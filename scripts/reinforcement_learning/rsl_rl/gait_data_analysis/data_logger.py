# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import time
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--display_weight", type=float, default=5.0, help="Weight value to display in graph titles.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
# 座標変換用ユーティリティ
import isaaclab.utils.math as math_utils 

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # ---------------------------------------------------------
    # リセット設定 (10m歩くまで止めない)
    # ---------------------------------------------------------
    env_cfg.episode_length_s = 30.0  # タイムアウトを防ぐ
    if hasattr(env_cfg, "terminations"):
        env_cfg.terminations = None    # 転倒等によるリセットを無効化
        print("[INFO] Terminations disabled for continuous data collection.")

    # 初期コマンド設定 (X=1.0m/s, Y=0, Rot=0)
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        print("[INFO] Initial velocity commands fixed: X=1.0, Y=0.0, AngZ=0.0")

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # ログディレクトリの設定
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir
    print(f"[INFO] Output files will be saved to: {log_dir}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # =========================================================================
    # カメラ位置の設定
    # =========================================================================
    env.unwrapped.sim.set_camera_view(eye=[1.8, -3.5, 1.2], target=[1.8, 0.0, 0.6])
    # =========================================================================

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # ---------------------------------------------------------
    # センサーとロボット情報の取得
    # ---------------------------------------------------------
    scene = env.unwrapped.scene
    robot = scene["robot"]
    
    # ロボットの質量を取得 (CoT計算用)
    robot_mass = torch.sum(robot.data.default_mass[0]).item()
    gravity = 9.81
    print(f"[INFO] Robot Mass: {robot_mass:.2f} kg (for CoT calculation)")

    # ---------------------------------------------------------
    # 計測対象の関節インデックスを取得
    # ---------------------------------------------------------
    left_joint_names = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint"
    ]
    right_joint_names = [
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"
    ]
    target_joint_names = left_joint_names + right_joint_names
    
    all_joint_names = robot.data.joint_names
    target_joint_indices = []
    
    print("[INFO] Searching for target joint indices...")
    for name in target_joint_names:
        if name in all_joint_names:
            idx = all_joint_names.index(name)
            target_joint_indices.append(idx)
        else:
            print(f"  [WARNING] Joint '{name}' not found in robot model!")
    
    # ---------------------------------------------------------
    # 接触力センサーの取得
    # ---------------------------------------------------------
    contact_sensor = None
    if "contact_forces" in scene.sensors:
        contact_sensor = scene["contact_forces"]
    else:
        print("[WARNING] 'contact_forces' sensor not found in scene. Forces will be 0.")

    body_names = robot.body_names
    try:
        left_foot_names = [name for name in body_names if "left_ankle_roll_link" in name]
        right_foot_names = [name for name in body_names if "right_ankle_roll_link" in name]
        
        if not left_foot_names or not right_foot_names:
            l_foot_idx_robot = [i for i, n in enumerate(body_names) if "left" in n and "ankle" in n][-1]
            r_foot_idx_robot = [i for i, n in enumerate(body_names) if "right" in n and "ankle" in n][-1]
        else:
            l_foot_idx_robot = body_names.index(left_foot_names[0])
            r_foot_idx_robot = body_names.index(right_foot_names[0])
            
        l_foot_idx_sensor = 0
        r_foot_idx_sensor = 0
        if contact_sensor is not None:
            sensor_bodies = contact_sensor.body_names
            l_foot_name = body_names[l_foot_idx_robot]
            r_foot_name = body_names[r_foot_idx_robot]
            try:
                l_foot_idx_sensor = sensor_bodies.index(l_foot_name)
                r_foot_idx_sensor = sensor_bodies.index(r_foot_name)
            except ValueError:
                l_foot_idx_sensor = [i for i, n in enumerate(sensor_bodies) if "left" in n and "ankle" in n][-1]
                r_foot_idx_sensor = [i for i, n in enumerate(sensor_bodies) if "right" in n and "ankle" in n][-1]
            
    except Exception as e:
        print(f"[ERROR] Could not determine foot indices: {e}")
        return

    # extract and export policy
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # ---------------------------------------------------------
    # 初期化と (0,0) へのリセット
    # ---------------------------------------------------------
    obs = env.get_observations()
    
    # ロボットの位置を強制的に (0, 0) スタートにする
    print("[INFO] Forcing robot to start at (0, 0) facing forward...")
    
    root_state = robot.data.default_root_state.clone()
    root_state[:, :2] = 0.0 
    root_state[:, 3] = 1.0 
    root_state[:, 4:] = 0.0 
    
    # シミュレーションに反映
    robot.write_root_state_to_sim(root_state)
    robot.reset()
    scene.reset() 

    start_pos = torch.tensor([0.0, 0.0, 0.0], device=env.unwrapped.device)

    timestep = 0
    sim_time = 0.0
    distance_traveled = 0.0
    
    target_distance = 10.0 # 10m進むまで

    # エネルギー計算用
    total_energy = 0.0
    measurement_time = 0.0
    measurement_started = False
    start_measure_x = 0.0

    # ---------------------------------------------------------
    # CSVファイル設定
    # ---------------------------------------------------------
    file_forces = os.path.join(log_dir, "data_forces.csv")
    file_foot_z = os.path.join(log_dir, "data_foot_z.csv")
    file_com = os.path.join(log_dir, "data_com.csv")
    file_vel = os.path.join(log_dir, "data_velocity.csv")
    file_joints = os.path.join(log_dir, "data_joints_all.csv")
    file_power = os.path.join(log_dir, "data_power.csv")

    try:
        f_forces = open(file_forces, "w", newline="", encoding="utf-8")
        f_foot_z = open(file_foot_z, "w", newline="", encoding="utf-8")
        f_com = open(file_com, "w", newline="", encoding="utf-8")
        f_vel = open(file_vel, "w", newline="", encoding="utf-8")
        f_joints = open(file_joints, "w", newline="", encoding="utf-8")
        f_power = open(file_power, "w", newline="", encoding="utf-8")

        w_forces = csv.writer(f_forces)
        w_foot_z = csv.writer(f_foot_z)
        w_com = csv.writer(f_com)
        w_vel = csv.writer(f_vel)
        w_joints = csv.writer(f_joints)
        w_power = csv.writer(f_power)

        w_forces.writerow(["Timestep", "Left_Foot_Force_N", "Right_Foot_Force_N"])
        w_foot_z.writerow(["Timestep", "Left_Foot_Height_Z", "Right_Foot_Height_Z"])
        w_com.writerow(["Timestep", "CoM_X", "CoM_Y", "CoM_Z"])
        w_vel.writerow(["Timestep", "Target_Vel_X", "Actual_Vel_X", "Target_Vel_Y", "Actual_Vel_Y", "Target_Ang_Vel_Z", "Actual_Ang_Vel_Z"])
        w_power.writerow(["Timestep", "Instant_Power_W", "Total_Energy_J"])

        joint_header = ["Timestep"]
        for name in target_joint_names: joint_header.append(f"{name}_Torque")
        for name in target_joint_names: joint_header.append(f"{name}_Vel")
        w_joints.writerow(joint_header)

        print("[INFO] CSV files opened for recording.")
        
        # -----------------------------------------------------
        # データリスト初期化
        # -----------------------------------------------------
        traj_com = []
        traj_l_foot = []
        traj_r_foot = []
        forces_log_l = []
        forces_log_r = []
        vel_log_target = []
        vel_log_actual = []
        
        # 関節データ用ログリスト
        joint_torques_log = []
        joint_vels_log = []

        # [追加] 安定性解析用リスト
        traj_roll = []
        traj_pitch = []

        try:
            vel_command_term = env.unwrapped.command_manager._terms.get("base_velocity")
        except:
            vel_command_term = None
            print("[WARNING] Could not access 'base_velocity' term. Dynamic heading correction disabled.")

        print(f"[INFO] Starting simulation... 0m/s for 1s, then target {target_distance}m.")
        
        while simulation_app.is_running():
            start_time = time.time()
            
            # コマンド制御: 1秒待機 -> 移動開始 & 補正
            cmd_x = 0.0 # ループ冒頭で初期化
            
            if vel_command_term is not None:
                # デフォルト: 移動コマンド (X=1.0)
                cmd_x = 1.0
                cmd_y = 0.0
                cmd_yaw = 0.0

                # 最初の1秒間は速度0を指令 (足踏み/バランス維持)
                if sim_time < 1.0:
                    cmd_x = 0.0
                    cmd_y = 0.0
                    cmd_yaw = 0.0
                
                # コマンド適用
                vel_command_term.vel_command_b[:, 0] = cmd_x
                vel_command_term.vel_command_b[:, 1] = cmd_y
                vel_command_term.vel_command_b[:, 2] = cmd_yaw


            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)

                # 計測処理 (コマンドが1.0m/s以上になってから記録開始)
                if cmd_x >= 1.0:
                    
                    # 現在位置の取得
                    current_root_pos = robot.data.root_pos_w[0]
                    current_x_val = current_root_pos[0].item()

                    # 計測開始時の初期位置を保存
                    if not measurement_started:
                        print(f"[INFO] Command is {cmd_x} m/s. Measurement STARTED at time {sim_time:.2f}s")
                        measurement_started = True
                        start_measure_x = current_x_val

                    # --- データ取得 ---
                    com_pos = current_root_pos
                    all_body_pos = robot.data.body_pos_w[0]
                    l_foot_pos_vec = all_body_pos[l_foot_idx_robot]
                    r_foot_pos_vec = all_body_pos[r_foot_idx_robot]

                    l_force_val = 0.0
                    r_force_val = 0.0
                    if contact_sensor is not None:
                        net_forces = contact_sensor.data.net_forces_w[0]
                        l_force_val = torch.norm(net_forces[l_foot_idx_sensor]).item()
                        r_force_val = torch.norm(net_forces[r_foot_idx_sensor]).item()

                    cmd = env.unwrapped.command_manager.get_command("base_velocity")[0] 
                    target_vel_x = cmd[0].item(); target_vel_y = cmd[1].item(); target_ang_z = cmd[2].item()

                    root_quat = robot.data.root_quat_w[0]
                    lin_vel_b = math_utils.quat_apply_inverse(root_quat, robot.data.root_lin_vel_w[0])
                    ang_vel_b = math_utils.quat_apply_inverse(root_quat, robot.data.root_ang_vel_w[0])
                    actual_vel_x = lin_vel_b[0].item(); actual_vel_y = lin_vel_b[1].item(); actual_ang_z = ang_vel_b[2].item()

                    all_torques = robot.data.applied_torque[0]
                    all_vels = robot.data.joint_vel[0]
                    
                    inst_power = torch.sum(torch.abs(all_torques * all_vels)).item()
                    
                    # エネルギーと計測時間を加算
                    total_energy += inst_power * dt
                    measurement_time += dt

                    current_torques = [all_torques[idx].item() for idx in target_joint_indices]
                    current_vels = [all_vels[idx].item() for idx in target_joint_indices]

                    # --- [追加] 姿勢の計算 (Roll, Pitch) ---
                    # マスユーティリティ用に次元を追加 (Batch dim)
                    root_quat_batch = root_quat.unsqueeze(0)
                    r, p, y = math_utils.euler_xyz_from_quat(root_quat_batch)
                    
                    roll_deg = torch.rad2deg(r).item()
                    pitch_deg = torch.rad2deg(p).item()

                    # --- CSV書き込み ---
                    # CSV上でも、グラフ描画時と合わせるため、右足をマイナスにして保存します
                    w_forces.writerow([timestep, l_force_val, -r_force_val])
                    
                    w_foot_z.writerow([timestep, l_foot_pos_vec[2].item(), r_foot_pos_vec[2].item()])
                    w_com.writerow([timestep, com_pos[0].item(), com_pos[1].item(), com_pos[2].item()])
                    w_vel.writerow([timestep, target_vel_x, actual_vel_x, target_vel_y, actual_vel_y, target_ang_z, actual_ang_z])
                    w_joints.writerow([timestep] + current_torques + current_vels)
                    w_power.writerow([timestep, inst_power, total_energy])

                    # --- リスト保存 ---
                    traj_com.append(com_pos.cpu().numpy())
                    traj_l_foot.append(l_foot_pos_vec.cpu().numpy())
                    traj_r_foot.append(r_foot_pos_vec.cpu().numpy())
                    forces_log_l.append(l_force_val)
                    forces_log_r.append(r_force_val)
                    vel_log_target.append([target_vel_x, target_vel_y, target_ang_z])
                    vel_log_actual.append([actual_vel_x, actual_vel_y, actual_ang_z])
                    
                    # 関節データの保存
                    joint_torques_log.append(current_torques)
                    joint_vels_log.append(current_vels)

                    # [追加] 姿勢データの保存
                    traj_roll.append(roll_deg)
                    traj_pitch.append(pitch_deg)

                    # 相対距離での終了判定
                    measured_dist_now = current_x_val - start_measure_x
                    distance_traveled = current_x_val

                    # Y軸の位置を取得
                    current_y_val = current_root_pos[1].item()

                    # 終了条件1: X軸方向に目標距離 (10m) 到達
                    if measured_dist_now >= target_distance:
                        print(f"[INFO] Measured distance {measured_dist_now:.4f}m (Target {target_distance}m) reached. Stopping.")
                        break

                    # 終了条件2: Y軸方向に ±3m 以上逸脱
                    if abs(current_y_val) > 3.0:
                        print(f"[INFO] Lateral deviation {current_y_val:.4f}m exceeds limit ±3.0m. Stopping.")
                        break

                    # 終了条件3: 計測時間 20秒 経過 (追加)
                    if measurement_time >= 20.0:
                        print(f"[INFO] Measurement time limit {measurement_time:.2f}s reached. Stopping.")
                        break

            timestep += 1
            sim_time += dt

            if args_cli.video and timestep >= args_cli.video_length: pass
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0: time.sleep(sleep_time)

    except IOError as e:
        print(f"[ERROR] Failed to handle CSV files: {e}")
    finally:
        if 'f_forces' in locals(): f_forces.close()
        if 'f_foot_z' in locals(): f_foot_z.close()
        if 'f_com' in locals(): f_com.close()
        if 'f_vel' in locals(): f_vel.close()
        if 'f_joints' in locals(): f_joints.close()
        if 'f_power' in locals(): f_power.close()
        print("[INFO] All CSV data saved.")

    # ---------------------------------------------------------
    # メトリクス計算とレポート保存
    # ---------------------------------------------------------
    measured_distance = distance_traveled - start_measure_x
    
    if measured_distance > 0:
        cot = total_energy / (robot_mass * gravity * measured_distance)
    else:
        cot = float('inf')
        
    avg_power = total_energy / measurement_time if measurement_time > 0 else 0.0

    report_str = (
        f"========================================\n"
        f"METRICS REPORT (Measurement Phase Only)\n"
        f"Meas. Dist:   {measured_distance:.4f} m\n"
        f"Meas. Time:   {measurement_time:.2f} s\n"
        f"Total Energy: {total_energy:.2f} J\n"
        f"Avg Power:    {avg_power:.2f} W\n"
        f"CoT:          {cot:.4f}\n"
        f"========================================"
    )

    print(report_str)

    report_file_path = os.path.join(log_dir, "metrics_report.txt")
    with open(report_file_path, "w") as f:
        f.write(report_str)
    
    summary_csv_path = os.path.join(log_dir, "metrics_summary.csv")
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Meas_Dist_m", "Meas_Time_s", "Total_Energy_J", "Avg_Power_W", "CoT"])
        writer.writerow([measured_distance, measurement_time, total_energy, avg_power, cot])

    print(f"[INFO] Metrics report saved to: {report_file_path}")
    print(f"[INFO] Metrics summary CSV saved to: {summary_csv_path}")

    # ---------------------------------------------------------
    # グラフ生成処理 (更新版)
    # ---------------------------------------------------------
    if len(vel_log_target) > 0:
        # フォントサイズを大きく設定
        plt.rcParams.update({'font.size': 14})

        # =========================================================
        # [USER SETTINGS] グラフ設定
        # =========================================================
        Y_LIM_TORQUE = 100.0   # トルク Y軸範囲 (+/- Nm)
        Y_LIM_VEL = 20.0       # 速度 Y軸範囲 (+/- rad/s)
        
        # グラフタイトルに使われるタスク名 (Noneの場合は自動取得)
        DISPLAY_TASK_NAME = "feet_air_time" 
        # ↑ ここを変更するとグラフのタイトル名が変わります
        
        # グラフタイトルに使われるWeight値 (任意に変更可能)
        #DISPLAY_WEIGHT = 5.0
        DISPLAY_WEIGHT = args_cli.display_weight
        # ↑ ここを変更するとタイトルのWeight値が変わります
        # =========================================================

        # タスク名の決定
        if DISPLAY_TASK_NAME is None:
            plot_task_name = task_name
        else:
            plot_task_name = DISPLAY_TASK_NAME
            
        # タイトルプレフィックスの作成 (共通部分)
        title_prefix = f"Reward : {plot_task_name}, Weight : {DISPLAY_WEIGHT}"

        vel_log_target = np.array(vel_log_target)
        vel_log_actual = np.array(vel_log_actual)
        traj_com = np.array(traj_com)
        traj_l_foot = np.array(traj_l_foot)
        traj_r_foot = np.array(traj_r_foot)
        forces_l_arr = np.array(forces_log_l)
        forces_r_arr = np.array(forces_log_r)
        
        # 関節データの配列化 (Timesteps x NumJoints)
        joint_torques_arr = np.array(joint_torques_log)
        joint_vels_arr = np.array(joint_vels_log)

        # 時間軸の生成
        time_axis = np.arange(len(vel_log_target)) * dt

        # ---------------------------------------------------------
        # 速度追従グラフ
        # ---------------------------------------------------------
        fig_vel, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # タイトルを設定
        axs[0].set_title(f"{title_prefix}, Velocity_Tracking")
        
        axs[0].plot(time_axis, vel_log_target[:, 0], 'r--', label="Target", linewidth=2.5)
        axs[0].plot(time_axis, vel_log_actual[:, 0], 'b', alpha=0.7, label="Actual", linewidth=2.5)
        axs[0].set_ylabel("Vel X [m/s]")
        axs[0].legend()
        axs[0].grid(True)
        
        axs[1].plot(time_axis, vel_log_target[:, 1], 'r--', label="Target", linewidth=2.5)
        axs[1].plot(time_axis, vel_log_actual[:, 1], 'b', alpha=0.7, label="Actual", linewidth=2.5)
        axs[1].set_ylabel("Vel Y [m/s]")
        axs[1].grid(True)
        axs[1].set_ylim(-0.5, 0.5)
        
        axs[2].plot(time_axis, vel_log_target[:, 2], 'r--', label="Target", linewidth=2.5)
        axs[2].plot(time_axis, vel_log_actual[:, 2], 'b', alpha=0.7, label="Actual", linewidth=2.5)
        axs[2].set_ylabel("Ang Vel Z [rad/s]")
        axs[2].set_xlabel("Time [s]") # 横軸ラベル追加
        axs[2].grid(True)
        axs[2].set_ylim(-1.0, 1.0)

        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "velocity_tracking.png"))
        plt.close(fig_vel)

        # ---------------------------------------------------------
        # 関節トルク・速度のグラフ化 (Left/Right x Torque/Velocity)
        # ---------------------------------------------------------
        # target_joint_names = left_joint_names (0-5) + right_joint_names (6-11)
        num_left_joints = len(left_joint_names)
        num_right_joints = len(right_joint_names)

        def plot_joint_group(data, start_idx, end_idx, joint_names, suffix, title, filename, y_limit=None, y_label_text=""):
            """
            関節グループごとのグラフを作成するヘルパー関数
            """
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # 線の色を区別するためのカラーマップ
            colors = ['#FF4500', '#007FFF', '#009E73', '#56B4E9', '#E69F00', '#F0E442'] 
            
            for i, name in enumerate(joint_names):
                col_idx = start_idx + i
                if col_idx < data.shape[1]:
                    # ラベルの作成 (例: left_hip_pitch_joint_Torque)
                    label_name = f"{name}_{suffix}"
                    # データプロット (線を太く維持: linewidth=3.0)
                    ax.plot(time_axis, data[:, col_idx], label=label_name, linewidth=3.0, color=colors[i % len(colors)])

            ax.set_title(title)
            
            # --- 軸ラベルの設定 (追加) ---
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(y_label_text)
            # ---------------------------

            ax.grid(True, linestyle='-', alpha=0.6)
            
            # Y軸の範囲を固定設定
            if y_limit is not None:
                ax.set_ylim(-y_limit, y_limit)

            # 凡例を下に配置
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False)
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2) # 凡例スペース確保
            plt.savefig(os.path.join(log_dir, filename))
            plt.close(fig)
            print(f"[INFO] Saved graph: {filename}")

        # 1. Right_Velocity.png
        plot_joint_group(
            data=joint_vels_arr,
            start_idx=num_left_joints,
            end_idx=num_left_joints + num_right_joints,
            joint_names=right_joint_names,
            suffix="Vel",
            title=f"{title_prefix}, Right_Velocity",
            filename="Right_Velocity.png",
            y_limit=Y_LIM_VEL,
            y_label_text="Velocity [rad/s]"
        )

        # 2. Left_Torques.png
        plot_joint_group(
            data=joint_torques_arr,
            start_idx=0,
            end_idx=num_left_joints,
            joint_names=left_joint_names,
            suffix="Torque",
            title=f"{title_prefix}, Left_Torques",
            filename="Left_Torques.png",
            y_limit=Y_LIM_TORQUE,
            y_label_text="Torque [Nm]"
        )

        # 3. Right_Torques.png
        plot_joint_group(
            data=joint_torques_arr,
            start_idx=num_left_joints,
            end_idx=num_left_joints + num_right_joints,
            joint_names=right_joint_names,
            suffix="Torque",
            title=f"{title_prefix}, Right_Torques",
            filename="Right_Torques.png",
            y_limit=Y_LIM_TORQUE,
            y_label_text="Torque [Nm]"
        )

        # 4. Left_Velocity.png
        plot_joint_group(
            data=joint_vels_arr,
            start_idx=0,
            end_idx=num_left_joints,
            joint_names=left_joint_names,
            suffix="Vel",
            title=f"{title_prefix}, Left_Velocity",
            filename="Left_Velocity.png",
            y_limit=Y_LIM_VEL,
            y_label_text="Velocity [rad/s]"
        )

        # ---------------------------------------------------------
        # 2D 側面軌跡プロット (Side View: X-Z)
        # ---------------------------------------------------------
        print("[INFO] Generating 2D Side View plot (X-Z plane)...")
        fig_side, ax_side = plt.subplots(figsize=(14, 7)) 
        
        # タイトル追加
        ax_side.set_title(f"{title_prefix}, Side_View_Trajectory")

        # 重心 (CoM) の軌跡
        ax_side.plot(traj_com[:, 0], traj_com[:, 2], 'k-', linewidth=3, label='CoM Height')

        # 足首の軌跡
        ax_side.plot(traj_l_foot[:, 0], traj_l_foot[:, 2], 'b--', linewidth=2.0, alpha=0.6, label='Left Ankle')
        ax_side.plot(traj_r_foot[:, 0], traj_r_foot[:, 2], 'r--', linewidth=2.0, alpha=0.6, label='Right Ankle')
        
        # 地面 (Z=0) のラインを描画
        ax_side.axhline(y=0.0, color='gray', linestyle='-', linewidth=1.0, alpha=0.5)

        # スタートとゴール位置を点で表示
        ax_side.scatter(traj_com[0, 0], traj_com[0, 2], color='green', s=80, zorder=5, label='Start')
        ax_side.scatter(traj_com[-1, 0], traj_com[-1, 2], color='magenta', s=80, zorder=5, label='End')
        
        ax_side.set_xlabel("X Position (Distance) [m]")
        ax_side.set_ylabel("Z Position (Height) [m]")
        
        ax_side.grid(True)
        ax_side.legend(loc='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "trajectory_side_view.png"))
        plt.close(fig_side)

        # ---------------------------------------------------------
        # 2D 足跡プロット (着地時のみ)
        # ---------------------------------------------------------
        print("[INFO] Generating 2D Footprint plot (Stance Phase only)...")
        contact_threshold = 1.0 
        l_stance_idx = forces_l_arr > contact_threshold
        r_stance_idx = forces_r_arr > contact_threshold
        l_footprints = traj_l_foot[l_stance_idx]
        r_footprints = traj_r_foot[r_stance_idx]
        
        fig2d, ax2d = plt.subplots(figsize=(12, 8)) # Larger size
        # タイトル追加
        ax2d.set_title(f"{title_prefix}, Footprint")

        ax2d.plot(traj_com[:, 0], traj_com[:, 1], 'k-', linewidth=3.0, alpha=0.6, label='CoM Trajectory')
        if len(l_footprints) > 0:
            ax2d.scatter(l_footprints[:, 0], l_footprints[:, 1], c='blue', s=20, alpha=0.5, label='Left')
        if len(r_footprints) > 0:
            ax2d.scatter(r_footprints[:, 0], r_footprints[:, 1], c='red', s=20, alpha=0.5, label='Right')
            
        # 理想ライン (0,0)-(10,0) を描画
        ax2d.plot([0, 10], [0, 0], 'g--', linewidth=2.0, label='Target Line')
        
        ax2d.set_xlabel("X Position (m)")
        ax2d.set_ylabel("Y Position (m)")
        
        ax2d.axis('equal') 
        ax2d.set_ylim(-1.5, 1.5)

        ax2d.grid(True)
        ax2d.legend(loc='upper left')
        plt.savefig(os.path.join(log_dir, "footprint_2d.png"))
        plt.close(fig2d)
        
        # ---------------------------------------------------------
        # 足裏反力 (Foot Forces) のグラフ (面グラフ版)
        # ---------------------------------------------------------
        print("[INFO] Generating Foot Force plot (Area Chart)...")
        fig_force, ax_force = plt.subplots(figsize=(14, 7)) # Larger size
        # タイトル追加
        ax_force.set_title(f"{title_prefix}, Foot_Forces")

        # --- Left Foot (青・プラス側) ---
        # 0から値までの領域を塗りつぶす (面グラフ化)
        ax_force.fill_between(time_axis, forces_l_arr, 0, color='blue', alpha=0.3)
        # 境界線を描画
        ax_force.plot(time_axis, forces_l_arr, color='blue', linewidth=2.0, alpha=0.9, label='Left Foot Force')

        # --- Right Foot (赤・マイナス側) ---
        # 0からマイナス値までの領域を塗りつぶす (面グラフ化)
        ax_force.fill_between(time_axis, -forces_r_arr, 0, color='red', alpha=0.3)
        # 境界線を描画
        ax_force.plot(time_axis, -forces_r_arr, color='red', linewidth=2.0, alpha=0.9, label='Right Foot Force')

        ax_force.set_xlabel("Time [s]")
        ax_force.set_ylabel("Contact Force [N]")
        
        
        # 静的（固定）な値でY軸の範囲を対称に設定
        y_static_limit = 1500.0 
        ax_force.set_ylim(-y_static_limit, y_static_limit)
        
        ax_force.grid(True, linestyle='--', alpha=0.6)
        ax_force.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "foot_force_plot.png"))
        plt.close(fig_force)
        
        # ---------------------------------------------------------
        # 歩幅 (Stride Length) の解析グラフとCSV保存
        # ---------------------------------------------------------
        print("[INFO] Calculating Stride Lengths...")

        def get_stride_lengths(forces, traj_foot, name="Foot"):
            # 接触判定の閾値 (N)
            threshold = 1.0 
            # 接触状態 (True/False)
            is_contact = forces > threshold
            
            # 着地瞬間 (Swing -> Stance) のインデックスを検出
            # diffが 1 になる箇所が立ち上がり（着地）
            contact_indices = np.where(np.diff(is_contact.astype(int)) == 1)[0]
            
            if len(contact_indices) < 2:
                print(f"[WARNING] Not enough steps detected for {name}.")
                return [], []

            # 着地した瞬間のX座標を取得
            step_x_positions = traj_foot[contact_indices, 0]
            
            # 差分をとって歩幅 (Stride Length) を計算
            # stride[i] = pos[i+1] - pos[i] (同じ足の連続する着地間距離)
            strides = np.diff(step_x_positions)
            
            return strides, step_x_positions

        # 左右それぞれの歩幅を計算
        l_strides, l_step_pos = get_stride_lengths(forces_l_arr, traj_l_foot, "Left")
        r_strides, r_step_pos = get_stride_lengths(forces_r_arr, traj_r_foot, "Right")

        if len(l_strides) > 0 and len(r_strides) > 0:
            fig_stride, ax_stride = plt.subplots(figsize=(10, 6))
            # タイトル追加
            ax_stride.set_title(f"{title_prefix}, Stride_Length")
            
            # グラフ描画 (横軸: 歩数, 縦軸: 歩幅[m])
            # 1歩目, 2歩目... と数えられるように index + 1 しています
            ax_stride.plot(np.arange(1, len(l_strides) + 1), l_strides, 'b-o', label='Left Stride (L->L)', linewidth=2.5)
            ax_stride.plot(np.arange(1, len(r_strides) + 1), r_strides, 'r-o', label='Right Stride (R->R)', linewidth=2.5)
            
            # 平均値のライン
            l_mean = np.mean(l_strides)
            r_mean = np.mean(r_strides)
            ax_stride.axhline(y=l_mean, color='blue', linestyle='--', alpha=0.5, label=f'Left Mean: {l_mean:.3f}m')
            ax_stride.axhline(y=r_mean, color='red', linestyle='--', alpha=0.5, label=f'Right Mean: {r_mean:.3f}m')

            ax_stride.set_xlabel("Step Count")
            ax_stride.set_ylabel("Stride Length [m]")
            
            ax_stride.grid(True)
            ax_stride.legend(loc='lower right') 

            plt.tight_layout()
            plt.savefig(os.path.join(log_dir, "stride_length_analysis.png"))
            plt.close(fig_stride)
            print(f"[INFO] Stride analysis saved. Avg L: {l_mean:.3f}m, Avg R: {r_mean:.3f}m")
            
            # 歩幅データのCSV保存
            with open(os.path.join(log_dir, "data_strides.csv"), "w", newline="") as f_stride:
                w_stride = csv.writer(f_stride)
                w_stride.writerow(["Step_Index", "Left_Stride_m", "Right_Stride_m"])
                
                # 左右の歩数が異なる場合を考慮して保存
                max_len = max(len(l_strides), len(r_strides))
                for i in range(max_len):
                    val_l = l_strides[i] if i < len(l_strides) else ""
                    val_r = r_strides[i] if i < len(r_strides) else ""
                    w_stride.writerow([i+1, val_l, val_r])
            print(f"[INFO] Stride CSV saved to: {os.path.join(log_dir, 'data_strides.csv')}")

        else:
            print("[WARNING] Could not generate stride graph (steps < 2).")

        # ---------------------------------------------------------
        # 姿勢偏差 (Stability Analysis: Roll & Pitch)
        # ---------------------------------------------------------
        if len(traj_roll) > 0:
            print("[INFO] Generating Stability Analysis plot...")
            traj_roll_arr = np.array(traj_roll)
            traj_pitch_arr = np.array(traj_pitch)
            
            roll_std = np.std(traj_roll_arr)
            pitch_std = np.std(traj_pitch_arr)
            
            fig_stab, ax_stab = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            # Roll
            ax_stab[0].plot(time_axis, traj_roll_arr, 'b', label='Roll (L/R)')
            ax_stab[0].set_title(f"{title_prefix}, Roll Stability (Std: {roll_std:.4f} deg)")
            ax_stab[0].set_ylabel("Roll [deg]")
            ax_stab[0].grid(True)
            ax_stab[0].legend()
            
            # Pitch
            ax_stab[1].plot(time_axis, traj_pitch_arr, 'r', label='Pitch (F/B)')
            ax_stab[1].set_title(f"{title_prefix}, Pitch Stability (Std: {pitch_std:.4f} deg)")
            ax_stab[1].set_ylabel("Pitch [deg]")
            ax_stab[1].set_xlabel("Time [s]")
            ax_stab[1].grid(True)
            ax_stab[1].legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(log_dir, "stability_analysis.png"))
            plt.close(fig_stab)
            print(f"[INFO] Saved graph: stability_analysis.png")
        
    else:
        print("[WARNING] No data collected.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()