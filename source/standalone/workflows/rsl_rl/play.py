# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

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
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import time

from torch.utils.tensorboard import SummaryWriter
from rsl_rl.runners import OnPolicyRunner

from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.dict import print_dict

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

# ログディレクトリの作成
log_dir = os.path.join("logs", "play_metrics/2025-01-10_05-40-41e10v15flat")
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
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

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )


    obs, _ = env.get_observations()
    start_time = time.time()
    removed_agents = set()  # 転倒したエージェントを追跡
    removed_agents_count = 0
    total_agents = env.num_envs  # 全エージェント数
    follow_ratios_x = []  # X軸の追従率を記録
    follow_ratios_z = []  # Z軸の追従率を記録
    speed_differences = []  # 速度差の記録

    timestep = 0
    last_log_time = start_time
    fall_time_buffer = 4.0  # 転倒直前4秒分のデータを除外
    agent_last_fall_time = [None] * total_agents  # エージェントごとの最後の転倒時間を記録

    # 速度変化を記録するための初期速度
    previous_velocities = torch.zeros((total_agents, 3), device=obs.device)
    no_movement_start_time = [None] * total_agents  # 動きが停止した開始時刻を記録

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, reward, done, info = env.step(actions)
            current_time = time.time()

            # 転倒エージェントの記録
            for idx, is_done in enumerate(done):
                if is_done and idx not in removed_agents:
                    removed_agents.add(idx)
                    removed_agents_count += 1
                    agent_last_fall_time[idx] = current_time

            # 転倒したエージェントの速度変化を確認し記録
            for idx in range(total_agents):
                if idx not in removed_agents:
                    actual_velocity = obs[idx, 0:3]  # 実際の速度 (x, y, z)
                    velocity_change = torch.abs(actual_velocity - previous_velocities[idx])

                    if velocity_change[0].item() < 0.1 and velocity_change[2].item() < 0.1:
                        if no_movement_start_time[idx] is None:
                            no_movement_start_time[idx] = current_time
                        elif current_time - no_movement_start_time[idx] >= 5.0:  # 5秒間変化しない場合
                            removed_agents.add(idx)
                            removed_agents_count += 1
                            agent_last_fall_time[idx] = current_time
                    else:
                        no_movement_start_time[idx] = None

                    previous_velocities[idx] = actual_velocity


            # 転倒したエージェント数をログに記録
            elapsed_time = current_time - start_time
            writer.add_scalar("Metrics/Fallen_Agents", removed_agents_count, elapsed_time)

            # 各エージェントの追従率と速度差を計算
            step_follow_ratios_x = []
            step_follow_ratios_z = []
            step_speed_differences = []
            policy_tensor = info["observations"]["policy"]
            for agent_id, observation in enumerate(policy_tensor):
                # 転倒したエージェントや転倒直前4秒分のデータを除外
                if agent_id in removed_agents or (
                    agent_last_fall_time[agent_id] is not None and current_time - agent_last_fall_time[agent_id] <= fall_time_buffer
                ):
                    continue

                # 実際の速度と目標速度の取得
                actual_velocity = observation[0:2]  # 実際の速度 (x, y)
                target_velocity = observation[9:11]  # 目標速度 (x, y)
                actual_speed = torch.norm(actual_velocity)  # 実際の速度の大きさ
                target_speed = torch.norm(target_velocity)  # 目標速度の大きさ

                # x軸での追従率の計算
                follow_ratio_x = (actual_speed / target_speed * 100) if target_speed > 0 else 0.0
                step_follow_ratios_x.append(follow_ratio_x)

                # y軸での速度差の計算
                speed_difference = actual_speed - target_speed
                step_speed_differences.append(speed_difference)

                # z軸の追従率計算
                actual_z = observation[5]  # 実際のz軸速度
                target_z = observation[11]  # 目標z軸速度
                z_range = 3.14159  # ±πの範囲
                z_deviation = abs((actual_z - target_z + z_range) % (2 * z_range) - z_range)  # 周期的な差を考慮
                z_follow_ratio = (1 - z_deviation / z_range) * 100
                z_follow_ratio = max(z_follow_ratio, 0)  # 負の値が発生しないように制限
                step_follow_ratios_z.append(z_follow_ratio)

            # 各ステップの平均を記録
            if step_follow_ratios_x:
                avg_follow_ratio_x = sum(step_follow_ratios_x) / len(step_follow_ratios_x)
                follow_ratios_x.append(avg_follow_ratio_x)
                writer.add_scalar("Metrics/Average_Follow_Ratio_X", avg_follow_ratio_x, elapsed_time - 10)  # 10秒遅延

            if step_follow_ratios_z:
                avg_follow_ratio_z = sum(step_follow_ratios_z) / len(step_follow_ratios_z)
                follow_ratios_z.append(avg_follow_ratio_z)
                writer.add_scalar("Metrics/Average_Follow_Ratio_Z", avg_follow_ratio_z, elapsed_time - 10)  # 10秒遅延

            if step_speed_differences:
                avg_speed_difference = sum(step_speed_differences) / len(step_speed_differences)
                speed_differences.append(avg_speed_difference)
                writer.add_scalar("Metrics/Average_Speed_Difference", avg_speed_difference, elapsed_time - 10)  # 10秒遅延

            # 10ステップごとの追従率と速度差の履歴を記録
            if timestep % 10 == 0:
                valid_follow_ratios_x = [val for val, agent_id in zip(follow_ratios_x[-10:], range(total_agents))
                                         if agent_id not in removed_agents]
                if valid_follow_ratios_x:
                    x_time_based_histogram = torch.tensor(valid_follow_ratios_x).mean()
                    writer.add_histogram("Metrics/Follow_Ratio_X_Time_Based", x_time_based_histogram, elapsed_time - 10)

                valid_follow_ratios_z = [val for val, agent_id in zip(follow_ratios_z[-10:], range(total_agents))
                                         if agent_id not in removed_agents]
                if valid_follow_ratios_z:
                    z_time_based_histogram = torch.tensor(valid_follow_ratios_z).mean()
                    writer.add_histogram("Metrics/Follow_Ratio_Z_Time_Based", z_time_based_histogram, elapsed_time - 10)

                valid_speed_differences = [val for val, agent_id in zip(speed_differences[-10:], range(total_agents))
                                           if agent_id not in removed_agents]
                if valid_speed_differences:
                    avg_speed_diff_histogram = torch.tensor(valid_speed_differences).mean()
                    writer.add_histogram("Metrics/Speed_Difference_Distribution", avg_speed_diff_histogram, elapsed_time - 10)

            timestep += 1


            # 全エージェントが転倒した場合、シミュレーションを終了
            if removed_agents_count == total_agents:
                print(f"[INFO] All agents have fallen at {elapsed_time:.2f} seconds.")
                break

            # 10秒ごとに経過時間をログに出力
            if current_time - last_log_time >= 10:
                print(f"[INFO] Elapsed time: {elapsed_time:.2f} seconds.")
                last_log_time = current_time

            # 転倒したエージェントのアクションをゼロに設定
            for idx in removed_agents:
                actions[idx] = 0

            obs, reward, done, info = env.step(actions)

    # 環境とログライターをクローズ
    env.close()
    writer.close()

if __name__ == "__main__":
    main()
    simulation_app.close()


