# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import time
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

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    removed_agents = set()
    removed_agents_count = 0
    total_agents = env.num_envs
    start_time = time.time()
    last_checkpoint_time = start_time
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, reward, done, info = env.step(actions)
            current_time = time.time()
            
            policy_tensor = info["observations"]["policy"]

            # 転倒エージェントの判定と追跡
            for idx, is_done in enumerate(done):
                if is_done and idx not in removed_agents:
                    removed_agents.add(idx)
                    removed_agents_count += 1
                    print(f"[INFO] Agent {idx} has fallen. Total fallen: {removed_agents_count}")

            # 全エージェントが転倒した場合
            if removed_agents_count == total_agents:
                elapsed_time = current_time - start_time
                print(f"[INFO] All agents have fallen at {elapsed_time:.2f} seconds.")
                break

            # 各エージェントの追従率計算
            policy_tensor = info["observations"]["policy"]
            for agent_id, observation in enumerate(policy_tensor):
                if agent_id in removed_agents:
                    continue  # 転倒エージェントはスキップ

                # 実際の速度 (x, y)
                actual_velocity = observation[0:2]  # base_lin_vel の x, y
                # 目標速度 (x, y)
                target_velocity = observation[9:11]  # velocity_commands の x, y

                # z軸の値
                actual_z = observation[5]  # base_lin_vel の z 軸成分
                target_z = observation[11]  # velocity_commands の z 軸成分

                # 実際の速度と目標速度の大きさ（スカラー）
                actual_speed = torch.norm(actual_velocity)
                target_speed = torch.norm(target_velocity)

                # 追従率計算（x, y 軸）
                follow_ratio = (actual_speed / target_speed * 100) if target_speed > 0 else 0.0

                # 速度の差（スカラー）
                speed_difference = actual_speed - target_speed

                # z軸の追従率計算
                if target_z == 0:
                    actual_direction = actual_velocity / (torch.norm(actual_velocity) + 1e-6)
                    target_direction = torch.tensor([1.0, 0.0], device=actual_velocity.device)
                    dot_product = torch.dot(actual_direction, target_direction)
                    angle_error = torch.acos(dot_product.clip(-1.0, 1.0))
                    z_follow_ratio = (1 - angle_error / torch.pi) * 100
                else:
                    z_follow_ratio = (1 - abs(actual_z - target_z) / abs(target_z)) * 100
                    z_follow_ratio = max(z_follow_ratio, 0)

                # 結果の表示
                print(f"Agent {agent_id}:")
                print(f"  X-axis: Speed Follow Ratio = {follow_ratio:.2f} %")
                print(f"  Y-axis: Speed Difference = {speed_difference:.3f}")
                print(f"  Z-axis: Z-Axis Follow Ratio = {z_follow_ratio:.2f} %\n")

            # 10秒ごとの時間表示
            if current_time - last_checkpoint_time >= 10:
                elapsed_time = current_time - start_time
                print(f"[INFO] Elapsed time: {elapsed_time:.2f} seconds.")
                last_checkpoint_time = current_time
            for idx in removed_agents:
               actions[idx] = 0
            obs, reward, done, info = env.step(actions)
            
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
