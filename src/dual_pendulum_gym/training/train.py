"""GRPO (Group Relative Policy Optimization) trainer for DualPendulum."""
import argparse
import os
import time
from collections import deque

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical

import dual_pendulum_gym  # noqa: F401
import gymnasium as gym

from dual_pendulum_gym.training.actor_critic import ActorCritic


def parse_args():
    p = argparse.ArgumentParser(description="Train GRPO on DualPendulum-v0")
    p.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    p.add_argument("--clip-eps", type=float, default=0.2, help="PPO-style clip epsilon")
    p.add_argument("--entropy-coef", type=float, default=0.05, help="Entropy bonus coefficient")
    p.add_argument("--rollout-steps", type=int, default=1000, help="Steps per trajectory (one episode)")
    p.add_argument("--group-size", type=int, default=4, help="Group size: trajectories to compare")
    p.add_argument("--ppo-epochs", type=int, default=5, help="Update epochs per group")
    p.add_argument("--mini-batch-size", type=int, default=128, help="Mini-batch size")
    p.add_argument("--max-episodes", type=int, default=10000, help="Max training episodes (groups)")
    p.add_argument("--checkpoint-interval", type=int, default=200, help="Save every N groups")
    p.add_argument("--save-dir", type=str, default="checkpoints", help="Checkpoint directory")
    p.add_argument("--log-interval", type=int, default=10, help="Print stats every N groups")
    p.add_argument("--render", action="store_true", help="Render best env in real-time")
    p.add_argument("--render-fps", type=int, default=120, help="Target FPS for render mode")
    p.add_argument("--pretrained", type=str, default=None, help="Path to pretrained model")
    return p.parse_args()


def collect_trajectory(env, model, max_steps, render=False, render_info=None):
    """Collect one full trajectory (episode) using current policy."""
    obs, _ = env.reset()
    traj_obs = []
    traj_actions = []
    traj_log_probs = []
    traj_rewards = []
    total_reward = 0.0

    for step in range(max_steps):
        if render:
            import pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    return None  # signal to quit

        traj_obs.append(obs.copy())

        with torch.no_grad():
            action, log_prob, _ = model.act(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)

        traj_actions.append(action)
        traj_log_probs.append(log_prob.item())
        traj_rewards.append(reward)
        total_reward += reward

        if render:
            status = info.get("status", "red")
            env.unwrapped.renderer.render(
                env.unwrapped.state,
                extra_text=render_info or "",
                status=status,
                info=info,
            )

        if terminated or truncated:
            break
        obs = next_obs

    return {
        "obs": np.array(traj_obs),
        "actions": np.array(traj_actions),
        "log_probs": np.array(traj_log_probs),
        "rewards": traj_rewards,
        "total_reward": total_reward,
        "length": len(traj_rewards),
    }


def compute_discounted_rewards(rewards, gamma):
    """Compute per-step discounted returns."""
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def train():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Create environments: one for rendering, rest headless
    envs = []
    for i in range(args.group_size):
        rm = "human" if (args.render and i == 0) else None
        envs.append(gym.make("DualPendulum-v0", render_mode=rm))

    model = ActorCritic()
    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained, map_location="cpu", weights_only=True))
        print(f"Loaded pretrained model: {args.pretrained}")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.render:
        envs[0].reset()
        envs[0].render()
        envs[0].unwrapped.renderer.clock_fps = args.render_fps

    reward_history = deque(maxlen=100)
    best_avg = -float("inf")
    total_steps = 0
    start_time = time.time()

    for group_idx in range(1, args.max_episodes + 1):
        # --- Step 1: Collect a group of trajectories ---
        trajectories = []
        avg = np.mean(reward_history) if reward_history else 0.0

        for i, env in enumerate(envs):
            render_this = args.render and i == 0
            render_info = (
                f"Group {group_idx}  Env {i}  Avg: {avg:.0f}  Steps: {total_steps}"
                if render_this else None
            )
            traj = collect_trajectory(
                env, model, args.rollout_steps,
                render=render_this, render_info=render_info,
            )
            if traj is None:  # user quit
                for e in envs:
                    e.close()
                return
            trajectories.append(traj)
            total_steps += traj["length"]

        # --- Step 2: Compute group-relative advantages ---
        group_rewards = [t["total_reward"] for t in trajectories]
        mean_reward = np.mean(group_rewards)
        std_reward = np.std(group_rewards) + 1e-8

        # Track all episode rewards
        for r in group_rewards:
            reward_history.append(r)

        # Per-step advantages: trajectory-level relative score spread to each step
        all_obs = []
        all_actions = []
        all_old_log_probs = []
        all_advantages = []
        all_returns = []

        for traj in trajectories:
            # Group-relative advantage for this trajectory
            traj_advantage = (traj["total_reward"] - mean_reward) / std_reward

            # Also use per-step discounted returns for finer signal
            step_returns = compute_discounted_rewards(traj["rewards"], args.gamma)
            step_returns_t = torch.tensor(step_returns, dtype=torch.float32)

            # Combine: group-relative baseline + per-step structure
            # Normalize step returns within trajectory
            sr_mean = step_returns_t.mean()
            sr_std = step_returns_t.std() + 1e-8
            step_advantages = (step_returns_t - sr_mean) / sr_std

            # Blend: group-level ranking + step-level detail
            combined_advantages = 0.5 * traj_advantage + 0.5 * step_advantages

            all_obs.append(torch.FloatTensor(traj["obs"]))
            all_actions.append(torch.LongTensor(traj["actions"]))
            all_old_log_probs.append(torch.FloatTensor(traj["log_probs"]))
            all_advantages.append(combined_advantages)
            all_returns.append(step_returns_t)

        obs_t = torch.cat(all_obs)
        act_t = torch.cat(all_actions)
        old_lp_t = torch.cat(all_old_log_probs)
        adv_t = torch.cat(all_advantages)
        ret_t = torch.cat(all_returns)

        # Final normalization across entire group
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # --- Step 3: PPO-style clipped update (no critic) ---
        dataset_size = len(obs_t)
        for _ in range(args.ppo_epochs):
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, args.mini_batch_size):
                end = min(start + args.mini_batch_size, dataset_size)
                mb_idx = indices[start:end]

                mb_obs = obs_t[mb_idx]
                mb_act = act_t[mb_idx]
                mb_old_lp = old_lp_t[mb_idx]
                mb_adv = adv_t[mb_idx]

                # Forward pass (ignore value head)
                logits, _ = model(mb_obs)
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(mb_act)
                entropy = dist.entropy()

                ratio = torch.exp(log_probs - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                loss = policy_loss - args.entropy_coef * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        # --- Logging ---
        avg_reward = np.mean(reward_history)
        best_in_group = max(group_rewards)
        worst_in_group = min(group_rewards)

        if group_idx % args.log_interval == 0:
            elapsed = time.time() - start_time
            print(
                f"Group {group_idx:>5d} | "
                f"Best: {best_in_group:>8.1f} | "
                f"Worst: {worst_in_group:>8.1f} | "
                f"Avg(100): {avg_reward:>8.1f} | "
                f"Steps: {total_steps:>8d} | "
                f"Time: {elapsed:.0f}s"
            )

        if group_idx % args.checkpoint_interval == 0:
            path = os.path.join(args.save_dir, f"model_grp{group_idx}.pt")
            torch.save(model.state_dict(), path)
            print(f"  Saved checkpoint: {path}")

        if avg_reward > best_avg and len(reward_history) >= 100:
            best_avg = avg_reward
            path = os.path.join(args.save_dir, "model_best.pt")
            torch.save(model.state_dict(), path)

    # Final save
    path = os.path.join(args.save_dir, "model_final.pt")
    torch.save(model.state_dict(), path)
    print(f"\nTraining complete. Final model: {path}")
    print(f"Best avg reward (100-ep): {best_avg:.1f}")

    for env in envs:
        env.close()


if __name__ == "__main__":
    train()
