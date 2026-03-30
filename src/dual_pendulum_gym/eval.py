import argparse

import torch

import dual_pendulum_gym  # noqa: F401
import gymnasium as gym

from dual_pendulum_gym.training.actor_critic import ActorCritic
from dual_pendulum_gym.envs.dual_pendulum import compute_status


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained DualPendulum agent")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    args = parser.parse_args()

    import pygame

    model = ActorCritic()
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    model.eval()

    env = gym.make("DualPendulum-v0", render_mode="human")

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        cumulative_reward = 0.0
        done = False
        step = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    env.close()
                    return

            with torch.no_grad():
                x = torch.FloatTensor(obs).unsqueeze(0)
                logits, _ = model(x)
                action = logits.argmax(dim=1).item()

            obs, reward, terminated, truncated, info = env.step(action)
            cumulative_reward += reward
            step += 1
            done = terminated or truncated

            status = info.get("status", "red")
            env.unwrapped.renderer.render(
                env.unwrapped.state,
                extra_text=f"Ep {ep}/{args.episodes}  Score: {cumulative_reward:.1f}  Step: {step}",
                status=status,
                info=info,
            )

        print(f"Episode {ep}: reward={cumulative_reward:.1f}, steps={step}")

    env.close()
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
