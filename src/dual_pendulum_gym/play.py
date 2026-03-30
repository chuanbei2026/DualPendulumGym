import argparse
import os
import numpy as np
import gymnasium as gym
import dual_pendulum_gym  # noqa: F401 — triggers env registration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=str, default=None,
                        help="Path to save recorded demos (e.g. demos/demo.npz)")
    args = parser.parse_args()

    import pygame

    env = gym.make("DualPendulum-v0", render_mode="human")
    obs, info = env.reset()
    env.render()

    cumulative_reward = 0.0
    running = True
    episode = 1

    # Recording buffers
    recording = args.record is not None
    all_obs = []
    all_actions = []
    step_count = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        if not running:
            break

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action = 0
        elif keys[pygame.K_RIGHT]:
            action = 2
        else:
            action = 1

        # Record before stepping
        if recording:
            all_obs.append(obs.copy())
            all_actions.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward
        step_count += 1

        # Render
        status = info.get("status", "red")
        rec_text = f"  [REC {step_count} steps]" if recording else ""
        env.unwrapped.renderer.render(
            env.unwrapped.state,
            extra_text=(
                f"Ep {episode}  Score: {cumulative_reward:.1f}"
                f"{rec_text}  [arrows=move, ESC=quit]"
            ),
            status=status,
            info=info,
        )

        if terminated or truncated:
            print(f"Episode {episode}: reward={cumulative_reward:.1f}, steps={step_count}")
            obs, info = env.reset()
            cumulative_reward = 0.0
            episode += 1

    env.close()

    # Save recording
    if recording and all_obs:
        os.makedirs(os.path.dirname(args.record) or ".", exist_ok=True)
        np.savez(
            args.record,
            observations=np.array(all_obs, dtype=np.float32),
            actions=np.array(all_actions, dtype=np.int64),
        )
        print(f"\nSaved {len(all_obs)} steps to {args.record}")


if __name__ == "__main__":
    main()
