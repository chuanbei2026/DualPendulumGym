import gymnasium

gymnasium.register(
    id="DualPendulum-v0",
    entry_point="dual_pendulum_gym.envs.dual_pendulum:DualPendulumEnv",
    max_episode_steps=1000,
)
