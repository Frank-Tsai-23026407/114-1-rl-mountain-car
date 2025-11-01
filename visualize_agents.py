import gymnasium as gym
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

def main():
    """
    Visualizes the trained agents and records a video of their performance.
    """
    env = DummyVecEnv([lambda: gym.make("MountainCar-v0", render_mode="rgb_array")])

    models = {
        "DQN": DQN.load("dqn_mountaincar"),
        "PPO": PPO.load("ppo_mountaincar"),
        "A2C": A2C.load("a2c_mountaincar"),
    }

    for name, model in models.items():
        video_folder = f"videos/{name}/"
        video_length = 200  # Max episode length

        # Create a VecVideoRecorder
        vec_env = VecVideoRecorder(
            env,
            video_folder,
            record_video_trigger=lambda x: x == 0,
            video_length=video_length,
            name_prefix=f"{name}-mountaincar",
        )

        obs = vec_env.reset()
        for _ in range(video_length + 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, _ = vec_env.step(action)

        vec_env.close()

if __name__ == "__main__":
    main()
