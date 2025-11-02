import gymnasium as gym
import torch
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder
from DQN import DQN_Network, DQN_Agent

def visualize_stable_baselines_agent(name, model_path, vec_normalize_path):
    """
    Visualizes a trained stable-baselines3 agent.
    """
    # Create a DummyVecEnv
    env = DummyVecEnv([lambda: gym.make("MountainCar-v0", render_mode="rgb_array")])

    # Load the normalization statistics
    env = VecNormalize.load(vec_normalize_path, env)

    # We do not want to continue training at test time
    env.training = False
    # We do not want to update running mean and variance at test time
    env.norm_reward = False

    # Load the trained model
    model = PPO.load(model_path) if name == "PPO" else A2C.load(model_path)

    # Create a VecVideoRecorder
    video_folder = f"videos/{name}/"
    video_length = 200  # Max episode length
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

def visualize_dqn_agent(model_path):
    """
    Visualizes the trained custom DQN agent.
    """
    # Create the environment
    env = gym.make("MountainCar-v0", render_mode="rgb_array")

    # Create a VecVideoRecorder
    video_folder = "videos/DQN/"
    video_length = 200
    env = DummyVecEnv([lambda: env])
    vec_env = VecVideoRecorder(
        env,
        video_folder,
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix="DQN-mountaincar",
    )

    # Initialize the agent
    agent = DQN_Agent(
        env=env,
        epsilon_max=-1,  # No exploration
        epsilon_min=-1,
        epsilon_decay=0,
        clip_grad_norm=0,
        learning_rate=0,
        discount=0,
        memory_capacity=0,
    )

    # Load the trained model weights
    agent.main_network.load_state_dict(torch.load(model_path))
    agent.main_network.eval()

    obs = vec_env.reset()
    for _ in range(video_length + 1):
        action = agent.select_action(obs)
        obs, _, _, _ = vec_env.step([action])

    vec_env.close()


def main():
    """
    Visualizes the trained agents and records a video of their performance.
    """
    visualize_stable_baselines_agent("PPO", "ppo_mountaincar.zip", "ppo_vecnormalize.pkl")
    visualize_stable_baselines_agent("A2C", "a2c_mountaincar.zip", "a2c_vecnormalize.pkl")
    visualize_dqn_agent("final_weights_1000.pth") # Assuming the final weights are saved with this name

if __name__ == "__main__":
    main()
