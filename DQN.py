import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

# TensorBoard log directory
LOG_DIR = "./tensorboard_logs/dqn/"

def main():
    """
    Trains a DQN agent on the MountainCar-v0 environment.
    """
    # Create the environment and wrap it with a Monitor
    env = gym.make("MountainCar-v0")
    env = Monitor(env, LOG_DIR)
    env = DummyVecEnv([lambda: env])

    # Configure the logger
    new_logger = configure(LOG_DIR, ["stdout", "tensorboard"])

    # --- Hyperparameters ---
    policy_kwargs = dict(net_arch=[256, 256])
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=100000,
        learning_starts=500,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    # Set the logger
    model.set_logger(new_logger)

    # Train the model
    model.learn(total_timesteps=100000)

    # Save the trained model
    model.save("dqn_mountaincar")

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()
