import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

# TensorBoard log directory
LOG_DIR = "./tensorboard_logs/a2c/"

def main():
    """
    Trains an A2C agent on the MountainCar-v0 environment.
    """
    # Create the environment and wrap it with a Monitor
    env = gym.make("MountainCar-v0")
    env = Monitor(env, LOG_DIR)
    env = DummyVecEnv([lambda: env])

    # Configure the logger
    new_logger = configure(LOG_DIR, ["stdout", "tensorboard"])

    # --- Hyperparameters ---
    policy_kwargs = dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])])
    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=7e-4,
        n_steps=5,
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_rms_prop=True,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    # Set the logger
    model.set_logger(new_logger)

    # Train the model
    model.learn(total_timesteps=100000)

    # Save the trained model
    model.save("a2c_mountaincar")

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()
