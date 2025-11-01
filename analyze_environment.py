import gymnasium as gym

def main():
    """
    Analyzes the MountainCar-v0 environment from the Gymnasium library.

    This function creates an instance of the MountainCar-v0 environment and prints out
    key information about its observation space, action space, and reward structure.
    """
    # Create the MountainCar-v0 environment
    env = gym.make("MountainCar-v0")

    # --- Observation Space ---
    print(f"Observation Space: {env.observation_space}")
    print(f"  - Low: {env.observation_space.low}")
    print(f"  - High: {env.observation_space.high}")
    print("-" * 30)

    # --- Action Space ---
    print(f"Action Space: {env.action_space}")
    print(f"  - Number of Actions: {env.action_space.n}")
    print("-" * 30)

    # --- Reward Structure ---
    print("Reward Structure:")
    print("  - A reward of -1 is given for each time step.")
    print("  - The goal is to reach the flag on top of the right hill as quickly as possible.")
    print("  - There is no positive reward for reaching the goal, but the episode terminates.")
    print("-" * 30)

    # --- Episode Termination ---
    print("Episode Termination:")
    print("  - The episode ends if the car reaches the goal position (position >= 0.5).")
    print("  - The episode also ends if the number of steps exceeds 200.")
    print("-" * 30)

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()
