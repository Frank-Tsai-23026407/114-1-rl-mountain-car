import gymnasium as gym
import numpy as np


class reward_wrapper(gym.RewardWrapper):
    """
    Wrapper class for modifying rewards in the MountainCar-v0 environment.

    Args:
    env (gym.Env): The environment to wrap.
    """

    def __init__(self, env):
        super().__init__(env)


    def reward(self, reward):
        """
        Modifies the reward based on the current state of the environment.

        Args:
        state (numpy.ndarray): The current state of the environment.

        Returns:
        float: The modified reward.
        """
        state = self.env.unwrapped.state
        current_position, current_velocity = state # extract the position and current velocity based on the state

        # Interpolate the value to the desired range (because the velocity normalized value would be in range of 0 to 1 and now it would be in range of -0.5 to 0.5)
        current_velocity = np.interp(current_velocity, np.array([-0.07, 0.07]), np.array([-0.5, 0.5]))

        # (1) Calculate the modified reward based on the current position and velocity of the car.
        degree = (current_position + 1.2) / 1.8 * 360
        degree2radian = np.deg2rad(degree)
        modified_reward = 0.2 * (np.cos(degree2radian) + 2 * np.abs(current_velocity))

        # (2) Step limitation
        modified_reward -= 0.5 # Subtract 0.5 to adjust the base reward (to limit useless steps).

        # (3) Check if the car has surpassed a threshold of the path and is closer to the goal
        if current_position > 0.5:
            modified_reward += 20 # Add a bonus reward (Reached the goal)
        elif current_position > 0.25:
            modified_reward += 10 # So close to the goal
        elif current_position > 0:
            modified_reward += 6 # car is closer to the goal
        elif current_position > -0.5:
            modified_reward += 1 - np.exp(-2 * (current_position + 1.2) / 1.8) # car is getting close. Thus, giving reward based on the position and the further it reached


        # (4) Check if the car is coming down with velocity from left and goes with full velocity to right
        initial_position = -0.5 # Normalized value of initial position of the car which is extracted manually

        if current_velocity > 0.3 and current_position > initial_position + 0.1:
            modified_reward += 1 + 2 * (current_position + 1.2) / 1.8 # Add a bonus reward for this desired behavior

        return modified_reward + reward