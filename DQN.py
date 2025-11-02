import os
import gc
import torch
import pygame
import warnings
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from custom_wrappers import reward_wrapper
from dqn_agent import DQN_Agent


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gc.collect()
torch.cuda.empty_cache()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # Used for debugging; CUDA related errors shown immediately.

# Seed everything for reproducible results
seed = 2024
np.random.seed(seed)
np.random.default_rng(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Model_TrainTest():
    def __init__(self, hyperparams):

        # Define RL Hyperparameters
        self.train_mode = hyperparams["train_mode"]
        self.RL_load_path = hyperparams["RL_load_path"]
        self.save_path = hyperparams["save_path"]
        self.save_interval = hyperparams["save_interval"]

        self.clip_grad_norm = hyperparams["clip_grad_norm"]
        self.learning_rate = hyperparams["learning_rate"]
        self.discount_factor = hyperparams["discount_factor"]
        self.batch_size = hyperparams["batch_size"]
        self.update_frequency = hyperparams["update_frequency"]
        self.max_episodes = hyperparams["max_episodes"]
        self.max_steps = hyperparams["max_steps"]
        self.render = hyperparams["render"]

        self.epsilon_max = hyperparams["epsilon_max"]
        self.epsilon_min = hyperparams["epsilon_min"]
        self.epsilon_decay = hyperparams["epsilon_decay"]

        self.memory_capacity = hyperparams["memory_capacity"]

        self.render_fps = hyperparams["render_fps"]

        # Define Env
        self.env = gym.make('MountainCar-v0', max_episode_steps=self.max_steps,
                            render_mode="human" if self.render else None)
        self.env.metadata['render_fps'] = self.render_fps # For max frame rate make it 0

        # Apply RewardWrapper
        self.env = reward_wrapper(self.env)
        self.env = DummyVecEnv([lambda: self.env])
        self.env = VecNormalize(self.env, norm_obs=True, norm_reward=False, clip_obs=10.)

        # Define the agent class
        self.agent = DQN_Agent(env = self.env,
                               epsilon_max = self.epsilon_max,
                               epsilon_min = self.epsilon_min,
                               epsilon_decay = self.epsilon_decay,
                               clip_grad_norm = self.clip_grad_norm,
                               learning_rate = self.learning_rate,
                               discount = self.discount_factor,
                               memory_capacity = self.memory_capacity,
                               seed = seed)


    def train(self):
        """
        Reinforcement learning training loop.
        """

        total_steps = 0
        self.reward_history = []

        # Training loop over episodes
        for episode in range(1, self.max_episodes+1):
            state = self.env.reset()
            done = False
            truncation = False
            step_size = 0
            episode_reward = 0

            while not done and not truncation:
                action = self.agent.select_action(state)
                next_state, reward, done, info = self.env.step([action])

                self.agent.replay_memory.store(state, action, next_state, reward, done)

                if len(self.agent.replay_memory) > self.batch_size:
                    self.agent.learn(self.batch_size, (done or truncation))

                # Update target-network weights
                if total_steps % self.update_frequency == 0:
                    self.agent.hard_update()

                state = next_state
                episode_reward += reward[0]
                step_size +=1

            # Appends for tracking history
            self.reward_history.append(episode_reward) # episode reward
            total_steps += step_size

            # Decay epsilon at the end of each episode
            self.agent.update_epsilon()

            #-- based on interval
            if episode % self.save_interval == 0:
                self.agent.save(self.save_path + '_' + f'{episode}' + '.pth')
                if episode != self.max_episodes:
                    self.plot_training(episode)
                print('\n~~~~~~Interval Save: Model saved.\n')

            result = (f"Episode: {episode}, "
                      f"Total Steps: {total_steps}, "
                      f"Ep Step: {step_size}, "
                      f"Raw Reward: {episode_reward:.2f}, "
                      f"Epsilon: {self.agent.epsilon_max:.2f}")
            print(result)
        self.plot_training(episode)
        self.env.save("dqn_vecnormalize.pkl")


    def test(self, max_episodes):
        """
        Reinforcement learning policy evaluation.
        """

        # Load the weights of the test_network
        self.agent.main_network.load_state_dict(torch.load(self.RL_load_path))
        self.agent.main_network.eval()

        # Testing loop over episodes
        for episode in range(1, max_episodes+1):
            state = self.env.reset()
            done = False
            truncation = False
            step_size = 0
            episode_reward = 0

            while not done and not truncation:
                action = self.agent.select_action(state)
                next_state, reward, done, info = self.env.step([action])

                state = next_state
                episode_reward += reward
                step_size += 1

            # Print log
            result = (f"Episode: {episode}, "
                      f"Steps: {step_size:}, "
                      f"Reward: {episode_reward:.2f}, ")
            print(result)

        pygame.quit() # close the rendering window


    def plot_training(self, episode):
        # Calculate the Simple Moving Average (SMA) with a window size of 50
        sma = np.convolve(self.reward_history, np.ones(50)/50, mode='valid')

        # Clip max (high) values for better plot analysis
        reward_history = np.clip(self.reward_history, a_min=None, a_max=100)
        sma = np.clip(sma, a_min=None, a_max=100)

        plt.figure()
        plt.title("Obtained Rewards")
        plt.plot(reward_history, label='Raw Reward', color='#4BA754', alpha=1)
        plt.plot(sma, label='SMA 50', color='#F08100')
        plt.xlabel("Episode")
        plt.ylabel("Rewards")
        plt.legend()

        # Only save as file if last episode
        if episode == self.max_episodes:
            plt.savefig('./reward_plot.png', format='png', dpi=600, bbox_inches='tight')
            plt.tight_layout()
            plt.grid(True)
            plt.show()
            plt.clf()
            plt.close()


        plt.figure()
        plt.title("Network Loss")
        plt.plot(self.agent.loss_history, label='Loss', color='#8921BB', alpha=1)
        plt.xlabel("Episode")
        plt.ylabel("Loss")

        # Only save as file if last episode
        if episode == self.max_episodes:
            plt.savefig('./Loss_plot.png', format='png', dpi=600, bbox_inches='tight')
            plt.tight_layout()
            plt.grid(True)
            plt.show()


if __name__ == '__main__':
    # Parameters:
    train_mode = False
    render = not train_mode
    RL_hyperparams = {
        "train_mode" : train_mode,
        "RL_load_path" : './final_weights' + '_' + '1000' + '.pth',
        "save_path" : './final_weights',
        "save_interval" : 100,

        "clip_grad_norm" : 5,
        "learning_rate" : 75e-5,
        "discount_factor" : 0.96,
        "batch_size" : 64,
        "update_frequency" : 20,
        "max_episodes" : 1000 if train_mode else 2,
        "max_steps" : 200,
        "render" : render,

        "epsilon_max" : 0.999 if train_mode else -1,
        "epsilon_min" : 0.01,
        "epsilon_decay" : 0.997,

        "memory_capacity" : 125_000 if train_mode else 0,

        "render_fps" : 60,
        }


    # Run
    DRL = Model_TrainTest(RL_hyperparams) # Define the instance
    # Train
    if train_mode:
        DRL.train()
    else:
        # Test
        DRL.test(max_episodes = RL_hyperparams['max_episodes'])