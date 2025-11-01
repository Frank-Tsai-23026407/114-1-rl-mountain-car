import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def main():
    """
    Generates and saves training curves from TensorBoard logs.
    """
    log_dirs = {
        "DQN": "./tensorboard_logs/dqn/",
        "PPO": "./tensorboard_logs/ppo/",
        "A2C": "./tensorboard_logs/a2c/",
    }

    plt.figure(figsize=(12, 6))

    # --- Plot Rewards ---
    plt.subplot(1, 2, 1)
    for label, log_dir in log_dirs.items():
        event_file = next((os.path.join(log_dir, f) for f in os.listdir(log_dir) if "events.out.tfevents" in f), None)
        if not event_file:
            print(f"No event file found in {log_dir}")
            continue

        ea = event_accumulator.EventAccumulator(
            event_file,
            size_guidance={event_accumulator.SCALARS: 0},
        )
        ea.Reload()

        if "rollout/ep_rew_mean" in ea.Tags()["scalars"]:
            rewards = pd.DataFrame(ea.Scalars("rollout/ep_rew_mean")).value
            steps = pd.DataFrame(ea.Scalars("rollout/ep_rew_mean")).step
            plt.plot(steps, rewards, label=label)
        else:
            print(f"'rollout/ep_rew_mean' not found for {label}")

    plt.title("Training Rewards")
    plt.xlabel("Timesteps")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(True)

    # --- Plot Episode Lengths ---
    plt.subplot(1, 2, 2)
    for label, log_dir in log_dirs.items():
        event_file = next((os.path.join(log_dir, f) for f in os.listdir(log_dir) if "events.out.tfevents" in f), None)
        if not event_file:
            continue

        ea = event_accumulator.EventAccumulator(
            event_file,
            size_guidance={event_accumulator.SCALARS: 0},
        )
        ea.Reload()

        if "rollout/ep_len_mean" in ea.Tags()["scalars"]:
            ep_lens = pd.DataFrame(ea.Scalars("rollout/ep_len_mean")).value
            steps = pd.DataFrame(ea.Scalars("rollout/ep_len_mean")).step
            plt.plot(steps, ep_lens, label=label)
        else:
            print(f"'rollout/ep_len_mean' not found for {label}")

    plt.title("Episode Lengths")
    plt.xlabel("Timesteps")
    plt.ylabel("Average Episode Length")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()

if __name__ == "__main__":
    main()
