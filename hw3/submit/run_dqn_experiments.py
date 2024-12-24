import subprocess
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
import os
import pandas as pd

def run_experiment(exp_name, seed, double_q=False):
    """Run a single experiment with given parameters"""
    # Running from root directory
    cmd = [
        "python",
        "rob831/scripts/run_hw3_dqn.py",
        "--env_name", "LunarLander-v3",
        "--exp_name", exp_name,
        "--seed", str(seed),
        "--no_gpu"
    ]
    if double_q:
        cmd.append("--double_q")
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)

def get_tensorboard_data(log_dir, metric="Train_AverageReturn"):
    """Extract data from tensorboard logs"""
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    # Get wall time, step, and value for the metric
    _, step, value = zip(*event_acc.Scalars(metric))
    return np.array(step), np.array(value)

def run_all_experiments():
    # Run all experiments
    for seed in [1, 2, 3]:
        print(f"\nRunning DQN with seed {seed}")
        run_experiment(f"q1_dqn_{seed}", seed)
        print(f"\nRunning Double DQN with seed {seed}")
        run_experiment(f"q1_doubledqn_{seed}", seed, double_q=True)

def create_comparison_plot():
    # Using the data directory from root
    data_path = "data"
    
    # Lists to store results
    dqn_returns = []
    ddqn_returns = []
    steps = None
    
    # Get data for regular DQN
    for seed in [1, 2, 3]:
        pattern = f"{data_path}/q1_dqn_{seed}_LunarLander-v3_*/events.*"
        log_files = glob.glob(pattern)
        if not log_files:
            raise FileNotFoundError(f"No log file found for pattern: {pattern}")
        log_file = log_files[0]
        step, values = get_tensorboard_data(os.path.dirname(log_file))
        dqn_returns.append(values)
        if steps is None:
            steps = step
    
    # Get data for Double DQN
    for seed in [1, 2, 3]:
        pattern = f"{data_path}/q1_doubledqn_{seed}_LunarLander-v3_*/events.*"
        log_files = glob.glob(pattern)
        if not log_files:
            raise FileNotFoundError(f"No log file found for pattern: {pattern}")
        log_file = log_files[0]
        step, values = get_tensorboard_data(os.path.dirname(log_file))
        ddqn_returns.append(values)

    # Calculate means and standard errors
    dqn_mean = np.mean(dqn_returns, axis=0)
    dqn_std = np.std(dqn_returns, axis=0) / np.sqrt(3)  # Standard error
    ddqn_mean = np.mean(ddqn_returns, axis=0)
    ddqn_std = np.std(ddqn_returns, axis=0) / np.sqrt(3)  # Standard error

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, dqn_mean, label='DQN', color='blue')
    plt.fill_between(steps, dqn_mean - dqn_std, dqn_mean + dqn_std, alpha=0.2, color='blue')
    plt.plot(steps, ddqn_mean, label='Double DQN', color='red')
    plt.fill_between(steps, ddqn_mean - ddqn_std, ddqn_mean + ddqn_std, alpha=0.2, color='red')
    
    plt.xlabel('Number of Steps', fontsize=12)
    plt.ylabel('Average Return', fontsize=12)
    plt.title('DQN vs Double DQN on LunarLander-v3', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Use scientific notation for x-axis
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    # Save plot
    plt.savefig('dqn_comparison.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as dqn_comparison.png")
    plt.close()

def main():
    # Run all experiments
    print("Starting experiments...")
    run_all_experiments()
    
    # Create comparison plot
    print("\nCreating comparison plot...")
    create_comparison_plot()
    
    print("\nDone! Results plotted in dqn_comparison.png")

if __name__ == "__main__":
    main()