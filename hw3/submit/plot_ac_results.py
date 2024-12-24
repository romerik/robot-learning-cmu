import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
import os

def get_tensorboard_data(log_dir, metric="Eval_AverageReturn"):
    """Extract evaluation returns from tensorboard logs"""
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    # Get wall time, step, and value for the metric
    _, step, value = zip(*event_acc.Scalars(metric))
    return np.array(step), np.array(value)

def create_eval_plot(data_path):
    # Find the log directory
    pattern = f"{data_path}/q2_10_10_CartPole-v0_*/events.*"
    log_files = glob.glob(pattern)
    
    if not log_files:
        raise FileNotFoundError(f"No log file found for pattern: {pattern}")
    
    log_file = log_files[0]
    steps, returns = get_tensorboard_data(os.path.dirname(log_file))
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, returns, label='Actor-Critic', color='blue')
    
    plt.xlabel('Number of Steps', fontsize=12)
    plt.ylabel('Average Return', fontsize=12)
    plt.title('Actor-Critic Performance on CartPole-v0', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add target line at 200
    plt.axhline(y=200, color='r', linestyle='--', label='Target Return')
    
    plt.legend(fontsize=10)
    
    # Save plot
    plt.savefig('actor_critic_performance.png', dpi=300, bbox_inches='tight')
    print("Plot saved as actor_critic_performance.png")
    plt.close()

def main():
    data_path = "data"  # Assuming running from root directory
    print("Creating evaluation plot...")
    create_eval_plot(data_path)

if __name__ == "__main__":
    main()