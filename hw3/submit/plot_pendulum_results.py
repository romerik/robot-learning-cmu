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

def create_inverted_pendulum_plot(data_path):
    # Find the log directory
    pattern = f"{data_path}/q3_10_10_InvertedPendulum-v4_*/events.*"
    log_files = glob.glob(pattern)
    
    if not log_files:
        raise FileNotFoundError(f"No log file found for pattern: {pattern}")
    
    log_file = log_files[0]
    steps, returns = get_tensorboard_data(os.path.dirname(log_file))
    
    # Create plot
    plt.figure(figsize=(12, 7))
    plt.plot(steps, returns, label='Actor-Critic', color='blue', linewidth=2)
    
    # Add target line and debugging checkpoint
    plt.axhline(y=1000, color='r', linestyle='--', label='Target Return (1000)', alpha=0.7)
    
    # Find the return value at iteration 20
    iter_20_idx = np.searchsorted(steps, steps[0] + 20 * (steps[1] - steps[0]))
    if iter_20_idx < len(returns):
        return_at_20 = returns[iter_20_idx]
        plt.scatter([steps[iter_20_idx]], [return_at_20], color='green', s=100, 
                   label=f'Return at 20 iter: {return_at_20:.1f}', zorder=5)
        plt.axhline(y=100, color='g', linestyle='--', 
                   label='Minimum Expected at 20 iter', alpha=0.5)
    
    plt.xlabel('Number of Steps', fontsize=12)
    plt.ylabel('Average Return', fontsize=12)
    plt.title('Actor-Critic Performance on InvertedPendulum-v4', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Adjust y-axis to show full range
    plt.ylim(bottom=min(min(returns) - 50, 0), top=max(1100, max(returns) + 50))
    
    # Add legend
    plt.legend(fontsize=10, loc='lower right')
    
    # Save plot
    plt.savefig('inverted_pendulum_performance.png', dpi=300, bbox_inches='tight')
    print("Plot saved as inverted_pendulum_performance.png")
    
    # Print some statistics
    if iter_20_idx < len(returns):
        print(f"\nDebugging Statistics:")
        print(f"Return at iteration 20: {return_at_20:.1f}")
    print(f"Final return: {returns[-1]:.1f}")
    print(f"Max return: {np.max(returns):.1f}")
    print(f"Average return (last 10 iterations): {np.mean(returns[-10:]):.1f}")
    
    plt.close()

def main():
    data_path = "data"  # Assuming running from root directory
    print("Creating InvertedPendulum evaluation plot...")
    create_inverted_pendulum_plot(data_path)

if __name__ == "__main__":
    main()