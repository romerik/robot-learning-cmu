import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
import os

def get_tensorboard_data(log_dir, metric="Eval_AverageReturn"):
    """Extract data from tensorboard logs"""
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    _, step, value = zip(*event_acc.Scalars(metric))
    return np.array(value)

def plot_comparison(base_dir, experiment_type, configs, title):
    """Create comparison plot for different configurations"""
    plt.figure(figsize=(10, 6))
    
    for config in configs:
        # Find matching log directory
        pattern = os.path.join(base_dir, f'*q4_reacher_{experiment_type}{config}*')
        log_dirs = glob.glob(pattern)
        
        if log_dirs:
            values = get_tensorboard_data(log_dirs[0])
            plt.plot(range(len(values)), values, 
                    marker='o', label=f'{config}',
                    markersize=4, linewidth=2)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Evaluation Return', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    save_name = f'{experiment_type}_comparison.png'
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {save_name}")
    plt.close()

def main():
    # Base directory where logs are stored
    base_dir = "data"  # Modify this path as needed
    
    # Plot horizon comparison
    plot_comparison(
        base_dir,
        'horizon',
        ['5', '15', '30'],
        'Effect of Planning Horizon on Performance'
    )
    
    # Plot action sequences comparison
    plot_comparison(
        base_dir,
        'numseq',
        ['100', '1000'],
        'Effect of Number of Action Sequences on Performance'
    )
    
    # Plot ensemble size comparison
    plot_comparison(
        base_dir,
        'ensemble',
        ['1', '3', '5'],
        'Effect of Ensemble Size on Performance'
    )

if __name__ == "__main__":
    main()