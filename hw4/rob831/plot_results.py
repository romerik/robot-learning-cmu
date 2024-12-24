import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def plot_from_tensorboard(logdir, save_name='training_curves.png'):
    """
    Create plots from tensorboard logs
    
    Args:
        logdir: path to tensorboard log directory
        save_name: name of output plot file
    """
    # Load the tensorboard data
    event_acc = EventAccumulator(logdir)
    event_acc.Reload()

    # Get tags (metrics names)
    tags = event_acc.Tags()['scalars']

    # Get training and eval returns
    train_returns = [(s.step, s.value) for s in event_acc.Scalars('Train_AverageReturn')]
    eval_returns = [(s.step, s.value) for s in event_acc.Scalars('Eval_AverageReturn')]

    # Convert to numpy arrays
    train_steps, train_values = zip(*train_returns)
    eval_steps, eval_values = zip(*eval_returns)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_values, 'b-', label='Train Returns', marker='o')
    plt.plot(eval_steps, eval_values, 'r-', label='Eval Returns', marker='o')

    plt.xlabel('Iteration')
    plt.ylabel('Average Return')
    plt.title('Training and Evaluation Returns over Iterations')
    plt.legend()
    plt.grid(True)

    # Add target range for obstacles environment
    plt.axhline(y=-25, color='g', linestyle='--', alpha=0.3, label='Target Upper Bound (-25)')
    plt.axhline(y=-20, color='g', linestyle='--', alpha=0.3, label='Target Lower Bound (-20)')

    plt.savefig(save_name)
    plt.close()

# Use the function
logdir = "data/hw4_q3_obstacles_obstacles-hw4_part1-v0_12-11-2024_19-14-38"
plot_from_tensorboard(logdir, 'obstacles_performance_1.png')