import subprocess
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import glob
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cem_experiment_progress.log'),
        logging.StreamHandler()
    ]
)

class CEMExperimentRunner:
    def __init__(self, base_dir="rob831/hw4_part1/scripts/../../data"):
        self.base_dir = base_dir
        self.start_time = datetime.now()
        
    def run_experiment(self, exp_name, exp_params):
        """Run a single experiment with given parameters"""
        cmd = f"python rob831/hw4_part1/scripts/run_hw4_mb.py {exp_params}"
        
        start_time = datetime.now()
        logging.info(f"\n{'='*50}")
        logging.info(f"Starting experiment: {exp_name}")
        logging.info(f"Command: {cmd}")
        logging.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            process = subprocess.run(cmd, shell=True, check=True, 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            end_time = datetime.now()
            duration = end_time - start_time
            logging.info(f"Experiment completed successfully")
            logging.info(f"Duration: {duration}")
            return True
            
        except subprocess.CalledProcessError as e:
            logging.error(f"Experiment failed with error:\n{e.stderr.decode()}")
            return False

    def get_experiment_data(self, exp_name):
        """Extract evaluation returns from tensorboard logs"""
        try:
            pattern = os.path.join(self.base_dir, f"hw4_{exp_name}*")
            matching_dirs = glob.glob(pattern)
            
            if not matching_dirs:
                logging.warning(f"No directories found matching pattern: {pattern}")
                return None
            
            exp_dir = sorted(matching_dirs)[-1]
            event_acc = EventAccumulator(exp_dir)
            event_acc.Reload()
            
            eval_returns = [(s.step, s.value) for s in event_acc.Scalars('Eval_AverageReturn')]
            _, values = zip(*eval_returns)
            
            return np.array(values)
            
        except Exception as e:
            logging.error(f"Error processing {exp_name}: {str(e)}")
            return None

    def create_comparison_plot(self, experiment_data, save_name="cem_comparison.png"):
        """Create and save comparison plot"""
        plt.figure(figsize=(10, 6))
        
        for label, values in experiment_data.items():
            if values is not None:
                plt.plot(range(len(values)), values, marker='o', 
                        label=f'{label} (Final: {values[-1]:.2f})', 
                        markersize=4)
        
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Average Return', fontsize=12)
        plt.title('Comparison of CEM vs Random Shooting', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        # Add horizontal line at reward=800 for reference
        plt.axhline(y=800, color='r', linestyle='--', 
                   label='Target Performance (800)', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        logging.info(f"Saved plot: {save_name}")
        plt.close()

def main():
    runner = CEMExperimentRunner()
    
    # Define experiments
    experiments = {
        "Random Shooting": {
            "name": "q5_cheetah_random",
            "params": "--exp_name q5_cheetah_random --env_name 'cheetah-hw4_part1-v0' --mpc_horizon 15 --add_sl_noise --num_agent_train_steps_per_iter 1500 --batch_size_initial 5000 --batch_size 5000 --n_iter 5 --video_log_freq -1 --mpc_action_sampling_strategy 'random'"
        },
        "CEM (2 iterations)": {
            "name": "q5_cheetah_cem_2",
            "params": "--exp_name q5_cheetah_cem_2 --env_name 'cheetah-hw4_part1-v0' --mpc_horizon 15 --add_sl_noise --num_agent_train_steps_per_iter 1500 --batch_size_initial 5000 --batch_size 5000 --n_iter 5 --video_log_freq -1 --mpc_action_sampling_strategy 'cem' --cem_iterations 2"
        },
        "CEM (4 iterations)": {
            "name": "q5_cheetah_cem_4",
            "params": "--exp_name q5_cheetah_cem_4 --env_name 'cheetah-hw4_part1-v0' --mpc_horizon 15 --add_sl_noise --num_agent_train_steps_per_iter 1500 --batch_size_initial 5000 --batch_size 5000 --n_iter 5 --video_log_freq -1 --mpc_action_sampling_strategy 'cem' --cem_iterations 4"
        }
    }
    
    # Run experiments
    for label, exp in experiments.items():
        logging.info(f"\nRunning {label} experiment...")
        success = runner.run_experiment(exp["name"], exp["params"])
        if not success:
            logging.error(f"Failed to complete {label} experiment")
    
    # Create comparison plot
    data = {}
    for label, exp in experiments.items():
        values = runner.get_experiment_data(exp["name"])
        if values is not None:
            data[label] = values
            logging.info(f"Successfully loaded data for {label}")
    
    if data:
        runner.create_comparison_plot(data)
    
    # Log completion
    total_duration = datetime.now() - runner.start_time
    logging.info(f"\nAll experiments completed!")
    logging.info(f"Total duration: {total_duration}")

if __name__ == "__main__":
    logging.info("Starting CEM comparison experiments...")
    main()
    logging.info("All experiments and analysis complete!")