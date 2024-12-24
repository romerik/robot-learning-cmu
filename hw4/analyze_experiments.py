import subprocess
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import glob
import logging
import time
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment_progress.log'),
        logging.StreamHandler()
    ]
)

class ExperimentRunner:
    def __init__(self, base_dir="rob831/hw4_part1/scripts/../../data"):
        self.base_dir = base_dir
        self.start_time = None
        self.experiments_completed = 0
        
    def run_experiment(self, exp_name, exp_params):
        """Run a single experiment and monitor its progress"""
        start_time = datetime.now()
        cmd = f"python rob831/hw4_part1/scripts/run_hw4_mb.py {exp_params}"
        
        logging.info(f"\n{'='*50}")
        logging.info(f"Starting experiment: {exp_name}")
        logging.info(f"Command: {cmd}")
        logging.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            process = subprocess.run(cmd, shell=True, check=True, 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            end_time = datetime.now()
            duration = end_time - start_time
            self.experiments_completed += 1
            
            logging.info(f"Experiment completed successfully")
            logging.info(f"Duration: {duration}")
            
            time.sleep(2)  # Wait for file writing to complete
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

    def create_comparison_plot(self, experiment_data, title, save_name):
        """Create and save comparison plot"""
        plt.figure(figsize=(10, 6))
        
        for label, values in experiment_data.items():
            if values is not None:
                plt.plot(range(len(values)), values, marker='o', 
                        label=f'{label} (Final: {values[-1]:.2f})', 
                        markersize=4)
        
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Average Return', fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        logging.info(f"Saved plot: {save_name}")
        plt.close()

def main():
    runner = ExperimentRunner()
    
    # Define experiments
    experiments = {
        "Planning Horizon": {
            "Horizon 5": "--exp_name q4_reacher_horizon5 --env_name reacher-hw4_part1-v0 --add_sl_noise --mpc_horizon 5 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --video_log_freq -1 --mpc_action_sampling_strategy 'random'",
            "Horizon 15": "--exp_name q4_reacher_horizon15 --env_name reacher-hw4_part1-v0 --add_sl_noise --mpc_horizon 15 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --video_log_freq -1 --mpc_action_sampling_strategy 'random'",
            "Horizon 30": "--exp_name q4_reacher_horizon30 --env_name reacher-hw4_part1-v0 --add_sl_noise --mpc_horizon 30 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --video_log_freq -1 --mpc_action_sampling_strategy 'random'"
        },
        "Action Sequences": {
            "100 Sequences": "--exp_name q4_reacher_numseq100 --env_name reacher-hw4_part1-v0 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --mpc_num_action_sequences 100 --mpc_action_sampling_strategy 'random'",
            "1000 Sequences": "--exp_name q4_reacher_numseq1000 --env_name reacher-hw4_part1-v0 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --video_log_freq -1 --mpc_num_action_sequences 1000 --mpc_action_sampling_strategy 'random'"
        },
        "Ensemble Size": {
            "1 Model": "--exp_name q4_reacher_ensemble1 --env_name reacher-hw4_part1-v0 --ensemble_size 1 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --video_log_freq -1 --mpc_action_sampling_strategy 'random'",
            "3 Models": "--exp_name q4_reacher_ensemble3 --env_name reacher-hw4_part1-v0 --ensemble_size 3 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --video_log_freq -1 --mpc_action_sampling_strategy 'random'",
            "5 Models": "--exp_name q4_reacher_ensemble5 --env_name reacher-hw4_part1-v0 --ensemble_size 5 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --video_log_freq -1 --mpc_action_sampling_strategy 'random'"
        }
    }
    
    # Run all experiments and create plots
    for exp_type, configs in experiments.items():
        logging.info(f"\nStarting {exp_type} experiments...")
        
        # Run experiments
        for label, params in configs.items():
            logging.info(f"Running experiment for {label}")
            success = runner.run_experiment(label, params)
            if not success:
                logging.error(f"Failed to complete experiment for {label}")
        
        # Create plots after each set of experiments
        logging.info(f"\nCreating plots for {exp_type}...")
        data = {}
        for label, params in configs.items():
            exp_name = params.split('--exp_name')[1].split()[0]
            values = runner.get_experiment_data(exp_name)
            if values is not None:
                data[label] = values
                logging.info(f"Successfully loaded data for {label}")
        
        if data:
            save_name = f"{exp_type.lower().replace(' ', '_')}_comparison.png"
            runner.create_comparison_plot(
                data,
                f"Effect of {exp_type} on Performance",
                save_name
            )

if __name__ == "__main__":
    logging.info("Starting experiments and analysis...")
    main()
    logging.info("All experiments and analysis complete!")