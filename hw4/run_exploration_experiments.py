# import subprocess
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
# import os
# import logging
# from datetime import datetime
# import glob

# logging.basicConfig(level=logging.INFO)

# class ExplorationExperimentRunner:
#     def __init__(self, base_dir="rob831/data"):
#         self.base_dir = base_dir
        
#     def run_experiment(self, exp_name, params):
#         cmd = f"python rob831/hw4_part2/scripts/run_hw4_expl.py {params}"
#         logging.info(f"Running experiment: {exp_name}\nCommand: {cmd}")
        
#         start_time = datetime.now()
#         try:
#             subprocess.run(cmd, shell=True, check=True)
#             logging.info(f"Experiment {exp_name} completed successfully")
#         except subprocess.CalledProcessError as e:
#             logging.error(f"Experiment {exp_name} failed: {str(e)}")
        
#         duration = datetime.now() - start_time
#         logging.info(f"Experiment {exp_name} completed in {duration}")

#     def load_experiment_data(self, exp_name):
#         """Load tensorboard data from experiment"""
#         try:
#             # Find the experiment directory
#             pattern = os.path.join(self.base_dir, f"hw4_part2_expl_{exp_name}*")
#             exp_dirs = glob.glob(pattern)
            
#             if not exp_dirs:
#                 logging.error(f"No directories found matching {pattern}")
#                 return None
                
#             exp_dir = sorted(exp_dirs)[-1]  # Get most recent
            
#             # Load tensorboard data
#             event_acc = EventAccumulator(exp_dir)
#             event_acc.Reload()
            
#             # Extract relevant metrics
#             returns = [(s.step, s.value) for s in event_acc.Scalars('Train_AverageReturn')]
#             _, values = zip(*returns)
            
#             return {
#                 'Returns': np.array(values),
#                 'Dir': exp_dir
#             }
#         except Exception as e:
#             logging.error(f"Error loading data for {exp_name}: {str(e)}")
#             return None

#     def create_plots(self, env_name, rnd_exp_name, random_exp_name):
#         """Create comparison plots"""
#         try:
#             # Load data
#             rnd_data = self.load_experiment_data(rnd_exp_name)
#             random_data = self.load_experiment_data(random_exp_name)
            
#             if rnd_data is None or random_data is None:
#                 logging.error("Could not create plots due to missing data")
#                 return
            
#             # Create learning curves plot
#             plt.figure(figsize=(10, 6))
#             plt.plot(rnd_data['Returns'], label='RND Exploration')
#             plt.plot(random_data['Returns'], label='Random Exploration')
#             plt.xlabel('Iteration')
#             plt.ylabel('Average Return')
#             plt.title(f'RND vs Random Exploration on {env_name}')
#             plt.legend()
#             plt.grid(True, alpha=0.3)
            
#             # Save plot
#             save_path = f'{env_name.lower().replace("-", "_")}_comparison.png'
#             plt.savefig(save_path)
#             plt.close()
#             logging.info(f"Saved comparison plot to {save_path}")
            
#             # Also display state density plots if available
#             self.display_density_plots(rnd_data['Dir'], random_data['Dir'], env_name)
            
#         except Exception as e:
#             logging.error(f"Error creating plots for {env_name}: {str(e)}")

#     def display_density_plots(self, rnd_dir, random_dir, env_name):
#         """Display state density plots from both experiments"""
#         try:
#             # Create a new figure for density plots
#             plt.figure(figsize=(15, 5))
            
#             # RND density plot
#             plt.subplot(1, 2, 1)
#             rnd_density = plt.imread(os.path.join(rnd_dir, 'state_density.png'))
#             plt.imshow(rnd_density)
#             plt.title('RND State Density')
#             plt.axis('off')
            
#             # Random density plot
#             plt.subplot(1, 2, 2)
#             random_density = plt.imread(os.path.join(random_dir, 'state_density.png'))
#             plt.imshow(random_density)
#             plt.title('Random State Density')
#             plt.axis('off')
            
#             plt.suptitle(f'State Density Comparison - {env_name}')
            
#             # Save density comparison
#             save_path = f'{env_name.lower().replace("-", "_")}_density_comparison.png'
#             plt.savefig(save_path)
#             plt.close()
#             logging.info(f"Saved density comparison plot to {save_path}")
            
#         except Exception as e:
#             logging.error(f"Error creating density plots: {str(e)}")

# def main():
#     runner = ExplorationExperimentRunner()
    
#     # Define experiments
#     experiments = [
#         {
#             "env": "PointmassEasy-v0",
#             "experiments": [
#                 {
#                     "name": "q6_env1_rnd",
#                     "params": "--env_name PointmassEasy-v0 --use_rnd --unsupervised_exploration --exp_name q6_env1_rnd"
#                 },
#                 {
#                     "name": "q6_env1_random",
#                     "params": "--env_name PointmassEasy-v0 --unsupervised_exploration --exp_name q6_env1_random"
#                 }
#             ]
#         },
#         {
#             "env": "PointmassHard-v0",
#             "experiments": [
#                 {
#                     "name": "q6_env2_rnd",
#                     "params": "--env_name PointmassHard-v0 --use_rnd --unsupervised_exploration --exp_name q6_env2_rnd"
#                 },
#                 {
#                     "name": "q6_env2_random",
#                     "params": "--env_name PointmassHard-v0 --unsupervised_exploration --exp_name q6_env2_random"
#                 }
#             ]
#         }
#     ]
    
#     # Run experiments and create plots
#     for env_config in experiments:
#         env_name = env_config["env"]
#         logging.info(f"\nProcessing environment: {env_name}")
        
#         # for exp in env_config["experiments"]:
#         #     runner.run_experiment(exp["name"], exp["params"])
        
#         runner.create_plots(
#             env_name,
#             env_config["experiments"][0]["name"],
#             env_config["experiments"][1]["name"]
#         )

# if __name__ == "__main__":
#     main()


import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import logging

logging.basicConfig(level=logging.INFO)

class ExplorationPlotter:
    def __init__(self, base_dir="rob831/data"):
        self.base_dir = base_dir
        
    def load_experiment_data(self, exp_name):
        """Load tensorboard data from experiment"""
        try:
            # Find the latest experiment directory for this name
            pattern = os.path.join(self.base_dir, f"hw4_part2_expl_{exp_name}*")
            exp_dirs = [d for d in os.listdir(self.base_dir) if d.startswith(f"hw4_part2_expl_{exp_name}")]
            
            if not exp_dirs:
                logging.error(f"No directories found matching pattern {pattern}")
                return None
                
            # Get the most recent directory
            exp_dir = sorted(exp_dirs)[-1]
            exp_path = os.path.join(self.base_dir, exp_dir)
            logging.info(f"Loading data from {exp_path}")
            
            # Load tensorboard data
            event_acc = EventAccumulator(exp_path)
            event_acc.Reload()
            
            # Extract relevant metrics
            try:
                returns = [(s.step, s.value) for s in event_acc.Scalars('Train_AverageReturn')]
                _, values = zip(*returns)
            except KeyError:
                logging.error(f"Could not find 'Train_AverageReturn' in the events file")
                return None
            
            return {
                'Returns': np.array(values),
                'Dir': exp_path
            }
        except Exception as e:
            logging.error(f"Error loading data for {exp_name}: {str(e)}")
            return None

    def create_plots(self, env_name, rnd_exp_name, random_exp_name):
        """Create comparison plots"""
        try:
            # Load data
            rnd_data = self.load_experiment_data(rnd_exp_name)
            random_data = self.load_experiment_data(random_exp_name)
            
            if rnd_data is None or random_data is None:
                logging.error("Could not create plots due to missing data")
                return
            
            # Create learning curves plot
            plt.figure(figsize=(10, 6))
            plt.plot(rnd_data['Returns'], label='RND Exploration')
            plt.plot(random_data['Returns'], label='Random Exploration')
            plt.xlabel('Iteration')
            plt.ylabel('Average Return')
            plt.title(f'RND vs Random Exploration on {env_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            save_path = f'{env_name.lower().replace("-", "_")}_comparison.png'
            plt.savefig(save_path)
            plt.close()
            logging.info(f"Saved comparison plot to {save_path}")
            
            # Display state density plots
            self.display_density_plots(rnd_data['Dir'], random_data['Dir'], env_name)
            
        except Exception as e:
            logging.error(f"Error creating plots for {env_name}: {str(e)}")

    def display_density_plots(self, rnd_dir, random_dir, env_name):
        """Display state density plots from both experiments"""
        try:
            # Create a new figure for density plots
            plt.figure(figsize=(15, 5))
            
            # Load density plots (looking for curr_state_density.png)
            rnd_density_path = os.path.join(rnd_dir, 'curr_state_density.png')
            random_density_path = os.path.join(random_dir, 'curr_state_density.png')
            
            if not os.path.exists(rnd_density_path) or not os.path.exists(random_density_path):
                logging.warning(f"Density plots not found in {rnd_dir} or {random_dir}")
                return
            
            # RND density plot
            plt.subplot(1, 2, 1)
            rnd_density = plt.imread(rnd_density_path)
            plt.imshow(rnd_density)
            plt.title('RND State Density')
            plt.axis('off')
            
            # Random density plot
            plt.subplot(1, 2, 2)
            random_density = plt.imread(random_density_path)
            plt.imshow(random_density)
            plt.title('Random State Density')
            plt.axis('off')
            
            plt.suptitle(f'State Density Comparison - {env_name}')
            
            save_path = f'{env_name.lower().replace("-", "_")}_density_comparison.png'
            plt.savefig(save_path)
            plt.close()
            logging.info(f"Saved density comparison plot to {save_path}")
            
        except Exception as e:
            logging.error(f"Error creating density plots: {str(e)}")

def main():
    plotter = ExplorationPlotter()
    
    # Define experiments
    experiments = [
        {
            "env": "PointmassEasy-v0",
            "rnd_exp": "q6_env1_rnd",
            "random_exp": "q6_env1_random"
        },
        {
            "env": "PointmassHard-v0",
            "rnd_exp": "q6_env2_rnd",
            "random_exp": "q6_env2_random"
        }
    ]
    
    # Create plots for each environment
    for exp in experiments:
        logging.info(f"\nProcessing environment: {exp['env']}")
        plotter.create_plots(
            exp["env"],
            exp["rnd_exp"],
            exp["random_exp"]
        )

if __name__ == "__main__":
    main()