import os
import time

from rob831.infrastructure.rl_trainer import RL_Trainer
from rob831.agents.pg_agent import PGAgent
import matplotlib.pyplot as plt
import numpy as np
import itertools

class PG_Trainer(object):

    def __init__(self, params):

        #####################
        ## SET AGENT PARAMS
        #####################

        computation_graph_args = {
            'n_layers': params['n_layers'],
            'size': params['size'],
            'learning_rate': params['learning_rate'],
            }

        estimate_advantage_args = {
            'gamma': params['discount'],
            'standardize_advantages': not(params['dont_standardize_advantages']),
            'reward_to_go': params['reward_to_go'],
            'nn_baseline': params['nn_baseline'],
            'gae_lambda': params['gae_lambda'],
        }

        train_args = {
            'num_agent_train_steps_per_iter': params['num_agent_train_steps_per_iter'],
        }

        agent_params = {**computation_graph_args, **estimate_advantage_args, **train_args}

        self.params = params
        self.params['agent_class'] = PGAgent
        self.params['agent_params'] = agent_params
        self.params['batch_size_initial'] = self.params['batch_size']

        ################
        ## RL TRAINER
        ################

        self.rl_trainer = RL_Trainer(self.params)

    def run_training_loop(self):

        iterations_logs = self.rl_trainer.run_training_loop(
            self.params['n_iter'],
            collect_policy = self.rl_trainer.agent.actor,
            eval_policy = self.rl_trainer.agent.actor,
            )

        return iterations_logs

def run_experiment(params):
    trainer = PG_Trainer(params)
    return trainer.run_training_loop()

def create_q5_sb_graph_from_logs(experiments_logs):
    # Create small batch graph
    plt.figure(figsize=(10, 6))
    for exp_name in ['q1_sb_no_rtg_dsa', 'q1_sb_rtg_dsa', 'q1_sb_rtg_na']:
        if exp_name in experiments_logs:
            data = [log['Eval_AverageReturn'] for log in experiments_logs[exp_name]]
            plt.plot(range(len(data)), data, label=exp_name)
    
    plt.xlabel('Iteration')
    plt.ylabel('Average Return')
    plt.title('Small Batch Experiments')
    plt.legend()
    plt.savefig('small_batch_experiments.png')
    plt.close()
    
    # Create large batch graph
    plt.figure(figsize=(10, 6))
    for exp_name in ['q1_lb_no_rtg_dsa', 'q1_lb_rtg_dsa', 'q1_lb_rtg_na']:
        if exp_name in experiments_logs:
            data = [log['Eval_AverageReturn'] for log in experiments_logs[exp_name]]
            plt.plot(range(len(data)), data, label=exp_name)
    
    plt.xlabel('Iteration')
    plt.ylabel('Average Return')
    plt.title('Large Batch Experiments')
    plt.legend()
    plt.savefig('large_batch_experiments.png')
    plt.close()

    print("Graphs have been saved as 'small_batch_experiments.png' and 'large_batch_experiments.png'")


def run_question_5_experiments(params, data_path, args):
    experiments = {
        'q1_sb_no_rtg_dsa': {'exp_name': 'q1_sb_no_rtg_dsa', 'batch_size': 1000, 'dont_standardize_advantages': True, 'reward_to_go': False},
        'q1_sb_rtg_dsa': {'exp_name': 'q1_sb_rtg_dsa', 'batch_size': 1000, 'dont_standardize_advantages': True, 'reward_to_go': True},
        'q1_sb_rtg_na': {'exp_name': 'q1_sb_rtg_na', 'batch_size': 1000, 'dont_standardize_advantages': False, 'reward_to_go': True},
        'q1_lb_no_rtg_dsa': {'exp_name': 'q1_lb_no_rtg_dsa', 'batch_size': 5000, 'dont_standardize_advantages': True, 'reward_to_go': False},
        'q1_lb_rtg_dsa': {'exp_name': 'q1_lb_rtg_dsa', 'batch_size': 5000, 'dont_standardize_advantages': True, 'reward_to_go': True},
        'q1_lb_rtg_na': {'exp_name': 'q1_lb_rtg_na', 'batch_size': 5000, 'dont_standardize_advantages': False, 'reward_to_go': True},
    }
    experiments_logs = {}

    for exp_name, exp_params in experiments.items():
        print(f"Running experiment: {exp_name}")

        exp_params = {**params, **exp_params}  # Merge with default params

        exp_params['logdir'] = os.path.join(data_path, f"{exp_name}_{args.env_name}_{time.strftime('%d-%m-%Y_%H-%M-%S')}")
        if not os.path.exists(exp_params['logdir']):
            os.makedirs(exp_params['logdir'])
        
        experiments_logs[exp_name] = run_experiment(exp_params)
    
    create_q5_sb_graph_from_logs(experiments_logs)

def run_inverted_pendulum_experiment(params, data_path, args):
    # Set the specific parameters for this experiment
    exp_params = {
        'env_name': 'InvertedPendulum-v4',
        'exp_name': 'q2_b55_r3e-2',
        'batch_size': 55,
        'learning_rate': 0.03,
        'n_iter': 100,
        'reward_to_go': True,
        'ep_len': 1000,
        'discount': 0.9,
    }

    # Merge with default params
    exp_params = {**params, **exp_params}

    # Set up logging directory
    exp_params['logdir'] = os.path.join(data_path, f"{exp_params['exp_name']}_{time.strftime('%d-%m-%Y_%H-%M-%S')}")
    if not os.path.exists(exp_params['logdir']):
        os.makedirs(exp_params['logdir'])

    # Run the experiment
    print(f"Running InvertedPendulum-v4 experiment with b*={exp_params['batch_size']}, r*={exp_params['learning_rate']}")
    experiment_logs = run_experiment(exp_params)

    # Extract the relevant data for plotting
    iterations = range(len(experiment_logs))
    average_returns = [log['Eval_AverageReturn'] for log in experiment_logs]

    # Create the learning curve plot
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, average_returns)
    plt.xlabel('Iteration')
    plt.ylabel('Average Return')
    plt.title(f"InvertedPendulum-v4 Learning Curve (b*={exp_params['batch_size']}, r*={exp_params['learning_rate']})")
    plt.ylim(0, 1050)  # Set y-axis limit to slightly above the maximum score
    plt.axhline(y=1000, color='r', linestyle='--', label='Optimum Score')
    plt.legend()

    # Save the plot
    plt.savefig('inverted_pendulum_learning_curve.png')
    plt.close()

    print("Learning curve has been saved as 'inverted_pendulum_learning_curve.png'")

    # Return the data in case it's needed for further analysis
    return iterations, average_returns

def plot_learning_curve(iterations_logs, params):
    iterations = range(len(iterations_logs))
    average_returns = [log['Eval_AverageReturn'] for log in iterations_logs]

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, average_returns)
    plt.xlabel('Iteration')
    plt.ylabel('Average Return')
    plt.title(f"Learning Curve for {params['env_name']}")
    plt.savefig(f"{params['logdir']}/learning_curve.png")
    plt.close()

    print(f"Learning curve has been saved as '{params['logdir']}/learning_curve.png'")


def run_halfcheetah_experiments(params, data_path, args):
    batch_sizes = [10000, 30000, 50000]
    learning_rates = [0.005, 0.01, 0.02]
    
    experiments = list(itertools.product(batch_sizes, learning_rates))
    results = {}

    for batch_size, lr in experiments:
        exp_name = f"q4_search_b{batch_size}_lr{lr}_rtg_nnbaseline"
        exp_params = {
            'env_name': 'HalfCheetah-v4',
            'exp_name': exp_name,
            'batch_size': batch_size,
            'learning_rate': lr,
            'n_iter': 100,
            'ep_len': 150,
            'discount': 0.95,
            'reward_to_go': True,
            'nn_baseline': True,
            'n_layers': 2,
            'size': 32,
        }

        # Merge with default params
        exp_params = {**params, **exp_params}

        # Set up logging directory
        exp_params['logdir'] = os.path.join(data_path, f"{exp_name}_{time.strftime('%d-%m-%Y_%H-%M-%S')}")
        if not os.path.exists(exp_params['logdir']):
            os.makedirs(exp_params['logdir'])

        # Run the experiment
        print(f"Running experiment: {exp_name}")
        experiment_logs = run_experiment(exp_params)

        # Store results
        results[(batch_size, lr)] = [log['Eval_AverageReturn'] for log in experiment_logs]

    # Plotting
    plt.figure(figsize=(12, 8))
    for (batch_size, lr), returns in results.items():
        plt.plot(range(len(returns)), returns, label=f'b={batch_size}, lr={lr}')

    plt.xlabel('Iteration')
    plt.ylabel('Average Return')
    plt.title('HalfCheetah-v4 Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('halfcheetah_learning_curves.png')
    plt.close()

    print("Learning curves have been saved as 'halfcheetah_learning_curves.png'")

    # Find optimal parameters
    final_returns = {params: returns[-1] for params, returns in results.items()}
    optimal_params = max(final_returns, key=final_returns.get)
    
    print(f"Optimal parameters: batch_size (b*) = {optimal_params[0]}, learning_rate (r*) = {optimal_params[1]}")
    print(f"Best final return: {final_returns[optimal_params]}")

    return results, optimal_params


def run_halfcheetah_optimal_experiments(params, data_path, args):
    b_star = 30000
    r_star = 0.02
    
    experiments = [
        {'name': f'q4_b{b_star}_r{r_star}', 'rtg': False, 'nn_baseline': False},
        {'name': f'q4_b{b_star}_r{r_star}_rtg', 'rtg': True, 'nn_baseline': False},
        {'name': f'q4_b{b_star}_r{r_star}_nnbaseline', 'rtg': False, 'nn_baseline': True},
        {'name': f'q4_b{b_star}_r{r_star}_rtg_nnbaseline', 'rtg': True, 'nn_baseline': True},
    ]
    
    results = {}

    for exp in experiments:
        exp_params = {
            'env_name': 'HalfCheetah-v4',
            'exp_name': exp['name'],
            'batch_size': b_star,
            'learning_rate': r_star,
            'n_iter': 100,
            'ep_len': 150,
            'discount': 0.95,
            'reward_to_go': exp['rtg'],
            'nn_baseline': exp['nn_baseline'],
            'n_layers': 2,
            'size': 32,
        }

        # Merge with default params
        exp_params = {**params, **exp_params}

        # Set up logging directory
        exp_params['logdir'] = os.path.join(data_path, f"{exp['name']}_{time.strftime('%d-%m-%Y_%H-%M-%S')}")
        if not os.path.exists(exp_params['logdir']):
            os.makedirs(exp_params['logdir'])

        # Run the experiment
        print(f"Running experiment: {exp['name']}")
        experiment_logs = run_experiment(exp_params)

        # Store results
        results[exp['name']] = [log['Eval_AverageReturn'] for log in experiment_logs]

    # Plotting
    plt.figure(figsize=(12, 8))
    for name, returns in results.items():
        plt.plot(range(len(returns)), returns, label=name)

    plt.xlabel('Iteration')
    plt.ylabel('Average Return')
    plt.title(f'HalfCheetah-v4 Learning Curves (b*={b_star}, r*={r_star})')
    plt.legend()
    plt.grid(True)
    plt.savefig('halfcheetah_optimal_learning_curves.png')
    plt.close()

    print("Learning curves have been saved as 'halfcheetah_optimal_learning_curves.png'")

    # Check if the rtg + nn_baseline run achieved close to 200
    rtg_baseline_return = results[f'q4_b{b_star}_r{r_star}_rtg_nnbaseline'][-1]
    print(f"Final average return for RTG + NN Baseline: {rtg_baseline_return}")
    if abs(rtg_baseline_return - 200) <= 20:  # Within 10% of 200
        print("The run with both reward-to-go and baseline achieved an average score close to 200.")
    else:
        print("The run with both reward-to-go and baseline did not achieve an average score close to 200.")

    return results

def run_hopper_gae_lambda_experiments(params, data_path, args):
    lambda_values = [0, 0.95, 0.99, 1]
    
    experiments = [
        {'name': f'q5_b2000_r0.001_lambda{lambda_val}', 'gae_lambda': lambda_val}
        for lambda_val in lambda_values
    ]
    
    results = {}

    for exp in experiments:
        exp_params = {
            'env_name': 'Hopper-v4',
            'exp_name': exp['name'],
            'ep_len': 1000,
            'discount': 0.99,
            'n_iter': 300,
            'n_layers': 2,
            'size': 32,
            'batch_size': 2000,
            'learning_rate': 0.001,
            'reward_to_go': True,
            'nn_baseline': True,
            'action_noise_std': 0.5,
            'gae_lambda': exp['gae_lambda'],
        }

        # Merge with default params
        exp_params = {**params, **exp_params}

        # Set up logging directory
        exp_params['logdir'] = os.path.join(data_path, f"{exp['name']}_{time.strftime('%d-%m-%Y_%H-%M-%S')}")
        if not os.path.exists(exp_params['logdir']):
            os.makedirs(exp_params['logdir'])

        # Run the experiment
        print(f"Running experiment: {exp['name']}")
        experiment_logs = run_experiment(exp_params)

        # Store results
        results[exp['name']] = [log['Eval_AverageReturn'] for log in experiment_logs]

    # Plotting
    plt.figure(figsize=(12, 8))
    for name, returns in results.items():
        plt.plot(range(len(returns)), returns, label=name)

    plt.xlabel('Iteration')
    plt.ylabel('Average Return')
    plt.title('Hopper-v4 Learning Curves with Different GAE-λ Values')
    plt.legend()
    plt.grid(True)
    plt.savefig('hopper_gae_lambda_learning_curves.png')
    plt.close()

    print("Learning curves have been saved as 'hopper_gae_lambda_learning_curves.png'")

    # Find the best performing λ
    best_lambda = max(results, key=lambda k: np.max(results[k]))
    best_score = np.max(results[best_lambda])
    print(f"Best performing λ: {best_lambda.split('lambda')[1]}")
    print(f"Best average score: {best_score}")

    if abs(best_score - 400) <= 40:  # Within 10% of 400
        print("The best run achieved an average score close to 400.")
    else:
        print("The best run did not achieve an average score close to 400.")

    return results

def run_halfcheetah_experiments_7_2_1(params, data_path, args):
    experiments = [
        {'name': 'q4_search_b10000_lr0.02', 'rtg': False, 'nn_baseline': False},
        {'name': 'q4_search_b10000_lr0.02_rtg', 'rtg': True, 'nn_baseline': False},
        {'name': 'q4_search_b10000_lr0.02_nnbaseline', 'rtg': False, 'nn_baseline': True},
        {'name': 'q4_search_b10000_lr0.02_rtg_nnbaseline', 'rtg': True, 'nn_baseline': True},
    ]
    
    results = {}

    for exp in experiments:
        exp_params = {
            'env_name': 'HalfCheetah-v4',
            'exp_name': exp['name'],
            'batch_size': 10000,
            'learning_rate': 0.02,
            'n_iter': 100,
            'ep_len': 150,
            'discount': 0.95,
            'reward_to_go': exp['rtg'],
            'nn_baseline': exp['nn_baseline'],
            'n_layers': 2,
            'size': 32,
        }

        # Merge with default params
        exp_params = {**params, **exp_params}

        # Set up logging directory
        exp_params['logdir'] = os.path.join(data_path, f"{exp['name']}_{time.strftime('%d-%m-%Y_%H-%M-%S')}")
        if not os.path.exists(exp_params['logdir']):
            os.makedirs(exp_params['logdir'])

        # Run the experiment
        print(f"Running experiment: {exp['name']}")
        trainer = PG_Trainer(exp_params)
        experiment_logs = trainer.run_training_loop()

        # Store results
        results[exp['name']] = [log['Eval_AverageReturn'] for log in experiment_logs]

    # Plotting
    plt.figure(figsize=(12, 8))
    for name, returns in results.items():
        plt.plot(range(len(returns)), returns, label=name)

    plt.xlabel('Iteration')
    plt.ylabel('Average Return')
    plt.title('HalfCheetah-v4 Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('halfcheetah_learning_curves.png')
    plt.close()

    print("Learning curves have been saved as 'halfcheetah_learning_curves.png'")

    return results

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='todo')
    parser.add_argument('--n_iter', '-n', type=int, default=200)

    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--nn_baseline', action='store_true')
    parser.add_argument('--gae_lambda', type=float, default=None)
    parser.add_argument('--dont_standardize_advantages', '-dsa', action='store_true')
    parser.add_argument('--batch_size', '-b', type=int, default=1000) #steps collected per train iteration
    parser.add_argument('--eval_batch_size', '-eb', type=int, default=400) #steps collected per eval iteration
    parser.add_argument('--train_batch_size', '-tb', type=int, default=1000) ##steps used per gradient step

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1)
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)

    parser.add_argument('--ep_len', type=int) #students shouldn't change this away from env's default
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--video_log_freq', type=int, default=-1)
    parser.add_argument('--scalar_log_freq', type=int, default=1)

    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--action_noise_std', type=float, default=0)

    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)

    # for policy gradient, we made a design decision
    # to force batch_size = train_batch_size
    # note that, to avoid confusion, you don't even have a train_batch_size argument anymore (above)
    params['train_batch_size'] = params['batch_size']

##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    # logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    # logdir = os.path.join(data_path, logdir)
    # params['logdir'] = logdir
    # if not(os.path.exists(logdir)):
    #     os.makedirs(logdir)

    ###################
    ### RUN TRAINING
    ###################

    # trainer = PG_Trainer(params)
    # iterations_logs = trainer.run_training_loop()

    # # Plot the learning curve
    # plot_learning_curve(iterations_logs, params)

    # Run the InvertedPendulum-v4 experiment
    # run_inverted_pendulum_experiment(params, data_path, args)

    # run_question_5_experiments(params, data_path, args)

    # Run the HalfCheetah-v4 experiments
    # results, optimal_params = run_halfcheetah_experiments(params, data_path, args)

    # Run the HalfCheetah-v4 experiments with optimal parameters
    # results = run_halfcheetah_optimal_experiments(params, data_path, args)

    # Run the Hopper-v4 experiments with different GAE-λ values
    # results = run_hopper_gae_lambda_experiments(params, data_path, args)

    # Run the HalfCheetah-v4 experiments
    results = run_halfcheetah_experiments_7_2_1(params, data_path, args)


if __name__ == "__main__":
    main()
