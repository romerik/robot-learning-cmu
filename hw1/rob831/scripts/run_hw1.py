import os
import time
import sys

from rob831.infrastructure.rl_trainer import RL_Trainer
from rob831.agents.bc_agent import BCAgent
from rob831.policies.loaded_gaussian_policy import LoadedGaussianPolicy
import matplotlib.pyplot as plt
import numpy as np

class BC_Trainer(object):

    def __init__(self, params):

        #######################
        ## AGENT PARAMS
        #######################

        agent_params = {
            'n_layers': params['n_layers'],
            'size': params['size'],
            'learning_rate': params['learning_rate'],
            'max_replay_buffer_size': params['max_replay_buffer_size'],
            }

        self.params = params
        self.params['agent_class'] = BCAgent ## HW1: you will modify this
        self.params['agent_params'] = agent_params

        ################
        ## RL TRAINER
        ################

        self.rl_trainer = RL_Trainer(self.params) ## HW1: you will modify this

        #######################
        ## LOAD EXPERT POLICY
        #######################

        print('Loading expert policy from...', self.params['expert_policy_file'])
        self.loaded_expert_policy = LoadedGaussianPolicy(self.params['expert_policy_file'])
        print('Done restoring expert policy...')

    def run_training_loop(self):

        return self.rl_trainer.run_training_loop(
            n_iter=self.params['n_iter'],
            initial_expertdata=self.params['expert_data'],
            collect_policy=self.rl_trainer.agent.actor,
            eval_policy=self.rl_trainer.agent.actor,
            relabel_with_expert=self.params['do_dagger'],
            expert_policy=self.loaded_expert_policy,
        )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_policy_file', '-epf', type=str, required=True)  # relative to where you're running this script from
    parser.add_argument('--expert_data', '-ed', type=str, required=True) #relative to where you're running this script from
    parser.add_argument('--env_name', '-env', type=str, help='choices: Ant-v2, Humanoid-v2, Walker2d-v2, HalfCheetah-v2, Hopper-v2', required=True)
    parser.add_argument('--exp_name', '-exp', type=str, default='pick an experiment name', required=True)
    parser.add_argument('--do_dagger', action='store_true')
    parser.add_argument('--ep_len', type=int, default=1000)

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1000)  # number of gradient steps for training policy (per iter in n_iter)
    parser.add_argument('--n_iter', '-n', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=1000)  # training data collected (in the env) during each iteration
    parser.add_argument('--eval_batch_size', type=int,
                        default=5000)  # eval data collected (in the env) for logging metrics
    parser.add_argument('--train_batch_size', type=int,
                        default=1000)  # number of sampled data points to be used per gradient/train step

    parser.add_argument('--n_layers', type=int, default=2)  # depth, of policy to be learned
    parser.add_argument('--size', type=int, default=64)  # width of each layer, of policy to be learned
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)  # LR for supervised learning

    parser.add_argument('--video_log_freq', type=int, default=5)
    parser.add_argument('--scalar_log_freq', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', type=int, default=0)
    parser.add_argument('--max_replay_buffer_size', type=int, default=1000000)
    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    # convert args to dictionary
    params = vars(args)

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    if args.do_dagger:
        logdir_prefix = 'q2_'
        assert args.n_iter>1, ('DAGGER needs more than 1 iteration (n_iter>1) of training, to iteratively query the expert and train (after 1st warmstarting from behavior cloning).')
    else:
        logdir_prefix = 'q1_'
        assert args.n_iter==1, ('Vanilla behavior cloning collects expert data just once (n_iter=1)')

    ## directory for logging
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = logdir_prefix + args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)


    ###################
    ### RUN TRAINING
    ###################

    #Gather avg return and std for graphing
    # avg_returns = []
    # std_returns = []
    # expert_avg_returns = []
    # expert_std_returns = []

    # for num_train_steps in range(1000, 20000, 1000):
    #     print("-"*50)
    #     print("Training with parameter num_agent_train_steps_per_iter = {}".format(num_train_steps))
    #     params['num_agent_train_steps_per_iter'] = num_train_steps
    #     trainer = BC_Trainer(params)
    #     avg_return, std_return, expert_avg_return, expert_std_return = trainer.run_training_loop()
    #     avg_returns.append(avg_return)
    #     std_returns.append(std_return)
    #     expert_avg_returns.append(expert_avg_return)
    #     expert_std_returns.append(expert_std_return)

    # # Create the plot
    # plt.figure(figsize=(10, 6))

    # num_train_steps = list(range(1000, 20000, 1000))

    # # Plot average returns
    # plt.plot(num_train_steps, avg_returns, 'b-', label='Average Returns')

    # # Plot standard deviation
    # plt.plot(num_train_steps, std_returns, 'r-', label='Standard Deviation')

    # # Plot expert average returns
    # plt.plot(num_train_steps, expert_avg_returns, 'g-', label='Expert Average Returns')

    # # Plot expert standard deviation
    # plt.plot(num_train_steps, expert_std_returns, 'y-', label='Expert Standard Deviation')

    # # Add labels and title
    # plt.xlabel('Number of Training Steps per Iteration')
    # plt.ylabel('Returns')
    # plt.title('Average Returns and Standard Deviation vs. Training Steps')

    # # Add legend
    # plt.legend()

    # # Show grid
    # plt.grid(True, linestyle='--', alpha=0.7)

    # # Save the plot instead of showing it
    # plt.tight_layout()
    # plt.savefig('training_steps_plot.png')
    # plt.close()  # Close the figure to free up memory

    # print("Plot saved as 'training_steps_plot.png'")

    trainer = BC_Trainer(params)
    trainer.run_training_loop()

    # eval_avg_returns, eval_std_returns, expert_avg_returns, bc_avg_returns = trainer.run_training_loop()

    # # Create the plot
    # plt.figure(figsize=(10, 6))

    # # Plot DAgger learning curve
    # plt.errorbar(np.arange(params['n_iter']), eval_avg_returns, yerr=eval_std_returns, fmt='-o', capsize=5, label='DAgger')

    # # Plot expert and BC performance
    # plt.axhline(y=expert_avg_returns[0], color='r', linestyle='--', label='Expert')
    # plt.axhline(y=bc_avg_returns[0], color='g', linestyle='--', label='Behavioral Cloning')

    # plt.xlabel('DAgger Iterations')
    # plt.ylabel('Mean Return')
    # plt.title('DAgger Learning Curve - Humanoid-v2')
    # plt.legend()
    # plt.grid(True)

    # plt.tight_layout()
    # plt.savefig('dagger_learning_curve_humanoid.png')
    # plt.close()

    # print("Learning curve saved as 'dagger_learning_curve_humanoid.png'")

if __name__ == "__main__":
    main()
