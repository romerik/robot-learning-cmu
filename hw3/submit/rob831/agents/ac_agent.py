from collections import OrderedDict

from rob831.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from rob831.infrastructure.replay_buffer import ReplayBuffer
from rob831.infrastructure.utils import *
from rob831.policies.MLP_policy import MLPPolicyAC
import torch
from .base_agent import BaseAgent
from rob831.infrastructure import pytorch_util as ptu

class ACAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(ACAgent, self).__init__()

        self.env = env
        self.agent_params = agent_params

        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']

        self.actor = MLPPolicyAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
        )
        self.critic = BootstrappedContinuousCritic(self.agent_params)

        self.replay_buffer = ReplayBuffer()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # TODO Implement the following pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic

        # advantage = estimate_advantage(...)

        # for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor

        # Convert numpy arrays to tensors first
        # Convert inputs to tensors if they aren't already
        ob_no = ptu.from_numpy(ob_no) if not torch.is_tensor(ob_no) else ob_no
        ac_na = ptu.from_numpy(ac_na) if not torch.is_tensor(ac_na) else ac_na
        re_n = ptu.from_numpy(re_n) if not torch.is_tensor(re_n) else re_n
        next_ob_no = ptu.from_numpy(next_ob_no) if not torch.is_tensor(next_ob_no) else next_ob_no
        terminal_n = ptu.from_numpy(terminal_n) if not torch.is_tensor(terminal_n) else terminal_n

        loss = OrderedDict()

        # Update critic multiple times
        critic_loss = []
        for _ in range(self.agent_params['num_critic_updates_per_agent_update']):
            # Compute critic targets: r + γV(s')
            critic_targets = re_n + self.gamma * self.critic.forward(next_ob_no) * (1 - terminal_n)
            critic_targets = critic_targets.detach()  # Don't backprop through targets
            
            # Update critic
            critic_loss.append(self.critic.update(ob_no, ac_na, next_ob_no, re_n, terminal_n))

        # Save mean critic loss
        loss['Loss_Critic'] = np.mean(critic_loss)

        # Estimate advantages for actor update
        advantages = self.estimate_advantage(ob_no, next_ob_no, re_n, terminal_n)

        # Update actor multiple times
        actor_loss = []
        for _ in range(self.agent_params['num_actor_updates_per_agent_update']):
            actor_loss.append(self.actor.update(ob_no, ac_na, adv_n=advantages))

        # Save mean actor loss
        loss['Loss_Actor'] = np.mean(actor_loss)

        return loss

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
        # TODO Implement the following pseudocode:
        # 1) query the critic with ob_no, to get V(s)
        # 2) query the critic with next_ob_no, to get V(s')
        # 3) estimate the Q value as Q(s, a) = r(s, a) + gamma*V(s')
        # HINT: Remember to cut off the V(s') term (ie set it to 0) at terminal states (ie terminal_n=1)
        # 4) calculate advantage (adv_n) as A(s, a) = Q(s, a) - V(s)

        # Get value estimates from critic
        v_t = self.critic.forward(ob_no)  # V(s)
        v_tp1 = self.critic.forward(next_ob_no)  # V(s')
        
        # Calculate Q(s,a) = r + γV(s') with terminal state handling
        q_t = re_n + self.gamma * v_tp1 * (1 - terminal_n)

        # Calculate advantages A(s,a) = Q(s,a) - V(s)
        adv_n = q_t - v_t

        if self.standardize_advantages:
            adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)
        return adv_n

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size)
