import numpy as np

# ArgMaxPolicy Implementation
class ArgMaxPolicy(object):
    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
            
        # Get Q-values from critic
        qa_values = self.critic.qa_values(observation)
        # Return action with highest Q-value
        action = np.argmax(qa_values, axis=1)
        return action.squeeze()