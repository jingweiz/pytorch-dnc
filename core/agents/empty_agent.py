from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import torch
from torch.autograd import Variable

from utils.helpers import Experience
from core.agent import Agent

class EmptyAgent(Agent):
    def __init__(self, args, env_prototype, circuit_prototype):
        super(EmptyAgent, self).__init__(args, env_prototype, circuit_prototype)
        self.logger.warning("<===================================> Empty")

        # env
        self.env_params.batch_size = args.batch_size
        self.env = self.env_prototype(self.env_params)
        self.state_shape = self.env.state_shape
        self.action_dim  = self.env.action_dim

        self._reset_experience()

    def _preprocessState(self, state):
        state_ts = torch.from_numpy(state).type(self.dtype)
        return state_ts

    def _forward(self, observation):
        # NOTE: we update the output_vb and target_vb here
        input_ts = self._preprocessState(observation[0])
        self.target_vb = Variable(self._preprocessState(observation[1]))
        self.mask_ts   = self._preprocessState(observation[2]).expand_as(self.target_vb)

        self.env.visual(input_ts, self.target_vb.data, self.mask_ts)

    def _backward(self, reward, terminal):
        pass

    def _eval_model(self):
        pass

    def fit_model(self):    # the most basic control loop, to ease integration of new envs
        self.step = 0
        should_start_new = True
        while self.step < self.steps:
            if should_start_new:
                self._reset_experience()
                self.experience = self.env.reset()
                assert self.experience.state1 is not None
                should_start_new = False
            # action = random.randrange(self.action_dim)      # thus we only randomly sample actions here, since the model hasn't been updated at all till now
            action = self._forward(self.experience.state1)
            self.experience = self.env.step(action)
            if self.experience.terminal1 or self.early_stop and (episode_steps + 1) >= self.early_stop:
                should_start_new = True

            self.step += 1
            raw_input()

    def test_model(self):
        pass
