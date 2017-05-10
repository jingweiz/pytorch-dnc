from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.optim as optim

from utils.helpers import Experience

class Agent(object):
    def __init__(self, args, env_prototype, circuit_prototype):
        # logging
        self.mode = args.mode                       # NOTE: when mode==2 we visualize accessor states
        self.logger = args.logger

        # prototypes for env & model & memory
        self.env_prototype = env_prototype          # NOTE: instantiated in inherited Agents
        self.env_params = args.env_params
        self.circuit_prototype = circuit_prototype  # NOTE: instantiated in inherited Agents
        self.circuit_params = args.circuit_params

        # TODO: let's decide what to save later
        # params
        self.model_name = args.model_name           # NOTE: will save the current model to model_name
        self.model_file = args.model_file           # NOTE: will load pretrained model_file if not None

        self.render = args.render
        self.visualize = args.visualize
        if self.visualize:
            self.vis = args.vis
            self.refs = args.refs

        self.save_best = args.save_best
        if self.save_best:
            self.best_step   = None                 # NOTE: achieves best_reward at this step
            self.best_reward = None                 # NOTE: only save a new model if achieves higher reward

        self.use_cuda = args.use_cuda
        self.dtype = args.dtype

        # agent_params
        # criteria and optimizer
        self.criteria = args.criteria
        self.optim = args.optim
        # hyperparameters
        self.steps = args.steps
        self.batch_size = args.batch_size
        self.early_stop = args.early_stop
        self.clip_grad = args.clip_grad
        # self.clip_value = args.clip_value
        self.lr = args.lr
        self.optim_eps = args.optim_eps
        self.optim_alpha = args.optim_alpha
        self.eval_freq = args.eval_freq
        self.eval_steps = args.eval_steps
        self.prog_freq = args.prog_freq
        self.test_nepisodes = args.test_nepisodes

    def _reset_experience(self):
        self.experience = Experience(state0 = None,
                                     action = None,
                                     reward = None,
                                     state1 = None,
                                     terminal1 = False)

    def _load_model(self, model_file):
        if model_file:
            self.logger.warning("Loading Model: " + self.model_file + " ...")
            self.circuit.load_state_dict(torch.load(model_file))
            self.logger.warning("Loaded  Model: " + self.model_file + " ...")
        else:
            self.logger.warning("No Pretrained Model. Will Train From Scratch.")

    def _save_model(self, step, curr_reward=0.):
        self.logger.warning("Saving Model    @ Step: " + str(step) + ": " + self.model_name + " ...")
        if self.save_best:
            if self.best_step is None:
                self.best_step   = step
                self.best_reward = curr_reward
            if curr_reward >= self.best_reward:
                self.best_step   = step
                self.best_reward = curr_reward
                torch.save(self.circuit.state_dict(), self.model_name)
            self.logger.warning("Saved  Model    @ Step: " + str(step) + ": " + self.model_name + ". {Best Step: " + str(self.best_step) + " | Best Reward: " + str(self.best_reward) + "}")
        else:
            torch.save(self.circuit.state_dict(), self.model_name)
            self.logger.warning("Saved  Model    @ Step: " + str(step) + ": " + self.model_name + ".")

    def _forward(self, observation):
        raise NotImplementedError("not implemented in base calss")

    def _backward(self, reward, terminal):
        raise NotImplementedError("not implemented in base calss")

    def fit_model(self):    # training
        raise NotImplementedError("not implemented in base calss")

    def _eval_model(self):  # evaluation during training
        raise NotImplementedError("not implemented in base calss")

    def test_model(self):   # testing pre-trained models
        raise NotImplementedError("not implemented in base calss")
