from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.controller import Controller

class LSTMController(Controller):
    def __init__(self, args):
        super(LSTMController, self).__init__(args)

        # build model
        self.in_2_hid = nn.LSTMCell(self.input_dim + self.read_vec_dim, self.hidden_dim, 1)

        self._reset()

    def _init_weights(self):
        pass

    def forward(self, input_vb, read_vec_vb):
        self.lstm_hidden_vb = self.in_2_hid(torch.cat((input_vb.contiguous().view(-1, self.input_dim),
                                                       read_vec_vb.contiguous().view(-1, self.read_vec_dim)), 1),
                                            self.lstm_hidden_vb)

        # we clip the controller states here
        self.lstm_hidden_vb[0].clamp(min=-self.clip_value, max=self.clip_value)
        self.lstm_hidden_vb[1].clamp(min=-self.clip_value, max=self.clip_value)
        return self.lstm_hidden_vb[0]
