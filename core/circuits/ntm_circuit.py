from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn

from core.circuit import Circuit
from core.controllers.lstm_controller import LSTMController as Controller
from core.accessors.static_accessor import StaticAccessor as Accessor

class NTMCircuit(Circuit):
    def __init__(self, args):
        super(NTMCircuit, self).__init__(args)

        # functional components
        self.controller = Controller(self.controller_params)
        self.accessor = Accessor(self.accessor_params)

        # build model
        self.hid_to_out = nn.Linear(self.hidden_dim + self.read_vec_dim, self.output_dim)

        self._reset()

    def _init_weights(self):
        pass
