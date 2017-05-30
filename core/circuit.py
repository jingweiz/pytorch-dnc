from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Circuit(nn.Module):   # NOTE: basically this whole module is treated as a custom rnn cell
    def __init__(self, args):
        super(Circuit, self).__init__()
        # logging
        self.logger = args.logger
        # params
        self.use_cuda = args.use_cuda
        self.dtype = args.dtype
        # params
        self.batch_size = args.batch_size
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.hidden_dim = args.hidden_dim
        self.num_write_heads = args.num_write_heads
        self.num_read_heads = args.num_read_heads
        self.mem_hei = args.mem_hei
        self.mem_wid = args.mem_wid
        self.clip_value = args.clip_value

        # functional components
        self.controller_params = args.controller_params
        self.accessor_params = args.accessor_params

        # now we fill in the missing values for each module
        self.read_vec_dim = self.num_read_heads * self.mem_wid
        # controller
        self.controller_params.batch_size = self.batch_size
        self.controller_params.input_dim = self.input_dim
        self.controller_params.read_vec_dim = self.read_vec_dim
        self.controller_params.output_dim = self.output_dim
        self.controller_params.hidden_dim = self.hidden_dim
        self.controller_params.mem_hei = self.mem_hei
        self.controller_params.mem_wid = self.mem_wid
        self.controller_params.clip_value = self.clip_value
        # accessor: {write_heads, read_heads, memory}
        self.accessor_params.batch_size = self.batch_size
        self.accessor_params.hidden_dim = self.hidden_dim
        self.accessor_params.num_write_heads = self.num_write_heads
        self.accessor_params.num_read_heads = self.num_read_heads
        self.accessor_params.mem_hei = self.mem_hei
        self.accessor_params.mem_wid = self.mem_wid
        self.accessor_params.clip_value = self.clip_value

        self.logger.warning("<-----------------------------======> Circuit:    {Controller, Accessor}")

    def _init_weights(self):
        raise NotImplementedError("not implemented in base calss")

    def print_model(self):
        self.logger.warning("<-----------------------------======> Circuit:    {Overall Architecture}")
        self.logger.warning(self)

    def _reset_states(self): # should be called at the beginning of forwarding a new input sequence
        # we first reset the previous read vector
        self.read_vec_vb = Variable(self.read_vec_ts).type(self.dtype)
        # we then reset the controller's hidden state
        self.controller._reset_states()
        # we then reset the write/read weights of heads
        self.accessor._reset_states()

    def _reset(self):
        self._init_weights()
        self.type(self.dtype)
        self.print_model()
        # reset internal states
        self.read_vec_ts = torch.zeros(self.batch_size, self.read_vec_dim).fill_(1e-6)
        self._reset_states()

    def forward(self, input_vb):
        # NOTE: the operation order must be the following: control, access{write, read}, output

        # 1. first feed {input, read_vec_{t-1}} to controller
        hidden_vb = self.controller.forward(input_vb, self.read_vec_vb)
        # 2. then we write to memory_{t-1} to get memory_{t}; then read from memory_{t} to get read_vec_{t}
        self.read_vec_vb = self.accessor.forward(hidden_vb)
        # 3. finally we concat the output from the controller and the current read_vec_{t} to get the final output
        output_vb = self.hid_to_out(torch.cat((hidden_vb.view(-1, self.hidden_dim),
                                               self.read_vec_vb.view(-1, self.read_vec_dim)), 1))

        # we clip the output values here
        return F.sigmoid(torch.clamp(output_vb, min=-self.clip_value, max=self.clip_value)).view(1, self.batch_size, self.output_dim)
