from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn

class Accessor(nn.Module):
    def __init__(self, args):
        super(Accessor, self).__init__()
        # logging
        self.logger = args.logger
        # params
        self.use_cuda = args.use_cuda
        self.dtype = args.dtype

        # params
        self.batch_size = args.batch_size
        self.hidden_dim = args.hidden_dim
        self.num_write_heads = args.num_write_heads
        self.num_read_heads = args.num_read_heads
        self.mem_hei = args.mem_hei
        self.mem_wid = args.mem_wid
        self.clip_value = args.clip_value

        # functional components
        self.write_head_params = args.write_head_params
        self.read_head_params = args.read_head_params
        self.memory_params = args.memory_params

        # fill in the missing values
        # write_heads
        self.write_head_params.num_heads = self.num_write_heads
        self.write_head_params.batch_size = self.batch_size
        self.write_head_params.hidden_dim = self.hidden_dim
        self.write_head_params.mem_hei = self.mem_hei
        self.write_head_params.mem_wid = self.mem_wid
        # read_heads
        self.read_head_params.num_heads = self.num_read_heads
        self.read_head_params.batch_size = self.batch_size
        self.read_head_params.hidden_dim = self.hidden_dim
        self.read_head_params.mem_hei = self.mem_hei
        self.read_head_params.mem_wid = self.mem_wid
        # memory
        self.memory_params.batch_size = self.batch_size
        self.memory_params.clip_value = self.clip_value
        self.memory_params.mem_hei = self.mem_hei
        self.memory_params.mem_wid = self.mem_wid

    def _init_weights(self):
        raise NotImplementedError("not implemented in base calss")

    def _reset_states(self):
        raise NotImplementedError("not implemented in base calss")

    def _reset(self):           # NOTE: should be called at each child's __init__
        raise NotImplementedError("not implemented in base calss")

    def visual(self):
        self.write_heads.visual()
        self.read_heads.visual()
        self.memory.visual()

    def forward(self, lstm_hidden_vb):
        raise NotImplementedError("not implemented in base calss")
