from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
from torch.autograd import Variable

from core.accessor import Accessor
from core.heads.static_write_head import StaticWriteHead as WriteHead
from core.heads.static_read_head import StaticReadHead as ReadHead
from core.memory import External2DMemory as ExternalMemory

class StaticAccessor(Accessor):
    def __init__(self, args):
        super(StaticAccessor, self).__init__(args)
        # logging
        self.logger = args.logger
        # params
        self.use_cuda = args.use_cuda
        self.dtype = args.dtype

        self.logger.warning("<--------------------------------===> Accessor:   {WriteHead, ReadHead, Memory}")

        # functional components
        self.write_heads = WriteHead(self.write_head_params)
        self.read_heads = ReadHead(self.read_head_params)
        self.memory = ExternalMemory(self.memory_params)

        self._reset()

    def _init_weights(self):
        pass

    def _reset_states(self):
        # we reset the write/read weights of heads
        self.write_heads._reset_states()
        self.read_heads._reset_states()
        # we also reset the memory to bias value
        self.memory._reset_states()

    def _reset(self):           # NOTE: should be called at __init__
        self._init_weights()
        self.type(self.dtype)   # put on gpu if possible
        # reset internal states
        self._reset_states()

    def forward(self, hidden_vb):
        # 1. first write to memory_{t-1} to get memory_{t}
        self.memory.memory_vb = self.write_heads.forward(hidden_vb, self.memory.memory_vb)
        # 2. then read from memory_{t} to get read_vec_{t}
        read_vec_vb = self.read_heads.forward(hidden_vb, self.memory.memory_vb)
        return read_vec_vb
