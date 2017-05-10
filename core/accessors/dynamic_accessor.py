from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable

from core.accessor import Accessor
from core.heads.dynamic_write_head import DynamicWriteHead as WriteHead
from core.heads.dynamic_read_head import DynamicReadHead as ReadHead
from core.memory import External2DMemory as ExternalMemory

class DynamicAccessor(Accessor):
    def __init__(self, args):
        super(DynamicAccessor, self).__init__(args)
        # logging
        self.logger = args.logger
        # params
        self.use_cuda = args.use_cuda
        self.dtype = args.dtype
        # dynamic-accessor-specific params
        self.read_head_params.num_read_modes = self.write_head_params.num_heads * 2 + 1

        self.logger.warning("<--------------------------------===> Accessor:   {WriteHead, ReadHead, Memory}")

        # functional components
        self.usage_vb = None    # for dynamic allocation, init in _reset
        self.link_vb = None     # for temporal link, init in _reset
        self.preced_vb = None   # for temporal link, init in _reset
        self.write_heads = WriteHead(self.write_head_params)
        self.read_heads = ReadHead(self.read_head_params)
        self.memory = ExternalMemory(self.memory_params)

        self._reset()

    def _init_weights(self):
        pass

    def _reset_states(self):
        # reset the usage (for dynamic allocation) & link (for temporal link)
        self.usage_vb  = Variable(self.usage_ts).type(self.dtype)
        self.link_vb   = Variable(self.link_ts).type(self.dtype)
        self.preced_vb = Variable(self.preced_ts).type(self.dtype)
        # we reset the write/read weights of heads
        self.write_heads._reset_states()
        self.read_heads._reset_states()
        # we also reset the memory to bias value
        self.memory._reset_states()

    def _reset(self):           # NOTE: should be called at __init__
        self._init_weights()
        self.type(self.dtype)   # put on gpu if possible
        # reset internal states
        self.usage_ts  = torch.zeros(self.batch_size, self.mem_hei)
        self.link_ts   = torch.zeros(self.batch_size, self.write_head_params.num_heads, self.mem_hei, self.mem_hei)
        self.preced_ts = torch.zeros(self.batch_size, self.write_head_params.num_heads, self.mem_hei)
        self._reset_states()

    def forward(self, hidden_vb):
        # 1. first we update the usage using the read/write weights from {t-1}
        self.usage_vb = self.write_heads._update_usage(self.usage_vb)
        self.usage_vb = self.read_heads._update_usage(hidden_vb, self.usage_vb)
        # 2. then write to memory_{t-1} to get memory_{t}
        self.memory.memory_vb = self.write_heads.forward(hidden_vb, self.memory.memory_vb, self.usage_vb)
        # 3. then we update the temporal link
        self.link_vb, self.preced_vb = self.write_heads._temporal_link(self.link_vb, self.preced_vb)
        # 4. then read from memory_{t} to get read_vec_{t}
        read_vec_vb = self.read_heads.forward(hidden_vb, self.memory.memory_vb, self.link_vb, self.write_head_params.num_heads)
        return read_vec_vb
