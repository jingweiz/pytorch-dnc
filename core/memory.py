from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
from torch.autograd import Variable

class ExternalMemory(object):
    def __init__(self, args):
        # logging
        self.logger = args.logger
        # params
        self.visualize = args.visualize
        if self.visualize:
            self.vis        = args.vis
            self.refs       = args.refs
            self.win_memory = "win_memory"
        self.use_cuda = args.use_cuda
        self.dtype = args.dtype

        self.batch_size = args.batch_size
        self.mem_hei = args.mem_hei
        self.mem_wid = args.mem_wid

        self.logger.warning("<-----------------------------------> Memory:     {" + str(self.batch_size) + "(batch_size) x " + str(self.mem_hei) + "(mem_hei) x " + str(self.mem_wid) + "(mem_wid)}")

    def _save_memory(self):
        raise NotImplementedError("not implemented in base calss")

    def _load_memory(self):
        raise NotImplementedError("not implemented in base calss")

    def _reset_states(self):
        self.memory_vb = Variable(self.memory_ts).type(self.dtype)

    def _reset(self):           # NOTE: should be called at each child's __init__
        self.memory_ts = torch.zeros(self.batch_size, self.mem_hei, self.mem_wid).fill_(1e-6)
        self._reset_states()

    def visual(self):
        if self.visualize:      # here we visualize the memory of batch0
            self.win_memory = self.vis.heatmap(self.memory_vb.data[0].clone().cpu().numpy(), env=self.refs, win=self.win_memory, opts=dict(title="memory"))

class External2DMemory(ExternalMemory):
    def __init__(self, args):
        super(External2DMemory, self).__init__(args)
        self._reset()

class External3DMemory(ExternalMemory):
    def __init__(self, args):
        super(External3DMemory, self).__init__(args)
        self._reset()
