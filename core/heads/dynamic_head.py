from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.head import Head
from utils.similarities import batch_cosine_sim

class DynamicHead(Head):
    def __init__(self, args):
        super(DynamicHead, self).__init__(args)

        # build model
        # for content focus
        self.hid_2_key  = nn.Linear(self.hidden_dim, self.num_heads * self.mem_wid)
        self.hid_2_beta = nn.Linear(self.hidden_dim, self.num_heads * 1)

    def _update_usage(self, prev_usage_vb):
        raise NotImplementedError("not implemented in base calss")

    def _content_focus(self, memory_vb):
        """
        variables needed:
            key_vb:    [batch_size x num_heads x mem_wid]
                    -> similarity key vector, to compare to each row in memory
                    -> by cosine similarity
            beta_vb:   [batch_size x num_heads x 1]
                    -> NOTE: refer here: https://github.com/deepmind/dnc/issues/9
                    -> \in (1, +inf) after oneplus(); similarity key strength
                    -> amplify or attenuate the pecision of the focus
            memory_vb: [batch_size x mem_hei   x mem_wid]
        returns:
            wc_vb:     [batch_size x num_heads x mem_hei]
                    -> the attention weight by content focus
        """
        K_vb = batch_cosine_sim(self.key_vb, memory_vb)  # [batch_size x num_heads x mem_hei]
        self.wc_vb = K_vb * self.beta_vb.expand_as(K_vb) # [batch_size x num_heads x mem_hei]
        self.wc_vb = F.softmax(self.wc_vb.transpose(0, 2)).transpose(0, 2)

    def _location_focus(self):
        raise NotImplementedError("not implemented in base calss")

    def forward(self, hidden_vb, memory_vb):
        # outputs for computing addressing for heads
        # NOTE: to be consistent w/ the dnc paper, we use
        # NOTE: sigmoid to constrain to [0, 1]
        # NOTE: oneplus to constrain to [1, +inf]
        self.key_vb   = F.tanh(self.hid_2_key(hidden_vb)).view(-1, self.num_heads, self.mem_wid)    # TODO: relu to bias the memory to store positive values ??? check again
        self.beta_vb  = F.softplus(self.hid_2_beta(hidden_vb)).view(-1, self.num_heads, 1)          # beta >=1: https://github.com/deepmind/dnc/issues/9

        # now we compute the addressing mechanism
        self._content_focus(memory_vb)
