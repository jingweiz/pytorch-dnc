from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.heads.static_head import StaticHead

class StaticWriteHead(StaticHead):
    def __init__(self, args):
        super(StaticWriteHead, self).__init__(args)
        # params
        if self.visualize:
            self.win_head = "win_write_head"

        # buid model: add additional outs for write
        # for access
        self.hid_2_erase = nn.Linear(self.hidden_dim, self.num_heads * self.mem_wid)
        self.hid_2_add   = nn.Linear(self.hidden_dim, self.num_heads * self.mem_wid)

        # logging
        self.logger.warning("<-----------------------------------> WriteHeads: {" + str(self.num_heads) + " heads}")
        self.logger.warning(self)

        # reset
        self._reset()

    def visual(self):
        if self.visualize:      # here we visualize the wl_curr of the first batch
            self.win_head = self.vis.heatmap(self.wl_curr_vb.data[0].clone().cpu().transpose(0, 1).numpy(), env=self.refs, win=self.win_head, opts=dict(title="write_head"))

    def forward(self, hidden_vb, memory_vb):
        # content & location focus
        super(StaticWriteHead, self).forward(hidden_vb, memory_vb)
        self.wl_prev_vb = self.wl_curr_vb
        # access
        self.erase_vb = F.sigmoid(self.hid_2_erase(hidden_vb)).view(-1, self.num_heads, self.mem_wid)
        self.add_vb   = self.hid_2_add(hidden_vb).view(-1, self.num_heads, self.mem_wid)
        return self._access(memory_vb)

    def _access(self, memory_vb): # write
        """
        variables needed:
            wl_curr_vb: [batch_size x num_heads x mem_hei]
            erase_vb:   [batch_size x num_heads x mem_wid]
                     -> /in (0, 1)
            add_vb:     [batch_size x num_heads x mem_wid]
                     -> w/ no restrictions in range
            memory_vb:  [batch_size x mem_hei x mem_wid]
        returns:
            memory_vb:  [batch_size x mem_hei x mem_wid]
        NOTE: IMPORTANT: https://github.com/deepmind/dnc/issues/10
        """

        # first let's do erasion
        weighted_erase_vb = torch.bmm(self.wl_curr_vb.contiguous().view(-1, self.mem_hei, 1),
                                      self.erase_vb.contiguous().view(-1, 1, self.mem_wid)).view(-1, self.num_heads, self.mem_hei, self.mem_wid)
        keep_vb = torch.prod(1. - weighted_erase_vb, dim=1)
        memory_vb = memory_vb * keep_vb
        # finally let's write (do addition)
        return memory_vb + torch.bmm(self.wl_curr_vb.transpose(1, 2), self.add_vb)
