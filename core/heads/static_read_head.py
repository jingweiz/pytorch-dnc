from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch

from core.heads.static_head import StaticHead

class StaticReadHead(StaticHead):
    def __init__(self, args):
        super(StaticReadHead, self).__init__(args)
        # params
        if self.visualize:
            self.win_head = "win_read_head"
        # logging
        self.logger.warning("<-----------------------------------> ReadHeads:  {" + str(self.num_heads) + " heads}")
        self.logger.warning(self)

        # reset
        self._reset()

    def visual(self):
        if self.visualize:      # here we visualize the wl_curr of the first batch
            self.win_head = self.vis.heatmap(self.wl_curr_vb.data[0].clone().cpu().transpose(0, 1).numpy(), env=self.refs, win=self.win_head, opts=dict(title="read_head"))

    def forward(self, hidden_vb, memory_vb):
        # content & location focus
        super(StaticReadHead, self).forward(hidden_vb, memory_vb)
        self.wl_prev_vb = self.wl_curr_vb
        # access
        return self._access(memory_vb)

    def _access(self, memory_vb): # read
        """
        variables needed:
            wl_curr_vb:   [batch_size x num_heads x mem_hei]
                       -> location focus of {t}
            memory_vb:    [batch_size x mem_hei   x mem_wid]
        returns:
            read_vec_vb:  [batch_size x num_heads x mem_wid]
                       -> read vector of {t}
        """
        return torch.bmm(self.wl_curr_vb, memory_vb)
