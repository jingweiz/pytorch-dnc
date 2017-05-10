from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.heads.dynamic_head import DynamicHead

class DynamicReadHead(DynamicHead):
    def __init__(self, args):
        super(DynamicReadHead, self).__init__(args)
        # params
        if self.visualize:
            self.win_head = "win_read_head"
        # dynamic-accessor-specific params
        # NOTE: mixing between backwards and forwards position (for each write head)
        # NOTE: and content-based lookup, for each read head
        self.num_read_modes = args.num_read_modes

        # build model: add additional outs for read
        # for locatoin focus: temporal linkage
        self.hid_2_free_gate = nn.Linear(self.hidden_dim, self.num_heads * 1)
        self.hid_2_read_mode = nn.Linear(self.hidden_dim, self.num_heads * self.num_read_modes)

        # logging
        self.logger.warning("<-----------------------------------> ReadHeads:  {" + str(self.num_heads) + " heads}")
        self.logger.warning(self)

        # reset
        self._reset()

    def visual(self):
        if self.visualize:      # here we visualize the wl_curr of the first batch
            self.win_head = self.vis.heatmap(self.wl_curr_vb.data[0].clone().cpu().transpose(0, 1).numpy(), env=self.refs, win=self.win_head, opts=dict(title="read_head"))

    def _update_usage(self, hidden_vb, prev_usage_vb):
        """
        calculates the new usage after reading and freeing from memory
        variables needed:
            hidden_vb:     [batch_size x hidden_dim]
            prev_usage_vb: [batch_size x mem_hei]
            free_gate_vb:  [batch_size x num_heads x 1]
            wl_prev_vb:    [batch_size x num_heads x mem_hei]
        returns:
            usage_vb:      [batch_size x mem_hei]
        """
        self.free_gate_vb = F.sigmoid(self.hid_2_free_gate(hidden_vb)).view(-1, self.num_heads, 1)
        free_read_weights_vb = self.free_gate_vb.expand_as(self.wl_prev_vb) * self.wl_prev_vb
        psi_vb = torch.prod(1. - free_read_weights_vb, 1)
        return prev_usage_vb * psi_vb

    def _directional_read_weights(self, link_vb, num_write_heads, forward):
        """
        calculates the forward or the backward read weights
        for each read head (at a given address), there are `num_writes` link
        graphs to follow. thus this function computes a read address for each of
        the `num_reads * num_writes` pairs of read and write heads.
        we calculate the forward and backward directions for each pair of read
        and write heads; hence we need to tile the read weights and do a sort of
        "outer product" to get this.
        variables needed:
            link_vb:    [batch_size x num_read_heads x mem_hei x mem_hei]
                     -> {L_t}, current link graph
            wl_prev_vb: [batch_size x num_read_heads x mem_hei]
                     -> containing the previous read weights w_{t-1}^r.
            num_write_heads: NOTE: self.num_heads here is num_read_heads
            forward:    boolean
                     -> indicating whether to follow the "future" (True)
                     -> direction in the link graph or the "past" (False)
                     -> direction
        returns:
            directional_weights_vb: [batch_size x num_read_heads x num_write_heads x mem_hei]
        """
        expanded_read_weights_vb = self.wl_prev_vb.unsqueeze(1).expand_as(torch.Tensor(self.batch_size, num_write_heads, self.num_heads, self.mem_hei)).contiguous()
        if forward:
            directional_weights_vb = torch.bmm(expanded_read_weights_vb.view(-1, self.num_heads, self.mem_hei), link_vb.view(-1, self.mem_hei, self.mem_hei).transpose(1, 2))
        else:
            directional_weights_vb = torch.bmm(expanded_read_weights_vb.view(-1, self.num_heads, self.mem_hei), link_vb.view(-1, self.mem_hei, self.mem_hei))
        return directional_weights_vb.view(-1, num_write_heads, self.num_heads, self.mem_hei).transpose(1, 2)

    def _location_focus(self, link_vb, num_write_heads): # temporal linkage
        """
        calculates the read weights after location focus
        variables needed:
            link_vb:      [batch_size x num_heads x mem_hei x mem_hei]
                       -> {L_t}, current link graph
            num_write_heads: NOTE: self.num_heads here is num_read_heads
            wc_vb:        [batch_size x num_heads x mem_hei]
                       -> containing the focus by content of {t}
            read_mode_vb: [batch_size x num_heads x num_read_modes]
        returns:
            wl_curr_vb:   [batch_size x num_read_heads x num_write_heads x mem_hei]
                       -> focus by content of {t}
        """
        # calculates f_t^i & b_t^i
        forward_weights_vb  = self._directional_read_weights(link_vb, num_write_heads, True)
        backward_weights_vb = self._directional_read_weights(link_vb, num_write_heads, False)
        backward_mode_vb = self.read_mode_vb[:, :, :num_write_heads]
        forward_mode_vb  = self.read_mode_vb[:, :, num_write_heads:2*num_write_heads]
        content_mode_vb  = self.read_mode_vb[:, :, 2*num_write_heads:]
        self.wl_curr_vb = content_mode_vb.expand_as(self.wc_vb) * self.wc_vb\
                        + torch.sum(forward_mode_vb.unsqueeze(3).expand_as(forward_weights_vb) * forward_weights_vb, 2) \
                        + torch.sum(backward_mode_vb.unsqueeze(3).expand_as(backward_weights_vb) * backward_weights_vb, 2)

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

    def forward(self, hidden_vb, memory_vb, link_vb, num_write_heads):
        # content focus
        super(DynamicReadHead, self).forward(hidden_vb, memory_vb)
        # location focus
        self.read_mode_vb = F.softmax(self.hid_2_read_mode(hidden_vb).view(-1, self.num_heads, self.num_read_modes).transpose(0, 2)).transpose(0, 2)
        self._location_focus(link_vb, num_write_heads)
        self.wl_prev_vb = self.wl_curr_vb
        # access
        return self._access(memory_vb)
