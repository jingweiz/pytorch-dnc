from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.fake_ops import fake_cumprod
from core.heads.dynamic_head import DynamicHead

class DynamicWriteHead(DynamicHead):
    def __init__(self, args):
        super(DynamicWriteHead, self).__init__(args)
        # params
        if self.visualize:
            self.win_head = "win_write_head"

        # buid model: add additional outs for write
        # for location focus: dynamic allocation
        self.hid_2_alloc_gate = nn.Linear(self.hidden_dim, self.num_heads * 1)
        self.hid_2_write_gate = nn.Linear(self.hidden_dim, self.num_heads * 1)
        # for access
        self.hid_2_erase = nn.Linear(self.hidden_dim, self.num_heads * self.mem_wid)
        self.hid_2_add   = nn.Linear(self.hidden_dim, self.num_heads * self.mem_wid) # the write vector in dnc, called "add" in ntm

        # logging
        self.logger.warning("<-----------------------------------> WriteHeads: {" + str(self.num_heads) + " heads}")
        self.logger.warning(self)

        # reset
        self._reset()

    def visual(self):
        if self.visualize:      # here we visualize the wl_curr of the first batch
            self.win_head = self.vis.heatmap(self.wl_curr_vb.data[0].clone().cpu().transpose(0, 1).numpy(), env=self.refs, win=self.win_head, opts=dict(title="write_head"))

    def _update_usage(self, prev_usage_vb):
        """
        calculates the new usage after writing to memory
        variables needed:
            prev_usage_vb: [batch_size x mem_hei]
            wl_prev_vb:    [batch_size x num_write_heads x mem_hei]
        returns:
            usage_vb:      [batch_size x mem_hei]
        """
        # calculate the aggregated effect of all write heads
        # NOTE: how multiple write heads are delt w/ is not discussed in the paper
        # NOTE: this part is only shown in the source code
        write_weights_vb = 1. - torch.prod(1. - self.wl_prev_vb, 1)
        return prev_usage_vb + (1. - prev_usage_vb) * write_weights_vb

    def _allocation(self, usage_vb, epsilon=1e-6):
        """
        computes allocation by sorting usage, a = a_t[\phi_t[j]]
        variables needed:
            usage_vb: [batch_size x mem_hei]
                   -> indicating current memory usage, this is equal to u_t in
                      the paper when we only have one write head, but for
                      multiple write heads, one should update the usage while
                      iterating through the write heads to take into account the
                      allocation returned by this function
        returns:
            alloc_vb: [batch_size x num_write_heads x mem_hei]
        """
        # ensure values are not too small prior to cumprod
        usage_vb = epsilon + (1 - epsilon) * usage_vb
        # NOTE: we sort usage in ascending order
        sorted_usage_vb, indices_vb = torch.topk(usage_vb, k=self.mem_hei, dim=1, largest=False)
        # to imitate tf.cumrprod(exclusive=True) https://discuss.pytorch.org/t/cumprod-exclusive-true-equivalences/2614/8
        cat_sorted_usage_vb = torch.cat((Variable(torch.ones(self.batch_size, 1)).type(self.dtype), sorted_usage_vb), 1)[:, :-1]
        # TODO: seems we have to wait for this PR: https://github.com/pytorch/pytorch/pull/1439
        prod_sorted_usage_vb = fake_cumprod(cat_sorted_usage_vb)
        # prod_sorted_usage_vb = torch.cumprod(cat_sorted_usage_vb, dim=1) # TODO: use this once the PR is ready
        alloc_weight_vb = (1 - sorted_usage_vb) * prod_sorted_usage_vb  # equ. (1)
        _, indices_vb = torch.topk(indices_vb, k=self.mem_hei, dim=1, largest=False)
        alloc_weight_vb = alloc_weight_vb.gather(1, indices_vb)
        return alloc_weight_vb

    def _location_focus(self, usage_vb):
        """
        Calculates freeness-based locations for writing to.
        This finds unused memory by ranking the memory locations by usage, for
        each write head. (For more than one write head, we use a "simulated new
        usage" which takes into account the fact that the previous write head
        will increase the usage in that area of the memory.)
        variables needed:
            usage_vb:         [batch_size x mem_hei]
                           -> representing current memory usage
            write_gate_vb:    [batch_size x num_write_heads x 1]
                           -> /in [0, 1] indicating how much each write head
                              does writing based on the address returned here
                              (and hence how much usage increases)
        returns:
            alloc_weights_vb: [batch_size x num_write_heads x mem_hei]
                            -> containing the freeness-based write locations
                               Note that this isn't scaled by `write_gate`;
                               this scaling must be applied externally.
        """
        alloc_weights_vb = []
        for i in range(self.num_heads):
            alloc_weights_vb.append(self._allocation(usage_vb))
            # update usage to take into account writing to this new allocation
            # NOTE: checked: if not operate directly on _vb.data, then the _vb
            # NOTE: outside of this func will not change
            usage_vb += (1 - usage_vb) * self.write_gate_vb[:, i, :].expand_as(usage_vb) * alloc_weights_vb[i]
        # pack the allocation weights for write heads into one tensor
        alloc_weight_vb = torch.stack(alloc_weights_vb, dim=1)
        self.wl_curr_vb = self.write_gate_vb.expand_as(alloc_weight_vb) * (self.alloc_gate_vb.expand_as(self.wc_vb) * alloc_weight_vb + \
                                                                           (1. - self.alloc_gate_vb.expand_as(self.wc_vb)) * self.wc_vb)

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

    def forward(self, hidden_vb, memory_vb, usage_vb):
        # content focus
        super(DynamicWriteHead, self).forward(hidden_vb, memory_vb)
        # location focus
        self.alloc_gate_vb = F.sigmoid(self.hid_2_alloc_gate(hidden_vb)).view(-1, self.num_heads, 1)
        self.write_gate_vb = F.sigmoid(self.hid_2_write_gate(hidden_vb)).view(-1, self.num_heads, 1)
        self._location_focus(usage_vb)
        self.wl_prev_vb = self.wl_curr_vb
        # access
        self.erase_vb = F.sigmoid(self.hid_2_erase(hidden_vb)).view(-1, self.num_heads, self.mem_wid)
        self.add_vb   = self.hid_2_add(hidden_vb).view(-1, self.num_heads, self.mem_wid)
        return self._access(memory_vb)

    def _update_link(self, prev_link_vb, prev_preced_vb):
        """
        calculates the new link graphs
        For each write head, the link is a directed graph (represented by a
        matrix with entries in range [0, 1]) whose vertices are the memory
        locations, and an edge indicates temporal ordering of writes.
        variables needed:
            prev_link_vb:   [batch_size x num_heads x mem_hei x mem_wid]
                         -> {L_t-1}, previous link graphs
            prev_preced_vb: [batch_size x num_heads x mem_hei]
                         -> {p_t}, the previous aggregated precedence
                         -> weights for each write head
            wl_curr_vb:     [batch_size x num_heads x mem_hei]
                         -> location focus of {t}
        returns:
            link_vb:        [batch_size x num_heads x mem_hei x mem_hei]
                         -> {L_t}, current link graph
        """
        write_weights_i_vb = self.wl_curr_vb.unsqueeze(3).expand_as(prev_link_vb)
        write_weights_j_vb = self.wl_curr_vb.unsqueeze(2).expand_as(prev_link_vb)
        prev_preced_j_vb = prev_preced_vb.unsqueeze(2).expand_as(prev_link_vb)
        prev_link_scale_vb = 1 - write_weights_i_vb - write_weights_j_vb
        new_link_vb = write_weights_i_vb * prev_preced_j_vb
        link_vb = prev_link_scale_vb * prev_link_vb + new_link_vb
        # Return the link with the diagonal set to zero, to remove self-looping edges.
        # TODO: set diag as 0 if there's a specific method to do that w/ later releases
        diag_mask_vb = Variable(1 - torch.eye(self.mem_hei).unsqueeze(0).unsqueeze(0).expand_as(link_vb)).type(self.dtype)
        link_vb = link_vb * diag_mask_vb
        return link_vb

    def _update_precedence_weights(self, prev_preced_vb):
        """
        calculates the new precedence weights given the current write weights
        the precedence weights are the "aggregated write weights" for each write
        head, where write weights with sum close to zero will leave the
        precedence weights unchanged, but with sum close to one will replace the
        precedence weights.
        variables needed:
            prev_preced_vb: [batch_size x num_write_heads x mem_hei]
            wl_curr_vb:     [batch_size x num_write_heads x mem_hei]
        returns:
            preced_vb:      [batch_size x num_write_heads x mem_hei]
        """
        write_sum_vb = torch.sum(self.wl_curr_vb, 2)
        return (1 - write_sum_vb).expand_as(prev_preced_vb) * prev_preced_vb + self.wl_curr_vb

    def _temporal_link(self, prev_link_vb, prev_preced_vb):
        link_vb = self._update_link(prev_link_vb, prev_preced_vb)
        preced_vb = self._update_precedence_weights(prev_preced_vb)
        return link_vb, preced_vb
