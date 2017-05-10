from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from torch.autograd import Variable

# NOTE: since currently cumprod does not support autograd
# NOTE: we implement this op using exisiting torch ops
# TODO: replace fake_cumprod w/ cumprod once this PR is ready:
# TODO: https://github.com/pytorch/pytorch/pull/1439
# NOTE: https://discuss.pytorch.org/t/cumprod-exclusive-true-equivalences/2614/8https://discuss.pytorch.org/t/cumprod-exclusive-true-equivalences/2614/8
def fake_cumprod(vb):
    """
    args:
        vb:  [hei x wid]
          -> NOTE: we are lazy here so now it only supports cumprod along wid
    """
    # real_cumprod = torch.cumprod(vb.data, 1)
    vb = vb.unsqueeze(0)
    mul_mask_vb = Variable(torch.zeros(vb.size(2), vb.size(1), vb.size(2))).type_as(vb)
    for i in range(vb.size(2)):
       mul_mask_vb[i, :, :i+1] = 1
    add_mask_vb = 1 - mul_mask_vb
    vb = vb.expand_as(mul_mask_vb) * mul_mask_vb + add_mask_vb
    vb = torch.prod(vb, 2).transpose(0, 2)
    # print(real_cumprod - vb.data) # NOTE: checked, ==0
    return vb
