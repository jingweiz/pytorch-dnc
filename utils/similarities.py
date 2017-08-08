from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from torch.autograd import Variable

def batch_cosine_sim(u, v, epsilon=1e-6):
    """
    u: content_key: [batch_size x num_heads x mem_wid]
    v: memory:      [batch_size x mem_hei   x mem_wid]
    k: similarity:  [batch_size x num_heads x mem_hei]
    """
    assert u.dim() == 3 and v.dim() == 3
    numerator = torch.bmm(u, v.transpose(1, 2))
    # denominator = torch.sqrt(torch.bmm(u.norm(2, 2).pow(2) + epsilon, v.norm(2, 2).pow(2).transpose(1, 2) + epsilon))                             # 0.1.12
    denominator = torch.sqrt(torch.bmm(u.norm(2, 2, keepdim=True).pow(2) + epsilon, v.norm(2, 2, keepdim=True).pow(2).transpose(1, 2) + epsilon))   # 0.2.0
    k = numerator / (denominator + epsilon)
    return k

# batch_size = 3
# num_heads = 2
# mem_hei = 5
# mem_wid = 7
# u = torch.ones(batch_size, num_heads, mem_wid)
# u[0][0][4] = 0
# u[1][0][4] = 10
# u[1][1][6] = 10
#
# v = torch.ones(batch_size, mem_hei, mem_wid)
# v[0] = v[0] * 2
# v[1][0][0] = 0
# v[1][0][1] = 1
# v[1][0][2] = 2
# v[1][0][3] = 3
# v[1][0][4] = 4
# v[1][1][4] = 0
# print(u)
# print(v)
#
# batch_cosine_sim(Variable(u), Variable(v))
