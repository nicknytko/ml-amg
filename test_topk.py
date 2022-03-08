import torch
from ns.model.agg_interp import topk_vec as topk_ref
from ns.lib.helpers import topk_vec as topk_qs

z=torch.normal(torch.zeros(5), torch.ones(5))
print(topk_ref(z, 3))
print(topk_qs(z, 3))
