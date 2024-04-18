from collections import OrderedDict
import torch

def mapping(old_w, new_w):
    for key, value in new_w.items():
        if key.startswith(f"module.encode.attn1") or key.startswith(f"module.encode.attn2") or key.startswith(f"module.encode.attn3"):
            parts = key.split('.')
            k1 = ".".join(parts[:3])+"."+".".join(parts[4:])
            # print(k1)
            new_w[key] = old_w[k1]
        else:
            new_w[key] = old_w[key]
    new_w['module.semantic.output.2.weight'] = torch.rand((14, 64, 1,1))
    new_w['module.semantic.output.2.bias'] = torch.rand((14))
    return new_w