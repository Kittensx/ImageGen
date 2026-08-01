import torch
from typing import Optional

def check_device(*tensors, enabled: Optional[bool] = True):
    dev = tensors[0].device
    for t in tensors:
        if t.device != dev:
            raise RuntimeError(f"[DeviceMismatch] {t} is on {t.device}, expected {dev}")


'''
#how to use:

a = torch.randn(1).cuda()
b = torch.randn(1).cuda()
c = torch.randn(1).cpu()

check_device(a, b, )  # ✅ no error
check_device(a, c, )  # ❌ raises RuntimeError

'''
