import torch
from torch import linspace, tensor
from modules.ss_registry.schedulers.simple_kes_sched.schedulers.shared import apply_last_tail, apply_decay_tail, valid_decay_patterns, valid_decay_modes, blend_decay_tail, replace_tail, _to_tensor


        
def get_sigmas_karras(steps, sigma_min, sigma_max, rho=7., device='cpu', decay_pattern=None, decay_mode = None, tail_steps=None):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = linspace(0, 1, steps, device=device)
    tail_steps = tail_steps or 5    
    
    
    sigma_min = _to_tensor(sigma_min, device)
    sigma_max = _to_tensor(sigma_max, device)
    
    min_inv_rho = sigma_min.item()  ** (1 / rho)
    max_inv_rho = sigma_max.item()  ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho

    tails = None
    decay = None
    
    
    if decay_pattern:
        if decay_pattern in valid_decay_patterns:
            tails = apply_last_tail(sigmas, device, decay_pattern)
        elif decay_pattern not in valid_decay_patterns:
            print(f"[Warning] decay_pattern: {decay_pattern} not in valid decay patterns: {valid_decay_patterns}") 
    if decay_mode:
        if decay_mode in valid_decay_modes:
            if decay_mode == 'append':
                sigmas = apply_decay_tail(sigmas, device, decay_pattern, tail_steps)
            elif decay_mode == 'blend':
                blend_decay_tail(sigmas, device, decay_pattern, tail_steps)
            elif decay_mode == 'replace':
                sigmas = replace_tail(sigmas, device, decay_pattern, tail_steps)
                 
            elif decay_mode not in valid_decay_modes:
                print(f"[Warning] decay_mode: {decay_mode} not in valid decay modes: {valid_decay_modes}")
    return tails, decay, sigmas