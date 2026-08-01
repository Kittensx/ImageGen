import torch
from modules.ss_registry.schedulers.simple_kes_sched.schedulers.shared import apply_last_tail, apply_decay_tail, valid_decay_patterns, valid_decay_modes, blend_decay_tail, replace_tail, _to_tensor



        
def get_sigmas_geometric(steps:int, sigma_min:float, sigma_max:float, device, decay_pattern:str=None, decay_mode:str=None, tail_steps:int=None):
   

    # Convert sigma_min and sigma_max to tensors safely
    sigma_min = _to_tensor(sigma_min, device)
    sigma_max = _to_tensor(sigma_max, device)

    sigmas = [sigma_max]
    tail_steps = tail_steps or 5

    for step in range(1, steps):
        if len(sigmas) >= 2:
            deltas = torch.abs(torch.tensor(sigmas[:-1]) - torch.tensor(sigmas[1:]))
            avg_delta = torch.mean(deltas).item()
        else:
            avg_delta = sigmas[-1].item() * 0.1

        last_sigma = sigmas[-1]
        dynamic_decay_rate = max(1 - (avg_delta / (last_sigma + 1e-5)), 0.85)  # Clamp for stability

        next_sigma = max(last_sigma * dynamic_decay_rate, sigma_min.item())
        sigmas.append(next_sigma)

    sigmas = torch.tensor(sigmas, device=device)

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









