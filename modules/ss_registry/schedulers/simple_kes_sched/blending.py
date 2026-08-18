from __future__ import annotations

import copy

import numpy as np
import torch

class BlendingMixin:
    """Internal Simple KES responsibility mixin.

    This class is composed into ``SimpleKEScheduler`` so the public scheduler
    method surface remains unchanged while implementation concerns stay isolated.
    """

    def resolve_blend_weights(self, blend_weights, blending_style):
        if blending_style == 'softmax':
            blend_weights = torch.tensor(blend_weights)
            normalized_weights = torch.softmax(blend_weights, dim=0)
            return normalized_weights.tolist()
        if blending_style == 'explicit':
            return blend_weights
        raise ValueError(f"Unknown blending_style: {blending_style}")

    def extract_scalar(self, value):
        if isinstance(value, torch.Tensor):
            if value.numel() > 1:
                return value.mean().item()
            return value.item()
        return value

    def _call_legacy_mode(self, schedule_type):
        if schedule_type not in ['karras', 'exponential']:
            self.log(f"[Legacy Mode] Unsupported schedule_type: {schedule_type}")
            return

        target_attr = f"sigmas_{schedule_type}"
        scheduler_func = self.scheduler_registry.get(schedule_type)

        tails, decay, extras, sigmas = self.call_scheduler_function(
            scheduler_func,
            steps=self.steps,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
            device=self.device,
            decay_pattern=self.decay_pattern
        )

        setattr(self, target_attr, sigmas)
        self.log(f"[Legacy Mode] Loaded sigma sequence for {schedule_type}. Assigned to self.{target_attr}")

    def blend_sigma_sequence(self, sigmas_karras=None, sigmas_exponential=None, pre_pass=False, blend_methods=None, blend_weights=None):
        active_methods = [
            method for method, config in self.blend_method_dict.items()
            if float(config.get('weight', 0.0)) > 0.0
        ]
        if not active_methods:
            raise ValueError(
                "[SimpleKEScheduler] No active schedulers selected. "
                "At least one blend method must have a weight greater than zero."
            )

        self.blend_methods = active_methods
        self.blend_method_dict = {
            method: self.blend_method_dict[method]
            for method in active_methods
        }
        self.settings['blend_methods'] = copy.deepcopy(self.blend_method_dict)
        self.blend_weights = [self.blend_method_dict[m]['weight'] for m in self.blend_methods]

        if len(self.blend_methods) == 1:
            self.log(f"[Blend] Only one active scheduler: {self.blend_methods[0]}. Skipping blending, using it directly.")
            self.sigs = self.sigma_sequences[self.blend_methods[0]]['sigmas']

        if self.blending_mode in {'default', 'auto'}:
            self.blending_mode = 'weights'
        elif self.blending_mode == 'smooth_blend' and len(self.blend_methods) != 2:
            self.log("[Blend] smooth_blend requires exactly two active methods; using weighted blending.")
            self.blending_mode = 'weights'
        self.settings['blending_mode'] = self.blending_mode

        if not self.allow_step_expansion and self.auto_mode_enabled:
            self.auto_mode_enabled = False
            self.log("[Auto Mode] Step expansion disallowed. Auto mode forcibly disabled.")

        self.progress = torch.linspace(0, 1, len(self.sigs), device=self.device)
        self.blended_sigmas = []
        self.change_log = []
        self.relative_converged = False
        self.max_converged = False
        self.delta_converged = False
        self.early_stop_triggered = False

        # Weighted and smooth blending consume the active method sequences from
        # ``self.sigma_sequences``. Do not manufacture legacy Karras/Exponential
        # sequences when either method has a zero weight; zero-weight methods must
        # remain excluded from both generation and blending.
        self.prepass_blended_sigmas = []
        self.blended_sigma = None
        self.blended_sigmas = []
        for i in range(len(self.sigs)):
            if self.step_progress_mode == "linear":
                progress_value = self.progress[i]
            elif self.step_progress_mode == "exponential":
                progress_value = self.progress[i] ** self.exp_power
            elif self.step_progress_mode == "logarithmic":
                progress_value = torch.log1p(self.progress[i] * (torch.exp(torch.tensor(1.0, device=self.device)) - 1))
            elif self.step_progress_mode == "sigmoid":
                progress_value = 1 / (1 + torch.exp(-12 * (self.progress[i] - 0.5)))
            else:
                progress_value = self.progress[i]

            self.dynamic_blend_factor = self.start_blend * (1 - self.progress[i]) + self.end_blend * self.progress[i]
            self.smooth_blend = torch.sigmoid((self.dynamic_blend_factor - self.blend_midpoint) * self.smooth_blend_factor)
            self.noise_scale = self.initial_noise_scale * (1 - self.progress[i]) + self.final_noise_scale * self.progress[i] * self.noise_scale_factor
            self.step_size = self.initial_step_size * (1 - progress_value) + self.final_step_size * progress_value * self.step_size_factor

            if self.blending_mode == 'default':
                self.blended_sigma = self.sigmas_karras[i] * (1 - self.smooth_blend) + self.sigmas_exponential[i] * self.smooth_blend

            if self.blending_mode == 'smooth_blend' or (self.blending_mode == 'auto' and len(self.blend_methods) == 2):
                sigma_seq_a = self.sigma_sequences[self.blend_methods[0]]['sigmas']
                sigma_seq_b = self.sigma_sequences[self.blend_methods[1]]['sigmas']
                self.blended_sigma = sigma_seq_a[i] * (1 - self.smooth_blend) + sigma_seq_b[i] * self.smooth_blend

            elif self.blending_mode == 'weights' or (self.blending_mode == 'auto' and len(self.blend_methods) > 2):
                if self.blend_weights is None:
                    self.blend_weights = [1.0] * len(self.all_sigmas)
                if self.blending_style is None:
                    self.blending_style = 'softmax'
                resolved_blend_weights = self.resolve_blend_weights(self.blend_weights, self.blending_style)
                weighted_sum = sum(w * self.extract_scalar(s[i]) for w, s in zip(resolved_blend_weights, self.all_sigmas))
                total_weight = sum(resolved_blend_weights)
                self.blended_sigma = weighted_sum / total_weight

            for s in self.all_sigmas:
                self.log(f"[DEBUG]sigma sequence shape: {s.shape}")

            self.sigs[i] = self.blended_sigma * self.step_size * self.noise_scale
            self.change = torch.abs(self.sigs[i] - self.sigs[i - 1])
            self.change_log.append(self.extract_scalar(self.change))
            relative_sigma_progress = (self.blended_sigma - self.sigs[-1].item()) / self.blended_sigma
            recent_changes = torch.abs(torch.tensor(self.change_log[-5:]))
            max_change = torch.max(recent_changes).item()
            mean_change = torch.mean(recent_changes).item()
            self.delta_change = abs(max_change - mean_change)
            self.blended_sigmas.append(self.extract_scalar(self.blended_sigma))

            self.relative_converged = relative_sigma_progress < 0.05
            self.max_converged = max_change < self.early_stopping_threshold
            self.delta_converged = self.delta_change < self.recent_change_convergence_delta

            if pre_pass:
                self.prepass_blended_sigmas = self.blended_sigmas.copy()
                self.prepass_blended_sigma = self.blended_sigma
                if i >= 2:
                    sigma_rate = abs(self.prepass_blended_sigmas[i] - self.prepass_blended_sigmas[i - 1])
                    previous_sigma_rate = abs(self.prepass_blended_sigmas[i - 1] - self.prepass_blended_sigmas[i - 2])
                    if sigma_rate > previous_sigma_rate:
                        self.prepass_log(f"Sigma decline is slowing down → possible plateau at step {i+1}.")

                if i == 0:
                    self.prepass_log("\n--- Starting Pre-Pass Blending ---\n")
                    step_label = "Prepass First Step"
                elif i == len(self.sigs) - 1:
                    step_label = "Prepass Last Step"
                else:
                    step_label = None

                if step_label:
                    self.prepass_log(f"[{step_label} - Step {i}/{len(self.sigs)}] Prepass Blended Sigma: {self.prepass_blended_sigma:.6f}, Final Sigma: {self.sigs[i]:.6f}")
                    self.prepass_log(f"{step_label} Delta Converged: {self.delta_converged} delta_change: {self.delta_change:.6f}, Target Default Settings:{self.recent_change_convergence_delta}")

                if i > self.safety_minimum_stop_step and len(self.change_log) > 10:
                    self.blended_tensor = torch.tensor(self.prepass_blended_sigmas)
                    if self.device == 'cpu':
                        self.sigma_variance = np.var(self.prepass_blended_sigmas)
                    else:
                        self.sigma_variance = torch.var(self.sigs).item()

                    self.min_sigma_threshold = self.sigma_variance * self.sigma_variance_scale
                    self.prepass_log(f"\n--- Early Stopping Evaluation at Step {i} ---")
                    self.prepass_log(f"Current Blended Prepass Sigma: {self.prepass_blended_sigma:.6f}")
                    self.prepass_log(f"Sigma Variance: {self.sigma_variance:.6f}")
                    self.prepass_log(f"Relative Sigma Progress: {relative_sigma_progress:.6f}")
                    self.prepass_log(f"Max Recent Sigma Change: {max_change:.6f}")
                    self.prepass_log(f"Mean Recent Sigma Change: {mean_change:.6f}")

                    if self.prepass_blended_sigma > self.min_sigma_threshold:
                        self.prepass_log(f"Prepass Blended Sigma {self.prepass_blended_sigma:.6f} exceeds min sigma threshold {self.min_sigma_threshold:.6f} → Continuing.\n")

                    if self.early_stopping_method == "mean":
                        mean_change = sum(self.change_log) / len(self.change_log)
                        if mean_change < self.early_stopping_threshold:
                            skipped_steps = len(self.sigs) - i
                            self.prepass_log(f"Early stopping triggered by mean at step {i}. Mean change: {mean_change:.6f}. Steps used: {i}/{len(self.sigs)}, steps skipped: {skipped_steps}")
                    elif self.early_stopping_method == "max":
                        if max_change < self.early_stopping_threshold:
                            skipped_steps = len(self.sigs) - i
                            self.prepass_log(f"Early stopping triggered by mean at step {i}. Mean change: {max_change:.6f}. Steps used: {i}/{len(self.sigs)}, steps skipped: {skipped_steps}")
                    elif self.early_stopping_method == "sum":
                        stable_steps = sum(
                            1 for j in range(1, len(self.change_log))
                            if abs(self.change_log[j]) < self.early_stopping_threshold * abs(self.sigs[j])
                        )
                        if stable_steps >= 0.8 * len(self.change_log):
                            skipped_steps = len(self.sigs) - i
                            self.prepass_log(f"Early stopping triggered by sum at step {i}. Stable steps: {stable_steps}/{len(self.change_log)}. Steps used: {i}/{len(self.sigs)}, steps skipped: {skipped_steps}")

                    if self.relative_converged and self.max_converged and self.delta_converged:
                        self.early_stop_triggered = True
                        self.prepass_log(f"\n--- Early Stopping Evaluation at Step {i+1} ---")
                        self.prepass_log(f"Relative Sigma Progress: {relative_sigma_progress:.6f}")
                        self.prepass_log(f"Max Recent Sigma Change: {max_change:.6f}")
                        self.prepass_log(f"Mean Recent Sigma Change: {mean_change:.6f}")
                        self.prepass_log(f"Delta Change: {self.delta_change:.6f} (Target: {self.recent_change_convergence_delta})")
                        self.prepass_log(f"Early stopping criteria met at step {i+1} based on all convergence checks.")
                        self.predicted_stop_step = i
                        break

            if not pre_pass:
                if i == 0:
                    step_label = "First Step"
                    self.log("\n" + "=" * 10 + "\n[Start of Sigma Sequence Logging]\n" + "=" * 10)
                    self.log(f"[{step_label} - Step {i}/{len(self.sigs)}]"
                             f"\nStep Size: {self.step_size:.6f}"
                             f"\nDynamic Blend Factor: {self.dynamic_blend_factor:.6f}"
                             f"\nNoise Scale: {self.noise_scale:.6f}"
                             f"\nSmooth Blend: {self.smooth_blend:.6f}"
                             f"\nBlended Sigma: {self.blended_sigma:.6f}"
                             f"\nFinal Sigma: {self.sigs[i]:.6f}")
                elif i == len(self.sigs) // 2:
                    step_label = "Middle Step"
                    self.log(f"[{step_label} - Step {i}/{len(self.sigs)}]"
                             f"\nStep Size: {self.step_size:.6f}"
                             f"\nDynamic Blend Factor: {self.dynamic_blend_factor:.6f}"
                             f"\nNoise Scale: {self.noise_scale:.6f}"
                             f"\nSmooth Blend: {self.smooth_blend:.6f}"
                             f"\nBlended Sigma: {self.blended_sigma:.6f}"
                             f"\nFinal Sigma: {self.sigs[i]:.6f}")
                elif i == len(self.sigs) - 1:
                    step_label = "Last Step"
                    self.log(f"[{step_label} - Step {i}/{len(self.sigs)}]"
                             f"\nStep Size: {self.step_size:.6f}"
                             f"\nDynamic Blend Factor: {self.dynamic_blend_factor:.6f}"
                             f"\nNoise Scale: {self.noise_scale:.6f}"
                             f"\nSmooth Blend: {self.smooth_blend:.6f}"
                             f"\nBlended Sigma: {self.blended_sigma:.6f}"
                             f"\nFinal Sigma: {self.sigs[i]:.6f}")
                    self.log("\n" + "=" * 10 + "\n[End of Sigma Sequence Logging]\n" + "=" * 10)

                if i > 0:
                    self.change = torch.abs(self.sigs[i] - self.sigs[i - 1])
                    self.change_log.append(self.extract_scalar(self.change))

                if i > self.safety_minimum_stop_step and len(self.change_log) > 5:
                    final_target_sigma = self.sigs[-1].item()
                    if self.blended_sigma != 0:
                        relative_sigma_progress = (self.blended_sigma - final_target_sigma) / self.blended_sigma
                    else:
                        relative_sigma_progress = 0
                    self.sigma_variance = torch.var(self.sigs).item() if self.device != 'cpu' else np.var(self.blended_sigmas)
                    self.log(f"Sigma Variance: {self.sigma_variance:.6f}")

        num_methods = len(self.all_sigmas)
        if not self.blend_weights or len(self.blend_weights) != num_methods:
            self.blend_weights = [1.0] * num_methods

        if not getattr(self, "blending_style", None):
            self.blending_style = "softmax"

        resolved_blend_weights = self.resolve_blend_weights(self.blend_weights, self.blending_style)

        s = sum(resolved_blend_weights)
        if s <= 0:
            resolved_blend_weights = [1.0] * num_methods
            s = float(num_methods)
        resolved_blend_weights = [w / s for w in resolved_blend_weights]

        if not self.auto_mode_enabled:
            if not pre_pass:
                if self.apply_tail_steps:
                    for i, tail in enumerate(self.all_tails):
                        if tail is not None:
                            self.log(f"Appending tail from method: {self.blend_methods[i]}")
                            self.sigs = torch.cat([self.sigs, tail])
                            self._tail_steps_applied = True
                            self._step_expansion_applied = True

                if self.apply_decay_tail:
                    for i, decay in enumerate(self.all_decays):
                        if decay is not None:
                            self.log(f"Appending decay from method: {self.blend_methods[i]}")
                            self.sigs = torch.cat([self.sigs, decay])
                            self._decay_tail_applied = True
                            self._step_expansion_applied = True

                if self.apply_progressive_decay and any(self.all_decays):
                    progressive_decay = None
                    total_weight = 0.0
                    for w, decay in zip(resolved_blend_weights, self.all_decays):
                        if decay is not None:
                            decay = decay[:len(self.sigs)]
                            progressive_decay = (w * decay) if progressive_decay is None else (progressive_decay + w * decay)
                            total_weight += w
                    if progressive_decay is not None and total_weight > 0:
                        progressive_decay /= total_weight
                        self.sigs = self.sigs * progressive_decay
                        self._progressive_decay_applied = True

                if self.apply_blended_tail and any(self.all_tails):
                    blended_tail = None
                    total_weight = 0.0
                    for w, tail in zip(resolved_blend_weights, self.all_tails):
                        if tail is not None:
                            blended_tail = (w * tail) if blended_tail is None else (blended_tail + w * tail)
                            total_weight += w
                    if blended_tail is not None and total_weight > 0:
                        blended_tail /= total_weight
                        self.sigs = torch.cat([self.sigs, blended_tail])
                        self._blended_tail_applied = True
                        self._step_expansion_applied = True
        else:
            if len(self.sigs) > self.steps:
                self.auto_stabilization_sequence = []
                self.log(f"[Auto Mode] Sigma sequence length {len(self.sigs)} exceeds requested steps {self.steps}. Disabling auto stabilization.")
                self.auto_mode_enabled = False
                self.sigs = self.sigs[:self.steps]
                return self.sigs
            self.run_auto_stabilization(self.sigs)

        if pre_pass and self.early_stop_triggered:
            return self.sigs[:self.predicted_stop_step]
        return self.sigs

    def load_blend_method_sigmas(self, mode=None):
        self.all_sigmas = []
        shared_tail_config = {
            'decay_pattern': self.decay_pattern if self.decay_pattern is not None else 'zero',
            'decay_mode': self.decay_mode if self.decay_mode is not None else 'blend',
            'tail_steps': self.tail_steps if self.tail_steps is not None else 1,
        }

        for method in self.blend_methods:
            self.method_config = dict(self.blend_method_dict.get(method) or {})
            self.current_config = dict(shared_tail_config)

            tails, decay, extras, sigmas = self.call_scheduler_function(
                self.scheduler_registry.get(method),
                steps=self.steps,
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max,
                rho=self.rho,
                device=self.device,
                decay_pattern=self.current_config['decay_pattern'],
                decay_mode=self.current_config['decay_mode'],
                tail_steps=self.current_config['tail_steps']
            )
            if self.debug:
                if isinstance(sigmas, torch.Tensor):
                    flat = sigmas.detach().flatten().cpu()
                    if flat.numel() > 0:
                        self.log(
                            f"[Raw scheduler {method}] count={flat.numel()}, "
                            f"first={flat[0].item():.10f}, last={flat[-1].item():.10f}"
                        )
                else:
                    self.log(
                        f"[Raw scheduler {method}] non-tensor result: {type(sigmas)}"
                    )
            self.sigma_sequences[method] = {
                'sigmas': sigmas,
                'tails': tails,
                'decay': decay,
                'extras': extras
            }
            setattr(self, f"sigmas_{method}", sigmas)

        self.all_sigmas = [self.sigma_sequences[method]['sigmas'] for method in self.blend_methods]
        self.all_tails = [self.sigma_sequences[method]['tails'] for method in self.blend_methods]
        self.all_decays = [self.sigma_sequences[method]['decay'] for method in self.blend_methods]
        self.all_extras = [self.sigma_sequences[method].get('extras', []) for method in self.blend_methods]
        self._log_extras_to_file(self.all_extras)

        norm = []
        for i, s in enumerate(self.all_sigmas):
            if isinstance(s, list):
                s = torch.tensor(s, dtype=torch.float32, device=self.device)
            elif isinstance(s, torch.Tensor):
                s = s.detach().to(self.device, dtype=torch.float32).flatten()
            else:
                raise TypeError(f"Sigma sequence for {self.blend_methods[i]} must be list or Tensor, got {type(s)}")
            norm.append(s)

        maxlen = max(len(s) for s in norm)
        for i, s in enumerate(norm):
            if len(s) < maxlen:
                pad = s[-1].repeat(maxlen - len(s))
                norm[i] = torch.cat([s, pad])

        self.all_sigmas = norm
        self.log(f"Loaded sigma schedules for blend methods: {self.blend_methods} using mode: {mode}")

    def validate_and_align_sigmas(self):
        if not self.all_sigmas or len(self.all_sigmas) == 0:
            raise ValueError("No sigma sequences were loaded for blending.")

        target_length = max(len(s) for s in self.all_sigmas)

        for idx, sigmas in enumerate(self.all_sigmas):
            if sigmas is None or len(sigmas) == 0:
                raise ValueError(f"Sigma sequence at index {idx} is invalid or empty: {sigmas}")

            if len(sigmas) < target_length:
                padding = torch.full((target_length - len(sigmas),), sigmas[-1]).to(sigmas.device)
                self.all_sigmas[idx] = torch.cat([sigmas, padding])

        self.log(f"Validated and aligned all sigma sequences to length {target_length}.")
