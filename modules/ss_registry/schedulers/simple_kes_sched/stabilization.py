from __future__ import annotations

import torch
import torch.nn.functional as F

from modules.ss_registry.schedulers.simple_kes_sched.schedulers.shared import apply_decay_tail as apply_decay_tail_fn

class StabilizationMixin:
    """Internal Simple KES responsibility mixin.

    This class is composed into ``SimpleKEScheduler`` so the public scheduler
    method surface remains unchanged while implementation concerns stay isolated.
    """

    def _repair_non_increasing_sigmas(self, sigmas: torch.Tensor) -> torch.Tensor:
        """Repair advanced tail/blend combinations that introduce upward sigma jumps.

        Simple KES exposes several experimental tail, decay, blending, and step-expansion
        features that may be combined freely. Some valid combinations can append a segment
        whose first value is above the preceding sigma. The sampler contract requires a
        non-increasing sequence, so normalize only the offending upward transitions rather
        than failing the entire generation before sampling starts.
        """
        values = sigmas.flatten()
        if values.numel() < 2:
            return values

        increases = values[1:] - values[:-1]
        violation_mask = increases > 0
        if not bool(torch.any(violation_mask).item()):
            return values

        violation_count = int(torch.count_nonzero(violation_mask).item())
        max_increase = float(torch.max(increases[violation_mask]).item())

        # Sigma values are non-negative. cummin preserves every already-valid decline and
        # clips only values that would rise above the lowest sigma reached so far.
        repaired = torch.cummin(torch.clamp(values, min=0), dim=0).values
        if repaired.numel():
            repaired[-1] = torch.zeros((), device=repaired.device, dtype=repaired.dtype)

        self._monotonic_repair_applied = True
        self._monotonic_repair_count = violation_count
        self._monotonic_repair_max_increase = max_increase
        warning = (
            "Simple KES repaired "
            f"{violation_count} upward sigma transition(s) produced by the selected "
            "advanced scheduler features so the schedule remains non-increasing."
        )
        warnings = list(self.settings.get("_validation_warnings", []) or [])
        if warning not in warnings:
            warnings.append(warning)
        self.settings["_validation_warnings"] = warnings
        self.log(
            "[Scheduler Safety] Repaired non-increasing sigma schedule: "
            f"violations={violation_count}, max_increase={max_increase:.8f}."
        )
        return repaired

    def _resolve_effective_history_window(
        self,
        full_sigmas: torch.Tensor,
        target_len: int,
        requested_history_window: int,
    ) -> int:
        max_history_window = max(3, min(int(full_sigmas.numel()) - 1, target_len - 1))
        requested_history_window = max(3, int(requested_history_window))
        requested_history_window = min(requested_history_window, max_history_window)

        if not bool(getattr(self, "auto_history", False)):
            return requested_history_window

        # Conservative first-pass heuristic:
        # start from a percentage of visible target length, capped by configured value.
        adaptive_history = max(4, int(target_len * 0.25))
        adaptive_history = min(adaptive_history, requested_history_window)

        # If the visible tail already looks flat, bias more local.
        visible = full_sigmas[:target_len]
        if visible.numel() >= 5:
            tail = visible[-5:]
            deltas = torch.abs(tail[:-1] - tail[1:])
            if deltas.numel() >= 2:
                delta_var = torch.var(deltas).item() if deltas.numel() > 1 else 0.0
                max_delta = torch.max(deltas).item()
                jaggedness_threshold = float(getattr(self, "jaggedness_threshold", 0.01))
                auto_tail_threshold = float(getattr(self, "auto_tail_threshold", 0.05))

                if max_delta <= auto_tail_threshold and delta_var <= jaggedness_threshold:
                    adaptive_history = max(3, min(adaptive_history, 4))

        self.log(
            f"[Compatibility] auto_history enabled: resolved history_window="
            f"{adaptive_history} (requested={requested_history_window}, target_len={target_len})"
        )
        return adaptive_history

    def _infer_tail_repair_pattern(
        self,
        history: torch.Tensor,
        fallback_pattern: str = "extrapolate",
    ) -> str:
        """
        Infer the best tail repair pattern from recent delta behavior.
        Returns one of the existing shared decay patterns.
        """
        if not isinstance(history, torch.Tensor) or history.numel() < 4:
            return fallback_pattern

        history = history.detach().flatten()
        deltas = torch.abs(history[:-1] - history[1:])

        if deltas.numel() < 2:
            return fallback_pattern

        delta_var = torch.var(deltas).item() if deltas.numel() > 1 else 0.0
        mean_delta = torch.mean(deltas).item()
        max_delta = torch.max(deltas).item()

        eps = 1e-8
        ratios = deltas[1:] / (deltas[:-1] + eps)
        ratio_var = torch.var(ratios).item() if ratios.numel() > 1 else 0.0
        mean_ratio = torch.mean(ratios).item() if ratios.numel() > 0 else 1.0

        jaggedness_threshold = float(getattr(self, "jaggedness_threshold", 0.01))

        # Very stable delta size -> linear-like tail
        if delta_var <= jaggedness_threshold * 0.25:
            return "linear"

        # Strongly multiplicative decay -> geometric/exponential-like
        if ratio_var <= jaggedness_threshold * 0.25:
            if mean_ratio < 0.85:
                return "exponential"
            if mean_ratio < 1.0:
                return "geometric"

        # Shrinking deltas but not multiplicatively stable -> harmonic/log-like
        if mean_ratio < 1.0:
            if max_delta > mean_delta * 1.5:
                return "harmonic"
            return "logarithmic"

        # Nearly constant last delta -> extrapolate current trend
        tail_delta_change = abs(deltas[-1].item() - deltas[-2].item()) if deltas.numel() >= 2 else 0.0
        if tail_delta_change <= jaggedness_threshold:
            return "extrapolate"

        return fallback_pattern

    def _truncate_and_repair_sigmas(
        self,
        sigmas: torch.Tensor,
        requested_steps: int,
        history_window: int = 10,
        repair_steps: int = 4,
    ) -> torch.Tensor:
        """
        Truncate a sigma sequence to requested_steps + 1 values and repair the
        exposed tail using an inferred shared decay pattern that best matches
        the local curve behavior.
        """
        if not isinstance(sigmas, torch.Tensor):
            raise TypeError(f"sigmas must be a torch.Tensor, got {type(sigmas)}")

        target_len = max(int(requested_steps), 0) + 1
        current_len = int(sigmas.numel())

        if current_len <= target_len:
            return sigmas

        full_sigmas = sigmas.detach().clone()
        truncated = full_sigmas[:target_len].clone()

        if truncated.numel() < 4:
            return truncated

        max_repair_steps = max(2, truncated.numel() - 2)
        repair_steps = max(2, int(repair_steps))
        repair_steps = min(repair_steps, max_repair_steps)

        history_window = self._resolve_effective_history_window(
            full_sigmas,
            target_len,
            history_window,
        )

        history_start = max(0, target_len - history_window - 1)
        history = full_sigmas[history_start:target_len]

        active_decay_pattern = str(getattr(self, "decay_pattern", "extrapolate") or "extrapolate").strip().lower()
        inferred_pattern = self._infer_tail_repair_pattern(
            history,
            fallback_pattern=active_decay_pattern,
        )

        

        # Keep prefix unchanged, rebuild only the last repair_steps values.
        prefix_len = truncated.numel() - repair_steps
        if prefix_len < 2:
            return truncated
        
        

        prefix = truncated[:prefix_len].clone()

        # Reuse the current shared tail generator behavior by extending from prefix.
        repaired_full = apply_decay_tail_fn(
            prefix,
            truncated.device,
            decay_pattern=inferred_pattern,
            tail_steps=repair_steps,
        )

        # apply_decay_tail_fn appends repair_steps values to prefix, giving us the target length.
        if repaired_full.numel() != truncated.numel():
            repaired_full = repaired_full[:truncated.numel()]

        # Safety clamp and monotonic cleanup
        repaired_full = torch.clamp(repaired_full, min=1e-5)

        if repaired_full.numel() >= 2:
            for i in range(1, repaired_full.numel()):
                if repaired_full[i] > repaired_full[i - 1]:
                    repaired_full[i] = repaired_full[i - 1]

        self.log(
            f"[Compatibility] Truncated sigma sequence from {current_len} to {target_len} "
            f"and repaired tail using pattern '{inferred_pattern}' over {repair_steps} steps."
        )

        return repaired_full

    def run_auto_stabilization(self, *_args):
        if not self.allow_step_expansion:
            self.log("[Auto Mode] Step expansion is disabled by configuration. Skipping auto stabilization.")
            return self.sigs
        unstable = self.detect_sequence_instability()

        if not unstable:
            self.log("[Auto Mode] Sigma sequence is already stable.")
            return

        self.log("[Auto Mode] Detected instability in sigma sequence. Starting stabilization sequence.")

        for method in self.auto_stabilization_sequence:
            if not unstable:
                self.log(f"[Auto Mode] Sequence stabilized after {method}. Stopping further corrections.")
                break

            if method == 'smooth_interpolation':
                unstable = self.smooth_interpolation()
            elif method == 'append_tail':
                unstable = self.append_tail()
            elif method == 'blend_tail':
                unstable = self.blend_tail()
            elif method == 'apply_decay':
                unstable = self.apply_decay()
            elif method == 'progressive_decay':
                unstable = self.progressive_decay()
            else:
                self.log(f"[Auto Mode] Unknown stabilization method: {method}")

    def detect_sequence_instability(self):
        delta_sigmas = self.sigs[:-1] - self.sigs[1:]
        second_deltas = torch.diff(delta_sigmas)

        steep_drop_detected = torch.any(delta_sigmas > self.auto_tail_threshold)
        jaggedness_score = torch.var(second_deltas[-5:]) if len(second_deltas) >= 5 else 0
        jagged_transition_detected = jaggedness_score > self.jaggedness_threshold

        if steep_drop_detected:
            self.log(f"[Auto Mode] Steep drop detected. Max drop: {torch.max(delta_sigmas).item():.6f}")
        if jagged_transition_detected:
            self.log(f"[Auto Mode] Jagged transition detected. Jaggedness score: {jaggedness_score:.6f}")

        return steep_drop_detected or jagged_transition_detected

    def smooth_interpolation(self):
        self.log("[Auto Mode] Applying smooth interpolation to last 5 steps.")
        if len(self.sigs) >= 5:
            start = self.sigs[-6].item()
            end = self.sigs[-1].item()
            interpolated = torch.linspace(start, end, steps=6, device=self.device)[1:]
            self.sigs[-5:] = interpolated

        return self.detect_sequence_instability()

    def append_tail(self):
        self.log("[Auto Mode] Attempting to append available tail.")
        if hasattr(self, 'all_tails') and self.all_tails:
            for tail in self.all_tails:
                if tail is not None:
                    tail = tail.to(self.device)
                    if tail.shape[0] > self.sigs.shape[0]:
                        tail = tail[:len(self.sigs)]
                    self.sigs = torch.cat([self.sigs, tail])
                    self.log("[Auto Mode] Appended tail to sigma sequence.")
                    self._auto_stabilization_applied = True
                    self._tail_steps_applied = True
                    self._step_expansion_applied = True
                    break

        return self.detect_sequence_instability()

    def blend_tail(self):
        if not hasattr(self, 'all_tails') or not self.all_tails:
            self.log("[Auto Mode] No available tails to blend.")
            return self.detect_sequence_instability()

        self.log("[Auto Mode] Attempting to blend multiple tails.")
        blended_tail = None
        total_weight = 0

        for w, tail in zip(self.blend_weights, self.all_tails):
            if tail is not None:
                tail = tail.to(self.device)
                if tail.shape[0] > self.sigs.shape[0]:
                    tail = tail[:len(self.sigs)]

                if blended_tail is None:
                    blended_tail = w * tail
                else:
                    blended_tail += w * tail
                total_weight += w

        if blended_tail is not None and total_weight > 0:
            blended_tail /= total_weight
            self.sigs = torch.cat([self.sigs, blended_tail])
            self.log("[Auto Mode] Appended blended tail to sigma sequence.")
            self._auto_stabilization_applied = True
            self._blended_tail_applied = True
            self._step_expansion_applied = True

        return self.detect_sequence_instability()

    def apply_decay(self):
        self.log("[Auto Mode] Attempting to append decay tails.")
        if hasattr(self, 'all_decays') and self.all_decays:
            for decay in self.all_decays:
                if decay is not None:
                    decay = decay.to(self.device)
                    if decay.shape[0] > self.sigs.shape[0]:
                        decay = decay[:len(self.sigs)]

                    self.sigs = torch.cat([self.sigs, decay])
                    self.log("[Auto Mode] Appended decay tail to sigma sequence.")
                    self._auto_stabilization_applied = True
                    self._decay_tail_applied = True
                    self._step_expansion_applied = True
                    break

        return self.detect_sequence_instability()

    def progressive_decay(self):
        self.log("[Auto Mode] Applying progressive decay to sigma sequence.")
        progressive_decay = None
        total_weight = 0

        for w, decay in zip(self.blend_weights, self.all_decays):
            if decay is not None:
                decay = decay.to(self.device)
                if decay.shape[0] != self.sigs.shape[0]:
                    decay = decay.view(1, 1, -1)
                    decay = F.interpolate(decay, size=self.sigs.shape[0], mode='linear', align_corners=False)
                    decay = decay.view(-1)

                if progressive_decay is None:
                    progressive_decay = w * decay
                else:
                    progressive_decay += w * decay

                total_weight += w

        if progressive_decay is not None and total_weight > 0:
            progressive_decay /= total_weight
            self.sigs = self.sigs * progressive_decay
            self.log("[Auto Mode] Applied progressive decay to sigma sequence.")
            self._auto_stabilization_applied = True
            self._progressive_decay_applied = True

        return self.detect_sequence_instability()
