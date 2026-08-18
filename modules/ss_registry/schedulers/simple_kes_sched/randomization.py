from __future__ import annotations

import math
import random

from modules.ss_registry.schedulers.simple_kes_sched.simple_kes_config import KES_RANDOMIZATION_SAFE_BOUNDS

class RandomizationMixin:
    """Internal Simple KES responsibility mixin.

    This class is composed into ``SimpleKEScheduler`` so the public scheduler
    method surface remains unchanged while implementation concerns stay isolated.
    """

    def apply_global_randomization(self):
        """Enable only the scheduler's declared randomization controls.

        The global switch never derives or stretches ranges. It only turns on
        the known min/max controls already present in the effective settings.
        """
        for base_key in KES_RANDOMIZATION_SAFE_BOUNDS:
            min_key = f"{base_key}_rand_min"
            max_key = f"{base_key}_rand_max"
            if min_key in self.settings and max_key in self.settings:
                self.settings[f"{base_key}_rand"] = True

    def _effective_randomization_bounds(self, key_prefix, default_value):
        safe_min, safe_max = KES_RANDOMIZATION_SAFE_BOUNDS.get(
            key_prefix,
            (float(default_value), float(default_value)),
        )
        try:
            rand_min = float(self.settings.get(f"{key_prefix}_rand_min", safe_min))
        except (TypeError, ValueError):
            rand_min = float(safe_min)
        try:
            rand_max = float(self.settings.get(f"{key_prefix}_rand_max", safe_max))
        except (TypeError, ValueError):
            rand_max = float(safe_max)

        if not math.isfinite(rand_min):
            rand_min = float(safe_min)
        if not math.isfinite(rand_max):
            rand_max = float(safe_max)
        if rand_min > rand_max:
            rand_min, rand_max = rand_max, rand_min

        override_enabled = bool(self.settings.get("allow_randomization_range_override", False))
        if not override_enabled:
            rand_min = max(rand_min, float(safe_min))
            rand_max = min(rand_max, float(safe_max))
            if rand_min > rand_max:
                rand_min, rand_max = float(safe_min), float(safe_max)

        return rand_min, rand_max

    def _clamp_randomized_value(self, key_prefix, value, default_value):
        rand_min, rand_max = self._effective_randomization_bounds(key_prefix, default_value)
        clamped = min(max(float(value), rand_min), rand_max)
        if clamped != float(value):
            self.log(
                f"[Randomization Safety] {key_prefix}: clamped {value} to {clamped} "
                f"inside configured range {rand_min} to {rand_max}."
            )
        return clamped

    def get_randomization_type(self, key_prefix):
        randomization_type_raw = self.settings.get(f'{key_prefix}_randomization_type', 'asymmetric')
        return self.RANDOMIZATION_TYPE_ALIASES.get(str(randomization_type_raw).lower(), 'asymmetric')

    def get_randomization_percent(self, key_prefix):
        try:
            value = float(self.settings.get(f'{key_prefix}_randomization_percent', 0.2))
        except (TypeError, ValueError):
            value = 0.2
        return max(0.0, value)

    def get_random_between_min_max(self, key_prefix, default_value):
        randomize_flag = self.settings.get(f'{key_prefix}_rand', False)
        if randomize_flag:
            rand_min, rand_max = self._effective_randomization_bounds(key_prefix, default_value)
            if rand_min == rand_max:
                self.log(f"[Random Range] {key_prefix}: min and max are equal ({rand_min}). Using single value.")
                return rand_min
            value = random.uniform(rand_min, rand_max)
            self.log(f"[Random Range] {key_prefix}: Picked random value {value} between {rand_min} and {rand_max}")
            return value
        self.log(f"[Random Range] {key_prefix}: Randomization is OFF. Using base value {default_value}")
        return default_value

    def get_random_by_type(self, key_prefix, default_value):
        randomization_enabled = self.settings.get(f'{key_prefix}_enable_randomization_type', False)
        if not randomization_enabled:
            self.log(f"[Randomization Type] {key_prefix}: Randomization type is OFF. Using base value {default_value}")
            return default_value

        randomization_type = self.get_randomization_type(key_prefix)
        randomization_percent = self.get_randomization_percent(key_prefix)
        base_value = float(default_value)

        if randomization_type == 'symmetric':
            candidate_min = base_value * (1 - randomization_percent)
            candidate_max = base_value * (1 + randomization_percent)
            value = random.uniform(candidate_min, candidate_max)
            self.log(f"[Symmetric Randomization] {key_prefix}: Range {candidate_min} to {candidate_max}")
        elif randomization_type == 'asymmetric':
            candidate_min = base_value * (1 - randomization_percent)
            candidate_max = base_value * (1 + (randomization_percent * 2))
            value = random.uniform(candidate_min, candidate_max)
            self.log(f"[Asymmetric Randomization] {key_prefix}: Range {candidate_min} to {candidate_max}")
        elif randomization_type == 'logarithmic':
            positive_floor = max(base_value * (1 - randomization_percent), 1e-12)
            positive_ceiling = max(base_value * (1 + randomization_percent), positive_floor)
            value = math.exp(random.uniform(math.log(positive_floor), math.log(positive_ceiling)))
            self.log(f"[Logarithmic Randomization] {key_prefix}: Log-space randomization resulted in {value}")
        elif randomization_type == 'exponential':
            candidate_min = base_value * (1 - randomization_percent)
            candidate_max = base_value * (1 + randomization_percent)
            try:
                value = math.exp(random.uniform(candidate_min, candidate_max))
            except OverflowError:
                _range_min, range_max = self._effective_randomization_bounds(key_prefix, default_value)
                value = range_max
                self.log(
                    f"[Exponential Randomization] {key_prefix}: exponential candidate overflowed; "
                    f"using configured maximum {range_max}."
                )
            else:
                self.log(f"[Exponential Randomization] {key_prefix}: Randomized exponential value {value}")
        else:
            self.log(f"[Randomization Type] {key_prefix}: Invalid randomization type {randomization_type}. Using base value.")
            value = base_value

        return self._clamp_randomized_value(key_prefix, value, default_value)

    def _apply_runtime_randomization(self):
        for key in self.re_randomizable_keys:
            base_value = self.settings.get(key, getattr(self, key, None))
            if base_value is None:
                continue
            runtime_value = self.get_random_or_default(key, base_value)
            setattr(self, key, runtime_value)
            self.settings[key] = runtime_value

    def get_random_or_default(self, key_prefix, default_value):
        rand_type_enabled = self.settings.get(f'{key_prefix}_enable_randomization_type', False)
        min_max_enabled = self.settings.get(f'{key_prefix}_rand', False)

        if self.global_randomize:
            result_value = self.get_random_between_min_max(key_prefix, default_value)
            self.log(
                f"[Global Randomization] {key_prefix}: used the configured min/max range. "
                f"Final value: {result_value}"
            )
        elif rand_type_enabled and min_max_enabled:
            self.log(
                f"[Randomization Policy] Both min/max and randomization type enabled for {key_prefix}. "
                "The randomization type is applied and then clamped to the configured min/max range."
            )
            result_value = self.get_random_by_type(key_prefix, default_value)
        elif rand_type_enabled:
            result_value = self.get_random_by_type(key_prefix, default_value)
            self.log(f"[Randomization] {key_prefix}: Applied randomization type. Final value: {result_value}")
        elif min_max_enabled:
            result_value = self.get_random_between_min_max(key_prefix, default_value)
            self.log(f"[Randomization] {key_prefix}: Applied min/max randomization. Final value: {result_value}")
        else:
            result_value = default_value
            self.log(f"[Randomization] {key_prefix}: No randomization applied. Using default value: {result_value}")

        return result_value

    def is_sigma_randomized(self):
        return (
            self.settings.get('sigma_min_rand', False) or
            self.settings.get('sigma_max_rand', False) or
            self.settings.get('rho_rand', False) or
            self.settings.get('sigma_max_enable_randomization_type', False) or
            self.settings.get('sigma_min_enable_randomization_type', False) or
            self.settings.get('rho_enable_randomization_type', False)
        )
