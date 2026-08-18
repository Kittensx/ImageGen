from __future__ import annotations

import copy
import glob
import hashlib
import os
import re

import torch

class SigmaCacheMixin:
    """Internal Simple KES responsibility mixin.

    This class is composed into ``SimpleKEScheduler`` so the public scheduler
    method surface remains unchanged while implementation concerns stay isolated.
    """

    def _safe_sigma_loader(self, cache_key):
        cache_folder = self.sigma_save_folder
        if not os.path.exists(cache_folder) or not os.listdir(cache_folder):
            self.log(f"[Cache Check] Cache folder {cache_folder} is empty or missing. Skipping load.")
            return None

        matching_files = [f for f in os.listdir(cache_folder) if cache_key in f and f.endswith('.pt')]
        if not matching_files:
            self.log(f"[Cache Check] No matching cache file found for key: {cache_key}. Skipping load.")
            return None

        filename = os.path.join(cache_folder, matching_files[0])
        self.log(f"[Cache Hit] Loading sigma cache from: {filename}")
        loaded_data = torch.load(filename, map_location=self.device)
        return loaded_data['sigma_values'].to(self.device)



    def build_sigma_cache_filename(self, steps, sigma_min, sigma_max, rho=None, schedule_type='karras', decay_pattern='zero', cache_dir=None, suffix=None, ext=None):
        ext = ext or 'txt'
        if cache_dir is None:
            cache_dir = self.cache_dir or 'cache'
        if schedule_type == 'karras':
            base_filename = f'sigma_{schedule_type}_{steps}steps_rho{rho}_min{sigma_min}_max{sigma_max}_{decay_pattern}'
        else:
            base_filename = f'sigma_{schedule_type}_{steps}steps_min{sigma_min}_max{sigma_max}_{decay_pattern}'

        if suffix:
            base_filename += f'_{suffix}'
            version = self.get_next_version_number(cache_dir, base_filename, ext)
            filename = f'{version:03d}_{base_filename}.{ext}'
        else:
            filename = f'{base_filename}.{ext}'

        return os.path.join(cache_dir, filename)

    def get_next_version_number(self, cache_dir, base_filename, ext=None):
        pattern = os.path.join(cache_dir, f'*_{base_filename}')
        if ext:
            pattern = os.path.join(cache_dir, f'*_{base_filename}.{ext}')
        existing_files = glob.glob(pattern)

        version_numbers = []
        for file in existing_files:
            match = re.search(r'(\d{3})_' + re.escape(base_filename), os.path.basename(file))
            if match:
                version_numbers.append(int(match.group(1)))

        return max(version_numbers) + 1 if version_numbers else 1

    def get_sigma_with_cache(self, steps, sigma_min, sigma_max, rho=7.0, device='cpu',
                             schedule_type='karras', decay_pattern=None, cache_dir=None, cache_file=None,
                             suffix=None, ext=None, mode=None, cache_key=None):
        self.steps = steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.device = device
        self.schedule_type = schedule_type
        self.decay_pattern = decay_pattern
        self.cache_dir = cache_dir
        self.cache_file = cache_file
        self.suffix = suffix
        self.ext = ext
        self.mode = mode
        self.cache_key = cache_key

        cached_sigmas = self.get_sigma_from_cache(cache_key)
        if cached_sigmas is not None:
            return cached_sigmas

        if self.is_sigma_randomized():
            _, _, _, sigmas = self._generate_sigmas(steps, sigma_min, sigma_max, rho, device, schedule_type, decay_pattern)
            self.sigma_cache[cache_key] = sigmas
            return sigmas

        if self.loaded_sigmas is None:
            _, _, _, sigmas = self._generate_sigmas(steps, sigma_min, sigma_max, rho, device, schedule_type, decay_pattern)
            self.loaded_sigmas = sigmas
            self.sigma_cache[cache_key] = sigmas
            return sigmas

        if mode == 'prepass':
            self.cache_file = self.prepass_save_file
        elif mode == 'final':
            self.cache_file = self.final_save_file
        else:
            self.cache_file = self.build_sigma_cache_filename(steps, sigma_min, sigma_max, rho, device, schedule_type, decay_pattern, cache_dir)

        if mode in ['prepass', 'final'] and self.load_prepass_sigmas:
            loaded_sigmas = self.load_sigmas_with_hash_validation(
                filename=self.cache_file,
                steps=steps,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                rho=rho,
                device=device,
                schedule_type=schedule_type,
                decay_pattern=decay_pattern,
                cache_key=cache_key
            )

            if loaded_sigmas is not None:
                self.loaded_sigmas = loaded_sigmas
                self.sigma_cache[cache_key] = loaded_sigmas
                return loaded_sigmas.to(device)

            self.log("[Cache Recovery] Cache load failed. Recalculating sigma schedule.")
            _, _, _, sigmas = self._generate_sigmas(steps, sigma_min, sigma_max, rho, device, schedule_type, decay_pattern)
            self.sigma_cache[cache_key] = sigmas
            return sigmas

        _, _, _, sigmas = self._generate_sigmas(steps, sigma_min, sigma_max, rho, device, schedule_type, decay_pattern)
        self.sigma_cache[cache_key] = sigmas
        return sigmas

    def load_sigmas_with_hash_validation(self, filename, steps, sigma_min, sigma_max, rho, device, schedule_type, decay_pattern, save_data=None, cache_key=None, suffix=None):
        if self.load_prepass_sigmas and cache_key:
            try:
                loaded_data = torch.load(filename, map_location=self.device)
                self.loaded_sigmas = loaded_data['sigma_values'].to(self.device)
                loaded_hash = loaded_data['sigma_hash']
                expected_hash = self.generate_sigma_hash(steps, sigma_min, sigma_max, rho, device, schedule_type, decay_pattern, save_data, suffix)

                if loaded_hash != expected_hash:
                    self.log(f"[Sigma Validator] Hash mismatch. Expected: {expected_hash}, Found: {loaded_hash}. Recalculating.")
                    return None

                self.log(f"[Sigma Validator] Hash validated successfully for file: {filename}")
                return self.loaded_sigmas
            except Exception:
                self.log("[Cache Recovery] Sigma cache invalid or missing. Recalculating sigmas.")
                _, _, _, sigmas = self._generate_sigmas(steps, sigma_min, sigma_max, rho, device, schedule_type, decay_pattern)
                return sigmas
        return None

    def generate_sigma_hash(self, steps, sigma_min, sigma_max, rho, device, schedule_type, decay_pattern, save_data=None, suffix=None):
        data_string = f'{steps}_{sigma_min}_{sigma_max}_{rho}_{device}_{schedule_type}_{decay_pattern}_{suffix}'
        hash_object = hashlib.sha256(data_string.encode())
        return hash_object.hexdigest()[:12]

    def load_or_regenerate_sigmas(self, cache_key):
        if self.load_sigma_cache and cache_key:
            try:
                loaded_data = torch.load(self.cache_file, map_location=self.device)
                sigmas = loaded_data['sigma_values'].to(self.device)
                return sigmas
            except FileNotFoundError:
                self.log(f"[Cache Warning] Cache file not found: {self.cache_file}")
                self.log(f"[Cache Recovery] Automatically recomputing sigma schedule.")

        _, _, _, sigmas = self._generate_sigmas(
            self.steps,
            self.sigma_min,
            self.sigma_max,
            self.rho,
            self.device,
            self.schedule_type,
            self.decay_pattern
        )
        return sigmas

    def get_sigma_from_cache(self, cache_key):
        if cache_key in self.sigma_cache:
            cached_sigmas = self.sigma_cache[cache_key]
            self.log(f"[Cache Hit] Returning cached sigma sequence for key: {cache_key}")

            if isinstance(cached_sigmas, torch.Tensor):
                return cached_sigmas.clone().detach().to(self.device)
            elif isinstance(cached_sigmas, list):
                return copy.deepcopy(cached_sigmas)
            return cached_sigmas

        self.log(f"[Cache Miss] Cache key not found: {cache_key}")
        return None
