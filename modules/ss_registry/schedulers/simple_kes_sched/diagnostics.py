from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import json
import os

import numpy as np
import torch

from modules.ss_registry.schedulers.simple_kes_sched.utils.plot_sigma_sequence import plot_sigma_sequence

class SharedLogger:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.log_buffer = []
        self.prepass_log_buffer = []

    def log(self, message):
        if self.debug:
            self.log_buffer.append(message)

    def prepass_log(self, message):
        if self.debug:
            self.prepass_log_buffer.append(message)

class DiagnosticsMixin:
    """Internal Simple KES responsibility mixin.

    This class is composed into ``SimpleKEScheduler`` so the public scheduler
    method surface remains unchanged while implementation concerns stay isolated.
    """

    @staticmethod
    def _canonical_data_directory_name(value: str) -> str:
        """Correct known historical misspellings of image_generation_data."""

        replacements = {
            "image_gneraetion_data": "image_generation_data",
            "image_generaetion_data": "image_generation_data",
            "image_genaration_data": "image_generation_data",
        }
        path = Path(str(value or "image_generation_data").strip())
        corrected_name = replacements.get(path.name.casefold(), path.name)
        if corrected_name == path.name:
            return str(path)
        return str(path.with_name(corrected_name))

    def _resolve_data_directory(self, value: Any) -> str:
        raw = self._canonical_data_directory_name(str(value or "image_generation_data"))
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            project_root = Path(__file__).resolve().parents[4]
            candidate = project_root / candidate
        return str(candidate.resolve())

    def _create_directories(self):
        graph_directory = self._resolve_data_directory(
            self.settings.get("graph_save_directory", "image_generation_data")
        )
        log_directory = self._resolve_data_directory(
            self.settings.get("log_save_directory", "image_generation_data")
        )
        self.graph_save_directory = graph_directory
        self.log_save_directory = log_directory
        self.settings["graph_save_directory"] = graph_directory
        self.settings["log_save_directory"] = log_directory

        paths = [
            self.cache_dir,
            self.sigma_save_folder,
            graph_directory,
            log_directory,
        ]

        for directory in paths:
            try:
                if directory:
                    os.makedirs(directory, exist_ok=True)
            except Exception as exc:
                self.log(f"[Init Warning] Failed to create directory {directory}: {exc}")

    def _log_extras_to_file(self, all_extras):
        if not self.extras_log_filename:
            return

        try:
            with open(self.extras_log_filename, "a", encoding="utf-8") as f:
                f.write("\n=== New Scheduler Extras ===\n")

                for method in self.blend_methods:
                    seq = self.sigma_sequences.get(method, {})
                    sigmas = seq.get("sigmas")
                    tails = seq.get("tails")
                    decay = seq.get("decay")
                    extras = seq.get("extras", [])

                    f.write(f"\nScheduler: {method}\n")

                    # method config
                    method_cfg = self.blend_method_dict.get(method, {})
                    f.write("Method settings:\n")
                    try:
                        f.write(json.dumps(method_cfg, indent=2))
                    except TypeError:
                        f.write(str(method_cfg))
                    f.write("\n")

                    # sigma preview
                    if isinstance(sigmas, torch.Tensor):
                        flat = sigmas.detach().flatten().cpu()
                        f.write(f"Sigma count: {flat.numel()}\n")
                        if flat.numel() > 0:
                            f.write(f"First sigma: {flat[0].item():.10f}\n")
                            f.write(f"Last sigma: {flat[-1].item():.10f}\n")
                            f.write(
                                f"First 5 sigmas: {[float(x) for x in flat[:5]]}\n"
                            )
                            f.write(
                                f"Last 5 sigmas: {[float(x) for x in flat[-5:]]}\n"
                            )
                    else:
                        f.write(f"Sigmas: {sigmas}\n")

                    if tails is not None:
                        if isinstance(tails, torch.Tensor):
                            f.write(f"Tail count: {tails.numel()}\n")
                        else:
                            f.write(f"Tails: {tails}\n")

                    if decay is not None:
                        if isinstance(decay, torch.Tensor):
                            f.write(f"Decay count: {decay.numel()}\n")
                        else:
                            f.write(f"Decay: {decay}\n")

                    # extras
                    if extras:
                        f.write("Extras:\n")
                        try:
                            f.write(json.dumps(extras, indent=2))
                        except TypeError:
                            f.write(str(extras))
                        f.write("\n")
                    else:
                        f.write("Extras: []\n")

                f.write("\n============================\n")
        except Exception as e:
            self.log(f"[Extras Log Warning] Failed to write extras log: {e}")

    def initialize_generation_filename(self, folder=None, base_name="generation_log", ext="txt"):
        if folder is None:
            folder = self._resolve_data_directory(
                self.settings.get("log_save_directory", "image_generation_data")
            )
        else:
            folder = self._resolve_data_directory(folder)

        os.makedirs(folder, exist_ok=True)
        # Microseconds prevent two images prepared in the same second from
        # overwriting one another's final plot or generation log.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.log_filename = os.path.join(folder, f"{base_name}_{timestamp}.{ext}")

    def save_generation_settings(self):
        with open(self.log_filename, "w", encoding='utf-8') as f:
            for line in self.logger.log_buffer:
                f.write(f"{line}\n")
            for line in self.logger.prepass_log_buffer:
                f.write(f"{line}\n")
        self.log(f"[SimpleKEScheduler] Generation settings saved to {self.log_filename}")
        self.logger.log_buffer.clear()
        self.logger.prepass_log_buffer.clear()

    def save_image_plot(self, sigs, stopping_index):
        """Save the completed sigma schedule once without opening a window."""

        graph_plot = plot_sigma_sequence(
            sigs,
            stopping_index,
            self.log_filename,
            self.graph_save_directory,
            False,
        )
        self.log(f"Final sigma sequence plot saved to {graph_plot}")
        return graph_plot

    def save_sigmas_as_csv(self, sigmas, filename):
        np.savetxt(filename, sigmas.cpu().numpy(), delimiter=",")
