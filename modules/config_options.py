from __future__ import annotations

from datetime import datetime
from pathlib import Path

from modules.project_context import ProjectContext


class ConfigOptions:
    """Compatibility view over the canonical :class:`ProjectContext`.

    This constructor is deliberately read-only. It resolves configuration and
    paths but does not create model, data, output, cache, or temporary folders.
    """

    def __init__(self, project_context: ProjectContext | None = None):
        self.context = project_context or ProjectContext.load()
        context = self.context

        self.root_path = str(context.modules_root)
        self.project_root = str(context.project_root)
        self.user_config_path = str(context.config_path.parent)
        self.user_config_file = str(context.config_path)
        self.config = context.config

        self.images_root = str(context.output_root)
        self.txt2img_root = str(context.txt2img_output_root)
        today_str = datetime.now().strftime("%Y-%m-%d")
        self.txt2img_save_dir = str(context.txt2img_output_root / today_str)
        self.output_folder = str(context.output_root)

        self.local_config_dir = str(context.local_config_dir)
        self.local_tokenizer_dir = str(context.tokenizer_root)

        self.models_root = str(context.models_root)
        self.checkpoints_dir = str(context.checkpoints_dir)
        self.vae_dir = str(context.vae_dir)
        self.lora_dir = str(context.lora_dir)
        self.vae_approx_dir = str(context.vae_approx_dir)
        self.blip_dir = str(context.blip_dir)
        self.codeformer_dir = str(context.codeformer_dir)
        self.esrgan_dir = str(context.esrgan_dir)
        self.gfpgan_dir = str(context.gfpgan_dir)
        self.realesrgan_dir = str(context.realesrgan_dir)
        self.controlnet_dir = str(context.controlnet_dir)
        self.embeddings_dir = str(context.embeddings_dir)
        self.hypernetworks_dir = str(context.hypernetworks_dir)

        self.data_dir = str(context.data_root)
        self.cache_dir = str(context.cache_root)
        self.temporary_dir = str(context.temporary_root)
        self.registry_db_path = str(context.registry_db_path)
        self.MODEL_PATH = str(context.default_model_path) if context.default_model_path else None

    def ensure_runtime_directories(self) -> tuple[Path, ...]:
        """Explicitly create writable runtime-owned directories.

        Callers must opt into this operation; construction never mutates disk.
        Model-library and tokenizer directories are intentionally not created.
        """

        directories = (
            self.context.data_root,
            self.context.cache_root,
            self.context.temporary_root,
            self.context.output_root,
            self.context.txt2img_output_root,
        )
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        return directories
