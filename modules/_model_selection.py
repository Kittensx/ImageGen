class ModelSelector:
    def __init__(self, registry, config):
        self.registry = registry
        self.config = config

    def get_default_model_path(self) -> str:
        # 1. last successfully loaded model
        model = self._get_last_loaded_model()
        if model:
            return model.path

        # 2. most recent checkpoint in registry
        model = self._get_latest_checkpoint()
        if model:
            return model.path

        # 3. first file in checkpoints directory
        model = self._get_first_checkpoint_from_folder()
        if model:
            return model

        # 4. fallback
        return self.config.MODEL_PATH