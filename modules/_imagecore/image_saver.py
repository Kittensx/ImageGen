import os
from datetime import datetime
from PIL import Image
import numpy as np
from typing import Optional
import inspect

class ImageSaver:
    def __init__(
        self,
        image_np: np.ndarray,
        image_path: str,
        shared_state=None,
        bit_depth: Optional[int] = None,
        mode: Optional[str] = None,
        include_alpha: Optional[bool] = None,
        normalize: Optional[bool] = None,
        output_format: Optional[str] = None,
        verbose: Optional[bool]=False
    ):
        self.image_np = image_np
        self.image_path = image_path
        self.state = shared_state
        self.verbose = verbose
        if not self.state:
            # Get the caller function or class
            stack = inspect.stack()
            caller = stack[1]
            caller_name = caller.function
            caller_file = caller.filename
            caller_line = caller.lineno

            print(f"[Image Saver] ❌ No shared_state passed.")
            print(f"  ↳ Called by: {caller_name} in {caller_file}:{caller_line}")

            raise ValueError("SharedState was required but not provided.")

        # If shared_state.image exists, use its values as fallback
        image_cfg = getattr(self.state, "image", {}) if self.state else {}

        self.bit_depth = bit_depth if bit_depth is not None else getattr(image_cfg, "bit_depth", 8)
        self.mode = mode if mode is not None else getattr(image_cfg, "mode", "auto")
        self.include_alpha = include_alpha if include_alpha is not None else getattr(image_cfg, "include_alpha", False)
        self.normalize = normalize if normalize is not None else getattr(image_cfg, "normalize_float", True)
        self.output_format = output_format if output_format is not None else getattr(image_cfg, "output_format", "png")



    def save_image(self):                
        fmt = self.output_format
        bit_depth = self.bit_depth
        mode = self.mode
        include_alpha = self.include_alpha
        normalize = self.normalize_float

        if mode == "auto":
            shape = self.image_np.shape
            if len(shape) == 2:
                mode = "L"
            elif shape[2] == 3:
                mode = "RGB"
            elif shape[2] == 4:
                mode = "RGBA"

        # Bit depth handling
        if bit_depth == 8:
            image_out = Image.fromarray(self.image_np.astype(np.uint8), mode=mode)
        elif bit_depth == 16:
            image_out = Image.fromarray(self.image_np.astype(np.uint16), mode=mode)
        elif bit_depth == 32 and fmt in ["exr", "tiff"]:
            if normalize:
                self.image_np = np.clip(self.image_np, 0, 1)
            image_out = Image.fromarray(self.image_np.astype(np.float32), mode=mode)
        else:
            raise ValueError(f"Unsupported bit depth/format combination: {bit_depth}, {fmt}")

        # Format-specific params
        save_kwargs = {}
        if fmt == "jpeg":
            save_kwargs["quality"] = 95
        elif fmt == "tiff" and bit_depth == 16:
            save_kwargs["bits"] = (16,)

        final_path = self.get_final_image_path()
        image_out.save(final_path, format=fmt.upper(), **save_kwargs)
        return final_path

    def get_final_image_path(self) -> str:
        base, _ = os.path.splitext(self.image_path)
        ext = "." + self.output_format.lower()
        return base + ext