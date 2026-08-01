import torch.nn as nn
import torch
from dataclasses import dataclass, field, asdict, fields, is_dataclass
from typing import Optional, Tuple, List, Union, Dict, Any


def is_nested_dataclass(value: Any) -> bool:
    return is_dataclass(value) and not isinstance(value, type)

@dataclass
class LastSettingsUsed:  # lsu
    
    last_settings: Dict[str, Any] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)

   
    
    def update_settings(self, exclude: str = "settings"):
        self.settings = {
            k: v for k, v in asdict(self).items()
            if k != exclude
        }
   
    def update(self, source: Optional[Dict[str, Any]] = None):
        if source is None:
            print("[LSU] No source provided, skipping update.")
            return
        if not isinstance(source, dict):
            raise TypeError(f"[LSU] Expected dict, got {type(source)}")

        self.last_settings = self.settings.copy()
        self.settings = source.copy()
        
    def to_dict(self) -> dict:
        return asdict(self)
        
    def print_summary(self):
        print(f"[{self.__class__.__name__}] Field values:")
        for f in fields(self):
            print(f"  {f.name}: {getattr(self, f.name, None)}")
    def update_from_dict(self, d: dict):
        for f in fields(self):
            if f.name in d:
                setattr(self, f.name, d[f.name])
    def get_field_names(self) -> list:
        return [f.name for f in fields(self)]
    def is_valid(self) -> bool:
        """
        Returns True if all fields are non-None (basic validity check).
        Optional/complex types are not deeply inspected.
        """
        for f in fields(self):
            value = getattr(self, f.name, None)
            if value is None:
                return False
        return True
        
    def diff(self, return_diff: bool = False) -> Union[None, Dict[str, Dict[str, Any]]]:
        """
        Compares last_settings to current settings and returns or prints the difference.

        Returns:
            A dict of the changed keys with old and new values, if `return_diff` is True.
        """
        changes = {}
        for key in self.settings:
            current = self.settings.get(key)
            previous = self.last_settings.get(key)

            if current != previous:
                changes[key] = {
                    "from": previous,
                    "to": current
                }

        if return_diff:
            return changes

        if not changes:
            print("[LSU] No changes found.")
        else:
            print("[LSU] Detected setting changes:")
            for k, v in changes.items():
                print(f"  {k}: {v['from']} → {v['to']}")



@dataclass
class TensorData:
    # Single most recent / active values
    sigma: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    sample: Optional[torch.Tensor] = field(default=None, repr=False)
    model: Optional[nn.Module] = field(default=None, repr=False)

    # Historical or multi-entry versions
    sigmas: List[torch.Tensor] = field(default_factory=list, repr=False)
    samples: List[torch.Tensor] = field(default_factory=list, repr=False)
    models: List[nn.Module] = field(default_factory=list, repr=False)

    # Full resolved settings (optional debug)
    settings: dict = field(default_factory=dict)
    denoised: Optional[torch.Tensor] = field(default=None, repr=False)
    denoiseds: List[torch.Tensor] = field(default_factory=list, repr=False)
    
    def set_denoised(self, value: torch.Tensor):
        self.denoised = value
        self.denoiseds.append(value.clone().detach())
        
    def update_settings(self, exclude: str = "settings"):
        self.settings = {
            k: v for k, v in asdict(self).items()
            if k != exclude
        }
    def set_sigma(self, value: torch.Tensor):
        self.sigma = value
        self.sigmas.append(value.clone().detach())

    def set_sample(self, value: torch.Tensor):
        self.sample = value
        self.samples.append(value.clone().detach())

    def set_model(self, value: nn.Module):
        self.model = value
        self.models.append(value)  # you may choose `.copy()` or deep copy if needed

    def get_last_sigma(self) -> Optional[torch.Tensor]:
        return self.sigmas[-1] if self.sigmas else None

    def get_last_sample(self) -> Optional[torch.Tensor]:
        return self.samples[-1] if self.samples else None

    def get_last_model(self) -> Optional[nn.Module]:
        return self.models[-1] if self.models else None

    def clear_history(self):
        self.sigmas.clear()
        self.samples.clear()
        self.models.clear()

    def to_dict(self):
        return {
            "sigma": self.sigma,
            "sample": self.sample,
            "model": self.model,
            "sigmas": self.sigmas,
            "samples": self.samples,
            "models": self.models,
        }

      
    def get(self, key: str, default=None):
        return getattr(self, key, self.settings.get(key, default))
        
@dataclass
class GenParams: #gen
    # === Required system-level inputs
    steps: int = 20
    cfg_scale: float = 7.5
    batch_size: int = 1
    shape: Tuple[int, int, int, int] = (1, 4, 64, 64)
    device: str = "cuda"
    settings: dict = field(default_factory=dict)
    
    # Full resolved settings (optional debug)
    settings: dict = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Return the complete dataclass as a dictionary (including extras).
        """
        return asdict(self)
    
    def get(self, key: str, default=None):
        return getattr(self, key, self.settings.get(key, default))
    
    def update_settings(self, exclude: str = "settings"):
        self.settings = {
            k: v for k, v in asdict(self).items()
            if k != exclude
        }
  
@dataclass
class SamplerConfig:
    sampler_type: str = "euler"
    eta: float = 0.0
    add_noise: bool = True
    verbose: bool = False

    rescale_cfg: bool = False
    rescale_cfg_factor: float = 1.0
    clamp_range: list[float] = field(default_factory=lambda: [-1.0, 1.0])

    initial_noise_strength: float = 0.0
    eta_scale_factor: float = 1.0
    eta_schedule_mode: str = "none"
    noise_schedule_scaling: str = "none"

    use_adaptive_eta: bool = False
    adaptive_time_mode: str = "none"
    adaptive_delta_low_floor: float = 0.1
    adaptive_delta_high_floor: float = 1.0
    adaptive_low_adjustment_multiplier: float = 1.5
    adaptive_high_adjustment_multiplier: float = 0.5
    adaptive_denoised_floor: float = 0.05
    adaptive_denoised_adjustment_multiplier: float = 1.25
    adaptive_manual_low_adjustment: float = 1.0
    adaptive_manual_high_adjustment: float = 1.0

    settings: dict = field(default_factory=dict)  

        
@dataclass
class SamplerState: #sampler_state
    
    
    gen: GenParams = field(default_factory=GenParams)
    tensor_data: TensorData = field(default_factory=TensorData)
    lsu: LastSettingsUsed = field(default_factory=LastSettingsUsed)
    cfg: SamplerConfig = field(default_factory=SamplerConfig)
    settings: dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Return the complete dataclass as a dictionary (including extras).
        """
        return asdict(self)
    
    def update_settings(self, exclude: str = "settings"):
        self.settings = {
            k: v for k, v in asdict(self).items()
            if k != exclude
        }
        
    def push_update(self, destination: Any, exclude: set[str] = None):
        """
        Pushes an update from self into `destination`, using `.settings` and `.last_settings`.

        Args:
            destination: A class with `.settings` and `.last_settings` attributes (e.g., self.lsu)
            exclude: Optional set of field names to exclude (e.g., {"lsu", "settings"})
        """
        if not hasattr(destination, "settings") or not hasattr(destination, "last_settings"):
            raise AttributeError("Destination must have both `settings` and `last_settings` attributes.")

        exclude = exclude or {"settings", "last_settings", "lsu"}
        collected = {}

        for f in fields(self):
            if f.name in exclude:
                continue

            value = getattr(self, f.name)
            if is_nested_dataclass(value):
                collected[f.name] = asdict(value)
            else:
                collected[f.name] = value

        # Perform update
        destination.last_settings = destination.settings.copy()
        destination.settings = collected    
        