from typing import Optional, List, Tuple, Dict,  Any, Union
from dataclasses import dataclass, field, fields, asdict, is_dataclass
import random
import torch


def is_nested_dataclass(value: Any) -> bool:
    return is_dataclass(value) and not isinstance(value, type)

"""
# ===== Code Changes ====# need to patch this new way in
    #old way = 
    #_rand_min
    #_rand_max 
    #example:
    #smooth_blend_factor_rand_min: 6
    #smooth_blend_factor_rand_max: 11 

    #new way: 
    #_bounds = [x,y]
    #example:
    #smooth_blend_factor_bounds = [6,11]
# ===== 
# =====
# New Modes added but need testing on the new pipeline:
# "append" when "tail_steps" are greater than 1
# Notes about these: Will need to be tested with the new pipeline to ensure we can pass a request to the sampler/main program to increase the steps and provide a smoother sigma schedule. If approved according to system settings, then the steps would be passed from the scheduler back to the main program / or directly to the sampler, along with the higher sigma schedule. 
# To this end, we should modify the flow to pass both sigma schedules for the proposed step count, and the original step count / or possibly work directly with the sampler on providing a new schedule if needed by the sampler?

# =====
#Valid decay modes compatible with A1111: all if tail_steps is not greater than 1. If any methods add steps that increase steps higher than what was requested, it is not compatible
#decay modes have been tested and they work. However if they increase steps beyond the requested amount, it will not work in the A1111 pipeline. If a pipeline supports increasing steps to have a smoother transition for sigma/noise reduction, then this method would function as intended - to increase steps to have a smoother transition & no jaggedness between steps. 
# supported v1.3 schedulers: euler, euler_advanced, geometric, harmonic, logarithmic, karras, exponential,
# =====
"""


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
class MiscVariables: #msc
    
    # --- Core prompt/tail blending strategy flags ---
    global_randomize: Optional[bool] = False
    allow_step_expansion: Optional[bool] = False
    apply_tail_steps: Optional[bool] = False
    apply_decay_tail: Optional[bool] = False
    apply_blended_tail: Optional[bool] = False
    apply_progressive_decay: Optional[bool] = False
    skip_prepass: Optional[bool] = True

    # --- Step + blending style ---
    decay_pattern: Optional[List[str]] = field(default_factory=lambda: ["zero", "soft_landing", "extrapolate", "fractional"])
    decay_mode: Optional[List[str]] = field(default_factory=lambda: ["append", "blend", "replace"])
    blending_mode: Optional[List[str]] = field(default_factory=lambda: ["auto", "default", "smooth_blend", "weights"])
    blending_style: Optional[List[str]] = field(default_factory=lambda: ["softmax", "explicit"])
    step_progress_mode: Optional[List[str]] = field(default_factory=lambda: ["linear", "sigmoid", "exponential", "logarithmic"])
    early_stopping_method: Optional[List[str]] = field(default_factory=lambda: ["max", "mean", "sum"])

    # --- Numerical config values ---
    exp_power: Optional[float] = 2.0
    sigma_variance_scale: Optional[float] = 0.1
    safety_minimum_stop_step: Optional[int] = 10
    recent_change_convergence_delta: Optional[float] = 0.6
    sharpen_variance_threshold: Optional[float] = 0.01
    tail_steps: Optional[int] = 1
    auto_tail_threshold: Optional[float] = 0.05
    jaggedness_threshold: Optional[float] = 0.01
    blend_midpoint: Optional[float] = 0.5
    smooth_blend_factor: Optional[float] = 5.0

    # --- Device/runtime info ---
    device: Optional[str] = "cuda"
    sigma_auto_enabled: Optional[bool] = True
    sigma_auto_mode: Optional[List[str]] = field(default_factory=lambda: ["sigma_min", "sigma_max"])

    '''
    valid_decay_patterns: ClassVar[Dict[str, List[str]]] = [
        "geometric", "harmonic", "extrapolate", "fractional",
        "logarithmic", "exponential", "linear", "zero"
    ]
    valid_decay_modes: ClassVar[Dict[str, List[str]]] = ["append", "blend", "replace"]
    '''

    # --- Debug output snapshot ---
    settings: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)
        
    def update_settings(self, exclude: str = "settings"):
        self.settings = {
            k: v for k, v in asdict(self).items()
            if k != exclude
        }

@dataclass
class DefaultRangeValue:  # def_range
    sigma_scale_factor: Tuple[float, float] = (700, 1100)
    smooth_blend_factor: Tuple[float, float] = (6, 11)
    rho_bounds: Tuple[float, float] = (3, 8)
    sigma_min_bounds: Tuple[float, float] = (0.001, 0.2)
    sigma_max_bounds: Tuple[float, float] = (25, 60)
    start_blend_bounds: Tuple[float, float] = (0.04, 0.11)
    sharpness_bounds: Tuple[float, float] = (0.75, 0.95)
    end_blend_bounds: Tuple[float, float] = (0.4, 0.6)
    initial_step_size_bounds: Tuple[float, float] = (0.7, 1.0)
    final_step_size_bounds: Tuple[float, float] = (0.1, 0.3)
    step_size_factor_bounds: Tuple[float, float] = (0.65, 0.85)
    initial_noise_scale_bounds: Tuple[float, float] = (1.0, 1.5)
    final_noise_scale_bounds: Tuple[float, float] = (0.6, 1.0)
    noise_scale_factor_bounds: Tuple[float, float] = (0.75, 0.95)
    early_stopping_threshold_bounds: Tuple[float, float] = (0.001, 0.02)

    settings: Dict[str, Any] = field(default_factory=dict)

    # === Helper Methods ===
    def to_dict(self) -> dict:
        return asdict(self)
    
    def update_settings(self, exclude: str = "settings"):
        self.settings = {
            k: v for k, v in asdict(self).items()
            if k != exclude
        }

        
    def get_midpoints(self) -> Dict[str, float]:
        """Return midpoints for all range fields."""
        return {
            f.name: round((v[0] + v[1]) / 2, 6)
            for f in fields(self)
            if isinstance((v := getattr(self, f.name, None)), tuple)
        }

    def sample_random(self, stretch: float = 0.0) -> Dict[str, float]:
        """Randomly sample a value within each range (optionally stretch the bounds)."""
        result = {}
        for f in fields(self):
            val = getattr(self, f.name, None)
            if isinstance(val, tuple) and len(val) == 2:
                low, high = val
                delta = (high - low) * stretch
                result[f.name] = round(random.uniform(low - delta, high + delta), 6)
        return result

    def get_range(self, key: str) -> Optional[Tuple[float, float]]:
        """Get the bounds tuple for a given key name."""
        return getattr(self, key, None)

    def print_all_ranges(self):
        """Pretty-print all range values."""
        print("=== Default Range Values ===")
        for f in fields(self):
            val = getattr(self, f.name, None)
            if isinstance(val, tuple):
                print(f"{f.name}: {val[0]} to {val[1]}")

    def flatten(self) -> Dict[str, Tuple[float, float]]:
        """Flatten all range values into a dict."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if isinstance(getattr(self, f.name, None), tuple)
        }

  

@dataclass
class SchedRandomizationConfig: #sched_rand
    # Central bounds dictionary: each target key maps to a (min, max) tuple
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "sigma_min": (0.001, 0.2),
        "sigma_max": (25.0, 60.0),
        "rho": (3.0, 8.0),
        "start_blend": (0.04, 0.11),
        "end_blend": (0.4, 0.6),
        "sharpness": (0.75, 0.95),
        "initial_step_size": (0.7, 1.0),
        "final_step_size": (0.1, 0.3),
        "step_size_factor": (0.65, 0.85),
        "initial_noise_scale": (1.0, 1.5),
        "final_noise_scale": (0.6, 1.0),
        "noise_scale_factor": (0.75, 0.95),
        "early_stopping_threshold": (0.001, 0.02),
        "smooth_blend_factor": (6.0, 11.0),
        "sigma_scale_factor": (700.0, 1100.0),
    })
    
    settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
    def update_settings(self, exclude: str = "settings"):
        self.settings = {
            k: v for k, v in asdict(self).items()
            if k != exclude
        }

        
    # Automatically computed default values (center of each range)
    @property
    def defaults(self) -> Dict[str, float]:
        return {
            k: round((v[0] + v[1]) / 2, 6)
            for k, v in self._bounds.items()
        }

    # Apply randomized values within bounds to target
    def apply_randomized_values(self, target: Any, mode: str = "uniform", percent_stretch: float = 0.0):
        for k, (min_v, max_v) in self._bounds.items():
            # Optional: apply stretch beyond min/max if needed
            stretch = (max_v - min_v) * percent_stretch
            min_v -= stretch
            max_v += stretch

            if mode == "uniform":
                value = round(random.uniform(min_v, max_v), 6)
            elif mode == "normal":
                mu = (min_v + max_v) / 2
                sigma = (max_v - min_v) / 4  # ~95% coverage
                value = round(random.gauss(mu, sigma), 6)
                value = max(min_v, min(max_v, value))
            else:
                raise ValueError(f"Unknown randomization mode: {mode}")

            # Apply to dict or object
            if isinstance(target, dict):
                target[k] = value
            else:
                setattr(target, k, value)

    def print_bounds(self):
        print("=== Parameter Bounds ===")
        for k, (lo, hi) in self._bounds.items():
            print(f"{k}: [{lo}, {hi}]")

    def randomize_within_bounds(self, bounds: dict[str, tuple[float, float]]):
        """
        Applies random values to fields based on given bounds dictionary.
        """
        for key, (low, high) in bounds.items():
            if hasattr(self, key):
                setattr(self, key, random.uniform(low, high))
    def apply_default_bounds(self):
        if hasattr(self, "_bounds") and isinstance(self._bounds, dict):
            self.randomize_within_bounds(self._bounds)


@dataclass
class AppendFields: #app_field
    _rand: Optional[bool] = False # this mode is different from _enable_randomization_type
    _randomization_type: Optional[str] = "asymmetric"
    _randomization_type_options: List[str] = field(default_factory=lambda: ["asymmetric", "symmetric", "logarithmic", "exponential"])
    _randomization_percent: Optional[float] = 0.2
    _enabled_randomization_type: Optional[bool] = False #this randomizaiton pertains to the randomization_type options, and is only active if _rand is true
    
   
@dataclass
class AppendList: #app_list
    _base_targets: List[str] = field(default_factory=lambda: [
        "sigma_min", "sigma_max", "smooth_blend_factor", "rho", "start_blend",
        "end_blend", "sharpness", "initial_step_size", "final_step_size",
        "step_size_factor", "initial_noise_scale", "final_noise_scale",
        "noise_scale_factor", "sigma_scale_factor"
    ])

    # Main dictionary: each base param has its own AppendFields object
    randomization_fields: Dict[str, AppendFields] = field(init=False)

    def __post_init__(self):
        self.randomization_fields = {
            key: AppendFields()
            for key in self._base_targets
        }
    

    def get(self, key: str) -> AppendFields:
        return self.randomization_fields[key]

    def set_value(self, base_key: str, attr: str, value: Any):
        if base_key in self.randomization_fields:
            setattr(self.randomization_fields[base_key], attr, value)

    def build_expanded_keys(self) -> List[str]:
        """Returns a list of all field access paths (for UI or flat config use)."""
        sample = AppendFields()
        keys = []
        for base in self._base_targets:
            for suffix in sample.__dataclass_fields__.keys():
                keys.append(f"{base}_{suffix.lstrip('_')}")
        return keys

    def to_nested_dict(self) -> Dict[str, Dict[str, Any]]:
        """Serialize all fields per base target."""
        return {
            base: self.randomization_fields[base].to_dict()
            for base in self._base_targets
        }

    
@dataclass
class EarlyStopping: #earlystop
    enabled: Optional[bool]= True
    method: Optional[str] = "mean"
    threshold: Optional[float] =  0.01
    safety_minimum_stop_step: Optional[int] = 10
    recent_change_convergence_delta: Optional[float] = 0.02
    
    # Full resolved settings (optional debug)
    settings: dict = field(default_factory=dict)
   
    def to_dict(self) -> dict:
        return asdict(self)
    def update_settings(self, exclude: str = "settings"):
        self.settings = {
            k: v for k, v in asdict(self).items()
            if k != exclude
        }


@dataclass
class Sharpening:   #sharp

    enabled: Optional[bool]= True
    sharpen_mode: Optional[str] = "both"
    sharpen_last_n_steps: Optional[int] = 10
    sharpen_variance_threshold: Optional[float] = 0.01
    sharpness: Optional[float] = 0.8
    
    # Full resolved settings (optional debug)
    settings: dict = field(default_factory=dict)
  
    def to_dict(self) -> dict:
        return asdict(self)
    def update_settings(self, exclude: str = "settings"):
        self.settings = {
            k: v for k, v in asdict(self).items()
            if k != exclude
        }


@dataclass
class Cache:    #cache
    save_sigma_cache: Optional[bool]= True
    load_sigma_cache: Optional[bool]= True
    save_prepass_sigmas: Optional[bool]= True
    load_prepass_sigmas: Optional[bool]= True
    sigma_save_subfolder: Optional[str] = "saved_sigmas"
    
    # Full resolved settings (optional debug)
    settings: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)
    def update_settings(self, exclude: str = "settings"):
        self.settings = {
            k: v for k, v in asdict(self).items()
            if k != exclude
        }

 
@dataclass
class Logging: #log
    debug: Optional[bool]= False
    log_save_directory: Optional[str] = "modules/sd_simple_kes/image_generation_data" #should be saved as the absolute path
    graph_save_directory: Optional[str] = "modules/sd_simple_kes/image_generation_data"    #should be saved as the absolute path
    graph_save_enable: Optional[bool]= False
    
    # Full resolved settings (optional debug)
    settings: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)
    def update_settings(self, exclude: str = "settings"):
        self.settings = {
            k: v for k, v in asdict(self).items()
            if k != exclude
        }


@dataclass
class AutoMode: #auto

    auto_tail_smoothing: Optional[bool]= False
    allow_step_expansion: Optional[bool]= False
    auto_stabilization_sequence: Optional[List[str]] = field(default_factory=lambda: [
    "smooth_interpolation", "append_tail", "blend_tail", "apply_decay", "progressive_decay"
    ])  
    auto_tail_threshold: Optional[float]= 0.1
    jaggedness_threshold: Optional[float]=0.05
    
    # Full resolved settings (optional debug)
    settings: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)
 
@dataclass
class BlendMethod: #blendmeth
    weight: Optional[float] = 1.0
    decay_pattern: Optional[List[str]] = field(default_factory=lambda: ["zero", "geometric", "harmonic", "logarithmic",
    "extrapolate", "fractional", "exponential", "linear"
    ])
    decay_mode: Optional[List[str]] = field(default_factory=lambda:   ["blend", "append", "replace"])
    tail_steps: Optional[int] = 1
    
    # Full resolved settings (optional debug)
    settings: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)
        
    def update_settings(self, exclude: str = "settings"):
        self.settings = {
            k: v for k, v in asdict(self).items()
            if k != exclude
        }

    
     # === Helpers ===

    def set_weight(self, weight: float, value: float):
        self.weight = max(0.0, min(1.0, value))  # Clamp to 0–1

    def set_decay_mode(self, mode: str):
        if mode in self.decay_mode:
            self.settings["selected_decay_mode"] = mode
        else:
            raise ValueError(f"Invalid decay_mode: {mode}. Allowed: {self.decay_mode}")

    def set_decay_pattern(self, pattern: str):
        if pattern in self.decay_pattern:
            self.settings["selected_decay_pattern"] = pattern
        else:
            raise ValueError(f"Invalid decay_pattern: {pattern}. Allowed: {self.decay_pattern}")

    def set_tail_steps(self, value: int):
        self.tail_steps = max(1, value)  # Avoid 0

    def summary(self) -> str:
        selected_mode = self.settings.get("selected_decay_mode", "blend")
        selected_pattern = self.settings.get("selected_decay_pattern", "harmonic")
        return (f"Weight={self.weight}, "
                f"Mode={selected_mode}, "
                f"Pattern={selected_pattern}, "
                f"TailSteps={self.tail_steps}")
@dataclass
class BlendMethods: #bm
    methods: Dict[str, BlendMethod] = field(default_factory=lambda: {
        "euler": BlendMethod(weight=0.3, decay_pattern="harmonic", decay_mode="blend", tail_steps=1),
        "euler_advanced": BlendMethod(weight=0.7, decay_pattern="harmonic", decay_mode="blend", tail_steps=1),
        "geometric": BlendMethod(weight=0.5, decay_pattern="linear", decay_mode="blend", tail_steps=1),
        "harmonic": BlendMethod(weight=0.6, decay_pattern="logarithmic", decay_mode="blend", tail_steps=1),
        "logarithmic": BlendMethod(weight=0.4, decay_pattern="fractional", decay_mode="blend", tail_steps=1),
        "karras": BlendMethod(weight=0.8, decay_pattern="exponential", decay_mode="blend", tail_steps=1),
        "exponential": BlendMethod(weight=0.5, decay_pattern="geometric", decay_mode="blend", tail_steps=1),
    })
    
    # Full resolved settings (optional debug)
    settings: dict = field(default_factory=dict)
   
    # === Helpers ===
    
    def to_dict(self) -> dict:
        return asdict(self)
    def update_settings(self, exclude: str = "settings"):
        self.settings = {
            k: v for k, v in asdict(self).items()
            if k != exclude
        }


    def get_method(self, name: str) -> Optional[BlendMethod]:
        return self.methods.get(name)

    def update_method(self, name: str, **kwargs):
        """
        Example: update_method("euler", weight=0.6, tail_steps=3)
        """
        method = self.get_method(name)
        if not method:
            raise ValueError(f"Blend method '{name}' not found.")

        for k, v in kwargs.items():
            if hasattr(method, k):
                setattr(method, k, v)

    def set_global_decay_mode(self, mode: str):
        for method in self.methods.values():
            method.set_decay_mode(mode)

    def set_global_pattern(self, pattern: str):
        for method in self.methods.values():
            method.set_decay_pattern(pattern)

    def set_global_weight(self, value: float):
        for method in self.methods.values():
            method.set_weight(value)

    def summary(self):
        print("=== Blend Methods Summary ===")
        for name, method in self.methods.items():
            print(f"{name}: {method.summary()}")

    def get_enabled_patterns(self) -> List[str]:
        patterns = set()
        for m in self.methods.values():
            selected = m.settings.get("selected_decay_pattern")
            if selected:
                patterns.add(selected)
        return sorted(patterns)
        
    def resolve_blending_order(
        self,
        by: str = "weight",
        descending: bool = True,
        min_weight: float = 0.0,
        preferred_pattern: Optional[str] = None,
        preferred_mode: Optional[str] = None
    ) -> List[Tuple[str, BlendMethod]]:
        """
        Returns sorted list of (name, BlendMethod) pairs by:
        - weight (default)
        - tail_steps
        - presence of preferred decay_pattern or decay_mode

        You can also use:
          resolve_blending_order(by="pattern:harmonic")
          resolve_blending_order(by="mode:append")
          
        # Sort by weight, but only include those using harmonic pattern
        bm.resolve_blending_order(by="pattern:harmonic")

        # Sort by weight, filter by decay_mode=append
        bm.resolve_blending_order(by="mode:append")

        # Standard: sort by tail steps
        bm.resolve_blending_order(by="tail_steps", descending=False)
        
        """
        valid_keys = {"weight", "tail_steps", "decay_pattern", "decay_mode"}
        filter_pattern = None
        filter_mode = None

        # Detect filter mode
        if by.startswith("pattern:"):
            filter_pattern = by.split(":", 1)[1]
            by = "weight"
        elif by.startswith("mode:"):
            filter_mode = by.split(":", 1)[1]
            by = "weight"
        elif by not in valid_keys:
            raise ValueError(f"[BlendMethods] Unsupported sort key: {by}")

        # Filter methods by weight and optionally pattern/mode
        filtered = []
        for name, method in self.methods.items():
            if method.weight < min_weight:
                continue
            if filter_pattern and filter_pattern not in method.decay_pattern:
                continue
            if filter_mode and filter_mode not in method.decay_mode:
                continue
            filtered.append((name, method))

        return sorted(
            filtered,
            key=lambda item: getattr(item[1], by, 0),
            reverse=descending
        )
    def filter_by_pattern(self, pattern: str) -> Dict[str, BlendMethod]:
        return {
            name: bm for name, bm in self.methods.items()
            if pattern in bm.decay_pattern
        }


 
@dataclass
class Descriptions: #desc
    
    _blending_mode_options_descriptions: List[Dict[str, str]] = field(default_factory=lambda: [{
        "auto": "uses smart weights if more than 2 methods",
        "default": "Legacy blending mode between Karras and Exponential using smooth_blend",
        "smooth_blend": "enforces use even when weights are used to mix 2 methods",
        "weights": "enforce using weighting method even with only 2 methods"
    }])
    skip_prepass_description: str = "has no change to image quality - not currently functioning as intended for early stop purposes" 
    
    _experimental: Dict[str, str] = field(default_factory=lambda: {
    "early_stopping_threshold": "is not implemented",
    "prepass": "does not work as intended"
    })

   
   # Full resolved settings (optional debug)
    settings: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def update_settings(self, exclude: str = "settings"):
        self.settings = {
            k: v for k, v in asdict(self).items()
            if k != exclude
        }


@dataclass
class SigmaData: #sd
    
    sigma: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    sigmas: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    
    def add_schedule(self, new_sigma: torch.Tensor):
        self.sigmas.append(new_sigma)

    def get_primary_schedule(self) -> Optional[torch.Tensor]:
        return self.sigma or (self.sigmas[0] if self.sigmas else None)

    def get_all_schedules(self) -> List[torch.Tensor]:
        return self.sigmas

    def summary(self):
        print(f"[SigmaData] Stored {len(self.sigmas)} schedules.")
        if self.sigma is not None:
            print(f"Primary sigma: shape {tuple(self.sigma.shape)}")
        for i, s in enumerate(self.sigmas):
            print(f"  → sigmas[{i}]: shape {tuple(s.shape)}")
            
    
    # Full resolved settings (optional debug)
    settings: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)
    def update_settings(self, exclude: str = "settings"):
        self.settings = {
            k: v for k, v in asdict(self).items()
            if k != exclude
        }


@dataclass
class DeveloperSettings: #dev
# new - add this as an option
    
    sigma_min: Optional[float] = 0.13757067353874633,
    sigma_max: Optional[float] = 47.95768510805332,
    rho: Optional[float] = 7.959565031107985,
    end_blend: Optional[float] = 0.4
    start_blend: Optional[float] = 0.05
    sharpness: Optional[float] = 0.85 
    initial_step_size: Optional[float] = 0.9
    final_step_size: Optional[float] = 0.20
    step_size_factor: Optional[float] = 0.80814932869181
    initial_noise_scale: Optional[float] = 1.25
    final_noise_scale: Optional[float] = 0.80
    noise_scale_factor: Optional[float] = 0.8113992828873163
    early_stopping_threshold: Optional[float] = 0.06
    # ====
    auto_tail_threshold: Optional[float] = 0.05
    jaggedness_threshold: Optional[float] = 0.01
    allow_step_expansion: Optional[bool] = False  
    apply_tail_steps: Optional[bool] = False         
    apply_decay_tail: Optional[bool] = False        
    apply_blended_tail: Optional[bool] = False       
    apply_progressive_decay: Optional[bool] = False  
    auto_tail_smoothing: Optional[bool] = True
    decay_pattern: Optional[str] = "extrapolate"   
    decay_mode: Optional[str] = "append" 
    blending_style: Optional[str] = "softmax"
    blending_mode: Optional[str] = "default"
    sigma_variance_scale: Optional[float] = 0.1  
    safety_minimum_stop_step: Optional[int] = 10 
    recent_change_convergence_delta: Optional[float] = 0.6 
    sharpen_variance_threshold: Optional[float] = 0.01
    tail_steps: Optional[int] = 1
    blend_midpoint: Optional[float] =0.5 
    smooth_blend_factor: Optional[float] = 5 
    skip_prepass: Optional[bool] = True        
    sigma_auto_enabled: Optional[bool]= True
    blend_methods: BlendMethods = field(default_factory=BlendMethods)
    
    # Full resolved settings (optional debug)
    settings: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)
    def update_settings(self, exclude: str = "settings"):
        self.settings = {
            k: v for k, v in asdict(self).items()
            if k != exclude
        }

        
    def apply_to(self, target: Any, overwrite: bool = True, include_none: bool = False):
        """
        Apply all settings to a target object or dictionary.

        Args:
            target: a dict or an object (like `state.p`, `state.conditioning`, etc.)
            overwrite: whether to overwrite existing non-None values
            include_none: whether to apply None values
        """
        settings_dict = self.to_dict()
        for key, value in settings_dict.items():
            if not include_none and value is None:
                continue
            if isinstance(target, dict):
                if overwrite or key not in target:
                    target[key] = value
            else:
                if overwrite or not hasattr(target, key) or getattr(target, key) is None:
                    setattr(target, key, value)
                    
@dataclass
class SchedulerState: #scheduler_state
    
    
    def_range: DefaultRangeValue = field(default_factory=DefaultRangeValue)
    app_field: AppendFields = field(default_factory=AppendFields)
    app_list: AppendList = field(default_factory=AppendList)
    sched_rand: SchedRandomizationConfig = field(default_factory=SchedRandomizationConfig)
    earlystop: EarlyStopping = field(default_factory=EarlyStopping)
    sharp: Sharpening = field(default_factory=Sharpening)
    cache: Cache = field(default_factory=Cache)
    log: Logging = field(default_factory=Logging)
    auto: AutoMode = field(default_factory=AutoMode)
    desc: Descriptions = field(default_factory=Descriptions)
    blendmeth: BlendMethod = field(default_factory=BlendMethod)
    bm: BlendMethods = field(default_factory=BlendMethods)
    msc: MiscVariables = field(default_factory=MiscVariables)
    lsu: LastSettingsUsed = field(default_factory=LastSettingsUsed)
    dev: DeveloperSettings = field(default_factory=DeveloperSettings)
    sd: SigmaData = field(default_factory=SigmaData)
    
    # Full resolved settings (optional debug)
    settings: dict = field(default_factory=dict)
        
    def to_dict(self) -> dict:
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
    
    def apply_to_state_subsection(self, state_now, section: str, data: dict, overwrite: bool = True):
        """
        Applies a dictionary of key/values to a subsection of `state_now`.

        Args:
            state_now: The current state object.
            section (str): The attribute of state_now to target (e.g., "p", "dev", "aliases").
            data (dict): The settings to apply.
            overwrite (bool): Whether to overwrite existing values (default True).
        """
        if not state_now or not hasattr(state_now, section):
            raise ValueError(f"[Warning] state is missing or has no '{section}' attribute.")

        target = getattr(state_now, section)
        if not hasattr(target, "__dict__"):
            raise TypeError(f"[Warning] Target section '{section}' is not an object with assignable attributes.")

        for k, v in data.items():
            if overwrite or not hasattr(target, k):
                setattr(target, k, v)