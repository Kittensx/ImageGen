from dataclasses import fields, field, dataclass, asdict
from typing import Optional, Dict, Any, List, Union, Callable, Tuple
import torch
import torch.nn as nn
from pathlib import Path
import os 
    

@dataclass
class ParamAliases: #aliases
    _aliases: Dict[str, str] = field(default_factory=lambda: {
        "steps": ["S", "n"],
        "batch_size": "batch_size",
        "cfg_scale": ["unconditional_guidance_scale", "cfg"],
        "shape": "shape",
        "eta": "eta",
        "verbose": "verbose",        
        "uncond": ["unconditional_conditioning", "uncond"],
        "cond": ["conditional_conditioning", "conditioning", "cond"],
        "sigmas": ["sigmas", "sigs"],
        "timesteps": ["timesteps", "t"],
        
    })
   
    # Singleton accessor
    _instance = None
    
    _valid_internal_names = {
        "steps",
        "cfg_scale",
        "seed",
        "positive_prompt",
        "negative_prompt",                
        "sigma_min",
        "sigma_max",               
        "debug",
        "verbose",    
    }
    
    # Full resolved settings (optional debug)
    settings: dict = field(default_factory=dict)
    
    def register_safe(self, internal_name: str, new_alias: str):
        """Registers alias with override=True to suppress KeyError."""
        self.register(internal_name, new_alias, override=True)

    def register(self, internal_name: str, new_alias: str, override: bool = False):
        """
        Add or override an alias for an internal parameter name.
        If override=True, allows registering unknown names without raising an error.
        """
       
        if internal_name not in self._aliases and override==True:
            self._aliases[internal_name] = new_alias
        if internal_name not in self._valid_internal_names and not override:
            raise KeyError(f"[ParamAliases] '{internal_name}' is not a known internal parameter name.")
        self._aliases[new_alias] = internal_name
        
        self._aliases[internal_name] = new_alias

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)

    def __getitem__(self, key: str) -> str:
        return self._aliases.get(key, key)

    def __contains__(self, key: str) -> bool:
        return key in self._aliases
    
    def get_target_key(self, internal_name: str) -> str | None:
        """Return the alias (target key) for a given internal parameter name, if it exists."""
        alias = self._aliases.get(internal_name)
        if isinstance(alias, list):
            return alias[0]  # use first as canonical
        return alias

    def get_all_aliases_for(self, canonical_key: str) -> list[str]:
        """Return all alias keys that resolve to a given canonical key (reverse lookup)."""
        return [
            alias_key for alias_key, target in self._aliases.items()
            if (target == canonical_key) or
               (isinstance(target, list) and canonical_key in target)
        ]
    def allow_custom_keys(self, keys: list[str]):
        for key in keys:
            self._valid_internal_names.add(key)
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    

@dataclass
class ModelConfig:  # cfgm
    # === Core Root Structure ===
    root_dir: str = None                     # Root project path (e.g., base_dir)
    program_dir: str = "image_gen"           # Main folder (default: image_gen)
    model_folder_root: str = "models"        # Where all model types live (e.g., models/stable_diffusion)
    
    # === Model Subfolders & Identity ===
    model_folder: str = "stable_diffusion"   # Subdir under models (e.g., models/stable_diffusion)
    model_category: str = "stable_diffusion" # Used to organize models by type
    model_name: str = "default_model"        # File name (e.g., animore.safetensors)

    # === Configuration & Architecture ===
    config_folder: str = "_model_configs"    # Folder holding YAML config files
    config_name: Optional[str] = None        # Optional override (e.g., v1-inference.yaml)
    model_arch: Optional[str] = None         # Auto-detected: "v1", "sdxl", etc

    # === Optional Runtime State ===
    model: Optional[nn.Module] = None

    # === Computed Absolute Paths ===
    config_path: Optional[str] = None
    sched_folder_path: Optional[str] = None
    scheduler_folder: str = "schedulers"

    samp_folder_path: Optional[str] = None
    sampler_folder: str = "samplers"

    model_dir_path: Optional[str] = None  # Absolute path to selected model category folder

    # === Multi-Model Support ===
    additional_model_dirs: List[str] = field(default_factory=list)
    _model_path: Optional[str] = field(default=None, repr=False)

    model_folders: List[str] = field(default_factory=list)
    _model_folders: Optional[str] = field(default=None, repr=False)

    # === LoRA/VAE ===
    vae_folder: str = "vae"
    vae_folder_path: Optional[str] = None

    lora_folder: str = "lora"
    lora_folder_path: Optional[str] = None

    tokenizer_path: Optional[str] = None  # Not yet wired in
    
    # Full resolved settings (optional debug)
    settings: dict = field(default_factory=dict)

    def __post_init__(self):
        self.model_folders = [os.path.abspath(os.path.normpath(p)) for p in self.model_folders]
        if self._model_folders:
            self._model_folders = os.path.abspath(os.path.normpath(self._model_folders))
        if self.root_dir:
            self._resolve_all_paths()

    def _resolve_all_paths(self):
        root = os.path.abspath(os.path.normpath(self.root_dir))
        image_gen_root = os.path.join(root, self.program_dir)

        self.model_dir_path = os.path.join(image_gen_root, self.model_folder_root, self.model_folder)
        self.config_path = os.path.join(image_gen_root, self.model_folder_root, self.config_folder) #_model_configs
        self.sched_folder_path = os.path.join(image_gen_root, self.scheduler_folder)
        self.samp_folder_path = os.path.join(image_gen_root, self.sampler_folder)
        self.vae_folder_path = os.path.join(image_gen_root, self.model_folder_root, self.vae_folder)
        self.lora_folder_path = os.path.join(image_gen_root, self.model_folder_root, self.lora_folder)

        # Normalize all
        for key in ['model_dir_path', 'config_path', 'sched_folder_path', 'samp_folder_path', 'vae_folder_path', 'lora_folder_path']:
            val = getattr(self, key)
            setattr(self, key, os.path.abspath(os.path.normpath(val)))

    @property
    def model_path(self) -> str:
        if self._model_path:
            return self._model_path
        elif self.additional_model_dirs:
            return os.path.abspath(self.additional_model_dirs[0])
        else:
            return os.path.abspath(os.path.join(self.model_dir_path, self.model_name))

    @model_path.setter
    def model_path(self, value: str):
        self._model_path = os.path.abspath(os.path.normpath(value))

         

    
    @property
    def model_folders_path(self) -> List[str]:
        if self._model_folders:
            return [os.path.abspath(self._model_folders)]
        elif self.model_folders:
            return [os.path.abspath(p) for p in self.model_folders]
        else:
            return [os.path.abspath(self.model_folder_root)]

    @model_folders_path.setter
    def model_folders_path(self, value: str | List[str]):
        if isinstance(value, str):
            self._model_folders = os.path.abspath(os.path.normpath(value))
        elif isinstance(value, list):
            self.model_folders = [os.path.abspath(os.path.normpath(p)) for p in value]
            self._model_folders = None
        else:
            raise TypeError(f"Expected str or list[str] for model_folders_path, got {type(value)}")

    def to_dict(self):
        return asdict(self)


@dataclass
class MemoryOptimizations: #memop
    enable_tf32: Optional[bool] = False
    enable_xformers: Optional[bool] = False
    clear_cuda_cache: Optional[bool] = False
    set_alloc_conf: Optional[bool] = False
    limit_prompt_tokens: Optional[bool] = False
    
    # Full resolved settings (optional debug)
    settings: dict = field(default_factory=dict)
    
    def to_dict(self):
        return asdict(self)


@dataclass
class ImageState: #image
    preferred_output_format: Optional[str] = "png" #user config
    txt2img_folder: Optional[str] = None #user config #need to make sure or delay image output creation until after we load the config
    output_folder: Optional[str] = None #user config    #need to make sure or delay image output creation until after we load the config
    #### 
    bit_depth: int = 8
    mode: str = "auto"
    prefix: str = "img_"     #"name" prefix, date prefix, number prefix (needs implementing)
    generated_images: List[str] = field(default_factory=list)   
    include_alpha: bool = False
    normalize_float: bool = True 
    image_np: any = None        # numpy processed image (raw data)
    image_path: str = None      #stored image path when generating images
    output_format: str = "png"   #method to save according to a user config value (needs implementing)
    txt2image_path: str = None   #root/image/output/txt2image
    output_path: str = None  #root/image_output
    #compression_quality: Optional[int] = 90 #converted to decimal percentage #TODO
    
    # Full resolved settings (optional debug)
    settings: dict = field(default_factory=dict)
    
    def to_dict(self):
        return asdict(self)


        
  
@dataclass
class SchedulerParameters: #sched
    scheduler_settings: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    sigmas: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    timesteps: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    scheduler_fn: Optional[Callable] = None
    selected_scheduler_name: Optional[str] = None
    scheduler_name: Optional[str] = None
    scheduler_label: Optional[str] = None    
    scheduler_description: Optional[str] = None
    requested_steps: Optional[int] = None
    effective_steps: Optional[int] = None
    schedule_extra: Dict[str, Any] = field(default_factory=dict)
    compatibility_mode: Optional[str] = None
    
    # Full resolved settings (optional debug)
    settings: dict = field(default_factory=dict)
    
    def to_dict(self):
        return asdict(self)
 
@dataclass
class SampleRecord: #samprecord
    tensor:Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    device: Optional[str] = None
    samples_device: Optional[str] = None
    toggle_history: Optional[bool] = False
    samples_history: Dict[str, List[Dict[str, Union[torch.Tensor, str]]]] = field(default_factory=dict)
    max_keep: Optional[int] = 1
    min_keep: Optional[int] = 1
    offload_target: str = "cpu"  # Options: "cpu", "cuda:0", "cuda:1", "auto"
    enable_offload: bool = True
    samples: Optional[torch.Tensor] = field(default=None, repr=False)
    decoded_tensor: Optional[torch.Tensor] = None
    decoded_history: Dict[str, List[torch.Tensor]] = field(default_factory=dict)
    
    # Full resolved settings (optional debug)
    settings: dict = field(default_factory=dict)
    sampler_state: Optional[Any] = None
    
    def to_dict(self):
        return asdict(self)
        
@dataclass
class SamplerParameters: #samp
    validated_for_sampler: bool = False
    sampler_settings: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    sigmas: Optional[torch.Tensor] = field(default=None, init=False, repr=False)    
    sampler_fn: Optional[Callable] = None
    sampler_name: Optional[str] = None    
    selected_sampler_name: Optional[str] = None
    sampler_label: Optional[str] = None
    sampler_description: Optional[str] = None
    samples: Optional[torch.Tensor] = field(default=None, repr=False)
    requested_steps: Optional[int] = None
    effective_steps: Optional[int] = None
    schedule_is_compatible: Optional[bool] = None
    compatibility_mode: Optional[str] = None
    
    
    # Full resolved settings (optional debug)
    settings: dict = field(default_factory=dict)
    
    def to_dict(self):
        return asdict(self)

@dataclass
class DeviceParameters: #d
    device: Optional[torch.device] = None
    device_name: Optional[str] = None
    device_type: Optional[str] = None
    device_index: Optional[int] = None
    mode: Optional[str] = None
    strategy: Optional[str] = None
    user_device_type: Optional[str] = None
    model: Optional[nn.Module] = None #see cfgm for model
    
    # Full resolved settings (optional debug)
    settings: dict = field(default_factory=dict)
    
    def to_dict(self):
        return asdict(self)

@dataclass
class GenParameters:  # p
    
    seed: Optional[int] = None
    last_seed: Optional[int] = None
    use_seed_sequence: Optional[bool] = True 
    cfg_scale: Optional[float] = 6
    steps: Optional[int] = 20
    batch_size: Optional[int] = 1
    width: Optional[int] = 512
    height: Optional[int] = 640
    positive_prompt: Optional[str] = ""
    negative_prompt: Optional[str] = ""
    hires_steps: Optional[int]=None #need to swap all hires to new dataclass 
    hires_pos: Optional[str] = None #new
    hires_neg: Optional[str] = None #new
    hires_width: Optional[int] = None #new
    hires_height: Optional[int] = None #new
    
    
    # Full resolved settings (optional debug)
    settings: dict = field(default_factory=dict)
    
    def to_dict(self):
        return asdict(self)
   
@dataclass
class ConditioningParameters: #conditioning
    shape: Optional[List[int]] = None  
    cond: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    uncond: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    prompt_schedules: Dict[str, List[str]] = field(default_factory=lambda: {"positive": [], "negative": []}, repr=False)
    
    pos_schedule: Optional[Dict[str, List[Tuple[int, str]]]] = field(
        default_factory=lambda: {"positive": [], "negative": []},
        repr=False
    )
    neg_schedule: Optional[Dict[str, List[Tuple[int, str]]]] = field(
        default_factory=lambda: {"positive": [], "negative": []},
        repr=False
    )
    use_old_scheduling: Optional[bool] = False
    
    
    # Full resolved settings (optional debug)
    settings: dict = field(default_factory=dict)
    
    def to_dict(self):
        return asdict(self)

 
@dataclass
class UserConfig:    #usercfg
    seed: int = -1
    cfg_scale: float = 6.0
    steps: int = 20
    height: int = 512
    width: int = 512
    batch_size:int = 1
    use_seed_sequence: Optional[bool] = True
    positive_prompt:str = ""
    negative_prompt:str = ""
    
    # Full resolved settings (optional debug)
    settings: dict = field(default_factory=dict)
    
    def to_dict(self):
        return asdict(self)
        
@dataclass
class CombinedConfig:   #copied at config 
    seed: int = None
    cfg_scale: float = 6
    steps: int = 25
    height: int = 512
    width: int = 512
    batch_size:int = 1
    use_seed_sequence: Optional[bool] = False
    positive_prompt:str = ""
    negative_prompt:str = ""
    
    # Full resolved settings (optional debug)
    settings: dict = field(default_factory=dict)
    
    def to_dict(self):
        return asdict(self)
@dataclass
class HiresUserConfig:    #hires_user
    hires_seed: int = -1
    hires_cfg_scale: float = 7.0
    hires_steps: int = 25
    hires_height: int = 512
    hires_width: int = 512
    hires_batch_size:int = 1
    hires_use_seed_sequence: Optional[bool] = False
    hires_positive_prompt:str = ""
    hires_negative_prompt:str = ""
    hires_size_mode: str = "same_as_base"
    hires_scale: float = 2.0
    
    
    # Full resolved settings (optional debug)
    settings: dict = field(default_factory=dict)
    
    def to_dict(self):
        return asdict(self)
        
    
    
@dataclass
class SharedState:
    """
    SharedState holds core runtime state values and allows flexible storage of additional data
    through an `extra` dictionary.

    Fixed fields include commonly used generation parameters like `seed`, `cfg_scale`, etc.
    The `extra` dictionary is used to store values dynamically at runtime, such as
    user-defined settings, tool data, or plugin results.
    """
    memop: MemoryOptimizations = field(default_factory=MemoryOptimizations)
    image: ImageState = field(default_factory=ImageState)
    p: GenParameters = field(default_factory=GenParameters)    
    d: DeviceParameters = field(default_factory=DeviceParameters)    
    conditioning: ConditioningParameters = field(default_factory=ConditioningParameters)    
    sched: SchedulerParameters = field(default_factory=SchedulerParameters)
    samp: SamplerParameters = field(default_factory=SamplerParameters)
    samprecord: SampleRecord = field(default_factory=SampleRecord)
    aliases: ParamAliases = field(default_factory=ParamAliases)
    cfgm: ModelConfig = field(default_factory=ModelConfig) #also contains model(nn.module)
    usercfg: UserConfig = field(default_factory=UserConfig) #TODO - ensure that user positive prompts get updated with config values
    cc: CombinedConfig = field(default_factory=CombinedConfig)
    #h: HiresGenParameters = field(default_factory=HiresGenParameters) #needs implementing
    #hires_cond: HiresConditioning = field(default_factory=HiresConditioning) #needs implementing
    #hires_sched: HiresScheduler = field(default_factory=HiresScheduler) #needs implementing
    #hires_samp: HiresSampler = field(default_factory=HiresSampler) #needs implementing    
    #hires_user: HiresUserConfig = field(default_factory=HiresUserConfigs) #needs implementing
    #hires_device: HiresDeviceParameters = field(default_factory=HiresDeviceParameters) #needs implementing
    #hires_image: Hires_ImageState = field(default_factory=Hires_ImageState) #needs implementing
    #hires_memop: Hires_MemoryOptimizations = field(default_factory=Hires_MemoryOptimizations) #needs implementing
   
    
    root_dir: Optional[Path] = None
    base_dir: Optional[Path] = None
      
    config_index_path: Optional[Path] = None    #index.yaml
    default_config_dict: Dict[str, str] = field(default_factory=lambda: {
        "yaml": "default_config.yaml",
        "json": "default_config.json"
    })
    user_config_dict: Dict[str, str] = field(default_factory=lambda: {
        "yaml": "user_config.yaml",
        "json": "user_config.json"
    })   
    # Actual parsed config values after loading
    default_config: Dict[str, Any] = field(default_factory=dict) #actual loaded dicts
    user_config: Dict[str, Any] = field(default_factory=dict) #actual loaded dicts
    injections_override: Optional[Dict[str, Any]] = field(default_factory=dict) #actual loaded dicts / scripts/ api calls
    extra: Dict[str, Any] = field(default_factory=dict)    # command line extra commands
    #combined_config: Dict[str, Any] = field(default_factory=dict) #actual loaded dicts = combining all loaded dicts default + user + injections_override + extra
    
    
    
    user_config_yaml_file: Optional[str] = "user_config.yaml"
    user_config_json_file: Optional[str] = "user_config.json"
    default_config_yaml_file: Optional[str] = "default_config.yaml"
    default_config_json_file: Optional[str] = "default_config.json"
    
    user_config_path_yaml: Optional[Path] = None 
    user_config_path_json: Optional[Path] = None
    default_config_path_yaml: Optional[Path] = None
    default_config_path_json: Optional[Path] = None
    userconfiguration_folder: Optional[Path] = None
    userconfiguration_path: Optional[Path] = None
    
    # Full resolved settings (optional debug)
    settings: dict = field(default_factory=dict)
    
    
    @property
    def effective_config(self) -> Dict[str, Any]:
        """Returns final config with all overrides applied."""
        return {**self.default_config, **self.user_config, **self.injections_override, **self.extra}

   
    def to_dict(self) -> Dict[str, Any]:
        """
        Return the complete dataclass as a dictionary (including extras).
        """
        return asdict(self)
   
    def add_image(self, filename: str) -> None:
        """
        Append an image filename or identifier to the `generated_images` list.

        Example:
            state.add_image("img_001.png")
        """
        self.generated_images.append(filename)

    def set_extra(self, key: str, value: Any) -> None: #need to update this for the different categories we use for different dataclasses
        """
        Add or update a custom key-value pair in the `extra` dictionary.

        Example:
            state.set_extra("sampler_name", "Euler")
        """
        self.extra[key] = value

    def get_extra(self, key: str, default: Any = None) -> Any: #need to update this for the different categories we use for different dataclasses
        """
        Retrieve a value from the `extra` dictionary.

        Example:
            sampler = state.get_extra("sampler_name")
        """
        return self.extra.get(key, default)

    def has_extra(self, key: str) -> bool: #need to update this for the different categories we use for different dataclasses
        """
        Check if a key exists in the `extra` dictionary.

        Example:
            if state.has_extra("scheduler_config"):
                ...
        """
        return key in self.extra
    
    def to_generation_params(self) -> "GenParameters":
   
        # Get the field names that GenParameters accepts
        target_fields = {f.name for f in fields(GenParameters)}

        # Use vars(self) to get actual SharedState values
        source_dict = vars(self)

        # Match only keys that are in GenParameters and not None
        matched = {k: source_dict[k] for k in target_fields if k in source_dict and source_dict[k] is not None}

        return GenParameters(**asdict(self.p))
        
    def update_from_generation_params(self, gen_params: "GenParameters") -> None:
    

        source = vars(gen_params)
        allowed = {f.name for f in fields(SharedState)}

        for key, value in vars(gen_params).items():
            if hasattr(self.p, key):
                setattr(self.p, key, value)


