from modules.ss_registry.schedulers.simple_kes_sched.arg_schema import ARG_SCHEMA_ORDER, ArgSchema
from typing import Optional, Dict, Any

class PluginSupport:
     # ====== Scheduler Plugin Support
    
    
       
    @staticmethod
    def update_state(state: Any, section: str, keys_to_sync: list[str] = None, exclude_keys: list[str] = None):
        """
        Syncs selected attributes from `self` into `state.section`.

        Parameters:
            state (Any): Shared state object (e.g., self.state).
            section (str): Name of the section to sync into (e.g., 'p', 'sched').
            keys_to_sync (list[str], optional): If provided, only these keys will be synced.
            exclude_keys (list[str], optional): Keys to explicitly exclude from syncing.
        """
        target_section = getattr(state, section, None)
        if target_section is None:
            raise ValueError(f"[update_sched_state] Section '{section}' not found in state.")

        for attr_name in dir():
            if attr_name.startswith("_"):
                continue  # Skip private/internal attributes
            if exclude_keys and attr_name in exclude_keys:
                continue  # Skip explicitly excluded keys
            if keys_to_sync and attr_name not in keys_to_sync:
                continue  # Skip non-specified keys
            if hasattr(target_section, attr_name) and hasattr(attr_name):
                attr_value = getattr(attr_name)
                setattr(target_section, attr_name, attr_value)
                print(f"[update_sched_state] → {section}.{attr_name} = {attr_value}")

    '''
    def resolve_aliases(self) -> dict:
        """
        Resolves kwargs from main program's canonical keys (e.g., 'steps') to
        plugin-specific expected names (e.g., 'n') based on self.sched_aliases.

        Returns:
            dict: A dictionary with keys renamed to the plugin's internal expectations.
        """
        resolved = {}

        for plugin_key, canonical_key in self.sched_aliases.items():
            if canonical_key in self.kwargs:
                resolved[plugin_key] = self.kwargs[canonical_key]

        # Also pass through unmapped args
        for key, value in self.kwargs.items():
            if key not in self.sched_aliases.values():  # Don't overwrite remapped keys
                resolved[key] = value

        return resolved
    '''
        
    @staticmethod
    def checkstate(state, section: str):
        """
        Verifies that the provided state object contains the required section attribute.

        Parameters:
        - state: The shared state object.
        - section (str): The name of the attribute expected to exist in the state.

        Raises:
        - ValueError if the state is None or does not contain the expected section.
        """
        import inspect
        caller = inspect.stack()[1]

        if not state:
            print(f"[SimpleKESSchedulerPlugin] ❌ State is None or not initialized (expected SharedState object).")
            print(f"↳ Called by: {caller.function} in {caller.filename}:{caller.lineno}")
            raise ValueError("SharedState is missing or None.")

        if not hasattr(state, section):
            print(f"[SimpleKESSchedulerPlugin] ❌ State is missing expected section: '{section}'")
            print(f"↳ Provided state type: {type(state).__name__}")
            print(f"↳ Called by: {caller.function} in {caller.filename}:{caller.lineno}")
            raise AttributeError(f"SharedState does not contain the section '{section}'.")


    @staticmethod
    def validate_meta(meta: dict):
        """
        Validates the structure and type integrity of a plugin's metadata.

        This method checks both the main `meta` dictionary and the `runtime_meta` dictionary
        for required keys and expected types. It ensures that all necessary metadata fields
        are present and conform to the expected structure for scheduler plugins.

        Raises:
            ValueError: If any required metadata keys are missing from either `meta` or `runtime_meta`.
            TypeError: If any metadata values are present but do not match their expected types.

        Required fields in `self.meta`:
            - name (str): Unique name identifier for the plugin.
            - label (str): Human-readable display name.
            - required_args (dict): Arguments required to run the scheduler.
            - optional_args (dict): Optional arguments with defaults.
            - config_file (dict): Paths to associated config files (e.g., YAML, JSON).

        Required fields in `self.runtime_meta`:
            - default_config_loader (callable): Function to load default config.
            - entry (object): The callable plugin instance or class reference.

        Returns:
            - True: This function returns a boolean True if it passes. This return is a hint only (it could be removed, but kept for clarity). 
            - It raises exceptions if validation fails.
        """
        required_keys = {
            "name",
            "label",
            "args",           
            "config_file"
        }
      

        # Check for missing keys in meta
        missing_meta = [key for key in required_keys if key not in meta]
       

        if missing_meta:
            print(f"[⚠️] Missing required meta keys: {missing_meta}")
            raise ValueError(
                f"[Plugin Meta] Missing required meta fields: {', '.join(missing_meta)}"
            )
        
        # Optional: type checking for meta
        type_errors = []

        for key, expected_type in required_keys.items():
            actual = meta.get(key)
            if not isinstance(actual, expected_type):
                type_errors.append(
                    f"meta['{key}'] should be {expected_type.__name__}, got {type(actual).__name__}"
                )
        
        if type_errors:
            raise TypeError(
                "[Plugin Meta] One or more fields have incorrect types:\n" +
                "\n".join(f"  - {e}" for e in type_errors)
            )
        return True

    @staticmethod
    def validate_arg_definition(name: str, values: list):
        """
        Validates that a list-based argument schema matches the expected format.

        This function checks that the number of elements in an argument's definition list
        matches the expected schema order defined by `ARG_SCHEMA_ORDER`. Each field is expected
        to appear in a specific position, corresponding to elements such as type, default, desc, etc.

        Parameters:
            name (str): The name of the argument being validated.
            values (list): A list of values representing the argument schema. 
                           The list must match the length and order defined in `ARG_SCHEMA_ORDER`.

        Raises:
            ValueError: If the number of elements in `values` does not match `ARG_SCHEMA_ORDER`.

        Returns:
            True: This function returns an optional Boolean value to signify it passed
        """
        if len(values) != len(ARG_SCHEMA_ORDER):
            raise ValueError(
                f"[ArgSchema] '{name}' has {len(values)} values, "
                f"expected {len(ARG_SCHEMA_ORDER)}: {ARG_SCHEMA_ORDER}"
            )
        return True

       
       
    @staticmethod
    def make_arg_schema_from_list(arg_list):
        """
        Converts a positional argument list into a structured ArgSchema instance.

        This function maps a flat list of values (ordered according to ARG_SCHEMA_ORDER)
        into a dictionary, and then uses it to instantiate an ArgSchema dataclass.

        Parameters:
            arg_list (list): A list of values matching the order defined by ARG_SCHEMA_ORDER.
                             Example:
                                 [int, 25, "Short desc", "Long desc", None, [1, 150], True]

        Returns:
            ArgSchema: A fully constructed ArgSchema instance representing the argument.

        Raises:
            ValueError: If the number of elements in the list does not match ARG_SCHEMA_ORDER.
        """
        if len(arg_list) != len(ARG_SCHEMA_ORDER):
            raise ValueError(f"[make_arg_schema_from_list] Expected {len(ARG_SCHEMA_ORDER)} items, got {len(arg_list)}")

        schema_dict = dict(zip(ARG_SCHEMA_ORDER, arg_list))
        
        return ArgSchema(
            type_    = schema_dict["type"],
            default  = schema_dict["default"],
            short_desc  = schema_dict["short_desc"],
            long_desc = schema_dict["long_desc"],
            choices  = schema_dict["choices"],
            range_   = schema_dict["range"],
            required = schema_dict["required"]
        )