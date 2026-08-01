from lark import Lark, Visitor
import lark
from typing import Optional

class LearnedConditioning:
    def __init__(self, shared_state, steps, hires_steps: Optional[int]=None, prompts = None, use_old_scheduling:Optional[bool]=False ):
        self.state = shared_state
        self.prompts: SdConditioning | list[str] = prompts if prompts is not None else ""
        self.steps: int = steps if steps is not None else self.state.p.steps
        self.hires_steps= hires_steps
        self.use_old_scheduling=use_old_scheduling if use_old_scheduling is not None else self.state.conditioning.use_old_scheduling
        
        
        self.schedule_parser = Lark(r"""
        !start: (prompt | /[][():]/+)*
        prompt: (emphasized | scheduled | grouped | alternate | alternate1 | alternate2 |  top_level_sequence | compound | numbered | and_rule | plain | WHITESPACE)*

        !emphasized: "(" prompt ")" 
                | "(" prompt ":" prompt ")"
                | "[" prompt "]"
        scheduled: "[" [prompt (":" prompt)+] "]" ":" NUMBER (step_range_list | reverse_flag | step_range_list reverse_flag)?
        reverse_flag: "reverse" | "r"
        step_range_list: step_range ("," step_range)*
        step_range: NUMBER "-" NUMBER | NUMBER "%" "-" NUMBER "%"
           
        alternate: "[" prompt ("|" [prompt])+ "]"
        alternate1: (prompt) "|" (prompt)+
        alternate2: (plain | compound) ("|" (plain | compound))+
        grouped: "{" ((NUMBER_Q | prompt | sequence | grouped) (","| "|")?)+ "}"

        top_level_sequence: prompt "::" sequence ("::" sequence)*  "!!"
        sequence: prompt "::" prompt ("," | WHITESPACE)* nested_sequence* "!"  
        nested_sequence: "::" prompt ("," | WHITESPACE)* ("~" | "!")


        compound: /[a-zA-Z0-9]+(_[a-zA-Z0-9]+)+/
        numbered: NUMBER_Q ("!")? (grouped | sequence | compound | and_rule | plain)
        and_rule: (plain | compound) ("&" (plain | compound))+
        WHITESPACE: /\s+/
        plain: /([^\\\[\]();]|\\.)+/

        %import common.SIGNED_NUMBER -> NUMBER // For weights and general numbers
        %import common.INT -> NUMBER_Q // For quantities

        """)

    def get_learned_cond(self, prompts, steps:int, hires_steps:Optional[int]=None):
        """
        >>> g = lambda p: get_learned_cond([p], 10)[0]
        >>> g("test")
        [[10, 'test']]
        >>> g("a [b:3]")
        [[3, 'a '], [10, 'a b']]
        >>> g("a [b: 3]")
        [[3, 'a '], [10, 'a b']]
        >>> g("a [[[b]]:2]")
        [[2, 'a '], [10, 'a [[b]]']]
        >>> g("[(a:2):3]")
        [[3, ''], [10, '(a:2)']]
        >>> g("a [b : c : 1] d")
        [[1, 'a b  d'], [10, 'a  c  d']]
        >>> g("a[b:[c:d:2]:1]e")
        [[1, 'abe'], [2, 'ace'], [10, 'ade']]
        >>> g("a [unbalanced")
        [[10, 'a [unbalanced']]
        >>> g("a [b:.5] c")
        [[5, 'a  c'], [10, 'a b c']]
        >>> g("a [{b|d{:.5] c")  # not handling this right now
        [[5, 'a  c'], [10, 'a {b|d{ c']]
        >>> g("((a][:b:c [d:3]")
        [[3, '((a][:b:c '], [10, '((a][:b:c d']]
        >>> g("[a|(b:1.1)]")
        [[1, 'a'], [2, '(b:1.1)'], [3, 'a'], [4, '(b:1.1)'], [5, 'a'], [6, '(b:1.1)'], [7, 'a'], [8, '(b:1.1)'], [9, 'a'], [10, '(b:1.1)']]
        >>> g("[fe|]male")
        [[1, 'female'], [2, 'male'], [3, 'female'], [4, 'male'], [5, 'female'], [6, 'male'], [7, 'female'], [8, 'male'], [9, 'female'], [10, 'male']]
        >>> g("[fe|||]male")
        [[1, 'female'], [2, 'male'], [3, 'male'], [4, 'male'], [5, 'female'], [6, 'male'], [7, 'male'], [8, 'male'], [9, 'female'], [10, 'male']]
        >>> g = lambda p: get_learned_cond([p], 10, 10)[0]
        >>> g("a [b:.5] c")
        [[10, 'a b c']]
        >>> g("a [b:1.5] c")
        [[5, 'a  c'], [10, 'a b c']]
        """

        if self.hires_steps is None or self.use_old_scheduling:
            int_offset = 0
            flt_offset = 0
            steps = self.steps
        else:
            int_offset = self.steps
            flt_offset = 1.0
            steps = self.hires_steps

        def collect_steps(steps, tree):
            #if not tree or not hasattr(tree, 'children') or not tree.children: #debugs
                #print("Invalid tree structure:", tree)
                #return []
            res = [steps]  # Always include the final step
            def resolve_tree(tree):
                """Recursively resolve a tree node to its final string representation."""
                if isinstance(tree, lark.Tree):
                    # Recursively resolve each child
                    return "".join(resolve_tree(child) for child in tree.children)
                return str(tree)

            class CollectSteps(lark.Visitor):
                def alternate2(self, tree):
                    # Resolve all alternates
                    options = []
                    for child in tree.children:
                        if isinstance(child, lark.Tree):
                            options.append(resolve_tree(child))
                        else:
                            options.append(str(child).strip())

                    # Combine options with `_` where applicable
                    combined_options = []
                    for option in options:
                        if "_" in option:
                            prefix, suffix = option.split("_", 1)
                            combined_options.append(f"{prefix}_{suffix}")
                        else:
                            combined_options.append(option)

                    # Add combined alternates to results
                    res.append("|".join(combined_options))
                
                def compound(self, tree):
                    # Treat compound phrases as a single unit
                    res.append("".join(tree.children))  

                def top_level_sequence(self, tree):
                    """
                    Handles top-level sequences by ensuring a defined owner (prompt) before '::'.
                    Example: woman:::hair_style::ponytail!, eye_color::blue!
                    """
                    owner = resolve_tree(tree.children[0])  # Get the owner prompt (e.g., "woman")
                    sequences = []

                    for child in tree.children[1:]:
                        if isinstance(child, lark.Tree) and child.data == "sequence":
                            sequences.append(self.sequence(child, owner))
                        elif isinstance(child, str):
                            if child.strip() == "!!":  # Ensure closure is respected
                                break
                            sequences.append(child.strip(" ~!")) # Strip trailing markers                       

                    # Store the structured sequence
                    res.append(f"{owner} -> {', '.join(sequences)}")

                def sequence(self, tree, parent=None):
                    """
                    Handles individual sequences. Ensures sequences have an owner.
                    Example: hair_style::ponytail~hair_color::blonde!
                    """
                    if parent is None:
                        # If this is a root sequence, the first child is the owner.
                        owner = resolve_tree(tree.children[0])
                        children = tree.children[1:]
                    else:
                        # This is a child sequence under a parent (top-level sequence).
                        owner = parent
                        children = tree.children

                    descriptors = []
                    for child in children:
                        if isinstance(child, lark.Tree):
                            if child.data == "nested_sequence":  # Handle nested sequences separately
                                descriptors.append(self.nested_sequence(child))
                            else:
                                descriptors.append(resolve_tree(child))  # Recursively resolve other structures
                        elif isinstance(child, str):
                            descriptors.append(child.strip(" ~!"))  # Strip trailing "~" or "!"

                    # Resolve the sequence by combining the described object and its attributes
                    combined_description = f"{owner}: {', '.join(descriptors)}"
                    res.append(combined_description)

                def nested_sequence(self, tree):
                    """
                    Handles nested sequences, ensuring they remain part of their parent sequence.
                    Example: ::hair_color::blonde!
                    """
                    sequence_elements = []
                    for child in tree.children:
                        if isinstance(child, lark.Tree):
                            sequence_elements.append(resolve_tree(child))
                        elif isinstance(child, str):
                            sequence_elements.append(child.strip(" ~!"))  # Strip trailing "~" or "!"

                    # Return formatted nested sequence
                    return f"[{' | '.join(sequence_elements)}]"  # Format it like an embedded list

                   
                def alternate1(self, tree):
                    # Randomly resolve `alternate1`
                    #options = [str(child) if not isinstance(child, lark.Tree) else resolve_tree(child) for child in tree.children]               
                    options = [resolve_tree(child) for child in tree.children]
                    res.append(random.choice(options))  # Random choice from options
                    
                
                def grouped(self,tree):
                    # Collect all descriptions within the group       
                    ##group_descriptions = [
                        ##self._resolve_tree(child) if isinstance(child, lark.Tree) else str(child) 
                        ##for child in tree.children]
                    
                    #print(f"Group: {group_descriptions}") #debug
                    
                    # Handle the group as a cohesive unit (e.g., append to results)               
                    ##res.append(", ".join(group_descriptions))
                    group_descriptions = [
                        resolve_tree(child) if isinstance(child, lark.Tree) else str(child)
                        for child in tree.children
                    ]
                    res.append(", ".join(group_descriptions))
                def scheduled(self, tree):
                    # Validate tree structure and children
                    if not hasattr(tree, "children") or not tree.children:
                        return

                    # Extract prompts and the scheduling number
                    prompts = tree.children[:-2]  # All but the last two children are options
                    number_node = tree.children[-2]  # Second-to-last child is the scheduling number
                    additional_info = tree.children[-1] if len(tree.children) > 2 else None

                    # Initialize parameters
                    is_reverse = False
                    step_intervals = []

                    # Safeguard for missing or invalid children
                    if not prompts or not number_node:
                        return

                    # Convert number_node to a float (scheduling weight or total steps percentage)
                    try:
                        v = float(number_node)
                    except ValueError:
                        return

                    # Handle additional parameters (reverse flag or step ranges)
                    if additional_info:
                        if isinstance(additional_info, str) and additional_info.lower() in ("reverse", "r"):
                            is_reverse = True
                        elif isinstance(additional_info, list):
                            # Process step ranges
                            for r in additional_info:
                                start, end = r.split("-")
                                if "%" in start or "%" in end:  # Handle percentage-based ranges                                
                                    start = round(float(start.strip("%")) / 100 * steps)
                                    end = round(float(end.strip("%")) / 100 * steps)                                
                                else:  # Handle absolute step ranges
                                    start, end = int(start), int(end)
                                 # Clamp ranges to valid boundaries
                                if start > steps:
                                    start = steps  # Adjust start to max steps
                                if end > steps:
                                    end = steps  # Adjust end to max steps
                                # Ignore invalid ranges where start > end
                                if start > end:
                                    print(f"Warning: Ignored invalid range {start}-{end}.")
                                    continue
                                step_intervals.append((start, end))

                    # If no step ranges are specified, generate default intervals based on weight
                    if not step_intervals:
                        num_prompts = len(prompts)
                        step_intervals = [
                            (int(i * (v * steps) / num_prompts), int((i + 1) * (v * steps) / num_prompts))
                            for i in range(num_prompts)
                        ]

                    # Handle reverse scheduling
                    if is_reverse:
                        prompts = prompts[::-1]  # Reverse prompts
                        step_intervals = step_intervals[::-1]  # Reverse intervals

                    # Replace number_node with numeric step intervals in the tree
                    tree.children[-2] = step_intervals

                    # Extend the results with calculated intervals
                    res.extend(step_intervals)
        
            # Visit the tree and collect step intervals
            CollectSteps().visit(tree)
            #return sorted(set(res))  # Remove duplicates and sort
            return res #does not remove duplicates or sort them


        def at_step(step, tree):
            class AtStep(lark.Transformer):
                def and_rule(self, args):
                    # All elements in args must be resolved
                    resolved_items = [self._resolve_tree(arg) if isinstance(arg, lark.Tree) else str(arg) for arg in args]
                    return " and ".join(resolved_items)  # Join items with "and"
                
                def compound(self, args):
                    # Return the compound phrase as a single string                
                    return "_".join(str(arg) for arg in args)                
                
               
                def top_level_sequence(self, args):
                    """
                    Handles top-level sequences where an explicit owner is required.
                    Example: woman:::hair_style::ponytail!, eye_color::blue!
                    """
                    owner = args[0]  # Extracts the first prompt (e.g., "woman")
                    sequences = []

                    for child in args[1:]:
                        if isinstance(child, list):
                            sequences.append(self.sequence(child, owner))
                        elif isinstance(child, str):
                            if child.strip() == "!!":  # Stop processing at `!!`
                                break
                            sequences.append(child.strip(" ~!"))

                    # Return structured output with owner and its sequences
                    return f"{owner} -> {', '.join(sequences)}"

                def sequence(self, args, parent=None):
                    """
                    Handles sequences by ensuring an owner (prompt) is defined before '::'.
                    Example: hair_style::ponytail~hair_color::blonde!
                    """
                    if parent is None:
                        # If this is a root sequence, the first argument is the owner.
                        owner = args[0]
                        children = args[1:]
                    else:
                        # If it's a child sequence, it inherits the parent's owner.
                        owner = parent
                        children = args

                    descriptors = []
                    for child in children:
                        if isinstance(child, str):
                            descriptors.append(child.strip(" ~!"))  # Remove sequence markers
                        elif isinstance(child, list):  
                            descriptors.append(self.nested_sequence(child))  # Handle nested sequences

                    # Format the sequence
                    return f"{owner}: {', '.join(descriptors)}"

                def nested_sequence(self, args):
                    """
                    Handles nested sequences within sequences.
                    Example: ::hair_color::blonde!
                    """
                    sequence_elements = []
                    for desc in args:
                        if isinstance(desc, str):
                            sequence_elements.append(desc.strip(" ~!"))  # Strip trailing markers
                        elif isinstance(desc, list):
                            sequence_elements.append(self.nested_sequence(desc))  # Handle deep nesting

                    # Format the nested sequence
                    return f"[{' | '.join(sequence_elements)}]"

                
                
                def alternate1(self, args):
                    # Randomly select one of the alternates
                    return random.choice(args)
                
                def alternate2(self, args):
                    # Resolve all alternates into a list
                    resolved_options = []
                    for arg in args:
                        if isinstance(arg, str):  # If it's already plain text
                            resolved_options.append(arg)
                        elif isinstance(arg, lark.Tree):  # If it's a compound or nested structure
                            resolved_options.append(self._resolve_tree(arg))

                    # Process each option to combine with `_` if necessary
                    combined_options = []
                    for option in resolved_options:
                        if "_" in option:  # Handle compounds like "green_eyes"
                            combined_options.append(option)  # Already combined
                        else:  # Combine plain text alternates
                            suffix = option.split("_")[-1] if "_" in resolved_options[0] else ""
                            combined_options.append(f"{option}_{suffix}" if suffix else option)

                    # Return combined alternates separated by `|`
                    return " | ".join(combined_options)
                    
                def scheduled(self, args):  
                    # Ensure args is valid
                    if not args or len(args) < 2:
                        return

                    # Extract components
                    *prompts, when, _, is_reverse, weight = args
                    step_intervals = args[-2]  # Step intervals
                    is_reverse = args[-1].lower() in ("reverse", "r") if len(args) > 2 else False

                    # Validate `when`
                    if not isinstance(when, list):
                        return

                    # Handle reverse scheduling
                    if is_reverse:
                        prompts = prompts[::-1]  # Reverse prompts
                        when = when[::-1]        # Reverse boundaries

                    # Iterate over the step intervals
                    for i, (start, end) in enumerate(step_intervals):
                        # Skip invalid or clamped ranges
                        if start > end:
                            continue
                        if start <= step <= end:
                            #yield f"({prompts[i]}:focus)"  # Apply focus during the range
                            yield f"({prompts[i]}:{weight})"  # Uniform weight
                            return

                    # Select the appropriate prompt based on the step
                    for i, boundary in enumerate(when):
                        if step <= boundary:
                            yield f"({prompts[i]}:{weight})"  # Apply weight (de-emphasis)
                            return
                    
                    # Default to the last prompt with the weight if step exceeds boundaries
                    yield f"({prompts[-1]}:{weight})"
                    
                def alternate(self, args):
                    # Handle alternates with a cycle
                    args = ["" if not arg else arg for arg in args]
                    yield args[(step - 1) % len(args)]
                def start(self, args):
                    #flatten nested structures into a single string
                    def flatten(x):
                        if isinstance(x, str):
                            yield x
                        else:
                            for gen in x:
                                yield from flatten(gen)
                    return ''.join(flatten(args))
                def plain(self, args):
                    #handle plain text nodes
                    yield args[0].value
                def grouped(self, args):
                    ### Return the group as a cohesive string
                    ##return ", ".join(args)
                    # Combine all grouped elements into a single string
                    return f"{{{', '.join(args)}}}"
                def __default__(self, data, children, meta):
                    #handle all other nodes
                    for child in children:
                        yield child
                def _resolve_tree(self, tree):
                    """Recursively resolve a tree node to its final string representation."""
                    if isinstance(tree, lark.Tree):
                        return "".join(self._resolve_tree(child) for child in tree.children)
                    return str(tree)
                    
            return AtStep().transform(tree)

        def get_schedule(prompt):
            try:
                tree = self.schedule_parser.parse(prompt)
                #print(tree.pretty())  # Debugging: visualize the tree structure
           
            except lark.exceptions.LarkError as e:
                #print(f"Parsing error for prompt: {prompt}")
                #if 0:
                #    import traceback
                #    traceback.print_exc()
                return [[steps, prompt]]            
                
            return [[t, at_step(t, tree)] for t in collect_steps(steps, tree)]

        promptdict = {prompt: get_schedule(prompt) for prompt in set(prompts)}
        return [promptdict[prompt] for prompt in prompts]

