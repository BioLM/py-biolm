"""Protocol schema validation and execution for BioLM."""
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List

# Protocol schema path for validation
PROTOCOL_SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "schema",
    "protocol_schema.json"
)


@dataclass
class ValidationError:
    """Represents a single validation error."""
    message: str
    path: str = ""  # JSONPath-like path to error location
    error_type: str = "unknown"  # schema, semantic, syntax, etc.


@dataclass
class ProtocolValidationResult:
    """Result of protocol validation."""
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, message: str, path: str = "", error_type: str = "unknown"):
        """Add an error to the result."""
        self.errors.append(ValidationError(message=message, path=path, error_type=error_type))
        self.is_valid = False
    
    def add_warning(self, message: str):
        """Add a warning to the result."""
        self.warnings.append(message)


class Protocol:
    """
    Protocol definition and execution.
    
    Args:
        yaml_path (str): Path to protocol YAML file.
    """
    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.data = self._load_yaml(yaml_path)
        self._validate()
    
    def _load_yaml(self, yaml_path: str) -> dict:
        """Load YAML file."""
        try:
            import yaml
            with open(yaml_path, 'r') as f:
                return yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML is required for protocol support. Install with: pip install pyyaml")
        except Exception as e:
            raise ValueError(f"Failed to load protocol YAML: {e}")
    
    def _validate(self):
        """Validate protocol against JSON schema (legacy method for __init__)."""
        result = self.validate(self.yaml_path)
        if not result.is_valid:
            # For backward compatibility, raise on first error
            if result.errors:
                error = result.errors[0]
                raise ValueError(f"Protocol validation failed: {error.message}")
    
    @classmethod
    def validate(cls, yaml_path: str) -> ProtocolValidationResult:
        """
        Validate a protocol YAML file.
        
        Args:
            yaml_path: Path to protocol YAML file.
            
        Returns:
            ProtocolValidationResult with validation results.
        """
        result = ProtocolValidationResult(is_valid=True)
        
        # Phase 1: YAML syntax validation
        try:
            data = cls._load_yaml_static(yaml_path)
        except Exception as e:
            result.add_error(
                f"Failed to parse YAML: {e}",
                path="",
                error_type="syntax"
            )
            return result  # Can't continue without valid YAML
        
        # Phase 2: Schema validation
        cls._validate_schema(data, result)
        
        # Phase 3: Semantic validation (only if we have valid YAML structure)
        # Only do semantic validation if we successfully parsed YAML
        if data:
            cls._validate_task_references(data, result)
            cls._validate_circular_dependencies(data, result)
            cls._validate_template_expressions(data, result)
            cls._collect_statistics(data, result)
        
        return result
    
    @staticmethod
    def _load_yaml_static(yaml_path: str) -> dict:
        """Load YAML file (static version for validate method)."""
        try:
            import yaml
            with open(yaml_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            raise ImportError("PyYAML is required for protocol support. Install with: pip install pyyaml")
        except Exception as e:
            raise ValueError(f"Failed to load protocol YAML: {e}")
    
    @staticmethod
    def _validate_schema(data: dict, result: ProtocolValidationResult):
        """Validate protocol against JSON schema."""
        if not os.path.exists(PROTOCOL_SCHEMA_PATH):
            result.add_warning("Schema file not found, skipping schema validation")
            return
        
        try:
            import jsonschema
        except ImportError:
            result.add_warning("jsonschema not installed, skipping schema validation")
            return
        
        try:
            with open(PROTOCOL_SCHEMA_PATH, 'r') as f:
                schema = json.load(f)
            
            validator = jsonschema.Draft202012Validator(schema)
            errors = list(validator.iter_errors(data))
            
            for error in errors:
                path = ".".join(str(p) for p in error.path)
                result.add_error(
                    error.message,
                    path=path if path else "root",
                    error_type="schema"
                )
        except Exception as e:
            result.add_error(
                f"Schema validation error: {e}",
                path="",
                error_type="schema"
            )
    
    @staticmethod
    def _validate_task_references(data: dict, result: ProtocolValidationResult):
        """Validate that all task references point to valid task IDs."""
        tasks = data.get("tasks", [])
        if not isinstance(tasks, list):
            return
        
        # Extract all task IDs
        task_ids = set()
        for i, task in enumerate(tasks):
            if isinstance(task, dict):
                task_id = task.get("id")
                if task_id:
                    if task_id in task_ids:
                        result.add_error(
                            f"Duplicate task ID: '{task_id}'",
                            path=f"tasks[{i}].id",
                            error_type="semantic"
                        )
                    task_ids.add(task_id)
        
        # Check references in depends_on, from, foreach, outputs
        for i, task in enumerate(tasks):
            if not isinstance(task, dict):
                continue
            
            task_id = task.get("id", f"tasks[{i}]")
            base_path = f"tasks[{i}]"
            
            # Check depends_on
            depends_on = task.get("depends_on", [])
            if isinstance(depends_on, list):
                for j, dep in enumerate(depends_on):
                    if isinstance(dep, str) and dep not in task_ids:
                        result.add_error(
                            f"Task '{task_id}' references unknown task ID '{dep}' in depends_on",
                            path=f"{base_path}.depends_on[{j}]",
                            error_type="semantic"
                        )
            
            # Check from
            from_task = task.get("from")
            if isinstance(from_task, str) and from_task not in task_ids:
                result.add_error(
                    f"Task '{task_id}' references unknown task ID '{from_task}' in from",
                    path=f"{base_path}.from",
                    error_type="semantic"
                )
            
            # Check foreach (can be a template expression, so we check if it's a simple string reference)
            foreach = task.get("foreach")
            if isinstance(foreach, str) and not foreach.startswith("${{"):
                # Simple string reference (not a template expression)
                if foreach not in task_ids:
                    result.add_error(
                        f"Task '{task_id}' references unknown task ID '{foreach}' in foreach",
                        path=f"{base_path}.foreach",
                        error_type="semantic"
                    )
            
            # Check outputs references
            outputs = data.get("outputs", [])
            if isinstance(outputs, list):
                for j, output in enumerate(outputs):
                    if isinstance(output, dict):
                        from_task = output.get("from")
                        if isinstance(from_task, str) and from_task not in task_ids:
                            result.add_error(
                                f"Output rule references unknown task ID '{from_task}'",
                                path=f"outputs[{j}].from",
                                error_type="semantic"
                            )
    
    @staticmethod
    def _validate_circular_dependencies(data: dict, result: ProtocolValidationResult):
        """Detect circular dependencies in task dependency graph."""
        tasks = data.get("tasks", [])
        if not isinstance(tasks, list):
            return
        
        # Build dependency graph
        graph = {}
        task_ids = {}
        
        for i, task in enumerate(tasks):
            if not isinstance(task, dict):
                continue
            task_id = task.get("id")
            if not task_id:
                continue
            
            task_ids[i] = task_id
            depends_on = task.get("depends_on", [])
            if isinstance(depends_on, list):
                graph[task_id] = [dep for dep in depends_on if isinstance(dep, str)]
            else:
                graph[task_id] = []
        
        # DFS to detect cycles
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: str, path: List[str]) -> Optional[List[str]]:
            """Check for cycles starting from node."""
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                return path[cycle_start:] + [node]
            
            if node in visited:
                return None
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in graph.get(node, []):
                if neighbor in graph:  # Only check if neighbor is a valid task
                    cycle = has_cycle(neighbor, path)
                    if cycle:
                        return cycle
            
            rec_stack.remove(node)
            path.pop()
            return None
        
        # Check all nodes
        for task_id in graph:
            if task_id not in visited:
                cycle = has_cycle(task_id, [])
                if cycle:
                    result.add_error(
                        f"Circular dependency detected: {' -> '.join(cycle)}",
                        path="tasks",
                        error_type="semantic"
                    )
                    break  # Report first cycle found
    
    @staticmethod
    def _validate_template_expressions(data: dict, result: ProtocolValidationResult):
        """Validate template expression syntax."""
        template_pattern = re.compile(r'\$\{\{[^}]*\}\}')
        
        def check_value(value: Any, path: str):
            """Recursively check values for template expressions."""
            if isinstance(value, str):
                # Check for template expressions
                if value.startswith("${{") and value.endswith("}}"):
                    # Valid template expression syntax
                    inner = value[3:-2].strip()
                    if not inner:
                        result.add_error(
                            "Empty template expression",
                            path=path,
                            error_type="semantic"
                        )
                    # Check for balanced braces (basic check)
                    brace_count = 0
                    for char in inner:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count < 0:
                                result.add_error(
                                    "Unbalanced braces in template expression",
                                    path=path,
                                    error_type="semantic"
                                )
                                break
                    if brace_count != 0:
                        result.add_error(
                            "Unbalanced braces in template expression",
                            path=path,
                            error_type="semantic"
                        )
                elif "${{" in value or "}}" in value:
                    # Malformed template expression
                    result.add_error(
                        "Malformed template expression (missing ${{ or }})",
                        path=path,
                        error_type="semantic"
                    )
            elif isinstance(value, dict):
                for key, val in value.items():
                    check_value(val, f"{path}.{key}" if path else key)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    check_value(item, f"{path}[{i}]" if path else f"[{i}]")
        
        # Check all data recursively
        check_value(data, "")
    
    @staticmethod
    def _collect_statistics(data: dict, result: ProtocolValidationResult):
        """Collect protocol statistics."""
        stats = {}
        
        # Count inputs
        inputs = data.get("inputs", {})
        if isinstance(inputs, dict):
            stats["input_count"] = len(inputs)
        else:
            stats["input_count"] = 0
        
        # Count tasks
        tasks = data.get("tasks", [])
        if isinstance(tasks, list):
            stats["task_count"] = len(tasks)
            stats["model_task_count"] = sum(
                1 for t in tasks
                if isinstance(t, dict) and t.get("type") != "gather" and "slug" in t
            )
            stats["gather_task_count"] = sum(
                1 for t in tasks
                if isinstance(t, dict) and t.get("type") == "gather"
            )
        else:
            stats["task_count"] = 0
            stats["model_task_count"] = 0
            stats["gather_task_count"] = 0
        
        # Count output rules
        outputs = data.get("outputs", [])
        if isinstance(outputs, list):
            stats["output_rule_count"] = len(outputs)
        else:
            stats["output_rule_count"] = 0
        
        # Protocol name
        stats["protocol_name"] = data.get("name", "unknown")
        
        result.statistics = stats
    
    @staticmethod
    def _get_examples_dir() -> Path:
        """Get path to examples directory."""
        # Get the package root directory (biolmai/)
        package_dir = Path(__file__).parent
        # Go up one level to project root, then into examples/
        project_root = package_dir.parent
        examples_dir = project_root / "examples"
        return examples_dir
    
    @staticmethod
    def _list_available_examples() -> List[str]:
        """List all available example protocol files."""
        examples_dir = Protocol._get_examples_dir()
        if not examples_dir.exists():
            return []
        
        examples = []
        for file_path in examples_dir.glob("*.yaml"):
            # Return name without extension
            examples.append(file_path.stem)
        
        return sorted(examples)
    
    @staticmethod
    def _load_example(name: str) -> str:
        """Load example protocol file content.
        
        Args:
            name: Example name (with or without .yaml extension)
            
        Returns:
            File content as string
            
        Raises:
            FileNotFoundError: If example file doesn't exist
        """
        # Strip .yaml extension if provided
        if name.endswith('.yaml') or name.endswith('.yml'):
            name = name.rsplit('.', 1)[0]
        
        examples_dir = Protocol._get_examples_dir()
        example_path = examples_dir / f"{name}.yaml"
        
        if not example_path.exists():
            available = Protocol._list_available_examples()
            raise FileNotFoundError(
                f"Example '{name}' not found. Available examples: {', '.join(available)}"
            )
        
        with open(example_path, 'r') as f:
            return f.read()
    
    @staticmethod
    def _generate_minimal_template(name: Optional[str] = None) -> str:
        """Generate minimal valid protocol YAML template.
        
        Args:
            name: Protocol name (defaults to "My_Protocol")
            
        Returns:
            YAML template as string
        """
        if name is None:
            name = "My_Protocol"
        
        template = f"""name: {name}
schema_version: 1

inputs:
  # Add your input parameters here
  # Example: input_param: string

tasks:
  # Add your tasks here
  # Example:
  # - id: my_task
  #   slug: model-slug
  #   action: predict
  #   request_body:
  #     items: []
"""
        return template
    
    @classmethod
    def init(cls, output_path: str, example: Optional[str] = None, force: bool = False) -> str:
        """Initialize a new protocol YAML file.
        
        Args:
            output_path: Path where the protocol file should be created
            example: Optional example template name to use
            force: If True, overwrite existing file
            
        Returns:
            Path to the created file
            
        Raises:
            FileExistsError: If file exists and force=False
            ValueError: If example name is invalid
            FileNotFoundError: If example file doesn't exist
        """
        output_path_obj = Path(output_path)
        
        # Check if file exists
        if output_path_obj.exists() and not force:
            raise FileExistsError(
                f"File '{output_path}' already exists. Use --force to overwrite."
            )
        
        # Generate protocol content
        if example:
            # Load from example
            content = cls._load_example(example)
        else:
            # Generate minimal template
            # Derive name from filename if not provided
            protocol_name = output_path_obj.stem.replace('_', ' ').title().replace(' ', '_')
            if not protocol_name or protocol_name == '.':
                protocol_name = "My_Protocol"
            content = cls._generate_minimal_template(protocol_name)
        
        # Write file
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path_obj, 'w') as f:
            f.write(content)
        
        return str(output_path_obj)
    
    def execute(self, inputs: Optional[Dict[str, Any]] = None):
        """Execute protocol with given inputs.
        
        Args:
            inputs: Input values for the protocol (optional, uses defaults from protocol if not provided).
            
        Returns:
            Protocol execution results.
            
        Note:
            Protocol execution is not yet implemented. This is a placeholder.
        """
        raise NotImplementedError("Protocol execution is not yet implemented.")

