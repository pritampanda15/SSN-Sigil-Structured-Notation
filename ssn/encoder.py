"""
SSN Encoder - Converts Python objects to SSN text.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import SSN


class SSNEncoder:
    """
    Encoder for converting dictionaries to SSN notation.
    
    Strategies:
        1. Detect action/command patterns and use @ sigil
        2. Convert nested dicts to .scope notation
        3. Convert booleans to #flags
        4. Use > for key:value pairs
    """
    
    # Keys that indicate an action
    ACTION_KEYS = {"action", "task", "command", "method", "_action", "operation"}
    
    # Keys that should be positional args
    ARG_KEYS = {"input", "file", "target", "source", "_args"}
    
    def __init__(self, ssn_instance: Optional["SSN"] = None):
        self.ssn = ssn_instance
        self.schemas = ssn_instance.schemas if ssn_instance else {}
        self.reverse_schemas = {v.action: k for k, v in self.schemas.items()}
    
    def encode(self, data: Dict[str, Any], indent: int = 0) -> str:
        """Encode dictionary to SSN text."""
        lines = []
        prefix = "." * indent if indent > 0 else ""
        
        # Check for action pattern
        action = self._extract_action(data)
        if action:
            args = self._extract_args(data)
            remaining = {k: v for k, v in data.items() 
                        if k not in self.ACTION_KEYS and k not in self.ARG_KEYS and k != "_args"}
            
            # Check for schema compression
            action_name = self.reverse_schemas.get(action, action)
            
            line = f"{prefix}@{action_name}"
            if args:
                line += "|" + "|".join(str(a) for a in args)
            lines.append(line)
            
            # Encode remaining fields
            for key, value in remaining.items():
                lines.extend(self._encode_field(key, value, indent))
        else:
            # No action, encode all fields
            for key, value in data.items():
                lines.extend(self._encode_field(key, value, indent))
        
        return "\n".join(lines)
    
    def _extract_action(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract action name from data."""
        for key in self.ACTION_KEYS:
            if key in data:
                return str(data[key])
        return None
    
    def _extract_args(self, data: Dict[str, Any]) -> List[Any]:
        """Extract positional arguments."""
        args = []
        for key in self.ARG_KEYS:
            if key in data:
                val = data[key]
                if isinstance(val, list):
                    args.extend(val)
                else:
                    args.append(val)
        return args
    
    def _encode_field(self, key: str, value: Any, indent: int = 0) -> List[str]:
        """Encode a single field."""
        lines = []
        prefix = "." * indent if indent > 0 else ""
        
        if isinstance(value, bool):
            if value:
                lines.append(f"{prefix}#{key}")
            # False booleans are omitted (absence = false)
        elif isinstance(value, dict):
            lines.append(f"{prefix}.{key}")
            nested = self.encode(value, indent + 1)
            if nested:
                lines.append(nested)
        elif isinstance(value, list):
            # Encode list as comma-separated or multiple lines
            if all(isinstance(v, (str, int, float)) for v in value):
                lines.append(f"{prefix}>{key}:{','.join(str(v) for v in value)}")
            else:
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        lines.append(f"{prefix}.{key}[{i}]")
                        lines.append(self.encode(item, indent + 1))
                    else:
                        lines.append(f"{prefix}>{key}[{i}]:{item}")
        elif value is None:
            lines.append(f"{prefix}>{key}:null")
        else:
            lines.append(f"{prefix}>{key}:{value}")
        
        return lines


def encode(data: Dict[str, Any], ssn_instance: Optional["SSN"] = None) -> str:
    """Encode dictionary to SSN text."""
    encoder = SSNEncoder(ssn_instance)
    return encoder.encode(data)


def to_ssn(data: Dict[str, Any], ssn_instance: Optional["SSN"] = None) -> str:
    """Alias for encode()."""
    return encode(data, ssn_instance)
