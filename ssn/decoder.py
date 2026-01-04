"""
SSN Decoder - High-level decoding utilities.
"""

from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import SSN

from .parser import parse, parse_file


def decode(ssn_text: str, ssn_instance: Optional["SSN"] = None) -> Dict[str, Any]:
    """
    Decode SSN text to dictionary.
    
    Args:
        ssn_text: SSN formatted string
        ssn_instance: Optional SSN instance with schemas
    
    Returns:
        Dictionary representation of SSN data
    
    Example:
        >>> decode("@analyze|protein.pdb\\n>chain:A\\n#verbose")
        {'_action': 'analyze', '_args': ['protein.pdb'], 'chain': 'A', 'verbose': True}
    """
    return parse(ssn_text, ssn_instance)


def to_dict(ssn_text: str, ssn_instance: Optional["SSN"] = None) -> Dict[str, Any]:
    """Alias for decode()."""
    return decode(ssn_text, ssn_instance)


def decode_file(filepath: str, ssn_instance: Optional["SSN"] = None) -> Dict[str, Any]:
    """Decode SSN file to dictionary."""
    return parse_file(filepath, ssn_instance)


def decode_with_schema(
    ssn_text: str,
    schema_name: str,
    ssn_instance: "SSN"
) -> Dict[str, Any]:
    """
    Decode SSN text using a specific schema for expansion.
    
    Args:
        ssn_text: SSN formatted string
        schema_name: Name of registered schema
        ssn_instance: SSN instance with schemas
    
    Returns:
        Dictionary with schema-expanded values
    """
    result = parse(ssn_text, ssn_instance)
    
    if schema_name in ssn_instance.schemas:
        schema = ssn_instance.schemas[schema_name]
        args = result.get("_args", [])
        expanded = schema.expand(args)
        result.update(expanded)
    
    return result
