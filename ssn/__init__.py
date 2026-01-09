"""
SSN - Sigil Structured Notation

A token-efficient notation format for LLM communication.
Reduces token usage by 60-75% compared to JSON.

Supports:
- Simple natural language queries
- Complex structured prompts (markdown-based)
- Auto-detection of prompt type
"""

from .core import SSN, SSNSchema
from .parser import parse, parse_file
from .encoder import encode, to_ssn
from .decoder import decode, to_dict
from .nl_converter import (
    nl_to_ssn,
    structured_to_ssn,
    template,
    NLToSSN,
    SSNTemplates,
    UnifiedConverter,
    StructuredPromptConverter,
    PromptType,
)

__version__ = "0.2.0"
__author__ = "Pritam Kumar Panda"
__all__ = [
    # Core
    "SSN",
    "SSNSchema",
    # Parser
    "parse",
    "parse_file",
    # Encoder/Decoder
    "encode",
    "to_ssn",
    "decode",
    "to_dict",
    # Natural Language Converter
    "nl_to_ssn",
    "structured_to_ssn",
    "template",
    "NLToSSN",
    "SSNTemplates",
    # Structured Prompt Converter (NEW)
    "UnifiedConverter",
    "StructuredPromptConverter",
    "PromptType",
]
