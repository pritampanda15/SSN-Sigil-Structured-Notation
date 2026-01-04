"""
SSN - Sigil Structured Notation

A token-efficient notation format for LLM communication.
Reduces token usage by 60-75% compared to JSON.
"""

from .core import SSN, SSNSchema
from .parser import parse, parse_file
from .encoder import encode, to_ssn
from .decoder import decode, to_dict
from .nl_converter import nl_to_ssn, template, NLToSSN, SSNTemplates

__version__ = "0.1.0"
__author__ = "Pritam Kumar Panda"
__all__ = [
    "SSN",
    "SSNSchema", 
    "parse",
    "parse_file",
    "encode",
    "to_ssn",
    "decode",
    "to_dict",
    "nl_to_ssn",
    "template",
    "NLToSSN",
    "SSNTemplates",
]
