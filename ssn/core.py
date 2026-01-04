"""
Core SSN classes and data structures.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum


class SigilType(Enum):
    """Sigil types in SSN notation."""
    ACTION = "@"      # Command/task
    CONTEXT = ">"     # Key-value context
    FLAG = "#"        # Boolean flag
    SCOPE = "."       # Nested scope
    REF = "~"         # Reference
    INHERIT = "^"     # Template inheritance
    SCHEMA = "$"      # Schema definition
    COMMENT = "//"    # Comment (ignored)


@dataclass
class SSNNode:
    """Single SSN statement node."""
    sigil: SigilType
    name: str
    args: List[str] = field(default_factory=list)
    value: Optional[str] = None
    children: List["SSNNode"] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary."""
        if self.sigil == SigilType.FLAG:
            return {self.name: True}
        elif self.sigil == SigilType.CONTEXT:
            return {self.name: self._parse_value(self.value)}
        elif self.sigil == SigilType.ACTION:
            result = {"_action": self.name}
            if self.args:
                result["_args"] = [self._parse_value(a) for a in self.args]
            for child in self.children:
                result.update(child.to_dict())
            return result
        elif self.sigil == SigilType.SCOPE:
            return {self.name: self._collect_children()}
        return {}
    
    def _parse_value(self, val: str) -> Union[str, int, float, bool, None]:
        """Parse string value to appropriate type."""
        if val is None:
            return None
        if val.lower() == "true":
            return True
        if val.lower() == "false":
            return False
        if val.lower() == "null" or val.lower() == "none":
            return None
        try:
            if "." in val:
                return float(val)
            return int(val)
        except ValueError:
            return val
    
    def _collect_children(self) -> Dict[str, Any]:
        """Collect all children into dict."""
        result = {}
        for child in self.children:
            result.update(child.to_dict())
        return result


@dataclass
class SSNSchema:
    """Schema definition for SSN compression."""
    name: str
    action: str
    arg_names: List[str] = field(default_factory=list)
    defaults: Dict[str, Any] = field(default_factory=dict)
    
    def expand(self, args: List[str]) -> Dict[str, Any]:
        """Expand positional args to named parameters."""
        result = dict(self.defaults)
        for i, arg in enumerate(args):
            if i < len(self.arg_names):
                result[self.arg_names[i]] = arg
        return result


class SSN:
    """
    Main SSN processor class.
    
    Handles parsing, encoding, and schema management.
    
    Example:
        >>> ssn = SSN()
        >>> ssn.register_schema("md", "molecular_dynamics", ["steps", "temp", "pressure"])
        >>> result = ssn.parse("@md|1000|300|1")
        >>> print(result)
        {'_action': 'molecular_dynamics', 'steps': 1000, 'temp': 300, 'pressure': 1}
    """
    
    def __init__(self):
        self.schemas: Dict[str, SSNSchema] = {}
        self.aliases: Dict[str, str] = {}
        self._register_defaults()
    
    def _register_defaults(self):
        """Register common bioinformatics schemas."""
        # Molecular dynamics
        self.register_schema(
            "md", "molecular_dynamics",
            ["steps", "temperature", "pressure"],
            {"ensemble": "NPT", "timestep": 2}
        )
        # Docking
        self.register_schema(
            "dock", "docking",
            ["receptor", "ligand", "center_x", "center_y", "center_z"],
            {"exhaustiveness": 8, "num_modes": 9}
        )
        # Sequence alignment
        self.register_schema(
            "align", "alignment",
            ["sequence", "database", "evalue"],
            {"algorithm": "blast", "max_hits": 100}
        )
        # ADMET prediction
        self.register_schema(
            "admet", "admet_prediction",
            ["smiles"],
            {"models": ["absorption", "distribution", "metabolism", "excretion", "toxicity"]}
        )
        # Protein analysis
        self.register_schema(
            "prot", "protein_analysis",
            ["pdb_file", "chain"],
            {"compute_contacts": True, "compute_sasa": True}
        )
        # RNA-seq
        self.register_schema(
            "rnaseq", "rna_sequencing",
            ["fastq_r1", "fastq_r2", "reference"],
            {"aligner": "star", "quantifier": "salmon"}
        )
        # Single-cell RNA-seq
        self.register_schema(
            "scrna", "single_cell_rnaseq",
            ["input_path", "reference"],
            {"tool": "cellranger", "chemistry": "auto"}
        )
        # Scanpy workflow
        self.register_schema(
            "scanpy", "scanpy_analysis",
            ["h5ad_file"],
            {"n_neighbors": 15, "n_pcs": 50}
        )
        # ATAC-seq
        self.register_schema(
            "atac", "atac_sequencing",
            ["fastq_r1", "fastq_r2", "reference"],
            {"aligner": "bowtie2", "peak_caller": "macs2"}
        )
        # Variant calling
        self.register_schema(
            "variant", "variant_calling",
            ["bam", "reference"],
            {"caller": "gatk", "mode": "germline"}
        )
        # ChIP-seq
        self.register_schema(
            "chip", "chip_sequencing",
            ["treatment", "control", "reference"],
            {"peak_type": "narrow", "caller": "macs2"}
        )
    
    def register_schema(
        self,
        short_name: str,
        full_name: str,
        arg_names: List[str] = None,
        defaults: Dict[str, Any] = None
    ):
        """Register a schema for compression."""
        self.schemas[short_name] = SSNSchema(
            name=short_name,
            action=full_name,
            arg_names=arg_names or [],
            defaults=defaults or {}
        )
        self.aliases[short_name] = full_name
    
    def parse(self, ssn_text: str) -> Dict[str, Any]:
        """Parse SSN text to dictionary."""
        from .parser import parse
        return parse(ssn_text, self)
    
    def encode(self, data: Dict[str, Any]) -> str:
        """Encode dictionary to SSN text."""
        from .encoder import encode
        return encode(data, self)
    
    def token_stats(self, ssn_text: str, json_text: str) -> Dict[str, Any]:
        """Compare token usage between SSN and JSON."""
        # Rough token estimation (4 chars ≈ 1 token)
        ssn_tokens = len(ssn_text) / 4
        json_tokens = len(json_text) / 4
        reduction = (1 - ssn_tokens / json_tokens) * 100
        return {
            "ssn_chars": len(ssn_text),
            "json_chars": len(json_text),
            "ssn_tokens_est": int(ssn_tokens),
            "json_tokens_est": int(json_tokens),
            "reduction_percent": round(reduction, 1)
        }
