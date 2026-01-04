# SSN - Sigil Structured Notation

<p align="center">
  <img src="docs/ssn_logo.svg" alt="SSN Logo" width="200"/>
</p>

**A token-efficient notation format for LLM communication. Reduces token usage by 60-75% compared to JSON.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/ssn-notation.svg)](https://badge.fury.io/py/ssn-notation)

---

## Why SSN?

Every token costs money and latency when communicating with LLMs. JSON is verbose:

Natural language (what you'd normally type):

"Can you help me build a RAG-based pipeline for drug discovery using Python, LangChain, and ChromaDB? 
I want to use PubMed abstracts, ChEMBL compound data, and PDB protein structures as data sources. 
Use BioBERT for embeddings, ChromaDB as the vector database, and GPT-4 as the LLM. 
Chunk size should be 512 tokens. 
I need citation tracking, molecule-aware retrieval, and protein context features. 
Please provide production-ready \br code with an architecture diagram and Docker Compose setup."

```
SYSTEM PROMPT (paste this first into any GPT):
---
You understand SSN (Sigil Structured Notation) - a token-efficient format:
@ = action/command with |args
> = key:value pair
# = boolean flag (presence = true)
. = nested scope

Parse SSN queries and respond with detailed, actionable output.
---

USER QUERY (paste this after):
---
@build|rag_pipeline
>domain:drug_discovery
>stack:python,langchain,chromadb
.data_sources
  >pubmed:abstracts
  >chembl:compounds
  >pdb:structures
.components
  >embeddings:biobert
  >vectordb:chromadb
  >llm:gpt-4
  >chunking:512_tokens
.features
  #citation_tracking
  #molecule_aware
  #protein_context
.output
  >include:code,architecture_diagram
  >style:production_ready
  #docker_compose

Token comparison:

Natural language: ~95 tokens
SSN format: ~45 tokens
Savings: ~53%

```

---

```json
{"task":"molecular_dynamics","input":{"pdb":"protein.pdb","chain":"A"},"parameters":{"steps":10000,"temperature":300,"pressure":1},"output":{"format":"xtc","save_frequency":100}}
```
**~180 characters → ~45 tokens**

SSN expresses the same information in:

```
@md|protein.pdb
>chain:A
>steps:10000
>temp:300
>pressure:1
.out>fmt:xtc;>freq:100
```
**~75 characters → ~19 tokens** (58% reduction)

---

## Installation

```bash
pip install ssn-notation
```

Or install from source:

```bash
git clone https://github.com/yourusername/ssn.git
cd ssn
pip install -e .
```

---

## Command Line Interface

After installing, use the `ssn` command:

### Convert Natural Language → SSN (Copy-Paste to ChatGPT)

```bash
# Direct text conversion
ssn convert "Find top LLM tools in bioinformatics"

# From file
echo "Build a RAG pipeline for drug discovery" > prompt.txt
ssn convert prompt.txt

# WITH SYSTEM PROMPT (ready to paste to ChatGPT/Claude)
ssn convert prompt.txt --full

# Save to file
ssn convert prompt.txt --full -o query.txt

# Show token savings
ssn convert prompt.txt --stats
```

### Generate from Templates

```bash
ssn template search topic=protein_design domain=bioinformatics filter=top count=10
ssn template code language=python task=molecular_docking
ssn template debug error=CUDA_OOM context=training env=A100
```

### Get System Prompt Only

```bash
ssn system-prompt           # Short version
ssn system-prompt --detailed  # With sigil reference table
```

### Parse & Encode

```bash
ssn parse config.ssn --pretty   # SSN → JSON
ssn encode data.json -o out.ssn # JSON → SSN
```

---

## Quick Start

```python
from ssn import SSN, encode, decode, nl_to_ssn

# Initialize with built-in schemas
ssn = SSN()

# Parse SSN to dictionary
text = """
@dock|receptor.pdb|ligand.mol2
>center_x:10.5
>center_y:20.3
>center_z:15.0
#flexible
"""
result = ssn.parse(text)
print(result)
# {'_action': 'docking', '_args': ['receptor.pdb', 'ligand.mol2'], 
#  'center_x': 10.5, 'center_y': 20.3, 'center_z': 15.0, 'flexible': True}

# Encode dictionary to SSN
data = {
    "action": "analyze",
    "input": "structure.pdb",
    "compute_sasa": True,
    "parameters": {"radius": 1.4, "resolution": 0.1}
}
ssn_text = ssn.encode(data)
print(ssn_text)
# @analyze|structure.pdb
# #compute_sasa
# .parameters
# >radius:1.4
# >resolution:0.1

# NEW: Convert natural language to SSN
ssn_query = nl_to_ssn("Find me top LLM projects in bioinformatics")
print(ssn_query)
# @search|LLM_projects
# >domain:bioinformatics
# >filter:top
# #ranked
```

---

## Natural Language → SSN Conversion

Convert plain English queries to token-efficient SSN format:

```python
from ssn import nl_to_ssn, template

# Automatic conversion
nl_to_ssn("Find me top LLM projects in bioinformatics")
# @search|LLM_projects
# >domain:bioinformatics
# >filter:top
# #ranked

nl_to_ssn("Explain how transformers work")
# @query|explain
# >topic:transformers
# >depth:moderate
# >focus:key_concepts

nl_to_ssn("Compare PyTorch vs TensorFlow for research")
# @compare|PyTorch|TensorFlow
# >aspects:all
# #pros_cons

nl_to_ssn("Debug CUDA out of memory error in training")
# @debug|CUDA_out_of_memory
# >context:training
# #root_cause
# #fix

# With context flags
nl_to_ssn("Explain attention mechanisms", expert_mode=True, include_code=True)
# @query|explain
# >topic:attention_mechanisms
# >depth:moderate
# #no_basics
# #code_examples

# Using templates directly
template("search", topic="protein_folding", domain="bioinformatics", filter="recent", count="10")
template("code", language="python", task="parse_pdb_files")
template("debug", error="segfault", context="C_extension", env="linux")
```

### Supported Query Types

| Pattern | Keywords | Example |
|---------|----------|---------|
| Search | find, search, list, show | "Find top ML papers" |
| Explain | explain, what is, how does | "Explain RLHF" |
| Compare | compare, vs, difference | "Compare Scanpy vs Seurat" |
| Create | create, generate, write | "Write a Python script" |
| Analyze | analyze, evaluate, assess | "Analyze this code" |
| Optimize | optimize, improve, speed up | "Optimize GPU usage" |
| Debug | debug, fix, error, bug | "Debug memory leak" |
| Summarize | summarize, tldr, overview | "Summarize this paper" |
| Recommend | recommend, suggest, best | "Best tools for scRNA" |
| Code | code, implement, function | "Implement in Rust" |

---

## Sigil Reference

| Sigil | Name | Usage | Example |
|-------|------|-------|---------|
| `@` | Action | Command/task with positional args | `@dock\|receptor\|ligand` |
| `>` | Context | Key-value pair | `>temperature:300` |
| `#` | Flag | Boolean true (absence = false) | `#verbose` |
| `.` | Scope | Nested structure | `.parameters` |
| `~` | Reference | Refer to previous definition | `~config1` |
| `^` | Inherit | Extend a template | `^base_config` |
| `$` | Schema | Define schema alias | `$md=molecular_dynamics` |
| `\|` | Separator | Argument separator | `@action\|arg1\|arg2` |
| `;` | Terminator | Statement separator (optional) | `>a:1;>b:2` |

---

## Built-in Schemas

SSN includes pre-registered schemas for common tasks:

### Drug Design / Computational Biology
| Short | Full Name | Arguments |
|-------|-----------|-----------|
| `md` | molecular_dynamics | steps, temperature, pressure |
| `dock` | docking | receptor, ligand, center_x, center_y, center_z |
| `admet` | admet_prediction | smiles |
| `prot` | protein_analysis | pdb_file, chain |
| `align` | alignment | sequence, database, evalue |

### NGS / Single-Cell
| Short | Full Name | Arguments |
|-------|-----------|-----------|
| `rnaseq` | rna_sequencing | fastq_r1, fastq_r2, reference |
| `scrna` | single_cell_rnaseq | input_path, reference |
| `scanpy` | scanpy_analysis | h5ad_file |
| `atac` | atac_sequencing | fastq_r1, fastq_r2, reference |
| `variant` | variant_calling | bam, reference |
| `chip` | chip_sequencing | treatment, control, reference |

```python
# Using schema shorthand
ssn.parse("@md|10000|300|1")
# Expands to: molecular_dynamics with steps=10000, temperature=300, pressure=1

ssn.parse("@scrna|/data/fastqs|GRCh38")
# Expands to: single_cell_rnaseq with input_path=/data/fastqs, reference=GRCh38
```

---

## Custom Schemas

Register your own schemas for domain-specific compression:

```python
ssn = SSN()

# Register FEP calculation schema
ssn.register_schema(
    short_name="fep",
    full_name="free_energy_perturbation",
    arg_names=["ligand_a", "ligand_b", "lambda_windows"],
    defaults={"equilibration": 5000, "production": 50000}
)

# Now use it
result = ssn.parse("@fep|propofol|analog1|20")
print(result)
# {'_action': 'free_energy_perturbation', 
#  'ligand_a': 'propofol', 'ligand_b': 'analog1', 'lambda_windows': 20,
#  'equilibration': 5000, 'production': 50000}
```

---

## Complex Example: Drug Discovery Pipeline

```
@pipeline|drug_discovery
.target
  @prot|GABAA_receptor.pdb|beta3
  >binding_site:orthosteric
  #compute_druggability
.screening
  @dock|~target|library.sdf
  >exhaustiveness:32
  >num_modes:20
  #flexible_residues
.optimization
  @admet|~top_hits
  >filters:ro5,pains,brenk
  #predict_herg
.dynamics
  @md|100000|310|1
  >ensemble:NPT
  #save_trajectory
```

## Complex Example: Single-Cell Analysis

```
@scanpy|pbmc_10k.h5ad
.preprocessing
  >min_genes:200
  >max_genes:5000
  >max_mt_pct:20
  #filter_genes
  #normalize_total
  >target_sum:10000
  #log1p
  #highly_variable
  >n_top_genes:2000
.dim_reduction
  #pca
  >n_comps:50
  #neighbors
  >n_neighbors:15
  #umap
.clustering
  >method:leiden
  >resolution:0.5,0.8,1.0
  #rank_genes
  >method:wilcoxon
.annotation
  >reference:celltypist
  >model:Immune_All_Low
```

Equivalent JSON would be 3-4x longer.

---

## Token Statistics

```python
import json
from ssn import SSN

ssn = SSN()

data = {
    "task": "molecular_dynamics",
    "input": {"pdb": "protein.pdb", "chain": "A"},
    "parameters": {"steps": 10000, "temperature": 300},
    "output": {"format": "xtc", "verbose": True}
}

json_text = json.dumps(data)
ssn_text = ssn.encode(data)

stats = ssn.token_stats(ssn_text, json_text)
print(f"JSON: {stats['json_tokens_est']} tokens")
print(f"SSN:  {stats['ssn_tokens_est']} tokens")
print(f"Reduction: {stats['reduction_percent']}%")
```

---

## LLM Integration Examples

### With Claude/GPT

```python
# System prompt
system = """
You understand SSN (Sigil Structured Notation):
@ = action, > = key:value, # = flag, . = scope
Respond in SSN format when asked for structured data.
"""

# User request
user = "Plan a docking study for EGFR with gefitinib"

# LLM response (SSN format - fewer tokens)
response = """
@dock|EGFR.pdb|gefitinib.mol2
>center_x:2.5
>center_y:15.3
>center_z:20.1
>box_size:25
#flexible
.preprocessing
  @prot|EGFR.pdb|A
  >remove_water:true
  >add_hydrogens:pH7.4
"""
```

### Parsing LLM Output

```python
from ssn import SSN

ssn = SSN()
result = ssn.parse(llm_response)
# Now you have structured data to work with
```

---

## API Reference

### SSN Class

```python
SSN()                                    # Initialize with default schemas
SSN.register_schema(short, full, args, defaults)  # Add custom schema
SSN.parse(text) -> dict                  # Parse SSN to dict
SSN.encode(data) -> str                  # Encode dict to SSN
SSN.token_stats(ssn, json) -> dict       # Compare token usage
```

### Functional API

```python
from ssn import parse, encode, decode, to_ssn, to_dict

parse(text, ssn=None) -> dict            # Parse SSN text
encode(data, ssn=None) -> str            # Encode to SSN
decode(text, ssn=None) -> dict           # Alias for parse
to_ssn(data, ssn=None) -> str            # Alias for encode
to_dict(text, ssn=None) -> dict          # Alias for decode
```

### Natural Language Converter

```python
from ssn import nl_to_ssn, template, NLToSSN, SSNTemplates

nl_to_ssn(text, **context) -> str        # Convert natural language to SSN
template(name, **kwargs) -> str          # Fill a predefined template

# Available templates: search, explain, compare, code, debug, analyze, summarize, recommend

# Context options for nl_to_ssn:
#   expert_mode=True      -> adds #no_basics
#   include_code=True     -> adds #code_examples
#   default_depth="deep"  -> sets depth level
#   default_language="python" -> for code queries
```

---

## File Operations

```python
from ssn import parse_file
from ssn.decoder import decode_file

# Read .ssn files
data = parse_file("config.ssn")
```

---

## Example Files

The `examples/` directory contains comprehensive examples:

| File | Contents |
|------|----------|
| `general_examples.py` | API requests, ML training, CI/CD, K8s, ETL, chatbots |
| `bioinformatics_examples.py` | MD, docking, ADMET, FEP, virtual screening |
| `ngs_examples.py` | RNA-seq, scRNA-seq, Scanpy, Seurat, ATAC-seq, spatial, variants |
| `llm_prompts_examples.py` | System prompts, resume mods, LLM queries, GPU optimization |
| `nl_conversion_examples.py` | Natural language → SSN conversion demos |

```bash
# Run examples
python examples/general_examples.py
python examples/bioinformatics_examples.py
python examples/ngs_examples.py
python examples/llm_prompts_examples.py
python examples/nl_conversion_examples.py  # NEW: NL to SSN conversion
```

---

## System Prompts in SSN

SSN excels at defining LLM system prompts with minimal tokens:

### Absolute Mode (Direct, No Fluff)
```
@system|absolute
#no_filler;#no_emoji;#no_hype;#no_transitions
>terminate:after_info
>mirror:never
>tone:blunt
```

### Expert Mode (Technical Depth)
```
@system|expert
>level:phd
>code:runnable
#no_basics;#no_disclaimers
>depth:implementation
```

### Query with Context
```
@query|explain
>topic:transformer_architecture
>depth:technical
>focus:attention,positional_encoding
.context
  >background:neural_networks
  >goal:implement_from_scratch
#no_history;#skip_basics
```

---

## Best Practices

1. **Use schemas** for repeated patterns - maximum compression
2. **Inline where possible** - `.out>fmt:csv` instead of separate lines
3. **Omit false booleans** - absence implies false
4. **Use semicolons** for single-line statements - `>a:1;>b:2;#flag`
5. **Register domain schemas** - your field's common operations

---

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Citation

If you use SSN in research:

```bibtex
@software{ssn2024,
  author = {Panda, Pritam Kumar},
  title = {SSN: Sigil Structured Notation for Token-Efficient LLM Communication},
  year = {2024},
  url = {https://github.com/yourusername/ssn}
}
```

---

## Acknowledgments

Developed to reduce costs and latency in AI-driven bioinformatics pipelines.
