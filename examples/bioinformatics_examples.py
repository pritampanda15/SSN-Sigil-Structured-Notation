"""
SSN Examples for Bioinformatics and Drug Design

Demonstrates token-efficient LLM communication patterns.
"""

import json
from ssn import SSN


def example_molecular_dynamics():
    """Molecular dynamics simulation setup."""
    
    ssn = SSN()
    
    # SSN format
    ssn_text = """
@md|system.pdb
>temperature:310
>pressure:1
>steps:1000000
>timestep:2
#save_trajectory
#compute_energy
.output
  >format:xtc
  >frequency:1000
"""
    
    # Equivalent JSON
    json_data = {
        "task": "molecular_dynamics",
        "input": "system.pdb",
        "parameters": {
            "temperature": 310,
            "pressure": 1,
            "steps": 1000000,
            "timestep": 2
        },
        "options": {
            "save_trajectory": True,
            "compute_energy": True
        },
        "output": {
            "format": "xtc",
            "frequency": 1000
        }
    }
    
    json_text = json.dumps(json_data)
    
    print("=== Molecular Dynamics Example ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")
    
    # Parse SSN
    result = ssn.parse(ssn_text)
    print(f"\nParsed result: {result}")


def example_docking():
    """Molecular docking setup."""
    
    ssn = SSN()
    
    ssn_text = """
@dock|GABAA_receptor.pdb|propofol.mol2|12.5|8.3|15.7
>box_size:20
>exhaustiveness:32
>num_modes:9
#flexible
.preprocessing
  #remove_water
  #add_hydrogens
  >ph:7.4
"""
    
    json_data = {
        "task": "docking",
        "receptor": "GABAA_receptor.pdb",
        "ligand": "propofol.mol2",
        "center": {"x": 12.5, "y": 8.3, "z": 15.7},
        "box_size": 20,
        "exhaustiveness": 32,
        "num_modes": 9,
        "flexible": True,
        "preprocessing": {
            "remove_water": True,
            "add_hydrogens": True,
            "ph": 7.4
        }
    }
    
    json_text = json.dumps(json_data)
    
    print("\n=== Docking Example ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")


def example_admet_screening():
    """ADMET property prediction."""
    
    ssn = SSN()
    
    ssn_text = """
@admet|CC(=O)Oc1ccccc1C(=O)O
>models:absorption,distribution,metabolism,excretion,toxicity
#predict_herg
#predict_cyp
#lipinski_filter
.thresholds
  >logp_max:5
  >hbd_max:5
  >hba_max:10
  >mw_max:500
"""
    
    json_data = {
        "task": "admet_prediction",
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "models": ["absorption", "distribution", "metabolism", "excretion", "toxicity"],
        "predictions": {
            "herg": True,
            "cyp": True
        },
        "filters": {
            "lipinski": True
        },
        "thresholds": {
            "logp_max": 5,
            "hbd_max": 5,
            "hba_max": 10,
            "mw_max": 500
        }
    }
    
    json_text = json.dumps(json_data)
    
    print("\n=== ADMET Screening Example ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")


def example_fep_calculation():
    """Free Energy Perturbation setup."""
    
    ssn = SSN()
    
    # Register FEP schema
    ssn.register_schema(
        "fep", 
        "free_energy_perturbation",
        ["ligand_a", "ligand_b", "lambda_windows"],
        {"equilibration": 5000, "production": 50000}
    )
    
    ssn_text = """
@fep|propofol|etomidate|20
>temperature:300
>pressure:1
#soft_core
#restraints
.analysis
  >method:mbar
  >bootstrap:1000
  #overlap_matrix
"""
    
    json_data = {
        "task": "free_energy_perturbation",
        "ligand_a": "propofol",
        "ligand_b": "etomidate",
        "lambda_windows": 20,
        "equilibration": 5000,
        "production": 50000,
        "temperature": 300,
        "pressure": 1,
        "soft_core": True,
        "restraints": True,
        "analysis": {
            "method": "mbar",
            "bootstrap": 1000,
            "overlap_matrix": True
        }
    }
    
    json_text = json.dumps(json_data)
    
    print("\n=== FEP Calculation Example ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")
    
    # Show schema expansion
    result = ssn.parse(ssn_text)
    print(f"\nWith schema expansion: {result}")


def example_virtual_screening_pipeline():
    """Complete virtual screening pipeline."""
    
    ssn = SSN()
    
    ssn_text = """
@pipeline|virtual_screening
.prepare
  @prot|receptor.pdb|A
  >remove:water,ions
  #protonate
.library
  >source:zinc15
  >filters:ro5,pains
  >max_compounds:10000
.docking
  @dock|~receptor|~library|0|0|0
  >exhaustiveness:16
  #gpu
.scoring
  >methods:vina,rf-score,plp
  #consensus
  >top_n:100
.analysis
  @admet|~top_n
  #cluster
  >diversity:0.3
"""
    
    print("\n=== Virtual Screening Pipeline ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    
    result = ssn.parse(ssn_text)
    print(f"\nParsed structure: {json.dumps(result, indent=2)}")


def example_llm_prompt():
    """Example of using SSN in LLM prompts."""
    
    print("\n=== LLM Integration Example ===")
    
    system_prompt = """You understand SSN (Sigil Structured Notation):
- @ = action with |args
- > = key:value
- # = boolean flag
- . = nested scope
Respond in SSN for structured outputs."""
    
    user_prompt = "Design a binding affinity experiment for the N265M GABAA mutation"
    
    # Simulated LLM response in SSN format
    llm_response = """
@experiment|gabaa_n265m_binding
.systems
  >wildtype:GABAA_WT.pdb
  >mutant:GABAA_N265M.pdb
.ligands
  >primary:propofol
  >controls:etomidate,diazepam
.protocol
  @md|~systems|100ns
  >temperature:310
  >replicas:3
  @fep|propofol|propofol|12
  >mutation:N265M
.analysis
  #binding_energy
  #residence_time
  #contact_analysis
  >compare:wt_vs_mutant
"""
    
    print(f"System: {system_prompt}")
    print(f"\nUser: {user_prompt}")
    print(f"\nLLM Response (SSN):\n{llm_response}")
    
    ssn = SSN()
    ssn.register_schema("fep", "free_energy_perturbation", ["ligand_a", "ligand_b", "windows"])
    result = ssn.parse(llm_response)
    print(f"\nParsed for programmatic use: {json.dumps(result, indent=2)}")


def example_encoding():
    """Demonstrate encoding Python dicts to SSN."""
    
    ssn = SSN()
    
    data = {
        "action": "molecular_dynamics",
        "input": "protein.pdb",
        "temperature": 300,
        "pressure": 1,
        "save_trajectory": True,
        "verbose": True,
        "output": {
            "format": "xtc",
            "frequency": 100
        }
    }
    
    print("\n=== Encoding Example ===")
    print(f"\nInput dict:\n{json.dumps(data, indent=2)}")
    
    ssn_output = ssn.encode(data)
    print(f"\nEncoded SSN:\n{ssn_output}")
    
    # Round-trip
    parsed = ssn.parse(ssn_output)
    print(f"\nRound-trip parse:\n{parsed}")


if __name__ == "__main__":
    example_molecular_dynamics()
    example_docking()
    example_admet_screening()
    example_fep_calculation()
    example_virtual_screening_pipeline()
    example_llm_prompt()
    example_encoding()
