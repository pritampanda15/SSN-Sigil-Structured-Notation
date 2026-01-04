"""
SSN Natural Language Conversion Examples

Demonstrates converting plain English queries to token-efficient SSN format.
"""

from ssn import nl_to_ssn, template, NLToSSN


def demo_basic_conversions():
    """Basic natural language to SSN conversions."""
    
    queries = [
        # Search queries
        "Find me top LLM projects in bioinformatics",
        "Search for best protein structure prediction tools",
        "List popular single-cell analysis packages",
        
        # Explain queries
        "Explain how transformers work",
        "What is attention mechanism in deep learning",
        "How does AlphaFold predict protein structures",
        
        # Compare queries
        "Compare PyTorch vs TensorFlow for research",
        "Difference between Scanpy and Seurat",
        "Which is better: STAR or HISAT2 for RNA-seq alignment",
        
        # Code queries
        "Write Python code to parse FASTA files",
        "Implement a function to calculate GC content",
        "Create a script for batch BLAST searches",
        
        # Debug queries
        "Debug CUDA out of memory error in training loop",
        "Fix segmentation fault in my C extension",
        "Error: module not found when importing tensorflow",
        
        # Optimize queries
        "How to optimize GPU memory for training 70B model",
        "Speed up my molecular dynamics simulation",
        "Reduce memory usage in pandas dataframe operations",
        
        # Recommend queries
        "Recommend best tools for single-cell analysis",
        "Suggest top Python libraries for bioinformatics",
        "Best practices for LLM fine-tuning",
    ]
    
    print("=" * 60)
    print("NATURAL LANGUAGE → SSN CONVERSION EXAMPLES")
    print("=" * 60)
    
    for query in queries:
        print(f"\n📝 Input: {query}")
        print(f"📤 SSN Output:")
        ssn = nl_to_ssn(query)
        for line in ssn.split('\n'):
            print(f"   {line}")
        print("-" * 40)


def demo_context_aware_conversion():
    """Conversions with user context."""
    
    print("\n" + "=" * 60)
    print("CONTEXT-AWARE CONVERSIONS")
    print("=" * 60)
    
    query = "Explain attention mechanism"
    
    # Default
    print(f"\n📝 Query: {query}")
    print("\n[Default mode]")
    print(nl_to_ssn(query))
    
    # Expert mode
    print("\n[Expert mode]")
    print(nl_to_ssn(query, expert_mode=True))
    
    # With code examples
    print("\n[With code examples]")
    print(nl_to_ssn(query, include_code=True))
    
    # Both
    print("\n[Expert + code]")
    print(nl_to_ssn(query, expert_mode=True, include_code=True))


def demo_templates():
    """Using pre-built templates."""
    
    print("\n" + "=" * 60)
    print("PRE-BUILT TEMPLATES")
    print("=" * 60)
    
    templates_demo = [
        ("search", {"topic": "crispr_tools", "domain": "genomics", "filter": "top", "count": "10"}),
        ("explain", {"topic": "RLHF", "depth": "technical", "focus": "implementation"}),
        ("compare", {"item_a": "vina", "item_b": "glide", "aspects": "accuracy,speed,cost"}),
        ("code", {"language": "python", "task": "protein_ligand_docking"}),
        ("debug", {"error": "nan_loss", "context": "transformer_training", "env": "A100_GPU"}),
        ("recommend", {"category": "LLM_frameworks", "use_case": "research", "count": "5", "constraints": "open_source"}),
    ]
    
    for template_name, kwargs in templates_demo:
        print(f"\n[Template: {template_name}]")
        print(f"Args: {kwargs}")
        print("Output:")
        ssn = template(template_name, **kwargs)
        for line in ssn.split('\n'):
            print(f"   {line}")


def demo_batch_conversion():
    """Batch convert multiple queries."""
    
    print("\n" + "=" * 60)
    print("BATCH CONVERSION")
    print("=" * 60)
    
    converter = NLToSSN()
    
    queries = [
        "Find top papers on protein design",
        "Explain diffusion models",
        "Compare transformers vs RNNs",
    ]
    
    results = converter.batch_convert(queries, {"expert_mode": True})
    
    for query, ssn in zip(queries, results):
        print(f"\n📝 {query}")
        print(f"📤 {ssn}")


def demo_real_world_scenarios():
    """Real-world usage scenarios."""
    
    print("\n" + "=" * 60)
    print("REAL-WORLD SCENARIOS")
    print("=" * 60)
    
    scenarios = {
        "Research query": "Find recent papers on protein language models for drug discovery",
        "Debug help": "Getting CUDA error when training BERT model on multiple GPUs",
        "Code request": "Write a Python function to calculate binding free energy from MD trajectory",
        "Tool comparison": "Compare Cell Ranger vs STARsolo for single-cell RNA-seq processing",
        "Optimization": "How to reduce memory usage when fine-tuning LLaMA 70B on 8 A100s",
        "Resume help": "Rewrite my resume bullet about machine learning project for industry",
        "Paper summary": "Summarize the key findings of the AlphaFold3 paper",
    }
    
    for scenario, query in scenarios.items():
        print(f"\n🎯 {scenario}")
        print(f"📝 Query: {query}")
        print(f"📤 SSN:")
        ssn = nl_to_ssn(query)
        for line in ssn.split('\n'):
            print(f"   {line}")


def demo_using_ssn_with_llm():
    """Show how to use SSN output with any LLM."""
    
    print("\n" + "=" * 60)
    print("USING SSN WITH LLMs")
    print("=" * 60)
    
    print("""
HOW TO USE SSN WITH ANY LLM (ChatGPT, Claude, etc.):

1. Add this to your system prompt:
   "You understand SSN (Sigil Structured Notation):
    @ = action/command, > = key:value, # = flag, . = nested scope
    Parse SSN queries and respond accordingly."

2. Convert your natural language to SSN:
""")
    
    query = "Find me top 5 protein structure prediction tools for antibody design"
    ssn = nl_to_ssn(query)
    
    print(f"   Natural: {query}")
    print(f"   SSN:\n   {ssn.replace(chr(10), chr(10) + '   ')}")
    
    print("""
3. Send SSN to the LLM - uses fewer tokens than natural language!

4. Token savings example:
""")
    
    natural = "Can you please find me the top 5 protein structure prediction tools that would be good for antibody design? I want them ranked by popularity and with pros/cons for each."
    ssn_version = nl_to_ssn("Find top 5 protein structure prediction tools for antibody design")
    
    natural_tokens = len(natural) / 4  # rough estimate
    ssn_tokens = len(ssn_version) / 4
    
    print(f"   Natural language: ~{int(natural_tokens)} tokens")
    print(f"   SSN format: ~{int(ssn_tokens)} tokens")
    print(f"   Savings: ~{int((1 - ssn_tokens/natural_tokens) * 100)}%")


def main():
    demo_basic_conversions()
    demo_context_aware_conversion()
    demo_templates()
    demo_batch_conversion()
    demo_real_world_scenarios()
    demo_using_ssn_with_llm()
    
    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
