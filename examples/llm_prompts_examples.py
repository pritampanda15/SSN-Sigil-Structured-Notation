"""
SSN Examples for LLM Interactions, System Prompts, and Knowledge Queries

Demonstrates token-efficient notation for AI/LLM workflows.
"""

import json
from ssn import SSN


def example_system_prompt_absolute_mode():
    """System prompt for minimal, direct responses."""
    
    ssn = SSN()
    
    ssn_text = """
@system|absolute_mode
>tone:blunt,directive
>goal:cognitive_rebuilding
>outcome:user_self_sufficiency
#eliminate
  >emojis
  >filler
  >hype
  >soft_asks
  >transitions
  >cta_appendix
#disable
  >engagement_boosting
  >sentiment_softening
  >continuation_bias
  >satisfaction_metrics
#suppress
  >questions
  >offers
  >suggestions
  >motivational_content
.rules
  >mirror_user:never
  >speak_to:cognitive_tier
  >terminate:after_info
  >assume:high_perception
"""
    
    json_data = {
        "system": "absolute_mode",
        "tone": ["blunt", "directive"],
        "goal": "cognitive_rebuilding",
        "outcome": "user_self_sufficiency",
        "eliminate": {
            "emojis": True,
            "filler": True,
            "hype": True,
            "soft_asks": True,
            "transitions": True,
            "cta_appendix": True
        },
        "disable": {
            "engagement_boosting": True,
            "sentiment_softening": True,
            "continuation_bias": True,
            "satisfaction_metrics": True
        },
        "suppress": {
            "questions": True,
            "offers": True,
            "suggestions": True,
            "motivational_content": True
        },
        "rules": {
            "mirror_user": "never",
            "speak_to": "cognitive_tier",
            "terminate": "after_info",
            "assume": "high_perception"
        }
    }
    
    json_text = json.dumps(json_data)
    
    print("=== System Prompt: Absolute Mode ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")


def example_system_prompt_expert_mode():
    """System prompt for domain expert responses."""
    
    ssn = SSN()
    
    ssn_text = """
@system|expert_mode
>domain:computational_biology
>level:phd
>style:technical,precise
.assume
  >python:advanced
  >ml:intermediate
  >stats:advanced
  >linux:proficient
.format
  >code:always_runnable
  >math:latex
  >refs:cite_arxiv
  #no_basics
  #skip_disclaimers
.response
  >depth:implementation_ready
  >include:edge_cases,pitfalls
  >examples:production_grade
"""
    
    json_data = {
        "system": "expert_mode",
        "domain": "computational_biology",
        "level": "phd",
        "style": ["technical", "precise"],
        "assume": {
            "python": "advanced",
            "ml": "intermediate",
            "stats": "advanced",
            "linux": "proficient"
        },
        "format": {
            "code": "always_runnable",
            "math": "latex",
            "refs": "cite_arxiv",
            "no_basics": True,
            "skip_disclaimers": True
        },
        "response": {
            "depth": "implementation_ready",
            "include": ["edge_cases", "pitfalls"],
            "examples": "production_grade"
        }
    }
    
    json_text = json.dumps(json_data)
    
    print("\n=== System Prompt: Expert Mode ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")


def example_system_prompt_tutor_mode():
    """System prompt for educational/tutoring responses."""
    
    ssn = SSN()
    
    ssn_text = """
@system|tutor_mode
>subject:machine_learning
>level:beginner_to_intermediate
>style:socratic
.pedagogy
  #build_intuition
  #use_analogies
  #incremental_complexity
  >examples_before_theory:true
  >check_understanding:true
.format
  >code:commented,step_by_step
  >math:explain_notation
  >visuals:ascii_diagrams
.avoid
  >jargon_without_definition
  >assumed_prerequisites
  >information_overload
"""
    
    json_data = {
        "system": "tutor_mode",
        "subject": "machine_learning",
        "level": "beginner_to_intermediate",
        "style": "socratic",
        "pedagogy": {
            "build_intuition": True,
            "use_analogies": True,
            "incremental_complexity": True,
            "examples_before_theory": True,
            "check_understanding": True
        },
        "format": {
            "code": ["commented", "step_by_step"],
            "math": "explain_notation",
            "visuals": "ascii_diagrams"
        },
        "avoid": {
            "jargon_without_definition": True,
            "assumed_prerequisites": True,
            "information_overload": True
        }
    }
    
    json_text = json.dumps(json_data)
    
    print("\n=== System Prompt: Tutor Mode ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")


def example_query_transformer_architecture():
    """Query about transformer architecture."""
    
    ssn = SSN()
    
    ssn_text = """
@query|explain
>topic:transformer_architecture
>depth:technical
>focus:attention_mechanism,positional_encoding,ffn
.context
  >background:neural_networks,backprop
  >goal:implement_from_scratch
.output
  >include:math,pseudocode,complexity
  >format:structured
  #no_history
  #skip_basics
"""
    
    json_data = {
        "query": "explain",
        "topic": "transformer_architecture",
        "depth": "technical",
        "focus": ["attention_mechanism", "positional_encoding", "ffn"],
        "context": {
            "background": ["neural_networks", "backprop"],
            "goal": "implement_from_scratch"
        },
        "output": {
            "include": ["math", "pseudocode", "complexity"],
            "format": "structured",
            "no_history": True,
            "skip_basics": True
        }
    }
    
    json_text = json.dumps(json_data)
    
    print("\n=== Query: Transformer Architecture ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")


def example_query_gpu_optimization():
    """Query about GPU optimization and parallelization."""
    
    ssn = SSN()
    
    ssn_text = """
@query|optimize
>topic:gpu_parallelization
>framework:pytorch
>hardware:A100,multi_gpu
.techniques
  >data_parallel:DDP
  >model_parallel:FSDP,tensor_parallel
  >mixed_precision:bf16,fp16
  >gradient:checkpointing,accumulation
.constraints
  >model_size:70B_params
  >gpu_memory:80GB
  >num_gpus:8
.output
  >include:code,memory_calc,benchmarks
  >compare:naive_vs_optimized
  #production_ready
"""
    
    json_data = {
        "query": "optimize",
        "topic": "gpu_parallelization",
        "framework": "pytorch",
        "hardware": ["A100", "multi_gpu"],
        "techniques": {
            "data_parallel": "DDP",
            "model_parallel": ["FSDP", "tensor_parallel"],
            "mixed_precision": ["bf16", "fp16"],
            "gradient": ["checkpointing", "accumulation"]
        },
        "constraints": {
            "model_size": "70B_params",
            "gpu_memory": "80GB",
            "num_gpus": 8
        },
        "output": {
            "include": ["code", "memory_calc", "benchmarks"],
            "compare": "naive_vs_optimized",
            "production_ready": True
        }
    }
    
    json_text = json.dumps(json_data)
    
    print("\n=== Query: GPU Optimization ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")


def example_query_llm_training():
    """Query about LLM training pipeline."""
    
    ssn = SSN()
    
    ssn_text = """
@query|guide
>topic:llm_training_pipeline
>scale:7B_to_70B
>budget:limited_compute
.stages
  >pretrain:curriculum,data_mix
  >sft:instruction_tuning
  >alignment:dpo,rlhf
.infrastructure
  >framework:pytorch,deepspeed
  >cluster:slurm
  >storage:distributed_fs
.focus
  >data_quality:filtering,dedup
  >efficiency:flash_attention,rope
  >evaluation:benchmarks,human_eval
.output
  #timeline
  #cost_estimate
  #failure_modes
"""
    
    json_data = {
        "query": "guide",
        "topic": "llm_training_pipeline",
        "scale": "7B_to_70B",
        "budget": "limited_compute",
        "stages": {
            "pretrain": ["curriculum", "data_mix"],
            "sft": "instruction_tuning",
            "alignment": ["dpo", "rlhf"]
        },
        "infrastructure": {
            "framework": ["pytorch", "deepspeed"],
            "cluster": "slurm",
            "storage": "distributed_fs"
        },
        "focus": {
            "data_quality": ["filtering", "dedup"],
            "efficiency": ["flash_attention", "rope"],
            "evaluation": ["benchmarks", "human_eval"]
        },
        "output": {
            "timeline": True,
            "cost_estimate": True,
            "failure_modes": True
        }
    }
    
    json_text = json.dumps(json_data)
    
    print("\n=== Query: LLM Training Pipeline ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")


def example_resume_modification():
    """Resume modification request."""
    
    ssn = SSN()
    
    ssn_text = """
@task|resume_modify
>target_role:ML_Research_Scientist
>company:OpenAI,Anthropic,DeepMind
>level:senior
.current_profile
  >title:Postdoctoral_Scholar
  >field:computational_biology
  >years:8
  >pubs:50
  >citations:5000
  >funding:8M
.emphasize
  >skills:pytorch,transformers,distributed_training
  >projects:protein_design,diffusion_models
  >impact:open_source,publications
.modify
  >summary:rewrite_for_industry
  >experience:quantify_achievements
  >skills:prioritize_ml_infra
  #remove_academic_jargon
  #add_metrics
  #ats_optimize
.format
  >length:2_pages
  >style:impact_driven
  >sections:summary,skills,experience,publications
"""
    
    json_data = {
        "task": "resume_modify",
        "target_role": "ML_Research_Scientist",
        "company": ["OpenAI", "Anthropic", "DeepMind"],
        "level": "senior",
        "current_profile": {
            "title": "Postdoctoral_Scholar",
            "field": "computational_biology",
            "years": 8,
            "pubs": 50,
            "citations": 5000,
            "funding": "8M"
        },
        "emphasize": {
            "skills": ["pytorch", "transformers", "distributed_training"],
            "projects": ["protein_design", "diffusion_models"],
            "impact": ["open_source", "publications"]
        },
        "modify": {
            "summary": "rewrite_for_industry",
            "experience": "quantify_achievements",
            "skills": "prioritize_ml_infra",
            "remove_academic_jargon": True,
            "add_metrics": True,
            "ats_optimize": True
        },
        "format": {
            "length": "2_pages",
            "style": "impact_driven",
            "sections": ["summary", "skills", "experience", "publications"]
        }
    }
    
    json_text = json.dumps(json_data)
    
    print("\n=== Resume Modification ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")


def example_resume_bullet_rewrite():
    """Rewrite resume bullets for impact."""
    
    ssn = SSN()
    
    ssn_text = """
@task|rewrite_bullets
>style:STAR,quantified
>tone:action_oriented
.input
  >bullet:Worked on protein structure prediction using deep learning
  >context:lead_project,2_years,production
.requirements
  #start_with_verb
  #include_metrics
  #show_impact
  >max_words:25
.output
  >variations:3
  >rank_by:impact,specificity
"""
    
    json_data = {
        "task": "rewrite_bullets",
        "style": ["STAR", "quantified"],
        "tone": "action_oriented",
        "input": {
            "bullet": "Worked on protein structure prediction using deep learning",
            "context": ["lead_project", "2_years", "production"]
        },
        "requirements": {
            "start_with_verb": True,
            "include_metrics": True,
            "show_impact": True,
            "max_words": 25
        },
        "output": {
            "variations": 3,
            "rank_by": ["impact", "specificity"]
        }
    }
    
    json_text = json.dumps(json_data)
    
    print("\n=== Resume Bullet Rewrite ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")


def example_cover_letter():
    """Cover letter generation request."""
    
    ssn = SSN()
    
    ssn_text = """
@task|cover_letter
>role:Research_Scientist
>company:Anthropic
>team:Alignment
.match
  >their_needs:interpretability,safety,scaling
  >my_strengths:protein_ml,physics_background,publication_record
.structure
  >hook:specific_project_interest
  >body:skill_evidence_mapping
  >close:unique_value_prop
.tone
  >formal:moderate
  >enthusiasm:genuine_not_excessive
  #no_generic_phrases
  #specific_examples
.constraints
  >length:400_words
  >paragraphs:4
"""
    
    json_data = {
        "task": "cover_letter",
        "role": "Research_Scientist",
        "company": "Anthropic",
        "team": "Alignment",
        "match": {
            "their_needs": ["interpretability", "safety", "scaling"],
            "my_strengths": ["protein_ml", "physics_background", "publication_record"]
        },
        "structure": {
            "hook": "specific_project_interest",
            "body": "skill_evidence_mapping",
            "close": "unique_value_prop"
        },
        "tone": {
            "formal": "moderate",
            "enthusiasm": "genuine_not_excessive",
            "no_generic_phrases": True,
            "specific_examples": True
        },
        "constraints": {
            "length": "400_words",
            "paragraphs": 4
        }
    }
    
    json_text = json.dumps(json_data)
    
    print("\n=== Cover Letter Generation ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")


def example_query_attention_mechanism():
    """Deep dive query on attention mechanisms."""
    
    ssn = SSN()
    
    ssn_text = """
@query|deep_dive
>topic:attention_mechanisms
>variants:self,cross,multi_head,flash,linear
.explain
  >math:QKV_formulation,softmax,scaling
  >complexity:time,space,io
  >implementation:pytorch,triton
.compare
  >standard_vs_flash:memory,speed,accuracy
  >quadratic_vs_linear:tradeoffs
.practical
  >when_to_use:each_variant
  >common_bugs:numerical_stability,masking
  >optimization:kv_cache,chunking
#code_examples
#benchmarks
"""
    
    json_data = {
        "query": "deep_dive",
        "topic": "attention_mechanisms",
        "variants": ["self", "cross", "multi_head", "flash", "linear"],
        "explain": {
            "math": ["QKV_formulation", "softmax", "scaling"],
            "complexity": ["time", "space", "io"],
            "implementation": ["pytorch", "triton"]
        },
        "compare": {
            "standard_vs_flash": ["memory", "speed", "accuracy"],
            "quadratic_vs_linear": "tradeoffs"
        },
        "practical": {
            "when_to_use": "each_variant",
            "common_bugs": ["numerical_stability", "masking"],
            "optimization": ["kv_cache", "chunking"]
        },
        "code_examples": True,
        "benchmarks": True
    }
    
    json_text = json.dumps(json_data)
    
    print("\n=== Query: Attention Mechanisms ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")


def example_code_review_request():
    """Code review request specification."""
    
    ssn = SSN()
    
    ssn_text = """
@task|code_review
>language:python
>domain:ml_training
>priority:performance,correctness
.focus
  >memory_leaks:tensor_accumulation
  >gpu_util:idle_time,transfers
  >numerical:precision,overflow
  >distributed:deadlocks,sync
.style
  #skip_formatting
  #skip_naming
  >only:logic,performance,bugs
.output
  >severity:critical,major,minor
  >include:fix_suggestion,explanation
  >format:inline_comments
"""
    
    json_data = {
        "task": "code_review",
        "language": "python",
        "domain": "ml_training",
        "priority": ["performance", "correctness"],
        "focus": {
            "memory_leaks": "tensor_accumulation",
            "gpu_util": ["idle_time", "transfers"],
            "numerical": ["precision", "overflow"],
            "distributed": ["deadlocks", "sync"]
        },
        "style": {
            "skip_formatting": True,
            "skip_naming": True,
            "only": ["logic", "performance", "bugs"]
        },
        "output": {
            "severity": ["critical", "major", "minor"],
            "include": ["fix_suggestion", "explanation"],
            "format": "inline_comments"
        }
    }
    
    json_text = json.dumps(json_data)
    
    print("\n=== Code Review Request ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")


def example_paper_review():
    """Paper/manuscript review request."""
    
    ssn = SSN()
    
    ssn_text = """
@task|paper_review
>type:methods_paper
>venue:nature_methods
>role:reviewer
.evaluate
  >novelty:1-5
  >technical:soundness,reproducibility
  >clarity:writing,figures
  >impact:field_advancement
.focus
  >methods:statistical_validity,baselines
  >claims:supported_by_evidence
  >limitations:acknowledged
.output
  >summary:2_sentences
  >strengths:bullet_list
  >weaknesses:bullet_list
  >questions:for_authors
  >recommendation:accept,minor,major,reject
"""
    
    json_data = {
        "task": "paper_review",
        "type": "methods_paper",
        "venue": "nature_methods",
        "role": "reviewer",
        "evaluate": {
            "novelty": "1-5",
            "technical": ["soundness", "reproducibility"],
            "clarity": ["writing", "figures"],
            "impact": "field_advancement"
        },
        "focus": {
            "methods": ["statistical_validity", "baselines"],
            "claims": "supported_by_evidence",
            "limitations": "acknowledged"
        },
        "output": {
            "summary": "2_sentences",
            "strengths": "bullet_list",
            "weaknesses": "bullet_list",
            "questions": "for_authors",
            "recommendation": ["accept", "minor", "major", "reject"]
        }
    }
    
    json_text = json.dumps(json_data)
    
    print("\n=== Paper Review Request ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")


def example_debug_request():
    """Debugging request specification."""
    
    ssn = SSN()
    
    ssn_text = """
@task|debug
>error:CUDA_out_of_memory
>context:training_loop
>model:llama_7b
.environment
  >gpu:A100_40GB
  >pytorch:2.1
  >batch_size:8
  >seq_len:4096
.tried
  >gradient_checkpointing:enabled
  >mixed_precision:bf16
  >batch_size:reduced_to_4
.need
  >root_cause:memory_breakdown
  >solution:actionable_steps
  >prevention:future_guidelines
#no_generic_advice
#specific_to_config
"""
    
    json_data = {
        "task": "debug",
        "error": "CUDA_out_of_memory",
        "context": "training_loop",
        "model": "llama_7b",
        "environment": {
            "gpu": "A100_40GB",
            "pytorch": "2.1",
            "batch_size": 8,
            "seq_len": 4096
        },
        "tried": {
            "gradient_checkpointing": "enabled",
            "mixed_precision": "bf16",
            "batch_size": "reduced_to_4"
        },
        "need": {
            "root_cause": "memory_breakdown",
            "solution": "actionable_steps",
            "prevention": "future_guidelines"
        },
        "no_generic_advice": True,
        "specific_to_config": True
    }
    
    json_text = json.dumps(json_data)
    
    print("\n=== Debug Request ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")


def example_linkedin_post():
    """LinkedIn post generation request."""
    
    ssn = SSN()
    
    ssn_text = """
@task|linkedin_post
>topic:new_open_source_tool
>tool:SSN
>audience:ml_engineers,researchers
.content
  >hook:problem_statement
  >solution:token_reduction
  >proof:benchmarks
  >cta:github_link
.style
  >tone:professional,technical
  >length:200_words
  #no_hashtag_spam
  #no_emoji_overuse
  >structure:short_paragraphs
.include
  >code_snippet:before_after
  >metrics:percent_reduction
"""
    
    json_data = {
        "task": "linkedin_post",
        "topic": "new_open_source_tool",
        "tool": "SSN",
        "audience": ["ml_engineers", "researchers"],
        "content": {
            "hook": "problem_statement",
            "solution": "token_reduction",
            "proof": "benchmarks",
            "cta": "github_link"
        },
        "style": {
            "tone": ["professional", "technical"],
            "length": "200_words",
            "no_hashtag_spam": True,
            "no_emoji_overuse": True,
            "structure": "short_paragraphs"
        },
        "include": {
            "code_snippet": "before_after",
            "metrics": "percent_reduction"
        }
    }
    
    json_text = json.dumps(json_data)
    
    print("\n=== LinkedIn Post Generation ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")


# Pre-built system prompts as SSN constants
SYSTEM_PROMPTS = {
    "absolute": """@system|absolute
#no_filler;#no_emoji;#no_hype;#no_transitions
>terminate:after_info
>mirror:never
>tone:blunt""",
    
    "expert": """@system|expert
>level:phd
>code:runnable
#no_basics;#no_disclaimers
>depth:implementation""",
    
    "tutor": """@system|tutor
>style:socratic
#analogies;#incremental
>check:understanding""",
    
    "reviewer": """@system|reviewer
>focus:technical_correctness
#cite_evidence
>tone:constructive_critical""",
    
    "debugger": """@system|debugger
>approach:systematic
#root_cause;#minimal_fix
>output:code_diff""",
}


def print_preset_prompts():
    """Display preset system prompts."""
    print("\n=== Preset System Prompts ===")
    for name, prompt in SYSTEM_PROMPTS.items():
        print(f"\n[{name}]")
        print(prompt)


if __name__ == "__main__":
    example_system_prompt_absolute_mode()
    example_system_prompt_expert_mode()
    example_system_prompt_tutor_mode()
    example_query_transformer_architecture()
    example_query_gpu_optimization()
    example_query_llm_training()
    example_resume_modification()
    example_resume_bullet_rewrite()
    example_cover_letter()
    example_query_attention_mechanism()
    example_code_review_request()
    example_paper_review()
    example_debug_request()
    example_linkedin_post()
    print_preset_prompts()
