"""
SSN Natural Language Converter

Converts natural language queries to SSN format for token-efficient LLM communication.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class QueryPattern:
    """Pattern for matching natural language queries."""
    keywords: List[str]
    action: str
    template: str
    extractors: Dict[str, str]  # regex patterns for extracting values


class NLToSSN:
    """
    Convert natural language to SSN format.
    
    Example:
        >>> converter = NLToSSN()
        >>> ssn = converter.convert("Find me top LLM projects in bioinformatics")
        >>> print(ssn)
        @search|llm_projects
        >domain:bioinformatics
        >filter:top
        #ranked
    """
    
    def __init__(self):
        self.patterns = self._build_patterns()
        self.domain_keywords = self._build_domain_keywords()
    
    def _build_patterns(self) -> List[QueryPattern]:
        """Build query pattern matchers."""
        return [
            # Search/Find patterns
            QueryPattern(
                keywords=["find", "search", "look for", "get", "show", "list"],
                action="search",
                template="@search|{topic}\n>domain:{domain}\n>filter:{filter}\n#ranked",
                extractors={
                    "topic": r"(?:find|search|get|show|list)(?:\s+me)?\s+(?:the\s+)?(?:top\s+)?(.+?)(?:\s+in\s+|\s+for\s+|\s+about\s+|$)",
                    "domain": r"(?:in|for|about|related to)\s+(\w+(?:\s+\w+)?)",
                    "filter": r"(top|best|latest|recent|popular|trending)"
                }
            ),
            # Explain/Describe patterns
            QueryPattern(
                keywords=["explain", "describe", "what is", "what are", "how does", "how do"],
                action="explain",
                template="@query|explain\n>topic:{topic}\n>depth:{depth}\n>focus:{focus}",
                extractors={
                    "topic": r"(?:explain|describe|what (?:is|are)|how (?:does|do))\s+(?:the\s+)?(.+?)(?:\?|$)",
                    "depth": r"(simple|basic|detailed|technical|advanced)",
                    "focus": r"(?:focus on|specifically|especially)\s+(.+?)(?:\?|$)"
                }
            ),
            # Compare patterns
            QueryPattern(
                keywords=["compare", "difference", "vs", "versus", "better"],
                action="compare",
                template="@compare|{item_a}|{item_b}\n>aspects:{aspects}\n#pros_cons",
                extractors={
                    "item_a": r"(?:compare|difference between)\s+(\w+(?:\s+\w+)?)",
                    "item_b": r"(?:and|vs|versus|with|to)\s+(\w+(?:\s+\w+)?)",
                    "aspects": r"(?:in terms of|regarding|for)\s+(.+?)(?:\?|$)"
                }
            ),
            # Create/Generate patterns
            QueryPattern(
                keywords=["create", "generate", "make", "write", "build"],
                action="create",
                template="@create|{type}\n>topic:{topic}\n>style:{style}\n>format:{format}",
                extractors={
                    "type": r"(?:create|generate|make|write|build)\s+(?:a\s+)?(\w+)",
                    "topic": r"(?:about|for|on)\s+(.+?)(?:\?|$)",
                    "style": r"(formal|casual|technical|simple)",
                    "format": r"(?:in|as)\s+(\w+)\s+format"
                }
            ),
            # Analyze patterns
            QueryPattern(
                keywords=["analyze", "analysis", "evaluate", "assess", "review"],
                action="analyze",
                template="@analyze|{target}\n>aspects:{aspects}\n>depth:{depth}\n#actionable",
                extractors={
                    "target": r"(?:analyze|analysis of|evaluate|assess|review)\s+(?:the\s+)?(.+?)(?:\s+and|\s+for|$)",
                    "aspects": r"(?:for|looking at|considering)\s+(.+?)(?:\?|$)",
                    "depth": r"(quick|thorough|detailed|comprehensive)"
                }
            ),
            # Optimize patterns
            QueryPattern(
                keywords=["optimize", "improve", "speed up", "make faster", "reduce"],
                action="optimize",
                template="@optimize|{target}\n>goal:{goal}\n>constraints:{constraints}\n#benchmarks",
                extractors={
                    "target": r"(?:optimize|improve|speed up|make faster)\s+(?:the\s+)?(.+?)(?:\s+for|\s+to|$)",
                    "goal": r"(?:for|to achieve|targeting)\s+(.+?)(?:\?|$)",
                    "constraints": r"(?:with|given|under)\s+(.+?)(?:\?|$)"
                }
            ),
            # Debug patterns
            QueryPattern(
                keywords=["debug", "fix", "error", "bug", "issue", "problem", "not working"],
                action="debug",
                template="@debug|{error}\n>context:{context}\n>tried:{tried}\n#root_cause\n#fix",
                extractors={
                    "error": r"(?:debug|fix|error|bug|issue|problem)(?:\s+with)?\s*:?\s*(.+?)(?:\s+in|\s+when|$)",
                    "context": r"(?:in|when|while)\s+(.+?)(?:\?|$)",
                    "tried": r"(?:tried|already did)\s+(.+?)(?:\?|$)"
                }
            ),
            # Summarize patterns
            QueryPattern(
                keywords=["summarize", "summary", "tldr", "brief", "overview"],
                action="summarize",
                template="@summarize|{content}\n>length:{length}\n>focus:{focus}\n#key_points",
                extractors={
                    "content": r"(?:summarize|summary of|tldr|brief|overview of)\s+(?:the\s+)?(.+?)(?:\s+in|$)",
                    "length": r"(\d+)\s*(?:words|sentences|paragraphs)",
                    "focus": r"(?:focus on|highlighting|emphasizing)\s+(.+?)(?:\?|$)"
                }
            ),
            # Recommend patterns
            QueryPattern(
                keywords=["recommend", "suggest", "best", "top", "which", "should i"],
                action="recommend",
                template="@recommend|{category}\n>for:{use_case}\n>constraints:{constraints}\n>count:{count}\n#ranked",
                extractors={
                    "category": r"(?:recommend|suggest|best|top)\s+(\w+(?:\s+\w+)?)",
                    "use_case": r"(?:for|to)\s+(.+?)(?:\?|$)",
                    "constraints": r"(?:with|under|given)\s+(.+?)(?:\?|$)",
                    "count": r"(\d+)"
                }
            ),
            # Code patterns
            QueryPattern(
                keywords=["code", "implement", "function", "script", "program"],
                action="code",
                template="@code|{language}\n>task:{task}\n>style:{style}\n#runnable\n#commented",
                extractors={
                    "language": r"(?:in|using)\s+(python|javascript|rust|go|java|c\+\+)",
                    "task": r"(?:code|implement|function|script)\s+(?:to|for|that)\s+(.+?)(?:\s+in|$)",
                    "style": r"(clean|optimized|readable|production)"
                }
            ),
        ]
    
    def _build_domain_keywords(self) -> Dict[str, List[str]]:
        """Build domain keyword mappings."""
        return {
            "bioinformatics": ["bio", "genomics", "protein", "sequence", "dna", "rna", "molecular"],
            "machine_learning": ["ml", "ai", "neural", "deep learning", "model", "training"],
            "nlp": ["language", "text", "nlp", "llm", "transformer", "gpt", "bert"],
            "computer_vision": ["image", "vision", "cv", "detection", "recognition"],
            "data_science": ["data", "analytics", "statistics", "visualization"],
            "web_development": ["web", "frontend", "backend", "api", "react", "node"],
            "devops": ["devops", "ci/cd", "docker", "kubernetes", "deployment"],
            "security": ["security", "encryption", "auth", "vulnerability"],
        }
    
    def _detect_domain(self, text: str) -> str:
        """Detect domain from text."""
        text_lower = text.lower()
        for domain, keywords in self.domain_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return domain
        return "general"
    
    def _extract_value(self, text: str, pattern: str, default: str = "") -> str:
        """Extract value using regex pattern."""
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else default
    
    def _match_pattern(self, text: str) -> Optional[Tuple[QueryPattern, Dict[str, str]]]:
        """Match text against patterns and extract values."""
        text_lower = text.lower()
        
        for pattern in self.patterns:
            if any(kw in text_lower for kw in pattern.keywords):
                extracted = {}
                for key, regex in pattern.extractors.items():
                    extracted[key] = self._extract_value(text, regex)
                return pattern, extracted
        
        return None
    
    def convert(self, natural_language: str, context: Dict = None) -> str:
        """
        Convert natural language to SSN format.
        
        Args:
            natural_language: Natural language query
            context: Optional context dict with user preferences
        
        Returns:
            SSN formatted string
        """
        text = natural_language.strip()
        context = context or {}
        
        # Match against patterns
        match_result = self._match_pattern(text)
        
        if match_result:
            pattern, extracted = match_result
            
            # Fill in defaults
            if not extracted.get("domain"):
                extracted["domain"] = self._detect_domain(text)
            if not extracted.get("depth"):
                extracted["depth"] = context.get("default_depth", "moderate")
            if not extracted.get("filter"):
                extracted["filter"] = "relevant"
            if not extracted.get("style"):
                extracted["style"] = context.get("default_style", "technical")
            if not extracted.get("format"):
                extracted["format"] = "structured"
            if not extracted.get("count"):
                extracted["count"] = "5"
            if not extracted.get("length"):
                extracted["length"] = "concise"
            if not extracted.get("aspects"):
                extracted["aspects"] = "all"
            if not extracted.get("focus"):
                extracted["focus"] = "key_concepts"
            if not extracted.get("goal"):
                extracted["goal"] = "improvement"
            if not extracted.get("constraints"):
                extracted["constraints"] = "none"
            if not extracted.get("context"):
                extracted["context"] = "general"
            if not extracted.get("tried"):
                extracted["tried"] = "none"
            if not extracted.get("use_case"):
                extracted["use_case"] = "general"
            if not extracted.get("language"):
                extracted["language"] = context.get("default_language", "python")
            if not extracted.get("task"):
                extracted["task"] = text
            
            # Clean up topic extraction
            if not extracted.get("topic"):
                # Fallback: extract main noun phrases
                extracted["topic"] = self._extract_topic_fallback(text)
            
            # Build SSN from template
            ssn = pattern.template
            for key, value in extracted.items():
                if value:
                    ssn = ssn.replace(f"{{{key}}}", value.replace(" ", "_"))
            
            # Add context if provided
            if context.get("expert_mode"):
                ssn += "\n#no_basics"
            if context.get("include_code"):
                ssn += "\n#code_examples"
            
            return ssn
        
        # Fallback: generic query format
        return self._fallback_conversion(text, context)
    
    def _extract_topic_fallback(self, text: str) -> str:
        """Fallback topic extraction."""
        # Remove common words and extract key terms
        stopwords = {"find", "me", "the", "a", "an", "in", "for", "about", "can", "you", 
                     "please", "i", "want", "need", "looking", "search", "top", "best"}
        words = text.lower().split()
        key_words = [w for w in words if w not in stopwords and len(w) > 2]
        return "_".join(key_words[:3]) if key_words else "query"
    
    def _fallback_conversion(self, text: str, context: Dict) -> str:
        """Fallback conversion for unmatched patterns."""
        topic = self._extract_topic_fallback(text)
        domain = self._detect_domain(text)
        
        ssn = f"""@query|general
>input:{topic}
>domain:{domain}
>output:structured"""
        
        if context.get("expert_mode"):
            ssn += "\n#technical"
        
        return ssn
    
    def batch_convert(self, queries: List[str], context: Dict = None) -> List[str]:
        """Convert multiple queries to SSN."""
        return [self.convert(q, context) for q in queries]


class SSNTemplates:
    """Pre-built SSN templates for common tasks."""
    
    SEARCH = """@search|{topic}
>domain:{domain}
>filter:{filter}
>count:{count}
#ranked"""
    
    EXPLAIN = """@query|explain
>topic:{topic}
>depth:{depth}
>focus:{focus}
.output
  >format:structured
  >include:examples
  #no_fluff"""
    
    COMPARE = """@compare|{item_a}|{item_b}
>aspects:{aspects}
>depth:detailed
#pros_cons
#recommendation"""
    
    CODE = """@code|{language}
>task:{task}
>style:production
#runnable
#commented
#error_handling"""
    
    DEBUG = """@debug|{error}
>context:{context}
>environment:{env}
#root_cause
#minimal_fix
#prevention"""
    
    ANALYZE = """@analyze|{target}
>aspects:{aspects}
>depth:comprehensive
#actionable
#metrics"""
    
    SUMMARIZE = """@summarize|{content}
>length:{length}
>focus:{focus}
#key_points
#structured"""
    
    RECOMMEND = """@recommend|{category}
>for:{use_case}
>count:{count}
>constraints:{constraints}
#ranked
#pros_cons"""
    
    @classmethod
    def fill(cls, template_name: str, **kwargs) -> str:
        """Fill a template with values."""
        template = getattr(cls, template_name.upper(), None)
        if not template:
            raise ValueError(f"Unknown template: {template_name}")
        
        result = template
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", str(value))
        
        return result


# Convenience functions
def nl_to_ssn(text: str, **context) -> str:
    """Convert natural language to SSN."""
    converter = NLToSSN()
    return converter.convert(text, context)


def template(name: str, **kwargs) -> str:
    """Fill an SSN template."""
    return SSNTemplates.fill(name, **kwargs)


# Example conversions for reference
EXAMPLE_CONVERSIONS = {
    "Find me top LLM projects in bioinformatics": """@search|llm_projects
>domain:bioinformatics
>filter:top
>count:10
#ranked""",

    "Explain how transformers work": """@query|explain
>topic:transformers
>depth:technical
>focus:attention_mechanism,architecture
.output
  >format:structured
  >include:math,diagrams""",

    "Compare PyTorch vs TensorFlow for research": """@compare|pytorch|tensorflow
>aspects:research,flexibility,ecosystem
>for:academic_research
#pros_cons
#recommendation""",

    "How to optimize GPU memory for training 70B model": """@optimize|gpu_memory
>target:70B_model_training
>techniques:gradient_checkpointing,mixed_precision,FSDP
>constraints:A100_80GB
#benchmarks
#code""",

    "Debug CUDA out of memory error in training loop": """@debug|cuda_oom
>context:training_loop
>symptoms:memory_grows_each_step
#root_cause
#minimal_fix
>output:code_diff""",

    "Write a Python function to calculate protein binding affinity": """@code|python
>task:protein_binding_affinity_calculation
>style:production
>include:type_hints,docstring
#runnable
#tested""",

    "Summarize this paper about AlphaFold": """@summarize|paper
>topic:alphafold
>length:300_words
>focus:methods,results,impact
#key_points
#structured""",

    "Recommend best tools for single-cell analysis": """@recommend|tools
>category:single_cell_analysis
>for:10x_genomics_data
>count:5
#ranked
#comparison_table""",
}


if __name__ == "__main__":
    converter = NLToSSN()
    
    print("=== Natural Language to SSN Converter ===\n")
    
    for nl, expected in EXAMPLE_CONVERSIONS.items():
        print(f"Input: {nl}")
        result = converter.convert(nl)
        print(f"Output:\n{result}")
        print("-" * 50)
