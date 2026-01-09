"""
SSN Natural Language Converter

Converts natural language queries to SSN format for token-efficient LLM communication.
Supports both simple queries and complex structured prompts (markdown-based).
"""

import re
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum


class PromptType(Enum):
    """Type of input prompt."""
    SIMPLE_QUERY = "simple"
    STRUCTURED_PROMPT = "structured"


@dataclass
class ParsedSection:
    """A parsed section from a structured prompt."""
    level: int  # Header level (1-6) or 0 for root
    title: str
    content: List[str]  # Raw content lines
    subsections: List['ParsedSection'] = field(default_factory=list)
    bullets: List[str] = field(default_factory=list)
    numbered_items: List[str] = field(default_factory=list)
    key_values: Dict[str, str] = field(default_factory=dict)


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


class StructuredPromptConverter:
    """
    Convert complex structured prompts (markdown-based) to SSN format.

    Handles:
    - Markdown headers (# ## ###)
    - Bullet lists (* -)
    - Numbered lists (1. 2. 3.)
    - Bold/emphasis for key terms
    - Code blocks
    - Nested structures

    Example:
        >>> converter = StructuredPromptConverter()
        >>> ssn = converter.convert('''
        ... # Task: Build a RAG System
        ...
        ... ## Requirements
        ... * Support multi-hop reasoning
        ... * Handle uncertainty
        ...
        ... ## Data Sources
        ... * PubMed
        ... * DrugBank
        ... ''')
    """

    # Role keywords for detecting persona/role definitions
    ROLE_KEYWORDS = [
        "you are", "act as", "behave as", "role:", "persona:",
        "senior", "expert", "specialist", "architect", "engineer"
    ]

    # Task keywords for detecting main objectives
    TASK_KEYWORDS = [
        "task is to", "your task", "objective", "goal", "design",
        "implement", "create", "build", "develop", "analyze"
    ]

    # Requirement keywords
    REQUIREMENT_KEYWORDS = [
        "must", "should", "require", "need", "essential", "critical",
        "important", "necessary", "mandatory"
    ]

    # Domain keyword mappings
    DOMAIN_KEYWORDS = {
        "drug_discovery": ["drug", "pharmaceutical", "compound", "ligand", "admet", "binding"],
        "bioinformatics": ["protein", "gene", "genomic", "sequence", "molecular", "biological"],
        "machine_learning": ["model", "training", "neural", "deep learning", "ml", "ai"],
        "nlp": ["language", "text", "embedding", "transformer", "llm", "rag"],
        "data_engineering": ["pipeline", "etl", "database", "indexing", "retrieval"],
        "scientific": ["research", "hypothesis", "evidence", "literature", "citation"],
    }

    def __init__(self):
        self.indent_size = 2

    def _strip_blockquotes(self, text: str) -> str:
        """Remove blockquote markers (>) from text."""
        lines = text.split('\n')
        stripped = []
        for line in lines:
            # Remove leading > and optional space
            if line.startswith('> '):
                stripped.append(line[2:])
            elif line.startswith('>'):
                stripped.append(line[1:])
            else:
                stripped.append(line)
        return '\n'.join(stripped)

    def detect_prompt_type(self, text: str) -> PromptType:
        """Detect whether input is a simple query or structured prompt."""
        # First strip blockquotes if present
        if text.strip().startswith('>'):
            text = self._strip_blockquotes(text)

        lines = text.strip().split('\n')

        # Indicators of structured prompt
        has_headers = any(re.match(r'^\s*#{1,6}\s+', line) for line in lines)
        has_bullets = sum(1 for line in lines if re.match(r'^\s*[\*\-]\s+', line)) > 2
        has_numbered = sum(1 for line in lines if re.match(r'^\s*\d+\.\s+', line)) > 2
        has_sections = text.count('---') >= 2
        line_count = len([l for l in lines if l.strip()])

        # If multiple structural elements, it's structured
        if has_headers and (has_bullets or has_numbered or has_sections):
            return PromptType.STRUCTURED_PROMPT
        if line_count > 10 and (has_bullets or has_numbered):
            return PromptType.STRUCTURED_PROMPT
        if has_headers and line_count > 5:
            return PromptType.STRUCTURED_PROMPT

        return PromptType.SIMPLE_QUERY

    def parse_markdown(self, text: str) -> ParsedSection:
        """Parse markdown text into structured sections."""
        # Strip blockquotes if present
        if text.strip().startswith('>'):
            text = self._strip_blockquotes(text)

        lines = text.strip().split('\n')
        root = ParsedSection(level=0, title="root", content=[])

        current_section = root
        section_stack = [root]

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Skip horizontal rules
            if stripped == '---' or stripped == '***':
                i += 1
                continue

            # Check for headers
            header_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()

                # Remove numbering like "1." or "1:" from title
                title = re.sub(r'^\d+[\.\:]\s*', '', title)

                new_section = ParsedSection(level=level, title=title, content=[])

                # Find parent section
                while section_stack and section_stack[-1].level >= level:
                    section_stack.pop()

                if section_stack:
                    section_stack[-1].subsections.append(new_section)

                section_stack.append(new_section)
                current_section = new_section
                i += 1
                continue

            # Check for bullet points
            bullet_match = re.match(r'^\s*[\*\-]\s+(.+)$', stripped)
            if bullet_match:
                bullet_text = bullet_match.group(1).strip()
                # Handle nested bullets (check indentation)
                current_section.bullets.append(bullet_text)
                i += 1
                continue

            # Check for numbered items
            numbered_match = re.match(r'^\s*(\d+)\.\s+(.+)$', stripped)
            if numbered_match:
                item_text = numbered_match.group(2).strip()
                current_section.numbered_items.append(item_text)
                i += 1
                continue

            # Check for key:value or "key:" patterns
            kv_match = re.match(r'^\*?\*?([^:]+)\*?\*?\s*:\s*$', stripped)
            if kv_match and i + 1 < len(lines):
                # Key with value on next lines (as bullets)
                key = kv_match.group(1).strip().strip('*')
                i += 1
                continue

            # Regular content
            if stripped:
                current_section.content.append(stripped)

            i += 1

        return root

    def detect_domains(self, text: str) -> List[str]:
        """Detect domains from text content."""
        text_lower = text.lower()
        detected = []
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                detected.append(domain)
        return detected if detected else ["general"]

    def extract_role(self, text: str) -> Optional[str]:
        """Extract role/persona from text."""
        lines = text.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            line_lower = line.lower()
            if any(kw in line_lower for kw in self.ROLE_KEYWORDS):
                # Extract the role description
                # Remove common prefixes
                role = line.strip()
                role = re.sub(r'^>\s*', '', role)  # Remove blockquote
                role = re.sub(r'^you are a?\s*', '', role, flags=re.IGNORECASE)
                role = re.sub(r'\*\*(.+?)\*\*', r'\1', role)  # Remove bold
                # Truncate and clean
                role = role.strip().rstrip('.')
                if len(role) > 100:
                    role = role[:100]
                return self._to_ssn_id(role)
        return None

    def extract_main_task(self, text: str) -> Optional[str]:
        """Extract main task/objective from text."""
        lines = text.split('\n')
        for line in lines[:10]:
            line_lower = line.lower()
            if any(kw in line_lower for kw in self.TASK_KEYWORDS):
                task = line.strip()
                task = re.sub(r'^>\s*', '', task)
                task = re.sub(r'\*\*(.+?)\*\*', r'\1', task)
                # Find the main task description
                for kw in self.TASK_KEYWORDS:
                    if kw in task.lower():
                        idx = task.lower().find(kw)
                        task = task[idx + len(kw):].strip()
                        break
                task = task.strip().rstrip('.')
                if len(task) > 80:
                    task = task[:80]
                return self._to_ssn_id(task)
        return None

    def _to_ssn_id(self, text: str) -> str:
        """Convert text to SSN-compatible identifier."""
        # Remove special characters, keep alphanumeric and spaces
        text = re.sub(r'[^\w\s\-]', '', text)
        # Replace spaces with underscores
        text = re.sub(r'\s+', '_', text.strip())
        # Limit length
        if len(text) > 50:
            text = text[:50].rsplit('_', 1)[0]
        return text.lower()

    def _to_ssn_value(self, text: str) -> str:
        """Convert text to SSN value (can include more chars)."""
        # Remove markdown formatting
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.+?)\*', r'\1', text)  # Italic
        text = re.sub(r'`(.+?)`', r'\1', text)  # Code
        # Replace special chars
        text = text.replace(':', '-').replace('|', '-')
        text = re.sub(r'\s+', '_', text.strip())
        return text

    def _indent(self, level: int) -> str:
        """Generate indentation."""
        return ' ' * (level * self.indent_size)

    def section_to_ssn(self, section: ParsedSection, depth: int = 0) -> List[str]:
        """Convert a parsed section to SSN lines."""
        lines = []
        indent = self._indent(depth)

        if section.title and section.title != "root":
            # Convert section title to scope
            scope_name = self._to_ssn_id(section.title)
            lines.append(f"{indent}.{scope_name}")
            depth += 1
            indent = self._indent(depth)

        # Add content as key-value or description
        for content in section.content:
            if ':' in content and not content.startswith('http'):
                # Potential key-value
                parts = content.split(':', 1)
                if len(parts) == 2 and len(parts[0]) < 40:
                    key = self._to_ssn_id(parts[0])
                    value = self._to_ssn_value(parts[1])
                    lines.append(f"{indent}>{key}:{value}")
                else:
                    lines.append(f"{indent}>{self._to_ssn_id('desc')}:{self._to_ssn_value(content)[:60]}")
            elif content:
                # Add as description if it seems important
                if any(kw in content.lower() for kw in self.REQUIREMENT_KEYWORDS):
                    lines.append(f"{indent}#{'_'.join(content.split()[:3]).lower()}")

        # Add bullets as requirements or items
        if section.bullets:
            for bullet in section.bullets:
                bullet_clean = self._to_ssn_value(bullet)
                # Detect if it's a nested structure (contains sub-bullets)
                if '**' in bullet or bullet.endswith(':'):
                    # It's a category/header
                    category = re.sub(r'\*\*(.+?)\*\*:?', r'\1', bullet).strip().rstrip(':')
                    lines.append(f"{indent}.{self._to_ssn_id(category)}")
                elif len(bullet) < 80:
                    # Short enough to be a flag or simple item
                    if any(kw in bullet.lower() for kw in ['must', 'should', 'require']):
                        lines.append(f"{indent}#{self._to_ssn_id(bullet)[:50]}")
                    else:
                        lines.append(f"{indent}>{self._to_ssn_id('item')}:{bullet_clean[:70]}")
                else:
                    # Longer description
                    lines.append(f"{indent}>{self._to_ssn_id(bullet[:25])}:{bullet_clean[:70]}")

        # Add numbered items
        if section.numbered_items:
            for i, item in enumerate(section.numbered_items, 1):
                item_clean = self._to_ssn_value(item)
                lines.append(f"{indent}>step_{i}:{item_clean[:80]}")

        # Recursively process subsections
        for subsection in section.subsections:
            lines.extend(self.section_to_ssn(subsection, depth))

        return lines

    def convert(self, text: str, context: Dict = None) -> str:
        """
        Convert a structured prompt to SSN format.

        Args:
            text: The structured prompt (markdown format)
            context: Optional context dict

        Returns:
            SSN formatted string
        """
        context = context or {}

        # Strip blockquotes if present
        if text.strip().startswith('>'):
            text = self._strip_blockquotes(text)

        # Parse the markdown structure
        parsed = self.parse_markdown(text)

        # Extract high-level metadata
        role = self.extract_role(text)
        task = self.extract_main_task(text)
        domains = self.detect_domains(text)

        # Build SSN output
        ssn_lines = []

        # Add main action/command
        if task:
            ssn_lines.append(f"@prompt|{task[:40]}")
        else:
            ssn_lines.append("@prompt|structured_task")

        # Add role if detected
        if role:
            ssn_lines.append(f">role:{role[:50]}")

        # Add domains
        ssn_lines.append(f">domain:{','.join(domains)}")

        # Add expert flag if context says so
        if context.get("expert_mode"):
            ssn_lines.append("#expert")

        # Convert all sections
        for section in parsed.subsections:
            ssn_lines.extend(self.section_to_ssn(section, 0))

        # Only process root-level content/bullets if there are no subsections
        # (to avoid duplication)
        if not parsed.subsections and (parsed.content or parsed.bullets):
            root_lines = self.section_to_ssn(parsed, 0)
            # Skip the first line which would be ".root"
            ssn_lines.extend(root_lines)

        return '\n'.join(ssn_lines)


class UnifiedConverter:
    """
    Unified converter that auto-detects prompt type and uses appropriate converter.
    """

    def __init__(self):
        self.simple_converter = NLToSSN()
        self.structured_converter = StructuredPromptConverter()

    def convert(self, text: str, context: Dict = None) -> str:
        """
        Auto-detect prompt type and convert to SSN.

        Args:
            text: Input text (simple query or structured prompt)
            context: Optional context dict

        Returns:
            SSN formatted string
        """
        prompt_type = self.structured_converter.detect_prompt_type(text)

        if prompt_type == PromptType.STRUCTURED_PROMPT:
            return self.structured_converter.convert(text, context)
        else:
            return self.simple_converter.convert(text, context)

    def batch_convert(self, texts: List[str], context: Dict = None) -> List[str]:
        """Convert multiple texts to SSN."""
        return [self.convert(t, context) for t in texts]


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
    """Convert natural language or structured prompt to SSN (auto-detects type)."""
    converter = UnifiedConverter()
    return converter.convert(text, context)


def structured_to_ssn(text: str, **context) -> str:
    """Convert structured prompt (markdown) to SSN."""
    converter = StructuredPromptConverter()
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
