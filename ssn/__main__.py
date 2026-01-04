"""
SSN Command Line Interface.

Usage:
    ssn convert prompt.txt              # Convert natural language to SSN
    ssn convert prompt.txt -o out.ssn   # Save to file
    ssn convert "Find top LLM tools"    # Direct text input
    ssn parse input.ssn                 # Parse SSN to JSON
    ssn encode input.json               # Encode JSON to SSN
    ssn stats input.ssn input.json      # Compare token usage
    ssn template search --topic=x       # Generate from template
"""

import argparse
import json
import sys
from pathlib import Path

from .core import SSN
from .parser import parse_file
from .encoder import encode
from .nl_converter import nl_to_ssn, template, NLToSSN, SSNTemplates


# System prompt for LLMs to understand SSN
SSN_SYSTEM_PROMPT = """You understand SSN (Sigil Structured Notation) - a token-efficient format:
@ = action/command with |args
> = key:value pair
# = boolean flag (presence = true)
. = nested scope

Parse SSN queries and respond with detailed, actionable output."""

SSN_SYSTEM_PROMPT_DETAILED = """You understand SSN (Sigil Structured Notation) - a token-efficient format for structured queries.

SIGIL REFERENCE:
| Sigil | Meaning | Example |
|-------|---------|---------|
| @ | Action/command with args | @search|topic |
| > | Key:value pair | >depth:technical |
| # | Boolean flag (true) | #include_code |
| . | Nested scope | .config |
| | | Argument separator | @task|arg1|arg2 |
| ; | Statement separator | >a:1;>b:2 |

Parse SSN queries and respond with detailed, actionable, structured output.
When you see #flag, treat it as that option being enabled.
When you see .scope, treat following items as nested under that scope."""


def get_system_prompt(detailed: bool = False) -> str:
    """Get SSN system prompt."""
    return SSN_SYSTEM_PROMPT_DETAILED if detailed else SSN_SYSTEM_PROMPT


def cmd_convert(args):
    """Convert natural language to SSN."""
    # Get input text
    if Path(args.input).exists():
        text = Path(args.input).read_text().strip()
    else:
        text = args.input  # Direct text input
    
    # Convert to SSN
    context = {}
    if args.expert:
        context["expert_mode"] = True
    if args.code:
        context["include_code"] = True
    
    converter = NLToSSN()
    ssn_output = converter.convert(text, context)
    
    # Build final output
    if args.full:
        # Include system prompt
        output = f"""SYSTEM PROMPT:
---
{get_system_prompt(args.detailed)}
---

USER QUERY:
---
{ssn_output}
---"""
    elif args.system_only:
        output = get_system_prompt(args.detailed)
    else:
        output = ssn_output
    
    # Output
    if args.output:
        Path(args.output).write_text(output)
        print(f"Written to {args.output}")
    else:
        print(output)
    
    # Show stats if requested
    if args.stats:
        ssn_tokens = len(ssn_output) / 4
        nl_tokens = len(text) / 4
        print(f"\n--- Stats ---")
        print(f"Original: ~{int(nl_tokens)} tokens")
        print(f"SSN: ~{int(ssn_tokens)} tokens")
        print(f"Reduction: ~{int((1 - ssn_tokens/nl_tokens) * 100)}%")


def cmd_parse(args):
    """Parse SSN to JSON."""
    ssn = SSN()
    result = parse_file(args.input, ssn)
    indent = 2 if args.pretty else None
    output = json.dumps(result, indent=indent)
    
    if args.output:
        Path(args.output).write_text(output)
        print(f"Written to {args.output}")
    else:
        print(output)


def cmd_encode(args):
    """Encode JSON to SSN."""
    ssn = SSN()
    with open(args.input) as f:
        data = json.load(f)
    
    output = encode(data, ssn)
    
    if args.output:
        Path(args.output).write_text(output)
        print(f"Written to {args.output}")
    else:
        print(output)


def cmd_stats(args):
    """Compare token usage."""
    ssn = SSN()
    ssn_text = Path(args.ssn_file).read_text()
    json_text = Path(args.json_file).read_text()
    
    stats = ssn.token_stats(ssn_text, json_text)
    
    print(f"SSN:  {stats['ssn_chars']:>6} chars  ~{stats['ssn_tokens_est']:>4} tokens")
    print(f"JSON: {stats['json_chars']:>6} chars  ~{stats['json_tokens_est']:>4} tokens")
    print(f"Reduction: {stats['reduction_percent']}%")


def cmd_template(args):
    """Generate SSN from template."""
    kwargs = {}
    for item in args.params:
        if "=" in item:
            key, value = item.split("=", 1)
            kwargs[key] = value
    
    try:
        output = template(args.name, **kwargs)
        
        if args.full:
            output = f"""SYSTEM PROMPT:
---
{get_system_prompt()}
---

USER QUERY:
---
{output}
---"""
        
        print(output)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        print(f"Available templates: search, explain, compare, code, debug, analyze, summarize, recommend")
        sys.exit(1)


def cmd_system_prompt(args):
    """Print SSN system prompt."""
    print(get_system_prompt(args.detailed))


def cmd_validate(args):
    """Validate SSN syntax."""
    ssn = SSN()
    try:
        result = parse_file(args.input, ssn)
        print(f"Valid SSN: {len(result)} top-level keys")
    except Exception as e:
        print(f"Invalid: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="ssn",
        description="SSN - Sigil Structured Notation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ssn convert "Find top LLM tools in bioinformatics"
  ssn convert prompt.txt --full -o query.txt
  ssn convert prompt.txt --stats
  ssn template search --topic=protein_design --domain=bioinformatics
  ssn system-prompt > system.txt
  ssn parse config.ssn --pretty
  ssn encode data.json -o config.ssn
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Convert command
    convert_cmd = subparsers.add_parser("convert", help="Convert natural language to SSN")
    convert_cmd.add_argument("input", help="Input file or text string")
    convert_cmd.add_argument("-o", "--output", help="Output file")
    convert_cmd.add_argument("--full", action="store_true", help="Include system prompt")
    convert_cmd.add_argument("--detailed", action="store_true", help="Use detailed system prompt")
    convert_cmd.add_argument("--system-only", action="store_true", help="Output only system prompt")
    convert_cmd.add_argument("--expert", action="store_true", help="Add expert mode flags")
    convert_cmd.add_argument("--code", action="store_true", help="Add code example flags")
    convert_cmd.add_argument("--stats", action="store_true", help="Show token statistics")
    
    # Parse command
    parse_cmd = subparsers.add_parser("parse", help="Parse SSN to JSON")
    parse_cmd.add_argument("input", help="Input SSN file")
    parse_cmd.add_argument("-o", "--output", help="Output JSON file")
    parse_cmd.add_argument("--pretty", action="store_true", help="Pretty print JSON")
    
    # Encode command
    encode_cmd = subparsers.add_parser("encode", help="Encode JSON to SSN")
    encode_cmd.add_argument("input", help="Input JSON file")
    encode_cmd.add_argument("-o", "--output", help="Output SSN file")
    
    # Stats command
    stats_cmd = subparsers.add_parser("stats", help="Compare token usage")
    stats_cmd.add_argument("ssn_file", help="SSN file")
    stats_cmd.add_argument("json_file", help="JSON file")
    
    # Template command
    template_cmd = subparsers.add_parser("template", help="Generate from template")
    template_cmd.add_argument("name", help="Template name (search, explain, compare, code, debug, etc.)")
    template_cmd.add_argument("params", nargs="*", help="Template params as key=value")
    template_cmd.add_argument("--full", action="store_true", help="Include system prompt")
    
    # System prompt command
    sysprompt_cmd = subparsers.add_parser("system-prompt", help="Print SSN system prompt")
    sysprompt_cmd.add_argument("--detailed", action="store_true", help="Use detailed version")
    
    # Validate command
    validate_cmd = subparsers.add_parser("validate", help="Validate SSN syntax")
    validate_cmd.add_argument("input", help="Input SSN file")
    
    args = parser.parse_args()
    
    if args.command == "convert":
        cmd_convert(args)
    elif args.command == "parse":
        cmd_parse(args)
    elif args.command == "encode":
        cmd_encode(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "template":
        cmd_template(args)
    elif args.command == "system-prompt":
        cmd_system_prompt(args)
    elif args.command == "validate":
        cmd_validate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
