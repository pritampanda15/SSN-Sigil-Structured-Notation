"""
SSN Parser - Converts SSN text to Python objects.
"""

import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import SSN

from .core import SSNNode, SigilType


class SSNParser:
    """
    Parser for SSN notation.
    
    Grammar:
        document    := statement*
        statement   := action | context | flag | scope | comment
        action      := '@' identifier ('|' arg)*
        context     := '>' identifier ':' value
        flag        := '#' identifier
        scope       := '.' identifier statement*
        comment     := '//' text
        identifier  := [a-zA-Z_][a-zA-Z0-9_]*
        arg         := [^|;\n]+
        value       := [^;\n]+
    """
    
    SIGIL_PATTERN = re.compile(r'^([@>#\.\~\^\$]|//)')
    
    def __init__(self, ssn_instance: Optional["SSN"] = None):
        self.ssn = ssn_instance
        self.schemas = ssn_instance.schemas if ssn_instance else {}
        self.lines: List[str] = []
        self.pos: int = 0
        self.indent_stack: List[int] = [0]
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse SSN text to dictionary."""
        self.lines = self._preprocess(text)
        self.pos = 0
        
        nodes = []
        while self.pos < len(self.lines):
            node = self._parse_statement()
            if node:
                nodes.append(node)
        
        return self._nodes_to_dict(nodes)
    
    def _preprocess(self, text: str) -> List[str]:
        """Clean and split text into lines."""
        lines = []
        for line in text.strip().split('\n'):
            # Handle semicolon separators
            parts = line.split(';')
            for part in parts:
                stripped = part.strip()
                if stripped and not stripped.startswith('//'):
                    lines.append(stripped)
        return lines
    
    def _parse_statement(self) -> Optional[SSNNode]:
        """Parse a single statement."""
        if self.pos >= len(self.lines):
            return None
        
        line = self.lines[self.pos]
        self.pos += 1
        
        if not line:
            return None
        
        sigil = line[0]
        
        if sigil == '@':
            return self._parse_action(line)
        elif sigil == '>':
            return self._parse_context(line)
        elif sigil == '#':
            return self._parse_flag(line)
        elif sigil == '.':
            return self._parse_scope(line)
        elif sigil == '~':
            return self._parse_reference(line)
        elif sigil == '^':
            return self._parse_inherit(line)
        elif sigil == '$':
            return self._parse_schema_def(line)
        
        # Fallback: treat as implicit context
        if ':' in line:
            return self._parse_context('>' + line)
        
        return None
    
    def _parse_action(self, line: str) -> SSNNode:
        """Parse action statement: @action|arg1|arg2"""
        content = line[1:]  # Remove @
        parts = content.split('|')
        name = parts[0].strip()
        args = [p.strip() for p in parts[1:] if p.strip()]
        
        # Check for schema expansion
        if name in self.schemas:
            schema = self.schemas[name]
            name = schema.action
        
        node = SSNNode(
            sigil=SigilType.ACTION,
            name=name,
            args=args
        )
        
        # Parse nested children
        node.children = self._parse_children()
        
        return node
    
    def _parse_context(self, line: str) -> SSNNode:
        """Parse context statement: >key:value"""
        content = line[1:]  # Remove >
        if ':' in content:
            key, value = content.split(':', 1)
            return SSNNode(
                sigil=SigilType.CONTEXT,
                name=key.strip(),
                value=value.strip()
            )
        return SSNNode(
            sigil=SigilType.CONTEXT,
            name=content.strip(),
            value=None
        )
    
    def _parse_flag(self, line: str) -> SSNNode:
        """Parse flag statement: #flag_name"""
        name = line[1:].strip()
        return SSNNode(
            sigil=SigilType.FLAG,
            name=name
        )
    
    def _parse_scope(self, line: str) -> SSNNode:
        """Parse scope statement: .scope_name"""
        content = line[1:]  # Remove .
        
        # Check for inline action: .scope@action|args
        if '@' in content:
            scope_name, action_part = content.split('@', 1)
            node = SSNNode(
                sigil=SigilType.SCOPE,
                name=scope_name.strip()
            )
            # Parse the inline action
            action_node = self._parse_action('@' + action_part)
            node.children = [action_node]
            return node
        
        # Check for inline context: .scope>key:value
        if '>' in content:
            scope_name, ctx_part = content.split('>', 1)
            node = SSNNode(
                sigil=SigilType.SCOPE,
                name=scope_name.strip()
            )
            ctx_node = self._parse_context('>' + ctx_part)
            node.children = [ctx_node]
            return node
        
        node = SSNNode(
            sigil=SigilType.SCOPE,
            name=content.strip()
        )
        node.children = self._parse_children()
        return node
    
    def _parse_reference(self, line: str) -> SSNNode:
        """Parse reference statement: ~ref_name"""
        name = line[1:].strip()
        return SSNNode(
            sigil=SigilType.REF,
            name=name
        )
    
    def _parse_inherit(self, line: str) -> SSNNode:
        """Parse inherit statement: ^template_name"""
        name = line[1:].strip()
        return SSNNode(
            sigil=SigilType.INHERIT,
            name=name
        )
    
    def _parse_schema_def(self, line: str) -> SSNNode:
        """Parse schema definition: $short=full"""
        content = line[1:]  # Remove $
        if '=' in content:
            short, full = content.split('=', 1)
            return SSNNode(
                sigil=SigilType.SCHEMA,
                name=short.strip(),
                value=full.strip()
            )
        return None
    
    def _parse_children(self) -> List[SSNNode]:
        """Parse child statements (indented or dot-prefixed)."""
        children = []
        while self.pos < len(self.lines):
            line = self.lines[self.pos]
            # Check if this is a child (starts with . or indented context/flag)
            if line.startswith('.') or line.startswith('>') or line.startswith('#'):
                if line.startswith('.'):
                    child = self._parse_statement()
                else:
                    child = self._parse_statement()
                if child:
                    children.append(child)
            else:
                break
        return children
    
    def _nodes_to_dict(self, nodes: List[SSNNode]) -> Dict[str, Any]:
        """Convert node list to dictionary."""
        result = {}
        for node in nodes:
            node_dict = node.to_dict()
            if "_action" in node_dict:
                # Merge action into result
                result.update(node_dict)
            else:
                result.update(node_dict)
        return result


def parse(text: str, ssn_instance: Optional["SSN"] = None) -> Dict[str, Any]:
    """Parse SSN text to dictionary."""
    parser = SSNParser(ssn_instance)
    return parser.parse(text)


def parse_file(filepath: str, ssn_instance: Optional["SSN"] = None) -> Dict[str, Any]:
    """Parse SSN file to dictionary."""
    with open(filepath, 'r') as f:
        return parse(f.read(), ssn_instance)
