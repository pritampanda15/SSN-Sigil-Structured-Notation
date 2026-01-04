"""
Test suite for SSN.
"""

import pytest
from ssn import SSN, parse, encode, decode


class TestParser:
    """Tests for SSN parser."""
    
    def test_parse_action(self):
        result = parse("@analyze|protein.pdb")
        assert result["_action"] == "analyze"
        assert result["_args"] == ["protein.pdb"]
    
    def test_parse_action_multiple_args(self):
        result = parse("@dock|receptor.pdb|ligand.mol2|10|20|30")
        assert result["_action"] == "dock"
        assert result["_args"] == ["receptor.pdb", "ligand.mol2", 10, 20, 30]
    
    def test_parse_context(self):
        result = parse(">temperature:300")
        assert result["temperature"] == 300
    
    def test_parse_context_string(self):
        result = parse(">name:protein_analysis")
        assert result["name"] == "protein_analysis"
    
    def test_parse_context_float(self):
        result = parse(">value:3.14")
        assert result["value"] == 3.14
    
    def test_parse_flag(self):
        result = parse("#verbose")
        assert result["verbose"] is True
    
    def test_parse_multiple_flags(self):
        result = parse("#verbose\n#debug\n#save")
        assert result["verbose"] is True
        assert result["debug"] is True
        assert result["save"] is True
    
    def test_parse_combined(self):
        text = """
@analyze|protein.pdb
>chain:A
>temperature:300
#verbose
"""
        result = parse(text)
        assert result["_action"] == "analyze"
        assert result["chain"] == "A"
        assert result["temperature"] == 300
        assert result["verbose"] is True
    
    def test_parse_semicolon_separator(self):
        result = parse(">a:1;>b:2;#flag")
        assert result["a"] == 1
        assert result["b"] == 2
        assert result["flag"] is True
    
    def test_parse_boolean_values(self):
        result = parse(">enabled:true\n>disabled:false")
        assert result["enabled"] is True
        assert result["disabled"] is False
    
    def test_parse_null_value(self):
        result = parse(">empty:null")
        assert result["empty"] is None


class TestSchema:
    """Tests for schema expansion."""
    
    def test_schema_registration(self):
        ssn = SSN()
        ssn.register_schema("test", "test_action", ["arg1", "arg2"])
        assert "test" in ssn.schemas
        assert ssn.schemas["test"].action == "test_action"
    
    def test_schema_expansion(self):
        ssn = SSN()
        result = ssn.parse("@md|10000|300|1")
        assert result["_action"] == "molecular_dynamics"
    
    def test_builtin_schemas(self):
        ssn = SSN()
        assert "md" in ssn.schemas
        assert "dock" in ssn.schemas
        assert "align" in ssn.schemas
        assert "admet" in ssn.schemas
        assert "prot" in ssn.schemas


class TestEncoder:
    """Tests for SSN encoder."""
    
    def test_encode_action(self):
        data = {"action": "analyze", "input": "file.pdb"}
        result = encode(data)
        assert "@analyze" in result
        assert "file.pdb" in result
    
    def test_encode_context(self):
        data = {"temperature": 300, "pressure": 1}
        result = encode(data)
        assert ">temperature:300" in result
        assert ">pressure:1" in result
    
    def test_encode_flag(self):
        data = {"verbose": True, "debug": True}
        result = encode(data)
        assert "#verbose" in result
        assert "#debug" in result
    
    def test_encode_false_flag_omitted(self):
        data = {"verbose": False}
        result = encode(data)
        assert "#verbose" not in result


class TestRoundTrip:
    """Tests for encode-decode round trips."""
    
    def test_simple_roundtrip(self):
        original = """
@analyze|protein.pdb
>chain:A
>temperature:300
#verbose
"""
        ssn = SSN()
        parsed = ssn.parse(original)
        encoded = ssn.encode(parsed)
        reparsed = ssn.parse(encoded)
        
        # Core values should match
        assert reparsed.get("_action") == parsed.get("_action")
        assert reparsed.get("chain") == parsed.get("chain")
        assert reparsed.get("temperature") == parsed.get("temperature")


class TestTokenStats:
    """Tests for token statistics."""
    
    def test_token_reduction(self):
        ssn = SSN()
        
        ssn_text = "@md|protein.pdb\n>temp:300"
        json_text = '{"task":"molecular_dynamics","input":"protein.pdb","temperature":300}'
        
        stats = ssn.token_stats(ssn_text, json_text)
        
        assert stats["ssn_chars"] < stats["json_chars"]
        assert stats["reduction_percent"] > 0


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_input(self):
        result = parse("")
        assert result == {}
    
    def test_whitespace_only(self):
        result = parse("   \n\n   ")
        assert result == {}
    
    def test_special_characters_in_value(self):
        result = parse(">smiles:CC(=O)Oc1ccccc1C(=O)O")
        assert "smiles" in result
    
    def test_numeric_string(self):
        result = parse(">id:ABC123")
        assert result["id"] == "ABC123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
