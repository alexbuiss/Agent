#!/usr/bin/env python3
"""
Parameter Extraction Improved with Few-shot Prompting
Uses examples from the manifest to improve extraction
"""

import json
import yaml
from typing import Dict, List, Tuple, Any
from pathlib import Path


class FewShotExampleBuilder:
    """Build few-shot examples based on the manifest"""

    def __init__(self, manifest_path: str = "manifest.yaml"):
        """Load and parse the manifest"""
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = yaml.safe_load(f)
        
        self.scripts = {s["name"]: s for s in self.manifest.get("scripts", [])}
    
    def build_few_shot_examples(self, tool_name: str, num_examples: int = 2) -> List[Dict[str, str]]:
        """
        Build few-shot examples for a specific tool

        Returns:
            List of dicts: [{"query": "...", "extracted": {...}}, ...]
        """
        
        tool = self.scripts.get(tool_name)
        if not tool:
            return []
        
        examples = []

        # 1. Extract the examples from the manifest (if any)
        if "examples" in tool:
            # Format: ["./input ./output" or "description"]
            # We use them as inspiration but we need to create query-params examples
            pass

        # 2. Create synthetic examples based on the parameters
        examples = self._generate_synthetic_examples(tool, num_examples)
        
        return examples
    
    def _generate_synthetic_examples(self, tool: Dict, num_examples: int) -> List[Dict]:
        """
        Generate synthetic examples based on the tool's parameters
        """
        examples = []

        # Create example 1: with all values provided
        example1 = self._create_example(
            tool,
            example_num=1,
            description="All parameters provided"
        )
        if example1:
            examples.append(example1)
        
        # Create example 2: with optional parameters
        example2 = self._create_example(
            tool,
            example_num=2,
            description="Required parameters only"
        )
        if example2:
            examples.append(example2)
        
        return examples
    
    def _create_example(self, tool: Dict, example_num: int, description: str) -> Dict:
        """
        Create ONE example for a tool
        """

        tool_name = tool["name"]
        params = tool.get("parameters", [])

        # Create query and expected params
        query_parts = []
        extracted = {}
        
        for param in params:
            name = param["name"]
            ptype = param.get("type", "string")
            required = param.get("required", False)
            encode = param.get("encode", "")
            description_text = param.get("description", "")
            
            # Add to the query
            if required or example_num == 1:  # Example 1 = all params
                value = self._get_example_value(name, ptype, encode)

                # Format the query
                if ptype == "list" and encode:
                    if encode == "comma_no_brackets":
                        query_parts.append(f"{description_text}: {value.replace(' ', ',')}")
                    elif encode == "space_separated":
                        query_parts.append(f"{description_text}: {value}")
                    elif encode == "comma_separated":
                        query_parts.append(f"{description_text}: {value}")
                elif ptype == "bool":
                    bool_val = "true" if "true" in value else "false"
                    query_parts.append(f"({name}={bool_val})")
                else:
                    query_parts.append(f"{description_text}: {value}")
                
                # Add to the extracted params
                extracted[name] = self._get_example_value_formatted(name, ptype, encode)
        
        query = f"{tool_name}: " + ", ".join(query_parts)
        
        return {
            "query": query,
            "extracted": extracted,
            "note": description
        }
    
    def _get_example_value(self, param_name: str, ptype: str, encode: str) -> str:
        """
        Get an example value for a parameter
        """

        # Default values depending on the type
        if ptype == "path":
            if "folder" in param_name or "dir" in param_name:
                return "/data" if "input" in param_name else "/output"
            else:
                return "/data/file.nii.gz"
        
        elif ptype == "list":
            if "lm" in param_name or "landmark" in param_name:
                if encode == "comma_no_brackets":
                    return "1,2,3,4,5"
                elif encode == "space_separated":
                    return "1 2 3 4 5"
                else:
                    return "1,2,3,4,5"
            elif "teeth" in param_name:
                if encode == "space_separated":
                    return "1 2 3"
                else:
                    return "1,2,3"
            else:
                return "item1,item2,item3"
        
        elif ptype == "bool":
            return "true" if "input" not in param_name else "false"
        
        elif ptype == "int":
            return "10"
        
        elif ptype == "float":
            return "1.0"
        
        else:  # string
            if "suffix" in param_name or "name" in param_name:
                return "_processed"
            else:
                return "value"
    
    def _get_example_value_formatted(self, param_name: str, ptype: str, encode: str) -> Any:
        """
        Get the formatted value for the extraction JSON
        """
        value = self._get_example_value(param_name, ptype, encode)
        
        if ptype == "bool":
            return value == "true"
        elif ptype == "int":
            return int(value)
        elif ptype == "float":
            return float(value)
        elif ptype == "list" and encode == "python_literal":
            return value  # "[1.0, 0.3]" stays a string
        else:
            return value


class ImprovedParameterExtractor:
    """
    Improved parameter extractor with:
    - Few-shot prompting
    - Automatic type conversion
    - Better parsing
    """
    
    def __init__(self, manifest_path: str = "manifest.yaml"):
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = yaml.safe_load(f)
        
        self.scripts = {s["name"]: s for s in self.manifest.get("scripts", [])}
        self.example_builder = FewShotExampleBuilder(manifest_path)
    
    def build_prompt(self, tool_name: str, user_text: str) -> str:
        """
        Build an improved prompt with few-shot examples
        """

        tool_spec = self.scripts.get(tool_name)
        if not tool_spec:
            return ""

        # 1. Build the parameters section with types
        params_section = self._build_params_section(tool_spec)

        # 2. Build the few-shot examples
        few_shot_examples = self._build_few_shot_section(tool_name, tool_spec)

        # 3. Build the final prompt
        prompt = f"""Extract parameters from the user request for tool: {tool_name}

PARAMETER DEFINITIONS:
{params_section}

EXAMPLES OF CORRECT EXTRACTION:
{few_shot_examples}

EXTRACTION RULES:
1. For lists: Use the correct format based on 'encode':
   - comma_no_brackets: "1,2,3" (no brackets, no quotes around items)
   - space_separated: "1 2 3" (space separated, no brackets)
   - comma_separated: "1,2,3" (comma separated, no brackets)
   - python_literal: "[1.0, 0.3]" (keep Python array format as string)

2. For booleans: Use JSON format without quotes: true or false

3. For integers: Use JSON format without quotes: 10, 0, 64

4. For floats: Use JSON format without quotes: 1.0, 0.95

5. For paths:
   - If parameter name contains "folder" or "dir": Use folder path only (no filename)
   - Otherwise: Can be file path (with extension)

6. For strings: Use normal string format with quotes: "value"

7. ONLY extract parameters mentioned in the user request
   - Do NOT invent values
   - Leave unmentioned optional parameters absent

USER REQUEST: "{user_text}"

Return ONLY valid JSON on one line (no markdown, no explanation):
{{"extracted": {{...}}, "confidence": 0.0-1.0, "missing_required": [...], "notes": "..."}}
"""
        
        return prompt
    
    def _build_params_section(self, tool_spec: Dict) -> str:
        """Build the parameters section"""
        
        lines = []
        for param in tool_spec.get("parameters", []):
            name = param["name"]
            ptype = param.get("type", "string")
            required = "REQUIRED" if param.get("required") else "optional"
            encode = param.get("encode", "")
            description = param.get("description", "")
            
            encode_info = f" (encode: {encode})" if encode else ""
            lines.append(f"  - {name} ({ptype}) [{required}]{encode_info}: {description}")
        
        return "\n".join(lines)
    
    def _build_few_shot_section(self, tool_name: str, tool_spec: Dict) -> str:
        """Build the few-shot examples section"""
        
        examples = self.example_builder.build_few_shot_examples(tool_name)
        
        output = []
        for i, example in enumerate(examples, 1):
            output.append(f"\nExample {i}: {example.get('note', '')}")
            output.append(f"  Query: \"{example['query']}\"")
            output.append(f"  Extracted:")
            for key, value in example['extracted'].items():
                if isinstance(value, str):
                    output.append(f"    {key}: \"{value}\"")
                elif isinstance(value, bool):
                    output.append(f"    {key}: {str(value).lower()}")
                else:
                    output.append(f"    {key}: {value}")
        
        return "\n".join(output)
    
    def convert_types(self, extracted: Dict, tool_spec: Dict) -> Tuple[Dict, List[str]]:
        """
        Convert the extracted types according to the manifest spec

        Returns:
            (converted_params, errors)
        """
        
        converted = {}
        errors = []
        
        for param in tool_spec.get("parameters", []):
            name = param["name"]
            ptype = param.get("type", "string")
            
            if name not in extracted:
                continue
            
            value = extracted[name]
            
            try:
                if ptype == "bool":
                    converted[name] = self._to_bool(value)
                
                elif ptype == "int":
                    converted[name] = self._to_int(value)
                
                elif ptype == "float":
                    converted[name] = self._to_float(value)
                
                elif ptype == "list":
                    encode = param.get("encode", "")
                    converted[name] = self._to_list(value, encode)
                
                elif ptype == "path":
                    converted[name] = str(value).strip()
                
                else:  # string
                    converted[name] = str(value)
            
            except ValueError as e:
                errors.append(f"Error converting {name}: {e}")
        
        return converted, errors
    
    @staticmethod
    def _to_bool(value) -> bool:
        """Convert to boolean"""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            if value.lower() in ("true", "yes", "1", "on"):
                return True
            elif value.lower() in ("false", "no", "0", "off"):
                return False
        raise ValueError(f"Cannot convert {value} to bool")
    
    @staticmethod
    def _to_int(value) -> int:
        """Convert to integer"""
        if isinstance(value, int):
            return value
        try:
            return int(float(str(value)))
        except:
            raise ValueError(f"Cannot convert {value} to int")
    
    @staticmethod
    def _to_float(value) -> float:
        """Convert to float"""
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value))
        except:
            raise ValueError(f"Cannot convert {value} to float")
    
    @staticmethod
    def _to_list(value, encode: str) -> Any:
        """Convert to a list according to the encoding"""
        if isinstance(value, list):
            return value

        if isinstance(value, str):
            s = value.strip()

            # If it's a Python literal, return as-is
            if encode == "python_literal":
                return s  # "[1.0, 0.3]"

            # Parse according to the separator
            if encode == "space_separated":
                items = s.split()
            else:  # comma_no_brackets, comma_separated
                if s.startswith("[") and s.endswith("]"):
                    s = s[1:-1]
                items = [x.strip() for x in s.split(",")]
            
            return items
        
        raise ValueError(f"Cannot convert {value} to list")


if __name__ == "__main__":
    # Test
    extractor = ImprovedParameterExtractor()
    
    # Test prompt for ali_cbct
    print("=" * 80)
    print("PROMPT FOR ALI_CBCT")
    print("=" * 80)
    
    test_query = "Detect landmarks 1,2,3,4,5 in CBCT /data/cbct.nii.gz using models from /models/ali_cbct, save to /output"
    prompt = extractor.build_prompt("ali_cbct", test_query)
    print(prompt)
    
    print("\n" + "=" * 80)
    print("TYPE CONVERSION EXAMPLE")
    print("=" * 80)

    # Test conversion
    tool_spec = extractor.scripts["ali_cbct"]

    # Input with wrong types
    bad_extracted = {
        "input": "/data/cbct.nii.gz",
        "dir_models": "/models/ali_cbct",
        "lm_type": "1,2,3,4,5",  # String, but must stay a string according to encode
        "output_dir": "/output",
        "DCMInput": "false",  # String, must become a bool
        "spawn_radius": "10"  # String, must become an int
    }

    converted, errors = extractor.convert_types(bad_extracted, tool_spec)

    print("Before conversion:")
    print(json.dumps(bad_extracted, indent=2))

    print("\nAfter conversion:")
    print(json.dumps(converted, indent=2, default=str))

    if errors:
        print("\nErrors:")
        for err in errors:
            print(f"  - {err}")
