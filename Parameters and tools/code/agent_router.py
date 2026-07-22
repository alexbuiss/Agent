#!/usr/bin/env python3
"""
Advanced Agent Router with Parameter Extraction
Combine tool selection + intelligent parameter extraction
Uses an LLM to understand the context and extract the parameters
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

# ----- CONFIG -----
MODEL = os.environ.get("ROUTER_MODEL", "mistral:latest")
BACKEND = os.environ.get("ROUTER_BACKEND", "ollama")
CONF_THRESHOLD = float(os.environ.get("ROUTER_THRESHOLD", "0.40"))
PARAM_CONF_THRESHOLD = float(os.environ.get("PARAM_CONF_THRESHOLD", "0.35"))
VERBOSE = "--verbose" in sys.argv[2:] if len(sys.argv) > 2 else False
DRY_RUN = "--dry-run" in sys.argv[2:] if len(sys.argv) > 2 else False
INTERACTIVE = "--interactive" in sys.argv[2:] if len(sys.argv) > 2 else False
BENCHMARK = "--benchmark" in sys.argv[2:] if len(sys.argv) > 2 else False

# Logger
logging.basicConfig(
    level=logging.DEBUG if VERBOSE else logging.WARNING,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ===== IMPORTS =====
try:
    import yaml
except ImportError:
    print("Install PyYAML: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

try:
    import subprocess
except ImportError:
    print("subprocess module not found")
    sys.exit(1)

# ===== IMPORTS FOR IMPROVED EXTRACTION & VALIDATION =====
try:
    from Param_and_cli.code.parameter_extraction_improved import ImprovedParameterExtractor
    from Param_and_cli.code.parameter_validator import ParameterValidator, ValidationReport
    IMPROVED_EXTRACTION_AVAILABLE = True
except ImportError:
    IMPROVED_EXTRACTION_AVAILABLE = False
    if VERBOSE:
        print("Warning: Improved extraction modules not available", file=sys.stderr)

# ===== MANIFEST & TOOLS =====
def load_manifest(path="manifest.yaml"):
    """Load and cache manifest."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_tool_spec(manifest):
    """Build tool specs from manifest."""
    tools = []
    for s in manifest.get("scripts", []):
        params = set()
        required_params = set()
        
        # Extract parameters
        if "parameters" in s:
            for param in s.get("parameters", []):
                param_name = param.get("name", "")
                if param_name:
                    params.add(param_name)
                    if param.get("required", False):
                        required_params.add(param_name)
        
        if "defaults" in s:
            params.update(s["defaults"].keys())

        tools.append({
            "name": s["name"],
            "description": s.get("description", ""),
            "params": sorted(list(params)),
            "required_params": sorted(list(required_params)),
            "path": s["path"],
            "tags": s.get("tags", []),
            "priority": s.get("priority", 0),
        })
    return tools

# ===== STEP 1: TOOL SELECTION =====
def build_tool_selection_prompt(tools: List[Dict], user_text: str) -> str:
    """Prompt to select the right tool."""
    tool_lines = []
    for t in tools:
        tool_lines.append(
            f"- name: {t['name']}\n"
            f"  desc: {t['description']}\n"
            f"  params: {', '.join(t['params']) or '(none)'}\n"
            f"  tags: {', '.join(t['tags'])}"
        )
    catalog = "\n".join(tool_lines)
    
    return f"""You are a tool router. Your ONLY output must be a single JSON object on one line.
Rules:
- Choose the SINGLE best 'tool' from the provided list that solves the user request.
- If unsure, return "tool": null and suggest up to 3 'candidates'.
- Never output text outside JSON.

Available tools:
{catalog}

User request: {user_text}

Respond ONLY with JSON on one line:
{{"tool": "tool_name_or_null", "confidence": 0.0-1.0, "reason": "...", "candidates": []}}
"""

def select_tool(manifest: Dict, tools: List[Dict], user_text: str) -> Tuple[Optional[str], float, str]:
    """Step 1: Select the tool."""
    
    prompt = build_tool_selection_prompt(tools, user_text)
    
    try:
        if BACKEND.lower() == "ollama":
            import ollama
            res = ollama.chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a tool router. Output ONLY valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0.1}
            )
            response = res["message"]["content"].strip()
        else:
            raise ValueError(f"Backend {BACKEND} not supported")
    except Exception as e:
        logger.error(f"Tool selection error: {e}")
        return None, 0.0, str(e)
    
    # Parse JSON
    try:
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            logger.debug(f"Parsing JSON: {json_str[:200]}")
            data = json.loads(json_str)
        else:
            logger.warning(f"No JSON in response: {response[:200]}")
            return None, 0.0, "No JSON found"
    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode error: {e}, response: {response[:200]}")
        return None, 0.0, str(e)
    
    tool = data.get("tool")
    confidence = float(data.get("confidence", 0.0))
    reason = data.get("reason", "")
    
    logger.debug(f"Tool selection: {tool} (conf: {confidence:.2f})")
    return tool, confidence, reason

# ===== STEP 2: PARAMETER EXTRACTION =====
def build_parameter_extraction_prompt(tool_name: str, tool_spec: Dict, user_text: str) -> str:
    """Prompt to extract the parameters."""
    
    params_desc = []
    for p in tool_spec.get("parameters", []):
        name = p.get("name", "")
        ptype = p.get("type", "string")
        required = p.get("required", False)
        description = p.get("description", "")
        default = p.get("default")
        
        req_marker = "[REQUIRED]" if required else "[optional]"
        param_line = f"  - {name} ({ptype}) {req_marker}: {description}"
        if default is not None:
            param_line += f" Default: {default}"
        params_desc.append(param_line)
    
    params_section = "\n".join(params_desc) if params_desc else "  (no parameters)"
    
    return f"""Extract parameters from this user request.

Tool: {tool_name}
Description: {tool_spec.get('description', 'N/A')}

Parameters to extract:
{params_section}

User Request: "{user_text}"

For EACH parameter above, extract the value if mentioned in the request.
Assign confidence 0.0-1.0:
- 1.0: explicitly stated
- 0.7: clearly implied  
- 0.4: inferred from context
- <0.4: not found

Return JSON (one line, no markdown):
{{"extracted": {{"param_name": {{"value": "value_here", "confidence": 0.9}}}}, "missing_required": ["param1", "param2"]}}

Rules:
- Extract ONLY what's in the request
- Use exact paths/values mentioned
- Parse "1,2,3" as list items
- For yes/no: true/false
- Do NOT invent values
"""

def extract_parameters(manifest: Dict, tool_name: str, user_text: str) -> Tuple[Dict[str, Any], float, List[str]]:
    """Step 2: Extract the parameters for the tool (with improved validation)."""
    
    # Get tool spec
    scripts = {s["name"]: s for s in manifest.get("scripts", [])}
    tool_spec = scripts.get(tool_name)
    if not tool_spec:
        logger.warning(f"Tool {tool_name} not found in manifest")
        return {}, 0.0, []
    
    # ===== USE IMPROVED EXTRACTION IF AVAILABLE =====
    if IMPROVED_EXTRACTION_AVAILABLE:
        logger.debug("Using IMPROVED extraction with few-shot prompting")

        try:
            # Create the prompt with few-shot examples
            prompt = improved_extractor.build_prompt(tool_name, user_text)

            # Call the LLM
            if BACKEND.lower() == "ollama":
                import ollama
                res = ollama.chat(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "You are a parameter extraction expert. Output ONLY valid JSON on one line."},
                        {"role": "user", "content": prompt}
                    ],
                    options={"temperature": 0.1}
                )
                response = res["message"]["content"].strip()
            else:
                raise ValueError(f"Backend {BACKEND} not supported")
            
            # Parse JSON
            try:
                data = json.loads(response)
            except json.JSONDecodeError:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    data = json.loads(response[json_start:json_end])
                else:
                    raise json.JSONDecodeError("No JSON found", response, 0)
            
            extracted_raw = data.get("extracted", {})
            confidence = float(data.get("confidence", 0.0))
            
            # STEP 1.5: Type conversion
            logger.debug("Converting parameter types")
            extracted_converted, conversion_errors = improved_extractor.convert_types(
                extracted_raw, tool_spec
            )
            if conversion_errors and VERBOSE:
                logger.warning(f"Conversion errors: {conversion_errors}")
            
            # STEP 2: Validation
            logger.debug("Validating parameters")
            validation_result = parameter_validator.validate(tool_name, extracted_converted)
            
            if VERBOSE and not validation_result["valid"]:
                report = ValidationReport.generate_report(tool_name, validation_result)
                print(report, file=sys.stderr)
            
            # Return the validated parameters
            final_params = validation_result["params"]
            final_confidence = confidence if validation_result["valid"] else confidence * 0.6
            missing_required = validation_result["missing_required"]
            
            logger.debug(
                f"IMPROVED: Extracted {len(final_params)} params, confidence: {final_confidence:.2f}, "
                f"valid: {validation_result['valid']}, missing: {missing_required}"
            )
            return final_params, final_confidence, missing_required
        
        except Exception as e:
            logger.warning(f"IMPROVED extraction failed, falling back to simple: {e}")
            # Fall back to simple extraction
    
    # ===== FALLBACK: SIMPLE EXTRACTION (OLD METHOD) =====
    logger.debug("Using SIMPLE extraction (fallback)")
    
    prompt = build_parameter_extraction_prompt(tool_name, tool_spec, user_text)
    
    try:
        if BACKEND.lower() == "ollama":
            import ollama
            res = ollama.chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a parameter extraction expert. Output ONLY valid JSON on one line."},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0.1}
            )
            response = res["message"]["content"].strip()
        else:
            raise ValueError(f"Backend {BACKEND} not supported")
    except Exception as e:
        logger.error(f"Parameter extraction error: {e}")
        required_params = [p["name"] for p in tool_spec.get("parameters", []) if p.get("required")]
        return {}, 0.0, required_params
    
    # Parse JSON
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
            else:
                required_params = [p["name"] for p in tool_spec.get("parameters", []) if p.get("required")]
                return {}, 0.0, required_params
        except json.JSONDecodeError:
            required_params = [p["name"] for p in tool_spec.get("parameters", []) if p.get("required")]
            return {}, 0.0, required_params
    
    # Process extracted parameters
    extracted = {}
    min_confidence = 1.0
    
    for param_name, param_info in data.get("extracted", {}).items():
        if isinstance(param_info, dict):
            confidence = param_info.get("confidence", 0.0)
            value = param_info.get("value")
            value = convert_parameter_type(param_name, value, tool_spec)
            
            if confidence >= PARAM_CONF_THRESHOLD:
                extracted[param_name] = value
                min_confidence = min(min_confidence, confidence)
    
    missing_required = [p for p in data.get("missing_required", []) if p not in ["param1", "param2", "param3", "param4"]]
    
    if not missing_required:
        extracted_params = set(extracted.keys())
        required_params = {p["name"] for p in tool_spec.get("parameters", []) if p.get("required")}
        missing_required = list(required_params - extracted_params)
    
    logger.debug(f"SIMPLE: Extracted {len(extracted)} params, min_conf: {min_confidence:.2f}, missing: {missing_required}")
    return extracted, min_confidence, missing_required

def convert_parameter_type(param_name: str, value: Any, tool_spec: Dict) -> Any:
    """Convert a value to the right type."""
    
    for p in tool_spec.get("parameters", []):
        if p.get("name") == param_name:
            ptype = p.get("type", "string")
            
            if ptype == "bool":
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    return value.lower() in ["true", "yes", "on", "1", "enabled"]
                return bool(value)
            elif ptype == "int":
                return int(value) if value is not None else None
            elif ptype == "float":
                return float(value) if value is not None else None
            elif ptype.startswith("list"):
                if isinstance(value, list):
                    return value
                if isinstance(value, str):
                    try:
                        return eval(value)
                    except:
                        return [v.strip() for v in value.split(",")]
                return value
            
            return value
    
    return value

# ===== CLI ARGS BUILDING =====
def build_cli_args(tool_name: str, params: Dict[str, Any], manifest: Dict) -> List[str]:
    """Build the CLI arguments."""

    scripts = {s["name"]: s for s in manifest.get("scripts", [])}
    spec = scripts[tool_name]

    # Merge with the defaults
    defaults = spec.get("defaults", {}) or {}
    merged = dict(defaults)
    merged.update({k: v for k, v in (params or {}).items() if v is not None})
    
    cli_style = spec.get("cli_style", "positional")
    
    if cli_style == "positional":
        # Positional arguments
        param_list = spec.get("parameters", [])
        positional_order = spec.get("positional_order", [])
        
        cli = []
        if positional_order:
            for param_name in positional_order:
                if param_name in merged:
                    cli.append(str(merged[param_name]))
        else:
            for param in param_list:
                param_name = param.get("name", "")
                if param_name in merged:
                    cli.append(str(merged[param_name]))
        
        return [sys.executable, spec["path"]] + cli
    else:
        # Named arguments (--flag style)
        param2flag = {}
        for p in spec.get("parameters", []):
            param_name = p.get("name", "")
            if param_name:
                flag = p.get("flag", f"--{param_name}")
                param2flag[param_name] = flag
        
        cli = []
        for k, v in merged.items():
            flag = param2flag.get(k, f"--{k}")
            cli += [flag, str(v)]
        
        return [sys.executable, spec["path"]] + cli

# ===== MAIN FLOW =====
def main():
    if len(sys.argv) < 2:
        print("Usage: python agent_router.py \"<request>\" [--dry-run] [--interactive] [--verbose]", file=sys.stderr)
        sys.exit(64)
    
    user_text = sys.argv[1]
    metrics = {
        "tool_selection_time": 0,
        "parameter_extraction_time": 0,
        "cli_build_time": 0,
        "total_time": 0
    }
    t_start = time.time()
    
    # ===== INITIALIZE THE IMPROVED MODULES =====
    global improved_extractor, parameter_validator, IMPROVED_EXTRACTION_AVAILABLE
    if IMPROVED_EXTRACTION_AVAILABLE:
        try:
            improved_extractor = ImprovedParameterExtractor("manifest.yaml")
            parameter_validator = ParameterValidator("manifest.yaml")
            logger.info("✅ Improved extraction & validation modules loaded")
        except Exception as e:
            logger.warning(f"Failed to initialize improved modules: {e}")
            IMPROVED_EXTRACTION_AVAILABLE = False
    
    # Load manifest
    logger.info(f"Loading manifest...")

    manifest = load_manifest()
    tools = build_tool_spec(manifest)
    logger.info(f"Found {len(tools)} tools")
    
    # STEP 1: Select the tool
    logger.info(f"Selecting tool for: {user_text[:60]}...")
    t0 = time.time()
    selected_tool, tool_conf, tool_reason = select_tool(manifest, tools, user_text)
    metrics["tool_selection_time"] = time.time() - t0
    
    if not selected_tool or tool_conf < CONF_THRESHOLD:
        output = {
            "status": "no_tool_match",
            "tool": selected_tool,
            "confidence": tool_conf,
            "reason": tool_reason,
            "threshold": CONF_THRESHOLD,
        }
        if BENCHMARK:
            output["metrics"] = metrics
        print(json.dumps(output, ensure_ascii=False, indent=2))
        sys.exit(2)
    
    logger.info(f"Selected tool: {selected_tool} (confidence: {tool_conf:.2f})")
    
    # STEP 2: Extract the parameters
    logger.info(f"Extracting parameters...")
    t0 = time.time()
    params, param_conf, missing_required = extract_parameters(manifest, selected_tool, user_text)
    metrics["parameter_extraction_time"] = time.time() - t0
    
    logger.info(f"Extracted {len(params)} parameters")
    
    # STEP 3: Interactive mode if requested
    if INTERACTIVE and missing_required:
        print(f"\n⚠️  Missing required parameters: {', '.join(missing_required)}")
        for param in missing_required:
            value = input(f"Enter {param}: ").strip()
            if value:
                params[param] = value
    
    # STEP 4: Build the CLI arguments
    t0 = time.time()
    cli_args = build_cli_args(selected_tool, params, manifest)
    metrics["cli_build_time"] = time.time() - t0
    
    # Prepare the output
    output = {
        "status": "ready",
        "tool": selected_tool,
        "tool_confidence": round(tool_conf, 3),
        "parameters_confidence": round(param_conf, 3) if params else 0.0,
        "parameters": params,
        "missing_required": missing_required,
        "command": cli_args,
        "dry_run": DRY_RUN,
    }
    
    if BENCHMARK:
        metrics["total_time"] = time.time() - t_start
        output["metrics"] = metrics
    
    # Output JSON
    print(json.dumps(output, ensure_ascii=False, indent=2))
    
    if DRY_RUN:
        logger.info("Dry run mode - not executing")
        sys.exit(0)
    
    # Execute the CLI
    logger.info(f"Executing: {' '.join(cli_args)}")
    try:
        rc = subprocess.run(cli_args).returncode
        sys.exit(rc)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Execution error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
