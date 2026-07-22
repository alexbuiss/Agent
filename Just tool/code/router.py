import os, sys, json, shlex, subprocess, time
from pathlib import Path
import logging

# ----- CONFIG -----
MODEL = os.environ.get("ROUTER_MODEL", "gemma:latest")  # e.g. "mistral", "tinyllama", "phi3"
BACKEND = os.environ.get("ROUTER_BACKEND", "ollama")               # "ollama" or "llamacpp"
CONF_THRESHOLD = float(os.environ.get("ROUTER_THRESHOLD", "0.40"))
VERBOSE = "--verbose" in sys.argv[2:]
DRY_RUN = "--dry-run" in sys.argv[2:]
BENCHMARK = "--benchmark" in sys.argv[2:]

# ----- LOGGER -----
logging.basicConfig(
    level=logging.DEBUG if VERBOSE else logging.WARNING,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ----- MANIFEST LOADER -----
try:
    import yaml
except ImportError:
    print("Install PyYAML: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

def load_manifest(path="manifest.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ----- TOOL SPEC FROM MANIFEST -----
def build_tool_spec(manifest):
    """Build tool specifications with memoization support."""
    tools = []
    for s in manifest.get("scripts", []):
        # Extract parameters from "parameters" (new structure)
        params = set()
        required_params = set()
        
        # Support old structure (patterns) and new (parameters)
        if "patterns" in s:
            # Old structure
            for p in s.get("patterns", []) or []:
                for g, flag in (p.get("args") or {}).items():
                    params.add(g)
        
        if "parameters" in s:
            # New structure
            for param in s.get("parameters", []):
                param_name = param.get("name", "")
                if param_name:
                    params.add(param_name)
                    if param.get("required", False):
                        required_params.add(param_name)
        
        # Support old structure defaults
        if "defaults" in s:
            params.update(s["defaults"].keys())

        tools.append({
            "name": s["name"],
            "description": s.get("description",""),
            "params": sorted(list(params)),           # logical names (not "--flag")
            "required_params": sorted(list(required_params)),
            "path": s["path"],
            "tags": s.get("tags", []),
            "priority": s.get("priority", 0),
        })
    return tools

# ----- PROMPT -----
SYSTEM_PROMPT = """You are a tool router. Your only output must be a single JSON object on one line.
Rules:
- Choose exactly one 'tool' from the provided list that solves the user request.
- 'args' contains only primitive key/value pairs (string, number, boolean).
- If unsure, return "tool": null and suggest up to 3 'candidates' with a score.
- Respect the parameter names provided by the tool (from 'params' section).
- Do not propose invented values if the user does not provide them; leave them absent.
- Never output text outside JSON, no comments.
Strict format:
{"tool": "...", "args": {...}, "confidence": 0.0-1.0, "reason": "...", "candidates": [{"name":"...","score":0.0}, ...]}
"""

def build_user_prompt(tools, user_text):
    """Build optimized user prompt with tool catalog."""
    tool_lines = []
    for t in tools:
        tool_lines.append(f"- name: {t['name']}\n  desc: {t['description']}\n  params: {', '.join(t['params']) or '(none)'}\n  tags: {', '.join(t['tags'])}")
    catalog = "\n".join(tool_lines)

    # 2-3 mini-examples to guide the style
    few_shots = """
Example 1 (confident):
USER: "resize ./in to 800x600 and save to ./out"
LLM:
{"tool":"resize_images","args":{"input":"./in","width":800,"height":600,"output":"./out"},"confidence":0.78,"reason":"Matches resize and dimensions keywords"}

Example 2 (partial info):
USER: "sum the total column from data.csv"
LLM:
{"tool":"sum_csv","args":{"file":"data.csv","column":"total"},"confidence":0.72,"reason":"CSV summation task"}

Example 3 (uncertain):
USER: "create a business report"
LLM:
{"tool": null, "args": {}, "confidence": 0.20, "reason":"No tool matches this request","candidates":[{"name":"sum_csv","score":0.22},{"name":"scrape_site","score":0.18}]}
""".strip()

    return f"""Available tools:
{catalog}

{few_shots}

Now, route the following request.
USER: {user_text}
Respond ONLY with JSON.
"""

# ----- BACKENDS -----
def call_ollama(model, system_prompt, user_prompt, metrics=None):
    """Call Ollama with performance tracking."""
    t0 = time.time()
    try:
        import ollama
    except ImportError:
        raise RuntimeError("Install ollama: pip install ollama (and `ollama run <model>`)")

    res = ollama.chat(model=model, messages=[
        {"role":"system","content":system_prompt},
        {"role":"user","content":user_prompt},
    ], options={"temperature":0.1})
    
    elapsed = time.time() - t0
    result = res["message"]["content"].strip()
    
    if metrics is not None:
        metrics["llm_time"] = elapsed
        metrics["llm_tokens"] = res.get("eval_count", 0)
    
    logger.debug(f"LLM response in {elapsed:.3f}s")
    return result

def call_llamacpp(model_path, system_prompt, user_prompt, metrics=None):
    """Call llama.cpp server with performance tracking."""
    import json as _json, urllib.request
    
    t0 = time.time()
    payload = {
        "prompt": f"<<SYS>>{system_prompt}<<SYS>>\n{user_prompt}",
        "temperature": 0.1,
        "n_predict": 512,
        "stop": ["\n\n", "\nUSER:"]
    }
    req = urllib.request.Request("http://localhost:8080/completion",
                                 data=_json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        out = resp.read().decode("utf-8")
    
    elapsed = time.time() - t0
    
    # Adapt based on your server; assumes a 'content' field
    try:
        data = json.loads(out)
        result = data.get("content","").strip()
    except:
        result = out.strip()
    
    if metrics is not None:
        metrics["llm_time"] = elapsed
    
    logger.debug(f"LLM response in {elapsed:.3f}s")
    return result

# ----- ROBUST JSON PARSING -----
def safe_json(s):
    """Parse JSON with fallback handling."""
    try:
        return json.loads(s)
    except:
        start = s.find("{"); end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = s[start:end+1]
            try: return json.loads(snippet)
            except: pass
    raise ValueError("LLM response is not valid JSON")

# ----- BUILD CLI ARGS -----
def to_cli_args(tool, args, manifest):
    # Merge with defaults and map to CLI arguments
    scripts = {s["name"]: s for s in manifest["scripts"]}
    spec = scripts[tool]
    
    # Build param->flag mapping and defaults (new + old structure)
    param2flag = {}
    defaults = spec.get("defaults", {}) or {}
    
    # Support old structure (patterns)
    for p in spec.get("patterns", []) or []:
        for g, flag in (p.get("args") or {}).items():
            param2flag[g] = flag
    
    # Support new structure (parameters)
    cli_style = spec.get("cli_style", "positional")
    if "parameters" in spec and cli_style == "positional":
        # For positional args, pass arguments in order
        param_list = spec.get("parameters", [])
        positional_order = spec.get("positional_order", [])
        
        merged = dict(defaults)
        merged.update({k:v for k,v in (args or {}).items() if v is not None})
        
        cli = []
        # Use defined order if available
        if positional_order:
            for param_name in positional_order:
                if param_name in merged:
                    cli.append(str(merged[param_name]))
        else:
            # Otherwise, use manifest order
            for param in param_list:
                param_name = param.get("name", "")
                if param_name in merged:
                    cli.append(str(merged[param_name]))
        
        return [sys.executable, spec["path"]] + cli
    else:
        # For named args (--flag style)
        merged = dict(defaults)
        merged.update({k:v for k,v in (args or {}).items() if v is not None})

        cli = []
        for k, v in merged.items():
            flag = param2flag.get(k, f"--{k}")  # fallback flag = --param
            cli += [flag, str(v)]
        return [sys.executable, spec["path"]] + cli

# ----- MAIN -----
def main():
    if len(sys.argv) < 2:
        print("Usage: python router.py \"<natural language command>\" [--dry-run] [--benchmark] [--verbose]", file=sys.stderr)
        sys.exit(64)

    metrics = {"total_time": 0, "manifest_time": 0, "llm_time": 0, "parse_time": 0}
    t_start = time.time()
    
    user_text = sys.argv[1]
    
    # Load manifest
    t0 = time.time()
    manifest = load_manifest()
    metrics["manifest_time"] = time.time() - t0
    logger.debug(f"Manifest loaded in {metrics['manifest_time']:.3f}s ({len(manifest['scripts'])} tools)")
    
    # Build tool specs
    t0 = time.time()
    tools = build_tool_spec(manifest)
    metrics["spec_time"] = time.time() - t0
    logger.debug(f"Tool specs built in {metrics['spec_time']:.3f}s")
    
    # Build user prompt
    t0 = time.time()
    user_prompt = build_user_prompt(tools, user_text)
    metrics["prompt_time"] = time.time() - t0
    logger.debug(f"Prompt built in {metrics['prompt_time']:.3f}s (len={len(user_prompt)})")
    
    # Call LLM
    t0 = time.time()
    try:
        if BACKEND.lower() == "ollama":
            raw = call_ollama(MODEL, SYSTEM_PROMPT, user_prompt, metrics)
        else:
            raw = call_llamacpp(MODEL, SYSTEM_PROMPT, user_prompt, metrics)
    except Exception as e:
        logger.error(f"LLM error: {e}")
        print(json.dumps({"status": "error", "message": str(e)}), file=sys.stderr)
        sys.exit(1)
    
    # Parse response
    t0 = time.time()
    try:
        data = safe_json(raw)
    except ValueError as e:
        logger.error(f"JSON parse error: {e}")
        print(json.dumps({"status": "error", "message": f"Failed to parse LLM response: {e}"}), file=sys.stderr)
        sys.exit(1)
    metrics["parse_time"] = time.time() - t0
    logger.debug(f"Response parsed in {metrics['parse_time']:.3f}s")

    tool = data.get("tool")
    confidence = float(data.get("confidence") or 0.0)

    if not tool or confidence < CONF_THRESHOLD:
        out = {
            "status": "no_match",
            "message": "No matching tools (insufficient confidence)",
            "confidence": confidence,
            "candidates": data.get("candidates", []),
            "llm_reason": data.get("reason", "")
        }
        if BENCHMARK:
            metrics["total_time"] = time.time() - t_start
            out["metrics"] = metrics
        print(json.dumps(out, ensure_ascii=False, indent=2))
        sys.exit(2)

    cmd = to_cli_args(tool, data.get("args", {}), manifest)

    if DRY_RUN:
        out = {
            "status":"dry_run",
            "chosen": tool,
            "confidence": round(confidence,3),
            "command": cmd,
            "llm_reason": data.get("reason","")
        }
        if BENCHMARK:
            metrics["total_time"] = time.time() - t_start
            out["metrics"] = metrics
        print(json.dumps(out, ensure_ascii=False, indent=2))
        sys.exit(0)

    out = {
        "status":"running",
        "chosen": tool,
        "confidence": round(confidence,3),
        "command": cmd,
        "llm_reason": data.get("reason","")
    }
    if BENCHMARK:
        metrics["total_time"] = time.time() - t_start
        out["metrics"] = metrics
    
    print(json.dumps(out, ensure_ascii=False))
    rc = subprocess.run(cmd).returncode
    sys.exit(rc)

if __name__ == "__main__":
    main()
