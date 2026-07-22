import json
import time
import os
import numpy as np
from pydantic import BaseModel, Field
from ollama import chat
from pathlib import Path
from sentence_transformers import CrossEncoder
import torch
from tqdm.autonotebook import tqdm

# ----- CONFIGURATION GPU -----
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(f"CUDA Available: {torch.cuda.is_available()}")

# ----- GLOBAL VARIABLES -----
MODELS_TO_TEST = [
    "qwen3:8b", 
    # "qwen3.5:9b", 33.73
    # "qwen2.5-coder:7b", 
    # "hermes3:8b", 
    # "llama3:latest", 
    # "llama3.1:8b"
    
]
queries_type = "long" 
top_k = 3 # Number of top tools to retrieve

# ----- PATHS -----
script_folder = Path(__file__).parent.absolute()
tools_path = script_folder.parent / 'input' / 'documentation' / 'tools'
queries_path = script_folder.parent / 'input' / 'param' / f'{queries_type}.json'
result_dir = script_folder / 'output' / 'benchmark'
result_dir.mkdir(parents=True, exist_ok=True) # Folder dedicated to the benchmark

# ----- LOAD MODELS & DATA -----
print("Loading BAAI Reranker...")
reranker_model = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512, device="cuda")

tools = []
for file_path in tools_path.glob('*.json'):
    with open(file_path, 'r', encoding='utf-8') as f:
        tools.append(json.load(f))

with open(queries_path, 'r', encoding='utf-8') as f:
    queries = json.load(f)

total_queries = len(queries)

# ----- CORE FUNCTIONS -----
def reranker(tools, user_prompt, top_k):
    pairs = [[user_prompt, json.dumps(tool)] for tool in tools]
    scores = reranker_model.predict(pairs)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [{"score": float(scores[i]), "tool": tools[i]} for i in top_indices]

class RouteDecision(BaseModel):
    reasoning: str = Field(description="step-by-step analysis comparing the user's request against the available tools before making a decision.")
    selected_tool: str = Field(description="The exact name of the tool. Return 'none' if no tool matches.")

def router(relevant_tools, model_name, user_prompt):
    tools_formatted = "\n".join([
        f"{t.get('name', 'Unknown')}: {t.get('description', '')} (Tags: {', '.join(t.get('tags', []))})" 
        for t in relevant_tools
    ])
    system_prompt = f"""You are a Router Agent expert in medical and dental imaging (CBCT, IOS, MRI).
    Your role is to analyze the user's request and select the most relevant tool from the FILTERED list below.
    If none of these {len(relevant_tools)} tools fit perfectly, return 'none'.

    === FILTERED TOOLS ===
    {tools_formatted}

    === ROUTING GUIDELINES & EXAMPLES ===
    - Pay close attention to subtle differences. For example, if a user specifically asks for "batch processing" or "multiple scans", prioritize tools designed for batching (e.g., batchdentalseg).
    - If a user asks to "segment" or "split" specific teeth, ensure the tool handles instance segmentation (e.g., amasss_cli).
    - If the request is for registration, check if it's CBCT-to-CBCT, MRI-to-CBCT, or intraoral (IOS) and choose the specific tool accordingly.

    Carefully analyze the user's prompt step-by-step in the 'reasoning' field BEFORE selecting the tool. Output strictly matching the JSON schema."""
    try: 
        response = chat(
            model=model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            format=RouteDecision.model_json_schema(),
            options={"temperature": 0.0, "num_predict": 1000},
        )
        return RouteDecision.model_validate_json(response.message.content)
    except Exception as e:
        return RouteDecision(selected_tool="error", reasoning=f"Error : {str(e)}")

class ParameterExtraction(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning for extraction these parameters from user's prompt.")
    parameters: dict = Field(description="Extracted parameters as a key-value dictionary. Use the exact parameter names from the tool schema.")

def agent(selected_tool_dict, model_name, user_prompt):
    tool_formatted = json.dumps(selected_tool_dict, indent=2)
    system_prompt = f"""You are en expert Parameter Extraction Agent for medical and dental imaging tools.
    Your task is to analyse the user's request and extract the required parameters according to the selected tool's schema
    
    SELECTED TOOL SCHEMA:
    {tool_formatted}

    carefully read the user's prompt and extract the appropriate values for the tool's parameters.
    Output strictly matching the JSON schema. if an optional parameter is missing, do not invent it"""
    try:
        response = chat(
            model=model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            format=ParameterExtraction.model_json_schema(),
            options={"temperature": 0.0, "num_predict": 1500},
        )
        return ParameterExtraction.model_validate_json(response.message.content)
    except Exception as e:
        return ParameterExtraction(parameters={}, reasoning=f"Error: {str(e)}")

def evaluate_agent(extracted_params, expected_params):
    agent_correct = 0
    for key, expected_val in expected_params.items():
        if extracted_params.get(key) == expected_val:
            agent_correct += 1
            
    if len(expected_params) > 0:
        agent_perfect = (agent_correct == len(expected_params)) and (len(extracted_params) == len(expected_params))
    else:
        agent_perfect = (len(extracted_params) == 0)

    return agent_perfect, agent_correct

# ----- BENCHMARK EXECUTION -----
benchmark_summary = {}

# 1. Precompute the Reranker (identical for all LLMs)
print("\nStep 1/2: Precomputing Reranker results (BAAI)...")
precomputed_rerank_results = []
total_expected_params = 0
reranker_correct_count = 0
reranker_perfect_count = 0

for q in tqdm(queries, desc="Reranking"):
    prompt = q['query']
    expected_tool = q['expected_tool']
    expected_params = q.get('expected_params', {})
    total_expected_params += len(expected_params)
    
    reranker_res = reranker(tools, prompt, top_k)
    top_tools_dicts = [res["tool"] for res in reranker_res]
    
    # Reranker evaluation
    reranker_tools_names = [t.get("name") for t in top_tools_dicts]
    if expected_tool in reranker_tools_names:
        reranker_correct_count += 1
    if expected_tool == reranker_tools_names[0]:
        reranker_perfect_count += 1
        
    precomputed_rerank_results.append({
        "query": q,
        "top_tools_dicts": top_tools_dicts
    })

# 2. Model Execution
print("\nStep 2/2: LLM Evaluation (Router & Agent)...")

for model in MODELS_TO_TEST:
    print(f"\n[{model}] Starting the test...")
    router_perfect_count = 0
    agent_correct_count = 0
    agent_perfect_count = 0
    results_detail = []
    
    start_time = time.time()
    
    for item in tqdm(precomputed_rerank_results, desc=model):
        q = item["query"]
        top_tools_dicts = item["top_tools_dicts"]
        prompt = q['query']
        expected_tool = q['expected_tool']
        expected_params = q.get('expected_params', {})

        # II. Router 
        decision = router(top_tools_dicts, model, prompt)
        selected_tool = decision.selected_tool
        router_perfect = (selected_tool == expected_tool)
        if router_perfect: 
            router_perfect_count += 1

        # III. Agent 
        extracted_params = {}
        if router_perfect:
            selected_tool_dict = next((t for t in tools if t.get("name") == selected_tool), None)
            if selected_tool_dict:
                param_response = agent(selected_tool_dict, model, prompt)
                extracted_params = param_response.parameters
                agent_perfect, agent_correct = evaluate_agent(extracted_params, expected_params)

                agent_correct_count += agent_correct
                if agent_perfect: 
                    agent_perfect_count += 1

        results_detail.append({
            "prompt": prompt,
            "expected_tool": expected_tool,
            "selected_tool": selected_tool,
            "expected_params": expected_params,
            "selected_params": extracted_params,
        })
        
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Compute the percentages
    router_acc = (router_perfect_count / total_queries) * 100
    agent_param_corr_acc = (agent_correct_count / total_expected_params * 100) if total_expected_params > 0 else 0
    agent_perf_acc = (agent_perfect_count / total_queries) * 100
    
    # Record the stats for this model
    safe_model_name = model.replace(':', '_').replace('.', '_')
    model_summary = {
        "model": model,
        "execution_time_seconds": round(execution_time, 2),
        "metrics": {
            "total_queries": total_queries,
            "Router_accuracy_pct": round(router_acc, 2),
            "Agent_Param_Correct_pct": round(agent_param_corr_acc, 2),
            "Agent_Param_Perfect_pct": round(agent_perf_acc, 2),
            "Raw_Counts": {
                "router_perfect": router_perfect_count,
                "agent_correct": agent_correct_count,
                "agent_perfect": agent_perfect_count
            }
        },
        "details": results_detail,
    }
    
    # Save the detailed JSON specific to the model
    model_result_path = result_dir / f'{safe_model_name}_{queries_type}.json'
    with open(model_result_path, 'w', encoding='utf-8') as f:
        json.dump(model_summary, f, indent=4, ensure_ascii=False)
        
    # Add to the global summary
    benchmark_summary[model] = model_summary["metrics"]
    benchmark_summary[model]["execution_time"] = round(execution_time, 2)
    
    # Quick print of the results
    print(f"  -> Router Acc: {round(router_acc, 2)}% | Agent Param Correct: {round(agent_param_corr_acc, 2)}% | Agent Perfect: {round(agent_perf_acc, 2)}% | Time: {round(execution_time,1)}s")

# ----- FINAL BENCHMARK EXPORT -----
print(f"\n{'='*50}")
print("🏁 BENCHMARK FINISHED 🏁")
print(f"{'='*50}")
print(f"Reranker (top-3): {round((reranker_correct_count/total_queries)*100, 2)}%")
print(f"Reranker (top-1): {round((reranker_perfect_count/total_queries)*100, 2)}%")
print("-" * 50)

for model, stats in benchmark_summary.items():
    print(f"{model}:")
    print(f"   Router Acc  : {stats['Router_accuracy_pct']}%")
    print(f"   Agent Params: {stats['Agent_Param_Correct_pct']}%")
    print(f"   Time        : {stats['execution_time']}s")
    print("-" * 50)

global_benchmark_path = result_dir / f'GLOBAL_BENCHMARK_{queries_type}.json'
with open(global_benchmark_path, 'w', encoding='utf-8') as f:
    json.dump({
        "reranker_stats": {
            "top_3_acc": round((reranker_correct_count/total_queries)*100, 2),
            "top_1_acc": round((reranker_perfect_count/total_queries)*100, 2)
        },
        "models": benchmark_summary
    }, f, indent=4, ensure_ascii=False)

print(f"\n📁 Global results saved to: {global_benchmark_path.resolve()}")