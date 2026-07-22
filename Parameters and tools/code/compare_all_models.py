#!/usr/bin/env python3
"""
Comprehensive Model Performance Comparison
Compare all LLM models on tool selection AND parameter extraction
with improved extraction module on all 20 benchmark cases.
"""

import json
import yaml
import subprocess
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import statistics

# Import improved modules
try:
    from parameter_extraction_improved import ImprovedParameterExtractor
    from parameter_validator import ParameterValidator
    IMPROVED_AVAILABLE = True
except ImportError:
    IMPROVED_AVAILABLE = False
    print("⚠️  Improved modules not available")
    sys.exit(1)

class ComprehensiveModelComparator:
    def __init__(self, manifest_file, queries_file):
        # Load manifest
        with open(manifest_file, "r") as f:
            self.manifest = yaml.safe_load(f)
        
        # Load test queries
        with open(queries_file, "r") as f:
            self.test_cases = json.load(f)

        # Initialize improvement modules
        self.improved_extractor = ImprovedParameterExtractor(manifest_file)
        self.validator = ParameterValidator(manifest_file)
        
        print(f"✅ Loaded {len(self.test_cases)} benchmark test cases")
        print(f"✅ Initialized improved extraction & validation modules")
        print(f"✅ Manifest contains {len(self.manifest.get('scripts', []))} tools\n")
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                print("⚠️  Could not get model list from Ollama")
                return []
            
            # Parse ollama list output
            models = []
            for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                if not "deep" in line.strip():
                    model_name = line.split()[0]
                    models.append(model_name)
            
            return sorted(models)
        
        except Exception as e:
            print(f"⚠️  Error getting models: {e}")
            return []
    
    def compare_parameters(self, extracted: Dict, expected: Dict) -> Tuple[bool, float, List[str]]:
        """Compare extracted vs expected parameters."""
        errors = []
        correct_count = 0
        total_expected = len(expected)
        
        for param_name, expected_value in expected.items():
            if param_name not in extracted:
                errors.append(f"Missing: {param_name}")
                continue
            
            extracted_value = extracted[param_name]
            
            # Flexible type comparison
            if self._values_match(extracted_value, expected_value):
                correct_count += 1
            else:
                errors.append(f"'{param_name}': expected {repr(expected_value)[:30]}, got {repr(extracted_value)[:30]}")
        
        # Check for extra parameters
        for param_name in extracted:
            if param_name not in expected:
                errors.append(f"Extra: {param_name}")
        
        accuracy = correct_count / total_expected if total_expected > 0 else 0.0
        is_perfect = len(errors) == 0
        
        return is_perfect, accuracy, errors
    
    def _values_match(self, extracted, expected) -> bool:
        """Check if two values match with flexible comparison."""
        if extracted == expected:
            return True
        
        try:
            if isinstance(expected, str) and isinstance(extracted, str):
                return expected.strip() == extracted.strip()
            
            if isinstance(expected, (list, tuple)) and isinstance(extracted, (list, tuple)):
                return set(str(x) for x in expected) == set(str(x) for x in extracted)
            
            if isinstance(expected, bool):
                if isinstance(extracted, bool):
                    return extracted == expected
                if isinstance(extracted, str):
                    return str(extracted).lower() == str(expected).lower()
            
            if isinstance(expected, (int, float)):
                return float(extracted) == float(expected)
            
            return str(extracted).lower() == str(expected).lower()
        
        except (ValueError, TypeError):
            return False
    
    def test_model(self, model_name: str) -> Dict:
        """Test a specific model on all benchmark cases."""
        
        results = {
            "model": model_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_tests": len(self.test_cases),
            "tool_selection_correct": 0,
            "tool_selection_total": 0,
            "parameter_extraction_perfect": 0,
            "parameter_extraction_total": 0,
            "combined_perfect": 0,
            "avg_tool_selection_accuracy": 0.0,
            "avg_parameter_accuracy": 0.0,
            "avg_combined_accuracy": 0.0,
            "test_details": [],
            "latencies": [],
            "errors": []
        }
        
        print(f"\n{'='*90}")
        print(f"TESTING MODEL: {model_name}")
        print(f"TOTAL TEST CASES TO RUN: {len(self.test_cases)}")
        print(f"{'='*90}\n")
        
        tool_selection_accuracies = []
        parameter_accuracies = []
        combined_accuracies = []
        
        for i, test_case in enumerate(self.test_cases, 1):
            query = test_case.get("query", "")
            expected_tool = test_case.get("expected_tool", "")
            expected_params = test_case.get("expected_params", {})
            
            try:
                start_time = time.time()
                
                # Run agent_router with the model
                env = os.environ.copy()
                env["ROUTER_MODEL"] = model_name
                
                cmd = [
                    "python", "Param_and_cli/code/agent_router.py", query,
                    "--benchmark", "--dry-run"
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=150,  # Increased timeout to 60s
                    env=env
                )
                
                latency = time.time() - start_time
                results["latencies"].append(latency)
                
                if result.returncode not in [0, 2]:  # Accept both 0 and 2 (benchmark exit code)
                    error_msg = f"Exit code {result.returncode}"
                    if result.stderr:
                        error_msg += f" - {result.stderr[:100]}"
                    results["errors"].append({
                        "test": i,
                        "query": query[:80],
                        "error": error_msg
                    })
                    print(f"   [{i:2d}/{len(self.test_cases)}] ❌ ERROR: {error_msg[:60]}")
                    continue
                
                # Parse output
                if not result.stdout or not result.stdout.strip():
                    error_msg = f"Empty output from agent_router (exit code {result.returncode})"
                    if result.stderr:
                        error_msg += f" - stderr: {result.stderr[:100]}"
                    results["errors"].append({
                        "test": i,
                        "query": query[:80],
                        "error": error_msg,
                        "stderr": result.stderr[:200] if result.stderr else ""
                    })
                    print(f"   [{i:2d}/{len(self.test_cases)}] ❌ EMPTY OUTPUT: {error_msg[:60]}")
                    continue
                
                try:
                    output = json.loads(result.stdout)
                except json.JSONDecodeError as e:
                    error_msg = f"JSON parse error: {str(e)[:50]}"
                    results["errors"].append({
                        "test": i,
                        "query": query[:80],
                        "error": error_msg,
                        "stdout": result.stdout[:200] if result.stdout else ""
                    })
                    print(f"   [{i:2d}/{len(self.test_cases)}] ❌ JSON ERROR: {error_msg[:60]}")
                    continue
                
                # Extract results
                selected_tool = output.get("tool", "")
                extracted_params = output.get("parameters", {})
                
                # Tool selection accuracy
                tool_correct = selected_tool == expected_tool
                results["tool_selection_total"] += 1
                if tool_correct:
                    results["tool_selection_correct"] += 1
                
                # Parameter extraction accuracy (with improved conversion)
                param_accuracy = 0.0
                is_perfect = False
                
                if tool_correct:  # Only assess params if tool is correct
                    # Get tool spec
                    tool_spec = next(
                        (s for s in self.manifest.get("scripts", []) if s["name"] == expected_tool),
                        None
                    )
                    
                    if tool_spec:
                        # Convert types using improved extractor
                        converted, conv_errors = self.improved_extractor.convert_types(
                            extracted_params, tool_spec
                        )
                        
                        # Compare
                        is_perfect, param_accuracy, param_errors = self.compare_parameters(
                            converted, expected_params
                        )
                        
                        results["parameter_extraction_total"] += 1
                        if is_perfect:
                            results["parameter_extraction_perfect"] += 1
                        
                        if is_perfect:
                            results["combined_perfect"] += 1
                    else:
                        param_accuracy = 0.0
                        is_perfect = False
                else:
                    param_accuracy = 0.0
                    is_perfect = False
                
                parameter_accuracies.append(param_accuracy)
                tool_selection_accuracies.append(1.0 if tool_correct else 0.0)
                combined_acc = 1.0 if is_perfect else param_accuracy
                combined_accuracies.append(combined_acc)
                
                # ✅ PRINT DETAILED PROGRESS FOR THIS QUERY
                tool_status = "✅" if tool_correct else "❌"
                param_status = "✅" if is_perfect else f"{param_accuracy*100:.0f}%"
                
                print(f"\n   [{i:2d}/{len(self.test_cases)}] {tool_status} QUERY: {query[:65]}")
                print(f"           Expected Tool: {expected_tool:<20} | Got: {selected_tool:<20}")
                if expected_params and extracted_params:
                    print(f"           Expected Params: {str(expected_params)[:60]}")
                    print(f"           Extracted Params: {str(extracted_params)[:60]}")
                print(f"           Param Accuracy: {param_status:>4} | Latency: {latency:.3f}s")
                
                results["test_details"].append({
                    "index": i,
                    "query": query[:80],
                    "expected_tool": expected_tool,
                    "selected_tool": selected_tool,
                    "tool_correct": tool_correct,
                    "param_accuracy": param_accuracy,
                    "is_perfect": is_perfect,
                    "latency": latency
                })
            
            except subprocess.TimeoutExpired:
                error_msg = "Timeout (60s)"
                results["errors"].append({
                    "test": i,
                    "query": query[:80],
                    "error": error_msg
                })
                print(f"   [{i:2d}/{len(self.test_cases)}] ⏱️  TIMEOUT: {error_msg}")
            except Exception as e:
                error_msg = str(e)[:100]
                results["errors"].append({
                    "test": i,
                    "query": query[:80],
                    "error": error_msg
                })
                print(f"   [{i:2d}/{len(self.test_cases)}] 💥 EXCEPTION: {error_msg}")
        
        # Calculate averages - safely
        if tool_selection_accuracies and len(tool_selection_accuracies) > 0:
            results["avg_tool_selection_accuracy"] = sum(tool_selection_accuracies) / len(tool_selection_accuracies)
        else:
            results["avg_tool_selection_accuracy"] = 0.0
        
        if parameter_accuracies and len(parameter_accuracies) > 0:
            results["avg_parameter_accuracy"] = sum(parameter_accuracies) / len(parameter_accuracies)
        else:
            results["avg_parameter_accuracy"] = 0.0
        
        if combined_accuracies and len(combined_accuracies) > 0:
            results["avg_combined_accuracy"] = sum(combined_accuracies) / len(combined_accuracies)
        else:
            results["avg_combined_accuracy"] = 0.0
        
        if results["latencies"] and len(results["latencies"]) > 0:
            results["latency_stats"] = {
                "mean_s": statistics.mean(results["latencies"]),
                "median_s": statistics.median(results["latencies"]),
                "min_s": min(results["latencies"]),
                "max_s": max(results["latencies"])
            }
        else:
            results["latency_stats"] = {
                "mean_s": 0.0,
                "median_s": 0.0,
                "min_s": 0.0,
                "max_s": 0.0
            }
        
        # Print model summary with more details
        print(f"\n{'='*90}")
        print(f"📊 SUMMARY FOR {model_name}:")
        print(f"{'='*90}")
        
        tool_tests = results['tool_selection_total']
        param_tests = results['parameter_extraction_total']
        
        if tool_tests > 0:
            print(f"  ✅ Tool Selection:       {results['avg_tool_selection_accuracy']*100:.1f}% ({results['tool_selection_correct']}/{tool_tests})")
        else:
            print(f"  ⚠️  Tool Selection:       No valid tests")
            
        if param_tests > 0:
            print(f"  ✅ Parameter Extraction: {results['avg_parameter_accuracy']*100:.1f}% ({results['parameter_extraction_perfect']}/{param_tests})")
        else:
            print(f"  ⚠️  Parameter Extraction: No valid tests")
            
        print(f"  ✅ Combined Accuracy:    {results['avg_combined_accuracy']*100:.1f}%")
        
        if results["latencies"] and len(results["latencies"]) > 0:
            avg_latency = statistics.mean(results['latencies'])
            print(f"  ⏱️  Avg Latency:          {avg_latency:.3f}s ({avg_latency*1000:.1f}ms)")
            print(f"  ⏱️  Min/Max Latency:      {min(results['latencies']):.3f}s / {max(results['latencies']):.3f}s")
        
        if results["errors"] and len(results["errors"]) > 0:
            print(f"  ⚠️  ERRORS:              {len(results['errors'])} error(s)")
            for err in results["errors"][:3]:  # Show first 3 errors
                print(f"      • Test {err['test']}: {err['error'][:70]}")
            if len(results["errors"]) > 3:
                print(f"      ... and {len(results['errors'])-3} more errors")
        else:
            print(f"  ✅ NO ERRORS")
            
        print(f"  📝 Total Queries Tested: {len(results['test_details'])}/{len(self.test_cases)}")
        print(f"{'='*90}\n")
        
        return results
    
    def run_comparison(self, models: List[str]) -> List[Dict]:
        """Compare all models."""
        
        all_results = []
        
        for model_name in models:
            try:
                results = self.test_model(model_name)
                all_results.append(results)
            except Exception as e:
                print(f"❌ Error testing {model_name}: {e}")
        
        return all_results
    
    def print_summary(self, results: List[Dict]):
        """Print summary comparison."""
        
        print(f"\n{'='*130}")
        print(f"🏆 COMPREHENSIVE MODEL COMPARISON SUMMARY")
        print(f"{'='*130}\n")
        
        # Sort by combined accuracy
        sorted_results = sorted(
            results,
            key=lambda x: x["avg_combined_accuracy"],
            reverse=True
        )
        
        print(f"{'Rank':<5} {'Model':<25} {'Tool Sel':<12} {'Param Ext':<12} {'Combined':<12} {'Latency':<12} {'Tests':<8}")
        print(f"{'-'*130}")
        
        for rank, result in enumerate(sorted_results, 1):
            tool_acc = f"{result['avg_tool_selection_accuracy']*100:.1f}%"
            param_acc = f"{result['avg_parameter_accuracy']*100:.1f}%"
            combined_acc = f"{result['avg_combined_accuracy']*100:.1f}%"
            
            if "latency_stats" in result and result['latency_stats']['mean_s'] > 0:
                latency = f"{result['latency_stats']['mean_s']*1000:.1f}ms"
            else:
                latency = "N/A"
            
            tests = f"{len(result['test_details'])}/{len(self.test_cases)}"
            
            # Add star for best performer
            star = "⭐ " if rank == 1 else ""
            print(f"{star}{rank:<5} {result['model']:<25} {tool_acc:<12} {param_acc:<12} {combined_acc:<12} {latency:<12} {tests:<8}")
        
        # Best model
        best_model = sorted_results[0]
        print(f"\n🏆 BEST MODEL: {best_model['model']}")
        print(f"   Tool Selection: {best_model['avg_tool_selection_accuracy']*100:.1f}%")
        print(f"   Parameter Extraction: {best_model['avg_parameter_accuracy']*100:.1f}%")
        print(f"   Combined Accuracy: {best_model['avg_combined_accuracy']*100:.1f}%")
        
        if "latency_stats" in best_model:
            print(f"   Latency (avg): {best_model['latency_stats']['mean_s']*1000:.1f}ms")
        
        # Statistics
        combined_scores = [r["avg_combined_accuracy"] for r in sorted_results]
        print(f"\n📊 Statistics:")
        print(f"   Models tested: {len(sorted_results)}")
        print(f"   Best combined accuracy: {max(combined_scores)*100:.1f}%")
        print(f"   Worst combined accuracy: {min(combined_scores)*100:.1f}%")
        print(f"   Average: {statistics.mean(combined_scores)*100:.1f}%")
        
        if len(combined_scores) > 1:
            print(f"   Std Dev: {statistics.stdev(combined_scores)*100:.1f}%")
    
    def save_results(self, results: List[Dict], output_file: str = None,queryfile:str = "") -> str:
        """Save results to JSON."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"Param_and_cli/results/results_{queryfile}_new.json"
        
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_file}")
        return output_file

def main(query_file):
    print("\n" + "="*110)
    print("🚀 COMPREHENSIVE MODEL PERFORMANCE COMPARISON")
    print("Testing all LLM models on tool selection + parameter extraction")
    print("="*110 + "\n")
    
    # Initialize comparator
    comparator = ComprehensiveModelComparator(
        "manifest.yaml",
        f"Param_and_cli/queries/{query_file}.json"
    )
    
    # Get available models
    print("📊 Getting available models from Ollama...")
    available_models = comparator.get_available_models()
    
    if not available_models:
        print("❌ No models available! Please ensure Ollama is running and has models installed.")
        print("   Run: ollama list")
        sys.exit(1)
    
    print(f"✅ Found {len(available_models)} models:")
    for model in available_models:
        print(f"   • {model}")
    
    # Run comparison
    results = comparator.run_comparison(available_models)
    
    # Print summary
    if results:
        comparator.print_summary(results)
        
        # Save results
        output_file = comparator.save_results(results,queryfile=query_file)
        
        print(f"\n✨ COMPARISON COMPLETE\n")
    else:
        print("❌ No results to compare")
        sys.exit(1)

if __name__ == "__main__":
    main(sys.argv[1])
