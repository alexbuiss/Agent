import pandas as pd
import json
import numpy as np

def main():
    all_data = []
    list_json = ["beginners", "expert vocab", "missingparam", "normal vocab", "french", "portuguese", "spanish", "short", "long"]
    
    for query in list_json:
        try:
            with open(f"Param_and_cli/results/results_{query}.json", "r") as f:
                model_results = json.load(f)
            
            for result in model_results:
                model = result.get("model", "Unknown")
                latency = result.get("latency_stats", {})
                
                # We create a dictionary for each metric for this model and this JSON
                metrics = {
                    "total_tests": int(result.get("total_tests")),
                    "avg_tool_selection": np.round(result.get("avg_tool_selection_accuracy"),3),
                    "combined_perfect": int(result.get("combined_perfect")),
                    "param_extraction_perfect": int(result.get("parameter_extraction_perfect")),
                    "avg_param_accuracy": np.round(result.get("avg_parameter_accuracy"),3),
                    "latency_mean": np.round(latency.get("mean_s"),3),
                }
                
                for metric_name, value in metrics.items():
                    all_data.append({
                        "json_file": query,
                        "metric": metric_name,
                        "model": model,
                        "value": value
                    })
        except FileNotFoundError:
            print(f"File results_{query}.json not found.")

    # Create the initial DataFrame
    df_raw = pd.DataFrame(all_data)

    # Pivot to put the models in columns
    # The index will be composed of the JSON name and the metric
    df = df_raw.pivot(index=["json_file", "metric"], columns="model", values="value")

    final_rows = []
    # We iterate over the JSON files actually present in the DataFrame
    existing_jsons = df.index.get_level_values(0).unique()

    for json_name in existing_jsons:
        # 1. Add the JSON data block
        group = df.loc[json_name]
        # We put the JSON name back into the index so that the concat works
        group.index = pd.MultiIndex.from_product([[json_name], group.index])
        final_rows.append(group)

        # 2. Add the empty separator row
        # We use a string with a space " " for the index so that it is distinct
        empty_index = pd.MultiIndex.from_tuples([(f" ", " ")], names=["json_file", "metric"])
        empty_row = pd.DataFrame(index=empty_index, columns=df.columns)
        final_rows.append(empty_row)

    # Final concatenation
    df_final = pd.concat(final_rows)

    # Display
    print(df_final)
    # To save to Excel/CSV more easily:
    df_final.to_excel("resultats_full medgemma.xlsx")
    return df

if __name__ == "__main__":
    df_final = main()