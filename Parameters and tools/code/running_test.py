import compare_all_models
import compare_params_performance
import export_result
import json

def main():
    list_json = ["french"]#,"expert vocab","missingparam","normal vocab","portuguese","spanish","beginners","short","long"]
    for query in list_json:
        compare_all_models.main(query)
    # export_result.main()
    

if __name__ == "__main__":
    main()

