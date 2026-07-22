#!/usr/bin/env python3
"""
Parameter Validator - Validate and correct the extracted parameters
Step 2: Automatic Validation & Correction
"""

import json
import yaml
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path


class ParameterValidator:
    """Validate and correct the extracted parameters"""

    def __init__(self, manifest_path: str = "manifest.yaml"):
        """Load the manifest"""
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = yaml.safe_load(f)
        
        self.scripts = {s["name"]: s for s in self.manifest.get("scripts", [])}
    
    def validate(self, tool_name: str, extracted_params: Dict) -> Dict:
        """
        Validate the extracted parameters

        Returns:
            {
                "valid": bool,
                "params": dict (validated params),
                "errors": list,
                "warnings": list,
                "missing_required": list,
                "extra_params": list
            }
        """
        
        tool_spec = self.scripts.get(tool_name)
        if not tool_spec:
            return {
                "valid": False,
                "params": {},
                "errors": [f"Tool '{tool_name}' not found in manifest"],
                "warnings": [],
                "missing_required": [],
                "extra_params": []
            }
        
        errors = []
        warnings = []
        corrected_params = extracted_params.copy()
        
        # 1. CHECK REQUIRED PARAMETERS
        required_errors, missing_required = self._check_required_params(
            tool_spec, extracted_params
        )
        errors.extend(required_errors)

        # 2. CHECK TYPES
        type_errors, corrected_params = self._check_types(
            tool_spec, corrected_params
        )
        errors.extend(type_errors)

        # 3. CHECK PATHS (syntax)
        path_errors = self._check_paths(tool_spec, corrected_params)
        errors.extend(path_errors)

        # 4. CHECK ENUMS (choices)
        enum_errors = self._check_enums(tool_spec, corrected_params)
        errors.extend(enum_errors)

        # 5. CHECK RANGES (min/max)
        range_errors = self._check_ranges(tool_spec, corrected_params)
        errors.extend(range_errors)

        # 6. CHECK EXTRA PARAMETERS
        extra_params = self._check_extra_params(tool_spec, extracted_params)
        if extra_params:
            warnings.append(f"Extra parameters not in spec: {extra_params}")
        
        return {
            "valid": len(errors) == 0,
            "params": corrected_params,
            "errors": errors,
            "warnings": warnings,
            "missing_required": missing_required,
            "extra_params": extra_params
        }
    
    def _check_required_params(
        self,
        tool_spec: Dict,
        params: Dict
    ) -> Tuple[List[str], List[str]]:
        """Check that the required parameters are present"""
        
        errors = []
        missing = []
        
        for param in tool_spec.get("parameters", []):
            name = param["name"]
            if param.get("required", False) and name not in params:
                msg = f"MISSING REQUIRED PARAMETER: {name}"
                errors.append(msg)
                missing.append(name)
        
        return errors, missing
    
    def _check_types(
        self,
        tool_spec: Dict,
        params: Dict
    ) -> Tuple[List[str], Dict]:
        """
        Check and convert the types
        Returns (errors, corrected_params)
        """
        
        errors = []
        corrected = params.copy()
        
        for param in tool_spec.get("parameters", []):
            name = param["name"]
            if name not in params:
                continue
            
            ptype = param.get("type", "string")
            value = params[name]
            
            try:
                # Conversion attempt
                if ptype == "bool":
                    corrected[name] = self._convert_bool(value)
                
                elif ptype == "int":
                    corrected[name] = self._convert_int(value)
                
                elif ptype == "float":
                    corrected[name] = self._convert_float(value)
                
                elif ptype == "list":
                    encode = param.get("encode", "")
                    corrected[name] = self._convert_list(value, encode)
                
                elif ptype == "path":
                    corrected[name] = self._convert_path(value)
                
                else:  # string
                    corrected[name] = str(value)
            
            except ValueError as e:
                errors.append(f"TYPE ERROR in '{name}': {str(e)}")
        
        return errors, corrected
    
    def _check_paths(self, tool_spec: Dict, params: Dict) -> List[str]:
        """Check the syntax of the paths"""
        
        errors = []
        
        for param in tool_spec.get("parameters", []):
            name = param["name"]
            if name not in params or param.get("type") != "path":
                continue
            
            value = params[name]
            
            # Check that it's a valid string
            try:
                Path(str(value))
            except Exception as e:
                errors.append(f"INVALID PATH '{name}': {value}")

            # Check that folder paths do not have an extension
            if "folder" in name or "dir" in name:
                str_value = str(value)
                if str_value.endswith((".nii.gz", ".nii", ".stl", ".ply", ".vtk", ".json", ".txt", ".log")):
                    errors.append(
                        f"FOLDER PATH SHOULD NOT HAVE EXTENSION '{name}': {value}. "
                        f"Remove the filename part."
                    )
        
        return errors
    
    def _check_enums(self, tool_spec: Dict, params: Dict) -> List[str]:
        """Check that the values are part of the choices"""
        
        errors = []
        
        for param in tool_spec.get("parameters", []):
            name = param["name"]
            if name not in params or "choices" not in param:
                continue
            
            value = params[name]
            choices = param["choices"]
            
            if value not in choices:
                errors.append(
                    f"INVALID VALUE '{name}': {value}. "
                    f"Must be one of: {choices}"
                )
        
        return errors
    
    def _check_ranges(self, tool_spec: Dict, params: Dict) -> List[str]:
        """Check the ranges (min/max)"""
        
        errors = []
        
        for param in tool_spec.get("parameters", []):
            name = param["name"]
            if name not in params:
                continue
            
            value = params[name]
            
            # Check min
            if "min" in param:
                try:
                    if float(value) < float(param["min"]):
                        errors.append(
                            f"VALUE TOO SMALL '{name}': {value} < {param['min']}"
                        )
                except:
                    pass  # Skip if not a number

            # Check max
            if "max" in param:
                try:
                    if float(value) > float(param["max"]):
                        errors.append(
                            f"VALUE TOO LARGE '{name}': {value} > {param['max']}"
                        )
                except:
                    pass  # Skip if not a number
        
        return errors
    
    def _check_extra_params(self, tool_spec: Dict, params: Dict) -> List[str]:
        """Check the extra parameters"""
        
        spec_names = {p["name"] for p in tool_spec.get("parameters", [])}
        param_names = set(params.keys())
        
        return list(param_names - spec_names)
    
    # ===== CONVERSION METHODS =====
    
    @staticmethod
    def _convert_bool(value) -> bool:
        """Convert to boolean"""
        if isinstance(value, bool):
            return value
        
        if isinstance(value, str):
            if value.lower() in ("true", "yes", "1", "on"):
                return True
            elif value.lower() in ("false", "no", "0", "off"):
                return False
        
        if isinstance(value, int):
            return value != 0
        
        raise ValueError(f"Cannot convert '{value}' to boolean")
    
    @staticmethod
    def _convert_int(value) -> int:
        """Convert to integer"""
        if isinstance(value, int):
            return value
        
        try:
            return int(float(str(value)))
        except:
            raise ValueError(f"Cannot convert '{value}' to integer")
    
    @staticmethod
    def _convert_float(value) -> float:
        """Convert to float"""
        if isinstance(value, (int, float)):
            return float(value)
        
        try:
            return float(str(value))
        except:
            raise ValueError(f"Cannot convert '{value}' to float")
    
    @staticmethod
    def _convert_list(value, encode: str) -> Any:
        """Convert to a list according to the encoding"""

        if isinstance(value, list):
            return value

        if isinstance(value, str):
            s = value.strip()

            # If python_literal, return as-is
            if encode == "python_literal":
                return s  # "[1.0, 0.3]"

            # Parse according to the separator
            if encode == "space_separated":
                items = s.split()
            else:  # comma_no_brackets, comma_separated
                # Remove the brackets if present
                if s.startswith("[") and s.endswith("]"):
                    s = s[1:-1]
                items = [x.strip() for x in s.split(",")]
            
            return items
        
        raise ValueError(f"Cannot convert '{value}' to list")
    
    @staticmethod
    def _convert_path(value) -> str:
        """Convert to a path (just clean it up)"""
        return str(value).strip()


class ValidationReport:
    """Generate a readable validation report"""

    @staticmethod
    def generate_report(tool_name: str, validation_result: Dict) -> str:
        """Generate a report in a readable format"""
        
        report = []
        report.append(f"\n{'='*80}")
        report.append(f"VALIDATION REPORT FOR: {tool_name}")
        report.append(f"{'='*80}")
        
        # Status
        status = "✅ VALID" if validation_result["valid"] else "❌ INVALID"
        report.append(f"\nStatus: {status}")
        
        # Missing required
        if validation_result["missing_required"]:
            report.append(f"\n❌ Missing Required Parameters:")
            for param in validation_result["missing_required"]:
                report.append(f"   - {param}")
        
        # Errors
        if validation_result["errors"]:
            report.append(f"\n❌ Errors:")
            for error in validation_result["errors"]:
                report.append(f"   - {error}")
        
        # Warnings
        if validation_result["warnings"]:
            report.append(f"\n⚠️  Warnings:")
            for warning in validation_result["warnings"]:
                report.append(f"   - {warning}")
        
        # Extra params
        if validation_result["extra_params"]:
            report.append(f"\n⚠️  Extra Parameters (not in spec):")
            for param in validation_result["extra_params"]:
                report.append(f"   - {param}")
        
        # Validated params
        if validation_result["params"]:
            report.append(f"\n✅ Validated Parameters:")
            for key, value in validation_result["params"].items():
                if isinstance(value, list):
                    report.append(f"   {key}: {value} (list with {len(value)} items)")
                else:
                    report.append(f"   {key}: {value}")
        
        report.append(f"\n{'='*80}\n")
        
        return "\n".join(report)


if __name__ == "__main__":
    # Test
    validator = ParameterValidator()
    
    print("=" * 80)
    print("TEST 1: ALI_CBCT with all parameters correct")
    print("=" * 80)
    
    test_params_1 = {
        "input": "/data/cbct_patient1.nii.gz",
        "dir_models": "/models/ali_cbct",
        "lm_type": "1,2,3,4,5",
        "output_dir": "/output",
        "temp_fold": "/tmp/ali_cbct",
        "DCMInput": "false",  # String, will be converted
        "spacing": "[1.0, 0.3]",
        "speed_per_scale": "[1, 1]",
        "agent_FOV": "[64, 64, 64]",
        "spawn_radius": "10"  # String, will be converted
    }
    
    result_1 = validator.validate("ali_cbct", test_params_1)
    print(ValidationReport.generate_report("ali_cbct", result_1))
    
    print("\n" + "=" * 80)
    print("TEST 2: ALI_CBCT with a missing required parameter")
    print("=" * 80)
    
    test_params_2 = {
        "input": "/data/cbct.nii.gz",
        "dir_models": "/models/ali_cbct",
        # lm_type missing!
        "output_dir": "/output",
        "temp_fold": "/tmp/ali_cbct"
    }
    
    result_2 = validator.validate("ali_cbct", test_params_2)
    print(ValidationReport.generate_report("ali_cbct", result_2))
    
    print("\n" + "=" * 80)
    print("TEST 3: AREG_CBCT with a path error (folder with extension)")
    print("=" * 80)
    
    test_params_3 = {
        "t1_folder": "/data/cbct_baseline.nii.gz",  # ERROR!
        "t2_folder": "/data/cbct_followup",
        "output_folder": "/output",
        "reg_type": "MAND",
        "temp_folder": "/tmp/areg_cbct",
        "ApproxReg": False,
        "mask_folder_t1": "None",
        "DCMInput": False
    }
    
    result_3 = validator.validate("areg_cbct", test_params_3)
    print(ValidationReport.generate_report("areg_cbct", result_3))
    
    print("\n" + "=" * 80)
    print("TEST 4: ALI_IOS with type conversion")
    print("=" * 80)
    
    test_params_4 = {
        "input_surface": "/data/ios_scan.stl",
        "dir_models": "/models/ali_ios",
        "lm_type": "1 2 3 4 5",
        "teeth": "1 2 3",
        "output_dir": "/output",
        "log_path": "/logs/ali_ios.log"
    }
    
    result_4 = validator.validate("ali_ios", test_params_4)
    print(ValidationReport.generate_report("ali_ios", result_4))
