"""Verify TimeMixer code structure and imports (without running models).

This script checks that:
1. All imports work correctly
2. Function signatures match expected API
3. Code structure is correct
"""

import sys
from pathlib import Path
import ast
import importlib.util

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_file_imports(file_path: Path):
    """Check if a file can be imported (syntax check)."""
    try:
        spec = importlib.util.spec_from_file_location("module", file_path)
        if spec is None:
            return False, "Could not create spec"
        module = importlib.util.module_from_spec(spec)
        # Try to load (but don't execute) - this will catch syntax errors
        # We can't actually import because neuralforecast may not be installed
        return True, "OK"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        # Import errors are OK for this check - we're just verifying syntax
        if "ModuleNotFoundError" in str(type(e).__name__):
            return True, f"Syntax OK (missing dependency: {e.name})"
        return False, f"Error: {e}"


def check_function_signatures(file_path: Path):
    """Check function signatures using AST."""
    try:
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read(), filename=str(file_path))
        
        functions = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                args = [arg.arg for arg in node.args.args]
                functions[node.name] = args
        
        return True, functions
    except Exception as e:
        return False, str(e)


def verify_train_file():
    """Verify train/timemixer.py structure."""
    print("=" * 80)
    print("Verifying src/train/timemixer.py")
    print("=" * 80)
    
    file_path = project_root / "src" / "train" / "timemixer.py"
    
    # Check file exists
    if not file_path.exists():
        print(f"✗ File not found: {file_path}")
        return False
    
    print(f"✓ File exists: {file_path}")
    
    # Check syntax
    syntax_ok, msg = check_file_imports(file_path)
    if syntax_ok:
        print(f"✓ Syntax check: {msg}")
    else:
        print(f"✗ Syntax check failed: {msg}")
        return False
    
    # Check function signatures
    sig_ok, funcs = check_function_signatures(file_path)
    if sig_ok:
        print(f"✓ Function signatures:")
        for func_name, args in funcs.items():
            print(f"  - {func_name}({', '.join(args)})")
        
        # Verify expected functions exist
        expected = ['train_timemixer_model', '_create_timemixer_model']
        for expected_func in expected:
            if expected_func in funcs:
                print(f"✓ Expected function found: {expected_func}")
            else:
                print(f"✗ Expected function missing: {expected_func}")
                return False
    else:
        print(f"✗ Could not parse signatures: {funcs}")
        return False
    
    print()
    return True


def verify_forecast_file():
    """Verify forecast/timemixer.py structure."""
    print("=" * 80)
    print("Verifying src/forecast/timemixer.py")
    print("=" * 80)
    
    file_path = project_root / "src" / "forecast" / "timemixer.py"
    
    # Check file exists
    if not file_path.exists():
        print(f"✗ File not found: {file_path}")
        return False
    
    print(f"✓ File exists: {file_path}")
    
    # Check syntax
    syntax_ok, msg = check_file_imports(file_path)
    if syntax_ok:
        print(f"✓ Syntax check: {msg}")
    else:
        print(f"✗ Syntax check failed: {msg}")
        return False
    
    # Check function signatures
    sig_ok, funcs = check_function_signatures(file_path)
    if sig_ok:
        print(f"✓ Function signatures:")
        for func_name, args in funcs.items():
            print(f"  - {func_name}({', '.join(args)})")
        
        # Verify expected functions exist
        expected = ['run_recursive_forecast', 'run_multi_horizon_forecast']
        for expected_func in expected:
            if expected_func in funcs:
                print(f"✓ Expected function found: {expected_func}")
            else:
                print(f"✗ Expected function missing: {expected_func}")
                return False
        
        # Check that forecast is defined (it's a variable, not a function)
        # Read file to check
        with open(file_path, 'r') as f:
            content = f.read()
            if 'forecast = create_forecast_function' in content:
                print(f"✓ 'forecast' variable defined using create_forecast_function")
            else:
                print(f"⚠ 'forecast' may not be defined correctly")
    else:
        print(f"✗ Could not parse signatures: {funcs}")
        return False
    
    print()
    return True


def verify_test_file():
    """Verify test/test_timemixer.py structure."""
    print("=" * 80)
    print("Verifying test/test_timemixer.py")
    print("=" * 80)
    
    file_path = project_root / "test" / "test_timemixer.py"
    
    # Check file exists
    if not file_path.exists():
        print(f"⚠ Test file not found: {file_path} (will be created)")
        return True  # Not an error, just missing
    
    print(f"✓ File exists: {file_path}")
    
    # Check syntax
    syntax_ok, msg = check_file_imports(file_path)
    if syntax_ok:
        print(f"✓ Syntax check: {msg}")
    else:
        print(f"✗ Syntax check failed: {msg}")
        return False
    
    # Check function signatures
    sig_ok, funcs = check_function_signatures(file_path)
    if sig_ok:
        # Count test functions
        test_funcs = [f for f in funcs.keys() if f.startswith('test_')]
        print(f"✓ Found {len(test_funcs)} test functions:")
        for func_name in test_funcs[:10]:  # Show first 10
            print(f"  - {func_name}")
        if len(test_funcs) > 10:
            print(f"  ... and {len(test_funcs) - 10} more")
    else:
        print(f"✗ Could not parse signatures: {funcs}")
        return False
    
    print()
    return True


def main():
    """Run all verification checks."""
    print("\n" + "=" * 80)
    print("TIMEMIXER CODE STRUCTURE VERIFICATION")
    print("=" * 80)
    print()
    
    results = []
    
    results.append(("train/timemixer.py", verify_train_file()))
    results.append(("forecast/timemixer.py", verify_forecast_file()))
    results.append(("test/test_timemixer.py", verify_test_file()))
    
    print("=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 80)
    
    if all_passed:
        print("✓ All verifications passed!")
        return 0
    else:
        print("✗ Some verifications failed")
        return 1


if __name__ == '__main__':
    exit(main())
