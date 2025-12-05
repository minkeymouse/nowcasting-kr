# Pytest Issues Resolution Plan

**Created**: 2025-01-04  
**Current Status**: 27 failed, 110 passed, 16 skipped, 1 error (154 total tests)  
**Previous Status**: 33 failed, 104 passed, 16 skipped, 1 error

## Executive Summary

After running pytest, we identified **27 failures and 1 error** that need to be resolved. The issues fall into several categories:

1. **Config Property Setter Issue** (8 failures) - CRITICAL
2. **Function Signature Mismatches** (2 failures)
3. **Type Errors** (2 failures: 1 error + 1 failure)
4. **Test Expectations** (2 failures)
5. **Missing Attributes** (2 failures)
6. **DataModule Setup Issues** (2 failures)
7. **Trainer Configuration** (5 failures)
8. **Empty Test File** (1 file needs implementation)

## Detailed Issue Analysis

### Category 1: Config Property Setter Issue (8 failures) - CRITICAL

**Root Cause**: In `dfm.py` line 508, `self.config = config` is called, but `config` is a read-only property (defined in `base.py` line 44-47) without a setter. The code also correctly sets `self._config = config` on line 509.

**Affected Tests**:
- `test_base_factor_model_interface`
- `test_dfm_initialization`
- `test_dfm_load_config`
- `test_dfm_pipeline_config_loading` (test_pipeline.py)
- `test_dfm_pipeline_training` (test_pipeline.py)
- `test_dfm_pipeline_prediction` (test_pipeline.py)
- `test_dfm_pipeline_complete` (test_pipeline.py)
- `test_dfm_pipeline_with_columnwise_transformer` (test_pipeline.py)
- `test_dfm_pipeline_config_loading` (test_pipeline_dfm.py)
- `test_dfm_pipeline_training` (test_pipeline_dfm.py)
- `test_dfm_pipeline_prediction` (test_pipeline_dfm.py)
- `test_dfm_pipeline_complete` (test_pipeline_dfm.py)
- `test_dfm_pipeline_with_columnwise_transformer` (test_pipeline_dfm.py)
- `test_ddfm_pipeline_config_loading` (test_pipeline.py)
- `test_ddfm_pipeline_config_loading` (test_pipeline_dfm.py)

**Fix**: Remove line 508 `self.config = config` from `DFM.__init__()` since `self._config = config` is already set on line 509. The property getter will return `self._config`.

**Files to Modify**:
- `dfm-python/src/dfm_python/models/dfm.py` (line 508)

**Impact**: High - This is blocking 14+ test failures. Simple fix, high impact.

---

### Category 2: Function Signature Mismatches (2 failures)

#### Issue 2.1: log_convergence() Signature Mismatch

**Root Cause**: Test calls `log_convergence(logger=logger, converged=True, num_iter=50)` but the function signature is `log_convergence(converged, num_iter, final_loglik=None, reason=None)` - no `logger` parameter.

**Affected Test**: `test_log_convergence` in `test_logger.py`

**Fix Options**:
1. **Option A (Recommended)**: Update test to match function signature (remove `logger=` parameter)
2. **Option B**: Add `logger` parameter to function (but function doesn't use logger, it uses module-level `_logger`)

**Recommended Fix**: Option A - Update test to match actual function signature.

**Files to Modify**:
- `dfm-python/src/test/test_logger.py` (line 108)

**Impact**: Medium - Single test failure, easy fix.

---

### Category 3: Type Errors (2 failures: 1 error + 1 failure)

#### Issue 3.1: np.isnan() Type Error

**Root Cause**: `rem_nans_spline()` in `data.py` line 113 calls `np.isnan(X)` but `X` may not be a numeric type (could be object/string array).

**Affected Tests**:
- `test_nowcast_data_view` (ERROR)
- `test_dfm_with_real_data` (FAILED)

**Error Message**: `TypeError: ufunc 'isnan' not supported for the input types`

**Fix**: Add type checking/conversion before calling `np.isnan()`:
```python
# Ensure X is numeric
if not np.issubdtype(X.dtype, np.number):
    X = pd.DataFrame(X).select_dtypes(include=[np.number]).values
    # Or convert to numeric, handling errors
X = np.asarray(X, dtype=np.float64)
indNaN = np.isnan(X)
```

**Files to Modify**:
- `dfm-python/src/dfm_python/utils/data.py` (line 113)

**Impact**: High - Blocks 2 tests, could cause runtime errors in production.

---

### Category 4: Test Expectations (2 failures)

#### Issue 4.1: Config Block Derivation Test

**Root Cause**: Test `test_dfm_config_block_derivation` creates series S3 with `blocks=[DEFAULT_BLOCK_NAME]` but validation requires all series to load on global block (first block).

**Error**: `ValueError: Series 2 ('S3') must load on the global block (first block 'Block_Global'). All series must have blocks[0] = 1. Current value: 0`

**Fix**: Update test to set S3's blocks to include global block:
```python
SeriesConfig(
    series_id="S3",
    frequency="m",
    transformation="chg",
    blocks=[DEFAULT_BLOCK_NAME, "Block_Global"]  # Add global block
)
```

**Files to Modify**:
- `dfm-python/src/test/test_config.py` (line 144)

**Impact**: Low - Single test, easy fix.

#### Issue 4.2: Transformation Validation Test

**Root Cause**: Test expects `ValueError` to be raised, but validation only warns (doesn't raise).

**Fix**: Update test expectation - validation warns but doesn't raise errors for invalid transformations.

**Files to Modify**:
- `dfm-python/src/test/test_config.py` (line 306)

**Impact**: Low - Single test, easy fix.

---

### Category 5: Missing Attributes (2 failures)

#### Issue 5.1: DFMLinear Missing `result` Attribute

**Root Cause**: Test `test_dfm_linear_initialization` checks `model.result is None` but `DFMLinear` doesn't have a `result` property (only high-level `DFM` has it).

**Fix**: Remove `result` check from test, or add `result` property to `DFMLinear` (but it's a low-level class, so probably shouldn't have it).

**Recommended Fix**: Update test to not check `result` attribute for low-level `DFMLinear` class.

**Files to Modify**:
- `dfm-python/src/test/test_models.py` (line 74)

**Impact**: Low - Single test, easy fix.

---

### Category 6: DataModule Setup Issues (2 failures)

#### Issue 6.1: DataModule Mx is None

**Root Cause**: Test `test_dfm_pipeline_data_loading` expects `data_module.Mx is not None` after `setup()`, but `Mx` is None.

**Possible Causes**:
1. `setup()` not fully completing
2. Missing data causing `Mx` to not be initialized
3. Test data issues (93.6% missing data warning suggests data quality issues)

**Fix**: 
1. Check if `setup()` properly initializes `Mx`
2. Ensure test data is valid
3. Add proper error handling in test

**Files to Investigate**:
- `dfm-python/src/dfm_python/lightning/data_module.py` (setup method)
- `dfm-python/src/test/test_pipeline.py` (line 167)
- `dfm-python/src/test/test_pipeline_dfm.py` (line 167)

**Impact**: Medium - 2 test failures, requires investigation.

---

### Category 7: Trainer Configuration (5 failures)

#### Issue 7.1: DFMTrainer Default Values

**Root Cause**: Test expects specific default values that don't match actual defaults.

**Fix**: Review test expectations vs actual trainer defaults, update test or trainer defaults.

**Files to Investigate**:
- `dfm-python/src/test/test_trainer.py` (line ~50)
- `dfm-python/src/dfm_python/trainer/dfm.py`

**Impact**: Medium - 1 test failure.

#### Issue 7.2: DDFMTrainer Default Values

**Root Cause**: Similar to DFMTrainer, test expectations don't match defaults.

**Fix**: Review and align test expectations with actual defaults.

**Files to Investigate**:
- `dfm-python/src/test/test_trainer.py` (line ~74)
- `dfm-python/src/dfm_python/trainer/ddfm.py`

**Impact**: Medium - 1 test failure.

#### Issue 7.3: DDFMTrainer Config Loading

**Root Cause**: `test_ddfm_trainer_from_config` fails when loading config.

**Fix**: Investigate config loading logic in DDFMTrainer.

**Files to Investigate**:
- `dfm-python/src/test/test_trainer.py` (line ~80)
- `dfm-python/src/dfm_python/trainer/ddfm.py`

**Impact**: Medium - 1 test failure.

#### Issue 7.4: Trainer Device Handling

**Root Cause**: Test `test_trainer_device_handling` fails.

**Fix**: Review device handling logic in trainers.

**Files to Investigate**:
- `dfm-python/src/test/test_trainer.py` (line ~120)
- `dfm-python/src/dfm_python/trainer/dfm.py`
- `dfm-python/src/dfm_python/trainer/ddfm.py`

**Impact**: Medium - 1 test failure.

#### Issue 7.5: Trainer Precision

**Root Cause**: Test `test_trainer_precision` fails.

**Fix**: Review precision handling logic in trainers.

**Files to Investigate**:
- `dfm-python/src/test/test_trainer.py` (line ~130)
- `dfm-python/src/dfm_python/trainer/dfm.py`
- `dfm-python/src/dfm_python/trainer/ddfm.py`

**Impact**: Medium - 1 test failure.

---

### Category 8: Empty Test File (1 file)

#### Issue 8.1: test_pipeline_ddfm.py is Empty

**Root Cause**: File exists but contains no test code (0 lines).

**Impact**: No automated tests for DDFM pipeline workflow - **CRITICAL FOR DDFM STABILITY**

**Fix**: Implement DDFM pipeline tests similar to DFM pipeline tests in `test_pipeline_dfm.py`.

**Files to Modify**:
- `dfm-python/src/test/test_pipeline_ddfm.py` (currently empty)

**Approach**: 
1. Copy structure from `test_pipeline_dfm.py`
2. Adapt for DDFM-specific requirements
3. Use DDFM model instead of DFM
4. Adjust training parameters for DDFM (neural network training)

**Impact**: High - Critical for DDFM stability verification.

---

## Refactoring and Improvement Opportunities

### 1. Test Code Consolidation (High Priority)

**Issue**: `test_pipeline.py` and `test_pipeline_dfm.py` are nearly identical (~750 lines each, ~1500 lines total duplicate code).

**Impact**: 
- Maintenance burden (fixes must be applied twice)
- Potential inconsistencies
- Confusion about which file to use

**Constraint**: Cannot create `conftest.py` in dfm-python/ (file creation restriction).

**Solutions**:
1. **Option A (Recommended)**: Consolidate into single file `test_pipeline_dfm.py`, remove `test_pipeline.py`
2. **Option B**: Keep both but clearly document which is canonical
3. **Option C**: Move shared fixtures to one file, import in the other (but both files are identical, so this doesn't help)

**Recommended**: Option A - Remove `test_pipeline.py`, keep `test_pipeline_dfm.py` as the canonical DFM pipeline test file.

**Files to Modify**:
- Delete: `dfm-python/src/test/test_pipeline.py`
- Keep: `dfm-python/src/test/test_pipeline_dfm.py`

**Impact**: High - Reduces maintenance burden significantly.

---

### 2. Config Property Pattern Improvement

**Issue**: Config property is read-only but `__init__` tries to set it directly.

**Current Pattern**:
```python
self.config = config  # Fails - no setter
self._config = config  # Works
```

**Improvement**: 
- Remove direct `self.config = config` assignment
- Always use `self._config = config` in `__init__`
- Property getter returns `self._config`

**Files to Modify**:
- `dfm-python/src/dfm_python/models/dfm.py` (line 508)
- `dfm-python/src/dfm_python/models/ddfm.py` (check if same issue exists)

**Impact**: Medium - Prevents future bugs, improves code clarity.

---

### 3. Error Handling in Data Utilities

**Issue**: `rem_nans_spline()` doesn't handle non-numeric input types gracefully.

**Improvement**: 
- Add type checking/conversion at function entry
- Provide clear error messages for invalid input types
- Handle edge cases (empty arrays, all NaN, etc.)

**Files to Modify**:
- `dfm-python/src/dfm_python/utils/data.py` (line 113)

**Impact**: Medium - Improves robustness, prevents runtime errors.

---

### 4. Naming Consistency

**Issue**: Some inconsistencies in naming:
- `DFMLinear` vs `DFM` (low-level vs high-level)
- `DDFMModel` vs `DDFM` (low-level vs high-level)
- Property names: `config` vs `_config`

**Improvement**: 
- Document naming conventions clearly
- Ensure consistent patterns across codebase
- Update tests to use correct class names

**Impact**: Low - Improves code clarity and maintainability.

---

### 5. Test Organization

**Issue**: 
- Empty test file (`test_pipeline_ddfm.py`)
- Duplicate test files (`test_pipeline.py` and `test_pipeline_dfm.py`)
- Many skipped tests (16 skipped) due to missing data/config files

**Improvement**:
1. Implement empty test file
2. Consolidate duplicate tests
3. Document required test data/config files
4. Consider creating minimal test fixtures

**Impact**: High - Improves test coverage and maintainability.

---

## Implementation Plan

### Phase 1: Critical Fixes (High Priority, High Impact)

**Goal**: Fix issues blocking the most tests (14+ failures from config property issue).

1. **Fix Config Property Setter** (15 minutes)
   - Remove `self.config = config` from `DFM.__init__()` (line 508)
   - Verify `DDFM.__init__()` doesn't have same issue
   - Run tests to verify fix

2. **Fix np.isnan() Type Error** (30 minutes)
   - Add type checking/conversion in `rem_nans_spline()`
   - Test with various input types
   - Run tests to verify fix

**Expected Result**: ~16 failures fixed (14 config + 2 type errors)

---

### Phase 2: Function Signature and Test Expectations (Medium Priority)

**Goal**: Fix function signature mismatches and test expectation issues.

1. **Fix log_convergence() Test** (10 minutes)
   - Update test to match function signature
   - Run test to verify

2. **Fix Config Test Expectations** (20 minutes)
   - Fix block derivation test (add global block to S3)
   - Fix transformation validation test (update expectation)
   - Run tests to verify

3. **Fix DFMLinear result Attribute Test** (10 minutes)
   - Remove `result` check from test (low-level class doesn't have it)
   - Run test to verify

**Expected Result**: 4 failures fixed

---

### Phase 3: DataModule and Trainer Issues (Medium Priority)

**Goal**: Fix DataModule setup and trainer configuration issues.

1. **Investigate DataModule Mx Issue** (1 hour)
   - Review `setup()` method in `DFMDataModule`
   - Check test data quality
   - Fix initialization logic or test expectations

2. **Fix Trainer Default Values** (1 hour)
   - Review trainer default values
   - Align test expectations with actual defaults
   - Fix config loading if needed

3. **Fix Trainer Device/Precision Handling** (1 hour)
   - Review device handling logic
   - Review precision handling logic
   - Fix issues or update test expectations

**Expected Result**: 7 failures fixed

---

### Phase 4: Test File Implementation and Consolidation (High Priority)

**Goal**: Implement empty test file and consolidate duplicate tests.

1. **Implement test_pipeline_ddfm.py** (2 hours)
   - Copy structure from `test_pipeline_dfm.py`
   - Adapt for DDFM (use DDFM model, adjust training parameters)
   - Test implementation

2. **Consolidate Duplicate Tests** (30 minutes)
   - Remove `test_pipeline.py` (duplicate of `test_pipeline_dfm.py`)
   - Verify `test_pipeline_dfm.py` has all necessary tests
   - Update any references if needed

**Expected Result**: Test coverage improved, maintenance burden reduced

---

## Estimated Timeline

- **Phase 1**: 45 minutes (Critical fixes)
- **Phase 2**: 40 minutes (Function signatures and test expectations)
- **Phase 3**: 3 hours (DataModule and trainer issues)
- **Phase 4**: 2.5 hours (Test file implementation and consolidation)

**Total**: ~6.5 hours

---

## Success Criteria

1. ✅ All 27 failures fixed
2. ✅ 1 error fixed
3. ✅ Empty test file implemented
4. ✅ Duplicate test code consolidated
5. ✅ All tests pass (or skip gracefully with clear reasons)
6. ✅ Code quality improvements applied (refactoring, error handling, naming)

---

## Notes

- **File Creation Restriction**: Cannot create new files in dfm-python/ (submodule restriction). However, `test_pipeline_ddfm.py` already exists (empty), so we can edit it.
- **Test Data**: Some tests skip due to missing data/config files. This is acceptable if tests skip gracefully with clear messages.
- **Backward Compatibility**: All fixes should maintain backward compatibility with existing code.

---

## Next Steps

1. Start with Phase 1 (Critical Fixes) - highest impact, quickest wins
2. Verify fixes with pytest after each phase
3. Document any issues found during investigation
4. Update STATUS.md with progress
