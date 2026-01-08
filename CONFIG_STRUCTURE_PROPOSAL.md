# Hydra Config Structure Proposal

Based on Hydra documentation and best practices, here are three options for organizing the configuration structure:

## Current Structure Analysis

**Current Structure:**
```
config/
├── default.yaml
└── model/
    ├── dfm/
    │   ├── default.yaml      # Base DFM config
    │   ├── investment.yaml   # Extends default, adds target_series
    │   ├── consumption.yaml  # Extends default, adds target_series
    │   └── production.yaml   # Extends default, adds target_series
    └── ddfm/
        └── default.yaml
```

**Current Issues:**
1. `model=dfm` in defaults looks for `model/dfm.yaml`, not `model/dfm/default.yaml`
2. Manual YAML loading in `main.py` is a workaround, not idiomatic Hydra
3. Mixed approach: some models need data-specific configs, others don't

---

## Option 1: Nested Path Syntax (Recommended)

**Structure:**
```
config/
├── default.yaml
└── model/
    ├── dfm/
    │   ├── default.yaml
    │   ├── investment.yaml    # Uses defaults: [default]
    │   ├── consumption.yaml
    │   └── production.yaml
    └── ddfm/
        └── default.yaml
```

**Usage:**
```yaml
# config/default.yaml
defaults:
  - _self_
  - model: dfm/investment  # Nested path syntax
  # Or override via CLI: model=dfm/consumption

data: investment
train: true
```

**Command line:**
```bash
python -m src.main model=dfm/investment data=investment
python -m src.main model=dfm/consumption data=consumption
python -m src.main model=ddfm data=investment  # Uses model/ddfm/default.yaml
```

**Pros:**
- ✅ Idiomatic Hydra (nested paths)
- ✅ No manual config loading needed
- ✅ Clear and explicit
- ✅ Works with Hydra's built-in resolution

**Cons:**
- ⚠️ Requires updating `default.yaml` syntax
- ⚠️ Need to remove manual config loading from `main.py`

---

## Option 2: Separate Data Config Group

**Structure:**
```
config/
├── default.yaml
├── model/
│   ├── dfm.yaml              # Base DFM config (no default.yaml)
│   ├── ddfm.yaml
│   └── ...
└── dfm_data/                 # Separate config group for DFM data variants
    ├── investment.yaml       # Only target_series override
    ├── consumption.yaml
    └── production.yaml
```

**Usage:**
```yaml
# config/default.yaml
defaults:
  - _self_
  - model: dfm
  - dfm_data: null            # Optional, can override

data: investment
train: true
```

**Command line:**
```bash
python -m src.main model=dfm dfm_data=investment data=investment
python -m src.main model=ddfm data=investment  # No dfm_data needed
```

**Pros:**
- ✅ Clean separation of concerns
- ✅ Models without data variants don't need nested dirs
- ✅ Flexible: can use `dfm_data=null` to skip

**Cons:**
- ⚠️ More complex defaults list
- ⚠️ Need conditional logic in code to merge `dfm_data` config

---

## Option 3: Flat with Conditional Merging (Current + Improvements)

**Structure:**
```
config/
├── default.yaml
└── model/
    ├── dfm.yaml              # Base DFM config
    ├── dfm_investment.yaml   # Complete config (inherits via code)
    ├── dfm_consumption.yaml
    └── ddfm.yaml
```

**Usage:**
```yaml
# config/default.yaml
defaults:
  - _self_
  - model: dfm                # Can override: model=dfm_investment

data: investment
train: true
```

**Pros:**
- ✅ Simple, flat structure
- ✅ Easy to understand
- ✅ No nested directories

**Cons:**
- ⚠️ Code duplication (each variant needs full config)
- ⚠️ Less maintainable
- ⚠️ Not using Hydra's composition features

---

## Recommended: Option 1 (Nested Path Syntax)

**Implementation Steps:**

1. **Update `config/default.yaml`:**
   ```yaml
   defaults:
     - _self_
     - model: dfm/investment  # Use nested path

   data: investment
   train: true
   forecast: true
   ```

2. **Update `config/model/dfm/investment.yaml` (already correct):**
   ```yaml
   defaults:
     - default

   target_series:
     - KOEQUIPTE
   ```

3. **Remove manual config loading from `main.py`:**
   - Remove the YAML loading workaround
   - Let Hydra handle all config resolution

4. **For models without variants (ddfm, itf, etc.):**
   - Keep `model/ddfm/default.yaml` 
   - Use `model=ddfm` (Hydra automatically uses `default.yaml`)

**Benefits:**
- Fully idiomatic Hydra usage
- Cleaner codebase
- Better type checking and validation
- Easier to extend with more variants

---

## Alternative: Hybrid Approach

If you want backward compatibility with `model=dfm` syntax while supporting nested paths:

**Structure:**
```
config/
├── default.yaml
└── model/
    ├── dfm.yaml              # Points to dfm/default.yaml via package
    └── dfm/
        ├── default.yaml
        ├── investment.yaml
        └── ...
```

**`model/dfm.yaml`:**
```yaml
# @package _global_
defaults:
  - dfm/default

# Re-export everything from dfm/default.yaml
# This allows model=dfm to work
```

This gives both `model=dfm` and `model=dfm/investment` syntax, but is more complex.
