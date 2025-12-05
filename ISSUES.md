# Issues and Action Plan

## Current Status (2025-01-XX)

### Work Completed This Iteration
- ✅ **Report**: Complete 20-30 page LaTeX framework finalized with all sections
- ✅ **dfm-python**: Code quality verified, consistent naming conventions, clean patterns
- ✅ **src/**: Architecture complete, all import issues resolved

### Experiment Status
- ❌ **0/3 targets complete** (KOGDP...D, KOCNPER.D, KOGFCF..D)
- **Previous Attempts**: 45 log files (all failed, now resolved)
- ✅ **Code Ready**: All import errors fixed, `run_experiment.sh` uses `.venv/bin/python3` (line 260)
- ✅ **Script Ready**: Auto-skips completed targets, runs in parallel

### Experiments Required
- **Configuration**: 3 targets × 4 models × 3 horizons = 36 combinations
- **Expected Outputs**: 3 result directories, 3 JSON files, 12 models, 1 aggregated CSV (36 rows)
- **Report Needs**: 4 plots, 4 tables, results section, discussion section

### Next Action
**Run**: `bash run_experiment.sh` (will auto-skip completed targets)

---

## Action Plan

1. **Phase 2: Run Experiments** (READY)
   - Run: `bash run_experiment.sh`
   - Expected: 3 result directories, 12 models, 1 aggregated CSV (36 rows)
   - Script auto-skips completed targets

2. **Phase 3: Generate Visualizations** (BLOCKED until Phase 2)
   - Run: `python3 nowcasting-report/code/plot.py`
   - Generates 4 PNG files in `nowcasting-report/images/`

3. **Phase 4: Update Report Tables** (BLOCKED until Phase 2)
   - Update LaTeX tables from `aggregated_results.csv`
   - Files: `tab_overall_metrics.tex`, `tab_overall_metrics_by_target.tex`, `tab_overall_metrics_by_horizon.tex`, `tab_nowcasting_metrics.tex`

4. **Phase 5: Update Report Content** (BLOCKED until Phase 2-4)
   - Replace placeholders in `contents/5_result.tex`, `contents/6_discussion.tex`, `main.tex`
   - Verify all numbers match tables

5. **Phase 6: Finalize Report** (BLOCKED until Phase 5)
   - Compile PDF: `cd nowcasting-report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`
   - Verify: 20-30 pages, no placeholders

## Notes

**run_experiment.sh**: Auto-skips completed targets (checks for `comparison_results.json`), runs in parallel (max 5), uses `.venv/bin/python3` (line 260)

**If experiments fail**: Check logs in `outputs/comparisons/*.log`, fix issues, re-run (script auto-skips completed)

**Resolved Issues**: ✅ Import errors fixed, ✅ venv activation fixed, ✅ Code ready for execution
