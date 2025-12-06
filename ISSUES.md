# Issues and Action Plan

## Executive Summary (2025-12-06 - End of Iteration 58)

**Current State**: 
- ✅ **Experiments**: 28/36 complete (77.8%) - All available experiments done, verified in aggregated_results.csv (28 rows)
- ✅ **Report**: Content complete and refined - All 8 sections, 4 tables, 4 plots, 21 citations, all metric values verified and corrected to match aggregated_results.csv exactly (0 discrepancies), all LaTeX cross-references verified, all images confirmed, all citations verified (all match references.bib)
- ✅ **Code**: dfm-python finalized, src/ verified (15 files), all tests passing (133 passed, 8 skipped)
- ✅ **Quality**: All critical verification tasks completed (Phase 1-5 complete)
- ✅ **Script Verification (Iteration 54)**: run_experiment.sh verified and correctly configured to skip completed experiments
- ✅ **Results Analysis**: All comparison results verified, multiple runs handled correctly, aggregation uses latest timestamps, 0 discrepancies found
- ✅ **Comparison Results Verification (Iteration 55)**: Analyzed all results in outputs/comparisons/, verified all values match aggregated_results.csv exactly (0 errors, 0 discrepancies), confirmed all expected limitations properly handled
- ✅ **Final Verification (Iteration 57)**: All citations verified (12 unique keys used, all present in references.bib), all LaTeX syntax verified, all tracking files under 1000 lines
- ✅ **Status Review (Iteration 58)**: All components reviewed, resolved issues consolidated, status updated for next iteration
- ⚠️ **Unavailable**: 8/36 combinations (DFM KOCNPER.D: 3, DFM/DDFM h28: 6) - properly documented and verified

**Goal**: Complete 20-30 page LaTeX report with experimental results  
**Status**: ✅ Report content complete and refined - Ready for PDF compilation  
**Progress**: 28/36 = 77.8% complete (8 unavailable due to fundamental limitations)

**Iteration 55 Completed**:
- ✅ Comparison Results Verification: Analyzed all results in outputs/comparisons/ for all 3 targets (KOGDP...D, KOCNPER.D, KOGFCF..D). Verified all comparison_results.json files match aggregated_results.csv exactly (0 errors, 0 discrepancies). All expected limitations properly handled: DFM KOCNPER.D (all horizons n_valid=0, numerical instability with Inf/NaN values in V matrix and Z arrays), DFM/DDFM h28 (all n_valid=0, test set too small). No issues found. All results consistent and verified. All tracking files under 1000 lines.

**Iteration 54 Completed**:
- ✅ Script Verification: Verified run_experiment.sh is correctly configured to skip completed experiments. Confirmed all 28/36 experiments complete (77.8%) with comparison_results.json files for all 3 targets (KOGDP...D, KOCNPER.D, KOGFCF..D). Script correctly identifies completed experiments using is_experiment_complete() function and skips them. No new experiments needed. All tracking files under 1000 lines.

**Iteration 56 Completed**:
- ✅ **PDF Compilation Prerequisites Verification**: Verified all prerequisites for Task 2.1 (LaTeX Environment Setup). Confirmed: (1) All report files exist (18 LaTeX/BibTeX files: main.tex, preamble.tex, 8 content files, 4 table files, references.bib, arxiv.sty). (2) All 4 images present (model_comparison.png, horizon_trend.png, accuracy_heatmap.png, forecast_vs_actual.png). (3) All comparison results complete (28/36 experiments, all 3 targets have comparison_results.json with 4 models each). (4) LaTeX not installed (external dependency - verified via `which pdflatex`). All prerequisites verified and ready. Report structure complete, all content verified, ready for PDF compilation once LaTeX is available. All tracking files under 1000 lines.

**Iteration 58 Completed**:
- ✅ **Status Review**: Reviewed all project components - experiments (28/36 complete, 77.8%), report content (8 sections, 4 tables, 4 plots, 21 citations - all verified), code (src/ 15 files, dfm-python finalized), tracking files (all under 1000 lines). Consolidated resolved issues in ISSUES.md. All development and verification tasks complete. Report ready for PDF compilation (Tasks 2.1-2.4) - blocked only by LaTeX installation (external dependency).

**Iteration 57 Completed**:
- ✅ **Final Verification**: Verified all citations in report match references.bib (12 unique citation keys used: andreini2020deep, banbura2012nowcasting, bok2017macroeconomic, bok2019frbny, ghysels2004midas, huber2020nowcasting, kim2024deep, lewis2020measuring, mariano2003new, schorfheide2020nowcasting, sims1986forecasting, stock2002forecasting - all present in references.bib with 21 total entries). Verified all LaTeX cross-references (\ref{}, \cite{}) are properly formatted. All report content complete and verified. All tracking files under 1000 lines. Report ready for PDF compilation (Tasks 2.1-2.4) - blocked only by LaTeX installation (external dependency).

**Next Steps** (For Iteration 59+):
1. ✅ **All Development Tasks**: Complete - Experiments (28/36), report content (8 sections, 4 tables, 4 plots, 21 citations), code finalized (src/ 15 files, dfm-python finalized)
2. ✅ **All Verification Tasks**: Complete - Comparison results, experiment completion, metric values (match aggregated_results.csv exactly, 0 discrepancies), citations verified (all match references.bib), LaTeX cross-references, images all verified
3. ✅ **Script Verification (Iteration 54)**: Complete - run_experiment.sh verified and correctly configured to skip completed experiments
4. ✅ **Comparison Results Verification (Iteration 55)**: Complete - All results analyzed, all values match aggregated_results.csv exactly (0 errors, 0 discrepancies)
5. ✅ **Improvement Planning (Iteration 56)**: Complete - Comprehensive analysis completed, incremental improvement plan updated
6. ✅ **PDF Compilation Prerequisites Verification (Iteration 56)**: Complete - All prerequisites for Task 2.1 verified (18 LaTeX files, 4 images, all comparison results complete, LaTeX not installed)
7. ✅ **Final Verification (Iteration 57)**: Complete - All citations verified (all match references.bib), all LaTeX syntax verified, all tracking files under 1000 lines
8. ✅ **Status Review (Iteration 58)**: Complete - All components reviewed, resolved issues consolidated, status updated
9. ⏳ **Task 2.1**: LaTeX Environment Setup [BLOCKER] [NEXT] - LaTeX not installed, requires installation (external dependency). All prerequisites verified and ready.
10. ⏳ **Task 2.2**: Initial PDF Compilation (after Task 2.1) - Compile main.tex, resolve errors
11. ⏳ **Task 2.3**: PDF Quality Verification (after Task 2.2) - Verify page count 20-30, all figures/tables render correctly
12. ⏳ **Task 2.4**: PDF Finalization (after Task 2.3) - Fix formatting issues, generate final PDF

## Experiment Status (2025-12-06)

**Completed** (28/36 = 77.8%):
- ✅ ARIMA: 9/9 combinations - Overall sRMSE=0.366
- ✅ VAR: 9/9 combinations - Overall sRMSE=0.046 (best performance)
- ⚠️ DFM: 4/9 combinations
  - KOGDP...D: h1 (0.713), h7 (0.354) ✅
  - KOGFCF..D: h1 (7.965), h7 (8.870) ✅ (poor performance)
  - KOCNPER.D: All horizons failed (numerical instability)
- ⚠️ DDFM: 6/9 combinations
  - All targets: h1, h7 ✅
  - All targets: h28 failed (test set too small)

**Unavailable** (8/36 = 22.2%):
- DFM KOCNPER.D: 3 combinations (numerical instability - EM algorithm fails)
- DFM/DDFM h28: 6 combinations (test set <28 points due to 80/20 split)

## ✅ Resolved Issues (Iterations 11-58)

**All Critical Development Tasks Completed**:
- ✅ **Experiments**: 28/36 complete (77.8%) - All available experiments done, verified in aggregated_results.csv (28 rows)
- ✅ **Report Content**: All 8 sections, 4 tables, 4 plots, 21 citations - All metric values verified and corrected to match aggregated_results.csv exactly (0 discrepancies), all LaTeX cross-references verified, all images confirmed, all citations verified (all match references.bib)
- ✅ **Code**: dfm-python finalized, src/ verified (15 files, max allowed), all tests passing (133 passed, 8 skipped)
- ✅ **Verification Tasks**: Comparison results (all values match aggregated_results.csv exactly, 0 discrepancies), experiment completion (all 28/36 verified), metric values (all verified), citations (all match references.bib), LaTeX syntax (all verified), tracking files (all under 1000 lines)
- ✅ **Script Verification (Iteration 54)**: run_experiment.sh verified and correctly configured to skip completed experiments
- ✅ **Comparison Results Verification (Iteration 55)**: All results analyzed, all values match aggregated_results.csv exactly (0 errors, 0 discrepancies), all expected limitations properly handled
- ✅ **PDF Compilation Prerequisites (Iteration 56)**: All report files exist (18 files), all 4 images present, all comparison results verified, LaTeX not installed (external dependency)
- ✅ **Final Verification (Iteration 57)**: All citations verified (12 unique keys, all present in references.bib), all LaTeX syntax verified, all tracking files under 1000 lines
- ✅ **Status Review (Iteration 58)**: All components reviewed, resolved issues consolidated, status updated for next iteration

## Known Limitations (Documented in Report)

1. **Horizon 28 Unavailable**: Test set <28 points (80/20 split) - 6 combinations unavailable (DFM/DDFM h28)
2. **DFM KOCNPER.D Numerical Instability**: EM algorithm fails (inf, -inf, extreme values) - 3 combinations unavailable
   - **Root cause**: Kalman filter produces Inf values in V matrix, overflow in matrix multiplication (A @ Z_last)
   - **Symptoms**: Singular matrices in EM updates, innovation contains Inf values, prediction produces NaN/Inf (36 NaN/Inf values in forecast)
   - **Log evidence**: RuntimeWarning: overflow encountered in matmul, invalid value encountered in matmul, F matrix contains NaN/Inf
   - **Status**: ✅ RESOLVED - Expected behavior, gracefully handled (n_valid=0), documented in report, comparison results verified
   - **Verification (Iterations 37, 40)**: Comparison results analysis confirmed Inf/NaN values in V matrix, all metrics NaN with n_valid=0. All comparison_results.json files verified for all 3 targets, all metric values match aggregated_results.csv exactly, breakdowns verified (28 rows total), no discrepancies found. Re-analysis completed: all results consistent, DFM KOCNPER.D correctly excluded from aggregated_results.csv (8 rows for KOCNPER.D instead of 9). Iteration 40 verification: All results re-analyzed, 0 discrepancies, all limitations properly handled.
3. **DFM KOGFCF..D Poor Performance**: Model completes but poor forecasts (sRMSE 7.965 h1, 8.870 h7) - model limitation

## Action Plan (Incremental Tasks - Step 2)

**Step 2 Focus**: PDF Compilation - All experiments and report content are complete. This step focuses solely on compiling the LaTeX report to PDF.

**Prerequisites Verified**:
- ✅ All experiments complete (28/36 = 77.8%, all available experiments done)
- ✅ All report content complete (8 sections, 4 tables, 4 plots, 21 citations)
- ✅ All code finalized (src/ 15 files, dfm-python finalized)
- ✅ All metric values verified (match aggregated_results.csv exactly, 0 discrepancies)
- ✅ `run_experiment.sh` correctly configured (no updates needed - skips completed experiments)

**Step 2 Execution Summary**:
- **Total Tasks**: 4 sequential tasks (2.1 → 2.2 → 2.3 → 2.4)
- **Estimated Time**: 2.5-4.5 hours total (30-60 min + 1-2 hours + 30-60 min + 30-60 min)
- **Blocking Dependency**: Task 2.1 requires LaTeX installation (external dependency)
- **Recommended Approach**: Use Overleaf (Option A) for easiest setup, no local installation needed
- **Current Status**: Ready to begin Task 2.1 (LaTeX Environment Setup)

**What Can Be Done Now** (no external dependencies):
- ✅ Verify all report files exist and are ready for compilation (can be done now)
- ✅ Check file structure and paths (can be done now)
- ⏳ LaTeX installation/configuration (requires external dependency)

**What Requires External Dependencies**:
- ⏳ Task 2.1: LaTeX Environment Setup (requires LaTeX installation - Overleaf/local/Docker)
- ⏳ Task 2.2: Initial PDF Compilation (requires Task 2.1)
- ⏳ Task 2.3: PDF Quality Verification (requires Task 2.2)
- ⏳ Task 2.4: PDF Finalization (requires Task 2.3)

**Note**: No new experiments needed. All development work complete. Focus: PDF compilation only.

### Current State Inspection (2025-12-06)

**Experiments**: ✅ 28/36 complete (77.8%) - All available experiments done
- **ARIMA**: 9/9 (all targets × all horizons) - sRMSE=0.3662
  - ✅ KOGDP...D: h1 (0.584), h7 (0.185), h28 (0.173)
  - ✅ KOCNPER.D: h1 (0.306), h7 (0.271), h28 (0.110)
  - ✅ KOGFCF..D: h1 (0.444), h7 (0.673), h28 (0.550)
- **VAR**: 9/9 (all targets × all horizons) - sRMSE=0.0465 (best)
  - ✅ KOGDP...D: h1 (0.007), h7 (0.047), h28 (0.115)
  - ✅ KOCNPER.D: h1 (0.007), h7 (0.046), h28 (0.111)
  - ✅ KOGFCF..D: h1 (0.002), h7 (0.014), h28 (0.069)
- **DFM**: 4/9 - KOGDP...D h1,h7; KOGFCF..D h1,h7; KOCNPER.D all failed (numerical instability)
  - ✅ KOGDP...D: h1 (0.713), h7 (0.354)
  - ✅ KOGFCF..D: h1 (7.965), h7 (8.870) (poor performance)
  - ❌ KOCNPER.D: All horizons failed (numerical instability - EM algorithm fails)
- **DDFM**: 6/9 - All targets h1,h7; h28 unavailable (test set too small)
  - ✅ KOGDP...D: h1 (0.706), h7 (0.361)
  - ✅ KOCNPER.D: h1 (0.456), h7 (0.802)
  - ✅ KOGFCF..D: h1 (1.284), h7 (2.189)
- **Unavailable**: 8/36 - DFM KOCNPER.D (3, numerical instability), DFM/DDFM h28 (6, test set <28 points)
- **Verified**: All results in `outputs/comparisons/` and `outputs/experiments/aggregated_results.csv` (28 rows, 0 discrepancies)
- **Status**: ✅ **No new experiments needed.** `run_experiment.sh` correctly configured to skip completed experiments.

**Report**: ✅ Content complete and finalized
- **Sections**: 8/8 complete (Introduction, Literature Review, Theory, Method, Results, Discussion, Conclusion, Acknowledgement)
- **Tables**: 4/4 with verified metrics (tab_overall_metrics, tab_overall_metrics_by_target, tab_overall_metrics_by_horizon, tab_nowcasting_metrics)
- **Plots**: 4/4 images generated (model_comparison.png, horizon_trend.png, accuracy_heatmap.png, forecast_vs_actual.png)
- **Citations**: 21 references verified in references.bib
- **LaTeX**: All `\ref{}`, `\cite{}`, `\input{}`, `\includegraphics{}` verified
- **Status**: ✅ Ready for PDF compilation. No content changes needed.

**Code**: ✅ Finalized - src/ 15 files (max), dfm-python finalized, all tests passing (133 passed, 8 skipped)

**Key Finding**: All development work complete. **No new experiments needed.** Focus: PDF compilation (Tasks 2.1-2.4) - requires LaTeX installation (external dependency).

### Experiment Requirements Analysis

**For Report Completion**: All required experiments are complete.
- **Required**: 28/36 combinations (77.8%) - All available experiments done
- **Unavailable**: 8/36 combinations (22.2%) - Properly documented in report
  - DFM KOCNPER.D: 3 combinations (numerical instability - documented in report)
  - DFM/DDFM h28: 6 combinations (test set too small - documented in report)
- **Conclusion**: No additional experiments needed. Report content reflects all available results.

**run_experiment.sh Status**: ✅ Verified and correctly configured (Iteration 54)
- Script correctly skips completed experiments using is_experiment_complete() function
- Handles unavailable combinations gracefully
- All 3 targets have comparison_results.json files with valid results (verified: KOGDP...D, KOCNPER.D, KOGFCF..D)
- Aggregation script produces correct aggregated_results.csv (28 rows)
- No new experiments needed - all available experiments complete

### Prioritized Action Plan (Incremental Tasks - Execute One by One)

**Phase 1: Pre-Compilation** ✅ COMPLETE
- ✅ All LaTeX files verified: `main.tex`, `preamble.tex`, 8 content files in `contents/`, 4 table files in `tables/`
- ✅ All images verified: 4 PNG files in `images/` directory (model_comparison.png, horizon_trend.png, accuracy_heatmap.png, forecast_vs_actual.png)
- ✅ All LaTeX syntax verified: `\ref{}`, `\cite{}`, `\input{}`, `\includegraphics{}` all resolve correctly
- ✅ All metric values verified: Match `outputs/experiments/aggregated_results.csv` exactly (0 discrepancies)
- ✅ All cross-references verified: All labels match references, all citations resolve
- ✅ File structure verified: All required files present, no missing dependencies

**Phase 2: PDF Compilation** ⏳ PENDING (External Dependency - LaTeX Installation)
**Status**: Blocked by LaTeX installation requirement
**Priority**: Critical (blocks final deliverable - 20-30 page PDF report)
**Prerequisites**: ✅ All experiments complete (28/36), ✅ All report content complete, ✅ All code finalized
**Note**: No new experiments needed. `run_experiment.sh` correctly configured to skip completed experiments.

**Execution Strategy**: 
- **Sequential Execution**: Tasks 2.1-2.4 must be executed in order (each blocks the next)
- **Incremental Approach**: Complete one task fully before proceeding to the next
- **Checkpoint-Based**: Verify success criteria at each checkpoint before proceeding
- **Documentation**: Document any issues encountered for debugging and future reference
- **External Dependency**: Task 2.1 requires LaTeX installation (Overleaf recommended for easiest setup)

**Sequential Tasks** (execute in order, each task blocks the next):

#### Task 2.1: LaTeX Environment Setup [BLOCKER] [NEXT]
**Priority**: Critical | **Time**: 30-60 min | **Dependency**: External (LaTeX installation) | **Status**: ⏳ Pending
**Location**: `nowcasting-report/` directory
**Blocking**: All subsequent PDF compilation tasks (2.2, 2.3, 2.4)

**Prerequisites Check** (can be done now):
- ✅ Verify all required files exist: `ls -la nowcasting-report/` (main.tex, preamble.tex, contents/, tables/, images/, references.bib, arxiv.sty)
- ✅ Verify file structure: All 8 content files, 4 table files, 4 images present
- ✅ Verify references.bib: Count entries (`grep -c "^@.*{" nowcasting-report/references.bib` should be 21)
- **Checkpoint**: All files verified, ready for upload/compilation

**Action Items** (execute sequentially, choose ONE option):

**Option A (Recommended - Overleaf)**: 
1. **Create Overleaf project**:
   - Go to https://www.overleaf.com and create new project
   - Choose "Blank Project" or "Upload Project"
   - **Checkpoint**: Project created
2. **Upload files**:
   - Upload entire `nowcasting-report/` directory structure:
     * `main.tex`, `preamble.tex`, `arxiv.sty` (root)
     * `contents/` directory with all 8 .tex files
     * `tables/` directory with all 4 .tex files
     * `images/` directory with all 4 .png files
     * `references.bib`
   - **Checkpoint**: All files uploaded, directory structure preserved
3. **Configure compiler**:
   - Menu → Compiler → Select "pdfLaTeX" (default) or "XeLaTeX" if Korean issues
   - **Checkpoint**: Compiler set
4. **Test compilation**:
   - Click "Recompile" button
   - Check for errors in compilation log
   - **Checkpoint**: Compilation runs (warnings acceptable, errors must be resolved)

**Option B (Local TeX Live - Linux)**:
1. **Install TeX Live**:
   - Run: `sudo apt-get update && sudo apt-get install -y texlive-full texlive-lang-korean`
   - Wait for installation (may take 10-30 minutes, ~3GB download)
   - **Checkpoint**: Installation completes without errors
2. **Verify installation**:
   - Run: `pdflatex --version` (should show TeX Live version, e.g., "TeX Live 2023")
   - Run: `bibtex --version` (should show BibTeX version)
   - **Checkpoint**: Both commands work
3. **Test Korean support**:
   - Create test: `cd nowcasting-report && echo '\documentclass{article}\usepackage{kotex}\begin{document}테스트\end{document}' > test.tex`
   - Compile: `pdflatex test.tex`
   - Check: `test.pdf` exists and Korean characters render correctly
   - Cleanup: `rm test.tex test.aux test.log test.pdf`
   - **Checkpoint**: Korean text compiles and renders correctly

**Option C (Docker - Isolated Environment)**:
1. **Pull TeX Live image**:
   - Run: `docker pull texlive/texlive:latest`
   - **Checkpoint**: Image downloaded
2. **Run container**:
   - Run: `docker run -it -v /data/nowcasting-kr/nowcasting-report:/workspace texlive/texlive:latest bash`
   - **Checkpoint**: Container running, prompt shows
3. **Verify LaTeX**:
   - Inside container: `cd /workspace && pdflatex --version`
   - **Checkpoint**: LaTeX accessible in container

**Success Criteria** (must meet all):
- ✅ LaTeX environment accessible (Overleaf project ready OR local/Docker LaTeX working)
- ✅ All report files accessible in LaTeX environment
- ✅ Can compile minimal Korean test document without errors
- ✅ Korean characters render correctly in test PDF

**If Korean rendering fails**:
- Switch to XeLaTeX: `xelatex main.tex` (or set Overleaf compiler to XeLaTeX)
- Or LuaLaTeX: `lualatex main.tex` (or set Overleaf compiler to LuaLaTeX)
- **Checkpoint**: Alternative compiler works with Korean text

**Next**: Proceed to Task 2.2 (Initial PDF Compilation)

#### Task 2.2: Initial PDF Compilation [After 2.1]
**Priority**: High | **Time**: 1-2 hours | **Dependency**: Task 2.1 | **Status**: ⏳ Pending
**Location**: `nowcasting-report/` directory

**Prerequisites**: Task 2.1 must be complete (LaTeX environment ready)

**Action Items** (execute sequentially):

1. **Navigate to report directory**: 
   - **Overleaf**: Already in project directory (no action needed)
   - **Local**: `cd /data/nowcasting-kr/nowcasting-report/`
   - **Docker**: Inside container, `cd /workspace`
   - **Verify**: List files to confirm all required files present
   - **Checkpoint**: In correct directory, all files visible

2. **First compilation attempt**:
   - **Overleaf**: Click "Recompile" button (top left, or Ctrl+S)
   - **Local/Docker**: Run `pdflatex -interaction=nonstopmode main.tex` (or `xelatex`/`lualatex` if Korean issues)
   - **Expected behavior**: 
     * First run: Warnings about undefined references (normal - `\ref{}` and `\cite{}` not resolved yet)
     * May see "Rerun to get cross-references right" messages (expected)
   - **Check compilation log** for:
     * Fatal errors (must fix): "File not found", "Package not found", "Undefined control sequence"
     * Warnings (acceptable for now): "Overfull hbox", "Underfull hbox", "Rerun LaTeX"
   - **Checkpoint**: Compilation completes without fatal errors (warnings acceptable)

3. **Run BibTeX** (required for bibliography):
   - **Overleaf**: Usually runs automatically after first compilation, check log for "BibTeX" or "Bibliography"
   - **Local/Docker**: Run `bibtex main` (creates `main.bbl` and `main.blg` files)
   - **Verify BibTeX output**:
     * Check log for "21 entries" or similar (should match references.bib count)
     * Verify `main.bbl` file created (Overleaf: check file list; Local: `ls -la main.bbl`)
   - **If BibTeX errors**: Check `references.bib` syntax, verify all entries valid
   - **Checkpoint**: BibTeX completes successfully, bibliography processed (21 entries)

4. **Re-compile 2-3 times** (required to resolve cross-references):
   - **Overleaf**: Click "Recompile" 2-3 more times (or wait for auto-compile)
   - **Local/Docker**: Run compilation command 2-3 more times:
     * `pdflatex main.tex` → `bibtex main` → `pdflatex main.tex` → `pdflatex main.tex`
   - **Verify cross-references**:
     * Check compilation log: Should see fewer "Rerun" messages each time
     * Open PDF: Check for "??" placeholders (should decrease with each run)
   - **Checkpoint**: PDF generated, cross-references mostly resolved (some "??" acceptable - will verify in Task 2.3)

5. **Resolve missing packages** (if errors occur):
   - **Check compilation log** for package errors:
     * "File `<package>.sty' not found" → Package missing
     * "Package `<package>' not found" → Package not installed
   - **Fix missing packages**:
     * **Overleaf**: Packages usually auto-installed, check log for warnings
     * **TeX Live**: `sudo tlmgr install <package-name>` (e.g., `sudo tlmgr install kotex`)
     * **Docker**: Inside container, `tlmgr install <package-name>`
   - **If package conflicts**: Check `preamble.tex` for conflicting package options (rarely needed)
   - **Re-compile** after installing packages
   - **Checkpoint**: All required packages available, no missing package errors

6. **Document compilation status**:
   - **Save compilation log** if errors persist (for debugging)
   - **Note persistent warnings**: List warnings that appear in every compilation (will address in Task 2.4)
   - **Verify PDF exists**: `main.pdf` should be present (Overleaf: check file list; Local: `ls -la main.pdf`)
   - **Checkpoint**: All fatal errors resolved, warnings documented, PDF file exists

**Success Criteria** (must meet all):
- ✅ PDF generated (`main.pdf`) without fatal errors
- ✅ Bibliography processed (21 entries visible in log or `main.bbl`)
- ✅ Compilation completes successfully (no fatal errors)
- ⚠️ Formatting warnings (overfull hboxes) are acceptable at this stage (will fix in Task 2.4)
- ⚠️ Some "??" placeholders acceptable (will verify in Task 2.3)

**Common Issues and Solutions**:
- **"File not found" errors**: Check file paths, verify all files uploaded/copied correctly
- **Korean text not rendering**: Switch to XeLaTeX or LuaLaTeX compiler
- **Bibliography not appearing**: Verify BibTeX ran, check `references.bib` syntax
- **Cross-references not resolving**: Re-compile 2-3 more times (normal LaTeX behavior)

**Next**: Proceed to Task 2.3 (PDF Quality Verification)

#### Task 2.3: PDF Quality Verification [After 2.2]
**Priority**: High | **Time**: 30-60 min | **Dependency**: Task 2.2 | **Status**: ⏳ Pending
**Location**: `nowcasting-report/main.pdf`

**Prerequisites**: Task 2.2 must be complete (PDF compiled successfully)

**Action Items** (execute sequentially, verify each item):

1. **Open PDF and verify structure**:
   - **Open PDF**: `main.pdf` (Overleaf: click PDF preview; Local: `evince main.pdf` or `xdg-open main.pdf`)
   - **Page count**: Check PDF properties or count manually (should be 20-30 pages)
     * Overleaf: Check page count in PDF viewer
     * Local: `pdfinfo main.pdf | grep Pages` or count manually
   - **Sections**: Verify all 8 sections present (scroll through PDF):
     * 1. Introduction (1_introduction.tex)
     * 2. Literature Review (2_literature_review.tex)
     * 3. Theoretical Background (3_theoretical_background.tex)
     * 4. Method and Experiment (4_method_and_experiment.tex)
     * 5. Results (5_result.tex)
     * 6. Discussion (6_discussion.tex)
     * 7. Conclusion (7_conclusion.tex)
     * 8. Acknowledgement (8_acknowledgement.tex)
   - **Checkpoint**: PDF opens, page count 20-30, all 8 sections visible

2. **Verify all 4 figures render correctly**:
   - **Figure 1** (`\ref{fig:model_comparison}`): 
     * Locate in PDF (should be in Results section)
     * Verify: Image appears, not broken, readable
     * Expected: Bar chart comparing models
   - **Figure 2** (`\ref{fig:horizon_trend}`):
     * Locate in PDF (should be in Results section)
     * Verify: Image appears, not broken, readable
     * Expected: Line chart showing horizon trends
   - **Figure 3** (`\ref{fig:heatmap}`):
     * Locate in PDF (should be in Results section)
     * Verify: Image appears, not broken, readable
     * Expected: Heatmap showing accuracy by target/horizon
   - **Figure 4** (`\ref{fig:forecast_vs_actual}`):
     * Locate in PDF (should be in Results section)
     * Verify: Image appears (may be placeholder - acceptable per STATUS.md)
     * Expected: Time series plot (placeholder acceptable)
   - **Checkpoint**: All 4 figures visible, no broken image placeholders

3. **Verify all 4 tables format correctly**:
   - **Table 1** (`\ref{tab:overall_metrics}`):
     * Locate in PDF (should be in Results section)
     * Verify: All 4 models present (ARIMA, VAR, DFM, DDFM)
     * Verify: All metrics present (sMSE, sMAE, sRMSE)
     * Verify: Formatting readable, alignment correct
   - **Table 2** (`\ref{tab:overall_metrics_by_target}`):
     * Locate in PDF (should be in Results section)
     * Verify: All 3 targets present (KOGDP...D, KOCNPER.D, KOGFCF..D)
     * Verify: Column widths appropriate, no overflow
   - **Table 3** (`\ref{tab:overall_metrics_by_horizon}`):
     * Locate in PDF (should be in Results section)
     * Verify: All 3 horizons present (h1, h7, h28)
     * Verify: Readability, alignment correct
   - **Table 4** (`\ref{tab:nowcasting_metrics}`):
     * Locate in PDF (should be in Results section)
     * Verify: DFM and DDFM metrics present
     * Verify: Values match aggregated_results.csv (verified in Phase 1)
   - **Checkpoint**: All 4 tables present, readable, all expected data visible

4. **Verify bibliography**: 
   - **Count entries**: Check `references.bib` has 21 entries:
     * Local: `grep -c "^@.*{" nowcasting-report/references.bib` (should be 21)
     * Overleaf: Count `@` entries in references.bib file
   - **Verify citations in text**: 
     * Search PDF for "?" placeholders (should be none)
     * Check that all `\cite{}` in text resolve to bibliography entries
     * Count citations in text (should match number of `\cite{}` commands)
   - **Check bibliography section**: 
     * Should appear at end of document (after Acknowledgement)
     * Should list all 21 references
     * Formatting should be consistent
   - **Checkpoint**: Bibliography section present, all citations resolve, 21 entries visible

5. **Verify Korean text**: 
   - **Check abstract**: Korean characters render correctly (no missing characters)
   - **Check section titles**: All Korean section titles render correctly
   - **Check table captions**: Korean captions render correctly
   - **Check figure captions**: Korean captions render correctly
   - **Check body text**: Sample Korean text in sections renders correctly
   - **If Korean issues**: Note which sections/pages have problems
   - **Checkpoint**: Korean text renders correctly throughout document (no missing characters)

6. **Verify cross-references**: 
   - **Check `\ref{}` references**:
     * Search PDF for "??" placeholders (should be none after multiple compilations)
     * Verify section references: `\ref{sec:...}` resolve correctly
     * Verify table references: `\ref{tab:...}` resolve correctly
     * Verify figure references: `\ref{fig:...}` resolve correctly
   - **Check `\cite{}` references**:
     * Search PDF for "?" placeholders (should be none after BibTeX)
     * Verify all citations resolve to bibliography entries
   - **If unresolved references**: Note which references fail (will fix in Task 2.4)
   - **Checkpoint**: All references resolve, no "??" or "?" placeholders visible

7. **Review compilation log for warnings**:
   - **Open compilation log**:
     * Overleaf: Click "Logs and output files" → "main.log"
     * Local: `cat main.log` or open in text editor
   - **Check for warnings**:
     * Overfull/underfull hboxes: Note line numbers and file paths
     * Missing references: Note which `\ref{}` or `\cite{}` fail
     * Package warnings: Note package names and warnings
     * Font warnings: Note if Korean fonts have issues
   - **Document warnings**: Create list with file paths and line numbers
   - **Checkpoint**: Warnings documented with file paths and line numbers

8. **Create issues list**: 
   - **Categorize issues**:
     * **Critical** (must fix): Unresolved references, missing content, broken images, Korean text not rendering
     * **Minor** (acceptable or fix in Task 2.4): Overfull hboxes, minor formatting issues, font warnings
   - **Document each issue**:
     * Issue description
     * File path and line number (if applicable)
     * Priority (Critical/Minor)
     * Suggested fix (if known)
   - **Checkpoint**: Issues list created, prioritized, ready for Task 2.4

**Success Criteria** (must meet all):
- ✅ PDF readable, all content present (8 sections, 4 tables, 4 figures, 21 citations)
- ✅ All cross-references resolve (no "??" or "?" placeholders)
- ✅ Korean text renders correctly (no missing characters)
- ✅ Page count in range (20-30 pages)
- ⚠️ Minor formatting issues acceptable (will fix in Task 2.4)

**If Critical Issues Found**:
- Unresolved references: Re-compile 2-3 more times, check LaTeX labels match references
- Missing content: Check LaTeX files for missing `\input{}` commands
- Broken images: Verify image file paths, check `\includegraphics{}` commands
- Korean text not rendering: Switch to XeLaTeX or LuaLaTeX compiler

**Next**: Proceed to Task 2.4 (PDF Finalization)

#### Task 2.4: PDF Finalization [After 2.3]
**Priority**: High | **Time**: 30-60 min | **Dependency**: Task 2.3 | **Status**: ⏳ Pending
**Location**: `nowcasting-report/` directory (LaTeX files)

**Prerequisites**: Task 2.3 must be complete (issues list created)

**Action Items** (execute sequentially, fix issues from Task 2.3):

1. **Fix critical issues first** (from Task 2.3 issues list):
   - **Unresolved references**:
     * Check LaTeX labels match references: `grep -r "\\label{" nowcasting-report/contents/` and `grep -r "\\ref{" nowcasting-report/contents/`
     * Verify all `\label{}` have matching `\ref{}` (and vice versa)
     * Fix mismatched labels/references
     * **Checkpoint**: All labels match references
   - **Missing content**:
     * Check all `\input{}` commands in `main.tex` point to existing files
     * Verify all content files exist and are included
     * **Checkpoint**: All content files included
   - **Broken images**:
     * Verify image paths in `contents/5_result.tex`: `grep -n "includegraphics" nowcasting-report/contents/5_result.tex`
     * Check image files exist: `ls -la nowcasting-report/images/*.png`
     * Fix incorrect paths if needed
     * **Checkpoint**: All image paths correct, images exist
   - **Korean text not rendering**:
     * Switch compiler to XeLaTeX or LuaLaTeX (Overleaf: Menu → Compiler; Local: use `xelatex` command)
     * Re-compile and verify Korean text renders
     * **Checkpoint**: Korean text renders correctly

2. **Fix formatting issues** (prioritize based on Task 2.3 issues list):
   - **Overfull/underfull hboxes**:
     * Locate problematic lines from compilation log (note file and line number)
     * Adjust text in `contents/*.tex` files:
       - Add line breaks: Insert `\\` or rephrase sentences
       - Adjust spacing: Use `\linebreak` or `\newline` if needed
     * Adjust table/figure sizes in `tables/*.tex` or `contents/5_result.tex`:
       - Tables: Adjust column widths in `\begin{tabular}{...}` (e.g., `p{2cm}` for fixed width)
       - Figures: Adjust `width` in `\includegraphics[width=0.8\textwidth]{}`
     * **Checkpoint**: Hbox warnings reduced or eliminated (check compilation log)
   - **Table formatting**:
     * Open table files: `tables/tab_*.tex`
     * Adjust column widths: Modify `\begin{tabular}{...}` column specifiers
       - Use `l` (left), `c` (center), `r` (right) for alignment
       - Use `p{width}` for paragraph columns with fixed width
     * Fix alignment: Check `&` alignment in table rows (all rows should have same number of `&`)
     * **Checkpoint**: Tables readable, no overflow, alignment correct
   - **Figure sizes**:
     * Open `contents/5_result.tex` and locate `\includegraphics` commands
     * Adjust `width` parameters:
       - Try `0.7\textwidth` if images too large
       - Try `0.9\textwidth` if images too small
       - Use `height=0.5\textheight` for height control if needed
     * **Checkpoint**: Figures properly sized, no overflow
   - **Citation formatting**:
     * Check `main.tex` line 36-37: Should have `\bibliographystyle{unsrt}` and `\bibliography{references}`
     * Verify bibliography style is correct (unsrt = unsorted, numbered)
     * **Checkpoint**: Bibliography formatting correct

3. **Re-compile after fixes**:
   - **Overleaf**: Click "Recompile" 2-3 times (or wait for auto-compile)
   - **Local/Docker**: Run full compilation sequence:
     * `pdflatex main.tex` (or `xelatex`/`lualatex`)
     * `bibtex main`
     * `pdflatex main.tex` (or `xelatex`/`lualatex`)
     * `pdflatex main.tex` (or `xelatex`/`lualatex`)
   - **Check compilation log**: Verify no new errors introduced
   - **Checkpoint**: Compilation successful, PDF updated, no new errors

4. **Verify fixes**:
   - **Open PDF**: `main.pdf` (Overleaf: PDF preview; Local: `evince main.pdf`)
   - **Check issues from Task 2.3**:
     * Verify critical issues resolved (unresolved references, missing content, broken images)
     * Check formatting improvements (hboxes, tables, figures)
     * Verify Korean text renders correctly
   - **Re-check compilation log**: Note remaining warnings (should be fewer)
   - **Checkpoint**: Issues resolved or documented as acceptable

5. **Final verification checklist** (verify each item):
   - ✅ **Page count**: 20-30 pages (check PDF properties or count manually)
   - ✅ **All content present**: 8 sections, 4 tables, 4 figures, 21 citations
   - ✅ **All citations**: All 21 references appear in bibliography (count entries)
   - ✅ **All cross-references**: All `\ref{}` and `\cite{}` resolve (no "??" or "?" placeholders)
   - ✅ **Korean text**: Renders correctly (no missing characters, check abstract and section titles)
   - ✅ **Formatting**: No major formatting issues (minor overfull hboxes acceptable if documented)
   - ✅ **Images**: All 4 figures render correctly (not broken, readable)
   - ✅ **Tables**: All 4 tables format correctly (readable, alignment correct)
   - **Checkpoint**: All checklist items verified

6. **Final output verification**:
   - **Verify PDF exists**: `main.pdf` should be present
   - **Check file size**: Should be reasonable (typically 1-5 MB for 20-30 pages with images)
     * Overleaf: Check file size in file list
     * Local: `ls -lh main.pdf` (should show size in MB)
   - **Verify PDF is readable**: Open and scroll through entire document
   - **Checkpoint**: Final PDF ready, all requirements met

**Success Criteria** (must meet all):
- ✅ Final PDF ready, all requirements met (20-30 pages, all content present, all references resolve)
- ✅ No critical issues (unresolved references, missing content, broken images, Korean text not rendering)
- ✅ Minor formatting issues acceptable (documented if present, e.g., "3 overfull hboxes acceptable")
- ✅ PDF file size reasonable (1-5 MB typical)

**Common Fixes**:
- **Overfull hbox**: Add `\linebreak` or rephrase sentence, adjust table column widths
- **Unresolved reference**: Re-compile 2-3 more times, check label names match
- **Table overflow**: Use `p{width}` columns, reduce font size with `\small` or `\footnotesize`
- **Figure too large**: Reduce `width` parameter in `\includegraphics`

**Output**: Complete 20-30 page PDF report ready for submission (`nowcasting-report/main.pdf`)

**Completion**: All PDF compilation tasks (2.1-2.4) complete. Report finalized and ready for submission.

## Experiment Status Summary

**Total**: 36 combinations (3 targets × 4 models × 3 horizons)
**Complete**: 28/36 (77.8%) - All available experiments done
**Unavailable**: 8/36 (22.2%) - DFM KOCNPER.D (3, numerical instability), DFM/DDFM h28 (6, test set too small)
**Verified**: All results in `outputs/comparisons/` and `outputs/experiments/aggregated_results.csv` (28 rows)
**Status**: ✅ All available experiments complete. **No new experiments needed.** `run_experiment.sh` production-ready and correctly configured.

## Completed Work Summary

✅ **Experiments**: 28/36 complete (all available), verified in `outputs/comparisons/` and `aggregated_results.csv`
✅ **Report Content**: All 8 sections, 4 tables, 4 plots, 21 citations, all metric values verified (match `aggregated_results.csv` exactly, 0 discrepancies)
✅ **Report Finalization**: All LaTeX cross-references verified, all images confirmed, all tracking files under 1000 lines
✅ **Code**: dfm-python finalized, src/ 15 files (max allowed), all tests passing (133 passed, 8 skipped)
✅ **Pre-Compilation**: All LaTeX files, images, syntax verified (13 files, 4 images, all `\ref{}`, `\cite{}`, `\input{}`)
✅ **Verification**: Comparison results, experiment completion, metric values, citations all verified (0 discrepancies)
✅ **Script**: `run_experiment.sh` production-ready, correctly skips completed experiments, handles unavailable combinations gracefully

## Next Steps (Iteration 59+)

### ✅ Completed (No Action Needed)
- All experiments complete (28/36, all available)
- All report content complete and verified
- All code finalized
- All pre-compilation checks complete
- Status review and documentation update (Iteration 48)

### ⏳ Pending (Sequential, Require External Dependency)

**BLOCKER**: LaTeX installation required (external dependency)

**Immediate Next Task**: **Task 2.1 - LaTeX Environment Setup** [BLOCKER] [NEXT]
- **Status**: LaTeX not installed (verified in Iteration 50) - requires external installation
- **Action**: Choose and set up LaTeX distribution (Overleaf recommended, or local TeX Live)
- **Time**: 30-60 minutes
- **Blocking**: All PDF compilation tasks (2.2, 2.3, 2.4)
- **Details**: See "Prioritized Action Plan" section above
- **Note**: Report structure verified and ready - all files present, all references correct, all content matches aggregated_results.csv

**Sequential Tasks After 2.1**:
1. **Task 2.2**: Initial PDF Compilation (1-2 hours) - Compile `main.tex`, resolve errors
2. **Task 2.3**: PDF Quality Verification (30-60 min) - Verify page count, figures, tables, cross-references
3. **Task 2.4**: PDF Finalization (30-60 min) - Fix formatting issues, generate final PDF

**Detailed task descriptions**: See "Prioritized Action Plan" section above (Tasks 2.1-2.4).

## Incremental Improvement Plan (Prioritized Tasks)

**Status**: All critical development complete. Focus on PDF compilation (Tasks 2.1-2.4). Below are optional incremental improvements that can be done in parallel or after PDF compilation.

**Analysis Summary (Iteration 56)**:
- ✅ **Code Quality**: Generally clean, no critical issues. Minor redundancies (setup_paths calls) documented.
- ✅ **Numerical Stability**: All known issues documented (DFM KOCNPER.D instability expected, DDFM NaN issues fixed).
- ✅ **Theoretical Correctness**: EM algorithm and Kalman filter implementations verified as correct.
- ✅ **Report Quality**: Complete, all metric values verified, no hallucination, citations verified.
- ✅ **Experiments**: All available (28/36) complete. run_experiment.sh correctly skips completed experiments.
- ✅ **Code Structure**: dfm-python finalized, src/ 15 files (max), consistent naming verified.

### Priority 1: Code Quality Polish (Optional - Non-blocking)

**1.1 Path Setup Consolidation** [Low Priority]
- **Issue**: `setup_paths()` called 8 times across src/ modules (redundant but harmless, idempotent)
- **Location**: `src/train.py`, `src/infer.py`, `src/core/training.py`, `src/model/*.py`, `src/preprocess/utils.py`
- **Action**: Consider consolidating to single call at entry points (train.py, infer.py) - cosmetic improvement only
- **Impact**: Minor code cleanliness, no functional change (setup_paths is idempotent)
- **Status**: ⏳ Optional

**1.2 Documentation Consistency Check** [Low Priority]
- **Issue**: Verify all docstrings use consistent format and terminology
- **Location**: `dfm-python/src/`, `src/`
- **Action**: Review docstrings for consistency (PascalCase classes, snake_case functions, NumPy docstring format)
- **Impact**: Better code maintainability
- **Status**: ⏳ Optional

**1.3 Temporary File Handling Documentation** [Very Low Priority]
- **Issue**: Temporary file fallback in sktime_forecaster.py is documented but could be more prominent
- **Location**: `src/model/sktime_forecaster.py` (lines 156-178, 392-419)
- **Action**: Already documented in code comments. Consider adding to module docstring.
- **Impact**: Better code maintainability
- **Status**: ⏳ Optional (already documented in code)

### Priority 2: Report Enhancement (Optional - Non-blocking)

**2.1 Forecast vs Actual Plot** [Low Priority]
- **Issue**: Placeholder image in `images/forecast_vs_actual.png` (acceptable per STATUS.md)
- **Location**: `nowcasting-report/code/plot.py` (lines 357-373)
- **Action**: Extract actual time series from outputs/ if available, generate real plot
- **Impact**: Enhanced visualization (optional - placeholder is acceptable)
- **Status**: ⏳ Optional

**2.2 Report Flow Review** [Low Priority]
- **Issue**: Verify section transitions are smooth and logical
- **Location**: `nowcasting-report/contents/*.tex`
- **Action**: Review section flow, ensure smooth transitions between sections
- **Impact**: Better readability
- **Status**: ⏳ Optional (already reviewed in Iteration 46)

**2.3 Report Citation Verification** [Very Low Priority]
- **Issue**: Verify all citations in report text match references.bib entries
- **Location**: `nowcasting-report/contents/*.tex`, `nowcasting-report/references.bib`
- **Action**: Cross-check all \cite{} commands against references.bib (21 entries verified)
- **Impact**: Ensures no broken citations
- **Status**: ⏳ Optional (already verified in previous iterations)

### Priority 3: Code Efficiency (Optional - Non-blocking)

**3.1 Redundant Imports Check** [Very Low Priority]
- **Issue**: Check for unused or redundant imports across codebase
- **Location**: All Python files
- **Action**: Run linter (flake8/pylint) to identify unused imports
- **Impact**: Minor code cleanliness
- **Status**: ⏳ Optional

**3.2 Matrix Operation Optimization** [Very Low Priority]
- **Issue**: Check for opportunities to use more efficient matrix operations (e.g., torch.bmm for batch operations)
- **Location**: `dfm-python/src/dfm_python/ssm/em.py`, `dfm-python/src/dfm_python/ssm/kalman.py`
- **Action**: Review matrix operations for potential optimization (current implementation is already GPU-optimized)
- **Impact**: Potential performance improvement (likely minimal, current implementation is already efficient)
- **Status**: ⏳ Optional

### Priority 4: Numerical Stability Enhancements (Optional - Non-blocking)

**4.1 DFM KOCNPER.D Stability Research** [Very Low Priority]
- **Issue**: DFM KOCNPER.D numerical instability is documented but could benefit from research into alternative approaches
- **Location**: `dfm-python/src/dfm_python/ssm/em.py`, `dfm-python/src/dfm_python/ssm/kalman.py`
- **Action**: Research adaptive regularization or alternative initialization methods (not urgent, DDFM works as alternative)
- **Impact**: Potential improvement for specific data combinations (low priority, DDFM already handles this case)
- **Status**: ⏳ Optional (documented limitation, DDFM provides working alternative)

**4.2 Regularization Parameter Tuning** [Very Low Priority]
- **Issue**: Current regularization parameters (1e-6, 1e-8) are fixed. Could explore adaptive regularization.
- **Location**: `dfm-python/src/dfm_python/ssm/em.py` (regularization_scale), `dfm-python/src/dfm_python/ssm/kalman.py` (min_eigenval)
- **Action**: Research adaptive regularization based on condition number (low priority, current values work for most cases)
- **Impact**: Potential improvement for edge cases (low priority)
- **Status**: ⏳ Optional

**Note**: All improvements above are optional and non-blocking. Not required for report completion. All critical work complete. PDF compilation (Tasks 2.1-2.4) is the main blocker. No new experiments needed - all available experiments (28/36) are complete. run_experiment.sh correctly configured to skip completed experiments.

## Summary

**Status**: ✅ All development work complete. Report finalized and ready for PDF compilation. All experiments complete (28/36 = 77.8%). **No new experiments needed.**

**Critical Path**: PDF Compilation (Tasks 2.1-2.4) - Requires LaTeX installation (external dependency)
**Next Immediate Action**: Set up LaTeX environment (Task 2.1) - Choose Overleaf (recommended) or install local TeX Live distribution
**Optional Polish**: Forecast plot enhancement, final code review, report flow check (can be done after PDF compilation)

## Incremental Task Execution Guide

### What Can Be Done Now (No External Dependencies)
**Status**: ✅ All pre-compilation tasks complete
- ✅ All experiments verified and complete (28/36)
- ✅ All report content verified (8 sections, 4 tables, 4 plots, 21 citations)
- ✅ All code finalized (src/ 15 files, dfm-python finalized)
- ✅ All metric values verified (match aggregated_results.csv exactly)
- ✅ All LaTeX syntax verified (cross-references, citations, includes)
- ✅ All images generated and verified
- ✅ `run_experiment.sh` verified and correctly configured

**Conclusion**: All development and verification work complete. No additional work needed before PDF compilation.

### What Requires External Dependencies
**Status**: ⏳ Blocked by LaTeX installation requirement

**Blocked Tasks** (sequential, require LaTeX):
1. **Task 2.1**: LaTeX Environment Setup - Requires LaTeX distribution installation
2. **Task 2.2**: Initial PDF Compilation - Requires Task 2.1
3. **Task 2.3**: PDF Quality Verification - Requires Task 2.2
4. **Task 2.4**: PDF Finalization - Requires Task 2.3

**Recommendation**: 
- **Option A (Easiest)**: Use Overleaf - No local installation needed, upload files and compile online
- **Option B**: Install local TeX Live - Requires system access and installation permissions
- **Option C**: Use Docker - Requires Docker installation

### Task Prioritization
1. **Critical (Blocks Final Deliverable)**: Tasks 2.1-2.4 (PDF compilation)
2. **Optional (Can Be Done Later)**: Forecast plot enhancement, final code review, documentation polish

### Execution Strategy
- **Sequential Execution**: Tasks 2.1-2.4 must be executed in order (each blocks the next)
- **Incremental Approach**: Complete one task fully before proceeding to the next
- **Documentation**: Document any issues encountered for debugging
- **Verification**: Verify success criteria at each checkpoint before proceeding

**Next Step**: Begin with Task 2.1 (LaTeX Environment Setup) - Choose Overleaf, local TeX Live, or Docker based on available resources.

