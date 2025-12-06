# Issues and Action Plan

## Executive Summary (2025-12-06 - New Experiment Phase)

**Current State**: 
- ✅ **Configuration**: All 4 target configs created, series configs updated (block: null), data path fixed
- ✅ **Scripts**: `run_experiment.sh` and `run_test_experiment.sh` finalized and verified
- ✅ **Report Structure**: 6 sections ready (Introduction, Methodology, Production, Investment, Consumption, Conclusion) - condensed to under 15 pages
- ⏳ **Experiments**: Need to run for 4 new targets (48 combinations: 4 × 4 × 3)
- ⏳ **Report Content**: Need to populate with actual experiment results

**Goal**: Complete under 15 page LaTeX report comparing 4 forecasting models (ARIMA, VAR, DFM, DDFM) on 4 Korean macroeconomic targets (Production: KOIPALL.G, KOMPRI30G; Investment: KOEQUIPTE; Consumption: KOWRCCNSE)  
**Status**: ⏳ Configuration ready, experiments pending  
**Progress**: 0/48 = 0% (experiments not yet run)

**Next Steps**:
1. ⏳ **Task 1.1**: Verify setup with test script [NEXT] - Run `./run_test_experiment.sh` to verify all targets and models work
2. ⏳ **Task 1.2**: Run full experiments - Run `./run_experiment.sh` for all 4 targets × 4 models × 3 horizons
3. ⏳ **Task 1.3**: Generate results and plots - Aggregate results, generate plots for report
4. ⏳ **Task 1.4**: Update report - Populate tables and sections with actual results
5. ⏳ **Task 1.5**: Compile PDF - Verify under 15 pages, all content present

## Experiment Status (2025-12-06)

**New Configuration**:
- **Targets**: 4 (KOEQUIPTE, KOWRCCNSE, KOIPALL.G, KOMPRI30G)
- **Models**: 4 (ARIMA, VAR, DFM, DDFM)
- **Horizons**: 3 (1, 7, 28 days)
- **Total**: 48 combinations (4 × 4 × 3)

**Status**: Experiments not yet run. Use `run_test_experiment.sh` for verification before full run.

**Configuration Details**:
- All series configs: `block: null` (only global block for DFM/DDFM)
- Data file: `data/data.csv`
- Config files: `config/experiment/{koequipte,kowrccnse,koipallg,kompri30g}_report.yaml`

## ✅ Resolved Issues

**Configuration Updates Completed**:
- ✅ **Target Configs**: All 4 target configs created and verified
- ✅ **Series Configs**: All 101 series configs updated with `block: null`
- ✅ **Data Path**: All configs use `data/data.csv`
- ✅ **Scripts**: `run_experiment.sh` and `run_test_experiment.sh` finalized
- ✅ **Test Verification**: ARIMA and VAR tests passing

## Known Limitations

1. **Horizon 28**: May be unavailable for some targets if test set <28 points (80/20 split)
2. **DFM Numerical Instability**: Some targets may show numerical instability (EM algorithm fails) - will be documented as encountered
3. **Report Length**: Target is under 15 pages (condensed from previous 20-30 page target)

## Action Plan (Incremental Tasks)

**Phase 1: Experiment Execution** ⏳ IN PROGRESS
**Status**: Configuration ready, experiments pending
**Priority**: Critical (blocks report completion)

**Sequential Tasks** (execute in order):

#### Task 1.1: Verify Setup [NEXT]
**Priority**: Critical | **Time**: 30-60 min | **Status**: ⏳ Pending
**Actions**: Run `./run_test_experiment.sh` to verify all 4 targets and 4 models work correctly
**Success Criteria**: All test combinations pass (16 tests: 4 targets × 4 models)

#### Task 1.2: Run Full Experiments [After 1.1]
**Priority**: Critical | **Time**: Several hours | **Status**: ⏳ Pending
**Actions**: Run `./run_experiment.sh` for all 4 targets × 4 models × 3 horizons (48 combinations)
**Success Criteria**: All experiments complete, results saved in `outputs/comparisons/`

#### Task 1.3: Generate Results and Plots [After 1.2]
**Priority**: High | **Time**: 30-60 min | **Status**: ⏳ Pending
**Actions**: Generate aggregated results CSV, generate plots for report
**Success Criteria**: `aggregated_results.csv` created, all plots generated in `nowcasting-report/images/`

#### Task 1.4: Update Report [After 1.3]
**Priority**: High | **Time**: 1-2 hours | **Status**: ⏳ Pending
**Actions**: Update tables with actual metrics, update result sections with findings, update discussion and conclusion
**Success Criteria**: All tables populated, all sections updated with actual results, no placeholders

#### Task 1.5: Compile PDF [After 1.4]
**Priority**: High | **Time**: 30-60 min | **Status**: ⏳ Pending
**Actions**: Compile LaTeX report, verify under 15 pages, fix formatting issues
**Success Criteria**: PDF generated, under 15 pages, all content present, all references resolve

**Execution Strategy**: Sequential tasks (1.1 → 1.2 → 1.3 → 1.4 → 1.5), complete one task fully before proceeding to the next.

## Experiment Status Summary

**Total**: 48 combinations (4 targets × 4 models × 3 horizons)
**Complete**: 0/48 (0%) - Experiments not yet run
**Pending**: 48/48 (100%) - All experiments need to be run

**Targets**:
- KOEQUIPTE (Equipment Investment Index)
- KOWRCCNSE (Wholesale and Retail Trade Sales)
- KOIPALL.G (Industrial Production Index, All Industries)
- KOMPRI30G (Manufacturing Production Index)

**Models**: ARIMA, VAR, DFM, DDFM
**Horizons**: 1, 7, 28 days

**Status**: ⏳ Ready to run experiments. Use `run_test_experiment.sh` for verification, then `run_experiment.sh` for full run.

## Summary

**Status**: ⏳ Configuration ready, experiments pending. Report structure ready, need results.

**Experiment Status**:
- ⏳ **Pending**: 48/48 (100%) - All experiments need to be run for new targets

**Critical Path**: Experiment Execution (Tasks 1.1-1.5)
**Next Immediate Action**: Run test script (`./run_test_experiment.sh`) to verify setup, then run full experiments (`./run_experiment.sh`)
