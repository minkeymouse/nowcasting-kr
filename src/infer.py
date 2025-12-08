from pathlib import Path
import sys
import argparse
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pickle
import json

script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from src.utils import setup_cli_environment, get_project_root, parse_experiment_config, extract_experiment_params, validate_experiment_config, setup_logging

setup_cli_environment()
setup_logging()


def _load_model_for_inference(project_root: Path, target_series: str, model: str) -> Tuple[Any, Any]:
    """Load a trained model from checkpoint for inference.
    
    Args:
        project_root: Root directory of the project
        target_series: Target series name (e.g., 'KOEQUIPTE')
        model: Model name (e.g., 'dfm', 'ddfm', 'arima', 'var')
        
    Returns:
        Tuple of (forecaster, dfm_model) where dfm_model is None for non-DFM models
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        ValueError: If checkpoint file is invalid or missing required data
        pickle.UnpicklingError: If checkpoint file is corrupted
    """
    checkpoint_path = project_root / "checkpoint" / f"{target_series}_{model.lower()}" / "model.pkl"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Please train the model first using: bash run_train.sh"
        )
    
    if not checkpoint_path.is_file():
        raise ValueError(f"Checkpoint path is not a file: {checkpoint_path}")
    
    try:
        with open(checkpoint_path, 'rb') as f:
            model_data = pickle.load(f)
    except (pickle.UnpicklingError, EOFError, IOError) as e:
        raise ValueError(f"Failed to load checkpoint from {checkpoint_path}: {e}") from e
    
    if not isinstance(model_data, dict):
        raise ValueError(f"Invalid checkpoint format: expected dict, got {type(model_data)}")
    
    if 'forecaster' not in model_data:
        raise ValueError(f"Checkpoint missing 'forecaster' key. Available keys: {list(model_data.keys())}")
    
    forecaster = model_data['forecaster']
    
    if model.lower() not in ['dfm', 'ddfm']:
        return forecaster, None
    
    dfm_model = getattr(forecaster, '_dfm_model', None) or getattr(forecaster, '_ddfm_model', None)
    if dfm_model is None:
        raise ValueError(f"DFM/DDFM forecaster missing underlying model. Checkpoint may be corrupted.")
    
    if 'result' in model_data and dfm_model._result is None:
        dfm_model._result = model_data['result']
    if 'data_module' in model_data and dfm_model._data_module is None:
        dfm_model._data_module = model_data['data_module']
    
    return forecaster, dfm_model


def run_backtest_evaluation(config_name: str, model: str, train_start: str = "1985-01-01", train_end: str = "2019-12-31",
                            nowcast_start: str = "2024-01-01", nowcast_end: str = "2025-10-31",
                            weeks_before: Optional[List[int]] = None, config_dir: Optional[str] = None,
                            overrides: Optional[List[str]] = None) -> Dict[str, Any]:
    project_root = get_project_root()
    config_dir = config_dir or str(project_root / "config")
    weeks_before = weeks_before or [4, 1]
    
    cfg = parse_experiment_config(config_name, config_dir, overrides)
    validate_experiment_config(cfg, require_target=True, require_models=False)
    target_series = extract_experiment_params(cfg)['target_series']
    
    if model.lower() not in ['dfm', 'ddfm']:
        results = {'target_series': target_series, 'model': model.upper(), 'train_period': f"{train_start} to {train_end}",
                   'nowcast_period': f"{nowcast_start} to {nowcast_end}", 'weeks_before': weeks_before, 'status': 'not_supported',
                   'error': f"Nowcasting not supported for {model.upper()}", 'summary': {'total_timepoints': len(weeks_before), 'successful_timepoints': 0, 'failed_timepoints': len(weeks_before)}}
        output_file = project_root / "outputs" / "backtest" / f"{target_series}_{model.lower()}_backtest.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(output_file, 'w', encoding='utf-8'), indent=2, default=str, ensure_ascii=False)
        return results
    
    forecaster, dfm_model = _load_model_for_inference(project_root, target_series, model)
    
    start_date = datetime.strptime(nowcast_start, "%Y-%m-%d").replace(day=1)
    end_date = datetime.strptime(nowcast_end, "%Y-%m-%d")
    target_periods = []
    current = start_date
    while current <= end_date:
        target_periods.append(current.replace(day=31) if current.month == 12 else (current.replace(month=current.month + 1, day=1) - timedelta(days=1)))
        current += relativedelta(months=1)
    
    from src.preprocessing import resample_to_monthly
    data_path = project_root / "data" / "data.csv"
    if not data_path.exists():
        data_path = project_root / "data" / "sample_data.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {project_root / 'data' / 'data.csv'}\n"
            f"Also checked: {project_root / 'data' / 'sample_data.csv'}\n"
            f"Please ensure data file exists in data/ directory"
        )
    
    try:
        full_data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, IOError) as e:
        raise ValueError(f"Failed to read data file {data_path}: {e}") from e
    
    if len(full_data) == 0:
        raise ValueError(f"Data file {data_path} is empty")
    
    if target_series not in full_data.columns:
        raise ValueError(
            f"Target series '{target_series}' not found in data. "
            f"Available columns: {list(full_data.columns)}"
        )
    
    full_data_monthly = resample_to_monthly(full_data)
    
    train_data = full_data_monthly[(full_data_monthly.index >= pd.Timestamp(train_start)) & (full_data_monthly.index <= pd.Timestamp(train_end))]
    
    if len(train_data) == 0:
        raise ValueError(
            f"No training data available in period {train_start} to {train_end}. "
            f"Available data range: {full_data_monthly.index.min()} to {full_data_monthly.index.max()}"
        )
    
    if target_series not in train_data.columns:
        raise ValueError(f"Target series '{target_series}' not found in training data")
    
    train_std = train_data[target_series].std() if len(train_data) > 0 else 1.0
    if train_std == 0 or not np.isfinite(train_std):
        train_std = 1.0  # Fallback to avoid division by zero
    
    train_end_date = pd.Timestamp(train_end)
    results_by_timepoint = {}
    
    for weeks in weeks_before:
        monthly_results = []
        for target_month_end in target_periods:
            target_month_end_ts = pd.Timestamp(target_month_end)
            view_date = target_month_end_ts - timedelta(weeks=weeks)
            if view_date <= train_end_date:
                continue
            
            try:
                # Get standardization parameters from trained model
                result = dfm_model.result
                Mx = result.Mx  # Mean for standardization
                Wx = result.Wx  # Standard deviation for standardization
                
                # Prepare data up to view date
                data_up_to_view = full_data[full_data.index <= view_date].copy()
                
                # Resample to monthly and align with model config
                from src.preprocessing import resample_to_monthly
                data_monthly = resample_to_monthly(data_up_to_view)
                
                if hasattr(dfm_model, 'config') and dfm_model.config:
                    from dfm_python.utils.helpers import get_series_ids
                    available = [s for s in get_series_ids(dfm_model.config) if s in data_monthly.columns]
                    if available:
                        data_monthly = data_monthly[available]
                    elif len(data_monthly.columns) == 0:
                        continue
                
                # Get the last few periods for update (use last 5 periods)
                n_periods = min(5, len(data_monthly))
                if n_periods == 0:
                    continue
                
                X_new_raw = data_monthly.iloc[-n_periods:].values
                
                # Standardize new data using the same parameters from training
                X_new_std = (X_new_raw - Mx) / Wx
                
                # Handle NaN/Inf values
                X_new_std = np.where(np.isfinite(X_new_std), X_new_std, np.nan)
                
                # Update model state with new standardized data, then predict
                # Pattern: model.update(X_std).predict(horizon=1)
                X_nowcast, Z_nowcast = dfm_model.update(X_new_std).predict(horizon=1)
                
                # Extract nowcast for target series
                target_idx = list(data_monthly.columns).index(target_series) if target_series in data_monthly.columns else None
                if target_idx is None:
                    continue
                
                forecast_value = X_nowcast[0, target_idx]
                
                if not np.isfinite(forecast_value):
                    continue
            except Exception as e:
                # Log error for debugging but continue with next iteration
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Nowcast failed for {target_series} at {view_date}: {e}")
                continue
            
            available_dates = full_data_monthly.index[full_data_monthly.index <= target_month_end_ts]
            if len(available_dates) == 0:
                continue
            raw_value = full_data_monthly.loc[available_dates.max(), target_series] if target_series in full_data_monthly.columns else full_data_monthly.loc[available_dates.max()].iloc[0]
            if pd.isna(raw_value):
                continue
                
            error = forecast_value - float(raw_value)
            train_std_sq = train_std ** 2 if train_std > 0 else 1.0
            monthly_results.append({
                'month': target_month_end_ts.strftime('%Y-%m'), 'view_date': view_date.strftime('%Y-%m-%d'), 'weeks_before': weeks,
                'forecast_value': float(forecast_value), 'actual_value': float(raw_value), 'error': float(error),
                'abs_error': float(abs(error)), 'squared_error': float(error ** 2),
                'sMSE': float((error ** 2) / train_std_sq), 'sMAE': float(abs(error) / train_std if train_std > 0 else abs(error))
            })
        
        if monthly_results:
            results_by_timepoint[f"{weeks}weeks"] = {
                'weeks_before': weeks,
                'monthly_results': monthly_results,
                'overall_sMAE': float(np.nanmean([r['sMAE'] for r in monthly_results])),
                'overall_sMSE': float(np.nanmean([r['sMSE'] for r in monthly_results])),
                'overall_mae': float(np.nanmean([r['abs_error'] for r in monthly_results])),
                'overall_rmse': float(np.sqrt(np.nanmean([r['squared_error'] for r in monthly_results]))),
                'n_months': len(monthly_results)
            }
    
    results = {'target_series': target_series, 'model': model.upper(), 'train_period': f"{train_start} to {train_end}",
               'nowcast_period': f"{nowcast_start} to {nowcast_end}", 'weeks_before': weeks_before, 'results_by_timepoint': results_by_timepoint,
               'horizon': 1, 'status': 'completed' if results_by_timepoint else 'no_results',
               'summary': {'total_timepoints': len(weeks_before), 'successful_timepoints': len(results_by_timepoint), 'failed_timepoints': len(weeks_before) - len(results_by_timepoint)}}
    output_file = project_root / "outputs" / "backtest" / f"{target_series}_{model}_backtest.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(output_file, 'w', encoding='utf-8'), indent=2, default=str, ensure_ascii=False)
    return results


def main():
    parser = argparse.ArgumentParser(description="Run nowcasting backtest")
    subparsers = parser.add_subparsers(dest='command', required=True)
    backtest_parser = subparsers.add_parser('backtest', help='Run nowcasting backtest (DFM and DDFM only)')
    backtest_parser.add_argument("--config-name", required=True)
    backtest_parser.add_argument("--model", required=True, choices=['dfm', 'ddfm'])
    backtest_parser.add_argument("--train-start", default="1985-01-01")
    backtest_parser.add_argument("--train-end", default="2019-12-31")
    backtest_parser.add_argument("--nowcast-start", default="2024-01-01")
    backtest_parser.add_argument("--nowcast-end", default="2025-10-31")
    backtest_parser.add_argument("--weeks-before", nargs="+", type=int)
    backtest_parser.add_argument("--override", action="append")
    
    args = parser.parse_args()
    if args.command == 'backtest':
        run_backtest_evaluation(config_name=args.config_name, model=args.model, train_start=args.train_start,
                                train_end=args.train_end, nowcast_start=args.nowcast_start, nowcast_end=args.nowcast_end,
                                weeks_before=args.weeks_before, config_dir=str(get_project_root() / "config"),
                                overrides=args.override)


if __name__ == "__main__":
    main()
