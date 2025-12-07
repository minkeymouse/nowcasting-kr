"""Nowcasting module - generates JSON/CSV results only."""
from pathlib import Path
import sys
import argparse
import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pickle
import json

# Set up paths
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from src.utils import setup_cli_environment
setup_cli_environment()

from src.utils import (
    get_project_root,
    parse_experiment_config,
    extract_experiment_params,
    validate_experiment_config
)
try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def _load_model_for_inference(
    project_root: Path,
    target_series: str,
    model: str,
    supports_nowcast_manager: bool,
    train_start: str,
    train_end: str
) -> Tuple[Optional[Any], Any, Optional[Any]]:
    """Load trained model from checkpoint."""
    checkpoint_path = project_root / "checkpoint" / f"{target_series}_{model.lower()}" / "model.pkl"
    if not checkpoint_path.exists():
        comparisons_dir = project_root / "outputs" / "comparisons"
        if comparisons_dir.exists():
            for comparison_dir in comparisons_dir.glob(f"{target_series}_*"):
                model_path = comparison_dir / model.lower() / "model.pkl"
                if model_path.exists():
                    checkpoint_path = model_path
                    break
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Trained {model.upper()} model not found for {target_series}")
    
    with open(checkpoint_path, 'rb') as f:
        model_data = pickle.load(f)
    
    forecaster = model_data.get('forecaster')
    if forecaster is None:
        raise ValueError(f"Model file does not contain forecaster")
    
    if not supports_nowcast_manager:
        return None, forecaster, None
    
    if hasattr(forecaster, '_dfm_model'):
        dfm_model = forecaster._dfm_model
    elif hasattr(forecaster, '_ddfm_model'):
        dfm_model = forecaster._ddfm_model
    else:
        raise ValueError(f"Forecaster does not contain {model.upper()} model")
    
    if dfm_model is None:
        raise ValueError(f"{model.upper()} model is None")
    
    # Restore result if needed
    if hasattr(dfm_model, '_model') and dfm_model._model is not None:
        underlying_model = dfm_model._model
        if hasattr(underlying_model, '_result') and underlying_model._result is None:
            if hasattr(underlying_model, 'training_state') and underlying_model.training_state is not None:
                try:
                    underlying_model._result = underlying_model.get_result()
                except Exception:
                    pass
            if underlying_model._result is None:
                try:
                    result = dfm_model.get_result()
                    if result is not None:
                        underlying_model._result = result
                except Exception:
                    pass
            if underlying_model._result is None and 'result' in model_data:
                try:
                    underlying_model._result = model_data['result']
                except Exception:
                    pass
    
    # Ensure data_module exists
    if not hasattr(dfm_model, '_data_module') or dfm_model._data_module is None:
        from src.models import create_data_module_from_dataframe
        from dfm_python.lightning import DFMDataModule
        from src.preprocessing import create_transformer_from_config
        
        data_path_file = project_root / "data" / "data.csv"
        if not data_path_file.exists():
            data_path_file = project_root / "data" / "sample_data.csv"
        
        train_data = pd.read_csv(data_path_file, index_col=0, parse_dates=True)
        train_start_ts = pd.Timestamp(train_start)
        train_end_ts = pd.Timestamp(train_end)
        train_data_filtered = train_data[(train_data.index >= train_start_ts) & (train_data.index <= train_end_ts)]
        
        data_module = create_data_module_from_dataframe(
            model=dfm_model,
            data=train_data_filtered,
            dfm_data_module=DFMDataModule,
            create_transformer_func=create_transformer_from_config
        )
        dfm_model._data_module = data_module
    
    nowcast_manager = dfm_model.nowcast
    return nowcast_manager, forecaster, dfm_model


def _update_data_module_for_nowcasting(
    dfm_model: Any,
    nowcast_manager: Any,
    data_up_to_target: pd.DataFrame,
    target_month_end_ts: pd.Timestamp
) -> None:
    """Update data_module with data up to target month end."""
    if not hasattr(dfm_model, '_data_module') or dfm_model._data_module is None:
        raise RuntimeError("data_module not available")
    
    from src.preprocessing import resample_to_monthly
    from dfm_python.utils.time import TimeIndex
    
    data_monthly = resample_to_monthly(data_up_to_target)
    if len(data_monthly) == 0:
        raise ValueError(f"No data available up to {target_month_end_ts}")
    
    time_index = TimeIndex(data_monthly.index.to_pydatetime().tolist())
    existing_data_module = dfm_model._data_module
    existing_data_module.data = data_monthly
    existing_data_module.time_index = time_index
    dfm_model._data_module = existing_data_module
    nowcast_manager.data_module = existing_data_module


def run_backtest_evaluation(
    config_name: str,
    model: str,
    train_start: str = "1985-01-01",
    train_end: str = "2019-12-31",
    nowcast_start: str = "2024-01-01",
    nowcast_end: str = "2025-10-31",
    weeks_before: Optional[List[int]] = None,
    config_dir: Optional[str] = None,
    overrides: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Run backtest evaluation and generate JSON/CSV results."""
    project_root = get_project_root()
    
    if config_dir is None:
        config_dir = str(project_root / "config")
    
    if weeks_before is None:
        weeks_before = [4, 1]
    
    supports_nowcast_manager = model.lower() in ['dfm', 'ddfm']
    
    cfg = parse_experiment_config(config_name, config_dir, overrides)
    validate_experiment_config(cfg, require_target=True, require_models=False)
    
    params = extract_experiment_params(cfg)
    target_series = params['target_series']
    exp_cfg = params.get('exp_cfg', cfg)
    
    # Load release dates
    series_release_dates = {}
    if hasattr(exp_cfg, 'get'):
        series_ids_raw = exp_cfg.get('series', [])
    else:
        series_ids_raw = getattr(exp_cfg, 'series', [])
    
    if series_ids_raw:
        import yaml
        series_config_dir = project_root / "dfm-python" / "config" / "series"
        series_ids = OmegaConf.to_container(series_ids_raw, resolve=True) if hasattr(OmegaConf, 'to_container') else list(series_ids_raw)
        if not isinstance(series_ids, list):
            series_ids = []
        
        for series_id in series_ids:
            series_config_path = series_config_dir / f"{series_id}.yaml"
            if series_config_path.exists():
                try:
                    with open(series_config_path, 'r', encoding='utf-8') as f:
                        series_cfg = yaml.safe_load(f) or {}
                    release_date = series_cfg.get('release') or series_cfg.get('release_date')
                    if release_date is not None:
                        series_release_dates[series_id] = release_date
                except Exception:
                    pass
    
    # Load model
    nowcast_manager, forecaster, dfm_model = _load_model_for_inference(
        project_root, target_series, model, supports_nowcast_manager, train_start, train_end
    )
    
    # Generate target periods
    start_date = datetime.strptime(nowcast_start, "%Y-%m-%d")
    end_date = datetime.strptime(nowcast_end, "%Y-%m-%d")
    target_periods = []
    current = start_date.replace(day=1)
    while current <= end_date:
        if current.month == 12:
            last_day = current.replace(day=31)
        else:
            next_month = current.replace(month=current.month + 1, day=1)
            last_day = next_month - timedelta(days=1)
        target_periods.append(last_day)
        current += relativedelta(months=1)
    
    # Load data
    from src.preprocessing import resample_to_monthly
    data_path_file = project_root / "data" / "data.csv"
    if not data_path_file.exists():
        data_path_file = project_root / "data" / "sample_data.csv"
    full_data = pd.read_csv(data_path_file, index_col=0, parse_dates=True)
    full_data_monthly = resample_to_monthly(full_data)
    
    train_start_ts = pd.Timestamp(train_start)
    train_end_ts = pd.Timestamp(train_end)
    train_data_filtered = full_data[(full_data.index >= train_start_ts) & (full_data.index <= train_end_ts)]
    train_std = train_data_filtered[target_series].std() if target_series in train_data_filtered.columns else train_data_filtered.iloc[:, 0].std() if len(train_data_filtered.columns) > 0 else 1.0
    
    results_by_timepoint = {}
    train_end_date = datetime.strptime(train_end, "%Y-%m-%d")
    
    for weeks in weeks_before:
        monthly_results = []
        
        for target_month_end in target_periods:
            target_month_end_ts = pd.Timestamp(target_month_end) if not isinstance(target_month_end, pd.Timestamp) else target_month_end
            view_date = target_month_end_ts - timedelta(weeks=weeks)
            view_date_ts = pd.Timestamp(view_date)
            
            if view_date_ts <= pd.Timestamp(train_end_date):
                continue
            
            if len(full_data_monthly) == 0:
                continue
            
            try:
                if supports_nowcast_manager:
                    data_up_to_target = full_data[full_data.index <= target_month_end_ts]
                    try:
                        _update_data_module_for_nowcasting(dfm_model, nowcast_manager, data_up_to_target, target_month_end_ts)
                    except Exception:
                        continue
                    
                    target_period_for_nowcast = target_month_end_ts
                    if target_month_end_ts <= full_data_monthly.index.max():
                        available_dates = full_data_monthly.index[full_data_monthly.index <= target_month_end_ts]
                        if len(available_dates) > 0:
                            target_period_for_nowcast = available_dates.max().to_pydatetime()
                        elif len(full_data_monthly) > 0:
                            target_period_for_nowcast = full_data_monthly.index.max().to_pydatetime()
                        else:
                            continue
                    else:
                        target_period_for_nowcast = full_data_monthly.index.max().to_pydatetime()
                    
                    if hasattr(nowcast_manager, '_data_view_cache'):
                        nowcast_manager._data_view_cache.clear()
                    
                    try:
                        nowcast_result = nowcast_manager(
                            target_series=target_series,
                            view_date=view_date,
                            target_period=target_period_for_nowcast,
                            return_result=True
                        )
                        
                        if nowcast_result is None or not hasattr(nowcast_result, 'nowcast_value'):
                            continue
                        
                        forecast_value = nowcast_result.nowcast_value
                        if not np.isfinite(forecast_value):
                            continue
                    except Exception:
                        continue
                    
                    # Get actual value
                    actual_value = np.nan
                    if target_month_end_ts <= full_data_monthly.index.max():
                        available_dates = full_data_monthly.index[full_data_monthly.index <= target_month_end_ts]
                        if len(available_dates) > 0:
                            closest_date = available_dates.max()
                            if target_series in full_data_monthly.columns:
                                raw_value = full_data_monthly.loc[closest_date, target_series]
                            else:
                                raw_value = full_data_monthly.loc[closest_date].iloc[0]
                            if not pd.isna(raw_value):
                                actual_value = float(raw_value)
                
                else:
                    # ARIMA/VAR
                    train_data_filtered = full_data[full_data.index <= pd.Timestamp(view_date)].copy()
                    if len(train_data_filtered) == 0:
                        continue
                    
                    from src.preprocessing import resample_to_monthly
                    train_data_filtered = resample_to_monthly(train_data_filtered)
                    if len(train_data_filtered) == 0:
                        continue
                    
                    # Apply release date masking
                    if series_release_dates:
                        from src.nowcasting import calculate_release_date
                        view_date_dt = view_date if isinstance(view_date, datetime) else pd.Timestamp(view_date).to_pydatetime()
                        for series_id, release_offset in series_release_dates.items():
                            if series_id not in train_data_filtered.columns:
                                continue
                            for period_idx in train_data_filtered.index:
                                period_dt = period_idx.to_pydatetime() if isinstance(period_idx, pd.Timestamp) else period_idx
                                release_date_dt = calculate_release_date(release_offset, period_dt)
                                if release_date_dt > view_date_dt:
                                    train_data_filtered.loc[period_idx, series_id] = np.nan
                    
                    # Prepare data
                    is_multivariate = False
                    training_columns = None
                    if hasattr(forecaster, 'get_tag'):
                        try:
                            scitype_y = forecaster.get_tag('scitype:y', 'univariate')
                            is_multivariate = (scitype_y == 'multivariate')
                        except Exception:
                            pass
                    
                    if hasattr(forecaster, '_y'):
                        try:
                            if isinstance(forecaster._y, pd.DataFrame):
                                is_multivariate = len(forecaster._y.columns) > 1
                                training_columns = forecaster._y.columns.tolist()
                        except Exception:
                            pass
                    
                    if not is_multivariate:
                        is_multivariate = len(train_data_filtered.columns) > 1
                    
                    original_columns = None
                    if is_multivariate:
                        if training_columns is not None:
                            available_columns = [col for col in training_columns if col in train_data_filtered.columns]
                            original_columns = available_columns
                            if len(available_columns) < 2:
                                continue
                            fit_data = train_data_filtered[available_columns].copy()
                            fit_data = fit_data.reindex(columns=available_columns)
                        else:
                            fit_data = train_data_filtered.copy()
                            original_columns = list(fit_data.columns)
                        fit_data = fit_data.dropna(how='all')
                    else:
                        if target_series in train_data_filtered.columns:
                            fit_data = train_data_filtered[[target_series]].copy()
                        else:
                            fit_data = train_data_filtered[[train_data_filtered.columns[0]]].copy()
                        target_col_name = fit_data.columns[0]
                        if fit_data[target_col_name].isna().any():
                            fit_data = fit_data.ffill()
                        fit_data = fit_data.dropna()
                        if isinstance(fit_data, pd.Series):
                            fit_data = fit_data.to_frame()
                    
                    if len(fit_data) < 10:
                        continue
                    
                    # Predict
                    last_fit_month = fit_data.index.max()
                    if last_fit_month is None:
                        continue
                    
                    if isinstance(last_fit_month, pd.Timestamp):
                        last_fit_dt = last_fit_month.to_pydatetime()
                    else:
                        last_fit_dt = last_fit_month
                    target_dt = target_month_end_ts.to_pydatetime()
                    delta = relativedelta(target_dt, last_fit_dt)
                    months_ahead = delta.years * 12 + delta.months
                    if months_ahead < 0:
                        horizon_periods = 1
                    elif months_ahead == 0:
                        horizon_periods = 1
                    else:
                        horizon_periods = months_ahead
                    
                    try:
                        fh = np.asarray([horizon_periods])
                        y_pred = forecaster.predict(fh=fh)
                        
                        if model == 'var' and original_columns is not None and isinstance(y_pred, pd.DataFrame):
                            if len(y_pred.columns) == len(original_columns):
                                y_pred.columns = original_columns
                    except Exception:
                        continue
                    
                    # Extract forecast value
                    if y_pred is None:
                        forecast_value = np.nan
                    elif isinstance(y_pred, pd.DataFrame):
                        if len(y_pred) == 0:
                            forecast_value = np.nan
                        elif target_series in y_pred.columns:
                            raw_val = y_pred[target_series].iloc[0]
                            forecast_value = float(raw_val) if not pd.isna(raw_val) else np.nan
                        elif len(y_pred.columns) > 0:
                            raw_val = y_pred.iloc[0, 0]
                            forecast_value = float(raw_val) if not pd.isna(raw_val) else np.nan
                        else:
                            forecast_value = np.nan
                    elif isinstance(y_pred, pd.Series):
                        if len(y_pred) > 0:
                            raw_val = y_pred.iloc[0]
                            forecast_value = float(raw_val) if not pd.isna(raw_val) else np.nan
                        else:
                            forecast_value = np.nan
                    else:
                        try:
                            raw_val = y_pred[0] if hasattr(y_pred, '__getitem__') else y_pred
                            if pd.isna(raw_val) if hasattr(pd, 'isna') else (raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val))):
                                forecast_value = np.nan
                            else:
                                forecast_value = float(raw_val)
                        except Exception:
                            forecast_value = np.nan
                    
                    # Get actual value
                    actual_value = np.nan
                    if len(full_data_monthly) > 0 and target_month_end_ts <= full_data_monthly.index.max():
                        available_dates = full_data_monthly.index[full_data_monthly.index <= target_month_end_ts]
                        if len(available_dates) > 0:
                            closest_date = available_dates.max()
                            if target_series in full_data_monthly.columns:
                                raw_value = full_data_monthly.loc[closest_date, target_series]
                            else:
                                raw_value = full_data_monthly.loc[closest_date].iloc[0]
                            if not pd.isna(raw_value):
                                actual_value = float(raw_value)
                
                if np.isnan(forecast_value) or np.isnan(actual_value):
                    continue
                
                error = forecast_value - actual_value
                if train_std and train_std > 0:
                    sMSE = (error ** 2) / (train_std ** 2)
                    sMAE = abs(error) / train_std
                else:
                    sMSE = error ** 2
                    sMAE = abs(error)
                
                monthly_results.append({
                    'month': target_month_end_ts.strftime('%Y-%m'),
                    'view_date': view_date.strftime('%Y-%m-%d'),
                    'weeks_before': weeks,
                    'forecast_value': float(forecast_value),
                    'actual_value': float(actual_value),
                    'error': float(error),
                    'abs_error': float(abs(error)),
                    'squared_error': float(error ** 2),
                    'sMSE': float(sMSE),
                    'sMAE': float(sMAE)
                })
                
            except Exception:
                continue
        
        if len(monthly_results) > 0:
            results_by_timepoint[f"{weeks}weeks"] = {
                'weeks_before': weeks,
                'monthly_results': monthly_results,
                'overall_sMAE': float(np.nanmean([r['sMAE'] for r in monthly_results])),
                'overall_sMSE': float(np.nanmean([r['sMSE'] for r in monthly_results])),
                'overall_mae': float(np.nanmean([r['abs_error'] for r in monthly_results])),
                'overall_rmse': float(np.sqrt(np.nanmean([r['squared_error'] for r in monthly_results]))),
                'n_months': len(monthly_results)
            }
    
    # Save results
    if len(results_by_timepoint) > 0:
        results = {
            'target_series': target_series,
            'model': model.upper(),
            'train_period': f"{train_start} to {train_end}",
            'nowcast_period': f"{nowcast_start} to {nowcast_end}",
            'weeks_before': weeks_before,
            'results_by_timepoint': results_by_timepoint,
            'horizon': 1
        }
    else:
        results = {
            'target_series': target_series,
            'model': model.upper(),
            'status': 'no_results',
            'error': 'No valid results generated for any time point'
        }
    
    output_dir = project_root / "outputs" / "backtest"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{target_series}_{model}_backtest.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {output_file}")
    return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Run nowcasting backtest")
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    backtest_parser = subparsers.add_parser('backtest', help='Run nowcasting backtest')
    backtest_parser.add_argument("--config-name", required=True)
    backtest_parser.add_argument("--model", required=True, choices=['arima', 'var', 'dfm', 'ddfm'])
    backtest_parser.add_argument("--train-start", default="1985-01-01")
    backtest_parser.add_argument("--train-end", default="2019-12-31")
    backtest_parser.add_argument("--nowcast-start", default="2024-01-01")
    backtest_parser.add_argument("--nowcast-end", default="2025-10-31")
    backtest_parser.add_argument("--weeks-before", nargs="+", type=int)
    backtest_parser.add_argument("--override", action="append")
    
    args = parser.parse_args()
    config_path = str(get_project_root() / "config")
    
    if args.command == 'backtest':
        run_backtest_evaluation(
            config_name=args.config_name,
            model=args.model,
            train_start=args.train_start,
            train_end=args.train_end,
            nowcast_start=args.nowcast_start,
            nowcast_end=args.nowcast_end,
            weeks_before=args.weeks_before,
            config_dir=config_path,
            overrides=args.override
        )


if __name__ == "__main__":
    main()
