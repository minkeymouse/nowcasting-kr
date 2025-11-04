"""Example script to produce a nowcast using estimated DFM parameters."""

import sys
from pathlib import Path
import pickle

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.nowcasting import load_config, load_data, dfm, DFMResult, update_nowcast
import pandas as pd


def main():
    # User inputs
    series = 'GDPC1'  # Nowcasting real GDP
    period = '2016q4'  # Forecasting target quarter
    
    # Load model configuration
    config_file = Path(__file__).parent.parent.parent / 'matlab' / 'Spec_US_example.xls'
    config = load_config(config_file)
    
    # Load DFM estimation results (matches MATLAB example_Nowcast.m behavior)
    res_file = Path(__file__).parent.parent.parent / 'ResDFM.pkl'
    try:
        with open(res_file, 'rb') as f:
            data = pickle.load(f)
            # Check config consistency (like MATLAB: if isequal(Res.Config,Config))
            # Handle both legacy 'Spec' and new 'Config' keys
            saved_config = data.get('Config', data.get('Spec'))
            if saved_config and 'Res' in data:
                if hasattr(saved_config, 'SeriesID') and saved_config.SeriesID != config.SeriesID:
                    # Config differs - re-estimate (MATLAB behavior)
                    print('Warning: Configuration mismatch. Re-estimating...')
                    raise FileNotFoundError
                Res = data  # Pass full dict to update_nowcast for config check
            else:
                Res = data.get('Res', data)  # Fallback to Res if no Config/Spec key
    except FileNotFoundError:
        # Re-estimate if file not found or config mismatch
        print('Estimating DFM model...')
        vintage = '2016-12-23'
        data_file = Path(__file__).parent.parent.parent / 'data' / 'US' / f'{vintage}.xls'
        X, Time, Z = load_data(data_file, config)
        threshold = 1e-4
        Res = dfm(X, config, threshold)
        with open(res_file, 'wb') as f:
            pickle.dump({'Res': Res, 'Config': config}, f)
    
    # Nowcast update from week of December 7 to week of December 16, 2016
    vintage_old = '2016-12-16'
    vintage_new = '2016-12-23'
    
    datafile_old = Path(__file__).parent.parent.parent / 'data' / 'US' / f'{vintage_old}.xls'
    datafile_new = Path(__file__).parent.parent.parent / 'data' / 'US' / f'{vintage_new}.xls'
    
    # Load datasets for each vintage
    X_old, Time_old, _ = load_data(datafile_old, config)
    X_new, Time, _ = load_data(datafile_new, config)
    
    # Update nowcast
    update_nowcast(X_old, X_new, Time, config, Res, series, period,
                   vintage_old, vintage_new)


if __name__ == '__main__':
    main()

