"""Example script to estimate a Dynamic Factor Model (DFM)."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.nowcasting import load_config, load_data, dfm
from src.utils import summarize
import pandas as pd
import numpy as np
import pickle


def main():
    # User inputs
    vintage = '2016-06-29'
    country = 'US'
    sample_start = pd.Timestamp('2000-01-01')
    
    # Load model configuration
    config_file = Path(__file__).parent.parent.parent / 'matlab' / 'Spec_US_example.xls'
    config = load_config(config_file)
    
    # Load data
    data_file = Path(__file__).parent.parent.parent / 'data' / country / f'{vintage}.xls'
    X, Time, Z = load_data(data_file, config, sample_start=sample_start)
    
    # Summarize data
    summarize(X, Time, config, vintage)
    
    # Run DFM estimation
    threshold = 1e-4  # Set to 1e-5 for more robust estimates
    Res = dfm(X, config, threshold)
    
    # Save results
    output_file = Path(__file__).parent.parent.parent / 'ResDFM.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump({'Res': Res, 'Config': config}, f)
    print(f'\nResults saved to {output_file}')
    
    # Optional: Plot common factor (requires matplotlib)
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(Time, Res.Z[:, 0])
    # plt.title('Common Factor')
    # plt.show()


if __name__ == '__main__':
    main()

