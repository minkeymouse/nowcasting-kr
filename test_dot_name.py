
import sys
from pathlib import Path

# Add dfm-python to path
sys.path.append(str(Path("dfm-python/src").resolve()))

import pandas as pd
import numpy as np
import torch
import logging
from dfm_python.models import DFM
from dfm_python import DFMTrainer, DFMDataModule

# Setup logging
logging.basicConfig(level=logging.INFO)

def test_dot_name():
    print("Testing DFM with dot in series name...")
    
    # Create dummy data
    dates = pd.date_range('2020-01-01', periods=100, freq='M')
    # Series with a dot in the name
    series_name = "KOIPALL.G"
    data = pd.DataFrame(np.random.randn(100, 1), index=dates, columns=[series_name])
    
    # Config dict
    config_dict = {
        "series": [
            {
                "series_id": series_name,
                "frequency": "m",
                "transformation": "lin",
                "blocks": [1] # Global block
            }
        ],
        "blocks": {
            "Block_Global": {
                "factors": 1,
                "clock": "m"
            }
        }
    }
    
    try:
        model = DFM(
            max_iter=10,
            threshold=1e-5
        )
        model.load_config(mapping=config_dict)
        print("Model initialized and config loaded successfully.")
        
        # Create DataModule
        dm = DFMDataModule(config=model.config, data=data)
        dm.setup()
        
        # Fit model
        print("Fitting model...")
        trainer = DFMTrainer(max_epochs=10)
        trainer.fit(model, dm)
        print("Model fitted successfully.")
        
        # Predict
        print("Predicting...")
        pred = model.predict(horizon=1)
        print("Prediction successful.")
        print(pred)
        
    except Exception as e:
        print(f"FAILED with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dot_name()

