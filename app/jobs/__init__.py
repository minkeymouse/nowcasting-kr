"""Application jobs for GitHub Actions workflows.

This module contains the three main jobs:
1. ingest - Ingest data from APIs and update database
2. train - Train DFM model and save weights to storage
3. nowcast - Generate nowcasts/forecasts and update database
"""

__all__ = ['ingest', 'train', 'nowcast']

