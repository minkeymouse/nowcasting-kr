"""Setup and on-demand scripts.

This directory contains:
1. One-time setup scripts:
   - Collecting initial statistics metadata from APIs
   - Uploading initial metadata to the database

2. On-demand/periodic scripts (not for GitHub Actions):
   - Training DFM models (train_dfm.py)
   - Running forecasting/nowcasting (forecast_dfm.py)

For regular data ingestion (GitHub Actions), use scripts/ingest_data.py instead.
"""

