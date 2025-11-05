#!/usr/bin/env python3
"""
Script to check if migration 003_restart.sql was applied successfully to Supabase.
"""
import sys
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables (go up 3 levels from scripts/setup/)
project_root = Path(__file__).resolve().parent.parent.parent
env_paths = [
    project_root / '.env.local',
    project_root / '.env',
    Path.home() / '.env.local',
]

for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded environment from: {env_path}")
        break
else:
    print("⚠️  No .env file found")
    print("   Please set SUPABASE_URL and SUPABASE_KEY environment variables")

sys.path.insert(0, str(project_root))

try:
    from database.client import get_client
    
    print("\n" + "=" * 60)
    print("Checking Migration 003 Status")
    print("=" * 60)
    
    client = get_client()
    
    # All tables that should exist after migration 003
    required_tables = [
        'data_sources',
        'statistics_metadata',
        'statistics_items',
        'series',
        'data_vintages',
        'observations',
        'ingestion_jobs',
        'api_fetches',
        'model_configs',
        'model_block_assignments',
        'trained_models',
        'forecast_runs',
        'forecasts'
    ]
    
    print("\n📊 Checking Tables:")
    print("-" * 60)
    
    existing = []
    missing = []
    errors = []
    
    for table in required_tables:
        try:
            result = client.table(table).select('*', count='exact').limit(0).execute()
            count = result.count if hasattr(result, 'count') else 'N/A'
            existing.append(table)
            print(f"✅ {table:30} - EXISTS (rows: {count})")
        except Exception as e:
            error_msg = str(e)
            if 'not found' in error_msg.lower() or 'PGRST205' in error_msg:
                missing.append(table)
                print(f"❌ {table:30} - NOT FOUND")
            else:
                errors.append((table, str(e)))
                print(f"⚠️  {table:30} - ERROR: {str(e)[:50]}")
    
    print("-" * 60)
    
    # Check views
    print("\n📋 Checking Views:")
    print("-" * 60)
    
    required_views = [
        'active_statistics_by_source',
        'dfm_selected_statistics',
        'latest_forecasts_view',
        'model_training_history'
    ]
    
    existing_views = []
    missing_views = []
    
    for view in required_views:
        try:
            result = client.table(view).select('*', count='exact').limit(0).execute()
            existing_views.append(view)
            print(f"✅ {view:30} - EXISTS")
        except Exception as e:
            error_msg = str(e)
            if 'not found' in error_msg.lower() or 'PGRST205' in error_msg:
                missing_views.append(view)
                print(f"❌ {view:30} - NOT FOUND")
            else:
                print(f"⚠️  {view:30} - {str(e)[:50]}")
    
    print("-" * 60)
    
    # Summary
    print("\n" + "=" * 60)
    print("📈 Migration Status Summary")
    print("=" * 60)
    print(f"Tables:")
    print(f"  ✅ Existing: {len(existing)}/{len(required_tables)}")
    print(f"  ❌ Missing:  {len(missing)}/{len(required_tables)}")
    print(f"\nViews:")
    print(f"  ✅ Existing: {len(existing_views)}/{len(required_views)}")
    print(f"  ❌ Missing:  {len(missing_views)}/{len(required_views)}")
    
    if missing or missing_views:
        print("\n⚠️  Migration 003 appears NOT to be fully applied.")
        if missing:
            print(f"\n   Missing tables: {', '.join(missing[:5])}")
        if missing_views:
            print(f"   Missing views: {', '.join(missing_views)}")
        print("\n   To apply migration:")
        print("   1. Open Supabase Dashboard → SQL Editor")
        print("   2. Copy contents of migrations/003_restart.sql")
        print("   3. Paste and run the SQL")
    elif len(existing) == len(required_tables) and len(existing_views) == len(required_views):
        print("\n✅ Migration 003 appears to be applied successfully!")
        print("   All tables and views exist.")
    else:
        print("\n⚠️  Partial migration - some objects missing")
    
    print("=" * 60)
    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print("   Make sure you're in the virtual environment:")
    print("   source .venv/bin/activate")
    print("   pip install supabase python-dotenv")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
