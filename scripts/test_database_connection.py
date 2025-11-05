"""Test Supabase connection using database module functions."""

import sys
import os
from pathlib import Path

# Check if required packages are installed
try:
    import dotenv
    from dotenv import load_dotenv
except ImportError:
    print("⚠ python-dotenv not installed. Environment variables must be set manually.")
    load_dotenv = lambda path: None

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    print("❌ supabase package not installed.")
    print("   Install with: pip install supabase>=2.0.0")
    print("   Or use: uv sync (if using uv)")
    SUPABASE_AVAILABLE = False

# Load environment variables - check multiple locations
env_paths = [
    Path(__file__).parent.parent / '.env.local',  # Project root
    Path(__file__).parent.parent.parent / '.env.local',  # Parent directory
    Path.home() / 'Nowcasting' / '.env.local',  # Alternative location
]

env_loaded = False
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded environment variables from {env_path}")
        env_loaded = True
        break

if not env_loaded:
    print(f"⚠ No .env.local file found in expected locations:")
    for path in env_paths:
        print(f"   - {path}")
    print("   Environment variables must be set manually or in .env.local")

if not SUPABASE_AVAILABLE:
    print("\n❌ Cannot test connection - supabase package not installed")
    sys.exit(1)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from database import get_client, get_latest_vintage_id, list_series
    from database.operations import get_source_id, get_source_code
except ImportError as e:
    print(f"❌ Failed to import database module: {e}")
    sys.exit(1)


def test_connection():
    """Test Supabase connection using database module."""
    print("=" * 80)
    print("Testing Supabase Connection via Database Module")
    print("=" * 80)
    
    # Check environment variables
    print("\n1. Checking environment variables...")
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_KEY') or os.getenv('SUPABASE_SERVICE_ROLE_KEY') or os.getenv('SUPABASE_SECRET_KEY')
    
    if not url:
        print("❌ SUPABASE_URL not found in environment variables")
        print("   Set it with: export SUPABASE_URL='your-project-url'")
        print("   Or create .env.local file with: SUPABASE_URL=your-project-url")
        return False
    else:
        print(f"✓ SUPABASE_URL found: {url[:30]}...")
    
    if not key:
        print("❌ SUPABASE_KEY/SUPABASE_SERVICE_ROLE_KEY/SUPABASE_SECRET_KEY not found")
        print("   Set it with: export SUPABASE_KEY='your-service-role-key'")
        print("   Or create .env.local file with: SUPABASE_KEY=your-service-role-key")
        print("   Note: Use service_role key for admin operations, not anon key")
        return False
    else:
        print(f"✓ Key found: {key[:20]}...")
    
    # Test client creation
    print("\n2. Testing client creation...")
    try:
        client = get_client()
        print("✓ Client created successfully")
    except Exception as e:
        print(f"❌ Failed to create client: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test basic query - data sources
    print("\n3. Testing data sources query...")
    try:
        bok_id = get_source_id('BOK')
        print(f"✓ BOK source_id: {bok_id}")
        
        source_code = get_source_code(bok_id)
        print(f"✓ Source code for ID {bok_id}: {source_code}")
    except Exception as e:
        error_msg = str(e)
        # Handle both dict-style errors and string errors
        if hasattr(e, 'dict'):
            error_dict = e.dict()
            error_code = error_dict.get('code', '')
        elif isinstance(error_msg, dict):
            error_dict = error_msg
            error_code = error_dict.get('code', '')
        else:
            error_code = ''
        
        if "not found" in error_msg.lower() or "PGRST205" in error_code or "schema cache" in error_msg.lower():
            print(f"⚠ Data sources table does not exist yet")
            print(f"  This is expected if migrations haven't been run.")
            print(f"  Run migrations/001_initial_schema.sql to create tables")
            if error_code:
                print(f"  Error code: {error_code}")
        elif "not found in database" in error_msg.lower():
            print(f"⚠ Data source 'BOK' not found in database")
            print(f"  This is expected if initial data hasn't been inserted")
            print(f"  Tables exist but need initial data")
        else:
            print(f"❌ Query failed: {error_msg[:200]}")
            import traceback
            traceback.print_exc()
            return False
    
    # Test series query
    print("\n4. Testing series query...")
    try:
        series_list = list_series()
        print(f"✓ Found {len(series_list)} series")
        if series_list:
            print(f"  Sample series_id: {series_list[0].get('series_id', 'N/A')}")
            if len(series_list) > 5:
                print(f"  (Showing first 5 of {len(series_list)} total)")
    except Exception as e:
        error_msg = str(e)
        # Handle both dict-style errors and string errors
        if hasattr(e, 'dict'):
            error_dict = e.dict()
            error_code = error_dict.get('code', '')
        elif isinstance(error_msg, dict):
            error_dict = error_msg
            error_code = error_dict.get('code', '')
        else:
            error_code = ''
        
        if "relation" in error_msg.lower() or "does not exist" in error_msg.lower() or "PGRST205" in error_code or "schema cache" in error_msg.lower():
            print(f"⚠ Series table does not exist yet")
            print(f"  Run migrations/001_initial_schema.sql to create tables")
            if error_code:
                print(f"  Error code: {error_code}")
        else:
            print(f"❌ Query failed: {error_msg[:200]}")
            import traceback
            traceback.print_exc()
            return False
    
    # Test vintage query
    print("\n5. Testing vintage query...")
    try:
        latest_vintage_id = get_latest_vintage_id()
        if latest_vintage_id:
            print(f"✓ Latest vintage_id: {latest_vintage_id}")
        else:
            print("⚠ No vintages found (this is expected if no data has been ingested)")
    except Exception as e:
        error_msg = str(e)
        # Handle both dict-style errors and string errors
        if hasattr(e, 'dict'):
            error_dict = e.dict()
            error_code = error_dict.get('code', '')
        elif isinstance(error_msg, dict):
            error_dict = error_msg
            error_code = error_dict.get('code', '')
        else:
            error_code = ''
        
        if "relation" in error_msg.lower() or "does not exist" in error_msg.lower() or "PGRST205" in error_code or "schema cache" in error_msg.lower():
            print(f"⚠ Vintages table does not exist yet")
            print(f"  Run migrations/001_initial_schema.sql to create tables")
            if error_code:
                print(f"  Error code: {error_code}")
        else:
            print(f"❌ Query failed: {error_msg[:200]}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "=" * 80)
    print("✅ Connection test completed successfully!")
    print("=" * 80)
    return True


if __name__ == '__main__':
    try:
        success = test_connection()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

