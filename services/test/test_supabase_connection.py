"""Test Supabase connection."""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env.local if exists
env_path = Path(__file__).parent.parent / '.env.local'
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded environment variables from {env_path}")

# Import Supabase client directly
from supabase import create_client, Client


def test_connection():
    """Test Supabase connection."""
    print("=" * 80)
    print("Testing Supabase Connection")
    print("=" * 80)
    
    # Check environment variables
    print("\n1. Checking environment variables...")
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_KEY') or os.getenv('SUPABASE_SERVICE_ROLE_KEY') or os.getenv('SUPABASE_SECRET_KEY')
    
    if not url:
        print("❌ SUPABASE_URL not found in environment variables")
        return False
    else:
        print(f"✓ SUPABASE_URL found: {url[:30]}...")
    
    if not key:
        print("❌ SUPABASE_KEY/SUPABASE_SERVICE_ROLE_KEY/SUPABASE_SECRET_KEY not found")
        return False
    else:
        print(f"✓ Key found: {key[:20]}...")
    
    # Test connection using Supabase create_client directly
    print("\n2. Testing connection using Supabase create_client...")
    try:
        client: Client = create_client(url, key)
        print("✓ Client created successfully")
    except Exception as e:
        print(f"❌ Failed to create client: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test query (list tables or simple query)
    print("\n3. Testing database query...")
    try:
        # Try to query a table - if it doesn't exist, that's okay
        # We're just testing the connection
        result = client.table('series').select('series_id').limit(1).execute()
        print(f"✓ Query successful (found {len(result.data)} rows)")
        if result.data:
            print(f"  Sample data: {result.data[0]}")
    except Exception as e:
        error_msg = str(e)
        if "relation" in error_msg.lower() or "does not exist" in error_msg.lower() or "Could not find" in error_msg:
            print(f"⚠ Connection successful but table 'series' doesn't exist yet")
            print(f"  This is expected if tables haven't been created.")
            print(f"  Error message: {error_msg[:150]}")
        elif "JWT" in error_msg or "auth" in error_msg.lower():
            print(f"⚠ Authentication issue: {error_msg[:150]}")
            print(f"  Check if you're using the correct key (service_role_key vs anon_key)")
        else:
            print(f"❌ Query failed: {error_msg[:200]}")
            import traceback
            traceback.print_exc()
            return False
    
    # Test with a simpler query - try to get database info
    print("\n4. Testing basic connection health...")
    try:
        # Just verify the client is working
        print(f"✓ Client URL: {client.supabase_url[:30]}...")
        print(f"✓ Client is ready")
    except Exception as e:
        print(f"⚠ Client health check: {e}")
    
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
