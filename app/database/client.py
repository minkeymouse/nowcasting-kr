"""Supabase client initialization and connection management."""

import os
from typing import Optional
from supabase import create_client, Client
from supabase.client import ClientOptions


def get_supabase_client(
    url: Optional[str] = None,
    key: Optional[str] = None,
    options: Optional[ClientOptions] = None
) -> Client:
    """
    Create and return a Supabase client.
    
    Parameters
    ----------
    url : str, optional
        Supabase project URL. If None, reads from SUPABASE_URL environment variable.
    key : str, optional
        Supabase service role key. If None, reads from SUPABASE_SECRET_KEY environment variable.
    options : ClientOptions, optional
        Additional client options.
        
    Returns
    -------
    Client
        Supabase client instance
        
    Raises
    ------
    ValueError
        If URL or key is not provided and not in environment variables.
    """
    url = url or os.getenv('SUPABASE_URL')
    # Check multiple possible environment variable names for the Supabase key
    key = key or os.getenv('SUPABASE_SECRET_KEY')

    if not url:
        raise ValueError("Supabase URL must be provided or set in SUPABASE_URL environment variable")
    if not key:
        raise ValueError("Supabase key must be provided or set in SUPABASE_SECRET_KEY environment variable")
    
    return create_client(url, key, options=options)

# Global client instance (lazy initialization)
_client: Optional[Client] = None

def get_client(url: Optional[str] = None, key: Optional[str] = None) -> Client:
    """
    Get or create a global Supabase client instance.
    
    This function caches the client instance for reuse across the application.
    If the client hasn't been created yet, it will be initialized using the
    provided credentials or environment variables.
    
    Parameters
    ----------
    url : str, optional
        Supabase project URL. Only used if client hasn't been created yet.
    key : str, optional
        Supabase service role key. Only used if client hasn't been created yet.
        
    Returns
    -------
    Client
        Cached Supabase client instance
    """
    global _client
    if _client is None:
        _client = get_supabase_client(url=url, key=key)
    return _client


def reset_client():
    """Reset the global client instance (useful for testing)."""
    global _client
    _client = None

