"""Pydantic Settings for API data ingestion and database configuration."""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import json


class DatabaseConfig(BaseSettings):
    """Database connection settings."""
    model_config = SettingsConfigDict(
        env_prefix="SUPABASE_",
        case_sensitive=False
    )
    
    url: str = Field(..., description="Supabase project URL")
    key: str = Field(..., description="Supabase service role key")
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v.startswith(('http://', 'https://')):
            raise ValueError("Database URL must start with http:// or https://")
        return v


class BOKAPIConfig(BaseSettings):
    """BOK API configuration."""
    model_config = SettingsConfigDict(
        env_prefix="BOK_API_",
        case_sensitive=False
    )
    
    base_url: str = Field(
        default="https://ecos.bok.or.kr/api/",
        description="BOK API base URL"
    )
    auth_key: str = Field(..., description="BOK API authentication key")
    service_name: str = Field(
        default="StatisticSearch",
        description="API service name"
    )
    request_type: Literal["xml", "json"] = Field(
        default="json",
        description="Response format"
    )
    language: Literal["kr", "en"] = Field(
        default="kr",
        description="Response language"
    )
    default_start_count: int = Field(
        default=1,
        description="Default start count for pagination"
    )
    default_end_count: int = Field(
        default=1000,
        description="Default end count for pagination (max per request)"
    )
    timeout: int = Field(
        default=60,
        description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts"
    )
    retry_backoff: float = Field(
        default=2.0,
        description="Exponential backoff multiplier"
    )
    rate_limit_per_minute: Optional[int] = Field(
        default=None,
        description="Rate limit (requests per minute)"
    )


class KOSISAPIConfig(BaseSettings):
    """KOSIS API configuration."""
    model_config = SettingsConfigDict(
        env_prefix="KOSIS_API_",
        case_sensitive=False
    )
    
    base_url: str = Field(
        default="https://kosis.kr/openapi/",
        description="KOSIS API base URL"
    )
    api_key: str = Field(..., description="KOSIS API authentication key")
    default_view_code: str = Field(
        default="MT_ZTITLE",
        description="Default view code (MT_ZTITLE: 국내통계 주제별)"
    )
    timeout: int = Field(
        default=60,
        description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts"
    )
    retry_backoff: float = Field(
        default=2.0,
        description="Exponential backoff multiplier"
    )
    rate_limit_per_minute: Optional[int] = Field(
        default=None,
        description="Rate limit (requests per minute)"
    )


class DataSourceConfig(BaseModel):
    """Configuration for a single data source."""
    name: str = Field(..., description="Data source name")
    api_code: str = Field(..., description="Source-specific statistic code (e.g., BOK: 200Y101, KOSIS: orgId_tblId)")
    frequency: Literal["A", "S", "Q", "M", "SM", "D"] = Field(
        ...,
        description="Data frequency"
    )
    start_date: str = Field(
        ...,
        description="Start date in format matching frequency (e.g., '2020', '2020Q1', '202001')"
    )
    end_date: Optional[str] = Field(
        default=None,
        description="End date (None = fetch until latest)"
    )
    item_code1: Optional[str] = Field(
        default=None,
        description="Statistic item code 1"
    )
    item_code2: Optional[str] = Field(
        default=None,
        description="Statistic item code 2"
    )
    item_code3: Optional[str] = Field(
        default=None,
        description="Statistic item code 3"
    )
    item_code4: Optional[str] = Field(
        default=None,
        description="Statistic item code 4"
    )
    enabled: bool = Field(
        default=True,
        description="Whether this data source is enabled"
    )


class AppSettings(BaseSettings):
    """Main application settings."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )
    
    # Database settings
    supabase_url: str = Field(description="Supabase project URL")
    supabase_key: str = Field(description="Supabase service role key")
    
    # BOK API settings
    bok_api_base_url: str = Field(
        default="https://ecos.bok.or.kr/api/",
        description="BOK API base URL"
    )
    bok_api_auth_key: str = Field(description="BOK API authentication key")
    bok_api_service_name: str = Field(
        default="StatisticSearch",
        description="API service name"
    )
    bok_api_request_type: Literal["xml", "json"] = Field(
        default="json",
        description="Response format"
    )
    bok_api_language: Literal["kr", "en"] = Field(
        default="kr",
        description="Response language"
    )
    bok_api_timeout: int = Field(
        default=60,
        description="Request timeout in seconds"
    )
    bok_api_max_retries: int = Field(
        default=3,
        description="Maximum retry attempts"
    )
    bok_api_retry_backoff: float = Field(
        default=2.0,
        description="Exponential backoff multiplier"
    )
    
    # KOSIS API settings
    kosis_api_base_url: str = Field(
        default="https://kosis.kr/openapi/",
        description="KOSIS API base URL"
    )
    kosis_api_key: Optional[str] = Field(
        default=None,
        description="KOSIS API authentication key"
    )
    kosis_api_default_view_code: str = Field(
        default="MT_ZTITLE",
        description="Default view code"
    )
    kosis_api_timeout: int = Field(
        default=60,
        description="Request timeout in seconds"
    )
    kosis_api_max_retries: int = Field(
        default=3,
        description="Maximum retry attempts"
    )
    kosis_api_retry_backoff: float = Field(
        default=2.0,
        description="Exponential backoff multiplier"
    )
    
    # API sources configuration file path
    api_sources_file: Path = Field(
        default=Path("config/api_sources.json"),
        description="Path to API sources configuration file"
    )
    
    @classmethod
    def load(cls) -> "AppSettings":
        """Load settings from environment variables."""
        # BaseSettings automatically loads from environment variables
        # Type checker may complain but runtime is fine
        return cls()  # type: ignore
    
    @property
    def database(self) -> DatabaseConfig:
        """Get database configuration."""
        return DatabaseConfig(
            url=self.supabase_url,
            key=self.supabase_key
        )
    
    @property
    def bok_api_config(self) -> BOKAPIConfig:
        """Get BOK API configuration."""
        return BOKAPIConfig(
            base_url=self.bok_api_base_url,
            auth_key=self.bok_api_auth_key,
            service_name=self.bok_api_service_name,
            request_type=self.bok_api_request_type,
            language=self.bok_api_language,
            timeout=self.bok_api_timeout,
            max_retries=self.bok_api_max_retries,
            retry_backoff=self.bok_api_retry_backoff
        )
    
    @property
    def kosis_api_config(self) -> Optional[KOSISAPIConfig]:
        """Get KOSIS API configuration."""
        if not self.kosis_api_key:
            return None
        return KOSISAPIConfig(
            base_url=self.kosis_api_base_url,
            api_key=self.kosis_api_key,
            default_view_code=self.kosis_api_default_view_code,
            timeout=self.kosis_api_timeout,
            max_retries=self.kosis_api_max_retries,
            retry_backoff=self.kosis_api_retry_backoff
        )
    
    @classmethod
    def load_api_sources(cls, sources_file: Optional[Path] = None) -> List[DataSourceConfig]:
        """
        Load API sources configuration from JSON file.
        
        Parameters
        ----------
        sources_file : Path, optional
            Path to JSON file. If None, uses default from settings.
            
        Returns
        -------
        List[DataSourceConfig]
            List of data source configurations
        """
        if sources_file is None:
            settings = cls.load()
            sources_file = settings.api_sources_file
        
        if not sources_file.exists():
            return []
        
        with open(sources_file, 'r', encoding='utf-8') as f:
            sources_data = json.load(f)
        
        return [DataSourceConfig(**source) for source in sources_data]


# Frequency mapping for date format validation
FREQUENCY_FORMATS = {
    "A": "YYYY",      # Annual: 2020
    "S": "YYYYSS",    # Semi-annual: 2020S1, 2020S2
    "Q": "YYYYQN",    # Quarterly: 2020Q1
    "M": "YYYYMM",    # Monthly: 202001
    "SM": "YYYYMMDD", # Semi-monthly: 20200101
    "D": "YYYYMMDD"   # Daily: 20200101
}


# BOK API error codes mapping (BOK-specific)
# This mapping is used by BOKAPIClient for error handling
BOK_ERROR_CODES = {
    "정보-100": "Invalid authentication key",
    "정보-200": "No data available",
    "에러-100": "Missing required parameters",
    "에러-101": "Invalid date format for frequency",
    "에러-200": "Invalid file type",
    "에러-300": "Missing count parameters",
    "에러-301": "Invalid count parameter type",
    "에러-400": "Query timeout - reduce date range",
    "에러-500": "Server error - service not found",
    "에러-600": "Database connection error",
    "에러-601": "SQL error",
    "에러-602": "Rate limit exceeded - too many requests"
}

