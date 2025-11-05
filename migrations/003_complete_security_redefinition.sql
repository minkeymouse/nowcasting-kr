-- ============================================================================
-- Migration: 003_complete_security_redefinition.sql
-- Purpose: Complete redefinition of all database objects with proper security
-- 
-- This migration serves as a NEW STARTING POINT for security:
-- - Drops all existing objects (tables, views, functions, policies)
-- - Recreates everything with proper security from the start
-- - All tables have RLS enabled immediately
-- - All views use security_invoker = true
-- - All policies are separated (SELECT, INSERT, UPDATE, DELETE)
-- - Function has search_path = ''
--
-- IMPORTANT: This assumes no data exists yet. If data exists, this will
-- delete everything!
-- ============================================================================

-- ============================================================================
-- PART 1: DROP ALL EXISTING OBJECTS
-- ============================================================================

-- Drop all views first (they depend on tables)
DROP VIEW IF EXISTS active_statistics_by_source CASCADE;
DROP VIEW IF EXISTS dfm_selected_statistics CASCADE;
DROP VIEW IF EXISTS latest_forecasts_view CASCADE;
DROP VIEW IF EXISTS model_training_history CASCADE;

-- Drop all tables (CASCADE handles dependencies)
DROP TABLE IF EXISTS forecasts CASCADE;
DROP TABLE IF EXISTS forecast_runs CASCADE;
DROP TABLE IF EXISTS trained_models CASCADE;
DROP TABLE IF EXISTS model_block_assignments CASCADE;
DROP TABLE IF EXISTS model_configs CASCADE;
DROP TABLE IF EXISTS observations CASCADE;
DROP TABLE IF EXISTS data_vintages CASCADE;
DROP TABLE IF EXISTS series CASCADE;
DROP TABLE IF EXISTS ingestion_jobs CASCADE;
DROP TABLE IF EXISTS api_fetches CASCADE;
DROP TABLE IF EXISTS statistics_items CASCADE;
DROP TABLE IF EXISTS statistics_metadata CASCADE;
DROP TABLE IF EXISTS data_sources CASCADE;

-- Drop function
DROP FUNCTION IF EXISTS update_updated_at_column() CASCADE;

-- ============================================================================
-- PART 2: CREATE ALL TABLES WITH RLS ENABLED FROM START
-- ============================================================================
-- All tables will have RLS enabled immediately after creation
-- This ensures security is enforced from the beginning

-- ============================================================================
-- Migration: 001_initial_schema.sql
-- Purpose: Initial database schema for DFM nowcasting system
-- 
-- Note: Initial series specification is defined in migrations/001_initial_spec.csv
-- This CSV file contains the list of series, their transformations, and block assignments
-- that will be used for DFM initialization. The initialization script reads from this CSV
-- to fetch data and populate the database.
-- ============================================================================

-- ============================================================================
-- 1. Data Sources Table
-- ============================================================================
-- Stores metadata about data sources (BOK, KOSIS, etc.)
CREATE TABLE data_sources (
    id SERIAL PRIMARY KEY,
    source_code VARCHAR(20) NOT NULL UNIQUE,  -- BOK, KOSIS, etc.
    source_name VARCHAR(100) NOT NULL,  -- 한국은행, 통계청 등
    source_name_eng VARCHAR(100),  -- Bank of Korea, Statistics Korea
    api_base_url VARCHAR(500),  -- API base URL
    api_documentation_url VARCHAR(500),  -- API documentation URL
    is_active BOOLEAN DEFAULT TRUE,
    description TEXT,
    metadata JSONB,  -- Flexible metadata storage
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT idx_source_code UNIQUE (source_code)
);

-- Enable RLS immediately after table creation
ALTER TABLE data_sources ENABLE ROW LEVEL SECURITY;


COMMENT ON TABLE data_sources IS 'Data source metadata (BOK, KOSIS, etc.)';
COMMENT ON COLUMN data_sources.source_code IS 'Source identifier: BOK, KOSIS, etc.';
COMMENT ON COLUMN data_sources.metadata IS 'Flexible JSON storage for source-specific metadata';

-- ============================================================================
-- 2. Statistics Metadata Table (Unified)
-- ============================================================================
-- Unified table for all statistics metadata from different sources
CREATE TABLE statistics_metadata (
    id SERIAL PRIMARY KEY,
    
    -- Source reference
    source_id INTEGER NOT NULL REFERENCES data_sources(id) ON DELETE CASCADE,
    
    -- Source-specific identifiers
    source_stat_code VARCHAR(100) NOT NULL,  -- BOK: stat_code, KOSIS: stat_code, etc.
    source_stat_name VARCHAR(500),  -- Original name from source
    source_stat_name_eng VARCHAR(500),  -- English name if available
    
    -- Standardized fields
    cycle VARCHAR(10),  -- Frequency: A, S, Q, M, SM, D (BOK), Y, H, Q, M, D (KOSIS)
    frequency_code VARCHAR(10),  -- Normalized: annual, semi_annual, quarterly, monthly, daily
    
    -- Source-specific metadata
    org_name VARCHAR(200),  -- Organization name
    is_searchable BOOLEAN DEFAULT FALSE,  -- Whether data can be fetched via API
    
    -- Hierarchy (for categorized statistics)
    parent_stat_code VARCHAR(100),  -- Parent statistic code
    parent_item_code VARCHAR(100),  -- Parent item code
    hierarchy_level INTEGER,  -- Depth in hierarchy
    
    -- Source-specific raw metadata (JSONB for flexibility)
    source_metadata JSONB,  -- Store source-specific fields (e.g., p_cycle, p_stat_code for BOK)
    
    -- DFM Selection
    is_dfm_selected BOOLEAN DEFAULT FALSE,
    dfm_priority INTEGER,  -- Priority/rank for DFM (1-50, NULL if not selected)
    dfm_selected_at TIMESTAMP,
    dfm_selection_reason TEXT,  -- Why this statistic was selected
    
    -- Data Collection Status
    is_active BOOLEAN DEFAULT TRUE,  -- Whether to collect data for this statistic
    last_data_fetch_date DATE,  -- Last successful data fetch
    last_data_fetch_status VARCHAR(20),  -- success, failed, partial
    data_start_date DATE,  -- Earliest available data date
    data_end_date DATE,  -- Latest available data date
    total_observations INTEGER,  -- Total number of observations collected
    last_observation_date DATE,  -- Date of last observation
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT unique_source_stat_code UNIQUE (source_id, source_stat_code)
);

-- Enable RLS immediately after table creation
ALTER TABLE statistics_metadata ENABLE ROW LEVEL SECURITY;


-- Indexes for statistics_metadata
CREATE INDEX idx_statistics_metadata_source_id ON statistics_metadata(source_id);
CREATE INDEX idx_statistics_metadata_source_stat_code ON statistics_metadata(source_stat_code);
CREATE INDEX idx_statistics_metadata_cycle ON statistics_metadata(cycle);
CREATE INDEX idx_statistics_metadata_frequency_code ON statistics_metadata(frequency_code);
CREATE INDEX idx_statistics_metadata_dfm_selected ON statistics_metadata(is_dfm_selected) WHERE is_dfm_selected = TRUE;
CREATE INDEX idx_statistics_metadata_active ON statistics_metadata(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_statistics_metadata_searchable ON statistics_metadata(is_searchable) WHERE is_searchable = TRUE;

-- JSONB indexes for source_metadata queries
CREATE INDEX idx_statistics_metadata_source_metadata ON statistics_metadata USING GIN (source_metadata);

COMMENT ON TABLE statistics_metadata IS 'Unified statistics metadata from all sources (BOK, KOSIS, etc.)';
COMMENT ON COLUMN statistics_metadata.source_stat_code IS 'Source-specific statistic code (e.g., BOK stat_code, KOSIS stat_code)';
COMMENT ON COLUMN statistics_metadata.frequency_code IS 'Normalized frequency: annual, semi_annual, quarterly, monthly, daily';
COMMENT ON COLUMN statistics_metadata.source_metadata IS 'Source-specific fields stored as JSON (e.g., BOK p_cycle, p_stat_code)';
COMMENT ON COLUMN statistics_metadata.is_dfm_selected IS 'Whether this statistic is selected for DFM nowcasting';
COMMENT ON COLUMN statistics_metadata.dfm_priority IS 'Priority rank for DFM (1-50)';

-- ============================================================================
-- 3. Statistics Items Table
-- ============================================================================
-- Stores available items for each statistic from StatisticItemList API
CREATE TABLE statistics_items (
    id SERIAL PRIMARY KEY,
    
    -- Reference to statistics metadata
    statistics_metadata_id INTEGER NOT NULL REFERENCES statistics_metadata(id) ON DELETE CASCADE,
    
    -- Item identification
    item_code VARCHAR(50) NOT NULL,  -- Item code (e.g., "1101", "*AA")
    item_name VARCHAR(500),  -- Item name in Korean
    item_name_eng VARCHAR(500),  -- Item name in English
    
    -- Hierarchy
    parent_item_code VARCHAR(50),  -- Parent item code for hierarchical structure
    parent_item_name VARCHAR(500),  -- Parent item name
    
    -- Grouping
    grp_code VARCHAR(50),  -- Group code (e.g., "Group1")
    grp_name VARCHAR(200),  -- Group name (e.g., "계정항목")
    
    -- Frequency and date range
    cycle VARCHAR(10) NOT NULL,  -- Frequency: A, S, Q, M, SM, D
    start_time VARCHAR(20),  -- Start time as string (e.g., "1960Q1", "202401")
    end_time VARCHAR(20),  -- End time as string (e.g., "2024Q4", "202412")
    
    -- Data information
    data_count INTEGER,  -- Number of available data points
    unit_name VARCHAR(50),  -- Unit name (e.g., "십억원", "2020=100")
    weight VARCHAR(50),  -- Weight value if applicable
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,  -- Whether this item is active for data collection
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT unique_stat_item_cycle UNIQUE (statistics_metadata_id, item_code, cycle)
);

-- Enable RLS immediately after table creation
ALTER TABLE statistics_items ENABLE ROW LEVEL SECURITY;


-- Indexes for statistics_items
CREATE INDEX idx_statistics_items_metadata_id ON statistics_items(statistics_metadata_id);
CREATE INDEX idx_statistics_items_item_code ON statistics_items(item_code);
CREATE INDEX idx_statistics_items_cycle ON statistics_items(cycle);
CREATE INDEX idx_statistics_items_active ON statistics_items(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_statistics_items_parent ON statistics_items(parent_item_code) WHERE parent_item_code IS NOT NULL;

COMMENT ON TABLE statistics_items IS 'Available items for each statistic from StatisticItemList API';
COMMENT ON COLUMN statistics_items.item_code IS 'Item code (e.g., "1101" for GDP sectors, "*AA" for total index)';
COMMENT ON COLUMN statistics_items.cycle IS 'Frequency for which this item is available (A, S, Q, M, SM, D)';
COMMENT ON COLUMN statistics_items.start_time IS 'Start time as string matching cycle format (e.g., "2024Q1" for quarterly)';
COMMENT ON COLUMN statistics_items.end_time IS 'End time as string matching cycle format';

-- ============================================================================
-- 4. Series Table (for actual time-series data)
-- ============================================================================
-- Links to statistics_metadata and stores additional series metadata
CREATE TABLE series (
    series_id VARCHAR(100) PRIMARY KEY,  -- Unique identifier (e.g., BOK_200Y101, KOSIS_101Y001)
    series_name VARCHAR(500) NOT NULL,
    
    -- Reference to statistics metadata
    statistics_metadata_id INTEGER REFERENCES statistics_metadata(id) ON DELETE SET NULL,
    
    -- Standardized fields
    frequency VARCHAR(10) NOT NULL,  -- d, w, m, q, sa, a (for DFM compatibility)
    units VARCHAR(50),  -- Original units (KRW_BIL, PCT, etc.)
    transformation VARCHAR(50),  -- Transformation code (log, diff, etc.)
    category VARCHAR(100),  -- Category classification
    
    -- Source information
    api_source VARCHAR(20) NOT NULL,  -- BOK, KOSIS, MANUAL
    api_code VARCHAR(100),  -- Source-specific code (matches source_stat_code)
    
    -- Item code (for item-based series)
    item_code VARCHAR(50),  -- Item code for this series (used when series represents a specific item from a statistic)
    
    -- Additional metadata
    description TEXT,
    metadata JSONB,  -- Additional flexible metadata
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    last_updated TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT fk_statistics_metadata FOREIGN KEY (statistics_metadata_id) 
        REFERENCES statistics_metadata(id) ON DELETE SET NULL
);

-- Enable RLS immediately after table creation
ALTER TABLE series ENABLE ROW LEVEL SECURITY;


CREATE INDEX idx_series_statistics_metadata_id ON series(statistics_metadata_id);
CREATE INDEX idx_series_api_source ON series(api_source);
CREATE INDEX idx_series_api_code ON series(api_code);
CREATE INDEX idx_series_frequency ON series(frequency);
CREATE INDEX idx_series_active ON series(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_series_item_code ON series(item_code) WHERE item_code IS NOT NULL;

COMMENT ON TABLE series IS 'Time-series metadata for actual data collection';
COMMENT ON COLUMN series.series_id IS 'Unique identifier (format: SOURCE_STATCODE)';
COMMENT ON COLUMN series.statistics_metadata_id IS 'Reference to statistics_metadata table';
COMMENT ON COLUMN series.api_code IS 'Source-specific code (links to statistics_metadata.source_stat_code)';
COMMENT ON COLUMN series.item_code IS 'Item code for this series (used when series represents a specific item from a statistic)';

-- ============================================================================
-- 5. Data Vintages Table
-- ============================================================================
CREATE TABLE data_vintages (
    vintage_id SERIAL PRIMARY KEY,
    vintage_date DATE NOT NULL,
    country VARCHAR(10) DEFAULT 'KR',
    description TEXT,
    
    -- Status tracking
    fetch_status VARCHAR(20) DEFAULT 'pending',  -- pending, in_progress, completed, failed
    fetch_started_at TIMESTAMP,
    fetch_completed_at TIMESTAMP,
    
    -- GitHub Actions tracking
    github_run_id VARCHAR(100),
    github_workflow_run_url VARCHAR(500),
    
    -- Error handling
    error_message TEXT,
    
    created_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT unique_vintage_date_country UNIQUE (vintage_date, country)
);

-- Enable RLS immediately after table creation
ALTER TABLE data_vintages ENABLE ROW LEVEL SECURITY;


CREATE INDEX idx_data_vintages_vintage_date ON data_vintages(vintage_date);
CREATE INDEX idx_data_vintages_fetch_status ON data_vintages(fetch_status);
CREATE INDEX idx_data_vintages_country ON data_vintages(country);

COMMENT ON TABLE data_vintages IS 'Data vintage snapshots for weekly data collection';

-- ============================================================================
-- 6. Ingestion Jobs Table
-- ============================================================================
CREATE TABLE ingestion_jobs (
    job_id SERIAL PRIMARY KEY,
    github_run_id VARCHAR(100) NOT NULL,
    github_workflow_run_url VARCHAR(500),
    vintage_date DATE NOT NULL,
    country VARCHAR(10) DEFAULT 'KR',
    
    -- Job status
    status VARCHAR(20) DEFAULT 'pending',  -- pending, running, completed, failed, cancelled
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    
    -- Statistics
    total_series INTEGER,
    successful_series INTEGER,
    failed_series INTEGER,
    
    -- Error handling
    error_message TEXT,
    logs_json JSONB,  -- Detailed logs
    
    created_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT fk_vintage_date FOREIGN KEY (vintage_date, country) 
        REFERENCES data_vintages(vintage_date, country)
);

-- Enable RLS immediately after table creation
ALTER TABLE ingestion_jobs ENABLE ROW LEVEL SECURITY;


CREATE INDEX idx_ingestion_jobs_github_run_id ON ingestion_jobs(github_run_id);
CREATE INDEX idx_ingestion_jobs_vintage_date ON ingestion_jobs(vintage_date);
CREATE INDEX idx_ingestion_jobs_status ON ingestion_jobs(status);

COMMENT ON TABLE ingestion_jobs IS 'GitHub Actions ingestion job tracking';

-- ============================================================================
-- 7. Observations Table
-- ============================================================================
CREATE TABLE observations (
    id SERIAL PRIMARY KEY,
    series_id VARCHAR(100) NOT NULL,
    vintage_id INTEGER NOT NULL,
    date DATE NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    
    -- Job tracking
    job_id INTEGER REFERENCES ingestion_jobs(job_id) ON DELETE SET NULL,
    
    -- Metadata
    is_forecast BOOLEAN DEFAULT FALSE,
    api_source VARCHAR(20),
    
    -- Item codes (hierarchical structure from BOK API)
    item_code1 VARCHAR(50),  -- First level item code from BOK API (ITEM_CODE1)
    item_code2 VARCHAR(50),  -- Second level item code from BOK API (ITEM_CODE2)
    item_code3 VARCHAR(50),  -- Third level item code from BOK API (ITEM_CODE3)
    item_code4 VARCHAR(50),  -- Fourth level item code from BOK API (ITEM_CODE4)
    
    -- Item names (hierarchical structure from BOK API)
    item_name1 VARCHAR(500),  -- First level item name from BOK API (ITEM_NAME1)
    item_name2 VARCHAR(500),  -- Second level item name from BOK API (ITEM_NAME2)
    item_name3 VARCHAR(500),  -- Third level item name from BOK API (ITEM_NAME3)
    item_name4 VARCHAR(500),  -- Fourth level item name from BOK API (ITEM_NAME4)
    
    -- Weight
    weight DOUBLE PRECISION,  -- Weight value from BOK API (WGT field)
    
    created_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT fk_series FOREIGN KEY (series_id) REFERENCES series(series_id) ON DELETE CASCADE,
    CONSTRAINT fk_vintage FOREIGN KEY (vintage_id) REFERENCES data_vintages(vintage_id) ON DELETE CASCADE,
    CONSTRAINT unique_series_vintage_date UNIQUE (series_id, vintage_id, date)
);

-- Enable RLS immediately after table creation
ALTER TABLE observations ENABLE ROW LEVEL SECURITY;


CREATE INDEX idx_observations_series_id ON observations(series_id);
CREATE INDEX idx_observations_vintage_id ON observations(vintage_id);
CREATE INDEX idx_observations_date ON observations(date);
CREATE INDEX idx_observations_series_vintage ON observations(series_id, vintage_id);
CREATE INDEX idx_observations_job_id ON observations(job_id);
CREATE INDEX idx_observations_item_code1 ON observations(item_code1) WHERE item_code1 IS NOT NULL;
CREATE INDEX idx_observations_item_code2 ON observations(item_code2) WHERE item_code2 IS NOT NULL;
CREATE INDEX idx_observations_item_codes ON observations(item_code1, item_code2) WHERE item_code1 IS NOT NULL;
CREATE INDEX idx_observations_item_name1 ON observations(item_name1) WHERE item_name1 IS NOT NULL;
CREATE INDEX idx_observations_item_name2 ON observations(item_name2) WHERE item_name2 IS NOT NULL;
CREATE INDEX idx_observations_weight ON observations(weight) WHERE weight IS NOT NULL;
CREATE INDEX idx_observations_item_code_name ON observations(item_code1, item_name1) WHERE item_code1 IS NOT NULL AND item_name1 IS NOT NULL;

COMMENT ON TABLE observations IS 'Time-series observations data';
COMMENT ON COLUMN observations.item_code1 IS 'First level item code from BOK API (ITEM_CODE1)';
COMMENT ON COLUMN observations.item_code2 IS 'Second level item code from BOK API (ITEM_CODE2)';
COMMENT ON COLUMN observations.item_code3 IS 'Third level item code from BOK API (ITEM_CODE3)';
COMMENT ON COLUMN observations.item_code4 IS 'Fourth level item code from BOK API (ITEM_CODE4)';
COMMENT ON COLUMN observations.item_name1 IS 'First level item name from BOK API (ITEM_NAME1)';
COMMENT ON COLUMN observations.item_name2 IS 'Second level item name from BOK API (ITEM_NAME2)';
COMMENT ON COLUMN observations.item_name3 IS 'Third level item name from BOK API (ITEM_NAME3)';
COMMENT ON COLUMN observations.item_name4 IS 'Fourth level item name from BOK API (ITEM_NAME4)';
COMMENT ON COLUMN observations.weight IS 'Weight value from BOK API (WGT field)';

-- ============================================================================
-- 8. API Fetches Table (for tracking API calls)
-- ============================================================================
CREATE TABLE api_fetches (
    id SERIAL PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES data_sources(id) ON DELETE CASCADE,
    source_stat_code VARCHAR(100) NOT NULL,
    fetch_date DATE NOT NULL,
    fetch_status VARCHAR(20) DEFAULT 'pending',  -- pending, success, failed
    fetch_started_at TIMESTAMP,
    fetch_completed_at TIMESTAMP,
    records_fetched INTEGER,
    error_message TEXT,
    metadata JSONB,
    
    created_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT fk_source_stat FOREIGN KEY (source_id, source_stat_code) 
        REFERENCES statistics_metadata(source_id, source_stat_code) ON DELETE CASCADE
);

-- Enable RLS immediately after table creation
ALTER TABLE api_fetches ENABLE ROW LEVEL SECURITY;


CREATE INDEX idx_api_fetches_source_id ON api_fetches(source_id);
CREATE INDEX idx_api_fetches_source_stat_code ON api_fetches(source_stat_code);
CREATE INDEX idx_api_fetches_fetch_date ON api_fetches(fetch_date);
CREATE INDEX idx_api_fetches_status ON api_fetches(fetch_status);

COMMENT ON TABLE api_fetches IS 'API fetch tracking for each statistic';

-- ============================================================================
-- 9. Initial Data: Insert Data Sources
-- ============================================================================
INSERT INTO data_sources (source_code, source_name, source_name_eng, api_base_url, is_active)
VALUES 
    ('BOK', '한국은행', 'Bank of Korea', 'https://ecos.bok.or.kr/api/', TRUE),
    ('KOSIS', '통계청', 'Statistics Korea', 'https://kosis.kr/openapi/', TRUE)
ON CONFLICT (source_code) DO NOTHING;

-- ============================================================================
-- 10. Helper Views
-- ============================================================================
-- View for DFM-selected statistics
CREATE VIEW dfm_selected_statistics
WITH (security_invoker = true) AS
SELECT 
    sm.id,
    ds.source_code,
    ds.source_name,
    sm.source_stat_code,
    sm.source_stat_name,
    sm.source_stat_name_eng,
    sm.cycle,
    sm.frequency_code,
    sm.dfm_priority,
    sm.is_active,
    sm.last_data_fetch_date,
    sm.last_data_fetch_status,
    sm.data_start_date,
    sm.data_end_date,
    s.series_id,
    sm.created_at,
    sm.updated_at
FROM statistics_metadata sm
JOIN data_sources ds ON sm.source_id = ds.id
LEFT JOIN series s ON s.statistics_metadata_id = sm.id
WHERE sm.is_dfm_selected = TRUE
ORDER BY sm.dfm_priority NULLS LAST, sm.source_stat_code;

COMMENT ON VIEW dfm_selected_statistics IS 'View of all DFM-selected statistics with source information';

-- View for active statistics by source
CREATE VIEW active_statistics_by_source
WITH (security_invoker = true) AS
SELECT 
    ds.source_code,
    ds.source_name,
    COUNT(*) FILTER (WHERE sm.is_active = TRUE) as active_count,
    COUNT(*) FILTER (WHERE sm.is_dfm_selected = TRUE) as dfm_selected_count,
    COUNT(*) FILTER (WHERE sm.is_searchable = TRUE) as searchable_count,
    COUNT(*) as total_count
FROM data_sources ds
LEFT JOIN statistics_metadata sm ON ds.id = sm.source_id
WHERE ds.is_active = TRUE
GROUP BY ds.source_code, ds.source_name;

COMMENT ON VIEW active_statistics_by_source IS 'Summary of statistics by data source';

-- ============================================================================
-- 11. Triggers for updated_at
-- ============================================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_data_sources_updated_at BEFORE UPDATE ON data_sources
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_statistics_metadata_updated_at BEFORE UPDATE ON statistics_metadata
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_statistics_items_updated_at BEFORE UPDATE ON statistics_items
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_series_last_updated BEFORE UPDATE ON series
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- 12. Model Configurations Table
-- ============================================================================
-- Stores DFM model configurations (block structure, series selection, etc.)
CREATE TABLE model_configs (
    config_id SERIAL PRIMARY KEY,
    config_name VARCHAR(100) NOT NULL UNIQUE,
    country VARCHAR(10) DEFAULT 'KR',
    description TEXT,
    
    -- Configuration as JSON
    config_json JSONB NOT NULL,  -- Full ModelConfig as JSON
    
    -- Block structure
    block_names TEXT[],  -- Array of block names
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT unique_config_name UNIQUE (config_name)
);

-- Enable RLS immediately after table creation
ALTER TABLE model_configs ENABLE ROW LEVEL SECURITY;


CREATE INDEX idx_model_configs_config_name ON model_configs(config_name);
CREATE INDEX idx_model_configs_country ON model_configs(country);
CREATE INDEX idx_model_configs_config_json ON model_configs USING GIN (config_json);

COMMENT ON TABLE model_configs IS 'DFM model configurations (block structure, series selection)';
COMMENT ON COLUMN model_configs.config_json IS 'Full ModelConfig as JSON (SeriesID, Blocks, Frequency, etc.)';

-- ============================================================================
-- 13. Model Block Assignments Table
-- ============================================================================
-- Links series to blocks in a model configuration
CREATE TABLE model_block_assignments (
    id SERIAL PRIMARY KEY,
    config_id INTEGER NOT NULL REFERENCES model_configs(config_id) ON DELETE CASCADE,
    series_id VARCHAR(100) NOT NULL REFERENCES series(series_id) ON DELETE CASCADE,
    block_name VARCHAR(100) NOT NULL,
    block_index INTEGER NOT NULL,  -- Index of block in block_names array
    
    created_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT unique_config_series_block UNIQUE (config_id, series_id, block_name)
);

-- Enable RLS immediately after table creation
ALTER TABLE model_block_assignments ENABLE ROW LEVEL SECURITY;


CREATE INDEX idx_model_block_assignments_config_id ON model_block_assignments(config_id);
CREATE INDEX idx_model_block_assignments_series_id ON model_block_assignments(series_id);
CREATE INDEX idx_model_block_assignments_block_name ON model_block_assignments(block_name);
CREATE INDEX idx_model_block_assignments_config_block ON model_block_assignments(config_id, block_name);

COMMENT ON TABLE model_block_assignments IS 'Series-to-block assignments for each model configuration';

-- ============================================================================
-- 14. Trained Models Table
-- ============================================================================
-- Stores trained DFM model weights and parameters
CREATE TABLE trained_models (
    model_id SERIAL PRIMARY KEY,
    config_id INTEGER NOT NULL REFERENCES model_configs(config_id) ON DELETE CASCADE,
    vintage_id INTEGER NOT NULL REFERENCES data_vintages(vintage_id) ON DELETE CASCADE,
    
    -- Training parameters
    threshold FLOAT,  -- Convergence threshold used
    convergence_iter INTEGER,  -- Number of iterations to converge
    log_likelihood FLOAT,  -- Final log-likelihood
    
    -- Model parameters as JSON (serialized numpy arrays)
    parameters_json JSONB NOT NULL,  -- {C, A, Q, R, Z_0, V_0, etc.}
    
    -- Metadata
    trained_at TIMESTAMP DEFAULT NOW(),
    training_duration_seconds FLOAT,
    notes TEXT,
    
    CONSTRAINT unique_config_vintage UNIQUE (config_id, vintage_id)
);

-- Enable RLS immediately after table creation
ALTER TABLE trained_models ENABLE ROW LEVEL SECURITY;


CREATE INDEX idx_trained_models_config_id ON trained_models(config_id);
CREATE INDEX idx_trained_models_vintage_id ON trained_models(vintage_id);
CREATE INDEX idx_trained_models_trained_at ON trained_models(trained_at DESC);
CREATE INDEX idx_trained_models_config_vintage ON trained_models(config_id, vintage_id);

COMMENT ON TABLE trained_models IS 'Trained DFM model weights and parameters';
COMMENT ON COLUMN trained_models.parameters_json IS 'Serialized DFM parameters (C, A, Q, R, Z_0, V_0, etc.) as JSON';

-- ============================================================================
-- 15. Forecast Runs Table
-- ============================================================================
-- Tracks forecast runs (nowcast updates, batch forecasts)
CREATE TABLE forecast_runs (
    run_id SERIAL PRIMARY KEY,
    model_id INTEGER NOT NULL REFERENCES trained_models(model_id) ON DELETE CASCADE,
    vintage_id_old INTEGER REFERENCES data_vintages(vintage_id) ON DELETE SET NULL,
    vintage_id_new INTEGER NOT NULL REFERENCES data_vintages(vintage_id) ON DELETE CASCADE,
    
    -- Run metadata
    run_type VARCHAR(50),  -- 'nowcast', 'forecast', 'batch'
    target_series_id VARCHAR(100),  -- Series being forecasted
    target_period VARCHAR(20),  -- Target period (e.g., '2016q4')
    
    -- GitHub Actions tracking
    github_run_id VARCHAR(100),
    github_workflow_run_url VARCHAR(500),
    
    -- Status
    status VARCHAR(20) DEFAULT 'pending',  -- pending, running, completed, failed
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    
    -- Results summary
    forecasts_generated INTEGER,
    metadata_json JSONB,
    
    created_at TIMESTAMP DEFAULT NOW()
);

-- Enable RLS immediately after table creation
ALTER TABLE forecast_runs ENABLE ROW LEVEL SECURITY;


CREATE INDEX idx_forecast_runs_model_id ON forecast_runs(model_id);
CREATE INDEX idx_forecast_runs_vintage_id_new ON forecast_runs(vintage_id_new);
CREATE INDEX idx_forecast_runs_vintage_id_old ON forecast_runs(vintage_id_old) WHERE vintage_id_old IS NOT NULL;
CREATE INDEX idx_forecast_runs_target_series_id ON forecast_runs(target_series_id) WHERE target_series_id IS NOT NULL;
CREATE INDEX idx_forecast_runs_status ON forecast_runs(status);
CREATE INDEX idx_forecast_runs_github_run_id ON forecast_runs(github_run_id) WHERE github_run_id IS NOT NULL;
CREATE INDEX idx_forecast_runs_created_at ON forecast_runs(created_at DESC);

COMMENT ON TABLE forecast_runs IS 'Tracks forecast runs (nowcast updates, batch forecasts)';

-- ============================================================================
-- 16. Forecasts Table
-- ============================================================================
-- Stores individual forecasts
CREATE TABLE forecasts (
    forecast_id SERIAL PRIMARY KEY,
    model_id INTEGER NOT NULL REFERENCES trained_models(model_id) ON DELETE CASCADE,
    run_id INTEGER REFERENCES forecast_runs(run_id) ON DELETE SET NULL,
    series_id VARCHAR(100) NOT NULL REFERENCES series(series_id) ON DELETE CASCADE,
    
    -- Forecast details
    forecast_date DATE NOT NULL,  -- Target date for forecast
    forecast_value DOUBLE PRECISION NOT NULL,
    lower_bound DOUBLE PRECISION,
    upper_bound DOUBLE PRECISION,
    confidence_level FLOAT DEFAULT 0.95,
    
    -- Additional metadata
    metadata_json JSONB,  -- Additional forecast metadata
    
    created_at TIMESTAMP DEFAULT NOW()
);

-- Enable RLS immediately after table creation
ALTER TABLE forecasts ENABLE ROW LEVEL SECURITY;


CREATE INDEX idx_forecasts_model_id ON forecasts(model_id);
CREATE INDEX idx_forecasts_run_id ON forecasts(run_id) WHERE run_id IS NOT NULL;
CREATE INDEX idx_forecasts_series_id ON forecasts(series_id);
CREATE INDEX idx_forecasts_forecast_date ON forecasts(forecast_date);
CREATE INDEX idx_forecasts_model_series_date ON forecasts(model_id, series_id, forecast_date);
CREATE INDEX idx_forecasts_created_at ON forecasts(created_at DESC);

COMMENT ON TABLE forecasts IS 'Individual forecasts from DFM models';
COMMENT ON COLUMN forecasts.forecast_date IS 'Target date for the forecast';

-- ============================================================================
-- 17. Additional Optimized Indexes for Forecasting Queries
-- ============================================================================

-- Composite index for efficient vintage data retrieval (most common query pattern)
CREATE INDEX IF NOT EXISTS idx_observations_vintage_date_series 
    ON observations(vintage_id, date, series_id);

-- Composite index for series data retrieval across vintages
CREATE INDEX IF NOT EXISTS idx_observations_series_date_vintage 
    ON observations(series_id, date, vintage_id);

-- Index for DFM-selected series filtering
CREATE INDEX IF NOT EXISTS idx_series_dfm_compatible 
    ON series(series_id, frequency, is_active) 
    WHERE is_active = TRUE;

-- Index for efficient vintage comparison queries
CREATE INDEX IF NOT EXISTS idx_observations_series_vintage_date 
    ON observations(series_id, vintage_id, date);

-- ============================================================================
-- 18. Helper Views for Forecasting
-- ============================================================================

-- View for latest forecasts per series
CREATE VIEW latest_forecasts_view
WITH (security_invoker = true) AS
SELECT DISTINCT ON (f.series_id, f.forecast_date)
    f.forecast_id,
    f.model_id,
    f.run_id,
    f.series_id,
    s.series_name,
    f.forecast_date,
    f.forecast_value,
    f.lower_bound,
    f.upper_bound,
    f.confidence_level,
    f.created_at,
    tm.config_id,
    mc.config_name,
    tm.vintage_id,
    dv.vintage_date
FROM forecasts f
JOIN series s ON f.series_id = s.series_id
JOIN trained_models tm ON f.model_id = tm.model_id
JOIN model_configs mc ON tm.config_id = mc.config_id
JOIN data_vintages dv ON tm.vintage_id = dv.vintage_id
ORDER BY f.series_id, f.forecast_date, f.created_at DESC;

COMMENT ON VIEW latest_forecasts_view IS 'Latest forecast for each series and date combination';

-- View for model training history
CREATE VIEW model_training_history
WITH (security_invoker = true) AS
SELECT 
    tm.model_id,
    tm.config_id,
    mc.config_name,
    tm.vintage_id,
    dv.vintage_date,
    tm.convergence_iter,
    tm.log_likelihood,
    tm.threshold,
    tm.trained_at,
    tm.training_duration_seconds,
    COUNT(DISTINCT f.forecast_id) as forecast_count
FROM trained_models tm
JOIN model_configs mc ON tm.config_id = mc.config_id
JOIN data_vintages dv ON tm.vintage_id = dv.vintage_id
LEFT JOIN forecasts f ON tm.model_id = f.model_id
GROUP BY tm.model_id, tm.config_id, mc.config_name, tm.vintage_id, dv.vintage_date,
         tm.convergence_iter, tm.log_likelihood, tm.threshold, tm.trained_at, tm.training_duration_seconds
ORDER BY tm.trained_at DESC;

COMMENT ON VIEW model_training_history IS 'Model training history with forecast counts';

-- ============================================================================
-- 19. Additional Triggers
-- ============================================================================

CREATE TRIGGER update_model_configs_updated_at BEFORE UPDATE ON model_configs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();


-- ============================================================================
-- PART 3: CREATE ALL RLS POLICIES
-- ============================================================================
-- All policies are separated by operation to avoid multiple permissive policies warning
-- Each table has: 1 SELECT policy (PUBLIC) + 3 write policies (authenticated)

-- Policies for data_sources
DROP POLICY IF EXISTS data_sources_public_read ON public.data_sources;
DROP POLICY IF EXISTS data_sources_insert_service ON public.data_sources;
DROP POLICY IF EXISTS data_sources_update_service ON public.data_sources;
DROP POLICY IF EXISTS data_sources_delete_service ON public.data_sources;

CREATE POLICY data_sources_public_read ON public.data_sources
    FOR SELECT TO PUBLIC USING (true);

CREATE POLICY data_sources_insert_service ON public.data_sources
    FOR INSERT TO authenticated WITH CHECK (true);

CREATE POLICY data_sources_update_service ON public.data_sources
    FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

CREATE POLICY data_sources_delete_service ON public.data_sources
    FOR DELETE TO authenticated USING (true);

-- Policies for statistics_metadata
DROP POLICY IF EXISTS statistics_metadata_public_read ON public.statistics_metadata;
DROP POLICY IF EXISTS statistics_metadata_insert_service ON public.statistics_metadata;
DROP POLICY IF EXISTS statistics_metadata_update_service ON public.statistics_metadata;
DROP POLICY IF EXISTS statistics_metadata_delete_service ON public.statistics_metadata;

CREATE POLICY statistics_metadata_public_read ON public.statistics_metadata
    FOR SELECT TO PUBLIC USING (true);

CREATE POLICY statistics_metadata_insert_service ON public.statistics_metadata
    FOR INSERT TO authenticated WITH CHECK (true);

CREATE POLICY statistics_metadata_update_service ON public.statistics_metadata
    FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

CREATE POLICY statistics_metadata_delete_service ON public.statistics_metadata
    FOR DELETE TO authenticated USING (true);

-- Policies for statistics_items
DROP POLICY IF EXISTS statistics_items_public_read ON public.statistics_items;
DROP POLICY IF EXISTS statistics_items_insert_service ON public.statistics_items;
DROP POLICY IF EXISTS statistics_items_update_service ON public.statistics_items;
DROP POLICY IF EXISTS statistics_items_delete_service ON public.statistics_items;

CREATE POLICY statistics_items_public_read ON public.statistics_items
    FOR SELECT TO PUBLIC USING (true);

CREATE POLICY statistics_items_insert_service ON public.statistics_items
    FOR INSERT TO authenticated WITH CHECK (true);

CREATE POLICY statistics_items_update_service ON public.statistics_items
    FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

CREATE POLICY statistics_items_delete_service ON public.statistics_items
    FOR DELETE TO authenticated USING (true);

-- Policies for series
DROP POLICY IF EXISTS series_public_read ON public.series;
DROP POLICY IF EXISTS series_insert_service ON public.series;
DROP POLICY IF EXISTS series_update_service ON public.series;
DROP POLICY IF EXISTS series_delete_service ON public.series;

CREATE POLICY series_public_read ON public.series
    FOR SELECT TO PUBLIC USING (true);

CREATE POLICY series_insert_service ON public.series
    FOR INSERT TO authenticated WITH CHECK (true);

CREATE POLICY series_update_service ON public.series
    FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

CREATE POLICY series_delete_service ON public.series
    FOR DELETE TO authenticated USING (true);

-- Policies for data_vintages
DROP POLICY IF EXISTS data_vintages_public_read ON public.data_vintages;
DROP POLICY IF EXISTS data_vintages_insert_service ON public.data_vintages;
DROP POLICY IF EXISTS data_vintages_update_service ON public.data_vintages;
DROP POLICY IF EXISTS data_vintages_delete_service ON public.data_vintages;

CREATE POLICY data_vintages_public_read ON public.data_vintages
    FOR SELECT TO PUBLIC USING (true);

CREATE POLICY data_vintages_insert_service ON public.data_vintages
    FOR INSERT TO authenticated WITH CHECK (true);

CREATE POLICY data_vintages_update_service ON public.data_vintages
    FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

CREATE POLICY data_vintages_delete_service ON public.data_vintages
    FOR DELETE TO authenticated USING (true);

-- Policies for ingestion_jobs
DROP POLICY IF EXISTS ingestion_jobs_public_read ON public.ingestion_jobs;
DROP POLICY IF EXISTS ingestion_jobs_insert_service ON public.ingestion_jobs;
DROP POLICY IF EXISTS ingestion_jobs_update_service ON public.ingestion_jobs;
DROP POLICY IF EXISTS ingestion_jobs_delete_service ON public.ingestion_jobs;

CREATE POLICY ingestion_jobs_public_read ON public.ingestion_jobs
    FOR SELECT TO PUBLIC USING (true);

CREATE POLICY ingestion_jobs_insert_service ON public.ingestion_jobs
    FOR INSERT TO authenticated WITH CHECK (true);

CREATE POLICY ingestion_jobs_update_service ON public.ingestion_jobs
    FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

CREATE POLICY ingestion_jobs_delete_service ON public.ingestion_jobs
    FOR DELETE TO authenticated USING (true);

-- Policies for observations
DROP POLICY IF EXISTS observations_public_read ON public.observations;
DROP POLICY IF EXISTS observations_insert_service ON public.observations;
DROP POLICY IF EXISTS observations_update_service ON public.observations;
DROP POLICY IF EXISTS observations_delete_service ON public.observations;

CREATE POLICY observations_public_read ON public.observations
    FOR SELECT TO PUBLIC USING (true);

CREATE POLICY observations_insert_service ON public.observations
    FOR INSERT TO authenticated WITH CHECK (true);

CREATE POLICY observations_update_service ON public.observations
    FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

CREATE POLICY observations_delete_service ON public.observations
    FOR DELETE TO authenticated USING (true);

-- Policies for api_fetches
DROP POLICY IF EXISTS api_fetches_public_read ON public.api_fetches;
DROP POLICY IF EXISTS api_fetches_insert_service ON public.api_fetches;
DROP POLICY IF EXISTS api_fetches_update_service ON public.api_fetches;
DROP POLICY IF EXISTS api_fetches_delete_service ON public.api_fetches;

CREATE POLICY api_fetches_public_read ON public.api_fetches
    FOR SELECT TO PUBLIC USING (true);

CREATE POLICY api_fetches_insert_service ON public.api_fetches
    FOR INSERT TO authenticated WITH CHECK (true);

CREATE POLICY api_fetches_update_service ON public.api_fetches
    FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

CREATE POLICY api_fetches_delete_service ON public.api_fetches
    FOR DELETE TO authenticated USING (true);

-- Policies for model_configs
DROP POLICY IF EXISTS model_configs_public_read ON public.model_configs;
DROP POLICY IF EXISTS model_configs_insert_service ON public.model_configs;
DROP POLICY IF EXISTS model_configs_update_service ON public.model_configs;
DROP POLICY IF EXISTS model_configs_delete_service ON public.model_configs;

CREATE POLICY model_configs_public_read ON public.model_configs
    FOR SELECT TO PUBLIC USING (true);

CREATE POLICY model_configs_insert_service ON public.model_configs
    FOR INSERT TO authenticated WITH CHECK (true);

CREATE POLICY model_configs_update_service ON public.model_configs
    FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

CREATE POLICY model_configs_delete_service ON public.model_configs
    FOR DELETE TO authenticated USING (true);

-- Policies for model_block_assignments
DROP POLICY IF EXISTS model_block_assignments_public_read ON public.model_block_assignments;
DROP POLICY IF EXISTS model_block_assignments_insert_service ON public.model_block_assignments;
DROP POLICY IF EXISTS model_block_assignments_update_service ON public.model_block_assignments;
DROP POLICY IF EXISTS model_block_assignments_delete_service ON public.model_block_assignments;

CREATE POLICY model_block_assignments_public_read ON public.model_block_assignments
    FOR SELECT TO PUBLIC USING (true);

CREATE POLICY model_block_assignments_insert_service ON public.model_block_assignments
    FOR INSERT TO authenticated WITH CHECK (true);

CREATE POLICY model_block_assignments_update_service ON public.model_block_assignments
    FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

CREATE POLICY model_block_assignments_delete_service ON public.model_block_assignments
    FOR DELETE TO authenticated USING (true);

-- Policies for trained_models
DROP POLICY IF EXISTS trained_models_public_read ON public.trained_models;
DROP POLICY IF EXISTS trained_models_insert_service ON public.trained_models;
DROP POLICY IF EXISTS trained_models_update_service ON public.trained_models;
DROP POLICY IF EXISTS trained_models_delete_service ON public.trained_models;

CREATE POLICY trained_models_public_read ON public.trained_models
    FOR SELECT TO PUBLIC USING (true);

CREATE POLICY trained_models_insert_service ON public.trained_models
    FOR INSERT TO authenticated WITH CHECK (true);

CREATE POLICY trained_models_update_service ON public.trained_models
    FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

CREATE POLICY trained_models_delete_service ON public.trained_models
    FOR DELETE TO authenticated USING (true);

-- Policies for forecast_runs
DROP POLICY IF EXISTS forecast_runs_public_read ON public.forecast_runs;
DROP POLICY IF EXISTS forecast_runs_insert_service ON public.forecast_runs;
DROP POLICY IF EXISTS forecast_runs_update_service ON public.forecast_runs;
DROP POLICY IF EXISTS forecast_runs_delete_service ON public.forecast_runs;

CREATE POLICY forecast_runs_public_read ON public.forecast_runs
    FOR SELECT TO PUBLIC USING (true);

CREATE POLICY forecast_runs_insert_service ON public.forecast_runs
    FOR INSERT TO authenticated WITH CHECK (true);

CREATE POLICY forecast_runs_update_service ON public.forecast_runs
    FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

CREATE POLICY forecast_runs_delete_service ON public.forecast_runs
    FOR DELETE TO authenticated USING (true);

-- Policies for forecasts
DROP POLICY IF EXISTS forecasts_public_read ON public.forecasts;
DROP POLICY IF EXISTS forecasts_insert_service ON public.forecasts;
DROP POLICY IF EXISTS forecasts_update_service ON public.forecasts;
DROP POLICY IF EXISTS forecasts_delete_service ON public.forecasts;

CREATE POLICY forecasts_public_read ON public.forecasts
    FOR SELECT TO PUBLIC USING (true);

CREATE POLICY forecasts_insert_service ON public.forecasts
    FOR INSERT TO authenticated WITH CHECK (true);

CREATE POLICY forecasts_update_service ON public.forecasts
    FOR UPDATE TO authenticated USING (true) WITH CHECK (true);

CREATE POLICY forecasts_delete_service ON public.forecasts
    FOR DELETE TO authenticated USING (true);

-- ============================================================================
-- PART 4: CREATE FUNCTION WITH SEARCH_PATH FIXED
-- ============================================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER
LANGUAGE plpgsql
SET search_path = ''
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- ============================================================================
-- PART 5: CREATE TRIGGERS FOR updated_at
-- ============================================================================

CREATE TRIGGER update_data_sources_updated_at BEFORE UPDATE ON data_sources
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_statistics_metadata_updated_at BEFORE UPDATE ON statistics_metadata
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_statistics_items_updated_at BEFORE UPDATE ON statistics_items
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_series_last_updated BEFORE UPDATE ON series
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- Migration Complete
-- ============================================================================
-- All objects have been recreated with proper security from the start:
-- ✅ All tables have RLS enabled
-- ✅ All views use security_invoker = true
-- ✅ All policies are separated by operation
-- ✅ Function has search_path = ''
-- ============================================================================
