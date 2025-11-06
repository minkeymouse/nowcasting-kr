-- ============================================================================
-- Migration: 001_initial_schema.sql
-- Purpose: Initial database schema for DFM nowcasting system
-- 
-- This schema is designed around CSV-based series definition (001_initial_spec.csv)
-- Key features:
-- - CSV-driven series definition (series_id, api_group_id, etc.)
-- - API optimization via series_groups (same data_code = same group)
-- - Vintage system for data snapshots
-- - DFM model training and forecasting support
-- ============================================================================

-- ============================================================================
-- PART 0: DROP ALL EXISTING OBJECTS (CLEAN SLATE)
-- ============================================================================
-- Drop all existing tables, views, functions, and triggers in correct order

DROP TABLE IF EXISTS forecasts CASCADE;
DROP TABLE IF EXISTS trained_models CASCADE;
DROP TABLE IF EXISTS model_configs CASCADE;
DROP TABLE IF EXISTS observations CASCADE;
DROP TABLE IF EXISTS data_vintages CASCADE;
DROP TABLE IF EXISTS series CASCADE;
DROP TABLE IF EXISTS factors CASCADE;
DROP TABLE IF EXISTS factor_values CASCADE;
DROP TABLE IF EXISTS factor_loadings CASCADE;

-- Drop removed tables (simplified schema)
DROP TABLE IF EXISTS series_groups CASCADE;
DROP TABLE IF EXISTS model_block_assignments CASCADE;
DROP TABLE IF EXISTS forecast_runs CASCADE;
DROP TABLE IF EXISTS ingestion_jobs CASCADE;

-- Drop optional/legacy tables (not in current schema)
DROP TABLE IF EXISTS data_sources CASCADE;
DROP TABLE IF EXISTS api_fetches CASCADE;
DROP TABLE IF EXISTS statistics_items CASCADE;
DROP TABLE IF EXISTS statistics_metadata CASCADE;
-- Note: factors, factor_loadings, factor_values are now part of the schema
-- They will be dropped and recreated
DROP TABLE IF EXISTS factor_values CASCADE;
DROP TABLE IF EXISTS factor_loadings CASCADE;
DROP TABLE IF EXISTS factors CASCADE;

-- Drop views
DROP VIEW IF EXISTS latest_forecasts_view CASCADE;
DROP VIEW IF EXISTS model_training_history CASCADE;
DROP VIEW IF EXISTS series_with_groups CASCADE;
DROP VIEW IF EXISTS variables_view CASCADE;
DROP VIEW IF EXISTS variable_values_view CASCADE;

-- Drop functions
DROP FUNCTION IF EXISTS update_updated_at_column() CASCADE;

-- ============================================================================
-- 1. Series Table
-- ============================================================================
-- Primary table for time-series definitions from CSV
CREATE TABLE IF NOT EXISTS series (
    series_id VARCHAR(100) PRIMARY KEY,  -- Format: {api_source}_{data_code}_{item_id}
    series_name VARCHAR(500) NOT NULL,
    
    -- API source and codes
    api_source VARCHAR(20) NOT NULL,  -- BOK, KOSIS
    data_code VARCHAR(100) NOT NULL,  -- Statistic code (BOK: 200Y106, KOSIS: 101_DT_1DA7002S)
    item_id VARCHAR(50) NOT NULL,  -- Item identifier (BOK: 1400, KOSIS: T80)
    
    -- API optimization (simplified: group_id as string, no FK)
    api_group_id VARCHAR(100),  -- Group identifier: {api_source}_{data_code} (nullable for non-grouped series)
    
    -- Time-series properties
    frequency VARCHAR(10) NOT NULL,  -- d, m, q, a (daily, monthly, quarterly, annual)
    transformation VARCHAR(50) NOT NULL,  -- lin, pch, pca, chg, log, etc.
    category VARCHAR(100),  -- GDP, Investment, Labor, Financial, External
    units VARCHAR(50),  -- Billion Won, Percent, Index, etc.
    country VARCHAR(10) DEFAULT 'KR',  -- Country code
    
    -- Metadata
    description TEXT,
    priority INTEGER,  -- DFM priority/rank
    is_active BOOLEAN DEFAULT TRUE,
    is_kpi BOOLEAN DEFAULT FALSE,  -- KPI 여부 (프론트엔드 시각화용)
    metadata JSONB,  -- Additional flexible metadata
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT unique_series_identifier UNIQUE (api_source, data_code, item_id)
);

CREATE INDEX idx_series_api_source ON series(api_source);
CREATE INDEX idx_series_data_code ON series(data_code);
CREATE INDEX idx_series_item_id ON series(item_id);
CREATE INDEX idx_series_api_group_id ON series(api_group_id) WHERE api_group_id IS NOT NULL;
CREATE INDEX idx_series_frequency ON series(frequency);
CREATE INDEX idx_series_transformation ON series(transformation);
CREATE INDEX idx_series_category ON series(category);
CREATE INDEX idx_series_active ON series(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_series_country ON series(country);
CREATE INDEX idx_series_is_kpi ON series(is_kpi) WHERE is_kpi = TRUE;

COMMENT ON TABLE series IS 'Time-series definitions from CSV (001_initial_spec.csv)';
COMMENT ON COLUMN series.is_kpi IS 'KPI flag for frontend visualization';
COMMENT ON COLUMN series.series_id IS 'Unique identifier: {api_source}_{data_code}_{item_id}';
COMMENT ON COLUMN series.api_group_id IS 'Group identifier: {api_source}_{data_code} (same data_code = same group, for API optimization)';
COMMENT ON COLUMN series.data_code IS 'Statistic code from API (e.g., BOK: 200Y106, KOSIS: 101_DT_1DA7002S)';
COMMENT ON COLUMN series.item_id IS 'Item identifier from API (e.g., BOK: 1400, KOSIS: T80)';

-- ============================================================================
-- 3. Data Vintages Table (with ingestion job info integrated)
-- ============================================================================
-- Data snapshots at specific points in time
-- Includes ingestion job tracking (simplified: no separate ingestion_jobs table)
CREATE TABLE IF NOT EXISTS data_vintages (
    vintage_id SERIAL PRIMARY KEY,
    vintage_date DATE NOT NULL,
    country VARCHAR(10) DEFAULT 'KR',
    description TEXT,
    
    -- Status tracking
    fetch_status VARCHAR(20) DEFAULT 'pending',  -- pending, in_progress, completed, failed
    fetch_started_at TIMESTAMP,
    fetch_completed_at TIMESTAMP,
    
    -- GitHub Actions tracking (from ingestion_jobs)
    github_run_id VARCHAR(100),
    github_workflow_run_url VARCHAR(500),
    
    -- Ingestion statistics (from ingestion_jobs)
    total_series INTEGER,
    successful_series INTEGER,
    failed_series INTEGER,
    
    -- Error handling
    error_message TEXT,
    logs_json JSONB,  -- Detailed logs (from ingestion_jobs)
    
    created_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT unique_vintage_date_country UNIQUE (vintage_date, country)
);

CREATE INDEX idx_data_vintages_vintage_date ON data_vintages(vintage_date);
CREATE INDEX idx_data_vintages_fetch_status ON data_vintages(fetch_status);
CREATE INDEX idx_data_vintages_country ON data_vintages(country);

COMMENT ON TABLE data_vintages IS 'Data vintage snapshots for time-series data collection';
COMMENT ON COLUMN data_vintages.vintage_date IS 'Date of the data snapshot';

-- ============================================================================
-- 4. Observations Table
-- ============================================================================
-- Time-series observations data
CREATE TABLE IF NOT EXISTS observations (
    id SERIAL PRIMARY KEY,
    series_id VARCHAR(100) NOT NULL,
    vintage_id INTEGER NOT NULL,
    date DATE NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    
    -- Job tracking (simplified: github_run_id from vintage)
    github_run_id VARCHAR(100),
    
    -- Metadata
    is_forecast BOOLEAN DEFAULT FALSE,
    api_source VARCHAR(20),
    
    -- Additional metadata (JSON for flexibility)
    metadata JSONB,  -- Can store item codes, names, weights, etc.
    
    created_at TIMESTAMP DEFAULT NOW(),
    
    -- Foreign keys
    CONSTRAINT fk_series FOREIGN KEY (series_id) 
        REFERENCES series(series_id) ON DELETE CASCADE,
    CONSTRAINT fk_vintage FOREIGN KEY (vintage_id) 
        REFERENCES data_vintages(vintage_id) ON DELETE CASCADE,
    
    -- Constraints
    CONSTRAINT unique_series_vintage_date UNIQUE (series_id, vintage_id, date)
);

CREATE INDEX idx_observations_series_id ON observations(series_id);
CREATE INDEX idx_observations_vintage_id ON observations(vintage_id);
CREATE INDEX idx_observations_date ON observations(date);
CREATE INDEX idx_observations_series_vintage ON observations(series_id, vintage_id);
CREATE INDEX idx_observations_github_run_id ON observations(github_run_id) WHERE github_run_id IS NOT NULL;
CREATE INDEX idx_observations_series_vintage_date ON observations(series_id, vintage_id, date);
CREATE INDEX idx_observations_vintage_date_series ON observations(vintage_id, date, series_id);

COMMENT ON TABLE observations IS 'Time-series observations data';
COMMENT ON COLUMN observations.metadata IS 'Additional metadata (item codes, names, weights, etc.) as JSON';

-- ============================================================================
-- 5. Forecasts Table (with run info integrated)
-- ============================================================================
-- Stores individual forecasts
CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id SERIAL PRIMARY KEY,
    model_id INTEGER NOT NULL,  -- Model ID (no FK, stored locally as pickle)
    series_id VARCHAR(100) NOT NULL REFERENCES series(series_id) ON DELETE CASCADE,
    
    -- Forecast details
    forecast_date DATE NOT NULL,  -- Target date for forecast
    forecast_value DOUBLE PRECISION NOT NULL,
    lower_bound DOUBLE PRECISION,
    upper_bound DOUBLE PRECISION,
    confidence_level FLOAT DEFAULT 0.95,
    
    -- Run info (from forecast_runs, integrated)
    run_type VARCHAR(50),  -- 'nowcast', 'forecast', 'batch'
    vintage_id_old INTEGER REFERENCES data_vintages(vintage_id) ON DELETE SET NULL,
    vintage_id_new INTEGER REFERENCES data_vintages(vintage_id) ON DELETE SET NULL,
    github_run_id VARCHAR(100),
    
    -- Additional metadata
    metadata_json JSONB,  -- Additional forecast metadata (run info, etc.)
    
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_forecasts_model_id ON forecasts(model_id);
CREATE INDEX idx_forecasts_series_id ON forecasts(series_id);
CREATE INDEX idx_forecasts_forecast_date ON forecasts(forecast_date);
CREATE INDEX idx_forecasts_model_series_date ON forecasts(model_id, series_id, forecast_date);
CREATE INDEX idx_forecasts_created_at ON forecasts(created_at DESC);
CREATE INDEX idx_forecasts_run_type ON forecasts(run_type) WHERE run_type IS NOT NULL;
CREATE INDEX idx_forecasts_vintage_id_new ON forecasts(vintage_id_new) WHERE vintage_id_new IS NOT NULL;

COMMENT ON TABLE forecasts IS 'Individual forecasts from DFM models';
COMMENT ON COLUMN forecasts.forecast_date IS 'Target date for the forecast';

-- ============================================================================
-- 8. Factors Table (DFM Factors for Frontend Visualization)
-- ============================================================================
-- Stores DFM factor metadata
CREATE TABLE IF NOT EXISTS factors (
    id SERIAL PRIMARY KEY,
    model_id INTEGER NOT NULL,  -- Model ID (no FK, stored locally as pickle)
    name VARCHAR(100) NOT NULL,  -- Factor name (e.g., "Factor 1", "Global Factor")
    description TEXT,  -- Factor description
    factor_index INTEGER NOT NULL,  -- Index/order of factor in the model
    
    created_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT unique_model_factor_index UNIQUE (model_id, factor_index)
);

CREATE INDEX idx_factors_model_id ON factors(model_id);
CREATE INDEX idx_factors_factor_index ON factors(factor_index);

COMMENT ON TABLE factors IS 'DFM factor metadata for frontend visualization';
COMMENT ON COLUMN factors.factor_index IS 'Index/order of factor in the model (0-based or 1-based)';

-- ============================================================================
-- 9. Factor Values Table (Factor Time Series)
-- ============================================================================
-- Stores time series of factor estimates
CREATE TABLE IF NOT EXISTS factor_values (
    id SERIAL PRIMARY KEY,
    factor_id INTEGER NOT NULL REFERENCES factors(id) ON DELETE CASCADE,
    vintage_id INTEGER NOT NULL REFERENCES data_vintages(vintage_id) ON DELETE CASCADE,
    date DATE NOT NULL,  -- Observation date
    value DOUBLE PRECISION NOT NULL,  -- Factor value
    
    created_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT unique_factor_vintage_date UNIQUE (factor_id, vintage_id, date)
);

CREATE INDEX idx_factor_values_factor_id ON factor_values(factor_id);
CREATE INDEX idx_factor_values_vintage_id ON factor_values(vintage_id);
CREATE INDEX idx_factor_values_date ON factor_values(date);
CREATE INDEX idx_factor_values_factor_vintage ON factor_values(factor_id, vintage_id);
CREATE INDEX idx_factor_values_factor_date ON factor_values(factor_id, date);

COMMENT ON TABLE factor_values IS 'Time series of factor estimates (DFM factor values over time)';
COMMENT ON COLUMN factor_values.vintage_id IS 'Vintage snapshot this factor value belongs to';

-- ============================================================================
-- 10. Factor Loadings Table (Factor-Variable Loadings)
-- ============================================================================
-- Stores factor loading matrix: how each variable (series) loads onto each factor
CREATE TABLE IF NOT EXISTS factor_loadings (
    factor_id INTEGER NOT NULL REFERENCES factors(id) ON DELETE CASCADE,
    series_id VARCHAR(100) NOT NULL REFERENCES series(series_id) ON DELETE CASCADE,
    loading DOUBLE PRECISION NOT NULL,  -- Loading coefficient (can be positive or negative)
    
    created_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT pk_factor_loadings PRIMARY KEY (factor_id, series_id)
);

CREATE INDEX idx_factor_loadings_factor_id ON factor_loadings(factor_id);
CREATE INDEX idx_factor_loadings_series_id ON factor_loadings(series_id);
CREATE INDEX idx_factor_loadings_loading ON factor_loadings(loading);

COMMENT ON TABLE factor_loadings IS 'Factor loading matrix: how each variable (series) loads onto each factor';
COMMENT ON COLUMN factor_loadings.loading IS 'Loading coefficient (positive or negative)';

-- ============================================================================
-- ============================================================================
-- 11. Helper Views
-- ============================================================================

-- View for series with group information (simplified: no series_groups join)
CREATE OR REPLACE VIEW series_with_groups
WITH (security_invoker=true) AS
SELECT 
    series_id,
    series_name,
    api_source,
    data_code,
    item_id,
    api_group_id,
    frequency,
    transformation,
    category,
    units,
    country,
    is_active,
    is_kpi,
    created_at,
    updated_at
FROM series
WHERE is_active = TRUE;

COMMENT ON VIEW series_with_groups IS 'View of active series with their group information (api_group_id)';

-- View for latest forecasts per series
CREATE OR REPLACE VIEW latest_forecasts_view
WITH (security_invoker=true) AS
SELECT DISTINCT ON (f.series_id, f.forecast_date)
    f.forecast_id,
    f.model_id,
    f.series_id,
    s.series_name,
    f.forecast_date,
    f.forecast_value,
    f.lower_bound,
    f.upper_bound,
    f.confidence_level,
    f.created_at
FROM forecasts f
JOIN series s ON f.series_id = s.series_id
ORDER BY f.series_id, f.forecast_date, f.created_at DESC;

COMMENT ON VIEW latest_forecasts_view IS 'Latest forecast for each series and date combination';

-- View for variables (frontend visualization - based on series)
CREATE OR REPLACE VIEW variables_view
WITH (security_invoker=true) AS
SELECT 
    series_id AS id,
    series_name AS name,
    units AS unit,
    is_kpi,
    frequency,
    transformation,
    category,
    country,
    is_active,
    created_at,
    updated_at
FROM series
WHERE is_active = TRUE;

COMMENT ON VIEW variables_view IS 'Variables view for frontend visualization (based on series table)';

-- View for variable values (frontend visualization - based on observations)
CREATE OR REPLACE VIEW variable_values_view
WITH (security_invoker=true) AS
SELECT 
    series_id AS variable_id,
    date,
    value,
    vintage_id,
    github_run_id,
    is_forecast,
    created_at
FROM observations;

COMMENT ON VIEW variable_values_view IS 'Variable values view for frontend visualization (based on observations table)';

-- ============================================================================
-- 13. Triggers for updated_at
-- ============================================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;


CREATE TRIGGER update_series_updated_at BEFORE UPDATE ON series
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();


-- ============================================================================
-- 14. Row Level Security (RLS) Policies
-- ============================================================================
-- Enable RLS on all tables for security

-- Series
ALTER TABLE series ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow public read access to series"
    ON series FOR SELECT
    USING (true);
CREATE POLICY "Allow authenticated insert to series"
    ON series FOR INSERT
    TO authenticated
    WITH CHECK (true);
CREATE POLICY "Allow authenticated update to series"
    ON series FOR UPDATE
    TO authenticated
    USING (true);

-- Data Vintages
ALTER TABLE data_vintages ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow public read access to data_vintages"
    ON data_vintages FOR SELECT
    USING (true);
CREATE POLICY "Allow authenticated insert to data_vintages"
    ON data_vintages FOR INSERT
    TO authenticated
    WITH CHECK (true);
CREATE POLICY "Allow authenticated update to data_vintages"
    ON data_vintages FOR UPDATE
    TO authenticated
    USING (true);

-- Observations
ALTER TABLE observations ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow public read access to observations"
    ON observations FOR SELECT
    USING (true);
CREATE POLICY "Allow authenticated insert to observations"
    ON observations FOR INSERT
    TO authenticated
    WITH CHECK (true);
CREATE POLICY "Allow authenticated update to observations"
    ON observations FOR UPDATE
    TO authenticated
    USING (true);


-- Forecasts
ALTER TABLE forecasts ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow public read access to forecasts"
    ON forecasts FOR SELECT
    USING (true);
CREATE POLICY "Allow authenticated insert to forecasts"
    ON forecasts FOR INSERT
    TO authenticated
    WITH CHECK (true);
CREATE POLICY "Allow authenticated update to forecasts"
    ON forecasts FOR UPDATE
    TO authenticated
    USING (true);

-- Factors
ALTER TABLE factors ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow public read access to factors"
    ON factors FOR SELECT
    USING (true);
CREATE POLICY "Allow authenticated insert to factors"
    ON factors FOR INSERT
    TO authenticated
    WITH CHECK (true);
CREATE POLICY "Allow authenticated update to factors"
    ON factors FOR UPDATE
    TO authenticated
    USING (true);

-- Factor Values
ALTER TABLE factor_values ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow public read access to factor_values"
    ON factor_values FOR SELECT
    USING (true);
CREATE POLICY "Allow authenticated insert to factor_values"
    ON factor_values FOR INSERT
    TO authenticated
    WITH CHECK (true);
CREATE POLICY "Allow authenticated update to factor_values"
    ON factor_values FOR UPDATE
    TO authenticated
    USING (true);

-- Factor Loadings
ALTER TABLE factor_loadings ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow public read access to factor_loadings"
    ON factor_loadings FOR SELECT
    USING (true);
CREATE POLICY "Allow authenticated insert to factor_loadings"
    ON factor_loadings FOR INSERT
    TO authenticated
    WITH CHECK (true);
CREATE POLICY "Allow authenticated update to factor_loadings"
    ON factor_loadings FOR UPDATE
    TO authenticated
    USING (true);
