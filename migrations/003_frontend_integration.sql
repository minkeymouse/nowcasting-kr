-- ============================================================================
-- Migration: 003_frontend_integration.sql
-- Purpose: Frontend optimization and data cleanup
-- 
-- This migration:
-- 1. Creates optimized views for frontend (latest data only)
-- 2. Adds cleanup functions for old data
-- 3. Cleans up existing data to keep only latest models
-- 4. Removes unused tables and views from previous migrations
-- 
-- This is an incremental migration that adds views and functions.
-- It also cleans up unused objects from previous migrations.
-- This migration is idempotent and can be run multiple times safely.
-- ============================================================================

-- ============================================================================
-- PART -1: Cleanup Unused Tables and Views
-- ============================================================================
-- Remove tables and views that were created in previous migrations but are no longer used

-- Drop unused tables (these were dropped in 001 but may still exist in some databases)
DROP TABLE IF EXISTS trained_models CASCADE;
DROP TABLE IF EXISTS model_configs CASCADE;
DROP TABLE IF EXISTS series_groups CASCADE;
DROP TABLE IF EXISTS model_block_assignments CASCADE;
DROP TABLE IF EXISTS forecast_runs CASCADE;
DROP TABLE IF EXISTS ingestion_jobs CASCADE;
DROP TABLE IF EXISTS data_sources CASCADE;
DROP TABLE IF EXISTS api_fetches CASCADE;
DROP TABLE IF EXISTS statistics_items CASCADE;
DROP TABLE IF EXISTS statistics_metadata CASCADE;

-- Drop unused/duplicate views
-- series_with_groups (001) was replaced by series_with_blocks (002)
DROP VIEW IF EXISTS series_with_groups CASCADE;
-- model_training_history was never created but was dropped in 001
DROP VIEW IF EXISTS model_training_history CASCADE;
-- variable_values_view may not be used (check if needed)
-- Keeping it for now as it might be used by frontend

COMMENT ON SCHEMA public IS 'Cleaned up unused tables and views from previous migrations';

-- ============================================================================
-- PART 0: Cleanup Unused dfm_results Table (if exists from previous migration attempts)
-- ============================================================================
-- Note: dfm_results table was created but never used. All necessary data is already
-- stored in factors, factor_values, and factor_loadings tables. Model weights are
-- stored in Supabase storage as pickle files. This table is redundant and removed.

DROP TABLE IF EXISTS dfm_results CASCADE;
DROP VIEW IF EXISTS latest_dfm_results_view CASCADE;

-- ============================================================================
-- PART 1: Cleanup Functions
-- ============================================================================

-- Function to cleanup old factors (keeps only latest N models)
CREATE OR REPLACE FUNCTION cleanup_old_factors(keep_latest_models INTEGER DEFAULT 3)
RETURNS TABLE(
    deleted_factors INTEGER,
    deleted_factor_values BIGINT,
    deleted_factor_loadings BIGINT
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    model_ids_to_delete INTEGER[];
    factor_ids_to_delete INTEGER[];
    deleted_factors_count INTEGER := 0;
    deleted_factor_values_count BIGINT := 0;
    deleted_factor_loadings_count BIGINT := 0;
BEGIN
    -- Get model_ids to keep (latest N by created_at)
    WITH latest_models AS (
        SELECT model_id, MAX(created_at) as max_created_at
        FROM public.factors
        GROUP BY model_id
        ORDER BY max_created_at DESC
        LIMIT keep_latest_models
    ),
    all_models AS (
        SELECT DISTINCT model_id
        FROM public.factors
    )
    SELECT array_agg(am.model_id)
    INTO model_ids_to_delete
    FROM all_models am
    WHERE am.model_id NOT IN (SELECT model_id FROM latest_models);
    
    -- If no models to delete, return early
    IF model_ids_to_delete IS NULL OR array_length(model_ids_to_delete, 1) = 0 THEN
        RETURN QUERY SELECT 0, 0::BIGINT, 0::BIGINT;
        RETURN;
    END IF;
    
    -- Get factor_ids to delete
    SELECT array_agg(id)
    INTO factor_ids_to_delete
    FROM public.factors
    WHERE model_id = ANY(model_ids_to_delete);
    
    -- Count factor_values and factor_loadings that will be deleted
    IF factor_ids_to_delete IS NOT NULL AND array_length(factor_ids_to_delete, 1) > 0 THEN
        SELECT COUNT(*) INTO deleted_factor_values_count
        FROM public.factor_values
        WHERE factor_id = ANY(factor_ids_to_delete);
        
        SELECT COUNT(*) INTO deleted_factor_loadings_count
        FROM public.factor_loadings
        WHERE factor_id = ANY(factor_ids_to_delete);
    END IF;
    
    -- Delete factors (CASCADE will delete factor_values and factor_loadings)
    DELETE FROM public.factors
    WHERE model_id = ANY(model_ids_to_delete);
    
    GET DIAGNOSTICS deleted_factors_count = ROW_COUNT;
    
    RETURN QUERY SELECT deleted_factors_count, deleted_factor_values_count, deleted_factor_loadings_count;
END;
$$;

COMMENT ON FUNCTION cleanup_old_factors IS 'Cleanup old factors, keeping only latest N models. Returns counts of deleted records.';

-- Function to cleanup old forecasts (keeps only latest N forecasts per series/date)
CREATE OR REPLACE FUNCTION cleanup_old_forecasts(keep_latest_per_series INTEGER DEFAULT 1)
RETURNS INTEGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    deleted_count INTEGER := 0;
BEGIN
    -- Delete old forecasts, keeping only latest N per (series_id, forecast_date, run_type)
    WITH ranked_forecasts AS (
        SELECT 
            forecast_id,
            ROW_NUMBER() OVER (
                PARTITION BY series_id, forecast_date, run_type 
                ORDER BY created_at DESC
            ) AS rn
        FROM public.forecasts
    )
    DELETE FROM public.forecasts
    WHERE forecast_id IN (
        SELECT forecast_id
        FROM ranked_forecasts
        WHERE rn > keep_latest_per_series
    );
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$;

COMMENT ON FUNCTION cleanup_old_forecasts IS 'Cleanup old forecasts, keeping only latest N per (series_id, forecast_date, run_type).';

-- ============================================================================
-- PART 2: Frontend-Optimized Views (Latest Data Only)
-- ============================================================================

-- View for latest factors only (from latest model)
DROP VIEW IF EXISTS latest_factors_view CASCADE;
CREATE VIEW latest_factors_view
WITH (security_invoker=true) AS
SELECT 
    f.id,
    f.model_id,
    f.name,
    f.description,
    f.factor_index,
    f.block_name,
    -- Block ID: hash of block_name for consistent ID generation (matches frontend logic)
    CASE 
        WHEN f.block_name IS NULL THEN NULL
        ELSE abs(hashtext(f.block_name))::INTEGER
    END AS block_id,
    f.created_at
FROM factors f
WHERE f.model_id = (
    SELECT model_id
    FROM factors
    GROUP BY model_id
    ORDER BY MAX(created_at) DESC
    LIMIT 1
)
ORDER BY f.factor_index;

COMMENT ON VIEW latest_factors_view IS 'Latest factors from the most recent model (for frontend visualization). block_id is a hash of block_name for consistent ID generation.';

-- View for latest factor values only (from latest model)
DROP VIEW IF EXISTS latest_factor_values_view CASCADE;
CREATE VIEW latest_factor_values_view
WITH (security_invoker=true) AS
SELECT 
    fv.id,
    fv.factor_id,
    fv.vintage_id,
    fv.date,
    fv.value,
    fv.created_at,
    f.model_id,
    f.factor_index,
    f.name AS factor_name,
    f.block_name,
    -- Block ID: hash of block_name for consistent ID generation
    CASE 
        WHEN f.block_name IS NULL THEN NULL
        ELSE abs(hashtext(f.block_name))::INTEGER
    END AS block_id
FROM factor_values fv
JOIN factors f ON fv.factor_id = f.id
WHERE f.model_id = (
    SELECT model_id
    FROM factors
    GROUP BY model_id
    ORDER BY MAX(created_at) DESC
    LIMIT 1
)
ORDER BY f.factor_index, fv.date;

COMMENT ON VIEW latest_factor_values_view IS 'Latest factor values from the most recent model (for frontend visualization). block_id is a hash of block_name for consistent ID generation.';

-- View for latest factor loadings only (from latest model)
DROP VIEW IF EXISTS latest_factor_loadings_view CASCADE;
CREATE VIEW latest_factor_loadings_view
WITH (security_invoker=true) AS
SELECT 
    fl.factor_id,
    fl.series_id,
    fl.loading,
    fl.created_at,
    f.model_id,
    f.factor_index,
    f.name AS factor_name,
    f.block_name AS factor_block_name,
    -- Block ID for factor: hash of block_name
    CASE 
        WHEN f.block_name IS NULL THEN NULL
        ELSE abs(hashtext(f.block_name))::INTEGER
    END AS factor_block_id,
    s.series_name,
    -- Variable (series) block name from blocks table (most recent config)
    (SELECT DISTINCT b.block_name 
     FROM blocks b
     WHERE b.series_id = fl.series_id 
     AND b.config_name = (SELECT MAX(config_name) FROM blocks WHERE series_id = fl.series_id)
     LIMIT 1
    ) AS variable_block_name,
    -- Variable block ID: hash of variable_block_name
    CASE 
        WHEN (SELECT DISTINCT b.block_name 
              FROM blocks b
              WHERE b.series_id = fl.series_id 
              AND b.config_name = (SELECT MAX(config_name) FROM blocks WHERE series_id = fl.series_id)
              LIMIT 1) IS NULL THEN NULL
        ELSE abs(hashtext(
            (SELECT DISTINCT b.block_name 
             FROM blocks b
             WHERE b.series_id = fl.series_id 
             AND b.config_name = (SELECT MAX(config_name) FROM blocks WHERE series_id = fl.series_id)
             LIMIT 1)
        ))::INTEGER
    END AS variable_block_id
FROM factor_loadings fl
JOIN factors f ON fl.factor_id = f.id
JOIN series s ON fl.series_id = s.series_id
WHERE f.model_id = (
    SELECT model_id
    FROM factors
    GROUP BY model_id
    ORDER BY MAX(created_at) DESC
    LIMIT 1
)
ORDER BY f.factor_index, s.series_name;

COMMENT ON VIEW latest_factor_loadings_view IS 'Latest factor loadings from the most recent model (for frontend visualization). Includes both factor_block_name/id and variable_block_name/id for comprehensive block information.';

-- Enhanced latest forecasts view (with block information)
DROP VIEW IF EXISTS latest_forecasts_view CASCADE;
CREATE VIEW latest_forecasts_view
WITH (security_invoker=true) AS
SELECT DISTINCT ON (f.series_id, f.forecast_date, f.run_type)
    f.forecast_id,
    f.model_id,
    f.series_id,
    s.series_name,
    f.forecast_date,
    f.forecast_value,
    f.lower_bound,
    f.upper_bound,
    f.confidence_level,
    f.run_type,
    f.vintage_id_old,
    f.vintage_id_new,
    f.metadata_json,
    f.created_at,
    -- Block information
    (SELECT array_agg(DISTINCT b.block_name ORDER BY b.block_name)
     FROM blocks b
     WHERE b.series_id = f.series_id
     AND b.config_name = (SELECT MAX(config_name) FROM blocks WHERE series_id = f.series_id)
    ) AS block_names
FROM forecasts f
JOIN series s ON f.series_id = s.series_id
ORDER BY f.series_id, f.forecast_date, f.run_type, f.created_at DESC;

COMMENT ON VIEW latest_forecasts_view IS 'Latest forecast for each series, date, and run_type combination with block information (replaces views from 001 and 002)';

-- View for latest observations (most recent vintage only)
DROP VIEW IF EXISTS latest_observations_view CASCADE;
CREATE VIEW latest_observations_view
WITH (security_invoker=true) AS
SELECT 
    o.id AS observation_id,
    o.series_id,
    s.series_name,
    o.date,
    o.value,
    o.vintage_id,
    o.github_run_id,
    o.is_forecast,
    o.created_at,
    dv.vintage_date
FROM observations o
JOIN series s ON o.series_id = s.series_id
JOIN data_vintages dv ON o.vintage_id = dv.vintage_id
WHERE o.vintage_id = (
    SELECT vintage_id
    FROM data_vintages
    ORDER BY vintage_date DESC
    LIMIT 1
)
ORDER BY o.series_id, o.date;

COMMENT ON VIEW latest_observations_view IS 'Latest observations from the most recent vintage (for frontend visualization)';

-- ============================================================================
-- PART 3: View Consolidation and Cleanup
-- ============================================================================
-- Ensure all views are up-to-date and remove any duplicates

-- Update variables_view to include block information (if not already updated in 002)
-- This ensures consistency across all views
DROP VIEW IF EXISTS variables_view CASCADE;
CREATE VIEW variables_view
WITH (security_invoker=true) AS
SELECT 
    s.series_id AS id,
    s.series_name AS name,
    s.units AS unit,
    s.is_kpi,
    s.frequency,
    s.transformation,
    s.category,
    s.country,
    s.is_active,
    -- Block information from most recent config
    (SELECT array_agg(DISTINCT b.block_name ORDER BY b.block_name)
     FROM blocks b
     WHERE b.series_id = s.series_id
     AND b.config_name = (SELECT MAX(config_name) FROM blocks WHERE series_id = s.series_id)
    ) AS block_names,
    s.created_at,
    s.updated_at
FROM series s
WHERE s.is_active = TRUE;

COMMENT ON VIEW variables_view IS 'Variables view for frontend visualization with block information (updated from 001 and 002)';

-- Ensure series_with_blocks view exists (from 002, but ensure it's current)
DROP VIEW IF EXISTS series_with_blocks CASCADE;
CREATE VIEW series_with_blocks
WITH (security_invoker=true) AS
SELECT 
    s.series_id,
    s.series_name,
    s.api_source,
    s.data_code,
    s.item_id,
    s.api_group_id,
    s.frequency,
    s.transformation,
    s.category,
    s.units,
    s.country,
    s.is_active,
    s.is_kpi,
    -- Block information from most recent config
    (SELECT MAX(config_name) FROM blocks WHERE series_id = s.series_id) AS latest_config_name,
    (SELECT array_agg(DISTINCT b.block_name ORDER BY b.block_name)
     FROM blocks b
     WHERE b.series_id = s.series_id
     AND b.config_name = (SELECT MAX(config_name) FROM blocks WHERE series_id = s.series_id)
    ) AS block_names,
    s.created_at,
    s.updated_at
FROM series s
WHERE s.is_active = TRUE;

COMMENT ON VIEW series_with_blocks IS 'Active series with their block information from the most recent config (from 002, ensured current)';

-- ============================================================================
-- PART 4: Initial Data Cleanup (Run once)
-- ============================================================================

-- Cleanup old factors (keeps only latest 3 models)
-- This will delete old factors and cascade to factor_values and factor_loadings
DO $$
DECLARE
    cleanup_result RECORD;
BEGIN
    SELECT * INTO cleanup_result FROM cleanup_old_factors(3);
    RAISE NOTICE 'Cleanup completed: Deleted % factors, % factor_values, % factor_loadings',
        cleanup_result.deleted_factors,
        cleanup_result.deleted_factor_values,
        cleanup_result.deleted_factor_loadings;
END;
$$;

-- Cleanup old forecasts (keeps only latest 1 per series/date/run_type)
DO $$
DECLARE
    deleted_count INTEGER;
BEGIN
    SELECT cleanup_old_forecasts(1) INTO deleted_count;
    RAISE NOTICE 'Forecast cleanup completed: Deleted % old forecasts', deleted_count;
END;
$$;

-- ============================================================================
-- PART 5: Indexes for Performance
-- ============================================================================

-- Index for faster latest model queries
CREATE INDEX IF NOT EXISTS idx_factors_model_created ON factors(model_id, created_at DESC);

-- Index for faster latest vintage queries
CREATE INDEX IF NOT EXISTS idx_observations_vintage_date ON observations(vintage_id, date);

-- Index for faster latest forecast queries
CREATE INDEX IF NOT EXISTS idx_forecasts_series_date_created ON forecasts(series_id, forecast_date, created_at DESC);

-- Additional indexes for view query optimization
-- Index for latest_factor_values_view: optimize factor_id and date queries
CREATE INDEX IF NOT EXISTS idx_factor_values_factor_date ON factor_values(factor_id, date DESC);

-- Index for latest_observations_view: optimize series_id and date queries
CREATE INDEX IF NOT EXISTS idx_observations_series_date ON observations(series_id, date DESC);

-- Index for latest_factor_loadings_view: optimize series_id lookups in blocks table
CREATE INDEX IF NOT EXISTS idx_blocks_series_config ON blocks(series_id, config_name DESC);

COMMENT ON INDEX idx_factors_model_created IS 'Index for faster queries to find latest model';
COMMENT ON INDEX idx_observations_vintage_date IS 'Index for faster queries to find latest vintage observations';
COMMENT ON INDEX idx_forecasts_series_date_created IS 'Index for faster queries to find latest forecasts';
COMMENT ON INDEX idx_factor_values_factor_date IS 'Index for optimizing latest_factor_values_view queries';
COMMENT ON INDEX idx_observations_series_date IS 'Index for optimizing latest_observations_view queries';
COMMENT ON INDEX idx_blocks_series_config IS 'Index for optimizing block_name lookups in views';

-- ============================================================================
-- PART 6: KPI Series View (Optional but useful for frontend)
-- ============================================================================
-- View for KPI series with latest observation values
-- This provides a convenient way to fetch KPI series metadata and latest values

DROP VIEW IF EXISTS latest_kpi_series_view CASCADE;
CREATE VIEW latest_kpi_series_view
WITH (security_invoker=true) AS
SELECT 
    s.series_id,
    s.series_name,
    s.units,
    s.is_kpi,
    s.frequency,
    s.transformation,
    s.category,
    s.country,
    -- Latest observation value and date
    (
        SELECT value 
        FROM latest_observations_view 
        WHERE series_id = s.series_id 
        ORDER BY date DESC 
        LIMIT 1
    ) AS latest_value,
    (
        SELECT date 
        FROM latest_observations_view 
        WHERE series_id = s.series_id 
        ORDER BY date DESC 
        LIMIT 1
    ) AS latest_date,
    -- Block information
    (SELECT array_agg(DISTINCT b.block_name ORDER BY b.block_name)
     FROM blocks b
     WHERE b.series_id = s.series_id
     AND b.config_name = (SELECT MAX(config_name) FROM blocks WHERE series_id = s.series_id)
    ) AS block_names,
    s.created_at,
    s.updated_at
FROM series s
WHERE s.is_kpi = true
  AND s.is_active = true
ORDER BY s.series_name;

COMMENT ON VIEW latest_kpi_series_view IS 'KPI series with latest observation values and metadata (for frontend dashboard)';

-- ============================================================================
-- PART 7: Additional Utility Views (Optional but useful)
-- ============================================================================
-- Views for aggregated statistics and summaries

-- Block statistics view: summary statistics per block
DROP VIEW IF EXISTS latest_block_stats_view CASCADE;
CREATE VIEW latest_block_stats_view
WITH (security_invoker=true) AS
SELECT 
    f.block_id,
    f.block_name,
    COUNT(DISTINCT f.id) as factor_count,
    COUNT(DISTINCT fv.date) as data_point_count,
    MIN(fv.date) as min_date,
    MAX(fv.date) as max_date,
    AVG(ABS(fv.value)) as avg_abs_value,
    STDDEV(fv.value) as std_value
FROM latest_factors_view f
LEFT JOIN latest_factor_values_view fv ON f.id = fv.factor_id
WHERE f.block_id IS NOT NULL
GROUP BY f.block_id, f.block_name
ORDER BY f.block_id;

COMMENT ON VIEW latest_block_stats_view IS 'Block-level summary statistics (factor count, data points, date range, etc.)';

-- Factor summary view: per-factor statistics
DROP VIEW IF EXISTS latest_factor_summary_view CASCADE;
CREATE VIEW latest_factor_summary_view
WITH (security_invoker=true) AS
SELECT 
    f.id,
    f.name,
    f.description,
    f.factor_index,
    f.block_name,
    f.block_id,
    COUNT(fv.date) as data_point_count,
    MIN(fv.date) as min_date,
    MAX(fv.date) as max_date,
    AVG(fv.value) as avg_value,
    STDDEV(fv.value) as std_value,
    MIN(fv.value) as min_value,
    MAX(fv.value) as max_value
FROM latest_factors_view f
LEFT JOIN latest_factor_values_view fv ON f.id = fv.factor_id
GROUP BY f.id, f.name, f.description, f.factor_index, f.block_name, f.block_id
ORDER BY f.factor_index;

COMMENT ON VIEW latest_factor_summary_view IS 'Per-factor summary statistics (data points, date range, value statistics)';

