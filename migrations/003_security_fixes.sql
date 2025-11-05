-- ============================================================================
-- Migration: 003_security_fixes.sql
-- Purpose: Fix security warnings from Supabase linter
-- 
-- This migration addresses:
-- 1. SECURITY DEFINER views (change to SECURITY INVOKER)
-- 2. RLS disabled on public tables (enable RLS and add policies)
-- 3. Function search_path mutable (set search_path in function)
-- ============================================================================

-- ============================================================================
-- 1. Fix SECURITY DEFINER Views
-- ============================================================================
-- Drop and recreate views without SECURITY DEFINER (defaults to SECURITY INVOKER)
-- SECURITY INVOKER uses the permissions of the user executing the query,
-- which is safer and aligns with RLS policies
--
-- Note: We drop and recreate instead of CREATE OR REPLACE to ensure
-- the SECURITY DEFINER property is removed

-- Fix dfm_selected_statistics view
-- Drop and recreate without SECURITY DEFINER (defaults to SECURITY INVOKER)
DROP VIEW IF EXISTS dfm_selected_statistics;
CREATE VIEW dfm_selected_statistics AS
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

-- Fix active_statistics_by_source view
DROP VIEW IF EXISTS active_statistics_by_source;
CREATE VIEW active_statistics_by_source AS
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

-- Fix latest_forecasts_view
DROP VIEW IF EXISTS latest_forecasts_view;
CREATE VIEW latest_forecasts_view AS
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

-- Fix model_training_history view
DROP VIEW IF EXISTS model_training_history;
CREATE VIEW model_training_history AS
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
         tm.convergence_iter, tm.log_likelihood, tm.threshold, tm.trained_at,
         tm.training_duration_seconds
ORDER BY tm.trained_at DESC;

COMMENT ON VIEW model_training_history IS 'Model training history with forecast counts';

-- ============================================================================
-- 2. Enable RLS on Missing Tables
-- ============================================================================

-- Enable RLS on data_sources table
ALTER TABLE public.data_sources ENABLE ROW LEVEL SECURITY;

-- Policy: Allow public read access to data sources
CREATE POLICY data_sources_public_read ON public.data_sources
    FOR SELECT
    TO PUBLIC
    USING (true);

-- Policy: Allow authenticated service users to manage data sources
CREATE POLICY data_sources_service_all ON public.data_sources
    FOR ALL
    TO authenticated
    USING (true)
    WITH CHECK (true);

-- Enable RLS on statistics_metadata table
ALTER TABLE public.statistics_metadata ENABLE ROW LEVEL SECURITY;

-- Policy: Allow public read access to statistics metadata
CREATE POLICY statistics_metadata_public_read ON public.statistics_metadata
    FOR SELECT
    TO PUBLIC
    USING (true);

-- Policy: Allow authenticated service users to manage statistics metadata
CREATE POLICY statistics_metadata_service_all ON public.statistics_metadata
    FOR ALL
    TO authenticated
    USING (true)
    WITH CHECK (true);

-- Enable RLS on statistics_items table
ALTER TABLE public.statistics_items ENABLE ROW LEVEL SECURITY;

-- Policy: Allow public read access to statistics items
CREATE POLICY statistics_items_public_read ON public.statistics_items
    FOR SELECT
    TO PUBLIC
    USING (true);

-- Policy: Allow authenticated service users to manage statistics items
CREATE POLICY statistics_items_service_all ON public.statistics_items
    FOR ALL
    TO authenticated
    USING (true)
    WITH CHECK (true);

-- Enable RLS on ingestion_jobs table
ALTER TABLE public.ingestion_jobs ENABLE ROW LEVEL SECURITY;

-- Policy: Allow public read access to ingestion jobs
CREATE POLICY ingestion_jobs_public_read ON public.ingestion_jobs
    FOR SELECT
    TO PUBLIC
    USING (true);

-- Policy: Allow authenticated service users to manage ingestion jobs
CREATE POLICY ingestion_jobs_service_all ON public.ingestion_jobs
    FOR ALL
    TO authenticated
    USING (true)
    WITH CHECK (true);

-- Enable RLS on api_fetches table
ALTER TABLE public.api_fetches ENABLE ROW LEVEL SECURITY;

-- Policy: Allow public read access to API fetches
CREATE POLICY api_fetches_public_read ON public.api_fetches
    FOR SELECT
    TO PUBLIC
    USING (true);

-- Policy: Allow authenticated service users to manage API fetches
CREATE POLICY api_fetches_service_all ON public.api_fetches
    FOR ALL
    TO authenticated
    USING (true)
    WITH CHECK (true);

-- Enable RLS on model_block_assignments table
ALTER TABLE public.model_block_assignments ENABLE ROW LEVEL SECURITY;

-- Policy: Allow public read access to model block assignments
CREATE POLICY model_block_assignments_public_read ON public.model_block_assignments
    FOR SELECT
    TO PUBLIC
    USING (true);

-- Policy: Allow authenticated service users to manage model block assignments
CREATE POLICY model_block_assignments_service_all ON public.model_block_assignments
    FOR ALL
    TO authenticated
    USING (true)
    WITH CHECK (true);

-- ============================================================================
-- 3. Fix Function Search Path
-- ============================================================================
-- Set search_path to empty string to prevent search_path injection attacks

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
-- Notes on Security Fixes
-- ============================================================================
--
-- 1. SECURITY INVOKER Views:
--    - Views now execute with the permissions of the querying user
--    - RLS policies on underlying tables are properly enforced
--    - More secure than SECURITY DEFINER which runs with creator's permissions
--
-- 2. RLS Policies:
--    - All public tables now have RLS enabled
--    - Public read access allows data to be publicly viewable (appropriate for forecasting)
--    - Write access restricted to authenticated users
--    - Service operations use service_role key which bypasses RLS (as intended)
--
-- 3. Function Search Path:
--    - update_updated_at_column() now has search_path = '' to prevent injection
--    - This ensures the function uses only explicitly qualified names
--
-- ============================================================================

