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
-- Note: Using CASCADE to ensure all dependencies are dropped cleanly,
-- then recreating views with explicit security_invoker = true (PostgreSQL 15+)
-- This ensures views execute with the permissions of the querying user,
-- not the view creator, which properly enforces RLS policies

-- Fix dfm_selected_statistics view
-- Drop with CASCADE to remove all dependencies, then recreate cleanly
-- Explicitly set security_invoker = true (PostgreSQL 15+)
DROP VIEW IF EXISTS dfm_selected_statistics CASCADE;
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

-- Fix active_statistics_by_source view
-- Drop with CASCADE to remove all dependencies, then recreate cleanly
-- Explicitly set security_invoker = true (PostgreSQL 15+)
DROP VIEW IF EXISTS active_statistics_by_source CASCADE;
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

-- Fix latest_forecasts_view
-- Drop with CASCADE to remove all dependencies, then recreate cleanly
-- Explicitly set security_invoker = true (PostgreSQL 15+)
DROP VIEW IF EXISTS latest_forecasts_view CASCADE;
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

-- Fix model_training_history view
-- Drop with CASCADE to remove all dependencies, then recreate cleanly
-- Explicitly set security_invoker = true (PostgreSQL 15+)
DROP VIEW IF EXISTS model_training_history CASCADE;
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
         tm.convergence_iter, tm.log_likelihood, tm.threshold, tm.trained_at,
         tm.training_duration_seconds
ORDER BY tm.trained_at DESC;

COMMENT ON VIEW model_training_history IS 'Model training history with forecast counts';

-- ============================================================================
-- 2. Enable RLS on Missing Tables
-- ============================================================================

-- Enable RLS on data_sources table
ALTER TABLE public.data_sources ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist (for idempotency)
DROP POLICY IF EXISTS data_sources_public_read ON public.data_sources;
DROP POLICY IF EXISTS data_sources_insert_service ON public.data_sources;
DROP POLICY IF EXISTS data_sources_update_service ON public.data_sources;
DROP POLICY IF EXISTS data_sources_delete_service ON public.data_sources;

-- Policy: Allow public read access to data sources
CREATE POLICY data_sources_public_read ON public.data_sources
    FOR SELECT
    TO PUBLIC
    USING (true);

-- Policy: Allow authenticated service users to insert data sources
-- Note: Separated from SELECT to avoid multiple permissive policies warning
CREATE POLICY data_sources_insert_service ON public.data_sources
    FOR INSERT
    TO authenticated
    WITH CHECK (true);

-- Policy: Allow authenticated service users to update data sources
CREATE POLICY data_sources_update_service ON public.data_sources
    FOR UPDATE
    TO authenticated
    USING (true)
    WITH CHECK (true);

-- Policy: Allow authenticated service users to delete data sources
CREATE POLICY data_sources_delete_service ON public.data_sources
    FOR DELETE
    TO authenticated
    USING (true);

-- Enable RLS on statistics_metadata table
ALTER TABLE public.statistics_metadata ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist (for idempotency)
DROP POLICY IF EXISTS statistics_metadata_public_read ON public.statistics_metadata;
DROP POLICY IF EXISTS statistics_metadata_insert_service ON public.statistics_metadata;
DROP POLICY IF EXISTS statistics_metadata_update_service ON public.statistics_metadata;
DROP POLICY IF EXISTS statistics_metadata_delete_service ON public.statistics_metadata;

-- Policy: Allow public read access to statistics metadata
CREATE POLICY statistics_metadata_public_read ON public.statistics_metadata
    FOR SELECT
    TO PUBLIC
    USING (true);

-- Policy: Allow authenticated service users to insert statistics metadata
-- Note: Separated from SELECT to avoid multiple permissive policies warning
CREATE POLICY statistics_metadata_insert_service ON public.statistics_metadata
    FOR INSERT
    TO authenticated
    WITH CHECK (true);

-- Policy: Allow authenticated service users to update statistics metadata
CREATE POLICY statistics_metadata_update_service ON public.statistics_metadata
    FOR UPDATE
    TO authenticated
    USING (true)
    WITH CHECK (true);

-- Policy: Allow authenticated service users to delete statistics metadata
CREATE POLICY statistics_metadata_delete_service ON public.statistics_metadata
    FOR DELETE
    TO authenticated
    USING (true);

-- Enable RLS on statistics_items table
ALTER TABLE public.statistics_items ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist (for idempotency)
DROP POLICY IF EXISTS statistics_items_public_read ON public.statistics_items;
DROP POLICY IF EXISTS statistics_items_insert_service ON public.statistics_items;
DROP POLICY IF EXISTS statistics_items_update_service ON public.statistics_items;
DROP POLICY IF EXISTS statistics_items_delete_service ON public.statistics_items;

-- Policy: Allow public read access to statistics items
CREATE POLICY statistics_items_public_read ON public.statistics_items
    FOR SELECT
    TO PUBLIC
    USING (true);

-- Policy: Allow authenticated service users to insert statistics items
-- Note: Separated from SELECT to avoid multiple permissive policies warning
CREATE POLICY statistics_items_insert_service ON public.statistics_items
    FOR INSERT
    TO authenticated
    WITH CHECK (true);

-- Policy: Allow authenticated service users to update statistics items
CREATE POLICY statistics_items_update_service ON public.statistics_items
    FOR UPDATE
    TO authenticated
    USING (true)
    WITH CHECK (true);

-- Policy: Allow authenticated service users to delete statistics items
CREATE POLICY statistics_items_delete_service ON public.statistics_items
    FOR DELETE
    TO authenticated
    USING (true);

-- Enable RLS on ingestion_jobs table
ALTER TABLE public.ingestion_jobs ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist (for idempotency)
DROP POLICY IF EXISTS ingestion_jobs_public_read ON public.ingestion_jobs;
DROP POLICY IF EXISTS ingestion_jobs_insert_service ON public.ingestion_jobs;
DROP POLICY IF EXISTS ingestion_jobs_update_service ON public.ingestion_jobs;
DROP POLICY IF EXISTS ingestion_jobs_delete_service ON public.ingestion_jobs;

-- Policy: Allow public read access to ingestion jobs
CREATE POLICY ingestion_jobs_public_read ON public.ingestion_jobs
    FOR SELECT
    TO PUBLIC
    USING (true);

-- Policy: Allow authenticated service users to insert ingestion jobs
-- Note: Separated from SELECT to avoid multiple permissive policies warning
CREATE POLICY ingestion_jobs_insert_service ON public.ingestion_jobs
    FOR INSERT
    TO authenticated
    WITH CHECK (true);

-- Policy: Allow authenticated service users to update ingestion jobs
CREATE POLICY ingestion_jobs_update_service ON public.ingestion_jobs
    FOR UPDATE
    TO authenticated
    USING (true)
    WITH CHECK (true);

-- Policy: Allow authenticated service users to delete ingestion jobs
CREATE POLICY ingestion_jobs_delete_service ON public.ingestion_jobs
    FOR DELETE
    TO authenticated
    USING (true);

-- Enable RLS on api_fetches table
ALTER TABLE public.api_fetches ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist (for idempotency)
DROP POLICY IF EXISTS api_fetches_public_read ON public.api_fetches;
DROP POLICY IF EXISTS api_fetches_insert_service ON public.api_fetches;
DROP POLICY IF EXISTS api_fetches_update_service ON public.api_fetches;
DROP POLICY IF EXISTS api_fetches_delete_service ON public.api_fetches;

-- Policy: Allow public read access to API fetches
CREATE POLICY api_fetches_public_read ON public.api_fetches
    FOR SELECT
    TO PUBLIC
    USING (true);

-- Policy: Allow authenticated service users to insert API fetches
-- Note: Separated from SELECT to avoid multiple permissive policies warning
CREATE POLICY api_fetches_insert_service ON public.api_fetches
    FOR INSERT
    TO authenticated
    WITH CHECK (true);

-- Policy: Allow authenticated service users to update API fetches
CREATE POLICY api_fetches_update_service ON public.api_fetches
    FOR UPDATE
    TO authenticated
    USING (true)
    WITH CHECK (true);

-- Policy: Allow authenticated service users to delete API fetches
CREATE POLICY api_fetches_delete_service ON public.api_fetches
    FOR DELETE
    TO authenticated
    USING (true);

-- Enable RLS on model_block_assignments table
ALTER TABLE public.model_block_assignments ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist (for idempotency)
DROP POLICY IF EXISTS model_block_assignments_public_read ON public.model_block_assignments;
DROP POLICY IF EXISTS model_block_assignments_insert_service ON public.model_block_assignments;
DROP POLICY IF EXISTS model_block_assignments_update_service ON public.model_block_assignments;
DROP POLICY IF EXISTS model_block_assignments_delete_service ON public.model_block_assignments;

-- Policy: Allow public read access to model block assignments
CREATE POLICY model_block_assignments_public_read ON public.model_block_assignments
    FOR SELECT
    TO PUBLIC
    USING (true);

-- Policy: Allow authenticated service users to insert model block assignments
-- Note: Separated from SELECT to avoid multiple permissive policies warning
CREATE POLICY model_block_assignments_insert_service ON public.model_block_assignments
    FOR INSERT
    TO authenticated
    WITH CHECK (true);

-- Policy: Allow authenticated service users to update model block assignments
CREATE POLICY model_block_assignments_update_service ON public.model_block_assignments
    FOR UPDATE
    TO authenticated
    USING (true)
    WITH CHECK (true);

-- Policy: Allow authenticated service users to delete model block assignments
CREATE POLICY model_block_assignments_delete_service ON public.model_block_assignments
    FOR DELETE
    TO authenticated
    USING (true);

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
--    - Policies are separated by operation (SELECT, INSERT, UPDATE, DELETE) to avoid
--      multiple permissive policies warning for authenticated users
--    - Each table has: 1 SELECT policy (PUBLIC) + 3 write policies (authenticated)
--    - Service operations use service_role key which bypasses RLS (as intended)
--
-- 3. Function Search Path:
--    - update_updated_at_column() now has search_path = '' to prevent injection
--    - This ensures the function uses only explicitly qualified names
--
-- ============================================================================

