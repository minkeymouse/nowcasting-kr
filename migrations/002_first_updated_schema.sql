-- ============================================================================
-- Migration: 002_first_updated_schema.sql
-- Purpose: Enable Row-Level Security (RLS) and add security policies
-- 
-- This migration enables RLS on the forecasts table and other critical tables
-- to ensure proper access control. Since this system uses service_role key
-- for backend operations (which bypasses RLS), we enable RLS for public/anon
-- access while allowing authenticated service operations.
-- ============================================================================

-- ============================================================================
-- Row-Level Security for Forecasts Table
-- ============================================================================

-- Enable RLS on forecasts table
ALTER TABLE public.forecasts ENABLE ROW LEVEL SECURITY;

-- Policy: Allow public read access to forecasts
-- Forecasts are intended to be publicly accessible for viewing
CREATE POLICY forecasts_public_read ON public.forecasts
    FOR SELECT
    TO PUBLIC
    USING (true);

-- Policy: Allow authenticated service users to insert forecasts
-- Service operations use service_role key which bypasses RLS, but this
-- provides an additional layer for authenticated API access if needed
CREATE POLICY forecasts_insert_service ON public.forecasts
    FOR INSERT
    TO authenticated
    WITH CHECK (true);

-- Policy: Allow authenticated service users to update forecasts
CREATE POLICY forecasts_update_service ON public.forecasts
    FOR UPDATE
    TO authenticated
    USING (true)
    WITH CHECK (true);

-- Policy: Allow authenticated service users to delete forecasts
CREATE POLICY forecasts_delete_service ON public.forecasts
    FOR DELETE
    TO authenticated
    USING (true);

-- ============================================================================
-- Row-Level Security for Forecast Runs Table
-- ============================================================================

-- Enable RLS on forecast_runs table
ALTER TABLE public.forecast_runs ENABLE ROW LEVEL SECURITY;

-- Policy: Allow public read access to forecast runs
CREATE POLICY forecast_runs_public_read ON public.forecast_runs
    FOR SELECT
    TO PUBLIC
    USING (true);

-- Policy: Allow authenticated service users to manage forecast runs
CREATE POLICY forecast_runs_service_all ON public.forecast_runs
    FOR ALL
    TO authenticated
    USING (true)
    WITH CHECK (true);

-- ============================================================================
-- Row-Level Security for Trained Models Table
-- ============================================================================

-- Enable RLS on trained_models table
ALTER TABLE public.trained_models ENABLE ROW LEVEL SECURITY;

-- Policy: Allow public read access to trained models
CREATE POLICY trained_models_public_read ON public.trained_models
    FOR SELECT
    TO PUBLIC
    USING (true);

-- Policy: Allow authenticated service users to manage trained models
CREATE POLICY trained_models_service_all ON public.trained_models
    FOR ALL
    TO authenticated
    USING (true)
    WITH CHECK (true);

-- ============================================================================
-- Row-Level Security for Model Configs Table
-- ============================================================================

-- Enable RLS on model_configs table
ALTER TABLE public.model_configs ENABLE ROW LEVEL SECURITY;

-- Policy: Allow public read access to model configs
CREATE POLICY model_configs_public_read ON public.model_configs
    FOR SELECT
    TO PUBLIC
    USING (true);

-- Policy: Allow authenticated service users to manage model configs
CREATE POLICY model_configs_service_all ON public.model_configs
    FOR ALL
    TO authenticated
    USING (true)
    WITH CHECK (true);

-- ============================================================================
-- Row-Level Security for Data Vintages Table
-- ============================================================================

-- Enable RLS on data_vintages table
ALTER TABLE public.data_vintages ENABLE ROW LEVEL SECURITY;

-- Policy: Allow public read access to data vintages
CREATE POLICY data_vintages_public_read ON public.data_vintages
    FOR SELECT
    TO PUBLIC
    USING (true);

-- Policy: Allow authenticated service users to manage data vintages
CREATE POLICY data_vintages_service_all ON public.data_vintages
    FOR ALL
    TO authenticated
    USING (true)
    WITH CHECK (true);

-- ============================================================================
-- Row-Level Security for Observations Table
-- ============================================================================

-- Enable RLS on observations table
ALTER TABLE public.observations ENABLE ROW LEVEL SECURITY;

-- Policy: Allow public read access to observations
CREATE POLICY observations_public_read ON public.observations
    FOR SELECT
    TO PUBLIC
    USING (true);

-- Policy: Allow authenticated service users to manage observations
CREATE POLICY observations_service_all ON public.observations
    FOR ALL
    TO authenticated
    USING (true)
    WITH CHECK (true);

-- ============================================================================
-- Row-Level Security for Series Table
-- ============================================================================

-- Enable RLS on series table
ALTER TABLE public.series ENABLE ROW LEVEL SECURITY;

-- Policy: Allow public read access to series metadata
CREATE POLICY series_public_read ON public.series
    FOR SELECT
    TO PUBLIC
    USING (true);

-- Policy: Allow authenticated service users to manage series
CREATE POLICY series_service_all ON public.series
    FOR ALL
    TO authenticated
    USING (true)
    WITH CHECK (true);

-- ============================================================================
-- Notes on RLS Implementation
-- ============================================================================
-- 
-- 1. Service Role Key: The backend system uses service_role key for operations,
--    which bypasses RLS policies. These policies are primarily for public/anon
--    access control.
--
-- 2. Public Read Access: All tables allow public read access since this is a
--    forecasting system where data should be publicly accessible.
--
-- 3. Write Access: Only authenticated users can write. In practice, the service
--    operations use service_role key which bypasses RLS.
--
-- 4. Future Enhancements: If user-specific access is needed, policies can be
--    updated to include user_id or tenant_id checks.
--
-- ============================================================================


