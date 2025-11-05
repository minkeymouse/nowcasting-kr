-- Migration 004: Frontend Integration Schema
-- Adds tables and views needed for the DFM Dashboard frontend
-- This migration creates the factor-centric schema required by the Next.js dashboard
-- This migration is idempotent and can be run multiple times safely

-- ============================================================================
-- DROP EXISTING OBJECTS (for idempotency)
-- ============================================================================

-- Drop triggers first (they depend on tables)
DROP TRIGGER IF EXISTS update_factors_updated_at ON factors CASCADE;
DROP TRIGGER IF EXISTS update_factor_loadings_updated_at ON factor_loadings CASCADE;

-- Drop functions (they might be used by triggers)
-- Note: DROP FUNCTION with no parameters will drop all overloads, but we specify signature for precision
DROP FUNCTION IF EXISTS get_latest_factor_values(INTEGER, INTEGER) CASCADE;
DROP FUNCTION IF EXISTS get_top_loadings(INTEGER, INTEGER) CASCADE;

-- Drop views (they depend on tables)
DROP VIEW IF EXISTS variable_values CASCADE;
DROP VIEW IF EXISTS variables CASCADE;

-- Drop policies (they depend on tables)
-- Drop old policies (if they exist from previous runs)
DROP POLICY IF EXISTS "factors_select" ON factors;
DROP POLICY IF EXISTS "factors_modify" ON factors;
DROP POLICY IF EXISTS "factors_insert" ON factors;
DROP POLICY IF EXISTS "factors_update" ON factors;
DROP POLICY IF EXISTS "factors_delete" ON factors;
DROP POLICY IF EXISTS "factor_values_select" ON factor_values;
DROP POLICY IF EXISTS "factor_values_modify" ON factor_values;
DROP POLICY IF EXISTS "factor_values_insert" ON factor_values;
DROP POLICY IF EXISTS "factor_values_update" ON factor_values;
DROP POLICY IF EXISTS "factor_values_delete" ON factor_values;
DROP POLICY IF EXISTS "factor_loadings_select" ON factor_loadings;
DROP POLICY IF EXISTS "factor_loadings_modify" ON factor_loadings;
DROP POLICY IF EXISTS "factor_loadings_insert" ON factor_loadings;
DROP POLICY IF EXISTS "factor_loadings_update" ON factor_loadings;
DROP POLICY IF EXISTS "factor_loadings_delete" ON factor_loadings;

-- Drop tables (in reverse dependency order)
DROP TABLE IF EXISTS factor_loadings CASCADE;
DROP TABLE IF EXISTS factor_values CASCADE;
DROP TABLE IF EXISTS factors CASCADE;

-- ============================================================================
-- FACTORS TABLE
-- ============================================================================
-- Stores DFM latent factors (the extracted factors from the model)
CREATE TABLE factors (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    factor_index INTEGER NOT NULL, -- Position in factor array (0-indexed or 1-indexed)
    config_id INTEGER REFERENCES model_configs(config_id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(config_id, factor_index) -- One factor per index per config
);

CREATE INDEX idx_factors_config_id ON factors(config_id);
CREATE INDEX idx_factors_factor_index ON factors(factor_index);

COMMENT ON TABLE factors IS 'DFM latent factors extracted from the model';
COMMENT ON COLUMN factors.factor_index IS 'Position of this factor in the factor array (0-based or 1-based)';
COMMENT ON COLUMN factors.config_id IS 'Reference to the model configuration that generated this factor';

-- ============================================================================
-- FACTOR_VALUES TABLE
-- ============================================================================
-- Stores time series of factor estimates (the Z values from DFM)
CREATE TABLE factor_values (
    id SERIAL PRIMARY KEY,
    factor_id INTEGER NOT NULL REFERENCES factors(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    value NUMERIC NOT NULL,
    vintage_id INTEGER REFERENCES data_vintages(vintage_id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(factor_id, date, vintage_id) -- One value per factor per date per vintage
);

CREATE INDEX idx_factor_values_factor_id ON factor_values(factor_id);
CREATE INDEX idx_factor_values_date ON factor_values(date);
CREATE INDEX idx_factor_values_vintage_id ON factor_values(vintage_id);
CREATE INDEX idx_factor_values_factor_date ON factor_values(factor_id, date DESC);

COMMENT ON TABLE factor_values IS 'Time series of factor estimates (DFM factor values over time)';
COMMENT ON COLUMN factor_values.vintage_id IS 'Data vintage snapshot for this factor estimate';

-- ============================================================================
-- VARIABLES TABLE (Frontend View of Series)
-- ============================================================================
-- Maps existing series to variables for frontend consumption
-- Note: We can create a view or materialized view that maps series to variables
-- OR create a separate variables table that references series
-- For MVP, we'll create a view that uses series as variables

-- Create a view that exposes series as variables
-- Note: We'll add is_kpi column to series table if needed, but for MVP use is_active
CREATE OR REPLACE VIEW variables AS
SELECT 
    s.series_id AS id,
    s.series_name AS name,
    COALESCE(s.units, '') AS unit,
    s.category,
    s.frequency,
    s.transformation,
    COALESCE(s.is_active, FALSE) AS is_kpi, -- Use is_active as is_kpi for now
    s.api_source,
    s.api_code,
    s.statistics_metadata_id,
    s.created_at,
    s.last_updated
FROM series s
WHERE s.is_active = TRUE;

COMMENT ON VIEW variables IS 'View of active series exposed as variables for frontend dashboard';

-- Explicitly set security invoker to avoid SECURITY DEFINER warnings
-- This ensures views run with caller's permissions, not creator's permissions
ALTER VIEW variables SET (security_invoker = true);

-- If we want to mark specific variables as KPIs, we can add a column to series table
-- For now, we'll use is_active, but we could add:
-- ALTER TABLE series ADD COLUMN IF NOT EXISTS is_kpi BOOLEAN DEFAULT FALSE;

-- ============================================================================
-- VARIABLE_VALUES TABLE (Frontend View of Observations)
-- ============================================================================
-- Maps existing observations to variable_values for frontend
-- We can use a view that maps observations to variable_values
-- Note: observations table has id SERIAL PRIMARY KEY
CREATE OR REPLACE VIEW variable_values AS
SELECT 
    o.id,
    o.series_id AS variable_id,
    o.date,
    o.value,
    o.vintage_id,
    o.created_at
FROM observations o
WHERE o.series_id IN (SELECT series_id FROM series WHERE is_active = TRUE);

COMMENT ON VIEW variable_values IS 'View of observations exposed as variable_values for frontend dashboard';

-- Explicitly set security invoker to avoid SECURITY DEFINER warnings
-- This ensures views run with caller's permissions, not creator's permissions
ALTER VIEW variable_values SET (security_invoker = true);

-- ============================================================================
-- FACTOR_LOADINGS TABLE
-- ============================================================================
-- Stores the loading matrix (C matrix from DFM) that links factors to variables
-- This represents how each variable loads onto each factor
CREATE TABLE factor_loadings (
    factor_id INTEGER NOT NULL REFERENCES factors(id) ON DELETE CASCADE,
    series_id TEXT NOT NULL REFERENCES series(series_id) ON DELETE CASCADE,
    loading NUMERIC NOT NULL,
    config_id INTEGER REFERENCES model_configs(config_id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (factor_id, series_id, config_id) -- Composite PK
);

CREATE INDEX idx_factor_loadings_factor_id ON factor_loadings(factor_id);
CREATE INDEX idx_factor_loadings_series_id ON factor_loadings(series_id);
CREATE INDEX idx_factor_loadings_config_id ON factor_loadings(config_id);
CREATE INDEX idx_factor_loadings_loading ON factor_loadings(loading DESC); -- For top loadings queries

COMMENT ON TABLE factor_loadings IS 'Factor loading matrix: how each variable (series) loads onto each factor';
COMMENT ON COLUMN factor_loadings.loading IS 'Loading coefficient: strength of relationship between variable and factor';

-- ============================================================================
-- HELPER FUNCTIONS FOR FRONTEND
-- ============================================================================

-- Function to get latest factor values for a given factor
CREATE OR REPLACE FUNCTION get_latest_factor_values(
    p_factor_id INTEGER,
    p_limit INTEGER DEFAULT 100
)
RETURNS TABLE (
    factor_id INTEGER,
    date DATE,
    value NUMERIC
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        fv.factor_id,
        fv.date,
        fv.value
    FROM factor_values fv
    WHERE fv.factor_id = p_factor_id
    ORDER BY fv.date DESC
    LIMIT p_limit;
END;
$$;

COMMENT ON FUNCTION get_latest_factor_values IS 'Get latest factor values for a factor, ordered by date descending';

-- Function to get top loadings for a factor
CREATE OR REPLACE FUNCTION get_top_loadings(
    p_factor_id INTEGER,
    p_limit INTEGER DEFAULT 10
)
RETURNS TABLE (
    series_id TEXT,
    series_name TEXT,
    loading NUMERIC
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        fl.series_id,
        s.series_name,
        fl.loading
    FROM factor_loadings fl
    JOIN series s ON s.series_id = fl.series_id
    WHERE fl.factor_id = p_factor_id
    ORDER BY ABS(fl.loading) DESC
    LIMIT p_limit;
END;
$$;

COMMENT ON FUNCTION get_top_loadings IS 'Get top loadings (by absolute value) for a factor';

-- ============================================================================
-- UPDATED_AT TRIGGERS
-- ============================================================================
-- Note: update_updated_at_column() function is created in migration 003_restart.sql

-- Trigger for factors.updated_at
CREATE TRIGGER update_factors_updated_at
    BEFORE UPDATE ON factors
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Trigger for factor_loadings.updated_at
CREATE TRIGGER update_factor_loadings_updated_at
    BEFORE UPDATE ON factor_loadings
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- ROW LEVEL SECURITY (RLS)
-- ============================================================================

-- Enable RLS on new tables
ALTER TABLE factors ENABLE ROW LEVEL SECURITY;
ALTER TABLE factor_values ENABLE ROW LEVEL SECURITY;
ALTER TABLE factor_loadings ENABLE ROW LEVEL SECURITY;

-- Policy: Allow read access to factors for authenticated and anon users
CREATE POLICY "factors_select" ON factors
    FOR SELECT
    USING (true);

-- Policy: Allow read access to factor_values for authenticated and anon users
CREATE POLICY "factor_values_select" ON factor_values
    FOR SELECT
    USING (true);

-- Policy: Allow read access to factor_loadings for authenticated and anon users
CREATE POLICY "factor_loadings_select" ON factor_loadings
    FOR SELECT
    USING (true);

-- Policy: Only service role can insert/update/delete
-- Note: Split into separate policies for INSERT/UPDATE/DELETE to avoid multiple permissive policies for SELECT
-- Also use (select auth.role()) to avoid initplan performance issues
CREATE POLICY "factors_insert" ON factors
    FOR INSERT
    WITH CHECK ((select auth.role()) = 'service_role');

CREATE POLICY "factors_update" ON factors
    FOR UPDATE
    USING ((select auth.role()) = 'service_role');

CREATE POLICY "factors_delete" ON factors
    FOR DELETE
    USING ((select auth.role()) = 'service_role');

CREATE POLICY "factor_values_insert" ON factor_values
    FOR INSERT
    WITH CHECK ((select auth.role()) = 'service_role');

CREATE POLICY "factor_values_update" ON factor_values
    FOR UPDATE
    USING ((select auth.role()) = 'service_role');

CREATE POLICY "factor_values_delete" ON factor_values
    FOR DELETE
    USING ((select auth.role()) = 'service_role');

CREATE POLICY "factor_loadings_insert" ON factor_loadings
    FOR INSERT
    WITH CHECK ((select auth.role()) = 'service_role');

CREATE POLICY "factor_loadings_update" ON factor_loadings
    FOR UPDATE
    USING ((select auth.role()) = 'service_role');

CREATE POLICY "factor_loadings_delete" ON factor_loadings
    FOR DELETE
    USING ((select auth.role()) = 'service_role');

-- ============================================================================
-- INITIAL DATA POPULATION NOTES
-- ============================================================================
-- This migration creates the schema but does not populate data.
-- Data population should be done by:
-- 1. Extracting factors from model_weights (C matrix) after DFM training
-- 2. Extracting factor values (Z) from model_weights or forecasts
-- 3. Extracting loadings (C matrix columns) from model_weights
--
-- These will be populated by the DFM training/forecasting scripts after model runs.

