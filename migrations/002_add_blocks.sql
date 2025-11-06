-- ============================================================================
-- Migration: 002_add_blocks.sql
-- Purpose: Add blocks table for DFM block structure management
-- 
-- This migration adds the blocks table to store block assignments from spec CSV files.
-- Blocks are used to organize series into logical groups for DFM model training.
-- Source of truth: spec CSV files (e.g., 001_initial_spec.csv, 002_updated_spec.csv)
-- 
-- This is an incremental migration that only adds the blocks table.
-- It does NOT drop or modify existing tables.
-- This migration is idempotent and can be run multiple times safely.
-- ============================================================================

-- ============================================================================
-- Blocks Table (DFM Block Structure)
-- ============================================================================
-- Stores block assignments from spec CSV files
-- Source of truth: spec CSV files (e.g., 001_initial_spec.csv, 002_updated_spec.csv)
-- Each spec version has its own block structure stored independently

-- Drop blocks table if exists (for idempotency)
DROP TABLE IF EXISTS blocks CASCADE;

CREATE TABLE blocks (
    config_name VARCHAR(200) NOT NULL,  -- Spec version identifier (e.g., '001-initial-spec', '002-updated-spec')
    series_id VARCHAR(100) NOT NULL REFERENCES series(series_id) ON DELETE CASCADE,
    block_name VARCHAR(50) NOT NULL,    -- Block name (e.g., 'Global', 'Invest', 'Extern')
    series_order INTEGER NOT NULL,      -- Series order in the spec CSV (important for DFM training)
    created_at TIMESTAMP DEFAULT NOW(),
    
    PRIMARY KEY (config_name, series_id, block_name)
);

CREATE INDEX idx_blocks_config ON blocks(config_name);
CREATE INDEX idx_blocks_series ON blocks(series_id);
CREATE INDEX idx_blocks_config_order ON blocks(config_name, series_order);
CREATE INDEX idx_blocks_config_block ON blocks(config_name, block_name);

COMMENT ON TABLE blocks IS 'Block assignments from spec CSV files for DFM model structure';
COMMENT ON COLUMN blocks.config_name IS 'Spec version identifier derived from CSV filename (e.g., 001_initial_spec.csv → 001-initial-spec)';
COMMENT ON COLUMN blocks.series_id IS 'Series identifier';
COMMENT ON COLUMN blocks.block_name IS 'Block name (e.g., Global, Invest, Extern)';
COMMENT ON COLUMN blocks.series_order IS 'Series order in the spec CSV (row index, important for DFM training)';

-- Blocks RLS Policies (idempotent: drop if exists, then create)
ALTER TABLE blocks ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Allow public read access to blocks" ON blocks;
CREATE POLICY "Allow public read access to blocks"
    ON blocks FOR SELECT
    USING (true);

DROP POLICY IF EXISTS "Allow authenticated insert to blocks" ON blocks;
CREATE POLICY "Allow authenticated insert to blocks"
    ON blocks FOR INSERT
    TO authenticated
    WITH CHECK (true);

DROP POLICY IF EXISTS "Allow authenticated update to blocks" ON blocks;
CREATE POLICY "Allow authenticated update to blocks"
    ON blocks FOR UPDATE
    TO authenticated
    USING (true);

DROP POLICY IF EXISTS "Allow authenticated delete from blocks" ON blocks;
CREATE POLICY "Allow authenticated delete from blocks"
    ON blocks FOR DELETE
    TO authenticated
    USING (true);

-- ============================================================================
-- Add block_name to factors table
-- ============================================================================
-- DFM factors are organized by blocks:
-- - Global factor: block_name = NULL or 'Global' (applies to all series)
-- - Inner block factors: block_name = 'Invest', 'Extern', etc. (applies only to series in that block)

-- Add block_name column if not exists (idempotent)
ALTER TABLE factors ADD COLUMN IF NOT EXISTS block_name VARCHAR(50) NULL;

-- Create indexes if not exists (idempotent)
CREATE INDEX IF NOT EXISTS idx_factors_block_name ON factors(block_name);
CREATE INDEX IF NOT EXISTS idx_factors_model_block ON factors(model_id, block_name);

-- Update comment (idempotent - COMMENT ON COLUMN can be run multiple times)
COMMENT ON COLUMN factors.block_name IS 'Block name for inner block factors (NULL or Global for global factors)';

-- ============================================================================
-- Update views to include block information
-- ============================================================================

-- Drop existing views before recreating (for idempotency and column changes)
DROP VIEW IF EXISTS latest_forecasts_view CASCADE;
DROP VIEW IF EXISTS series_with_blocks CASCADE;
DROP VIEW IF EXISTS variables_view CASCADE;

-- Update latest_forecasts_view to include block information
-- Uses the most recent config_name for each series
CREATE VIEW latest_forecasts_view
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
    -- Block information from most recent config
    (SELECT array_agg(DISTINCT b.block_name ORDER BY b.block_name)
     FROM blocks b
     WHERE b.series_id = f.series_id
     AND b.config_name = (SELECT MAX(config_name) FROM blocks WHERE series_id = f.series_id)
    ) AS block_names,
    f.created_at
FROM forecasts f
JOIN series s ON f.series_id = s.series_id
ORDER BY f.series_id, f.forecast_date, f.created_at DESC;

COMMENT ON VIEW latest_forecasts_view IS 'Latest forecast for each series and date combination with block information';

-- Create series_with_blocks view for easy access to block information
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

COMMENT ON VIEW series_with_blocks IS 'Active series with their block information from the most recent config';

-- Update variables_view to include block information
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

COMMENT ON VIEW variables_view IS 'Variables view for frontend visualization with block information';
