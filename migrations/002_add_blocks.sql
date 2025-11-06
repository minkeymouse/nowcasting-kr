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
-- ============================================================================

-- ============================================================================
-- Blocks Table (DFM Block Structure)
-- ============================================================================
-- Stores block assignments from spec CSV files
-- Source of truth: spec CSV files (e.g., 001_initial_spec.csv, 002_updated_spec.csv)
-- Each spec version has its own block structure stored independently
CREATE TABLE IF NOT EXISTS blocks (
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

-- Blocks RLS Policies
ALTER TABLE blocks ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow public read access to blocks"
    ON blocks FOR SELECT
    USING (true);
CREATE POLICY "Allow authenticated insert to blocks"
    ON blocks FOR INSERT
    TO authenticated
    WITH CHECK (true);
CREATE POLICY "Allow authenticated update to blocks"
    ON blocks FOR UPDATE
    TO authenticated
    USING (true);
CREATE POLICY "Allow authenticated delete from blocks"
    ON blocks FOR DELETE
    TO authenticated
    USING (true);
