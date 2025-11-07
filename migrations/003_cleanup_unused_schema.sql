-- ============================================================================
-- Migration: 003_cleanup_unused_schema.sql
-- Purpose: Remove unused schema elements identified in code review
-- 
-- This migration:
-- 1. Drops legacy series_with_groups view (replaced by series_with_blocks in 002)
-- 2. Keeps model_configs and trained_models tables as optional (handled gracefully in code)
-- 
-- This is an incremental migration that only removes unused elements.
-- It does NOT drop or modify existing tables.
-- This migration is idempotent and can be run multiple times safely.
-- ============================================================================

-- ============================================================================
-- Drop Legacy Views
-- ============================================================================

-- Drop series_with_groups view (replaced by series_with_blocks in migration 002)
DROP VIEW IF EXISTS series_with_groups CASCADE;

COMMENT ON VIEW series_with_groups IS 'DEPRECATED: Replaced by series_with_blocks view in migration 002';

-- Note: variable_values_view is kept as it may be used by frontend
-- Note: model_configs and trained_models tables are not created as they are optional
--       and code handles their absence gracefully

