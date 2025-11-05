#!/usr/bin/env python3
"""
Generate migration 003 from migration 001 with complete security redefinition.
This creates a new starting point with all security settings correct from the start.
"""

import re

def main():
    # Read 001
    with open('migrations/001_initial_schema.sql', 'r') as f:
        content = f.read()
    
    output = []
    
    # Header
    output.append("""-- ============================================================================
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

""")
    
    # Process content: replace CREATE TABLE IF NOT EXISTS with CREATE TABLE
    content = re.sub(r'CREATE TABLE IF NOT EXISTS', 'CREATE TABLE', content, flags=re.IGNORECASE)
    
    # Find all CREATE TABLE blocks and add RLS enable after each
    # Split by CREATE TABLE to find each table
    table_pattern = r'CREATE TABLE (\w+)\s*\([^)]+\);'
    
    # More complex: find table definitions with proper bracket matching
    lines = content.split('\n')
    i = 0
    in_table = False
    table_lines = []
    current_table = None
    bracket_count = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Detect start of CREATE TABLE
        if re.search(r'CREATE TABLE (\w+)', line, re.IGNORECASE):
            if in_table:
                # Finish previous table
                output.extend(table_lines)
                output.append(f"\n-- Enable RLS immediately after table creation\n")
                output.append(f"ALTER TABLE {current_table} ENABLE ROW LEVEL SECURITY;\n\n")
            
            in_table = True
            match = re.search(r'CREATE TABLE (\w+)', line, re.IGNORECASE)
            current_table = match.group(1)
            table_lines = [line + '\n']
            bracket_count = line.count('(') - line.count(')')
            i += 1
            continue
        
        if in_table:
            table_lines.append(line + '\n')
            bracket_count += line.count('(') - line.count(')')
            
            # Check if table definition is complete
            if bracket_count == 0 and ';' in line:
                # End of table
                output.extend(table_lines)
                output.append(f"\n-- Enable RLS immediately after table creation\n")
                output.append(f"ALTER TABLE {current_table} ENABLE ROW LEVEL SECURITY;\n\n")
                in_table = False
                current_table = None
                table_lines = []
            
            i += 1
            continue
        
        # For views, modify to use security_invoker
        if re.search(r'CREATE (?:OR REPLACE )?VIEW', line, re.IGNORECASE):
            # Extract view name
            match = re.search(r'CREATE (?:OR REPLACE )?VIEW (?:IF NOT EXISTS )?(\w+)', line, re.IGNORECASE)
            if match:
                view_name = match.group(1)
                # Replace with CREATE VIEW ... WITH (security_invoker = true)
                modified_line = re.sub(
                    r'CREATE (?:OR REPLACE )?VIEW (?:IF NOT EXISTS )?(\w+)(\s+AS)',
                    r'CREATE VIEW \1\nWITH (security_invoker = true)\2',
                    line,
                    flags=re.IGNORECASE
                )
                output.append(modified_line + '\n')
            else:
                output.append(line + '\n')
            i += 1
            continue
        
        # For other lines, just copy them
        if not in_table:
            output.append(line + '\n')
        
        i += 1
    
    # Handle any remaining table
    if in_table and table_lines:
        output.extend(table_lines)
        output.append(f"\n-- Enable RLS immediately after table creation\n")
        output.append(f"ALTER TABLE {current_table} ENABLE ROW LEVEL SECURITY;\n\n")
    
    # Write output
    output_text = ''.join(output)
    
    # Write to file
    with open('migrations/003_complete_security_redefinition.sql', 'w') as f:
        f.write(output_text)
    
    print(f"Created migration 003 ({len(output_text.split(chr(10)))} lines)")

if __name__ == '__main__':
    main()

