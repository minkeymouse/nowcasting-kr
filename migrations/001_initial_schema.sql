-- ============================================================================
-- Initial Schema for Nowcasting Project
-- Supports multiple data sources: BOK, KOSIS, and future sources
-- ============================================================================

-- ============================================================================
-- 1. Data Sources Table
-- ============================================================================
-- Stores metadata about data sources (BOK, KOSIS, etc.)
CREATE TABLE IF NOT EXISTS data_sources (
    id SERIAL PRIMARY KEY,
    source_code VARCHAR(20) NOT NULL UNIQUE,  -- BOK, KOSIS, etc.
    source_name VARCHAR(100) NOT NULL,  -- 한국은행, 통계청 등
    source_name_eng VARCHAR(100),  -- Bank of Korea, Statistics Korea
    api_base_url VARCHAR(500),  -- API base URL
    api_documentation_url VARCHAR(500),  -- API documentation URL
    is_active BOOLEAN DEFAULT TRUE,
    description TEXT,
    metadata JSONB,  -- Flexible metadata storage
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT idx_source_code UNIQUE (source_code)
);

COMMENT ON TABLE data_sources IS 'Data source metadata (BOK, KOSIS, etc.)';
COMMENT ON COLUMN data_sources.source_code IS 'Source identifier: BOK, KOSIS, etc.';
COMMENT ON COLUMN data_sources.metadata IS 'Flexible JSON storage for source-specific metadata';

-- ============================================================================
-- 2. Statistics Metadata Table (Unified)
-- ============================================================================
-- Unified table for all statistics metadata from different sources
CREATE TABLE IF NOT EXISTS statistics_metadata (
    id SERIAL PRIMARY KEY,
    
    -- Source reference
    source_id INTEGER NOT NULL REFERENCES data_sources(id) ON DELETE CASCADE,
    
    -- Source-specific identifiers
    source_stat_code VARCHAR(100) NOT NULL,  -- BOK: stat_code, KOSIS: stat_code, etc.
    source_stat_name VARCHAR(500),  -- Original name from source
    source_stat_name_eng VARCHAR(500),  -- English name if available
    
    -- Standardized fields
    cycle VARCHAR(10),  -- Frequency: A, S, Q, M, SM, D (BOK), Y, H, Q, M, D (KOSIS)
    frequency_code VARCHAR(10),  -- Normalized: annual, semi_annual, quarterly, monthly, daily
    
    -- Source-specific metadata
    org_name VARCHAR(200),  -- Organization name
    is_searchable BOOLEAN DEFAULT FALSE,  -- Whether data can be fetched via API
    
    -- Hierarchy (for categorized statistics)
    parent_stat_code VARCHAR(100),  -- Parent statistic code
    parent_item_code VARCHAR(100),  -- Parent item code
    hierarchy_level INTEGER,  -- Depth in hierarchy
    
    -- Source-specific raw metadata (JSONB for flexibility)
    source_metadata JSONB,  -- Store source-specific fields (e.g., p_cycle, p_stat_code for BOK)
    
    -- DFM Selection
    is_dfm_selected BOOLEAN DEFAULT FALSE,
    dfm_priority INTEGER,  -- Priority/rank for DFM (1-50, NULL if not selected)
    dfm_selected_at TIMESTAMP,
    dfm_selection_reason TEXT,  -- Why this statistic was selected
    
    -- Data Collection Status
    is_active BOOLEAN DEFAULT TRUE,  -- Whether to collect data for this statistic
    last_data_fetch_date DATE,  -- Last successful data fetch
    last_data_fetch_status VARCHAR(20),  -- success, failed, partial
    data_start_date DATE,  -- Earliest available data date
    data_end_date DATE,  -- Latest available data date
    total_observations INTEGER,  -- Total number of observations collected
    last_observation_date DATE,  -- Date of last observation
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT unique_source_stat_code UNIQUE (source_id, source_stat_code)
);

-- Indexes for statistics_metadata
CREATE INDEX idx_statistics_metadata_source_id ON statistics_metadata(source_id);
CREATE INDEX idx_statistics_metadata_source_stat_code ON statistics_metadata(source_stat_code);
CREATE INDEX idx_statistics_metadata_cycle ON statistics_metadata(cycle);
CREATE INDEX idx_statistics_metadata_frequency_code ON statistics_metadata(frequency_code);
CREATE INDEX idx_statistics_metadata_dfm_selected ON statistics_metadata(is_dfm_selected) WHERE is_dfm_selected = TRUE;
CREATE INDEX idx_statistics_metadata_active ON statistics_metadata(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_statistics_metadata_searchable ON statistics_metadata(is_searchable) WHERE is_searchable = TRUE;

-- JSONB indexes for source_metadata queries
CREATE INDEX idx_statistics_metadata_source_metadata ON statistics_metadata USING GIN (source_metadata);

COMMENT ON TABLE statistics_metadata IS 'Unified statistics metadata from all sources (BOK, KOSIS, etc.)';
COMMENT ON COLUMN statistics_metadata.source_stat_code IS 'Source-specific statistic code (e.g., BOK stat_code, KOSIS stat_code)';
COMMENT ON COLUMN statistics_metadata.frequency_code IS 'Normalized frequency: annual, semi_annual, quarterly, monthly, daily';
COMMENT ON COLUMN statistics_metadata.source_metadata IS 'Source-specific fields stored as JSON (e.g., BOK p_cycle, p_stat_code)';
COMMENT ON COLUMN statistics_metadata.is_dfm_selected IS 'Whether this statistic is selected for DFM nowcasting';
COMMENT ON COLUMN statistics_metadata.dfm_priority IS 'Priority rank for DFM (1-50)';

-- ============================================================================
-- 3. Statistics Items Table
-- ============================================================================
-- Stores available items for each statistic from StatisticItemList API
CREATE TABLE IF NOT EXISTS statistics_items (
    id SERIAL PRIMARY KEY,
    
    -- Reference to statistics metadata
    statistics_metadata_id INTEGER NOT NULL REFERENCES statistics_metadata(id) ON DELETE CASCADE,
    
    -- Item identification
    item_code VARCHAR(50) NOT NULL,  -- Item code (e.g., "1101", "*AA")
    item_name VARCHAR(500),  -- Item name in Korean
    item_name_eng VARCHAR(500),  -- Item name in English
    
    -- Hierarchy
    parent_item_code VARCHAR(50),  -- Parent item code for hierarchical structure
    parent_item_name VARCHAR(500),  -- Parent item name
    
    -- Grouping
    grp_code VARCHAR(50),  -- Group code (e.g., "Group1")
    grp_name VARCHAR(200),  -- Group name (e.g., "계정항목")
    
    -- Frequency and date range
    cycle VARCHAR(10) NOT NULL,  -- Frequency: A, S, Q, M, SM, D
    start_time VARCHAR(20),  -- Start time as string (e.g., "1960Q1", "202401")
    end_time VARCHAR(20),  -- End time as string (e.g., "2024Q4", "202412")
    
    -- Data information
    data_count INTEGER,  -- Number of available data points
    unit_name VARCHAR(50),  -- Unit name (e.g., "십억원", "2020=100")
    weight VARCHAR(50),  -- Weight value if applicable
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,  -- Whether this item is active for data collection
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT unique_stat_item_cycle UNIQUE (statistics_metadata_id, item_code, cycle)
);

-- Indexes for statistics_items
CREATE INDEX idx_statistics_items_metadata_id ON statistics_items(statistics_metadata_id);
CREATE INDEX idx_statistics_items_item_code ON statistics_items(item_code);
CREATE INDEX idx_statistics_items_cycle ON statistics_items(cycle);
CREATE INDEX idx_statistics_items_active ON statistics_items(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_statistics_items_parent ON statistics_items(parent_item_code) WHERE parent_item_code IS NOT NULL;

COMMENT ON TABLE statistics_items IS 'Available items for each statistic from StatisticItemList API';
COMMENT ON COLUMN statistics_items.item_code IS 'Item code (e.g., "1101" for GDP sectors, "*AA" for total index)';
COMMENT ON COLUMN statistics_items.cycle IS 'Frequency for which this item is available (A, S, Q, M, SM, D)';
COMMENT ON COLUMN statistics_items.start_time IS 'Start time as string matching cycle format (e.g., "2024Q1" for quarterly)';
COMMENT ON COLUMN statistics_items.end_time IS 'End time as string matching cycle format';

-- ============================================================================
-- 4. Series Table (for actual time-series data)
-- ============================================================================
-- Links to statistics_metadata and stores additional series metadata
CREATE TABLE IF NOT EXISTS series (
    series_id VARCHAR(100) PRIMARY KEY,  -- Unique identifier (e.g., BOK_200Y101, KOSIS_101Y001)
    series_name VARCHAR(500) NOT NULL,
    
    -- Reference to statistics metadata
    statistics_metadata_id INTEGER REFERENCES statistics_metadata(id) ON DELETE SET NULL,
    
    -- Standardized fields
    frequency VARCHAR(10) NOT NULL,  -- d, w, m, q, sa, a (for DFM compatibility)
    units VARCHAR(50),  -- Original units (KRW_BIL, PCT, etc.)
    transformation VARCHAR(50),  -- Transformation code (log, diff, etc.)
    category VARCHAR(100),  -- Category classification
    
    -- Source information
    api_source VARCHAR(20) NOT NULL,  -- BOK, KOSIS, MANUAL
    api_code VARCHAR(100),  -- Source-specific code (matches source_stat_code)
    
    -- Item code (for item-based series)
    item_code VARCHAR(50),  -- Item code for this series (used when series represents a specific item from a statistic)
    
    -- Additional metadata
    description TEXT,
    metadata JSONB,  -- Additional flexible metadata
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    last_updated TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT fk_statistics_metadata FOREIGN KEY (statistics_metadata_id) 
        REFERENCES statistics_metadata(id) ON DELETE SET NULL
);

CREATE INDEX idx_series_statistics_metadata_id ON series(statistics_metadata_id);
CREATE INDEX idx_series_api_source ON series(api_source);
CREATE INDEX idx_series_api_code ON series(api_code);
CREATE INDEX idx_series_frequency ON series(frequency);
CREATE INDEX idx_series_active ON series(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_series_item_code ON series(item_code) WHERE item_code IS NOT NULL;

COMMENT ON TABLE series IS 'Time-series metadata for actual data collection';
COMMENT ON COLUMN series.series_id IS 'Unique identifier (format: SOURCE_STATCODE)';
COMMENT ON COLUMN series.statistics_metadata_id IS 'Reference to statistics_metadata table';
COMMENT ON COLUMN series.api_code IS 'Source-specific code (links to statistics_metadata.source_stat_code)';
COMMENT ON COLUMN series.item_code IS 'Item code for this series (used when series represents a specific item from a statistic)';

-- ============================================================================
-- 5. Data Vintages Table
-- ============================================================================
CREATE TABLE IF NOT EXISTS data_vintages (
    vintage_id SERIAL PRIMARY KEY,
    vintage_date DATE NOT NULL,
    country VARCHAR(10) DEFAULT 'KR',
    description TEXT,
    
    -- Status tracking
    fetch_status VARCHAR(20) DEFAULT 'pending',  -- pending, in_progress, completed, failed
    fetch_started_at TIMESTAMP,
    fetch_completed_at TIMESTAMP,
    
    -- GitHub Actions tracking
    github_run_id VARCHAR(100),
    github_workflow_run_url VARCHAR(500),
    
    -- Error handling
    error_message TEXT,
    
    created_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT unique_vintage_date_country UNIQUE (vintage_date, country)
);

CREATE INDEX idx_data_vintages_vintage_date ON data_vintages(vintage_date);
CREATE INDEX idx_data_vintages_fetch_status ON data_vintages(fetch_status);
CREATE INDEX idx_data_vintages_country ON data_vintages(country);

COMMENT ON TABLE data_vintages IS 'Data vintage snapshots for weekly data collection';

-- ============================================================================
-- 6. Ingestion Jobs Table
-- ============================================================================
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    job_id SERIAL PRIMARY KEY,
    github_run_id VARCHAR(100) NOT NULL,
    github_workflow_run_url VARCHAR(500),
    vintage_date DATE NOT NULL,
    
    -- Job status
    status VARCHAR(20) DEFAULT 'pending',  -- pending, running, completed, failed, cancelled
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    
    -- Statistics
    total_series INTEGER,
    successful_series INTEGER,
    failed_series INTEGER,
    
    -- Error handling
    error_message TEXT,
    logs_json JSONB,  -- Detailed logs
    
    created_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT fk_vintage_date FOREIGN KEY (vintage_date, 'KR') 
        REFERENCES data_vintages(vintage_date, country)
);

CREATE INDEX idx_ingestion_jobs_github_run_id ON ingestion_jobs(github_run_id);
CREATE INDEX idx_ingestion_jobs_vintage_date ON ingestion_jobs(vintage_date);
CREATE INDEX idx_ingestion_jobs_status ON ingestion_jobs(status);

COMMENT ON TABLE ingestion_jobs IS 'GitHub Actions ingestion job tracking';

-- ============================================================================
-- 7. Observations Table
-- ============================================================================
CREATE TABLE IF NOT EXISTS observations (
    id SERIAL PRIMARY KEY,
    series_id VARCHAR(100) NOT NULL,
    vintage_id INTEGER NOT NULL,
    date DATE NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    
    -- Job tracking
    job_id INTEGER REFERENCES ingestion_jobs(job_id) ON DELETE SET NULL,
    
    -- Metadata
    is_forecast BOOLEAN DEFAULT FALSE,
    api_source VARCHAR(20),
    
    -- Item codes (hierarchical structure from BOK API)
    item_code1 VARCHAR(50),  -- First level item code from BOK API (ITEM_CODE1)
    item_code2 VARCHAR(50),  -- Second level item code from BOK API (ITEM_CODE2)
    item_code3 VARCHAR(50),  -- Third level item code from BOK API (ITEM_CODE3)
    item_code4 VARCHAR(50),  -- Fourth level item code from BOK API (ITEM_CODE4)
    
    -- Item names (hierarchical structure from BOK API)
    item_name1 VARCHAR(500),  -- First level item name from BOK API (ITEM_NAME1)
    item_name2 VARCHAR(500),  -- Second level item name from BOK API (ITEM_NAME2)
    item_name3 VARCHAR(500),  -- Third level item name from BOK API (ITEM_NAME3)
    item_name4 VARCHAR(500),  -- Fourth level item name from BOK API (ITEM_NAME4)
    
    -- Weight
    weight DOUBLE PRECISION,  -- Weight value from BOK API (WGT field)
    
    created_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT fk_series FOREIGN KEY (series_id) REFERENCES series(series_id) ON DELETE CASCADE,
    CONSTRAINT fk_vintage FOREIGN KEY (vintage_id) REFERENCES data_vintages(vintage_id) ON DELETE CASCADE,
    CONSTRAINT unique_series_vintage_date UNIQUE (series_id, vintage_id, date)
);

CREATE INDEX idx_observations_series_id ON observations(series_id);
CREATE INDEX idx_observations_vintage_id ON observations(vintage_id);
CREATE INDEX idx_observations_date ON observations(date);
CREATE INDEX idx_observations_series_vintage ON observations(series_id, vintage_id);
CREATE INDEX idx_observations_job_id ON observations(job_id);
CREATE INDEX idx_observations_item_code1 ON observations(item_code1) WHERE item_code1 IS NOT NULL;
CREATE INDEX idx_observations_item_code2 ON observations(item_code2) WHERE item_code2 IS NOT NULL;
CREATE INDEX idx_observations_item_codes ON observations(item_code1, item_code2) WHERE item_code1 IS NOT NULL;
CREATE INDEX idx_observations_item_name1 ON observations(item_name1) WHERE item_name1 IS NOT NULL;
CREATE INDEX idx_observations_item_name2 ON observations(item_name2) WHERE item_name2 IS NOT NULL;
CREATE INDEX idx_observations_weight ON observations(weight) WHERE weight IS NOT NULL;
CREATE INDEX idx_observations_item_code_name ON observations(item_code1, item_name1) WHERE item_code1 IS NOT NULL AND item_name1 IS NOT NULL;

COMMENT ON TABLE observations IS 'Time-series observations data';
COMMENT ON COLUMN observations.item_code1 IS 'First level item code from BOK API (ITEM_CODE1)';
COMMENT ON COLUMN observations.item_code2 IS 'Second level item code from BOK API (ITEM_CODE2)';
COMMENT ON COLUMN observations.item_code3 IS 'Third level item code from BOK API (ITEM_CODE3)';
COMMENT ON COLUMN observations.item_code4 IS 'Fourth level item code from BOK API (ITEM_CODE4)';
COMMENT ON COLUMN observations.item_name1 IS 'First level item name from BOK API (ITEM_NAME1)';
COMMENT ON COLUMN observations.item_name2 IS 'Second level item name from BOK API (ITEM_NAME2)';
COMMENT ON COLUMN observations.item_name3 IS 'Third level item name from BOK API (ITEM_NAME3)';
COMMENT ON COLUMN observations.item_name4 IS 'Fourth level item name from BOK API (ITEM_NAME4)';
COMMENT ON COLUMN observations.weight IS 'Weight value from BOK API (WGT field)';

-- ============================================================================
-- 8. API Fetches Table (for tracking API calls)
-- ============================================================================
CREATE TABLE IF NOT EXISTS api_fetches (
    id SERIAL PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES data_sources(id) ON DELETE CASCADE,
    source_stat_code VARCHAR(100) NOT NULL,
    fetch_date DATE NOT NULL,
    fetch_status VARCHAR(20) DEFAULT 'pending',  -- pending, success, failed
    fetch_started_at TIMESTAMP,
    fetch_completed_at TIMESTAMP,
    records_fetched INTEGER,
    error_message TEXT,
    metadata JSONB,
    
    created_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT fk_source_stat FOREIGN KEY (source_id, source_stat_code) 
        REFERENCES statistics_metadata(source_id, source_stat_code) ON DELETE CASCADE
);

CREATE INDEX idx_api_fetches_source_id ON api_fetches(source_id);
CREATE INDEX idx_api_fetches_source_stat_code ON api_fetches(source_stat_code);
CREATE INDEX idx_api_fetches_fetch_date ON api_fetches(fetch_date);
CREATE INDEX idx_api_fetches_status ON api_fetches(fetch_status);

COMMENT ON TABLE api_fetches IS 'API fetch tracking for each statistic';

-- ============================================================================
-- 9. Initial Data: Insert Data Sources
-- ============================================================================
INSERT INTO data_sources (source_code, source_name, source_name_eng, api_base_url, is_active)
VALUES 
    ('BOK', '한국은행', 'Bank of Korea', 'https://ecos.bok.or.kr/api/', TRUE),
    ('KOSIS', '통계청', 'Statistics Korea', 'https://kosis.kr/openapi/', TRUE)
ON CONFLICT (source_code) DO NOTHING;

-- ============================================================================
-- 10. Helper Views
-- ============================================================================
-- View for DFM-selected statistics
CREATE OR REPLACE VIEW dfm_selected_statistics AS
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

-- View for active statistics by source
CREATE OR REPLACE VIEW active_statistics_by_source AS
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

-- ============================================================================
-- 11. Triggers for updated_at
-- ============================================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_data_sources_updated_at BEFORE UPDATE ON data_sources
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_statistics_metadata_updated_at BEFORE UPDATE ON statistics_metadata
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_statistics_items_updated_at BEFORE UPDATE ON statistics_items
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_series_last_updated BEFORE UPDATE ON series
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
