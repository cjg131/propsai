-- PropsAI v7 Migration: CLV Tracking + Evaluation Tables
-- Run this in the Supabase SQL Editor

-- ============================================================
-- LINE SNAPSHOTS (for CLV tracking + line movement analysis)
-- ============================================================
CREATE TABLE IF NOT EXISTS line_snapshots (
    id BIGSERIAL PRIMARY KEY,
    player_id TEXT,
    game_id TEXT,
    prop_type TEXT NOT NULL,
    sportsbook TEXT,
    line DOUBLE PRECISION,
    over_odds INTEGER,
    under_odds INTEGER,
    snapshot_time TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_line_snapshots_player_prop
    ON line_snapshots(player_id, prop_type, snapshot_time);
CREATE INDEX IF NOT EXISTS idx_line_snapshots_time
    ON line_snapshots(snapshot_time);

-- ============================================================
-- PREDICTION EVALUATIONS (auto-evaluation results)
-- ============================================================
CREATE TABLE IF NOT EXISTS prediction_evaluations (
    id BIGSERIAL PRIMARY KEY,
    evaluation_date DATE UNIQUE NOT NULL,
    total_predictions INTEGER,
    hits INTEGER,
    misses INTEGER,
    hit_rate DOUBLE PRECISION,
    rmse DOUBLE PRECISION,
    mae DOUBLE PRECISION,
    by_prop_type JSONB,
    calibration JSONB,
    clv_avg DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
