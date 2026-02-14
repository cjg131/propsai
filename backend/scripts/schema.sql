-- PropsAI Database Schema for Supabase (PostgreSQL)
-- Run this in the Supabase SQL Editor to create all tables

-- ============================================================
-- TEAMS
-- ============================================================
CREATE TABLE IF NOT EXISTS teams (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    abbreviation TEXT NOT NULL UNIQUE,
    city TEXT NOT NULL,
    conference TEXT NOT NULL,
    division TEXT NOT NULL,
    logo_url TEXT,
    arena_name TEXT,
    arena_city TEXT,
    arena_state TEXT,
    arena_latitude DOUBLE PRECISION,
    arena_longitude DOUBLE PRECISION,
    timezone TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- PLAYERS
-- ============================================================
CREATE TABLE IF NOT EXISTS players (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    team_id TEXT REFERENCES teams(id),
    position TEXT,
    jersey_number TEXT,
    height TEXT,
    weight INTEGER,
    birth_date DATE,
    college TEXT,
    draft_year INTEGER,
    draft_round INTEGER,
    draft_pick INTEGER,
    headshot_url TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    is_starter BOOLEAN DEFAULT FALSE,
    is_rookie BOOLEAN DEFAULT FALSE,
    is_recently_traded BOOLEAN DEFAULT FALSE,
    trade_date DATE,
    previous_team_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_players_team ON players(team_id);
CREATE INDEX idx_players_name ON players(name);

-- ============================================================
-- SEASONS
-- ============================================================
CREATE TABLE IF NOT EXISTS seasons (
    id TEXT PRIMARY KEY,  -- e.g., "2024-25"
    start_date DATE NOT NULL,
    end_date DATE,
    is_current BOOLEAN DEFAULT FALSE,
    data_loaded BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- GAMES
-- ============================================================
CREATE TABLE IF NOT EXISTS games (
    id TEXT PRIMARY KEY,
    season_id TEXT REFERENCES seasons(id),
    game_date DATE NOT NULL,
    home_team_id TEXT REFERENCES teams(id),
    away_team_id TEXT REFERENCES teams(id),
    home_score INTEGER,
    away_score INTEGER,
    status TEXT DEFAULT 'scheduled',  -- scheduled, in_progress, final
    is_playoff BOOLEAN DEFAULT FALSE,
    playoff_round TEXT,
    pace DOUBLE PRECISION,
    total_score INTEGER,
    spread DOUBLE PRECISION,
    over_under DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_games_date ON games(game_date);
CREATE INDEX idx_games_season ON games(season_id);
CREATE INDEX idx_games_home_team ON games(home_team_id);
CREATE INDEX idx_games_away_team ON games(away_team_id);

-- ============================================================
-- PLAYER GAME STATS
-- ============================================================
CREATE TABLE IF NOT EXISTS player_game_stats (
    id BIGSERIAL PRIMARY KEY,
    player_id TEXT REFERENCES players(id),
    game_id TEXT REFERENCES games(id),
    team_id TEXT REFERENCES teams(id),
    minutes DOUBLE PRECISION,
    points INTEGER DEFAULT 0,
    rebounds INTEGER DEFAULT 0,
    assists INTEGER DEFAULT 0,
    steals INTEGER DEFAULT 0,
    blocks INTEGER DEFAULT 0,
    turnovers INTEGER DEFAULT 0,
    three_pointers_made INTEGER DEFAULT 0,
    three_pointers_attempted INTEGER DEFAULT 0,
    field_goals_made INTEGER DEFAULT 0,
    field_goals_attempted INTEGER DEFAULT 0,
    free_throws_made INTEGER DEFAULT 0,
    free_throws_attempted INTEGER DEFAULT 0,
    offensive_rebounds INTEGER DEFAULT 0,
    defensive_rebounds INTEGER DEFAULT 0,
    personal_fouls INTEGER DEFAULT 0,
    plus_minus INTEGER DEFAULT 0,
    usage_rate DOUBLE PRECISION,
    is_starter BOOLEAN DEFAULT FALSE,
    dk_fantasy_score DOUBLE PRECISION,
    fd_fantasy_score DOUBLE PRECISION,
    yahoo_fantasy_score DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(player_id, game_id)
);

CREATE INDEX idx_pgs_player ON player_game_stats(player_id);
CREATE INDEX idx_pgs_game ON player_game_stats(game_id);
CREATE INDEX idx_pgs_player_game ON player_game_stats(player_id, game_id);

-- ============================================================
-- REFEREES
-- ============================================================
CREATE TABLE IF NOT EXISTS referees (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS game_referees (
    id BIGSERIAL PRIMARY KEY,
    game_id TEXT REFERENCES games(id),
    referee_id TEXT REFERENCES referees(id),
    UNIQUE(game_id, referee_id)
);

CREATE TABLE IF NOT EXISTS referee_stats (
    id BIGSERIAL PRIMARY KEY,
    referee_id TEXT REFERENCES referees(id),
    season_id TEXT REFERENCES seasons(id),
    avg_fouls_per_game DOUBLE PRECISION,
    avg_pace DOUBLE PRECISION,
    home_win_pct DOUBLE PRECISION,
    avg_total_score DOUBLE PRECISION,
    overtime_pct DOUBLE PRECISION,
    tech_fouls_per_game DOUBLE PRECISION,
    games_officiated INTEGER DEFAULT 0,
    UNIQUE(referee_id, season_id)
);

-- ============================================================
-- REFEREE-PLAYER HISTORICAL MATCHUPS
-- ============================================================
CREATE TABLE IF NOT EXISTS referee_player_stats (
    id BIGSERIAL PRIMARY KEY,
    referee_id TEXT REFERENCES referees(id),
    player_id TEXT REFERENCES players(id),
    games_together INTEGER DEFAULT 0,
    avg_points DOUBLE PRECISION,
    avg_rebounds DOUBLE PRECISION,
    avg_assists DOUBLE PRECISION,
    avg_fouls DOUBLE PRECISION,
    avg_free_throws_attempted DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(referee_id, player_id)
);

-- ============================================================
-- INJURY REPORTS
-- ============================================================
CREATE TABLE IF NOT EXISTS injury_reports (
    id BIGSERIAL PRIMARY KEY,
    player_id TEXT REFERENCES players(id),
    game_id TEXT REFERENCES games(id),
    status TEXT NOT NULL,  -- healthy, questionable, doubtful, out
    description TEXT,
    source TEXT,  -- official, twitter, rss
    reported_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_injuries_player ON injury_reports(player_id);
CREATE INDEX idx_injuries_game ON injury_reports(game_id);

-- ============================================================
-- PROP LINES (from sportsbooks)
-- ============================================================
CREATE TABLE IF NOT EXISTS prop_lines (
    id BIGSERIAL PRIMARY KEY,
    player_id TEXT REFERENCES players(id),
    game_id TEXT REFERENCES games(id),
    prop_type TEXT NOT NULL,
    sportsbook TEXT NOT NULL,
    line DOUBLE PRECISION NOT NULL,
    over_odds INTEGER,
    under_odds INTEGER,
    is_opening BOOLEAN DEFAULT FALSE,
    fetched_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_props_player_game ON prop_lines(player_id, game_id);
CREATE INDEX idx_props_type ON prop_lines(prop_type);
CREATE INDEX idx_props_sportsbook ON prop_lines(sportsbook);

-- ============================================================
-- PREDICTIONS
-- ============================================================
CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
    player_id TEXT REFERENCES players(id),
    game_id TEXT REFERENCES games(id),
    prop_type TEXT NOT NULL,
    line DOUBLE PRECISION NOT NULL,
    predicted_value DOUBLE PRECISION NOT NULL,
    prediction_range_low DOUBLE PRECISION,
    prediction_range_high DOUBLE PRECISION,
    over_probability DOUBLE PRECISION,
    under_probability DOUBLE PRECISION,
    confidence_score DOUBLE PRECISION,
    confidence_tier INTEGER,  -- 1-5
    edge_pct DOUBLE PRECISION,
    expected_value DOUBLE PRECISION,
    recommended_bet TEXT,  -- "over" or "under"
    kelly_bet_size DOUBLE PRECISION,
    best_book TEXT,
    best_odds INTEGER,
    ensemble_agreement DOUBLE PRECISION,
    model_contributions JSONB DEFAULT '[]',
    feature_importances JSONB DEFAULT '[]',
    preset_used TEXT DEFAULT 'balanced',
    actual_value DOUBLE PRECISION,
    was_correct BOOLEAN,
    is_paper_trade BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_predictions_player ON predictions(player_id);
CREATE INDEX idx_predictions_game ON predictions(game_id);
CREATE INDEX idx_predictions_date ON predictions(created_at);

-- ============================================================
-- BETS
-- ============================================================
CREATE TABLE IF NOT EXISTS bets (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
    prediction_id TEXT REFERENCES predictions(id),
    player_id TEXT REFERENCES players(id),
    player_name TEXT NOT NULL,
    team TEXT,
    opponent TEXT,
    game_date DATE,
    prop_type TEXT NOT NULL,
    line DOUBLE PRECISION NOT NULL,
    bet_type TEXT NOT NULL,  -- "over" or "under"
    odds INTEGER NOT NULL,
    sportsbook TEXT NOT NULL,
    stake DOUBLE PRECISION NOT NULL,
    status TEXT DEFAULT 'pending',  -- pending, won, lost, push
    actual_value DOUBLE PRECISION,
    profit DOUBLE PRECISION,
    confidence_tier INTEGER,
    notes TEXT,
    is_paper_trade BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ
);

CREATE INDEX idx_bets_status ON bets(status);
CREATE INDEX idx_bets_date ON bets(game_date);
CREATE INDEX idx_bets_player ON bets(player_id);

-- ============================================================
-- PARLAYS
-- ============================================================
CREATE TABLE IF NOT EXISTS parlays (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
    name TEXT,
    combined_odds INTEGER,
    stake DOUBLE PRECISION,
    potential_payout DOUBLE PRECISION,
    status TEXT DEFAULT 'pending',
    is_ai_suggested BOOLEAN DEFAULT FALSE,
    correlation_score DOUBLE PRECISION,
    profit DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS parlay_legs (
    id BIGSERIAL PRIMARY KEY,
    parlay_id TEXT REFERENCES parlays(id) ON DELETE CASCADE,
    bet_id TEXT REFERENCES bets(id),
    prediction_id TEXT REFERENCES predictions(id),
    status TEXT DEFAULT 'pending',
    UNIQUE(parlay_id, bet_id)
);

-- ============================================================
-- MODEL PRESETS
-- ============================================================
CREATE TABLE IF NOT EXISTS model_presets (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
    name TEXT NOT NULL,
    description TEXT,
    kelly_fraction DOUBLE PRECISION DEFAULT 0.5,
    min_confidence DOUBLE PRECISION DEFAULT 0.55,
    model_weights JSONB DEFAULT '{}',
    is_builtin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insert built-in presets
INSERT INTO model_presets (id, name, description, kelly_fraction, min_confidence, is_builtin)
VALUES
    ('conservative', 'Conservative', 'Lower risk, fewer picks, higher confidence threshold. Quarter Kelly sizing.', 0.25, 0.7, TRUE),
    ('balanced', 'Balanced', 'Moderate risk, balanced picks, standard confidence. Half Kelly sizing.', 0.5, 0.55, TRUE),
    ('aggressive', 'Aggressive', 'Higher risk, more picks, lower confidence threshold. Full Kelly sizing.', 1.0, 0.4, TRUE)
ON CONFLICT (id) DO NOTHING;

-- ============================================================
-- APP SETTINGS
-- ============================================================
CREATE TABLE IF NOT EXISTS app_settings (
    id TEXT PRIMARY KEY DEFAULT 'default',
    bankroll DOUBLE PRECISION DEFAULT 1000.0,
    unit_size DOUBLE PRECISION DEFAULT 10.0,
    active_preset TEXT DEFAULT 'balanced' REFERENCES model_presets(id),
    fantasy_format TEXT DEFAULT 'draftkings',
    preferred_books TEXT[] DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

INSERT INTO app_settings (id) VALUES ('default') ON CONFLICT (id) DO NOTHING;

-- ============================================================
-- MODEL TRAINING LOG
-- ============================================================
CREATE TABLE IF NOT EXISTS model_training_log (
    id BIGSERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    training_started_at TIMESTAMPTZ DEFAULT NOW(),
    training_completed_at TIMESTAMPTZ,
    status TEXT DEFAULT 'running',  -- running, completed, failed
    accuracy DOUBLE PRECISION,
    samples_used INTEGER,
    parameters JSONB DEFAULT '{}',
    error_message TEXT
);

-- ============================================================
-- API USAGE TRACKING
-- ============================================================
CREATE TABLE IF NOT EXISTS api_usage (
    id BIGSERIAL PRIMARY KEY,
    service TEXT NOT NULL,  -- sportsdataio, openai, twitter
    endpoint TEXT,
    requests_made INTEGER DEFAULT 1,
    date DATE DEFAULT CURRENT_DATE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(service, date)
);

-- ============================================================
-- DATA BACKUPS LOG
-- ============================================================
CREATE TABLE IF NOT EXISTS backup_log (
    id BIGSERIAL PRIMARY KEY,
    backup_type TEXT NOT NULL,  -- auto, manual
    format TEXT,  -- csv, json, full
    file_path TEXT,
    size_bytes BIGINT,
    status TEXT DEFAULT 'completed',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

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

-- ============================================================
-- UPDATED_AT TRIGGER
-- ============================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_teams_updated_at BEFORE UPDATE ON teams
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_players_updated_at BEFORE UPDATE ON players
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_games_updated_at BEFORE UPDATE ON games
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_model_presets_updated_at BEFORE UPDATE ON model_presets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_app_settings_updated_at BEFORE UPDATE ON app_settings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_referee_player_stats_updated_at BEFORE UPDATE ON referee_player_stats
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
