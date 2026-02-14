-- Paper Trading table for virtual bet tracking
CREATE TABLE IF NOT EXISTS paper_trades (
    id BIGSERIAL PRIMARY KEY,
    prediction_id TEXT,
    player_name TEXT NOT NULL,
    player_id TEXT,
    team TEXT,
    opponent TEXT,
    game_date DATE NOT NULL,
    prop_type TEXT NOT NULL,
    line NUMERIC(6,1) NOT NULL,
    predicted_value NUMERIC(6,1),
    recommended_bet TEXT NOT NULL,  -- 'over' or 'under'
    confidence_score NUMERIC(5,1),
    odds INTEGER DEFAULT -110,
    stake NUMERIC(10,2) NOT NULL,
    actual_value NUMERIC(6,1),
    result TEXT DEFAULT 'pending',  -- 'pending', 'win', 'loss', 'push'
    profit NUMERIC(10,2) DEFAULT 0,
    bankroll_after NUMERIC(12,2),
    session_id TEXT,  -- groups trades by session
    created_at TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_paper_trades_date ON paper_trades(game_date);
CREATE INDEX IF NOT EXISTS idx_paper_trades_session ON paper_trades(session_id);
CREATE INDEX IF NOT EXISTS idx_paper_trades_result ON paper_trades(result);
