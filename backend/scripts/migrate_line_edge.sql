-- Add line edge signal columns to predictions table
-- These track the proven profitable betting signal:
-- When a player's L10 avg is 50%+ above the sportsbook line and odds >= -110

ALTER TABLE predictions ADD COLUMN IF NOT EXISTS line_edge_signal TEXT DEFAULT NULL;
-- Values: 'strong_over', 'moderate_over', 'strong_under', 'moderate_under', NULL
-- 'strong_over' = L10 avg 50%+ above line (proven +24% ROI)
-- 'moderate_over' = L10 avg 30-50% above line (marginal edge)

ALTER TABLE predictions ADD COLUMN IF NOT EXISTS avg_vs_line_pct DOUBLE PRECISION DEFAULT NULL;
-- (L10 avg - line) / line * 100 — how far player's recent average deviates from the line

ALTER TABLE predictions ADD COLUMN IF NOT EXISTS pct_games_over_line DOUBLE PRECISION DEFAULT NULL;
-- % of last 20 games where player exceeded this line value

ALTER TABLE predictions ADD COLUMN IF NOT EXISTS l10_avg DOUBLE PRECISION DEFAULT NULL;
-- Player's last-10-game average for this stat
