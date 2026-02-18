"""
Dynamic Kalshi market scanner.
Discovers ALL open markets, categorizes them, parses multi-game parlays,
and identifies tradeable opportunities across weather, sports, and other categories.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any

from app.logging_config import get_logger
from app.services.kalshi_api import KalshiClient

logger = get_logger(__name__)

# ── Single-game sports series — comprehensive mapping ─────────────
# Maps Kalshi series tickers to Odds API sport keys and market types.
# Covers ALL sports categories on Kalshi: Basketball, Hockey, Soccer,
# Tennis, Golf, Football, MMA, Cricket, Baseball, Boxing, Esports,
# Lacrosse, Motorsport, Olympics, Rugby, Darts, Chess, etc.

def _sg(sport: str, mtype: str) -> dict[str, str]:
    return {"odds_sport": sport, "market_type": mtype}

SINGLE_GAME_SERIES: dict[str, dict[str, str]] = {
    # ── Basketball ────────────────────────────────────────────────
    # NBA (All-Star break Feb 2026, resumes ~Feb 20)
    "KXNBAGAME":        _sg("basketball_nba", "h2h"),
    "KXNBASPREAD":      _sg("basketball_nba", "spreads"),
    "KXNBATOTAL":       _sg("basketball_nba", "totals"),
    "KXNBATEAMTOTAL":   _sg("basketball_nba", "team_total"),
    "KXNBA1HWINNER":    _sg("basketball_nba", "h2h"),
    "KXNBA1HSPREAD":    _sg("basketball_nba", "spreads"),
    "KXNBA1HTOTAL":     _sg("basketball_nba", "totals"),
    "KXNBA2HWINNER":    _sg("basketball_nba", "h2h"),
    "KXNBA2HSPREAD":    _sg("basketball_nba", "spreads"),
    "KXNBA2HTOTAL":     _sg("basketball_nba", "totals"),
    "KXNBA1QWINNER":    _sg("basketball_nba", "h2h"),
    "KXNBA1QSPREAD":    _sg("basketball_nba", "spreads"),
    "KXNBA1QTOTAL":     _sg("basketball_nba", "totals"),
    "KXNBA2QWINNER":    _sg("basketball_nba", "h2h"),
    "KXNBA2QSPREAD":    _sg("basketball_nba", "spreads"),
    "KXNBA2QTOTAL":     _sg("basketball_nba", "totals"),
    "KXNBA3QWINNER":    _sg("basketball_nba", "h2h"),
    "KXNBA3QSPREAD":    _sg("basketball_nba", "spreads"),
    "KXNBA3QTOTAL":     _sg("basketball_nba", "totals"),
    "KXNBA4QWINNER":    _sg("basketball_nba", "h2h"),
    "KXNBA4QSPREAD":    _sg("basketball_nba", "spreads"),
    "KXNBA4QTOTAL":     _sg("basketball_nba", "totals"),
    "KXNBAALLSTARGAME": _sg("basketball_nba_all_stars", "h2h"),
    # NCAAB — Men's College Basketball
    "KXNCAABGAME":      _sg("basketball_ncaab", "h2h"),
    "KXNCAAMBSPREAD":   _sg("basketball_ncaab", "spreads"),
    "KXNCAAMBTOTAL":    _sg("basketball_ncaab", "totals"),
    # NCAAB — Women's College Basketball
    "KXNCAAWBGAME":     _sg("basketball_wncaab", "h2h"),
    "KXNCAAWBSPREAD":   _sg("basketball_wncaab", "spreads"),
    "KXNCAAWBTOTAL":    _sg("basketball_wncaab", "totals"),
    # Euroleague / Eurocup
    "KXEUROLEAGUEGAME": _sg("basketball_euroleague", "h2h"),
    "KXEUROCUPGAME":    _sg("basketball_euroleague", "h2h"),
    # Unrivaled
    "KXUNRIVALEDGAME":  _sg("basketball_wba", "h2h"),

    # ── Hockey ────────────────────────────────────────────────────
    "KXNHLGAME":        _sg("icehockey_nhl", "h2h"),
    "KXNHLSPREAD":      _sg("icehockey_nhl", "spreads"),
    "KXNHLTOTAL":       _sg("icehockey_nhl", "totals"),
    "KXAHLGAME":        _sg("icehockey_ahl", "h2h"),
    "KXSHLGAME":        _sg("icehockey_sweden_hockey_league", "h2h"),
    "KXLIIGAGAME":      _sg("icehockey_liiga", "h2h"),
    "KXKHLGAME":        _sg("icehockey_nhl", "h2h"),  # KHL not on Odds API, skip
    "KXDELGAME":        _sg("icehockey_nhl", "h2h"),  # DEL not on Odds API
    "KXELHGAME":        _sg("icehockey_nhl", "h2h"),  # ELH not on Odds API
    "KXIIHFGAME":       _sg("icehockey_nhl", "h2h"),  # IIHF not on Odds API

    # ── Soccer (30+ leagues) ──────────────────────────────────────
    # England
    "KXEPLGAME":              _sg("soccer_epl", "h2h"),
    "KXEPLSPREAD":            _sg("soccer_epl", "spreads"),
    "KXEPLTOTAL":             _sg("soccer_epl", "totals"),
    "KXEPLBTTS":              _sg("soccer_epl", "btts"),
    "KXEFLCHAMPIONSHIPGAME":  _sg("soccer_efl_champ", "h2h"),
    "KXEFLCUPGAME":           _sg("soccer_england_efl_cup", "h2h"),
    "KXEFLCUPTOTAL":          _sg("soccer_england_efl_cup", "totals"),
    "KXFACUPGAME":            _sg("soccer_fa_cup", "h2h"),
    "KXFACUPSPREAD":          _sg("soccer_fa_cup", "spreads"),
    "KXFACUPTOTAL":           _sg("soccer_fa_cup", "totals"),
    "KXEWSLGAME":             _sg("soccer_epl", "h2h"),  # Women's Super League
    # Spain
    "KXLALIGAGAME":           _sg("soccer_spain_la_liga", "h2h"),
    "KXLALIGASPREAD":         _sg("soccer_spain_la_liga", "spreads"),
    "KXLALIGATOTAL":          _sg("soccer_spain_la_liga", "totals"),
    "KXLALIGA2GAME":          _sg("soccer_spain_segunda_division", "h2h"),
    "KXCOPADELREYGAME":       _sg("soccer_spain_copa_del_rey", "h2h"),
    "KXCOPADELREYTOTAL":      _sg("soccer_spain_copa_del_rey", "totals"),
    # Italy
    "KXSERIEAGAME":           _sg("soccer_italy_serie_a", "h2h"),
    "KXSERIEASPREAD":         _sg("soccer_italy_serie_a", "spreads"),
    "KXSERIEATOTAL":          _sg("soccer_italy_serie_a", "totals"),
    "KXSERIEBGAME":           _sg("soccer_italy_serie_b", "h2h"),
    # Germany
    "KXBUNDESLIGAGAME":       _sg("soccer_germany_bundesliga", "h2h"),
    "KXBUNDESLIGASPREAD":     _sg("soccer_germany_bundesliga", "spreads"),
    "KXBUNDESLIGATOTAL":      _sg("soccer_germany_bundesliga", "totals"),
    "KXBUNDESLIGA2GAME":      _sg("soccer_germany_bundesliga2", "h2h"),
    "KXDFBPOKALGAME":         _sg("soccer_germany_bundesliga", "h2h"),  # DFB Pokal
    # France
    "KXLIGUE1GAME":           _sg("soccer_france_ligue_one", "h2h"),
    "KXLIGUE1SPREAD":         _sg("soccer_france_ligue_one", "spreads"),
    "KXLIGUE1TOTAL":          _sg("soccer_france_ligue_one", "totals"),
    "KXCOUPEDEFRANCEGAME":    _sg("soccer_france_ligue_one", "h2h"),
    "KXCOUPEDEFRANCETOTAL":   _sg("soccer_france_ligue_one", "totals"),
    # UEFA
    "KXUCLGAME":              _sg("soccer_uefa_champs_league", "h2h"),
    "KXUCLSPREAD":            _sg("soccer_uefa_champs_league", "spreads"),
    "KXUCLTOTAL":             _sg("soccer_uefa_champs_league", "totals"),
    "KXUCLBTTS":              _sg("soccer_uefa_champs_league", "btts"),
    "KXUCLWGAME":             _sg("soccer_uefa_champs_league_women", "h2h"),
    "KXUELGAME":              _sg("soccer_uefa_europa_league", "h2h"),
    "KXUECLGAME":             _sg("soccer_uefa_europa_conference_league", "h2h"),
    "KXUEFAGAME":             _sg("soccer_uefa_champs_league", "h2h"),
    # Netherlands
    "KXEREDIVISIEGAME":       _sg("soccer_netherlands_eredivisie", "h2h"),
    # Portugal
    "KXLIGAPORTUGALGAME":     _sg("soccer_portugal_primeira_liga", "h2h"),
    "KXTACAPORTGAME":         _sg("soccer_portugal_primeira_liga", "h2h"),
    "KXTACAPORTSPREAD":       _sg("soccer_portugal_primeira_liga", "spreads"),
    "KXTACAPORTTOTAL":        _sg("soccer_portugal_primeira_liga", "totals"),
    # Scotland
    "KXSCOTTISHPREMGAME":     _sg("soccer_spl", "h2h"),
    # Denmark
    "KXDANISHSUPERLIGAGAME":  _sg("soccer_denmark_superliga", "h2h"),
    "KXDENSUPERLIGAGAME":     _sg("soccer_denmark_superliga", "h2h"),
    # Poland
    "KXEKSTRAKLASAGAME":      _sg("soccer_poland_ekstraklasa", "h2h"),
    # Greece
    "KXSLGREECEGAME":         _sg("soccer_greece_super_league", "h2h"),
    # Turkey
    "KXSUPERLIGGAME":         _sg("soccer_turkey_super_league", "h2h"),
    # Switzerland
    "KXSWISSLEAGUEGAME":      _sg("soccer_switzerland_superleague", "h2h"),
    # Austria
    "KXAUSTRIABUNDESLIGAGAME": _sg("soccer_austria_bundesliga", "h2h"),
    # Saudi Arabia
    "KXSAUDIPLGAME":          _sg("soccer_saudi_arabia_pro_league", "h2h"),
    "KXSAUDIPLSPREAD":        _sg("soccer_saudi_arabia_pro_league", "spreads"),
    "KXSAUDIPLTOTAL":         _sg("soccer_saudi_arabia_pro_league", "totals"),
    # Brazil
    "KXBRASILEIROGAME":       _sg("soccer_brazil_campeonato", "h2h"),
    "KXBRASILEIROTOTAL":      _sg("soccer_brazil_campeonato", "totals"),
    # Argentina
    "KXARGENTINAGAME":        _sg("soccer_argentina_primera_division", "h2h"),
    # Mexico
    "KXLIGAMXGAME":           _sg("soccer_mexico_ligamx", "h2h"),
    "KXLIGAMXSPREAD":         _sg("soccer_mexico_ligamx", "spreads"),
    "KXLIGAMXTOTAL":          _sg("soccer_mexico_ligamx", "totals"),
    # MLS
    "KXMLSGAME":              _sg("soccer_usa_mls", "h2h"),
    "KXMLSSPREAD":            _sg("soccer_usa_mls", "spreads"),
    "KXMLSTOTAL":             _sg("soccer_usa_mls", "totals"),
    # Japan
    "KXJLEAGUEGAME":          _sg("soccer_japan_j_league", "h2h"),
    # Korea
    "KXKLEAGUEGAME":          _sg("soccer_japan_j_league", "h2h"),  # K League not on Odds API
    # Australia
    "KXALEAGUEGAME":          _sg("soccer_australia_aleague", "h2h"),
    "KXALEAGUETOTAL":         _sg("soccer_australia_aleague", "totals"),
    # Croatia
    "KXHNLGAME":              _sg("soccer_epl", "h2h"),  # HNL not on Odds API
    # Sweden
    "KXALLSVENSKANGAME":      _sg("soccer_sweden_allsvenskan", "h2h"),
    # FIFA / International
    "KXFIFAGAME":             _sg("soccer_fifa_world_cup", "h2h"),
    "KXINTLFRIENDLYGAME":     _sg("soccer_fifa_world_cup", "h2h"),
    "KXWCGAME":               _sg("soccer_fifa_world_cup", "h2h"),
    # Generic soccer (catch-all)
    "KXSOCCERSPREAD":         _sg("soccer_epl", "spreads"),
    "KXSOCCERTOTAL":          _sg("soccer_epl", "totals"),

    # ── Football (NFL) ────────────────────────────────────────────
    "KXNFLGAME":        _sg("americanfootball_nfl", "h2h"),
    "KXNFLSPREAD":      _sg("americanfootball_nfl", "spreads"),
    "KXNFLTOTAL":       _sg("americanfootball_nfl", "totals"),
    "KXNFLTEAMTOTAL":   _sg("americanfootball_nfl", "team_total"),
    "KXNFL1HWINNER":    _sg("americanfootball_nfl", "h2h"),
    "KXNFL1HSPREAD":    _sg("americanfootball_nfl", "spreads"),
    "KXNFL1HTOTAL":     _sg("americanfootball_nfl", "totals"),
    "KXNFL2HWINNER":    _sg("americanfootball_nfl", "h2h"),
    "KXNFL2HSPREAD":    _sg("americanfootball_nfl", "spreads"),
    "KXNFL2HTOTAL":     _sg("americanfootball_nfl", "totals"),
    "KXNFL1QWINNER":    _sg("americanfootball_nfl", "h2h"),
    "KXNFL1QSPREAD":    _sg("americanfootball_nfl", "spreads"),
    "KXNFL1QTOTAL":     _sg("americanfootball_nfl", "totals"),
    "KXNFL2QWINNER":    _sg("americanfootball_nfl", "h2h"),
    "KXNFL2QSPREAD":    _sg("americanfootball_nfl", "spreads"),
    "KXNFL2QTOTAL":     _sg("americanfootball_nfl", "totals"),
    "KXNFL3QWINNER":    _sg("americanfootball_nfl", "h2h"),
    "KXNFL3QSPREAD":    _sg("americanfootball_nfl", "spreads"),
    "KXNFL3QTOTAL":     _sg("americanfootball_nfl", "totals"),
    "KXNFL4QWINNER":    _sg("americanfootball_nfl", "h2h"),
    "KXNFL4QSPREAD":    _sg("americanfootball_nfl", "spreads"),
    "KXNFL4QTOTAL":     _sg("americanfootball_nfl", "totals"),
    # College Football
    "KXNCAAFGAME":      _sg("americanfootball_ncaaf", "h2h"),
    "KXNCAAFSPREAD":    _sg("americanfootball_ncaaf", "spreads"),
    "KXNCAAFTOTAL":     _sg("americanfootball_ncaaf", "totals"),
    "KXNCAAFTEAMTOTAL": _sg("americanfootball_ncaaf", "team_total"),
    "KXNCAAFCSGAME":    _sg("americanfootball_ncaaf", "h2h"),
    "KXNCAAF1HWINNER":  _sg("americanfootball_ncaaf", "h2h"),

    # ── Tennis ────────────────────────────────────────────────────
    "KXATPGAME":              _sg("tennis_atp_qatar_open", "h2h"),
    "KXATPMATCH":             _sg("tennis_atp_qatar_open", "h2h"),
    "KXATPTOTALSETS":         _sg("tennis_atp_qatar_open", "totals"),
    "KXWTAGAME":              _sg("tennis_wta_dubai", "h2h"),
    "KXWTAMATCH":             _sg("tennis_wta_dubai", "h2h"),
    "KXWTACHALLENGERMATCH":   _sg("tennis_wta_dubai", "h2h"),
    "KXWTADOUBLES":           _sg("tennis_wta_dubai", "h2h"),
    "KXDAVISCUPMATCH":        _sg("tennis_atp_qatar_open", "h2h"),
    "KXUNITEDCUPMATCH":       _sg("tennis_atp_qatar_open", "h2h"),
    "KXTENNISEXHIBITION":     _sg("tennis_atp_qatar_open", "h2h"),

    # ── MMA / Boxing ──────────────────────────────────────────────
    "KXUFCGAME":        _sg("mma_mixed_martial_arts", "h2h"),
    "KXUFCMATCH":       _sg("mma_mixed_martial_arts", "h2h"),
    "KXUFCROUNDS":      _sg("mma_mixed_martial_arts", "totals"),
    "KXUFCMOF":         _sg("mma_mixed_martial_arts", "h2h"),
    "KXBOXINGMATCH":    _sg("boxing_boxing", "h2h"),
    "KXBOXINGKNOCKOUT": _sg("boxing_boxing", "h2h"),

    # ── Baseball ──────────────────────────────────────────────────
    "KXMLBGAME":        _sg("baseball_mlb", "h2h"),
    "KXMLBSPREAD":      _sg("baseball_mlb", "spreads"),
    "KXMLBTOTAL":       _sg("baseball_mlb", "totals"),
    "KXNCAABBGAME":     _sg("baseball_ncaa", "h2h"),
    "KXNCAABBSPREAD":   _sg("baseball_ncaa", "spreads"),
    "KXNCAABBTOTAL":    _sg("baseball_ncaa", "totals"),

    # ── Cricket ───────────────────────────────────────────────────
    "KXIPLGAME":        _sg("cricket_ipl", "h2h"),
    "KXT20MATCH":       _sg("cricket_t20_world_cup", "h2h"),
    "KXWPLGAME":        _sg("cricket_ipl", "h2h"),

    # ── Rugby ─────────────────────────────────────────────────────
    "KXSIXNATIONSMATCH":  _sg("rugbyunion_six_nations", "h2h"),
    "KXRUGBYNRLMATCH":    _sg("rugbyleague_nrl", "h2h"),

    # ── Lacrosse ──────────────────────────────────────────────────
    "KXLAXGAME":        _sg("lacrosse_ncaa", "h2h"),

    # ── Esports ───────────────────────────────────────────────────
    "KXLOLGAME":        _sg("", "h2h"),  # No Odds API match
    "KXLOLGAMES":       _sg("", "h2h"),
    "KXCS2GAME":        _sg("", "h2h"),
    "KXVALORANTGAME":   _sg("", "h2h"),
    "KXDOTA2GAME":      _sg("", "h2h"),

    # ── Darts ─────────────────────────────────────────────────────
    "KXDARTSMATCH":     _sg("", "h2h"),  # No Odds API match

    # ── Golf ──────────────────────────────────────────────────────
    "KXPGAGAME":        _sg("golf_pga_championship_winner", "h2h"),
    "KXTGLMATCH":       _sg("", "h2h"),

    # ── Motorsport ────────────────────────────────────────────────
    "KXF1RACE":         _sg("", "h2h"),  # No Odds API match

    # ── Olympics ──────────────────────────────────────────────────
    "KXWOCURLGAME":     _sg("", "h2h"),
    "KXWOMHOCKEYSPREAD": _sg("icehockey_nhl", "spreads"),
    "KXWOMHOCKEYTOTAL": _sg("icehockey_nhl", "totals"),
}

# ── Weather series — confirmed from Kalshi API ───────────────────
# Kalshi uses two naming conventions:
#   Old: KXHIGH + 2-letter city (KXHIGHNY, KXHIGHCHI, etc.)
#   New: KXHIGHT + 3-letter city (KXHIGHTATL, KXHIGHTDC, etc.)
# We enumerate ALL known series to avoid missing any.

WEATHER_SERIES_ALL: dict[str, dict[str, str]] = {
    # High temperature — old format
    "KXHIGHNY":   {"city_code": "NYC", "type": "high_temp"},
    "KXHIGHCHI":  {"city_code": "CHI", "type": "high_temp"},
    "KXHIGHLAX":  {"city_code": "LAX", "type": "high_temp"},
    "KXHIGHMIA":  {"city_code": "MIA", "type": "high_temp"},
    "KXHIGHDEN":  {"city_code": "DEN", "type": "high_temp"},
    # High temperature — new format
    "KXHIGHTATL": {"city_code": "ATL", "type": "high_temp"},
    "KXHIGHTAUS": {"city_code": "AUS", "type": "high_temp"},
    "KXHIGHTBOS": {"city_code": "BOS", "type": "high_temp"},
    "KXHIGHTCHI": {"city_code": "CHI", "type": "high_temp"},
    "KXHIGHTDC":  {"city_code": "DCA", "type": "high_temp"},
    "KXHIGHTDEN": {"city_code": "DEN", "type": "high_temp"},
    "KXHIGHTDFW": {"city_code": "DFW", "type": "high_temp"},
    "KXHIGHTHOU": {"city_code": "HOU", "type": "high_temp"},
    "KXHIGHTLAX": {"city_code": "LAX", "type": "high_temp"},
    "KXHIGHTLV":  {"city_code": "LAS", "type": "high_temp"},
    "KXHIGHTMIA": {"city_code": "MIA", "type": "high_temp"},
    "KXHIGHTMSP": {"city_code": "MSP", "type": "high_temp"},
    "KXHIGHTNYC": {"city_code": "NYC", "type": "high_temp"},
    "KXHIGHTPHL": {"city_code": "PHL", "type": "high_temp"},
    "KXHIGHTPHX": {"city_code": "PHX", "type": "high_temp"},
    "KXHIGHTSEA": {"city_code": "SEA", "type": "high_temp"},
    "KXHIGHTSFO": {"city_code": "SFO", "type": "high_temp"},
    # Low temperature
    "KXLOWTAUS":  {"city_code": "AUS", "type": "low_temp"},
    "KXLOWTCHI":  {"city_code": "CHI", "type": "low_temp"},
    "KXLOWTDEN":  {"city_code": "DEN", "type": "low_temp"},
    "KXLOWTLAX":  {"city_code": "LAX", "type": "low_temp"},
    "KXLOWTMIA":  {"city_code": "MIA", "type": "low_temp"},
    "KXLOWTNYC":  {"city_code": "NYC", "type": "low_temp"},
    # Rain
    "KXRAINNYC":  {"city_code": "NYC", "type": "rain"},
    "KXRAINCHI":  {"city_code": "CHI", "type": "rain"},
    "KXRAINLAX":  {"city_code": "LAX", "type": "rain"},
    "KXRAINDC":   {"city_code": "DCA", "type": "rain"},
    # Snow
    "KXSNOWNYC":  {"city_code": "NYC", "type": "snow"},
    "KXSNOWCHI":  {"city_code": "CHI", "type": "snow"},
    "KXSNOWBOS":  {"city_code": "BOS", "type": "snow"},
    "KXSNOWLA":   {"city_code": "LAX", "type": "snow"},
    "KXSNOWLAX":  {"city_code": "LAX", "type": "snow"},
    "KXSNOWDEN":  {"city_code": "DEN", "type": "snow"},
    "KXSNOWDC":   {"city_code": "DCA", "type": "snow"},
    # High temperature — additional discovered
    "KXHIGHAUS":  {"city_code": "AUS", "type": "high_temp"},
    "KXHIGHTNOLA": {"city_code": "NOL", "type": "high_temp"},
    "KXLOWTPHL":  {"city_code": "PHL", "type": "low_temp"},
    "KXLOWTDC":   {"city_code": "DCA", "type": "low_temp"},
    "KXLOWTATL":  {"city_code": "ATL", "type": "low_temp"},
    "KXLOWTBOS":  {"city_code": "BOS", "type": "low_temp"},
    "KXLOWTHOU":  {"city_code": "HOU", "type": "low_temp"},
    "KXLOWTLV":   {"city_code": "LAS", "type": "low_temp"},
    "KXLOWTPHX":  {"city_code": "PHX", "type": "low_temp"},
    "KXLOWTSEA":  {"city_code": "SEA", "type": "low_temp"},
    "KXLOWTSFO":  {"city_code": "SFO", "type": "low_temp"},
    "KXLOWTDFW":  {"city_code": "DFW", "type": "low_temp"},
    "KXLOWTMSP":  {"city_code": "MSP", "type": "low_temp"},
    # Snow — monthly
    "KXSNOWDET":  {"city_code": "DET", "type": "snow"},
    "KXSNOWSLC":  {"city_code": "SLC", "type": "snow"},
    "KXSNOWPHL":  {"city_code": "PHL", "type": "snow"},
    # Rain — monthly
    "KXRAINNYCM": {"city_code": "NYC", "type": "rain_monthly"},
    # Standalone
    "KXTORNADO":  {"city_code": "", "type": "tornado"},
    "KXHURRICANE": {"city_code": "", "type": "hurricane"},
}

# City code mapping for weather forecasts
WEATHER_CITY_CODES = {
    "NYC": "New York", "CHI": "Chicago", "LAX": "Los Angeles", "MIA": "Miami",
    "DCA": "Washington DC", "HOU": "Houston", "ATL": "Atlanta", "DFW": "Dallas",
    "DEN": "Denver", "PHL": "Philadelphia", "SFO": "San Francisco", "SEA": "Seattle",
    "BOS": "Boston", "AUS": "Austin", "LAS": "Las Vegas", "PHX": "Phoenix",
    "MSP": "Minneapolis", "NOL": "New Orleans", "DET": "Detroit", "SLC": "Salt Lake City",
}

# ── Parlay leg patterns ────────────────────────────────────────────

# "yes TeamName" → moneyline win
RE_MONEYLINE = re.compile(r"^(yes|no)\s+(.+?)$")
# "yes TeamName wins by over X.5 Points" → spread
RE_SPREAD = re.compile(r"^(yes|no)\s+(.+?)\s+wins\s+by\s+over\s+([\d.]+)\s+Points$")
# "yes Over X.5 points scored" → total
RE_TOTAL = re.compile(r"^(yes|no)\s+Over\s+([\d.]+)\s+points\s+scored$")
# "yes Both Teams To Score" → BTTS (soccer)
RE_BTTS = re.compile(r"^(yes|no)\s+Both\s+Teams?\s+To\s+Score$", re.IGNORECASE)
# "yes Over X.5 goals scored" → soccer total
RE_GOALS = re.compile(r"^(yes|no)\s+Over\s+([\d.]+)\s+goals\s+scored$")


def parse_parlay_legs(title: str) -> list[dict[str, Any]]:
    """
    Parse a multi-game parlay title into individual legs.
    Title format: "yes TeamA,yes TeamB wins by over 3.5 Points,no Over 150.5 points scored"
    """
    legs = []
    # Split on comma, but be careful with team names containing commas
    raw_legs = [leg.strip() for leg in title.split(",") if leg.strip()]

    for raw in raw_legs:
        leg = _parse_single_leg(raw)
        if leg:
            legs.append(leg)

    return legs


def _parse_single_leg(raw: str) -> dict[str, Any] | None:
    """Parse a single parlay leg string."""
    raw = raw.strip()
    if not raw:
        return None

    # Try spread first (more specific)
    m = RE_SPREAD.match(raw)
    if m:
        return {
            "type": "spread",
            "direction": m.group(1),  # yes/no
            "team": m.group(2).strip(),
            "line": float(m.group(3)),
            "raw": raw,
        }

    # Try total points
    m = RE_TOTAL.match(raw)
    if m:
        return {
            "type": "total",
            "direction": m.group(1),
            "line": float(m.group(2)),
            "raw": raw,
        }

    # Try goals total (soccer)
    m = RE_GOALS.match(raw)
    if m:
        return {
            "type": "goals_total",
            "direction": m.group(1),
            "line": float(m.group(2)),
            "raw": raw,
        }

    # Try BTTS
    m = RE_BTTS.match(raw)
    if m:
        return {
            "type": "btts",
            "direction": m.group(1),
            "raw": raw,
        }

    # Try moneyline (catch-all for "yes TeamName" or "no TeamName")
    m = RE_MONEYLINE.match(raw)
    if m:
        team = m.group(2).strip()
        # Filter out things that look like other patterns we missed
        if team and not team.startswith("Over") and not team.startswith("Under"):
            return {
                "type": "moneyline",
                "direction": m.group(1),
                "team": team,
                "raw": raw,
            }

    return None


def categorize_market(market: dict[str, Any]) -> str:
    """Categorize a Kalshi market into weather, sports, politics, finance, or other."""
    ticker = market.get("ticker", "")
    title = market.get("title", "").lower()
    series = market.get("series_ticker", "")

    # Weather — check if series_ticker or ticker prefix matches any known weather series
    for ws in WEATHER_SERIES_ALL:
        if ticker.startswith(ws) or series.startswith(ws):
            return "weather"
    if any(kw in title for kw in ["high temp", "low temp", "snowfall", "rainfall", "tornado", "hurricane", "maximum temperature", "minimum temperature", "rain in", "snow in"]):
        return "weather"

    # Sports
    if "KXMVESPORTS" in ticker or "KXMVESPORTS" in series:
        return "sports_parlay"
    if any(ticker.startswith(p) for p in ["KXNBA", "KXNFL", "KXNHL", "KXMLB", "KXNCAA", "KXMLS", "KXUFC", "KXPGA"]):
        return "sports_futures"
    if any(ticker.startswith(p) for p in ["KXNBAPTS", "KXNBAAST", "KXNBAREB"]):
        return "sports_props"

    # Politics
    if any(kw in title for kw in ["president", "election", "congress", "senate", "democrat", "republican"]):
        return "politics"

    # Finance
    if any(kw in title for kw in ["s&p", "nasdaq", "bitcoin", "fed rate", "interest rate", "gdp", "inflation"]):
        return "finance"

    return "other"


class KalshiScanner:
    """
    Scans all open Kalshi markets, categorizes them, and identifies
    tradeable opportunities with liquidity.
    """

    def __init__(self, kalshi: KalshiClient) -> None:
        self.kalshi = kalshi

    async def scan_all_open_markets(
        self,
        min_volume: int = 0,
        max_pages: int = 15,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Fetch all open markets and categorize them.
        Returns dict of category → list of parsed markets.
        """
        all_markets: list[dict[str, Any]] = []
        cursor = None

        for i in range(max_pages):
            if i > 0:
                await asyncio.sleep(1.0)  # Rate limit
            try:
                data = await self.kalshi.get_markets(
                    status="open", limit=200, cursor=cursor,
                )
                markets = data.get("markets", [])
                all_markets.extend(markets)
                cursor = data.get("cursor")
                if not cursor or not markets:
                    break
            except Exception as e:
                logger.warning("Market scan page failed", page=i, error=str(e))
                break

        logger.info("Market scan complete", total=len(all_markets))

        # Categorize
        categorized: dict[str, list[dict[str, Any]]] = {}
        for m in all_markets:
            cat = categorize_market(m)
            if cat not in categorized:
                categorized[cat] = []

            parsed = self._enrich_market(m, cat)
            if parsed:
                categorized[cat].append(parsed)

        # Log summary
        for cat, ms in categorized.items():
            liquid = [m for m in ms if (m.get("volume", 0) or 0) >= min_volume]
            logger.info(
                "Market category",
                category=cat,
                total=len(ms),
                liquid=len(liquid),
            )

        return categorized

    async def scan_weather_markets(self) -> list[dict[str, Any]]:
        """Scan all confirmed Kalshi weather series for active markets.

        NOTE: Kalshi uses status='active' for live tradeable markets,
        but the API filter accepts 'open' which returns active markets.
        We fetch without status filter and check locally to be safe.
        """
        weather_markets: list[dict[str, Any]] = []

        for series_ticker, config in WEATHER_SERIES_ALL.items():
            try:
                await asyncio.sleep(0.35)  # Rate limit: ~3 req/sec
                data = await self.kalshi.get_markets(
                    series_ticker=series_ticker, limit=50,
                )
                for m in data.get("markets", []):
                    # Only include active (tradeable) markets
                    if m.get("status") not in ("active", "open"):
                        continue
                    parsed = self._enrich_market(m, "weather")
                    if parsed:
                        # Attach city code and market type from our config
                        if "weather" not in parsed:
                            parsed["weather"] = {}
                        parsed["weather"]["city_code"] = config["city_code"]
                        parsed["weather"]["market_type"] = config["type"]
                        parsed["weather"]["series_ticker"] = series_ticker
                        weather_markets.append(parsed)
            except Exception as e:
                logger.debug("Weather series scan failed", series=series_ticker, error=str(e))
                continue

        logger.info("Weather scan complete", series_checked=len(WEATHER_SERIES_ALL), found=len(weather_markets))
        return weather_markets

    async def scan_sports_parlays(self, min_volume: int = 10) -> list[dict[str, Any]]:
        """Scan for sports parlay markets with liquidity."""
        parlays: list[dict[str, Any]] = []
        cursor = None

        for i in range(10):
            if i > 0:
                await asyncio.sleep(1.0)
            try:
                data = await self.kalshi.get_markets(
                    status="open",
                    series_ticker="KXMVESPORTSMULTIGAMEEXTENDED",
                    limit=200,
                    cursor=cursor,
                )
                markets = data.get("markets", [])
                for m in markets:
                    vol = m.get("volume", 0) or 0
                    yes_ask = m.get("yes_ask", 0) or 0
                    no_ask = m.get("no_ask", 0) or 0

                    # Skip no-liquidity markets
                    if vol < min_volume:
                        continue
                    if yes_ask <= 0 or no_ask <= 0:
                        continue
                    if yes_ask <= 2 or no_ask <= 2:
                        continue

                    parsed = self._enrich_market(m, "sports_parlay")
                    if parsed:
                        parlays.append(parsed)

                cursor = data.get("cursor")
                if not cursor or not markets:
                    break
            except Exception as e:
                logger.warning("Parlay scan page failed", page=i, error=str(e))
                break

        logger.info("Parlay scan complete", found=len(parlays))
        return parlays

    async def scan_single_game_markets(self, min_volume: int = 0) -> list[dict[str, Any]]:
        """Scan all single-game sports series for tradeable markets."""
        all_markets: list[dict[str, Any]] = []

        for series_ticker, series_info in SINGLE_GAME_SERIES.items():
            odds_sport = series_info.get("odds_sport", "")
            market_type = series_info.get("market_type", "")
            if not odds_sport:
                continue

            cursor = None
            series_count = 0
            for page in range(5):
                if page > 0:
                    await asyncio.sleep(0.5)
                try:
                    data = await self.kalshi.get_markets(
                        status="open",
                        series_ticker=series_ticker,
                        limit=200,
                        cursor=cursor,
                    )
                    markets = data.get("markets", [])
                    for m in markets:
                        vol = m.get("volume", 0) or 0
                        yes_ask = m.get("yes_ask", 0) or 0
                        no_ask = m.get("no_ask", 0) or 0

                        if vol < min_volume:
                            continue
                        if yes_ask <= 2 or no_ask <= 2:
                            continue

                        enriched = self._enrich_market(m, "sports_single")
                        if enriched:
                            enriched["odds_sport"] = odds_sport
                            enriched["kalshi_market_type"] = market_type
                            all_markets.append(enriched)
                            series_count += 1

                    cursor = data.get("cursor")
                    if not cursor or not markets:
                        break
                except Exception as e:
                    logger.debug("Single game scan failed", series=series_ticker, error=str(e))
                    break

            if series_count > 0:
                logger.info("Single game scan", series=series_ticker, found=series_count)

            await asyncio.sleep(0.5)

        logger.info("Single game scan complete", total=len(all_markets), series_checked=len(SINGLE_GAME_SERIES))
        return all_markets

    async def scan_crypto_markets(self) -> list[dict[str, Any]]:
        """Scan for active 15-minute crypto up/down markets.

        Discovers series tickers dynamically by searching for crypto-related
        open markets. Filters for markets with > 2 minutes remaining.
        """
        crypto_markets: list[dict[str, Any]] = []

        # Known and candidate crypto series tickers on Kalshi
        # These cover 15-min, 1-hour, and daily crypto markets
        crypto_series = [
            # 15-minute markets (primary target)
            "KXBTC", "KXBTCZ", "KXBTC15",
            "KXETH", "KXETHZ", "KXETH15",
            "KXSOL", "KXSOLZ", "KXSOL15",
            "KXXRP", "KXXRPZ", "KXXRP15",
            # Broader crypto series
            "KXBITCOIN", "KXCRYPTO",
        ]

        for series_ticker in crypto_series:
            try:
                await asyncio.sleep(0.35)
                data = await self.kalshi.get_markets(
                    series_ticker=series_ticker, limit=50, status="open",
                )
                markets = data.get("markets", [])

                for m in markets:
                    title = (m.get("title", "") or "").lower()
                    ticker = m.get("ticker", "")

                    # Filter: must be a crypto up/down or price market
                    is_crypto = any(kw in title for kw in [
                        "bitcoin", "btc", "ethereum", "eth", "solana", "sol",
                        "xrp", "crypto", "15 min", "15-min", "15min",
                    ])
                    if not is_crypto and "KXBTC" not in ticker and "KXETH" not in ticker and "KXSOL" not in ticker and "KXXRP" not in ticker:
                        continue

                    yes_ask = m.get("yes_ask", 0) or 0
                    no_ask = m.get("no_ask", 0) or 0
                    if yes_ask <= 2 or no_ask <= 2:
                        continue

                    enriched = self._enrich_market(m, "crypto")
                    if enriched:
                        # Detect which coin this market is for
                        coin = None
                        upper_title = m.get("title", "").upper()
                        upper_ticker = ticker.upper()
                        if "BTC" in upper_ticker or "BITCOIN" in upper_title:
                            coin = "BTC"
                        elif "ETH" in upper_ticker or "ETHEREUM" in upper_title:
                            coin = "ETH"
                        elif "SOL" in upper_ticker or "SOLANA" in upper_title:
                            coin = "SOL"
                        elif "XRP" in upper_ticker:
                            coin = "XRP"

                        enriched["crypto"] = {
                            "coin": coin,
                            "close_time": m.get("close_time", ""),
                        }
                        crypto_markets.append(enriched)

            except Exception as e:
                logger.debug("Crypto series scan failed", series=series_ticker, error=str(e))
                continue

        # Also try a keyword-based search if series approach finds nothing
        if not crypto_markets:
            try:
                await asyncio.sleep(0.5)
                data = await self.kalshi.get_markets(status="open", limit=200)
                for m in data.get("markets", []):
                    title = (m.get("title", "") or "").lower()
                    ticker = m.get("ticker", "")
                    if not any(kw in title for kw in ["bitcoin", "btc", "ethereum", "eth", "solana", "sol", "xrp", "crypto"]):
                        continue
                    if not any(kw in title for kw in ["15 min", "15-min", "up", "down", "above", "below", "price"]):
                        continue

                    yes_ask = m.get("yes_ask", 0) or 0
                    no_ask = m.get("no_ask", 0) or 0
                    if yes_ask <= 2 or no_ask <= 2:
                        continue

                    enriched = self._enrich_market(m, "crypto")
                    if enriched:
                        coin = None
                        upper_title = m.get("title", "").upper()
                        if "BTC" in upper_title or "BITCOIN" in upper_title:
                            coin = "BTC"
                        elif "ETH" in upper_title or "ETHEREUM" in upper_title:
                            coin = "ETH"
                        elif "SOL" in upper_title or "SOLANA" in upper_title:
                            coin = "SOL"
                        elif "XRP" in upper_title:
                            coin = "XRP"

                        enriched["crypto"] = {
                            "coin": coin,
                            "close_time": m.get("close_time", ""),
                        }
                        crypto_markets.append(enriched)
            except Exception as e:
                logger.debug("Crypto keyword scan failed", error=str(e))

        logger.info("Crypto scan complete", found=len(crypto_markets))
        return crypto_markets

    async def scan_finance_markets(self) -> list[dict[str, Any]]:
        """Scan for active S&P 500 and Nasdaq daily markets on Kalshi."""
        finance_markets: list[dict[str, Any]] = []

        finance_series = [
            "KXINX", "KXINXD", "KXINXU",
            "KXSP500", "KXSPX", "KXSPY",
            "KXNAS", "KXNASDAQ", "KXQQQ", "KXNDX",
            "KXDOW", "KXDJIA",
        ]

        for series_ticker in finance_series:
            try:
                await asyncio.sleep(0.35)
                data = await self.kalshi.get_markets(
                    series_ticker=series_ticker, limit=50, status="open",
                )
                markets = data.get("markets", [])

                for m in markets:
                    title = (m.get("title", "") or "").lower()
                    ticker = m.get("ticker", "")

                    is_finance = any(kw in title for kw in [
                        "s&p", "s&p 500", "sp500", "nasdaq", "dow", "index",
                        "close above", "close below", "close up", "close down",
                    ])
                    if not is_finance:
                        continue

                    yes_ask = m.get("yes_ask", 0) or 0
                    no_ask = m.get("no_ask", 0) or 0
                    if yes_ask <= 2 or no_ask <= 2:
                        continue

                    enriched = self._enrich_market(m, "finance")
                    if enriched:
                        # Detect which index
                        index = None
                        upper_title = m.get("title", "").upper()
                        if "S&P" in upper_title or "SP500" in upper_title or "SPX" in upper_title:
                            index = "SP500"
                        elif "NASDAQ" in upper_title or "QQQ" in upper_title or "NDX" in upper_title:
                            index = "NASDAQ"
                        elif "DOW" in upper_title or "DJIA" in upper_title:
                            index = "DOW"

                        enriched["finance"] = {
                            "index": index,
                            "close_time": m.get("close_time", ""),
                        }
                        finance_markets.append(enriched)

            except Exception as e:
                logger.debug("Finance series scan failed", series=series_ticker, error=str(e))
                continue

        # Fallback keyword search
        if not finance_markets:
            try:
                await asyncio.sleep(0.5)
                data = await self.kalshi.get_markets(status="open", limit=200)
                for m in data.get("markets", []):
                    title = (m.get("title", "") or "").lower()
                    if not any(kw in title for kw in ["s&p", "nasdaq", "dow jones"]):
                        continue
                    if not any(kw in title for kw in ["close", "above", "below", "up", "down"]):
                        continue

                    yes_ask = m.get("yes_ask", 0) or 0
                    no_ask = m.get("no_ask", 0) or 0
                    if yes_ask <= 2 or no_ask <= 2:
                        continue

                    enriched = self._enrich_market(m, "finance")
                    if enriched:
                        upper_title = m.get("title", "").upper()
                        index = None
                        if "S&P" in upper_title or "SP500" in upper_title:
                            index = "SP500"
                        elif "NASDAQ" in upper_title:
                            index = "NASDAQ"
                        elif "DOW" in upper_title:
                            index = "DOW"
                        enriched["finance"] = {"index": index, "close_time": m.get("close_time", "")}
                        finance_markets.append(enriched)
            except Exception as e:
                logger.debug("Finance keyword scan failed", error=str(e))

        logger.info("Finance scan complete", found=len(finance_markets))
        return finance_markets

    async def scan_econ_markets(self) -> list[dict[str, Any]]:
        """Scan for active economic event markets (CPI, Fed, Gas, Unemployment)."""
        econ_markets: list[dict[str, Any]] = []

        econ_series = [
            "KXCPI", "KXINFL",
            "KXFED", "KXFOMC", "KXRATE",
            "KXGAS", "KXOIL",
            "KXJOBS", "KXUNEMP", "KXNFP",
            "KXGDP", "KXECON",
        ]

        for series_ticker in econ_series:
            try:
                await asyncio.sleep(0.35)
                data = await self.kalshi.get_markets(
                    series_ticker=series_ticker, limit=50, status="open",
                )
                markets = data.get("markets", [])

                for m in markets:
                    title = (m.get("title", "") or "").lower()

                    yes_ask = m.get("yes_ask", 0) or 0
                    no_ask = m.get("no_ask", 0) or 0
                    if yes_ask <= 2 or no_ask <= 2:
                        continue

                    enriched = self._enrich_market(m, "econ")
                    if enriched:
                        # Detect econ type
                        econ_type = None
                        if any(kw in title for kw in ["cpi", "inflation", "consumer price"]):
                            econ_type = "cpi"
                        elif any(kw in title for kw in ["fed", "fomc", "interest rate", "rate cut", "rate hike"]):
                            econ_type = "fed_funds"
                        elif any(kw in title for kw in ["gas price", "gasoline", "oil price"]):
                            econ_type = "gas_price"
                        elif any(kw in title for kw in ["unemployment", "jobless", "nonfarm", "payroll", "jobs report"]):
                            econ_type = "unemployment"
                        elif any(kw in title for kw in ["gdp", "gross domestic"]):
                            econ_type = "gdp"

                        enriched["econ"] = {
                            "type": econ_type,
                            "close_time": m.get("close_time", ""),
                        }
                        econ_markets.append(enriched)

            except Exception as e:
                logger.debug("Econ series scan failed", series=series_ticker, error=str(e))
                continue

        # Fallback keyword search
        if not econ_markets:
            try:
                await asyncio.sleep(0.5)
                data = await self.kalshi.get_markets(status="open", limit=200)
                for m in data.get("markets", []):
                    title = (m.get("title", "") or "").lower()
                    if not any(kw in title for kw in [
                        "cpi", "inflation", "fed", "fomc", "interest rate",
                        "gas price", "gasoline", "unemployment", "jobless",
                        "nonfarm", "payroll", "gdp",
                    ]):
                        continue

                    yes_ask = m.get("yes_ask", 0) or 0
                    no_ask = m.get("no_ask", 0) or 0
                    if yes_ask <= 2 or no_ask <= 2:
                        continue

                    enriched = self._enrich_market(m, "econ")
                    if enriched:
                        econ_type = None
                        if any(kw in title for kw in ["cpi", "inflation"]):
                            econ_type = "cpi"
                        elif any(kw in title for kw in ["fed", "fomc", "interest rate"]):
                            econ_type = "fed_funds"
                        elif any(kw in title for kw in ["gas", "gasoline"]):
                            econ_type = "gas_price"
                        elif any(kw in title for kw in ["unemployment", "jobless", "nonfarm", "payroll"]):
                            econ_type = "unemployment"
                        elif "gdp" in title:
                            econ_type = "gdp"
                        enriched["econ"] = {"type": econ_type, "close_time": m.get("close_time", "")}
                        econ_markets.append(enriched)
            except Exception as e:
                logger.debug("Econ keyword scan failed", error=str(e))

        logger.info("Econ scan complete", found=len(econ_markets))
        return econ_markets

    async def scan_nba_props_markets(self) -> list[dict[str, Any]]:
        """Scan for active NBA player prop markets on Kalshi.

        Looks for markets like:
          - "Will LeBron James score 25+ points?" (KXNBAPTS...)
          - "Will Nikola Jokic get 10+ rebounds?" (KXNBAREB...)
          - "Will Luka Doncic get 8+ assists?" (KXNBAAST...)
        """
        props_markets: list[dict[str, Any]] = []

        nba_prop_series = [
            "KXNBAPTS", "KXNBAAST", "KXNBAREB",
            "KXNBA3PM", "KXNBASTL", "KXNBABLK", "KXNBATOV",
            "KXNBAPRA", "KXNBAPR", "KXNBAPA", "KXNBARA",
            "KXNBADD", "KXNBATD",
        ]

        for series_ticker in nba_prop_series:
            try:
                await asyncio.sleep(0.35)
                data = await self.kalshi.get_markets(
                    series_ticker=series_ticker, limit=100, status="open",
                )
                markets = data.get("markets", [])

                for m in markets:
                    yes_ask = m.get("yes_ask", 0) or 0
                    no_ask = m.get("no_ask", 0) or 0
                    if yes_ask <= 2 or no_ask <= 2:
                        continue

                    enriched = self._enrich_market(m, "nba_props")
                    if enriched:
                        title = m.get("title", "")
                        title_lower = title.lower()

                        # Detect prop type from series or title
                        prop_type = None
                        if "KXNBAPTS" in series_ticker or "point" in title_lower or "score" in title_lower:
                            prop_type = "points"
                        elif "KXNBAREB" in series_ticker or "rebound" in title_lower:
                            prop_type = "rebounds"
                        elif "KXNBAAST" in series_ticker or "assist" in title_lower:
                            prop_type = "assists"
                        elif "KXNBA3PM" in series_ticker or "three" in title_lower or "3-pointer" in title_lower:
                            prop_type = "threes"
                        elif "KXNBASTL" in series_ticker or "steal" in title_lower:
                            prop_type = "steals"
                        elif "KXNBABLK" in series_ticker or "block" in title_lower:
                            prop_type = "blocks"
                        elif "KXNBATOV" in series_ticker or "turnover" in title_lower:
                            prop_type = "turnovers"

                        # Extract player name from title
                        # Typical format: "Will [Player Name] score 25+ points?"
                        player_match = re.search(
                            r'(?:will\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z\'-]+)+)',
                            title,
                        )
                        player_name = player_match.group(1).strip() if player_match else None

                        # Extract line from title (e.g., "25+" or "over 25.5")
                        line_match = re.search(r'(\d+\.?\d*)\+|over\s+(\d+\.?\d*)', title_lower)
                        line = None
                        if line_match:
                            line = float(line_match.group(1) or line_match.group(2))

                        enriched["nba_props"] = {
                            "prop_type": prop_type,
                            "player_name": player_name,
                            "line": line,
                            "close_time": m.get("close_time", ""),
                        }
                        props_markets.append(enriched)

            except Exception as e:
                logger.debug("NBA props series scan failed", series=series_ticker, error=str(e))
                continue

        # Fallback keyword search if no series matched
        if not props_markets:
            try:
                await asyncio.sleep(0.5)
                data = await self.kalshi.get_markets(status="open", limit=200)
                for m in data.get("markets", []):
                    title = (m.get("title", "") or "").lower()
                    ticker = m.get("ticker", "")

                    if not any(kw in title for kw in [
                        "points", "rebounds", "assists", "steals", "blocks",
                        "three-pointer", "3-pointer", "turnovers",
                    ]):
                        continue
                    if not any(kw in title for kw in ["nba", "basketball"]) and not ticker.startswith("KXNBA"):
                        continue

                    yes_ask = m.get("yes_ask", 0) or 0
                    no_ask = m.get("no_ask", 0) or 0
                    if yes_ask <= 2 or no_ask <= 2:
                        continue

                    enriched = self._enrich_market(m, "nba_props")
                    if enriched:
                        enriched["nba_props"] = {
                            "prop_type": None,
                            "player_name": None,
                            "line": None,
                            "close_time": m.get("close_time", ""),
                        }
                        props_markets.append(enriched)
            except Exception as e:
                logger.debug("NBA props keyword scan failed", error=str(e))

        logger.info("NBA props scan complete", found=len(props_markets))
        return props_markets

    def _enrich_market(self, market: dict[str, Any], category: str) -> dict[str, Any] | None:
        """Add parsed metadata to a raw Kalshi market."""
        ticker = market.get("ticker", "")
        title = market.get("title", "")
        yes_ask = market.get("yes_ask", 0) or 0
        no_ask = market.get("no_ask", 0) or 0

        result: dict[str, Any] = {
            "ticker": ticker,
            "title": title,
            "category": category,
            "series_ticker": market.get("series_ticker", ""),
            "event_ticker": market.get("event_ticker", ""),
            "yes_bid": market.get("yes_bid", 0) or 0,
            "yes_ask": yes_ask,
            "no_bid": market.get("no_bid", 0) or 0,
            "no_ask": no_ask,
            "volume": market.get("volume", 0) or 0,
            "open_interest": market.get("open_interest", 0) or 0,
            "close_time": market.get("close_time", ""),
            "status": market.get("status", ""),
            "strike_type": market.get("strike_type", ""),
            "floor_strike": market.get("floor_strike"),
            "cap_strike": market.get("cap_strike"),
        }

        # Weather-specific parsing
        if category == "weather":
            result["weather"] = {
                "strike_type": market.get("strike_type", ""),
                "floor_strike": market.get("floor_strike"),
                "cap_strike": market.get("cap_strike"),
            }
            # Extract city from known series mapping
            for ws, ws_config in WEATHER_SERIES_ALL.items():
                if ticker.startswith(ws) or result["series_ticker"] == ws:
                    result["weather"]["city_code"] = ws_config["city_code"]
                    result["weather"]["market_type"] = ws_config["type"]
                    result["weather"]["city_name"] = WEATHER_CITY_CODES.get(ws_config["city_code"], ws_config["city_code"])
                    break

        # Parlay-specific parsing
        if category == "sports_parlay":
            legs = parse_parlay_legs(title)
            result["parlay"] = {
                "legs": legs,
                "num_legs": len(legs),
                "sports_detected": _detect_sports(legs),
            }

        return result


def _detect_sports(legs: list[dict[str, Any]]) -> list[str]:
    """Detect which sports are in a parlay based on leg content."""
    sports = set()
    for leg in legs:
        team = leg.get("team", "")
        leg_type = leg.get("type", "")

        # Soccer indicators
        if leg_type in ("goals_total", "btts"):
            sports.add("soccer")
            continue

        # Known soccer teams
        soccer_teams = [
            "Liverpool", "Barcelona", "Bayern Munich", "Aston Villa", "Lyon",
            "Roma", "Mallorca", "Villarreal", "Arsenal", "Chelsea", "Man City",
            "Man United", "Tottenham", "Real Madrid", "Atletico", "PSG",
            "Juventus", "Inter Milan", "AC Milan", "Napoli", "Dortmund",
        ]
        if any(st.lower() in team.lower() for st in soccer_teams):
            sports.add("soccer")
            continue

        # Tennis indicators
        tennis_names = [
            "Fritz", "Shelton", "Navarro", "Tauson", "Jovic", "Cerundolo",
            "Djokovic", "Sinner", "Alcaraz", "Swiatek", "Gauff", "Sabalenka",
        ]
        if any(tn.lower() in team.lower() for tn in tennis_names):
            sports.add("tennis")
            continue

        # Points-based → basketball
        if leg_type == "total" or leg_type == "spread":
            line = leg.get("line", 0)
            if line > 100:  # Basketball totals are 100+
                sports.add("basketball")
            elif line > 30:  # Could be basketball spread
                sports.add("basketball")
            continue

        # Default: likely basketball (college)
        if leg_type == "moneyline":
            sports.add("basketball")

    return sorted(sports)
