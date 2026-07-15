"""Orchestrate all Layer 1 computations into a single bundle dict."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from layer1.data_loader import (
    load_prices,
    load_prices_multi,
    load_latest_fundamentals,
    load_ttm_quarters,
    load_annual_revenue,
    load_ticker_metadata,
    load_peers,
    load_tnx_rate,
    load_rd_revenue_ratio,
    load_capex_revenue_ratio,
    load_macro_etf_prices,
)
from layer1.returns import daily_returns, cumulative_return
from layer1.relative_strength import relative_strength
from layer1.alpha_beta import alpha_beta
from layer1.regimes import vix_regime, rate_regime, market_trend
from layer1.commodity import commodity_impact
from layer1.peer_gaps import peer_gaps, ttm_revenue_growth, compute_margin
from layer1.flags import emit_flags
from layer1.pestel.political import political_signals
from layer1.pestel.economic import economic_signals
from layer1.pestel.social import social_signals
from layer1.pestel.technological import technological_signals
from layer1.pestel.environmental import environmental_signals
from layer1.pestel.legal import legal_signals

_CFG = Path(__file__).parent.parent / "config"


def _load_yaml(name: str) -> dict:
    return yaml.safe_load((_CFG / name).read_text())


def _safe_slope_z(rate_info: dict) -> float | None:
    sd = rate_info.get("rate_slope_sigma")
    if not sd or sd == 0:
        return None
    return (rate_info["rate_slope_3m"] - rate_info["rate_slope_mu"]) / sd


def build_bundle(
    ticker: str,
    as_of_date: date | None = None,
    lookback_days: int = 126,
) -> dict:
    """
    Orchestrate: data load → all Layer 1 computations → flag emission.
    Returns a dict matching the output contract.
    """
    if as_of_date is None:
        as_of_date = date.today()

    cfg_etf     = _load_yaml("sector_etf_map.yaml")
    cfg_comm    = _load_yaml("commodity_sensitivity.yaml")
    cfg_margin  = _load_yaml("sector_margin_metric.yaml")
    cfg_thresh  = _load_yaml("regime_thresholds.yaml")
    cfg_pol     = _load_yaml("political_exposure.yaml")
    cfg_tech    = _load_yaml("tech_disruption.yaml")
    cfg_env     = _load_yaml("environmental_exposure.yaml")
    cfg_legal   = _load_yaml("legal_regulatory_burden.yaml")

    window_start = as_of_date - timedelta(days=lookback_days + 5)
    long_start   = date(as_of_date.year - 3, as_of_date.month, as_of_date.day)

    # ── Metadata ────────────────────────────────────────────────────────────
    meta = load_ticker_metadata(ticker)
    sector = meta["gics_sector"] if meta else None
    sector_etf = cfg_etf.get(sector) if sector else None

    # ── Price series ────────────────────────────────────────────────────────
    p_company = load_prices(ticker, window_start, as_of_date)
    p_sector  = load_prices(sector_etf, window_start, as_of_date) if sector_etf else pd.Series(dtype=float)
    p_gspc    = load_prices("^GSPC", as_of_date - timedelta(days=300), as_of_date)  # 300 cal ≈ 210 trading days

    vix_3y = load_prices("^VIX", long_start, as_of_date)
    tnx_3y = load_prices("^TNX", long_start, as_of_date)

    commodity_tickers = ["CL=F", "GC=F", "HG=F", "NG=F"]
    p_commodities = load_prices_multi(commodity_tickers, window_start, as_of_date)

    # PESTEL macro ETFs (Political: UUP; Economic: HYG, IEF, TIP; Social: XLY, XLP)
    p_macro = load_macro_etf_prices(window_start, as_of_date)

    # ── Returns ─────────────────────────────────────────────────────────────
    r_company = daily_returns(p_company) if len(p_company) > 1 else pd.Series(dtype=float)
    r_sector  = daily_returns(p_sector)  if len(p_sector) > 1  else pd.Series(dtype=float)

    cum_company = cumulative_return(p_company) if len(p_company) > 1 else None
    cum_sector  = cumulative_return(p_sector)  if len(p_sector) > 1  else None

    # ── Relative strength ───────────────────────────────────────────────────
    # RS = company return / sector return. RS > 1 = LEADING, < 1 = LAGGING.
    rs_val = None
    if len(p_company) > 1 and len(p_sector) > 1:
        common = p_company.index.intersection(p_sector.index)
        if len(common) > 1:
            rs_val = relative_strength(p_company.loc[common], p_sector.loc[common])

    # ── Alpha / Beta ────────────────────────────────────────────────────────
    rf_annual = load_tnx_rate(as_of_date)
    ab = {"alpha_annualised": None, "beta": None, "n_obs": 0}
    if len(r_company) >= 40 and len(r_sector) >= 40:
        ab = alpha_beta(r_company, r_sector, rf_annual)

    # ── Rate beta (regress company on Δyield) ───────────────────────────────
    beta_rate = None
    if len(tnx_3y) > 1 and len(r_company) > 40:
        r_tnx = daily_returns(tnx_3y)
        common = r_company.index.intersection(r_tnx.index)
        if len(common) >= 40:
            import statsmodels.api as sm
            X = sm.add_constant(r_tnx.loc[common].values)
            y = r_company.loc[common].values
            try:
                beta_rate = float(sm.OLS(y, X).fit().params[1])
            except Exception:
                beta_rate = None

    # ── Regimes ─────────────────────────────────────────────────────────────
    vix_info  = vix_regime(vix_3y,
                           thresh_high=cfg_thresh["vix_zscore_high"],
                           thresh_low=cfg_thresh["vix_zscore_low"]) if len(vix_3y) > 63 else {}
    rate_info = rate_regime(tnx_3y,
                            sigma_mult=cfg_thresh["rate_slope_sigma_mult"]) if len(tnx_3y) > 63 else {}
    trend     = market_trend(p_gspc)

    slope_z = _safe_slope_z(rate_info)

    # ── Commodity ────────────────────────────────────────────────────────────
    comm_info = commodity_impact(sector or "", cfg_comm, p_commodities)

    # ── Peer gaps ────────────────────────────────────────────────────────────
    # Revenue growth: YoY annual (year_0 / year_1 - 1) — annual income_stmt
    #   reliably returns 2+ years; quarterly only returns 4 quarters so
    #   TTM-over-TTM was always n/a.
    # Margin: latest quarter from quarterly_financials (1 quarter is enough).
    peers = load_peers(ticker)
    margin_metric = cfg_margin.get(sector, cfg_margin.get("default", "operating_margin"))

    peer_gap_result: dict = {}

    # Company fundamentals
    ann_co = load_annual_revenue(ticker)          # [yr0, yr1] newest-first
    qtrs_co = load_ttm_quarters(ticker, n=4)
    latest_co = dict(qtrs_co[0]) if qtrs_co else {}

    # YoY revenue growth for company
    def _yoy(ann: list) -> float | None:
        if len(ann) >= 2 and ann[0] and ann[1] and ann[1] != 0:
            return ann[0] / ann[1] - 1
        return None

    g_co = _yoy(ann_co)

    # Peers
    peer_data = []
    for p in peers:
        ann_p  = load_annual_revenue(p)
        qtrs_p = load_ttm_quarters(p, n=4)
        latest_p = dict(qtrs_p[0]) if qtrs_p else {}
        peer_data.append({"ann": ann_p, "latest": latest_p})

    if peer_data:
        import statistics

        # Revenue growth gap
        g_peers = [_yoy(pd_["ann"]) for pd_ in peer_data]
        g_peers = [x for x in g_peers if x is not None]
        g_peer_med = statistics.median(g_peers) if g_peers else None
        rev_growth_gap = (g_co - g_peer_med) if (g_co is not None and g_peer_med is not None) else None

        # Margin gap (quarterly latest)
        from layer1.peer_gaps import compute_margin
        m_co = compute_margin(latest_co, margin_metric)
        m_peers = [compute_margin(pd_["latest"], margin_metric) for pd_ in peer_data]
        m_peers = [x for x in m_peers if x is not None]
        m_peer_med = statistics.median(m_peers) if m_peers else None
        margin_gap = (m_co - m_peer_med) if (m_co is not None and m_peer_med is not None) else None

        peer_gap_result = {
            "rev_growth_company":     g_co,
            "rev_growth_peer_median": g_peer_med,
            "rev_growth_gap":         rev_growth_gap,
            "margin_metric":          margin_metric,
            "margin_company":         m_co,
            "margin_peer_median":     m_peer_med,
            "margin_gap":             margin_gap,
        }

    # ── PESTEL signals ──────────────────────────────────────────────────────────
    # --- R&D and CapEx for company and peers (Technological / Social) --------
    rd_co    = load_rd_revenue_ratio(ticker)
    capex_co = load_capex_revenue_ratio(ticker)
    rd_peers    = [load_rd_revenue_ratio(p) for p in peers]
    capex_peers = [load_capex_revenue_ratio(p) for p in peers]

    # --- Political -----------------------------------------------------------
    pol = political_signals(
        sector=sector or "",
        cfg=cfg_pol,
        p_uup=p_macro.get("UUP", pd.Series(dtype=float)),
    )

    # --- Economic (extends existing signals with credit + inflation) ----------
    eco = economic_signals(
        sector_rs_6m        = rs_val,
        alpha_annualised     = ab.get("alpha_annualised"),
        vix_zscore          = vix_info.get("vix_zscore"),
        rate_slope_z        = slope_z,
        market_trend        = trend,
        peer_rev_growth_gap = peer_gap_result.get("rev_growth_gap"),
        peer_margin_gap     = peer_gap_result.get("margin_gap"),
        p_hyg=p_macro.get("HYG", pd.Series(dtype=float)),
        p_ief=p_macro.get("IEF", pd.Series(dtype=float)),
        p_tip=p_macro.get("TIP", pd.Series(dtype=float)),
    )

    # --- Social (consumer sentiment + R&D gap) --------------------------------
    soc = social_signals(
        p_xly             = p_macro.get("XLY", pd.Series(dtype=float)),
        p_xlp             = p_macro.get("XLP", pd.Series(dtype=float)),
        rd_ratio_company  = rd_co,
        rd_ratio_peers    = rd_peers,
    )

    # --- Technological (R&D gap, CapEx gap, disruption exposure) --------------
    tech = technological_signals(
        sector              = sector or "",
        cfg                 = cfg_tech,
        rd_ratio_company    = rd_co,
        rd_ratio_peers      = rd_peers,
        capex_ratio_company = capex_co,
        capex_ratio_peers   = capex_peers,
    )

    # --- Environmental (carbon intensity + env burden + commodity) -----------
    env = environmental_signals(
        sector              = sector or "",
        cfg                 = cfg_env,
        commodity_impact_raw= comm_info.get("commodity_impact_raw"),
    )

    # --- Legal (regulatory complexity + litigation exposure) -----------------
    leg = legal_signals(
        sector = sector or "",
        cfg    = cfg_legal,
    )

    pestel_bundle = {
        # Political
        "political_govt_dependency":   pol["political_govt_dependency"],
        "political_trade_sensitivity": pol["political_trade_sensitivity"],
        "uup_return_6m":               pol["uup_return_6m"],
        "political_fx_impact":         pol["political_fx_impact"],
        "political_score_raw":         pol["political_score_raw"],
        # Economic
        "credit_spread_z":             eco["credit_spread_z"],
        "inflation_z":                 eco["inflation_z"],
        "economic_score_raw":          eco["economic_score_raw"],
        # Social
        "consumer_sentiment_z":        soc["consumer_sentiment_z"],
        "rd_intensity_company":        soc["rd_intensity_company"],
        "rd_intensity_peer_med":       soc["rd_intensity_peer_med"],
        "rd_intensity_gap":            soc["rd_intensity_gap"],
        "social_score_raw":            soc["social_score_raw"],
        # Technological
        "tech_disruption_exposure":    tech["tech_disruption_exposure"],
        "rd_capex_importance":         tech["rd_capex_importance"],
        "tech_rd_intensity_gap":       tech["rd_intensity_gap"],
        "capex_intensity_company":     tech["capex_intensity_company"],
        "capex_intensity_peer_med":    tech["capex_intensity_peer_med"],
        "capex_intensity_gap":         tech["capex_intensity_gap"],
        "tech_score_raw":              tech["tech_score_raw"],
        # Environmental
        "carbon_intensity":            env["carbon_intensity"],
        "environmental_regulatory_burden": env["environmental_regulatory_burden"],
        "environmental_score_raw":     env["environmental_score_raw"],
        # Legal
        "regulatory_complexity":       leg["regulatory_complexity"],
        "litigation_exposure":         leg["litigation_exposure"],
        "legal_score_raw":             leg["legal_score_raw"],
    }

    # ── Flags ────────────────────────────────────────────────────────────────
    flags = emit_flags(
        sector_rs_6m        = rs_val,
        rate_regime         = rate_info.get("rate_regime"),
        beta_rate           = beta_rate,
        vix_zscore          = vix_info.get("vix_zscore"),
        market_trend        = trend,
        commodity_tag       = comm_info.get("commodity_tag"),
        peer_rev_growth_gap = peer_gap_result.get("rev_growth_gap"),
        peer_margin_gap     = peer_gap_result.get("margin_gap"),
        rs_lead_thresh      = 1.0,
        rs_lag_thresh       = 1.0,
        rate_beta_thresh    = -2.0,
        vix_high            = cfg_thresh["vix_zscore_high"],
        vix_low             = cfg_thresh["vix_zscore_low"],
        # PESTEL-driven flags
        political_fx_impact        = pol["political_fx_impact"],
        political_govt_dependency  = pol["political_govt_dependency"],
        credit_spread_z            = eco["credit_spread_z"],
        inflation_z                = eco["inflation_z"],
        consumer_sentiment_z       = soc["consumer_sentiment_z"],
        rd_intensity_gap           = soc["rd_intensity_gap"],
        capex_intensity_gap        = tech["capex_intensity_gap"],
        tech_disruption_exposure   = tech["tech_disruption_exposure"],
        environmental_score_raw    = env["environmental_score_raw"],
        legal_score_raw            = leg["legal_score_raw"],
    )

    bundle = {
        "ticker":                   ticker,
        "as_of_date":               as_of_date.isoformat(),
        "sector":                   sector,
        "sector_etf":               sector_etf,
        # Returns
        "company_cum_return_6m":    cum_company,
        "sector_cum_return_6m":     cum_sector,
        # Relative strength
        "sector_rs_6m":             rs_val,
        # Alpha / Beta
        "company_alpha_annualised": ab.get("alpha_annualised"),
        "company_beta":             ab.get("beta"),
        "ab_n_obs":                 ab.get("n_obs"),
        # Rate beta
        "beta_rate":                beta_rate,
        # Regimes
        "vix_current":              vix_info.get("vix_current"),
        "vix_zscore":               vix_info.get("vix_zscore"),
        "vol_regime":               vix_info.get("vol_regime"),
        "rate_slope_3m":            rate_info.get("rate_slope_3m"),
        "rate_slope_z":             slope_z,
        "rate_regime":              rate_info.get("rate_regime"),
        "market_trend":             trend,
        # Commodity
        "commodity_impact_raw":     comm_info.get("commodity_impact_raw"),
        "commodity_tag":            comm_info.get("commodity_tag"),
        # Peer gaps
        "peer_rev_growth_gap":      peer_gap_result.get("rev_growth_gap"),
        "peer_margin_gap":          peer_gap_result.get("margin_gap"),
        "margin_metric":            margin_metric,
        "peer_margin_company":      peer_gap_result.get("margin_company"),
        "peer_margin_median":       peer_gap_result.get("margin_peer_median"),
        # Flags
        "flags":                    flags,
        # Misc
        "rf_annual":                rf_annual,
        # PESTEL sub-bundle (all quantitative signals by dimension)
        "pestel":                   pestel_bundle,
    }
    return bundle
