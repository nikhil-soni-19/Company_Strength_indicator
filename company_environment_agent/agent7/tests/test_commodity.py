import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml
from layer1.commodity import commodity_impact

_CFG = Path(__file__).parent.parent / "config" / "commodity_sensitivity.yaml"
SENS = yaml.safe_load(_CFG.read_text())


def _series(start, end, n=10):
    vals = [start + (end - start) * i / (n - 1) for i in range(n)]
    return pd.Series(vals, index=pd.date_range("2024-01-01", periods=n, freq="B"))


def test_energy_rising_oil_tailwind():
    prices = {"CL=F": _series(70, 80), "NG=F": _series(2, 2.5)}
    result = commodity_impact("Energy", SENS, prices)
    assert result["commodity_tag"] == "COMMODITY_TAILWIND"
    assert result["commodity_impact_raw"] > 0


def test_industrials_rising_oil_headwind():
    prices = {"CL=F": _series(70, 80), "HG=F": _series(4, 4.5)}
    result = commodity_impact("Industrials", SENS, prices)
    assert result["commodity_tag"] == "COMMODITY_HEADWIND"
    assert result["commodity_impact_raw"] < 0


def test_tech_not_applicable():
    prices = {"CL=F": _series(70, 80)}
    result = commodity_impact("Information Technology", SENS, prices)
    assert result["commodity_tag"] == "NOT_APPLICABLE"


def test_missing_commodity_data():
    # Sector mapped but commodity data absent
    result = commodity_impact("Energy", SENS, {})
    assert result["commodity_tag"] == "NO_DATA"
