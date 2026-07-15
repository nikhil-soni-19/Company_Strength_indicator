"""Resolve tickers from free-form user text.

The agent should be forgiving — a trader who types "Aple", "Apples", or
"apple inc" almost certainly means **AAPL**. This module turns natural
language into one or more canonical ticker symbols using four layers:

1. **Direct symbol regex** — explicit upper-case mentions like ``AAPL`` or
   ``BRK.B`` are picked up first.
2. **Curated alias dictionary** — common company names and brand variants
   are mapped to their primary US listing.
3. **Fuzzy matching (difflib)** — uni- and bi-grams from the question are
   matched against the alias table with a similarity cutoff, which catches
   typos such as ``Aple`` → ``Apple`` → ``AAPL`` or ``Goolge`` → ``Google``.
4. **LLM fallback** (``resolve_tickers_smart``) — if layers 1-3 return
   nothing, a cheap GPT-4o-mini call resolves company names the alias table
   doesn't cover (e.g. "ZScaler" → ``ZS``). Each LLM-returned symbol is
   validated against yfinance before being accepted, preventing hallucination.

Order is preserved (first mentioned, first returned) and results are
de-duplicated. A short blacklist filters out English words that happen to
look like tickers.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
from typing import Iterable, Optional

log = logging.getLogger(__name__)

TICKER_BLACKLIST: frozenset[str] = frozenset({
    "I", "A", "AN", "THE", "FOR", "IS", "OF", "ON", "AT", "BY", "AS", "TO",
    "IN", "IT", "WE", "OR", "AND", "BUT", "VS", "VERSUS", "ETF", "LLC",
    "INC", "USD", "EUR", "GBP", "ADV", "DTL", "TWAP", "VWAP", "CV",
    "TIER", "OK", "NO", "YES", "BE", "DO", "GO", "IF", "SO", "UP",
    "LLM", "AI", "API", "ML", "OHLCV", "ETC",
})

# ticker -> list of lowercase aliases (names, abbreviations, common typos)
_ALIASES: dict[str, list[str]] = {
    "AAPL": ["apple", "apple inc", "apple incorporated"],
    "MSFT": ["microsoft", "microsoft corp", "microsoft corporation"],
    "GOOGL": ["google", "alphabet", "alphabet inc"],
    "GOOG": ["alphabet class c"],
    "AMZN": ["amazon", "amazon.com"],
    "META": ["meta", "facebook", "meta platforms"],
    "TSLA": ["tesla", "tesla inc", "tesla motors"],
    "NVDA": ["nvidia", "nvidia corp"],
    "NFLX": ["netflix"],
    "GME": ["gamestop", "game stop"],
    "AMC": ["amc entertainment"],
    "BB": ["blackberry"],
    "JPM": ["jpmorgan", "jp morgan", "jpmorgan chase"],
    "BAC": ["bank of america", "bofa"],
    "GS": ["goldman", "goldman sachs"],
    "MS": ["morgan stanley"],
    "WFC": ["wells fargo"],
    "C": ["citigroup", "citi"],
    "BRK.B": ["berkshire", "berkshire hathaway"],
    "V": ["visa"],
    "MA": ["mastercard"],
    "DIS": ["disney", "walt disney"],
    "KO": ["coca cola", "coke", "coca-cola"],
    "PEP": ["pepsi", "pepsico"],
    "MCD": ["mcdonalds", "mcdonald", "mcdonald's"],
    "SBUX": ["starbucks"],
    "NKE": ["nike"],
    "WMT": ["walmart", "wal-mart"],
    "TGT": ["target corp"],
    "COST": ["costco"],
    "HD": ["home depot"],
    "LOW": ["lowes", "lowe's"],
    "BA": ["boeing"],
    "GE": ["general electric"],
    "F": ["ford", "ford motor"],
    "GM": ["general motors"],
    "T": ["at&t", "att"],
    "VZ": ["verizon"],
    "TMUS": ["t-mobile", "tmobile"],
    "INTC": ["intel"],
    "AMD": ["advanced micro devices"],
    "ORCL": ["oracle"],
    "IBM": ["international business machines"],
    "CSCO": ["cisco"],
    "ADBE": ["adobe"],
    "CRM": ["salesforce"],
    "PYPL": ["paypal"],
    "SQ": ["block inc", "square inc"],
    "UBER": ["uber"],
    "LYFT": ["lyft"],
    "ABNB": ["airbnb"],
    "PLTR": ["palantir"],
    "SNOW": ["snowflake"],
    "SHOP": ["shopify"],
    "ROKU": ["roku"],
    "SPOT": ["spotify"],
    "ZM": ["zoom"],
    "BABA": ["alibaba"],
    "JD": ["jd.com"],
    "BIDU": ["baidu"],
    "TSM": ["taiwan semiconductor", "tsmc"],
    "ASML": ["asml"],
    "PFE": ["pfizer"],
    "JNJ": ["johnson and johnson", "johnson & johnson"],
    "MRK": ["merck"],
    "LLY": ["eli lilly", "lilly"],
    "ABBV": ["abbvie"],
    "UNH": ["united health", "unitedhealth"],
    "CVS": ["cvs health", "cvs pharmacy"],
    "WBA": ["walgreens"],
    "XOM": ["exxon", "exxon mobil", "exxonmobil"],
    "CVX": ["chevron"],
    "BP": ["british petroleum"],
    "SHEL": ["shell"],
    "COP": ["conocophillips", "conoco"],
    "SLB": ["schlumberger"],
}

_REVERSE_INDEX: dict[str, str] = {
    alias.lower(): ticker
    for ticker, aliases in _ALIASES.items()
    for alias in aliases
}
# Treat each ticker's own lowercase form as an alias too. This lets users
# type lower-case ticker symbols (e.g. "nflx", "amd", "aapl") without
# triggering only the upper-case regex path.
for _ticker in _ALIASES:
    _REVERSE_INDEX.setdefault(_ticker.lower(), _ticker)

_ALL_ALIASES: tuple[str, ...] = tuple(_REVERSE_INDEX.keys())

_MIN_FUZZY_LEN: int = 4  # below this length only exact alias matches are accepted

_DIRECT_TICKER_RE = re.compile(r"\b([A-Z]{1,5}(?:\.[A-Z]{1,2})?)\b")
_WORD_SPLIT_RE = re.compile(r"[^A-Za-z&.\-']+")
_MAG7_RE = re.compile(r"\b(?:MAG\s*7|MAGNIFICENT\s+(?:7|SEVEN))\b", re.IGNORECASE)
_EXCHANGE_SUFFIX_RE = re.compile(
    r"\b([A-Z]{1,5}(?:\.[A-Z]{1,2})?)\s+"
    r"(US|LN|JP|HK|CN|CH|GR|GY|FP|IM|SW|AU|SS|SZ|TT|KS|KQ|IN|TO|CN|NA|SM)"
    r"(?:\s+EQUITY)?\b"
)

DEFAULT_FUZZY_CUTOFF: float = 0.80

_MAG7_TICKERS: tuple[str, ...] = ("AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA")


# ── LLM + yfinance fallback layer ────────────────────────────────────────────

def _llm_resolve(text: str, api_key: str, model: str = "gpt-4o-mini") -> list[str]:
    """Call the LLM to extract company references and return canonical tickers.

    Returns a list of uppercase ticker strings. If the call fails for any
    reason (network, bad JSON, etc.) it returns an empty list so the caller
    can degrade gracefully.
    """
    try:
        from openai import OpenAI  # optional dependency — already used elsewhere

        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a stock ticker resolver. Given a user query, identify all "
                        "company names, brand names, or informal references to publicly traded "
                        "stocks and return their official primary US exchange ticker symbols.\n\n"
                        "Rules:\n"
                        "- Return ONLY a JSON object with a single key 'tickers' containing "
                        "a list of ticker strings. Example: {\"tickers\": [\"ZS\", \"PANW\"]}\n"
                        "- Only include tickers you are highly confident about.\n"
                        "- Use the primary US listing (e.g. 'ZScaler' -> 'ZS', "
                        "'Palo Alto Networks' -> 'PANW', 'Crowdstrike' -> 'CRWD').\n"
                        "- If an explicit ticker symbol already appears verbatim, include it.\n"
                        "- If you cannot resolve a name with high confidence, omit it entirely.\n"
                        "- Return ONLY the JSON object — no explanation, no markdown fences."
                    ),
                },
                {"role": "user", "content": f"Query: {text}"},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = (completion.choices[0].message.content or "").strip()
        data = json.loads(raw)
        return [t.upper() for t in data.get("tickers", []) if isinstance(t, str) and t]
    except Exception as exc:  # pragma: no cover
        log.debug("LLM ticker resolve failed: %s", exc)
        return []


def _validate_via_yfinance(symbol: str) -> bool:
    """Return True if yfinance can confirm ``symbol`` is a live tradeable security.

    Uses ``fast_info`` (lightweight, no full info download) and checks that
    ``last_price`` is non-None — a reliable signal that the ticker exists and
    trades on an exchange. Falls back to False on any exception.
    """
    try:
        import yfinance as yf

        fast = yf.Ticker(symbol).fast_info
        return fast.last_price is not None
    except Exception as exc:  # pragma: no cover
        log.debug("yfinance validation failed for %s: %s", symbol, exc)
        return False


_HAS_LOWERCASE = re.compile(r"[a-z]")


def _query_is_natural_language(text: str) -> bool:
    """Return True when the query contains lowercase letters.

    A query like ``"AAPL TSLA GME"`` is all-caps and almost certainly already
    contains explicit ticker symbols — the LLM adds nothing there.  A query
    like ``"How liquid is ZScaler?"`` or ``"Compare Palo Alto to Microsoft"``
    has lowercase words that the heuristic may not fully resolve, so the LLM
    layer is worth firing.
    """
    return bool(_HAS_LOWERCASE.search(text))


def resolve_tickers_smart(
    text: str,
    fuzzy_cutoff: float = DEFAULT_FUZZY_CUTOFF,
    api_key: Optional[str] = None,
    llm_model: str = "gpt-4o-mini",
) -> list[str]:
    """Four-layer resolver: heuristic + optional LLM → yfinance validation.

    Layers 1-3 (direct regex, alias dict, fuzzy match) always run first and
    are fast and free.

    Layer 4 (LLM) fires when the query looks like natural language (contains
    lowercase letters), meaning the user may have written a company name that
    the alias table doesn't cover.  The LLM result is *merged* with the
    heuristic result so partial matches (e.g. heuristic finds MSFT but misses
    PANW in "Compare Palo Alto to Microsoft") are augmented rather than lost.

    Every ticker returned by the LLM is confirmed against yfinance before
    being accepted, preventing hallucinated symbols from reaching the scoring
    engine.  If the LLM or yfinance call fails for any reason, we fall back
    silently to whatever the heuristic found.
    """
    # ── Layers 1-3: heuristic ────────────────────────────────────────────────
    normalized_text, seed_tickers = _normalize_special_references(text)
    heuristic: list[str] = _resolve_tickers_from_normalized(
        normalized_text,
        seed_tickers=seed_tickers,
        fuzzy_cutoff=fuzzy_cutoff,
    )

    # ── Layer 4: LLM — only for natural-language queries ─────────────────────
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key or not _query_is_natural_language(normalized_text):
        return heuristic  # fast path: all-caps or offline

    llm_candidates = _llm_resolve(normalized_text, api_key=key, model=llm_model)
    if not llm_candidates:
        return heuristic

    # ── yfinance confirmation: drop hallucinated tickers ─────────────────────
    seen: set[str] = set(heuristic)
    merged: list[str] = list(heuristic)
    for symbol in llm_candidates:
        if symbol in seen:
            continue  # heuristic already found it, no need to re-validate
        if _validate_via_yfinance(symbol):
            seen.add(symbol)
            merged.append(symbol)
        else:
            log.debug(
                "LLM returned '%s' but yfinance could not confirm it — dropped.", symbol
            )

    return merged


def resolve_tickers(text: str, fuzzy_cutoff: float = DEFAULT_FUZZY_CUTOFF) -> list[str]:
    """Return the unique tickers referenced in ``text``, in mention order.

    Picks up explicit symbols (``AAPL``), known company names (``apple``,
    ``microsoft``), and typo'd variants (``Aple``, ``Apples``, ``Goolge``).
    """
    normalized_text, seed_tickers = _normalize_special_references(text)
    return _resolve_tickers_from_normalized(
        normalized_text,
        seed_tickers=seed_tickers,
        fuzzy_cutoff=fuzzy_cutoff,
    )


def _resolve_tickers_from_normalized(
    text: str,
    seed_tickers: Iterable[str] = (),
    fuzzy_cutoff: float = DEFAULT_FUZZY_CUTOFF,
) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()

    def _add(ticker: str) -> None:
        if ticker not in seen:
            seen.add(ticker)
            found.append(ticker)

    for ticker in seed_tickers:
        _add(ticker)

    for match in _DIRECT_TICKER_RE.finditer(text):
        symbol = match.group(1)
        if symbol in TICKER_BLACKLIST:
            continue
        if symbol in _ALIASES or _is_plausible_ticker(symbol):
            _add(symbol)

    for ticker in _resolve_by_alias(text, fuzzy_cutoff):
        _add(ticker)

    return found


def _normalize_special_references(text: str) -> tuple[str, list[str]]:
    """Handle market shorthand before generic uppercase ticker extraction."""
    seed_tickers: list[str] = []

    def _expand_mag7(match: re.Match[str]) -> str:
        seed_tickers.extend(_MAG7_TICKERS)
        return " "

    normalized = _MAG7_RE.sub(_expand_mag7, text)
    normalized = _EXCHANGE_SUFFIX_RE.sub(r"\1", normalized)
    return normalized, seed_tickers


def resolve_ticker(text: str, fuzzy_cutoff: float = DEFAULT_FUZZY_CUTOFF) -> str | None:
    """First-ticker convenience wrapper."""
    tickers = resolve_tickers(text, fuzzy_cutoff=fuzzy_cutoff)
    return tickers[0] if tickers else None


def known_tickers() -> Iterable[str]:
    """All tickers that have at least one curated alias."""
    return _ALIASES.keys()


def _is_plausible_ticker(symbol: str) -> bool:
    """Heuristic for the direct-regex pass.

    * A single letter is accepted only if it is a known curated ticker
      (``C``, ``F``, ``T``, ``V``).
    * If the lowercase form of ``symbol`` is a known company-name alias
      (e.g. "APPLE", "TESLA" shouted by the user), the direct path defers
      to alias resolution so the symbol is rewritten to its real ticker.
    """
    if symbol.lower() in _REVERSE_INDEX and symbol not in _ALIASES:
        return False
    if len(symbol) == 1:
        return symbol in _ALIASES
    return True


def _resolve_by_alias(text: str, fuzzy_cutoff: float) -> list[str]:
    lowered = text.lower()
    tokens = [t for t in _WORD_SPLIT_RE.split(lowered) if t]
    candidates: list[str] = []
    for i, tok in enumerate(tokens):
        candidates.append(tok)
        if i + 1 < len(tokens):
            candidates.append(f"{tok} {tokens[i + 1]}")
        if i + 2 < len(tokens):
            candidates.append(f"{tok} {tokens[i + 1]} {tokens[i + 2]}")

    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if len(candidate) < 3:
            continue
        ticker = _match_alias(candidate, fuzzy_cutoff)
        if ticker is not None and ticker not in seen:
            seen.add(ticker)
            out.append(ticker)
    return out


def _match_alias(candidate: str, fuzzy_cutoff: float) -> str | None:
    if candidate in _REVERSE_INDEX:
        return _REVERSE_INDEX[candidate]
    if len(candidate) < _MIN_FUZZY_LEN:
        # Short tokens (e.g. "for", "and", "the") are too prone to false
        # fuzzy matches against 3-4 letter tickers like FORD or AMD.
        return None
    matches = difflib.get_close_matches(candidate, _ALL_ALIASES, n=1, cutoff=fuzzy_cutoff)
    if matches:
        return _REVERSE_INDEX[matches[0]]
    return None
