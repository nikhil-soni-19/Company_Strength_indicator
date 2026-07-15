"""
LLM-based query parser.

Takes any natural language query and extracts the target ticker symbol.
Falls back to regex for obvious direct ticker inputs.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# Regex: 1-5 uppercase letters optionally preceded by obvious context
_TICKER_RE = re.compile(r'\b([A-Z]{1,5})\b')

# Common company-name → ticker map for fast resolution without an LLM call
_NAME_MAP: dict[str, str] = {
    "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL", "alphabet": "GOOGL",
    "amazon": "AMZN", "meta": "META", "facebook": "META", "nvidia": "NVDA",
    "tesla": "TSLA", "netflix": "NFLX", "salesforce": "CRM", "adobe": "ADBE",
    "oracle": "ORCL", "intel": "INTC", "amd": "AMD", "qualcomm": "QCOM",
    "broadcom": "AVGO", "jpmorgan": "JPM", "jp morgan": "JPM",
    "bank of america": "BAC", "goldman sachs": "GS", "morgan stanley": "MS",
    "berkshire": "BRK.B", "exxon": "XOM", "chevron": "CVX",
    "johnson & johnson": "JNJ", "j&j": "JNJ", "pfizer": "PFE",
    "unitedhealth": "UNH", "walmart": "WMT", "costco": "COST",
    "coca-cola": "KO", "coke": "KO", "pepsi": "PEP", "pepsico": "PEP",
    "visa": "V", "mastercard": "MA", "paypal": "PYPL",
    "boeing": "BA", "caterpillar": "CAT", "3m": "MMM",
    "home depot": "HD", "lowes": "LOW", "target": "TGT",
    "disney": "DIS", "comcast": "CMCSA", "att": "T",
    "verizon": "VZ", "t-mobile": "TMUS",
}


# All-English words that look like tickers — never treat these as symbols
_ENGLISH_WORDS = {
    "A", "I", "IT", "OR", "AT", "IN", "ON", "UP", "DO", "GO", "SO", "BE",
    "AS", "BY", "IS", "IF", "OF", "TO", "AN", "US", "ME", "HE", "WE",
    "FOR", "THE", "AND", "BUT", "NOT", "RUN", "NOW", "GET", "SET", "PUT",
    "LET", "CAN", "MAY", "HAS", "HAD", "DID", "WAS", "ARE", "HIM", "HER",
    "ITS", "OUR", "YOU", "ALL", "ANY", "HOW", "WHAT", "WHEN", "WHO", "WHY",
    "TELL", "SHOW", "GIVE", "TAKE", "MAKE", "LOOK", "DOES", "THAT", "THIS",
    "WITH", "FROM", "WILL", "WELL", "JUST", "ALSO", "BEEN", "INTO", "OVER",
    "THAN", "THEN", "THEM", "THEY", "WHAT", "LIKE", "VERY", "SOME", "MORE",
    "GOOD", "BEST", "LAST", "NEXT", "EACH", "BOTH", "MUCH", "MANY", "SUCH",
    "EVEN", "BACK", "LONG", "HELP", "MOST", "HIGH", "REAL", "ONLY", "SAME",
    "WANT", "NEED", "KNOW", "FEEL", "SEEM", "KEEP", "THINK", "ABOUT",
    "AFTER", "AGAIN", "BELOW", "COULD", "EVERY", "FIRST", "FOUND", "GOING",
    "GREAT", "MIGHT", "OTHER", "RIGHT", "STILL", "THEIR", "THERE", "THESE",
    "THOSE", "THREE", "TODAY", "UNDER", "UNTIL", "USING", "WHICH", "WHILE",
    "WOULD", "YEARS", "SACHS", "ABOUT", "POINT", "GIVEN", "SINCE",
}

# Trigger words that strongly signal the next token is a ticker
_TRIGGER_WORDS = {
    "analyse", "analyze", "run", "check", "for", "about", "on",
    "ticker", "stock", "company", "firm", "corp",
}


def _fast_lookup(query: str) -> str | None:
    """Resolve obvious cases without an LLM call."""
    q = query.strip()

    # 1. Company name lookup first (most reliable)
    ql = q.lower()
    for name, ticker in _NAME_MAP.items():
        if name in ql:
            return ticker

    # 2. Ticker after a trigger word: "analyse AAPL", "run MSFT", "for NVDA"
    words = q.split()
    for i, word in enumerate(words):
        if word.lower().rstrip(".,?!:;") in _TRIGGER_WORDS and i + 1 < len(words):
            candidate = words[i + 1].strip(".,?!:;\"'").upper()
            if (re.fullmatch(r'[A-Z]{1,5}(\.[A-Z])?', candidate)
                    and candidate not in _ENGLISH_WORDS
                    and len(candidate) >= 2):
                return candidate

    # 3. Query IS just a ticker symbol (e.g. "AAPL" or "AAPL?")
    bare = q.strip(".,?!:;\"' ").upper()
    if re.fullmatch(r'[A-Z]{1,5}(\.[A-Z])?', bare) and bare not in _ENGLISH_WORDS:
        return bare

    # 4. Single uppercase token anywhere that's not a common English word
    for word in words:
        candidate = word.strip(".,?!:;\"'").upper()
        if (re.fullmatch(r'[A-Z]{2,5}(\.[A-Z])?', candidate)
                and candidate not in _ENGLISH_WORDS):
            return candidate

    return None


def parse_ticker(query: str) -> str | None:
    """
    Extract a ticker symbol from any natural language query.

    Strategy:
    1. Fast regex/name lookup (no API call)
    2. LLM call if ambiguous or company name not in fast map
    """
    fast = _fast_lookup(query)
    if fast:
        return fast

    # LLM fallback
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    prompt = f"""You are a financial assistant. Extract the stock ticker symbol from the user's query.

Rules:
- Return ONLY the ticker symbol in uppercase (e.g. AAPL, MSFT, GOOGL)
- If the user mentions a company by name, return its primary US exchange ticker
- If multiple tickers are mentioned, return the first / most prominent one
- If no company or ticker can be identified, return exactly: UNKNOWN
- Return nothing else — just the ticker symbol or UNKNOWN

User query: {query}"""

    try:
        response = client.messages.create(
            model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            max_tokens=16,
            messages=[{"role": "user", "content": prompt}],
        )
        result = response.content[0].text.strip().upper()
        # Validate: must look like a ticker
        if re.fullmatch(r'[A-Z]{1,5}(\.[A-Z])?', result) and result != "UNKNOWN":
            return result
        return None
    except Exception as e:
        print(f"  [WARN] Query parser LLM call failed: {e}")
        return None
