"""
Step 0: Parse a natural language query into a structured QueryIntent.
Uses the LLM to extract ticker, time scope, hypothesis, and RAG keywords.
"""
import os
import json
from dotenv import load_dotenv
from models.intent import QueryIntent

load_dotenv()

PARSE_PROMPT = """
You are a query parser for a financial analysis agent.
Extract structured intent from the user's query.

Return ONLY valid JSON matching this schema:
{{
  "ticker": "<company ticker symbol>",
  "time_scope_quarters": <integer: 4 for "last year", 8 for "last 2 years">,
  "primary_signals": [<list of: "operating_leverage", "margins", "revenue_growth", "cash_quality", "working_capital">],
  "hypothesis": <null or one of: "management_credibility", "growth_investment_validity", "margin_compression_cause">,
  "rag_keywords": [<3-5 search phrases to find relevant management statements>],
  "layer2_question": "<specific yes/no or evaluative question for the LLM reasoning layer>"
}}

User query: {query}
"""

def parse_query(raw_query: str) -> QueryIntent:
    """Parse a natural language query into a structured QueryIntent."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    response_text = _call_llm(PARSE_PROMPT.format(query=raw_query), provider)

    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError:
        words = raw_query.upper().split()
        ticker = next((w for w in words if 2 <= len(w) <= 5 and w.isalpha()), "UNKNOWN")
        parsed = {
            "ticker": ticker,
            "time_scope_quarters": 4,
            "primary_signals": ["operating_leverage", "margins"],
            "hypothesis": "management_credibility",
            "rag_keywords": ["growth investment", "strategic hiring", "operating leverage"],
            "layer2_question": "Does the management narrative match the financial trajectory?"
        }

    return QueryIntent(ticker=parsed["ticker"], raw_query=raw_query, **{
        k: v for k, v in parsed.items() if k != "ticker"
    })


def _call_llm(prompt: str, provider: str) -> str:
    """Call the configured LLM provider."""
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return resp.choices[0].message.content

    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        resp = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6"),
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.content[0].text

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}. Use 'openai' or 'anthropic'.")
