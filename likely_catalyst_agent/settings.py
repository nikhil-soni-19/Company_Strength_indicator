"""
Central configuration management for the Likely Catalyst Agent.
All settings are loaded from environment variables with sane defaults.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from functools import lru_cache


@dataclass
class DatabaseConfig:
    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    name: str = os.getenv("POSTGRES_DB", "catalyst_agent")
    user: str = os.getenv("POSTGRES_USER", "postgres")
    password: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    pool_size: int = int(os.getenv("DB_POOL_SIZE", "10"))
    max_overflow: int = int(os.getenv("DB_MAX_OVERFLOW", "20"))

    @property
    def url(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @property
    def sync_url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


@dataclass
class QdrantConfig:
    host: str = os.getenv("QDRANT_HOST", "localhost")
    port: int = int(os.getenv("QDRANT_PORT", "6333"))
    api_key: Optional[str] = os.getenv("QDRANT_API_KEY")
    collection_filings: str = "filing_embeddings"
    collection_catalysts: str = "catalyst_embeddings"
    collection_earnings: str = "earnings_call_embeddings"
    vector_size: int = 768  # FinBERT / SentenceTransformer output


@dataclass
class LLMConfig:
    # FinBERT
    finbert_model: str = os.getenv("FINBERT_MODEL", "ProsusAI/finbert")
    finbert_max_length: int = 512

    # Sentence Transformer for embeddings
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2"
    )
    embedding_model_neon: str = "BAAI/bge-large-en-v1.5"  # 1024-dim, matches Neon
    # Switch embedding_model to this value to enable pgvector search

    # Llama / Mistral for narrative understanding
    narrative_model: str = os.getenv(
        "NARRATIVE_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct"
    )
    narrative_max_new_tokens: int = 1024
    narrative_temperature: float = 0.1

    # BART for summarization
    bart_model: str = os.getenv("BART_MODEL", "facebook/bart-large-cnn")

    # Use GPU if available
    device: str = os.getenv("DEVICE", "auto")  # "auto", "cpu", "cuda", "mps"
    use_quantization: bool = os.getenv("USE_QUANTIZATION", "true").lower() == "true"
    use_flash_attention: bool = os.getenv("USE_FLASH_ATTENTION", "false").lower() == "true"


@dataclass
class SECConfig:
    edgar_base_url: str = "https://data.sec.gov"
    edgar_submissions_url: str = "https://data.sec.gov/submissions"
    edgar_company_facts_url: str = "https://data.sec.gov/api/xbrl/companyfacts"
    # Static bulk file; lives on www.sec.gov, not data.sec.gov (404 if wrong host).
    company_tickers_url: str = os.getenv(
        "SEC_COMPANY_TICKERS_URL",
        "https://www.sec.gov/files/company_tickers.json",
    )
    user_agent: str = os.getenv(
        "SEC_USER_AGENT", "CatalystAgent research@example.com"
    )
    rate_limit_calls: int = 10       # calls per second (SEC allows 10/sec)
    rate_limit_period: float = 1.0
    request_timeout: int = 30


@dataclass
class FMPConfig:
    """Financial Modeling Prep API (optional; falls back to yfinance)."""
    api_key: Optional[str] = os.getenv("FMP_API_KEY")
    base_url: str = os.getenv("FMP_BASE_URL", "https://financialmodelingprep.com/api/v3")
    enabled: bool = bool(os.getenv("FMP_API_KEY"))


@dataclass
class UpcomingPredictorConfig:
    """Pre-event upcoming earnings predictor."""
    default_days_ahead: int = int(os.getenv("UPCOMING_DAYS_AHEAD", "30"))
    max_days_ahead: int = int(os.getenv("UPCOMING_MAX_DAYS", "90"))
    return_horizon_days: int = 20  # primary calibration window
    # Default class-mean returns when DB has no history
    default_bull_return: float = float(os.getenv("DEFAULT_BULL_RETURN", "0.058"))
    default_bear_return: float = float(os.getenv("DEFAULT_BEAR_RETURN", "-0.042"))
    default_return_std: float = float(os.getenv("DEFAULT_RETURN_STD", "0.031"))


@dataclass
class MarketDataConfig:
    # Yahoo Finance or equivalent
    price_lookback_days: int = 365
    forward_return_days: int = 60    # PEAD measurement window
    early_signal_days: int = 3       # 3-day early signal
    drift_threshold: float = 0.02    # 2% abnormal return = drift
    volatility_window: int = 20


@dataclass
class ModelConfig:
    # XGBoost / LightGBM
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 5
    scale_pos_weight: float = 1.0
    early_stopping_rounds: int = 50
    model_path: str = os.getenv("MODEL_PATH", "./models/saved/xgb_catalyst.pkl")

    # Confidence thresholds for decision engine
    buy_threshold: float = float(os.getenv("BUY_THRESHOLD", "0.60"))
    sell_threshold: float = float(os.getenv("SELL_THRESHOLD", "0.60"))
    hold_threshold: float = float(os.getenv("HOLD_THRESHOLD", "0.40"))


@dataclass
class CacheConfig:
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    embedding_ttl: int = 86400 * 7   # 7 days
    prediction_ttl: int = 3600       # 1 hour
    filing_ttl: int = 86400 * 30     # 30 days


@dataclass
class APIConfig:
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    workers: int = int(os.getenv("API_WORKERS", "4"))
    reload: bool = os.getenv("API_RELOAD", "false").lower() == "true"
    api_key: Optional[str] = os.getenv("API_KEY")
    cors_origins: list = field(
        default_factory=lambda: os.getenv("CORS_ORIGINS", "*").split(",")
    )


@dataclass
class LoggingConfig:
    level: str = os.getenv("LOG_LEVEL", "INFO")
    format: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    log_file: Optional[str] = os.getenv("LOG_FILE")
    structured: bool = os.getenv("STRUCTURED_LOGGING", "false").lower() == "true"


@dataclass
class Settings:
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    sec: SECConfig = field(default_factory=SECConfig)
    fmp: FMPConfig = field(default_factory=FMPConfig)
    upcoming: UpcomingPredictorConfig = field(default_factory=UpcomingPredictorConfig)
    market: MarketDataConfig = field(default_factory=MarketDataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Environment
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()


settings = get_settings()