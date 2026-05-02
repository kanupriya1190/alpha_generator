"""News sentiment utilities using FinBERT with safe fallbacks."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import requests

from config import SETTINGS

SYMBOL_QUERY_ALIASES: Dict[str, List[str]] = {
    "NVDA": ["NVIDIA", "NVDA"],
    "MSFT": ["Microsoft", "MSFT"],
    "GOOG": ["Google", "Alphabet", "GOOG"],
    "TLT": ["Treasury bond ETF", "TLT", "iShares 20+ Year Treasury Bond ETF"],
    "CRWV": ["CoreWeave", "CRWV"],
    "NBIS": ["Nebius", "NBIS"],
    "BE": ["Bloom Energy", "BE"],
}


def _label_to_signed_score(label: str, confidence: float) -> float:
    up = label.upper()
    if "POS" in up:
        return confidence
    if "NEG" in up:
        return -confidence
    return 0.0


@dataclass
class FinBERTNewsSentiment:
    """Fetches ticker news and scores headline sentiment with FinBERT."""

    model_name: str = "ProsusAI/finbert"
    cache_ttl_seconds: int = 900  # 15 minutes
    max_headlines: int = 20
    _pipe: object | None = field(default=None, init=False, repr=False)
    _cache: Dict[str, Tuple[float, float, int]] = field(default_factory=dict, init=False, repr=False)
    _pipeline_failed: bool = field(default=False, init=False, repr=False)

    def _load_pipeline(self) -> object | None:
        if self._pipe is not None:
            return self._pipe
        if self._pipeline_failed:
            return None
        try:
            # Lazy import avoids Streamlit Cloud watcher probing all transformers
            # submodules (which can trigger optional torchvision errors).
            from transformers import pipeline as hf_pipeline

            self._pipe = hf_pipeline("text-classification", model=self.model_name, tokenizer=self.model_name)
            return self._pipe
        except Exception:
            self._pipeline_failed = True
            return None

    def _news_api_key(self) -> str:
        return SETTINGS.news_api_key or os.getenv("NEWS_API_KEY", "")

    def _query_for_symbol(self, symbol: str) -> str:
        aliases = SYMBOL_QUERY_ALIASES.get(symbol.upper(), [symbol])
        # Example: ("NVIDIA" OR "NVDA") AND (stock OR shares OR earnings OR guidance)
        ticker_part = " OR ".join(f'"{a}"' for a in aliases)
        return f"({ticker_part}) AND (stock OR shares OR earnings OR guidance)"

    def _fetch_headlines(self, symbol: str) -> Tuple[List[str], str]:
        api_key = self._news_api_key()
        if not api_key:
            return [], "no_news_key"
        date_from = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
        params = {
            "q": self._query_for_symbol(symbol),
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": self.max_headlines,
            "from": date_from,
            "apiKey": api_key,
        }
        try:
            resp = requests.get("https://newsapi.org/v2/everything", params=params, timeout=20)
            resp.raise_for_status()
            payload = resp.json()
            articles = payload.get("articles", [])
            headlines = []
            for a in articles:
                title = (a.get("title") or "").strip()
                desc = (a.get("description") or "").strip()
                text = " - ".join([x for x in [title, desc] if x])
                if text:
                    headlines.append(text[:600])
            if not headlines:
                return [], "no_headlines"
            return headlines, "ok"
        except Exception:
            return [], "news_api_error"

    def score_symbol(self, symbol: str) -> Tuple[float, int]:
        details = self.score_symbol_details(symbol)
        return float(details["score"]), int(details["headline_count"])

    def score_symbol_details(self, symbol: str) -> Dict[str, object]:
        now = time.time()
        cached = self._cache.get(symbol.upper())
        if cached and now - cached[2] <= self.cache_ttl_seconds:
            return {
                "score": float(cached[0]),
                "headline_count": int(cached[1]),
                "source": "cache",
            }

        headlines, news_status = self._fetch_headlines(symbol)
        if not headlines:
            self._cache[symbol.upper()] = (0.5, 0, int(now))
            return {
                "score": 0.5,
                "headline_count": 0,
                "source": f"fallback_{news_status}",
            }

        clf = self._load_pipeline()
        if clf is None:
            self._cache[symbol.upper()] = (0.5, len(headlines), int(now))
            return {
                "score": 0.5,
                "headline_count": len(headlines),
                "source": "fallback_finbert_unavailable",
            }

        try:
            preds = clf(headlines, truncation=True)
            signed_scores = []
            for p in preds:
                signed_scores.append(_label_to_signed_score(str(p.get("label", "")), float(p.get("score", 0.0))))
            avg_signed = sum(signed_scores) / max(len(signed_scores), 1)
            score_01 = max(0.0, min(1.0, (avg_signed + 1.0) / 2.0))
            self._cache[symbol.upper()] = (score_01, len(headlines), int(now))
            return {
                "score": score_01,
                "headline_count": len(headlines),
                "source": "finbert_news",
            }
        except Exception:
            self._cache[symbol.upper()] = (0.5, len(headlines), int(now))
            return {
                "score": 0.5,
                "headline_count": len(headlines),
                "source": "fallback_finbert_inference_error",
            }

