"""ChatGPT Research Bot for conversational knowledge extraction."""

from __future__ import annotations

from .bot_registry import BotRegistry
from .data_bot import DataBot

from .coding_bot_interface import self_coding_managed
import json
import os
import logging
import time

from dynamic_path_router import resolve_dir

from .dependency_manager import DependencyManager

registry = BotRegistry()
data_bot = DataBot(start_server=False)

_deps = DependencyManager()

requests = _deps.load("requests", lambda: __import__("requests"))  # type: ignore
from dataclasses import dataclass, asdict, field
from typing import Iterable, List, Optional, Callable, Any
import re
from collections import Counter
np = _deps.load("numpy", lambda: __import__("numpy"))  # type: ignore

from .retry_utils import with_retry

logger = logging.getLogger(__name__)

from governed_embeddings import governed_embed, get_embedder


def _env_int(
    name: str,
    default: int,
    *,
    min_val: int | None = None,
    max_val: int | None = None,
) -> int:
    """Return integer environment variable with optional range validation."""
    val = os.getenv(name)
    if val is None:
        result = default
    else:
        try:
            result = int(val)
        except Exception:
            logger.warning("invalid integer for %s: %s", name, val)
            return default
    if min_val is not None and result < min_val:
        logger.warning("%s below minimum of %s; using default", name, min_val)
        return default
    if max_val is not None and result > max_val:
        logger.warning("%s above maximum of %s; using default", name, max_val)
        return default
    return result


def _env_float(
    name: str,
    default: float,
    *,
    min_val: float | None = None,
    max_val: float | None = None,
) -> float:
    """Return float environment variable with optional range validation."""
    val = os.getenv(name)
    if val is None:
        result = default
    else:
        try:
            result = float(val)
        except Exception:
            logger.warning("invalid float for %s: %s", name, val)
            return default
    if min_val is not None and result < min_val:
        logger.warning("%s below minimum of %s; using default", name, min_val)
        return default
    if max_val is not None and result > max_val:
        logger.warning("%s above maximum of %s; using default", name, max_val)
        return default
    return result



def configure_logging(
    level: str | int | None = None,
    filename: str | None = None,
    *,
    settings: "ResearchBotSettings | None" = None,
) -> None:
    """Configure module logging with optional level and output file."""
    cfg = settings or SETTINGS
    if level is None:
        level = cfg.log_level
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)
    log_file = filename or cfg.log_file
    if log_file:
        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)


from .chatgpt_idea_bot import ChatGPTClient
from gpt_memory_interface import GPTMemoryInterface
try:  # memory-aware wrapper
    from .memory_aware_gpt_client import ask_with_memory
except Exception:  # pragma: no cover - fallback for flat layout
    from memory_aware_gpt_client import ask_with_memory  # type: ignore
try:  # canonical tag constants
    from .log_tags import FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT
except Exception:  # pragma: no cover - fallback for flat layout
    from log_tags import FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT  # type: ignore
try:  # shared GPT memory instance
    from .shared_gpt_memory import GPT_MEMORY_MANAGER
except Exception:  # pragma: no cover - fallback for flat layout
    from shared_gpt_memory import GPT_MEMORY_MANAGER  # type: ignore

try:  # optional context builder
    from vector_service.context_builder import ContextBuilder
except Exception:  # pragma: no cover - fallback when service unavailable
    ContextBuilder = Any  # type: ignore

DBRouter = _deps.load(
    "DBRouter", lambda: __import__("menace.db_router", fromlist=["DBRouter"]).DBRouter
) or object  # type: ignore

summarize = _deps.load("gensim_summarize", lambda: __import__("gensim.summarization", fromlist=["summarize"]).summarize)

def _load_sklearn() -> tuple[object | None, object | None]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.decomposition import TruncatedSVD  # type: ignore
        return TfidfVectorizer, TruncatedSVD
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.debug("sklearn unavailable: %s", exc)
        return None, None

TfidfVectorizer, TruncatedSVD = _load_sklearn()

pipeline = _deps.load("transformers_pipeline", lambda: __import__("transformers", fromlist=["pipeline"]).pipeline)

NLP_OFFLINE_MODE = os.getenv("NLP_OFFLINE_MODE") == "1"


def _load_nltk() -> tuple[object | None, bool]:
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            if NLP_OFFLINE_MODE:
                logger.warning(
                    "nltk punkt data missing and offline mode enabled; sentence splitting may degrade"
                )
                return nltk, False
            try:
                nltk.download("punkt", quiet=True)
            except Exception as exc:  # pragma: no cover - optional
                logger.warning("failed to download nltk data: %s", exc)
                return nltk, False
        return nltk, True
    except Exception as exc:  # pragma: no cover - optional
        logger.warning("nltk unavailable: %s", exc)
        return None, False

nltk, _nltk_available = _load_nltk()

def _load_spacy() -> tuple[object | None, object | None, bool]:
    try:
        import spacy
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return spacy, nlp, True
    except Exception as exc:  # pragma: no cover - optional
        logger.warning("spacy unavailable: %s", exc)
        return None, None, False

spacy, _spacy_nlp, _spacy_available = _load_spacy()


@dataclass
class Exchange:
    """A single question and answer pair."""

    prompt: str
    response: str


@dataclass
class ResearchResult:
    """Container for conversation and summary."""

    conversation: List[Exchange]
    summary: str


@dataclass(frozen=True)
class ResearchBotSettings:
    """Runtime configuration loaded from environment variables."""

    log_level: str = os.getenv("CHATGPT_RESEARCH_LOG_LEVEL", "INFO")
    log_file: str | None = os.getenv("CHATGPT_RESEARCH_LOG_FILE")
    aggregator_url: str | None = os.getenv("AGGREGATOR_URL")
    aggregator_token: str | None = os.getenv("AGGREGATOR_TOKEN")
    aggregator_sign_secret: str | None = os.getenv("AGGREGATOR_SIGN_SECRET")
    fallback_log: str = os.getenv(
        "RESEARCH_FALLBACK_LOG",
        str(resolve_dir("logs") / "aggregator_failures.jsonl"),
    )
    aggregator_retry_queue: str | None = os.getenv("AGGREGATOR_RETRY_QUEUE")
    cloud_logging_url: str | None = os.getenv("CLOUD_LOGGING_URL")
    ask_attempts: int = field(
        default_factory=lambda: _env_int("CHATGPT_RESEARCH_ATTEMPTS", 3, min_val=1)
    )
    ask_backoff: float = field(
        default_factory=lambda: _env_float("CHATGPT_RESEARCH_BACKOFF", 1.0, min_val=0.1)
    )
    aggregator_attempts: int = field(
        default_factory=lambda: _env_int("AGGREGATOR_ATTEMPTS", 3, min_val=1)
    )
    aggregator_backoff: float = field(
        default_factory=lambda: _env_float("AGGREGATOR_BACKOFF", 1.0, min_val=0.1)
    )
    conversation_token_limit: int = field(
        default_factory=lambda: _env_int("CHATGPT_CONVO_TOKENS", 3000, min_val=1)
    )
    raise_db_errors: bool = os.getenv("CHATGPT_RESEARCH_RAISE_DB_ERRORS") == "1"

    def __post_init__(self) -> None:
        if self.aggregator_url and not self.aggregator_url.startswith(("http://", "https://")):
            logger.warning("invalid AGGREGATOR_URL %s", self.aggregator_url)
            object.__setattr__(self, "aggregator_url", None)
        for name in ("ask_attempts", "aggregator_attempts", "conversation_token_limit"):
            val = getattr(self, name)
            if val <= 0:
                logger.warning("%s must be positive; using default", name)
                object.__setattr__(self, name, 1)
        for name in ("ask_backoff", "aggregator_backoff"):
            val = getattr(self, name)
            if val <= 0:
                logger.warning("%s must be positive; using default", name)
                object.__setattr__(self, name, 1.0)


SETTINGS = ResearchBotSettings()


@dataclass(frozen=True)
class SummaryConfig:
    """Configuration parameters for text summarisation."""

    small_threshold: int = field(
        default_factory=lambda: _env_int("SUMMARY_SMALL_THRESHOLD", 5, min_val=1)
    )
    medium_threshold: int = field(
        default_factory=lambda: _env_int("SUMMARY_MEDIUM_THRESHOLD", 20, min_val=1)
    )
    ratio_small: float = field(
        default_factory=lambda: _env_float("SUMMARY_RATIO_SMALL", 0.8, min_val=0.0, max_val=1.0)
    )
    ratio_medium: float = field(
        default_factory=lambda: _env_float("SUMMARY_RATIO_MEDIUM", 0.3, min_val=0.0, max_val=1.0)
    )
    ratio_large: float = field(
        default_factory=lambda: _env_float("SUMMARY_RATIO_LARGE", 0.15, min_val=0.0, max_val=1.0)
    )
    filter_threshold: float = field(
        default_factory=lambda: _env_float("SUMMARY_FILTER_THRESHOLD", 0.8, min_val=0.0, max_val=1.0)
    )
    transformer_min_ratio: float = field(
        default_factory=lambda: _env_float("SUMMARY_TRANS_MIN_RATIO", 0.5, min_val=0.1, max_val=1.0)
    )
    transformer_len_mult: float = field(
        default_factory=lambda: _env_float("SUMMARY_LEN_MULT", 1.0, min_val=0.1)
    )
    transformer_max_len: int | None = (
        os.getenv("SUMMARY_MAX_LEN") and _env_int("SUMMARY_MAX_LEN", 0, min_val=1)
    ) or None
    batch_size: int = field(
        default_factory=lambda: _env_int("SUMMARY_BATCH_SIZE", 0, min_val=0)
    )
    cache_size: int = field(
        default_factory=lambda: _env_int("SUMMARY_CACHE_SIZE", 128, min_val=1)
    )
    transformer_timeout: float = field(
        default_factory=lambda: _env_float("SUMMARY_TRANS_TIMEOUT", 5.0, min_val=1.0)
    )
    # comma separated preferred strategies; 'benchmark' triggers dynamic benchmarking
    strategies: str = os.getenv(
        "SUMMARY_STRATEGIES", "benchmark"
    )

    def __post_init__(self) -> None:
        valid = {"gensim", "svd", "transformer", "frequency", "benchmark"}
        for strat in [s.strip() for s in self.strategies.split(',') if s.strip()]:
            if strat not in valid:
                logger.error("invalid summarization strategy '%s'", strat)
                raise ValueError(f"invalid summarization strategy {strat}")

from functools import lru_cache
import multiprocessing


def _split_sentences(text: str) -> List[str]:
    """Return a list of sentences using nltk or spaCy if available."""
    if _nltk_available:
        try:
            return [s.strip() for s in nltk.tokenize.sent_tokenize(text) if s.strip()]
        except Exception:  # pragma: no cover - fallback
            pass
    if _spacy_available:
        try:
            doc = _spacy_nlp(text)
            return [s.text.strip() for s in doc.sents if s.text.strip()]
        except Exception:  # pragma: no cover - fallback
            pass
    return [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]


@lru_cache(maxsize=SummaryConfig().cache_size)
def _summarise_chunk(text: str, ratio: float | None, config: SummaryConfig) -> str:
    """Summarise ``text`` without batching or caching logic."""
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return text

    if ratio is None:
        count = len(sentences)
        if count < config.small_threshold:
            ratio = config.ratio_small
        elif count < config.medium_threshold:
            ratio = config.ratio_medium
        else:
            ratio = config.ratio_large

    strategy_map = {
        "gensim": _summarise_gensim,
        "svd": _summarise_svd,
        "transformer": _summarise_transformer,
        "frequency": _summarise_frequency,
    }

    order = [s.strip() for s in config.strategies.split(",") if s.strip()]

    if not order or "benchmark" in order:
        timings: list[tuple[float, str, str]] = []
        for name, strat in strategy_map.items():
            start = time.time()
            try:
                result = strat(text, sentences, ratio, config)
            except Exception:
                result = None
            if result:
                timings.append((time.time() - start, name, result))
        if timings:
            timings.sort(key=lambda t: t[0])
            _BENCHMARK_ORDER[:] = [t[1] for t in timings]
            logger.debug("benchmark timings: %s", _BENCHMARK_ORDER)
            return timings[0][2]
        order = list(strategy_map)

    for name in order:
        strat = strategy_map.get(name)
        if not strat:
            continue
        result = strat(text, sentences, ratio, config)
        if result:
            logger.debug("summarization using %s", name)
            return result

    logger.info("falling back to frequency summarizer")
    return _summarise_frequency(text, sentences, ratio, config)


def _summarise_gensim(text: str, sentences: List[str], ratio: float, config: SummaryConfig) -> str | None:
    if not summarize:
        return None
    try:
        result = summarize(text, ratio=ratio)
        if result:
            return _filter_redundant_sentences(result, config.filter_threshold)
    except Exception:
        logger.exception("gensim summarization failed")
    return None


def _summarise_svd(text: str, sentences: List[str], ratio: float, config: SummaryConfig) -> str | None:
    if not (TfidfVectorizer and TruncatedSVD and np):
        return None
    try:
        vec = TfidfVectorizer()
        matrix = vec.fit_transform(sentences).T
        vocab_dim, sent_dim = matrix.shape

        def _choose_components(vdim: int, sdim: int) -> int:
            max_dim = min(vdim, sdim) - 1
            if max_dim <= 1:
                return 1
            est = int(np.sqrt(vdim * sdim) / 2)
            est = int(np.clip(est, 1, max_dim))
            return est

        n_components = _choose_components(vocab_dim, sent_dim)
        svd = TruncatedSVD(n_components=n_components)
        svd.fit(matrix)
        sigma = svd.singular_values_
        vt = svd.components_
        scores = np.sqrt((sigma[:, None] * vt) ** 2).sum(axis=0)
        ranked = [s for s, _ in sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)]
        count = max(1, int(len(ranked) * ratio))
        result = ". ".join(ranked[:count]).rstrip(".") + "."
        return _filter_redundant_sentences(result, config.filter_threshold)
    except Exception:
        logger.exception("SVD summarization failed")
    return None


def _call_with_timeout(func: Callable[[], str | None], timeout: float) -> str | None:
    result_queue: multiprocessing.Queue = multiprocessing.Queue()

    def runner(q: multiprocessing.Queue) -> None:
        try:
            q.put(func())
        except BaseException as e:  # pragma: no cover - defensive
            q.put(e)

    p = multiprocessing.Process(target=runner, args=(result_queue,))
    p.daemon = True
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        return None
    if not result_queue.empty():
        result = result_queue.get()
        if isinstance(result, BaseException):
            raise result
        return result
    return None


_TRANSFORMER_SUMMARIZER: Optional[Callable[[str], list]] = None
_BENCHMARK_ORDER: List[str] = []
_SBERT_MODEL: Optional[object] = None


def _get_sbert_model() -> object | None:
    """Lazily load the shared embedder for semantic similarity."""
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        _SBERT_MODEL = get_embedder()
    return _SBERT_MODEL


def _get_transformer_summarizer() -> Callable[[str], list] | None:
    global _TRANSFORMER_SUMMARIZER
    if _TRANSFORMER_SUMMARIZER is None and pipeline:
        try:
            _TRANSFORMER_SUMMARIZER = pipeline("summarization")
        except Exception as exc:  # pragma: no cover - optional
            logger.warning("failed to initialise transformer pipeline: %s", exc)
            _TRANSFORMER_SUMMARIZER = None
    return _TRANSFORMER_SUMMARIZER


def _summarise_transformer(text: str, sentences: List[str], ratio: float, config: SummaryConfig) -> str | None:
    summarizer = _get_transformer_summarizer()
    if not summarizer:
        return None

    def _run() -> str | None:
        max_input = getattr(getattr(summarizer, "tokenizer", None), "model_max_length", 1024)
        tokens = text.split()
        if len(tokens) > max_input:
            trimmed = " ".join(tokens[:max_input])
        else:
            trimmed = text
        length_est = int(len(text.split()) * ratio * config.transformer_len_mult)
        if config.transformer_max_len:
            length_est = min(length_est, config.transformer_max_len)
        max_length = max(20, min(length_est, max_input))
        min_length = max(10, int(max_length * config.transformer_min_ratio))
        result = summarizer(trimmed, max_length=max_length, min_length=min_length)
        if result and isinstance(result, list):
            summary_text = result[0].get("summary_text")
            if summary_text:
                return _filter_redundant_sentences(summary_text, config.filter_threshold)
        return None

    try:
        summary = _call_with_timeout(_run, config.transformer_timeout)
        if summary is None:
            logger.warning("transformer summarization timed out")
        return summary
    except Exception:
        logger.exception("transformer summarization failed")
        return None


def _summarise_frequency(text: str, sentences: List[str], ratio: float, config: SummaryConfig) -> str:
    words = re.findall(r"\w+", text.lower())
    freq = Counter(words)
    ranked = sorted(
        sentences,
        key=lambda s: sum(1.0 / (freq.get(w.lower(), 1)) for w in s.split()) / (len(s.split()) or 1),
        reverse=True,
    )
    count = max(1, int(len(ranked) * ratio))
    result = ". ".join(ranked[:count]).rstrip(".") + "."
    return _filter_redundant_sentences(result, config.filter_threshold)


def summarise_text(text: str, ratio: float | None = None, config: SummaryConfig | None = None) -> str:
    """Summarise long text with optional batching and caching."""
    cfg = config or SummaryConfig()
    text = text.strip()
    if not text:
        return ""

    words = text.split()
    if cfg.batch_size and len(words) > cfg.batch_size:
        segments = [" ".join(words[i : i + cfg.batch_size]) for i in range(0, len(words), cfg.batch_size)]
        parts = [_summarise_chunk(seg, ratio, cfg) for seg in segments]
        combined = " ".join(parts)
        return _summarise_chunk(combined, ratio, cfg)
    return _summarise_chunk(text, ratio, cfg)


def _filter_redundant_sentences(text: str, threshold: float = 0.8) -> str:
    """Remove highly similar sentences from ``text`` using semantic similarity."""
    sents = _split_sentences(text)
    if len(sents) <= 1:
        return text.strip()

    model = _get_sbert_model()
    if not (model and np is not None):
        logger.warning("semantic redundancy filter unavailable; returning text unchanged")
        return text.strip()
    try:
        vecs = []
        for s in sents:
            emb = governed_embed(s, model)
            if emb is None:
                return text.strip()
            vecs.append(np.array(emb))
        kept_idx: List[int] = []
        for i, vec in enumerate(vecs):
            if any(
                float(np.dot(vec, vecs[j]) / (np.linalg.norm(vec) * np.linalg.norm(vecs[j])))
                >= threshold
                for j in kept_idx
            ):
                continue
            kept_idx.append(i)
        kept_sents = [sents[i] for i in kept_idx]
        return ". ".join(kept_sents).rstrip(".") + ("." if kept_sents else "")
    except Exception:
        logger.exception("SBERT redundancy filter failed")
        return text.strip()


def _persist_payload(payload: dict, path: str) -> None:
    """Append ``payload`` to the fallback JSONL log."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
    except Exception as exc:  # pragma: no cover - best effort
        logger.error("failed to persist aggregator payload: %s", exc)


def _queue_payload(payload: dict, path: str) -> None:
    """Queue ``payload`` for later retry."""
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
    except Exception as exc:
        logger.error("failed to queue aggregator payload: %s", exc)


def _send_cloud_log(payload: dict, url: str) -> None:
    if not (url and requests):
        return
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as exc:  # pragma: no cover - best effort
        logger.error("cloud logging failed: %s", exc)


@self_coding_managed(bot_registry=registry, data_bot=data_bot)
class ChatGPTResearchBot:
    """Interact with ChatGPT recursively to gather research."""

    def __init__(
        self,
        context_builder: ContextBuilder,
        client: ChatGPTClient | None = None,
        send_callback: Optional[Callable[[Iterable[Exchange], str], None]] = None,
        db_steward: "DBRouter" | None = None,
        summary_config: SummaryConfig | None = None,
        *,
        settings: ResearchBotSettings | None = None,
        gpt_memory: GPTMemoryInterface | None = GPT_MEMORY_MANAGER,
    ) -> None:
        self.context_builder = context_builder
        if client is None:
            client = ChatGPTClient(
                context_builder=context_builder, gpt_memory=gpt_memory
            )
        else:
            if not isinstance(client, ChatGPTClient):
                raise TypeError("client must be ChatGPTClient")
            if client.context_builder is None:
                raise ValueError("client.context_builder must not be None")
        self.client = client
        self.send_callback = send_callback
        if db_steward is not None and not isinstance(db_steward, DBRouter):
            raise TypeError("db_steward must be DBRouter or None")
        self.db_steward = db_steward
        self.summary_config = summary_config or SummaryConfig()
        self.settings = settings or ResearchBotSettings()
        self.gpt_memory = gpt_memory
        if getattr(self.client, "gpt_memory", None) is None:
            try:
                self.client.gpt_memory = self.gpt_memory
            except Exception:
                logger.debug("failed to attach gpt_memory to client", exc_info=True)

    def _truncate_history(self, text: str) -> str:
        limit = self.settings.conversation_token_limit
        tokens = text.split()
        if len(tokens) > limit:
            tokens = tokens[-limit:]
        return " ".join(tokens)

    def _budget_prompt(self, prompt: str) -> str:
        limit = self.settings.conversation_token_limit
        tokens = prompt.split()
        if len(tokens) > limit:
            tokens = tokens[:limit]
        return " ".join(tokens)

    def _ask(self, prompt: str, attempts: int | None = None, delay: float | None = None) -> str:
        """Query the ChatGPT client with retries and exponential backoff."""
        attempts = attempts or self.settings.ask_attempts
        delay = delay or self.settings.ask_backoff

        def _do() -> str:
            b_prompt = self._budget_prompt(prompt)
            data = ask_with_memory(
                self.client,
                "chatgpt_research_bot._ask",
                b_prompt,
                memory=self.gpt_memory,
                context_builder=self.context_builder,
                tags=[FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT],
            )
            if not isinstance(data, dict):
                logger.warning("unexpected response type %s", type(data))
                return ""
            if data.get("error"):
                err = data["error"]
                logger.warning("api error: %s", err)
                raise RuntimeError("api_error")
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                message = choices[0].get("message", {})
                if isinstance(message, dict):
                    msg = message.get("content", "")
                    if msg:
                        return msg
                else:
                    logger.warning("unexpected message format from API")
            else:
                logger.warning("missing 'choices' in API response")
            return ""

        try:
            return with_retry(_do, attempts=attempts, delay=delay, logger=logger)
        except Exception as exc:  # pragma: no cover - network issues
            if "rate" in str(exc).lower():
                logger.warning("rate limited when querying API")
            else:
                logger.warning("ask failed after retries: %s", exc)
            return ""

    def conversation(self, instruction: str, depth: int = 1) -> List[Exchange]:
        logger.info("starting conversation: %s", instruction)
        convo: List[Exchange] = []
        prompt = instruction
        for i in range(depth):
            response = self._ask(prompt)
            convo.append(Exchange(prompt=prompt, response=response))
            if i + 1 >= depth:
                break
            history = self._truncate_history(" ".join(ex.response for ex in convo))
            prompt = (
                f"Based on our discussion so far ({history}), "
                f"explore a related subtopic or provide further detail about {instruction}."
            )
        return convo

    def summarise(self, convo: Iterable[Exchange], ratio: float | None = None) -> str:
        text = " ".join(ex.response for ex in convo)
        try:
            summary = summarise_text(text, ratio=ratio, config=self.summary_config)
            logger.info("conversation summarisation completed")
            return summary
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("summarisation failed: %s", exc)
            return ""

    def process(self, instruction: str, depth: int = 1, ratio: float | None = None) -> ResearchResult:
        if self.db_steward:
            try:
                matches = self.db_steward.query_all(instruction).info
            except Exception as exc:
                logger.error("database router query failed: %s", exc)
                if self.settings.raise_db_errors:
                    raise
                matches = []
            if matches:
                convo = [Exchange(prompt=instruction, response=m.content) for m in matches]
                summary = matches[0].summary if hasattr(matches[0], "summary") else ""
                return ResearchResult(conversation=convo, summary=summary)

        convo = self.conversation(instruction, depth=depth)
        summary = self.summarise(convo, ratio=ratio)
        if self.send_callback:
            try:
                self.send_callback(convo, summary)
                logger.info("conversation dispatched via callback")
            except Exception as exc:
                logger.exception("callback send failed: %s", exc)
                send_to_aggregator(convo, summary, settings=self.settings)
        else:
            send_to_aggregator(convo, summary, settings=self.settings)
        return ResearchResult(conversation=convo, summary=summary)


def send_to_aggregator(
    convo: Iterable[Exchange],
    summary: str,
    *,
    settings: ResearchBotSettings | None = None,
) -> None:
    """POST conversation and summary to the aggregator service with retries."""

    cfg = settings or SETTINGS

    payload = {"conversation": [asdict(ex) for ex in convo], "summary": summary}

    if not cfg.aggregator_url:
        logger.warning("AGGREGATOR_URL not configured; persisting payload")
        _persist_payload(payload, cfg.fallback_log)
        _queue_payload(payload, cfg.aggregator_retry_queue or "")
        _send_cloud_log(payload, cfg.cloud_logging_url or "")
        return

    url = cfg.aggregator_url

    if not requests:
        logger.warning("requests unavailable; persisting aggregator payload")
        _persist_payload(payload, cfg.fallback_log)
        _queue_payload(payload, cfg.aggregator_retry_queue or "")
        _send_cloud_log(payload, cfg.cloud_logging_url or "")
        return

    headers: dict[str, str] = {}
    if cfg.aggregator_token:
        headers["Authorization"] = f"Bearer {cfg.aggregator_token}"
    if cfg.aggregator_sign_secret:
        import hmac, hashlib

        digest = hmac.new(
            cfg.aggregator_sign_secret.encode(),
            json.dumps(payload, sort_keys=True).encode(),
            hashlib.sha256,
        ).hexdigest()
        headers["X-Signature"] = digest

    attempts = max(1, cfg.aggregator_attempts)
    delay = cfg.aggregator_backoff

    def _post() -> None:
        resp = requests.post(url, json=payload, headers=headers, timeout=5)
        if not (200 <= resp.status_code < 300):
            raise RuntimeError(resp.status_code)

    try:
        with_retry(_post, attempts=attempts, delay=delay, logger=logger)
        logger.info("aggregator POST succeeded")
    except Exception as exc:
        logger.error("Failed to send conversation to aggregator after %s attempts: %s", attempts, exc)
        _persist_payload(payload, cfg.fallback_log)
        _queue_payload(payload, cfg.aggregator_retry_queue or "")
        _send_cloud_log(payload, cfg.cloud_logging_url or "")


__all__ = ["ChatGPTResearchBot", "Exchange", "ResearchResult", "summarise_text", "send_to_aggregator"]