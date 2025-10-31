from .memory import ConversationMemory, DatabaseConversationMemory
from .metrics import metrics
from .embedding_memory import EmbeddingConversationMemory, DatabaseEmbeddingMemory
from .vector_db import VectorDB, DatabaseVectorDB
from .mongo_memory import MongoMemorySystem, MemoryRecord, Preference
from .profiles import ProfileManager
from .reactions import ReactionHistory
from .scoring import ResponsePriorityQueue, CandidateResponseScorer
from .trigger_profile import TriggerProfileScorer
from .trigger_graph import TriggerEffectGraph
from .trigger_phrase_db import TriggerPhraseDB
from .engagement_graph import EngagementGraph, pagerank, shortest_path
from .adaptive_confidence import AdaptiveConfidenceScorer
from .anomaly_detection import AnomalyDetector
from .behavioral_shift import BehavioralShiftDetector
from .few_shot_learning import FewShotZeroShotClassifier
from .prompt_cache import TriggerPromptCache, FewShotPromptEngine
from .meta_learning_preprocess import MetaLearningPreprocessor
from .self_learning import SelfLearningEngine
from .rlhf import RLHFPolicyManager, DatabaseRLHFPolicyManager
from .human_feedback import HumanFeedbackManager
from .adaptive_ranking import AdaptiveRanker
try:  # pragma: no cover - optional dependency
    from .response_generation import ResponseCandidateGenerator, redundancy_filter
except Exception:  # pragma: no cover - allow partial import when dependency missing
    ResponseCandidateGenerator = None  # type: ignore
    redundancy_filter = None  # type: ignore
from .influence_graph import InfluenceGraph
from .psychological_graph import PsychologicalGraph, RuleNode
from .archetype_graph import ArchetypeGraph, ArchetypeNode, RelationshipEdge
from .policy_learning import PolicyLearner, BanditScout, PPOBrain
from .genetic_hatchery import GeneticHatchery
from .rl_integration import (
    QLearningModule,
    ReplayBuffer,
    DatabaseReplayBuffer,
    MetaLearner,
    RLResponseRanker,
    DatabaseRLResponseRanker,
)
from .experience_replay import ExperienceReplayBuffer, ReplayExchange
from .adaptive_exploration import AdaptiveExplorer
from .confidence_action import ConfidenceBasedActionSelector
from .hierarchical_reward import HierarchicalRewardLearner
from .adaptive_goal_switch import AdaptiveGoalSwitcher
from .marl_grid import FactionScaleMarlGrid
from .faction_influence import FactionInfluenceEngine
from .multi_agent_strategy import MultiAgentStrategy, DiplomaticMemory
from .social_power import SocialPowerRanker, ArchetypeStats
from .adaptive_influence_propagation import AdaptiveInfluencePropagator
from .static_harvest import StaticHarvester
from .dynamic_harvest import DynamicHarvester
from .doi_crawler import DOICrawler
from .api_harvest import APIScraper
from .http_retry import ResilientRequester
from .adaptive_scraper import AdaptiveWebScraper
from .orchestrator import SandboxOrchestrator
from .brain_to_sales import BrainToSalesMapper
from .neuro_api_router import NeuroAPIRouter, EndpointTelemetry
from .external_integrations import (
    RedditHarvester,
    TwitterTracker,
    GPT4Client,
    PineconeLogger,
    InfluenceGraphUpdater,
)
from .trend_monitor import MediumFetcher, TrendMonitor
from .security import get_api_key, RateLimiter

from .cortex_responder import (
    CortexAwareResponder,
    InMemoryResponseDB,
    ResponseRecord,
)
from .api_gateway import create_app
from .sql_db import (
    create_session as create_sql_session,
    run_migrations,
    UserProfile,
    UserPreference,
    ConversationMessage,
    MatchHistory,
    SelfLearningEvent,
)
from .neuro_etl import (
    NeuroToken,
    InMemoryPostgres,
    InMemoryMongo,
    InMemoryFaiss,
    InMemoryNeo4j,
    tokenize_paragraph,
    process_study_sync,
    app as neuro_etl_app,
)
from .cache_system import (
    SessionMemoryCache,
    UserPreferenceCache,
    ResponseRankingCache,
    ArchetypeInfluenceCache,
    MemoryDecaySystem,
)
from .db_manager import DatabaseConnectionManager
from .rl_training import (
    save_feedback_dataset,
    train_models,
    schedule_periodic_training,
    schedule_feedback_export,
)
from .db_backup import schedule_database_backup
from .engagement_dataset import collect_engagement_logs

try:  # optional heavy deps
    from .preprocess import TextPreprocessor, PreprocessResult
    from .ner import IntentEntityExtractor, IntentProfile
    from .entity_detection import EntityDetector, Entity
    from .intent_detection import IntentDetector
    from .intent_classifier import IntentClassifier
    from .sentiment import SentimentAnalyzer, SentimentMemory
except Exception:  # pragma: no cover - allow partial import
    TextPreprocessor = PreprocessResult = None  # type: ignore
    IntentEntityExtractor = IntentProfile = None  # type: ignore
    EntityDetector = Entity = None  # type: ignore
    IntentDetector = None  # type: ignore
    IntentClassifier = None  # type: ignore
    SentimentAnalyzer = SentimentMemory = None  # type: ignore

try:  # pragma: no cover - optional dependency with richer graph libs
    from .emotion import (
        EmotionLabeler,
        RollingEmotionTensor,
        EmotionMemory,
        DatabaseEmotionMemory,
        GenderStyleAdapter,
        ReinforcementABTest,
    )
except Exception:  # pragma: no cover - allow partial import when deps missing
    EmotionLabeler = None  # type: ignore
    RollingEmotionTensor = None  # type: ignore
    EmotionMemory = None  # type: ignore
    DatabaseEmotionMemory = None  # type: ignore
    GenderStyleAdapter = None  # type: ignore
    ReinforcementABTest = None  # type: ignore
from .user_preferences import (
    PreferenceEngine,
    DatabasePreferenceEngine,
    PreferenceProfile,
    PerformanceTracker,
    RoleplayCoach,
)

__all__ = [
    "ConversationMemory",
    "DatabaseConversationMemory",
    "EmbeddingConversationMemory",
    "DatabaseEmbeddingMemory",
    "VectorDB",
    "DatabaseVectorDB",
    "MongoMemorySystem",
    "MemoryRecord",
    "Preference",
    "ProfileManager",
    "ReactionHistory",
    "ResponsePriorityQueue",
    "CandidateResponseScorer",
    "TriggerProfileScorer",
    "TriggerEffectGraph",
    "TriggerPhraseDB",
    "EngagementGraph",
    "TextPreprocessor",
    "PreprocessResult",
    "IntentEntityExtractor",
    "IntentProfile",
    "IntentDetector",
    "IntentClassifier",
    "SentimentAnalyzer",
    "SentimentMemory",
    "EntityDetector",
    "Entity",
    "pagerank",
    "shortest_path",
    "AdaptiveConfidenceScorer",
    "AnomalyDetector",
    "BehavioralShiftDetector",
    "FewShotZeroShotClassifier",
    "TriggerPromptCache",
    "FewShotPromptEngine",
    "MetaLearningPreprocessor",
    "PreferenceEngine",
    "DatabasePreferenceEngine",
    "PreferenceProfile",
    "PerformanceTracker",
    "RoleplayCoach",
    "EmotionLabeler",
    "RollingEmotionTensor",
    "EmotionMemory",
    "DatabaseEmotionMemory",
    "GenderStyleAdapter",
    "ReinforcementABTest",
    "SelfLearningEngine",
    "InfluenceGraph",
    "PsychologicalGraph",
    "RuleNode",
    "ArchetypeGraph",
    "ArchetypeNode",
    "RelationshipEdge",
    "ResponseCandidateGenerator",
    "redundancy_filter",
    "RLHFPolicyManager",
    "DatabaseRLHFPolicyManager",
    "HumanFeedbackManager",
    "AdaptiveRanker",
    "QLearningModule",
    "ReplayBuffer",
    "DatabaseReplayBuffer",
    "MetaLearner",
    "RLResponseRanker",
    "DatabaseRLResponseRanker",
    "PolicyLearner",
    "BanditScout",
    "PPOBrain",
    "GeneticHatchery",
    "ExperienceReplayBuffer",
    "ReplayExchange",
    "AdaptiveExplorer",
    "ConfidenceBasedActionSelector",
    "HierarchicalRewardLearner",
    "AdaptiveGoalSwitcher",
    "FactionScaleMarlGrid",
    "FactionInfluenceEngine",
    "MultiAgentStrategy",
    "DiplomaticMemory",
    "SocialPowerRanker",
    "AdaptiveInfluencePropagator",
    "StaticHarvester",
    "DynamicHarvester",
    "SandboxOrchestrator",
    "DOICrawler",
    "APIScraper",
    "ResilientRequester",
    "AdaptiveWebScraper",
    "BrainToSalesMapper",
    "NeuroToken",
    "InMemoryPostgres",
    "InMemoryMongo",
    "InMemoryFaiss",
    "InMemoryNeo4j",
    "tokenize_paragraph",
    "process_study_sync",
    "neuro_etl_app",
    "RedditHarvester",
    "TwitterTracker",
    "GPT4Client",
    "PineconeLogger",
    "InfluenceGraphUpdater",
    "MediumFetcher",
    "TrendMonitor",
    "CortexAwareResponder",
    "InMemoryResponseDB",
    "ResponseRecord",
    "create_app",
    "metrics",
    "NeuroAPIRouter",
    "EndpointTelemetry",
    "create_sql_session",
    "run_migrations",
    "UserProfile",
    "UserPreference",
    "ConversationMessage",
    "ArchetypeStats",
    "MatchHistory",
    "SelfLearningEvent",
    "SessionMemoryCache",
    "UserPreferenceCache",
    "ResponseRankingCache",
    "ArchetypeInfluenceCache",
    "MemoryDecaySystem",
    "DatabaseConnectionManager",
    "save_feedback_dataset",
    "train_models",
    "collect_engagement_logs",
    "schedule_periodic_training",
    "schedule_feedback_export",
    "schedule_database_backup",
    "get_api_key",
    "RateLimiter",
]
