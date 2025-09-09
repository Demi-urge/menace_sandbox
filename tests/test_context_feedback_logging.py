import logging
import sys
import types
from pathlib import Path
import importlib.util
import pytest

# ---- minimal stub environment to import SelfDebuggerSandbox ----
sys.modules.setdefault('cryptography', types.ModuleType('cryptography'))
sys.modules.setdefault('cryptography.hazmat', types.ModuleType('hazmat'))
sys.modules.setdefault('cryptography.hazmat.primitives', types.ModuleType('primitives'))
sys.modules.setdefault('cryptography.hazmat.primitives.asymmetric', types.ModuleType('asymmetric'))
sys.modules.setdefault('cryptography.hazmat.primitives.asymmetric.ed25519', types.ModuleType('ed25519'))
ed = sys.modules['cryptography.hazmat.primitives.asymmetric.ed25519']
ed.Ed25519PrivateKey = types.SimpleNamespace(generate=lambda: object())
ed.Ed25519PublicKey = object
serialization = types.ModuleType('serialization')
primitives = sys.modules['cryptography.hazmat.primitives']
primitives.serialization = serialization
sys.modules['cryptography.hazmat.primitives.serialization'] = serialization

jinja_mod = types.ModuleType('jinja2')
jinja_mod.Template = lambda *a, **k: None
jinja_mod.__spec__ = types.SimpleNamespace()
sys.modules['jinja2'] = jinja_mod
sys.modules['yaml'] = types.ModuleType('yaml')
sys.modules['numpy'] = types.ModuleType('numpy')
sys.modules['env_config'] = types.SimpleNamespace(DATABASE_URL='sqlite:///:memory:')
sys.modules['httpx'] = types.ModuleType('httpx')
sys.modules['sqlalchemy'] = types.ModuleType('sqlalchemy')
sys.modules['sqlalchemy.engine'] = types.ModuleType('engine')
cov_mod = types.ModuleType('coverage')
cov_mod.Coverage = object
sys.modules['coverage'] = cov_mod

sklearn_mod = types.ModuleType('sklearn')
sklearn_linear = types.ModuleType('linear_model')
class DummyLR:
    def fit(self, *a, **k):
        return self
    def predict(self, X):
        return [0.0 for _ in range(len(X))]
sklearn_linear.LinearRegression = DummyLR
sklearn_pre = types.ModuleType('preprocessing')
class DummyPF:
    def __init__(self, *a, **k):
        pass
sklearn_pre.PolynomialFeatures = DummyPF
sklearn_mod.linear_model = sklearn_linear
sklearn_mod.preprocessing = sklearn_pre
sys.modules['sklearn'] = sklearn_mod
sys.modules['sklearn.linear_model'] = sklearn_linear
sys.modules['sklearn.preprocessing'] = sklearn_pre

import os
os.environ.setdefault('MENACE_LIGHT_IMPORTS', '1')

sys.modules['neurosales'] = types.SimpleNamespace()
sys.modules['environment_bootstrap'] = types.SimpleNamespace(EnvironmentBootstrapper=object)
sys.modules['light_bootstrap'] = types.SimpleNamespace(EnvironmentBootstrapper=object)
sys.modules['vector_service.embedding_scheduler'] = types.SimpleNamespace(start_scheduler_from_env=lambda *a, **k: None)
sys.modules['unified_event_bus'] = types.SimpleNamespace(UnifiedEventBus=object)
sys.modules['automated_reviewer'] = types.SimpleNamespace(AutomatedReviewer=object)
sys.modules['jsonschema'] = types.SimpleNamespace(ValidationError=Exception, validate=lambda *a, **k: None)
sys.modules['quick_fix_engine'] = types.SimpleNamespace(generate_patch=lambda *a, **k: None)
sys.modules['menace.quick_fix_engine'] = sys.modules['quick_fix_engine']
sys.modules['menace.patch_score_backend'] = types.SimpleNamespace()
sys.modules['sandbox_runner.scoring'] = types.SimpleNamespace(record_run=lambda *a, **k: None)
sys.modules['menace.sandbox_runner.scoring'] = sys.modules['sandbox_runner.scoring']

menace_pkg = types.ModuleType('menace')
menace_pkg.__path__ = []
sys.modules['menace'] = menace_pkg

# relative module stubs
sys.modules['menace.logging_utils'] = types.SimpleNamespace(log_record=lambda **kw: None)
sys.modules['menace.retry_utils'] = types.SimpleNamespace(with_retry=lambda f: f)
class _ErrLog:
    def __init__(self, knowledge_graph=None, context_builder=None):
        pass

sys.modules['menace.error_logger'] = types.SimpleNamespace(ErrorLogger=_ErrLog, TelemetryEvent=object)
sys.modules['menace.knowledge_graph'] = types.SimpleNamespace(KnowledgeGraph=object)
sys.modules['menace.human_alignment_agent'] = types.SimpleNamespace(HumanAlignmentAgent=object)
sys.modules['menace.human_alignment_flagger'] = types.SimpleNamespace(_collect_diff_data=lambda *a, **k: None)
sys.modules['menace.violation_logger'] = types.SimpleNamespace(log_violation=lambda *a, **k: None)
_router_stub = types.SimpleNamespace(get_connection=lambda name: (_ for _ in ()).throw(RuntimeError("no db")))
sys.modules['db_router'] = types.SimpleNamespace(
    GLOBAL_ROUTER=_router_stub, init_db_router=lambda name: _router_stub
)
def _auto_init(self, telem, engine, context_builder):
    self.telemetry_db = telem
    self.engine = engine
    self.context_builder = context_builder
    self.logger = logging.getLogger('AutomatedDebugger')

sys.modules['menace.automated_debugger'] = types.SimpleNamespace(
    AutomatedDebugger=type(
        'AutomatedDebugger',
        (object,),
        {
            '__init__': _auto_init,
            '_generate_tests': lambda self, logs: [],
            '_recent_logs': lambda self, limit=5: ['dummy'],
        },
    )
)
sys.modules['menace.self_coding_engine'] = types.SimpleNamespace(SelfCodingEngine=object)
sys.modules['menace.audit_trail'] = types.SimpleNamespace(AuditTrail=object)
sys.modules['menace.code_database'] = types.SimpleNamespace(PatchHistoryDB=object, _hash_code=lambda b: 'x')
sys.modules['menace.self_improvement_policy'] = types.SimpleNamespace(SelfImprovementPolicy=object)
sys.modules['menace.roi_tracker'] = types.SimpleNamespace(ROITracker=object)
sys.modules['menace.error_cluster_predictor'] = types.SimpleNamespace(ErrorClusterPredictor=object)
sys.modules['menace.error_parser'] = types.SimpleNamespace(ErrorReport=type('ErrorReport',(object,),{}), FailureCache=object, parse_failure=lambda *a, **k: None)
sys.modules['sandbox_runner.environment'] = types.SimpleNamespace(create_ephemeral_env=lambda *a, **k: None)
sys.modules['menace.sandbox_settings'] = types.SimpleNamespace(
    SandboxSettings=type(
        'SandboxSettings',
        (object,),
        {
            'baseline_window': 5,
            'stagnation_iters': 10,
            'delta_margin': 0.0,
            'score_weights': (1, 1, 1, 1, 1, 1),
            'merge_threshold': 0.0,
            'flakiness_runs': 1,
            'weight_update_interval': 0.0,
            'test_run_timeout': 1,
            'test_run_retries': 0,
            'patch_score_backend': None,
            'patch_score_backend_url': None,
        },
    )
)
sys.modules['menace.sandbox_runner'] = types.SimpleNamespace(post_round_orphan_scan=lambda *a, **k: None)

# vector_service stubs
class _CB:
    def exclude_failed_strategies(self, tags):
        pass
    def query(self, *a, **k):
        return [], {}
    def build_context(self, query, **kwargs):
        return {}
    def refresh_db_weights(self):
        pass
vs_mod = types.ModuleType('vector_service')
vs_mod.ContextBuilder = _CB
sys.modules['vector_service'] = vs_mod
sys.modules['vector_service.context_builder'] = types.SimpleNamespace(record_failed_tags=lambda _tags: None)

# Load SelfDebuggerSandbox module
_spec = importlib.util.spec_from_file_location('menace.self_debugger_sandbox', Path(__file__).resolve().parents[1] / 'self_debugger_sandbox.py')  # path-ignore
sds = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(sds)


class DummyTelem:
    pass


class DummyEngine:
    pass


def test_context_feedback_logs_malformed_metadata(monkeypatch, caplog):
    class DummyBuilder:
        def exclude_failed_strategies(self, tags):
            pass
        def query(self, *args, **kwargs):
            return [], {'bucket': [object()]}
        def build_context(self, query, **kwargs):
            return {}
        def refresh_db_weights(self):
            pass

    monkeypatch.setattr(sds, 'ContextBuilder', DummyBuilder)
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(), DummyEngine(), context_builder=DummyBuilder()
    )
    report = types.SimpleNamespace(trace='x')

    with caplog.at_level(logging.ERROR, logger='AutomatedDebugger'):
        assert dbg._context_feedback(report) == []

    assert any('malformed context metadata' in rec.message for rec in caplog.records)
