import sys
import types
from pathlib import Path
import pytest
def setup_stubs(monkeypatch):
    stub_pm = types.ModuleType("menace.prediction_manager_bot")
    stub_pm.PredictionManager = object
    monkeypatch.setitem(sys.modules, "menace.prediction_manager_bot", stub_pm)
    monkeypatch.setitem(sys.modules, "menace.ga_prediction_bot", types.ModuleType("menace.ga_prediction_bot"))

    jinja_stub = types.ModuleType("jinja2")
    class _Template:
        def __init__(self, *a, **k):
            pass
        def render(self, *a, **k):
            return ""
    jinja_stub.Template = _Template
    monkeypatch.setitem(sys.modules, "jinja2", jinja_stub)

    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = lambda *a, **k: {}
    monkeypatch.setitem(sys.modules, "yaml", yaml_stub)

    sqlalchemy_stub = types.ModuleType("sqlalchemy")
    sqlalchemy_stub.engine = types.SimpleNamespace(Engine=object)
    sqlalchemy_stub.Column = object
    sqlalchemy_stub.Integer = object
    sqlalchemy_stub.MetaData = object
    sqlalchemy_stub.String = object
    sqlalchemy_stub.Table = object
    sqlalchemy_stub.create_engine = lambda *a, **k: object()
    sqlalchemy_stub.exc = types.SimpleNamespace(SQLAlchemyError=Exception)
    sqlalchemy_stub.orm = types.SimpleNamespace(sessionmaker=lambda **k: None)
    monkeypatch.setitem(sys.modules, "sqlalchemy", sqlalchemy_stub)
    monkeypatch.setitem(sys.modules, "sqlalchemy.engine", sqlalchemy_stub.engine)
    monkeypatch.setitem(sys.modules, "sqlalchemy.orm", sqlalchemy_stub.orm)

    conv_stub = types.ModuleType("menace.conversation_manager_bot")
    class _ConvBot:
        def notify(self, msg):
            pass
    conv_stub.ConversationManagerBot = _ConvBot
    monkeypatch.setitem(sys.modules, "menace.conversation_manager_bot", conv_stub)

    report_stub = types.ModuleType("menace.report_generation_bot")
    report_stub.ReportGenerationBot = type("X", (), {})
    report_stub.ReportOptions = type("Y", (), {})
    monkeypatch.setitem(sys.modules, "menace.report_generation_bot", report_stub)

    cgi_stub = types.ModuleType("menace.chatgpt_idea_bot")
    cgi_stub.ChatGPTClient = object
    monkeypatch.setitem(sys.modules, "menace.chatgpt_idea_bot", cgi_stub)


def test_management_ingest_queries_router(monkeypatch, tmp_path):
    setup_stubs(monkeypatch)
    import importlib
    import menace
    importlib.reload(menace)
    try:
        import menace.database_management_bot as dmb
    except Exception:
        pytest.skip("imports failed")
    from menace.preliminary_research_bot import BusinessData

    class Prelim:
        def process_model(self, name, urls):
            return BusinessData(model_name=name)
    class Capital:
        def energy_score(self, **k):
            return 0.0

    monkeypatch.setattr(dmb, 'process_idea', lambda *a, **k: 'ok')

    router = types.SimpleNamespace(terms=[])
    router.query_all = lambda term: router.terms.append(term) or {}

    bot = dmb.DatabaseManagementBot(prelim_bot=Prelim(), capital_bot=Capital(), db_router=router, db_path=tmp_path/'db.db')
    bot.ingest_idea('idea')
    assert 'idea' in router.terms
    for m in ('menace.database_management_bot',):
        sys.modules.pop(m, None)


def test_deployment_queries_router(monkeypatch, tmp_path):
    setup_stubs(monkeypatch)
    import importlib, menace
    importlib.reload(menace)
    try:
        import menace.deployment_bot as dep
    except Exception:
        pytest.skip("imports failed")

    router = types.SimpleNamespace(terms=[])
    router.query_all = lambda term: router.terms.append(term) or {}
    ddb = dep.DeploymentDB(tmp_path / 'd.db')
    bot = dep.DeploymentBot(ddb, db_router=router)
    monkeypatch.setattr(bot, 'prepare_environment', lambda spec: True)
    monkeypatch.setattr(bot, 'build_containers', lambda bots, model_id=None: ([], []))
    monkeypatch.setattr(bot, 'run_tests', lambda **k: (True, []))
    spec = dep.DeploymentSpec(name='test', resources={}, env={})
    bot.deploy('deployname', [], spec)
    assert 'deployname' in router.terms
    for m in ('menace.deployment_bot',):
        sys.modules.pop(m, None)


def test_steward_deduplicate_queries_router(monkeypatch):
    setup_stubs(monkeypatch)
    import importlib, menace
    importlib.reload(menace)
    try:
        import menace.database_steward_bot as dsb
    except Exception:
        pytest.skip("imports failed")

    router = types.SimpleNamespace(terms=[])
    router.query_all = lambda term: router.terms.append(term) or {}

    class FakeSQL:
        def __init__(self, *a, **k):
            self.records = [{'id':1,'name':'a'},{'id':2,'name':'a'}]
            class Eng:
                def begin(self_inner):
                    class Conn:
                        def __enter__(self_inner2): return self_inner2
                        def __exit__(self_inner2, *a): pass
                        def execute(self_inner2, *a, **k): pass
                    return Conn()
            self.engine = Eng()
            class Template:
                def delete(self_inner):
                    class Stmt:
                        def where(self_inner2, *a, **k): pass
                    return Stmt()
                class C:
                    id = 0
                c = C()
            self.templates = Template()
        def fetch(self):
            return list(self.records)
    monkeypatch.setattr(dsb, 'SQLStore', FakeSQL)
    bot = dsb.DatabaseStewardBot(db_router=router)
    bot.deduplicate()
    assert 'templates' in router.terms
    for m in ('menace.database_steward_bot',):
        sys.modules.pop(m, None)
