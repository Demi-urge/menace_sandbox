import os, sys, types
sys.modules.setdefault("stripe_billing_router", types.ModuleType("stripe_billing_router"))
sys.modules.setdefault("sqlalchemy", types.ModuleType("sqlalchemy"))
sqlalchemy_orm = types.ModuleType("sqlalchemy.orm")
sqlalchemy_orm.declarative_base = lambda *a, **k: None
sys.modules.setdefault("sqlalchemy.orm", sqlalchemy_orm)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
