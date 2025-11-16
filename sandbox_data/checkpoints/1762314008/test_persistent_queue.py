import types
import sys


def _stub_deps(monkeypatch):
    fastapi_mod = types.ModuleType("fastapi")
    fastapi_mod.FastAPI = type("App", (), {})
    fastapi_mod.HTTPException = type("HTTPException", (Exception,), {})
    fastapi_mod.Header = lambda default=None: default
    monkeypatch.setitem(sys.modules, "fastapi", fastapi_mod)

    pydantic_mod = types.ModuleType("pydantic")
    pydantic_mod.BaseModel = type("BaseModel", (), {})
    monkeypatch.setitem(sys.modules, "pydantic", pydantic_mod)

    uvicorn_mod = types.ModuleType("uvicorn")
    uvicorn_mod.Config = type("Config", (), {})
    uvicorn_mod.Server = type("Server", (), {})
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn_mod)

