from __future__ import annotations

"""FastAPI service to allocate and manage session identities."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .session_vault import SessionVault, SessionData

app = FastAPI(title="Session Allocator")
_vault = SessionVault()


class GetRequest(BaseModel):
    domain: str


class SessionOut(BaseModel):
    session_id: int
    cookies: dict
    user_agent: str
    fingerprint: str


class ReportIn(BaseModel):
    session_id: int
    status: str


@app.post("/session/get", response_model=SessionOut)
def get_session(req: GetRequest) -> SessionOut:
    data = _vault.get_least_recent(req.domain)
    if not data:
        raise HTTPException(status_code=404, detail="No session available")
    return SessionOut(
        session_id=data.session_id or 0,
        cookies=data.cookies,
        user_agent=data.user_agent,
        fingerprint=data.fingerprint,
    )


@app.post("/session/report")
def report_session(info: ReportIn) -> dict:
    status = info.status.lower()
    if status not in {"success", "captcha", "banned"}:
        raise HTTPException(status_code=400, detail="Invalid status")
    _vault.report(
        info.session_id,
        success=status == "success",
        captcha=status == "captcha",
        banned=status == "banned",
    )
    return {"ok": True}


__all__ = ["app"]
