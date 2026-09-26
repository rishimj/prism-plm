"""FastAPI service for the PRISM-Bio live demo."""
from __future__ import annotations

import os
import time
from collections import OrderedDict, defaultdict, deque
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

from .engine import Engine, clean_sequence

ALLOWED_ORIGINS = [o.strip() for o in os.environ.get(
    "ALLOWED_ORIGINS", "https://rishimj.github.io,http://localhost:5173,http://localhost:4173").split(",") if o.strip()]
RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "30"))
CACHE_SIZE = 256

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["engine"] = Engine()
    state["started"] = time.time()
    yield


app = FastAPI(title="PRISM-Bio demo API", version="1.0.0", lifespan=lifespan,
              description="Live interpretability of the ESM-2 protein language model.")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type"], max_age=3600)
app.add_middleware(GZipMiddleware, minimum_size=1024)

_hits: dict = defaultdict(deque)
_cache: "OrderedDict[tuple, dict]" = OrderedDict()


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "unknown")


def _rate_limit(request: Request) -> None:
    now = time.time()
    q = _hits[_client_ip(request)]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        raise HTTPException(429, "Too many requests; please wait a minute.")
    q.append(now)


def _cached(key: tuple, fn):
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    value = fn()
    _cache[key] = value
    if len(_cache) > CACHE_SIZE:
        _cache.popitem(last=False)
    return value


class AnalyzeRequest(BaseModel):
    sequence: str = Field(..., max_length=5000)
    layer: int = Field(6, ge=1, le=48)
    units: Optional[List[int]] = Field(None, max_length=8)
    marginals: bool = True


class SteerRequest(BaseModel):
    sequence: str = Field(..., max_length=5000)
    concept: str
    strength: float = Field(1.0, ge=-3.0, le=4.0)


@app.get("/api/health")
def health():
    engine: Engine = state["engine"]
    return {"status": "ok", "uptime_s": round(time.time() - state["started"]), **engine.info()}


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest, request: Request):
    _rate_limit(request)
    try:
        seq = clean_sequence(req.sequence)
    except ValueError as e:
        raise HTTPException(422, str(e))
    engine: Engine = state["engine"]
    key = ("analyze", seq, req.layer, tuple(req.units or []), req.marginals)
    return await run_in_threadpool(_cached, key, lambda: engine.analyze(seq, req.layer, req.units, req.marginals))


@app.post("/api/steer")
async def steer(req: SteerRequest, request: Request):
    _rate_limit(request)
    try:
        seq = clean_sequence(req.sequence)
    except ValueError as e:
        raise HTTPException(422, str(e))
    engine: Engine = state["engine"]
    key = ("steer", seq, req.concept, round(req.strength, 3))
    try:
        return await run_in_threadpool(_cached, key, lambda: engine.steer(seq, req.concept, req.strength))
    except ValueError as e:
        raise HTTPException(422, str(e))
