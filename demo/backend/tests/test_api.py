"""API tests. They load the real ESM-2 35M weights (~140 MB download on first run).

    pip install -r demo/backend/requirements.txt pytest httpx torch
    pytest demo/backend/tests -q
"""
import pytest
from fastapi.testclient import TestClient

from demo.backend.app.engine import clean_sequence
from demo.backend.app.main import app

UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_clean_sequence_accepts_fasta():
    assert clean_sequence(">sp|X\nmqif vktl\nTGKT 12\n") == "MQIFVKTLTGKT"


@pytest.mark.parametrize("bad", ["", "MKT", "MKT1@#$" * 3, "A" * 401])
def test_clean_sequence_rejects(bad):
    with pytest.raises(ValueError):
        clean_sequence(bad)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["layers"] == 12
    assert {c["key"] for c in body["concepts"]} >= {"zinc_finger", "p_loop"}


def test_analyze(client):
    r = client.post("/api/analyze", json={"sequence": UBIQUITIN, "layer": 6})
    assert r.status_code == 200
    body = r.json()
    L = len(UBIQUITIN)
    assert body["length"] == L
    assert len(body["p_native"]) == L
    assert set(body["probes"]) == {"helix", "strand", "transmembrane"}
    assert all(len(v) == L for v in body["probes"].values())
    assert body["tracks"] and all(len(t["v"]) == L for t in body["tracks"])
    assert len(body["fingerprint"]) > 0


def test_analyze_rejects_bad_sequence(client):
    r = client.post("/api/analyze", json={"sequence": "NOT A PROTEIN 123!"})
    assert r.status_code == 422


def test_steer_zero_strength_is_identity_like(client):
    r = client.post("/api/steer", json={"sequence": UBIQUITIN, "concept": "zinc_finger", "strength": 0})
    assert r.status_code == 200
    body = r.json()
    assert body["identity"] > 0.9
    assert all(abs(c - 1) < 1e-4 for c in body["drift"])


def test_steer_changes_prediction(client):
    base = client.post("/api/steer", json={"sequence": UBIQUITIN, "concept": "membrane", "strength": 0}).json()
    steered = client.post("/api/steer", json={"sequence": UBIQUITIN, "concept": "membrane", "strength": 3}).json()
    assert steered["concept_score"] > base["concept_score"]
    assert steered["drift"][-1] < 0.999


def test_steer_unknown_concept(client):
    r = client.post("/api/steer", json={"sequence": UBIQUITIN, "concept": "nope", "strength": 1})
    assert r.status_code == 422
