"""Tests for the concept probability shift metric in src/steering/analysis.py."""
import pytest

torch = pytest.importorskip("torch")

from src.steering.analysis import compute_concept_probability_shift, create_motif_evaluator  # noqa: E402

VOCAB = ["<cls>", "<pad>", "<eos>", "A", "C", "G", "H", "K", "S"]


class SpacedTokenizer:
    """Mimics ESM tokenizers, which decode residues separated by spaces."""

    all_special_ids = [0, 1, 2]

    def decode(self, ids, skip_special_tokens=False):
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        toks = [VOCAB[i] for i in ids if not (skip_special_tokens and i in self.all_special_ids)]
        return " ".join(toks)

    def encode(self, seq):
        return [0] + [VOCAB.index(a) for a in seq] + [2]


def logits_favouring(ids, pos, token, strength=20.0):
    """Logits that copy the input everywhere except `pos`, where `token` is favoured."""
    L = len(ids)
    logits = torch.full((1, L, len(VOCAB)), -strength)
    for i, t in enumerate(ids):
        logits[0, i, t] = strength
    logits[0, pos] = -strength
    logits[0, pos, VOCAB.index(token)] = strength
    return logits


@pytest.mark.unit
def test_masked_token_detects_motif_created_by_steering():
    tok = SpacedTokenizer()
    # "GAAAAAKS" is one substitution (position 6, A->G) away from the P-loop motif G.{4}GK[ST]
    seq = "GAAAAAKS"
    ids = tok.encode(seq)
    input_ids = torch.tensor([ids])
    baseline = logits_favouring(ids, 6, "A")  # token index 6 = residue index 5
    steered = logits_favouring(ids, 6, "G")
    res = compute_concept_probability_shift(
        baseline, steered, tok, create_motif_evaluator("G.{4}GK[ST]"), input_ids=input_ids, method="masked_token"
    )
    # only the substituted residue creates the motif, so its probability dominates the shift
    assert res["baseline_concept_prob"] < 0.01
    assert res["steered_concept_prob"] == pytest.approx(1 / len(seq), rel=1e-3)
    assert res["probability_shift"] > 0.1


@pytest.mark.unit
def test_sequence_level_decodes_without_spaces():
    tok = SpacedTokenizer()
    seq = "GAAAAGKS"  # already contains the motif
    ids = tok.encode(seq)
    input_ids = torch.tensor([ids])
    logits = logits_favouring(ids, 1, "G")
    res = compute_concept_probability_shift(
        logits, logits, tok, create_motif_evaluator("G.{4}GK[ST]"), input_ids=input_ids, method="sequence_level"
    )
    assert res["baseline_concept_prob"] == 1.0
    assert res["probability_shift"] == 0.0
