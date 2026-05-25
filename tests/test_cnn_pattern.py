"""
tests/test_cnn_pattern.py
---------------------------
Unit tests for the 1D CNN Pattern Recognition service.

Tests cover:
- Model construction (dual-head)
- Training with labelled data
- Score output types and ranges
- Pattern classification produces valid strings
- Save/load round-trip
"""

import numpy as np
import pytest
import tempfile

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")

from app.services.neural.cnn_pattern import (
    build_model,
    train,
    save_model,
    load_model,
    score,
    PATTERN_TYPES,
)

SEQ_LEN = 32
N_FEATURES = 1
N_SAMPLES = 60


def _make_labelled_data(n: int = N_SAMPLES):
    rng = np.random.default_rng(1)
    seqs, labels_b, labels_p = [], [], []

    per_class = n // 4
    for cls_idx in range(4):
        for _ in range(per_class):
            if cls_idx == 0:  # normal
                flow = rng.normal(0.5, 0.05, SEQ_LEN)
            elif cls_idx == 1:  # burst
                flow = rng.normal(0.5, 0.05, SEQ_LEN)
                spike_pos = rng.integers(5, SEQ_LEN - 5)
                flow[spike_pos: spike_pos + 3] += 3.0
            elif cls_idx == 2:  # background
                flow = rng.normal(0.5, 0.05, SEQ_LEN) + 0.3
            else:  # intermittent
                flow = rng.normal(0.5, 0.05, SEQ_LEN)
                for pos in range(0, SEQ_LEN, 8):
                    flow[pos: pos + 2] += rng.uniform(0.2, 0.5)

            flow = np.maximum(flow, 0)
            # Normalise
            std = flow.std()
            if std > 0:
                flow = (flow - flow.mean()) / std
            else:
                flow = np.zeros_like(flow)

            seqs.append(flow.reshape(SEQ_LEN, 1).astype(np.float32))
            labels_b.append(0 if cls_idx == 0 else 1)
            labels_p.append(cls_idx)

    return (
        np.array(seqs),
        np.array(labels_b, dtype=np.float32),
        np.array(labels_p, dtype=np.int32),
    )


class TestCNNBuild:
    def test_model_builds(self):
        model = build_model(seq_len=SEQ_LEN, n_features=N_FEATURES)
        assert model is not None

    def test_output_shapes(self):
        model = build_model(seq_len=SEQ_LEN, n_features=N_FEATURES)
        model.compile(optimizer="adam", loss={"leak_score": "binary_crossentropy", "pattern_type": "sparse_categorical_crossentropy"})
        x = np.zeros((2, SEQ_LEN, 1), dtype=np.float32)
        outputs = model.predict(x, verbose=0)
        assert outputs[0].shape == (2, 1)      # leak_score
        assert outputs[1].shape == (2, 4)      # pattern_type softmax


class TestCNNTrain:
    def test_train_returns_model(self):
        seqs, labels_b, labels_p = _make_labelled_data()
        model = train(seqs, labels_b, labels_p, epochs=2, batch_size=8)
        assert model is not None


class TestCNNScore:
    def test_score_returns_float_and_string(self):
        seqs, labels_b, labels_p = _make_labelled_data()
        model = train(seqs, labels_b, labels_p, epochs=2, batch_size=8)
        leak_score, pattern = score(model, seqs[0])
        assert isinstance(leak_score, float)
        assert isinstance(pattern, str)
        assert pattern in PATTERN_TYPES

    def test_score_in_range(self):
        seqs, labels_b, labels_p = _make_labelled_data()
        model = train(seqs, labels_b, labels_p, epochs=2, batch_size=8)
        for seq in seqs[:5]:
            s, _ = score(model, seq)
            assert 0.0 <= s <= 1.0


class TestCNNPersistence:
    def test_save_load_roundtrip(self):
        seqs, labels_b, labels_p = _make_labelled_data()
        model = train(seqs, labels_b, labels_p, epochs=2, batch_size=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_model(model, tmpdir, "zone_test")
            loaded = load_model(tmpdir, "zone_test")
            assert loaded is not None

            s1, p1 = score(model, seqs[0])
            s2, p2 = score(loaded, seqs[0])
            assert s1 == pytest.approx(s2, abs=0.01)
            assert p1 == p2

    def test_load_nonexistent_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert load_model(tmpdir, "nonexistent_zone") is None
