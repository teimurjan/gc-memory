from __future__ import annotations

import numpy as np
import torch

from gc_memory.mlp_adapter import DeltaPredictor, predict_delta, train_step


class TestDeltaPredictor:
    def test_output_shape(self) -> None:
        predictor = DeltaPredictor(embed_dim=384, hidden=128)
        q = torch.randn(384)
        e = torch.randn(384)
        s = torch.tensor(1.0)
        delta = predictor(q, e, s)
        assert delta.shape == (384,)

    def test_initial_deltas_small(self) -> None:
        predictor = DeltaPredictor(embed_dim=384, hidden=128)
        q = torch.randn(384)
        e = torch.randn(384)
        s = torch.tensor(0.0)
        delta = predictor(q, e, s)
        assert float(delta.norm()) < 1.0  # small due to xavier init with gain=0.1


class TestTrainStep:
    def test_loss_decreases_over_steps(self) -> None:
        predictor = DeltaPredictor(embed_dim=384, hidden=128)
        optimizer = torch.optim.SGD(predictor.parameters(), lr=1e-2)
        rng = np.random.default_rng(42)
        query = rng.standard_normal(384).astype(np.float32)
        query /= np.linalg.norm(query)
        embedding = rng.standard_normal(384).astype(np.float32)
        embedding /= np.linalg.norm(embedding)

        losses = []
        for _ in range(20):
            _, loss = train_step(predictor, optimizer, query, embedding, 5.0, 0.5)
            losses.append(loss)
        # Loss should decrease (high xenc = relevant, drives toward query)
        assert losses[-1] < losses[0]

    def test_delta_norm_clipped(self) -> None:
        predictor = DeltaPredictor(embed_dim=384, hidden=128)
        optimizer = torch.optim.SGD(predictor.parameters(), lr=1.0)  # large lr
        rng = np.random.default_rng(42)
        query = rng.standard_normal(384).astype(np.float32)
        query /= np.linalg.norm(query)
        embedding = rng.standard_normal(384).astype(np.float32)
        embedding /= np.linalg.norm(embedding)

        for _ in range(50):
            delta, _ = train_step(predictor, optimizer, query, embedding, 5.0, 0.3)
        assert float(np.linalg.norm(delta)) <= 0.3 + 1e-6


class TestPredictDelta:
    def test_inference_no_grad(self) -> None:
        predictor = DeltaPredictor(embed_dim=384, hidden=128)
        rng = np.random.default_rng(42)
        q = rng.standard_normal(384).astype(np.float32)
        e = rng.standard_normal(384).astype(np.float32)
        delta = predict_delta(predictor, q, e, 1.0, 0.5)
        assert delta.shape == (384,)
        assert delta.dtype == np.float32
