from __future__ import annotations

import pytest


def test_score_in_batches_preserves_order():
    torch = pytest.importorskip("torch")
    from iqa_ssc.learned_iqa import score_in_batches

    class Metric:
        def __call__(self, batch):
            return batch[:, :1, :1, :1].reshape(-1)

    images = torch.arange(5, dtype=torch.float32).reshape(5, 1, 1, 1)
    assert score_in_batches(Metric(), images, device="cpu", batch_size=2) == list(range(5))


def test_score_in_batches_rejects_nonpositive_batch_size():
    torch = pytest.importorskip("torch")
    from iqa_ssc.learned_iqa import score_in_batches

    with pytest.raises(ValueError):
        score_in_batches(lambda batch: batch, torch.zeros(1, 1, 1, 1), device="cpu", batch_size=0)
