from types import SimpleNamespace

import pytest
import torch

import comfy.model_prefetch
import comfy.utils
from comfy.text_encoders.llama import BaseGenerate


class Generator(BaseGenerate):
    def __init__(self, readback, sequence=None):
        self.readback = readback
        self.sequence = sequence
        self.calls = 0
        self.model = SimpleNamespace(
            config=SimpleNamespace(stop_tokens=[4]), graph_dynamic_vbar_blocks=False,
            embed_tokens=torch.nn.Embedding(5, 3), lm_head=torch.nn.Identity(), forward=self.forward)

    def init_kv_cache(self, *args):
        return []

    def _generation_readback_size(self, embeds, caches):
        return self.readback

    def forward(self, *args, **kwargs):
        self.calls += 1
        if self.sequence is None:
            logits = torch.tensor([-2., .3, .7, -.5, .1])
        else:
            logits = torch.full((5,), -100.)
            logits[self.sequence[min(self.calls - 1, len(self.sequence) - 1)]] = 100.
        return logits.reshape(1, 1, 5), None, []

    def run(self, **kwargs):
        return self.generate(embeds=torch.zeros(1, 2, 3), execution_dtype=torch.float32, **kwargs)


@pytest.mark.parametrize('readback', [1, 4, 8])
@pytest.mark.parametrize('sequence,limit,expected', [
    ([4, 1, 2], 9, [4]),
    ([1, 2, 4, 3], 9, [1, 2, 4]),
    ([1, 2, 3, 1, 4, 2], 9, [1, 2, 3, 1, 4]),
    ([1, 2, 3, 1, 2, 3], 5, [1, 2, 3, 1, 2]),
    ([1], 0, []),
])
def test_readback_preserves_eos_limit_and_progress(monkeypatch, readback, sequence, limit, expected):
    progress = []
    cleaned = []
    monkeypatch.setattr(comfy.utils, 'ProgressBar', lambda total: SimpleNamespace(update=progress.append))
    monkeypatch.setattr(comfy.model_prefetch, 'cleanup_prefetch_queues', lambda: cleaned.append(True))
    model = Generator(readback, sequence)
    assert model.run(max_length=limit, do_sample=False) == expected
    assert sum(progress) == len(expected)
    assert model.calls <= limit
    assert cleaned == [True]


@pytest.mark.parametrize('seed', [1, 17, 29])
def test_readback_preserves_sampling_and_request_rng(seed):
    options = dict(max_length=19, do_sample=True, seed=seed, top_k=3, top_p=.95,
                   presence_penalty=1.5)
    expected = Generator(1).run(**options)
    model = Generator(4)
    assert model.run(**options) == expected
    model.calls = 0
    assert model.run(**options) == expected


def test_readback_cleans_up_after_failure(monkeypatch):
    cleaned = []
    model = Generator(4, [1])

    def fail(*args, **kwargs):
        raise RuntimeError('generation interrupted')

    model.model.forward = fail
    monkeypatch.setattr(comfy.model_prefetch, 'cleanup_prefetch_queues', lambda: cleaned.append(True))
    with pytest.raises(RuntimeError, match='generation interrupted'):
        model.run(max_length=8, do_sample=False)
    assert cleaned == [True]
