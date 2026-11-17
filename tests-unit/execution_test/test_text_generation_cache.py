import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import nodes
from comfy_execution.caching import BasicCache, CacheKeySetInputSignature
from comfy_execution.graph import DynamicPrompt
from comfy_extras.nodes_primitive import StringMultiline
from comfy_extras.nodes_textgen import TextGenerate


@pytest.fixture
def text_cache(monkeypatch):
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, 'TextGenerate', TextGenerate)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, 'PrimitiveStringMultiline', StringMultiline)
    prompt = {
        'system': {'class_type': 'PrimitiveStringMultiline', 'inputs': {'value': 'Describe the scene.'}},
        'text': {'class_type': 'TextGenerate', 'inputs': {
            'prompt': 'A blue dress.', 'system_prompt': ['system', 0], 'max_length': 4096,
            'thinking': True, 'use_default_template': True, 'mtp': 'auto',
            'sampling_mode': 'on', 'sampling_mode.seed': 17, 'sampling_mode.temperature': 1.0,
        }},
    }
    return BasicCache(CacheKeySetInputSignature), prompt, SimpleNamespace(get=AsyncMock(return_value=False))


def select_prompt(cache, prompt, changed):
    asyncio.run(cache.set_prompt(DynamicPrompt(prompt), prompt.keys(), changed))


@pytest.mark.parametrize('node,field,value', [
    ('text', 'prompt', 'A pink dress.'),
    ('text', 'prompt', 'A blue dress. '),
    ('text', 'prompt', 'A blue dress with flowers.'),
    ('text', 'thinking', False),
    ('text', 'max_length', 4097),
    ('text', 'sampling_mode.seed', 18),
    ('text', 'sampling_mode.temperature', .7),
    ('text', 'use_default_template', False),
    ('text', 'mtp', 'off'),
    ('system', 'value', 'Describe the colors.'),
])
def test_text_generation_cache_invalidates_changed_inputs(text_cache, node, field, value):
    cache, prompt, changed = text_cache
    select_prompt(cache, prompt, changed)
    cache.set_local('text', 'original caption')
    modified = copy.deepcopy(prompt)
    modified[node]['inputs'][field] = value
    select_prompt(cache, modified, changed)
    assert cache.get_local('text') is None


def test_text_generation_cache_keeps_same_length_prompts_separate(text_cache):
    cache, blue, changed = text_cache
    pink = copy.deepcopy(blue)
    pink['text']['inputs']['prompt'] = 'A pink dress.'
    assert len(blue['text']['inputs']['prompt']) == len(pink['text']['inputs']['prompt'])
    select_prompt(cache, blue, changed)
    cache.set_local('text', 'blue caption')
    select_prompt(cache, pink, changed)
    assert cache.get_local('text') is None
    cache.set_local('text', 'pink caption')
    select_prompt(cache, blue, changed)
    assert cache.get_local('text') == 'blue caption'
    select_prompt(cache, pink, changed)
    assert cache.get_local('text') == 'pink caption'


def test_text_generation_cache_can_force_recomputation(text_cache):
    cache, prompt, changed = text_cache
    select_prompt(cache, prompt, changed)
    cache.set_local('text', 'previous caption')
    changed.get.side_effect = lambda node: float('nan') if node == 'text' else False
    select_prompt(cache, prompt, changed)
    assert cache.get_local('text') is None
    cache.set_local('text', 'recomputed caption')
    assert cache.get_local('text') == 'recomputed caption'
    select_prompt(cache, prompt, changed)
    assert cache.get_local('text') is None
