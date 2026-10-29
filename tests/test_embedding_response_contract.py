from types import SimpleNamespace

import pytest

from cogmem.config import CogMemConfig
from cogmem.models.llm_client import OpenAICompatClient
from cogmem.utils.cache import LRUCache


def client(create, batch_size):
    value = object.__new__(OpenAICompatClient)
    value._cfg = CogMemConfig()
    value._cfg.runtime.embedding_batch_size = batch_size
    value._embed_cache = LRUCache(8)
    value._client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    return value


def test_embedding_request_uses_configured_batch_and_restores_response_order():
    batches = []
    def create(model, input):
        batches.append(list(input))
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[float(input[i])]) for i in reversed(range(len(input)))])
    value = client(create, 32)
    inputs = [str(i) for i in range(35)]
    assert value.embed(inputs) == [[float(i)] for i in range(35)]
    assert [len(batch) for batch in batches] == [32, 3]
    assert value.embed(inputs) == [[float(i)] for i in range(35)]
    assert len(batches) == 2


@pytest.mark.parametrize('indices', [[0, 0], [0], [0, 2]])
def test_malformed_embedding_alignment_is_not_cached(indices):
    value = client(lambda **kwargs: SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[1.]) for i in indices]), 10)
    # Test one transport response without retrying a deterministic boundary failure.
    with pytest.raises(ValueError):
        OpenAICompatClient.embed.__wrapped__(value, ['a', 'b'])
    assert value._embed_cache.get((value._cfg.model.embedding_model, None, ('a', 'b'))) is None


def test_embedding_dimensions_are_native_request_options_and_part_of_cache_identity():
    calls=[]
    def create(model,input,dimensions):
        calls.append(dimensions)
        return SimpleNamespace(data=[SimpleNamespace(index=i,embedding=[1.]*dimensions) for i in range(len(input))])
    value=client(create,32)
    value._cfg.model.embedding_dimensions=2
    assert value.embed(['text']) == [[1.,1.]]
    value._cfg.model.embedding_dimensions=1
    assert value.embed(['text']) == [[1.]]
    assert calls == [2,1]


def test_wrong_requested_embedding_dimension_fails_before_caching():
    value=client(lambda **kwargs:SimpleNamespace(data=[SimpleNamespace(index=0,embedding=[1.])]),10)
    value._cfg.model.embedding_dimensions=2
    with pytest.raises(ValueError):OpenAICompatClient.embed.__wrapped__(value,['text'])
    assert value._embed_cache.get((value._cfg.model.embedding_model,2,('text',))) is None


def test_native_chat_options_are_sent_and_cache_respects_option_changes():
    calls=[]
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=str(len(calls))))],usage=None)
    value=object.__new__(OpenAICompatClient)
    value._cfg=CogMemConfig();value._chat_cache=LRUCache(8)
    value._cfg.model.chat_max_tokens=64
    value._cfg.model.chat_extra_body={'enable_thinking':False}
    value._client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert value.chat('system','question') == '1'
    assert value.chat('system','question') == '1'
    assert calls[0]['max_tokens'] == 64 and calls[0]['extra_body'] == {'enable_thinking':False}
    value._cfg.model.chat_max_tokens=128
    assert value.chat('system','question') == '2'


def test_native_sdk_has_only_one_retry_layer():
    config=CogMemConfig();config.model.api_base_url='https://example.test/v1';config.model.api_key='test-local-placeholder'
    value=OpenAICompatClient(config)
    try:assert value._client.max_retries == 0
    finally:value._client.close()
