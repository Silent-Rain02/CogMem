"""A malformed extraction is a failed operation, not an empty successful memory."""
from types import SimpleNamespace
import pytest
from cogmem.memory.indexer import Indexer


def instance(response):
    i=object.__new__(Indexer);i._llm=SimpleNamespace(chat=lambda **kwargs:response);return i


@pytest.mark.parametrize('response',['not JSON','[]','{}','{"events":[],"facts":{},"claims":[]}','{"events":["unsupported"],"facts":[],"claims":[]}'])
def test_failed_parse_or_wrong_shape_is_not_reported_as_empty_extraction(response):
    with pytest.raises(ValueError):instance(response)._extract_structured_from_chunk('Original text','source')


def test_explicit_valid_empty_extraction_retains_its_meaning():
    assert instance('{"events":[],"facts":[],"claims":[]}')._extract_structured_from_chunk('Hello','source')=={'events':[],'facts':[],'claims':[]}
