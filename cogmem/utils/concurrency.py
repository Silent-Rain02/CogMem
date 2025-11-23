from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, List, TypeVar

T = TypeVar("T")
U = TypeVar("U")


def parallel_map(func: Callable[[T], U], items: Iterable[T], max_workers: int = 8) -> List[U]:
    results: List[U] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_item = {ex.submit(func, it): it for it in items}
        for fut in as_completed(future_to_item):
            results.append(fut.result())
    return results
