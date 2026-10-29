from cogmem.memory.consolidator import MemoryConsolidator
from cogmem.config import CogMemConfig
from types import SimpleNamespace
import pytest


def windows(events, days=30):
    return {frozenset(e['id'] for e in group) for group in MemoryConsolidator._bounded_event_windows(events,days)}


def test_events_years_apart_cannot_be_synthesized_as_one_temporal_pattern():
    events=[dict(id='a',time_start='2020-01-01'),dict(id='b',time_start='2020-01-15'),dict(id='c',time_start='2022-01-01')]
    assert windows(events)=={frozenset(['a','b']),frozenset(['c'])}


def test_unknown_occurrence_time_is_not_replaced_by_recording_time():
    events=[dict(id='a',session_time='2020-01-01'),dict(id='b',time_start='2020')]
    assert windows(events)==set()


def test_maximal_overlapping_windows_do_not_lose_a_valid_later_cluster():
    events=[dict(id=str(i),time_start=d) for i,d in enumerate(['2020-01-01','2020-01-31','2020-02-01','2020-02-02'])]
    assert windows(events)=={frozenset(['0','1']),frozenset(['1','2','3'])}
    assert windows(events)==windows(list(reversed(events)))


def test_coarse_interval_must_fit_entirely_not_just_its_first_day():
    events=[dict(id='month',time_start='2020-02'),dict(id='day',time_start='2020-02-10')]
    assert windows(events,10)=={frozenset(['day'])}


def test_batch_results_keep_exact_support_for_multiple_same_person_concept_tasks():
    c=object.__new__(MemoryConsolidator);c.config=CogMemConfig()
    c._synthesize_fact=lambda p,name,concept,events:dict(content=events[0]['id'],predicate='observed')
    tasks=[('p','Person','c',[dict(id='early')]),('p','Person','c',[dict(id='late')])]
    results=c._synthesize_facts_batch(tasks,max_workers=2)
    assert len(results)==2
    assert all(task[3][0]['id']==fact['content'] for task,fact in results)
