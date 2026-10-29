"""Public calendar helpers obey the same identity/lifecycle/time contracts."""
import pytest
from cogmem.storage.graph_store import GraphStore
from cogmem.storage.schema import Fact, Event


def test_subject_filter_uses_the_declared_subject_id_and_coarse_validity():
    g=GraphStore();g.add_node(Fact(id='f',subject_id='p',object_id='c_place',predicate='resident',content='Resident',valid_from='2020',valid_to='2020'))
    assert [f.id for f in g.get_facts_by_validity('2020-06',subject='p')]==['f']
    assert g.get_facts_by_validity('2022',subject='p')==[]


def test_unknown_and_archived_validity_cannot_certify_a_date():
    g=GraphStore();g.add_node(Fact(id='unknown',subject_id='p',object_id='c_place',predicate='resident',content='Resident'))
    g.add_node(Fact(id='archived',subject_id='p',object_id='c_place',predicate='resident',content='Resident',valid_from='2020',is_archived=True))
    assert g.get_facts_by_validity('2020')==[]
    with pytest.raises(ValueError):g.get_facts_by_validity('invalid')


def test_one_sided_range_does_not_admit_old_or_unknown_occurrences():
    g=GraphStore()
    for key,time in [('old','2019-01-01'),('new','2022-01-01'),('unknown',None)]:
        g.add_node(Event(id=key,title=key,time_start=time,participants=['p'],session_time='2023-01-01'))
    assert [e.id for e in g.get_events_by_time_range(start_time='2021',entity_id='p')]==['new']
    assert [e.id for e in g.get_events_by_time_range(end_time='2020')]==['old']
    with pytest.raises(ValueError):g.get_events_by_time_range('2022','2020')
    with pytest.raises(ValueError):g.get_events_by_time_range('invalid')


def test_version_queries_follow_replacements_and_do_not_guess_across_forks():
    g=GraphStore()
    for key in ['old','new']:
        g.add_node(Fact(id=key,subject_id='p',object_id='c_place',predicate='resident',content=key))
    g._g.add_edge('new','old',rel_type='SUPERSEDES')
    assert g.get_supersedes_chain('old')==['old','new']
    assert g.get_latest_fact('p','resident').id=='new'
    g.add_node(Fact(id='fork',subject_id='p',object_id='c_other',predicate='resident',content='Unresolved alternative'))
    g._g.add_edge('fork','old',rel_type='SUPERSEDES')
    assert g.get_supersedes_chain('old')==['old','fork','new']
    assert g.get_latest_fact('p','resident') is None
    g._g.add_edge('old','new',rel_type='SUPERSEDES')
    assert set(g.get_supersedes_chain('old'))=={'old','new','fork'}
    assert g.get_latest_fact('p','resident') is None
