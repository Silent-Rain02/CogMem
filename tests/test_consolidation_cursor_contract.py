from datetime import datetime, timezone
from cogmem.memory.consolidator import MemoryConsolidator
from cogmem.storage.graph_store import GraphStore
import pytest


def instance():
    c=object.__new__(MemoryConsolidator);c.graph=GraphStore();c._last_consolidation_timestamp='2021-01-01T00:00:00+00:00';return c


def test_incremental_empty_scope_is_empty_and_failure_does_not_advance_cursor():
    c=instance();seen=[]
    c.run_consolidation=lambda **kwargs:(seen.append(kwargs['affected_person_ids']) or {'errors':['Failed']})
    c.run_incremental_consolidation(affected_person_ids=[])
    assert seen==[[]] and c._last_consolidation_timestamp=='2021-01-01T00:00:00+00:00'


def test_checkpoint_precedes_work_so_concurrent_updates_are_not_skipped():
    c=instance();inside=[]
    def run(**kwargs):inside.append(datetime.now(timezone.utc));return {'errors':[]}
    c.run_consolidation=run;c.run_incremental_consolidation(affected_person_ids=[])
    assert datetime.fromisoformat(c._last_consolidation_timestamp)<=inside[0]


def test_updates_unknown_times_and_graph_participants_are_not_silently_omitted():
    c=instance();g=c.graph._g
    g.add_node('old',node_type='event',participants=['old-person'],metadata={'created_at':'2020-01-01T00:00:00Z'})
    g.add_node('updated',node_type='event',participants=['updated-person'],metadata={'created_at':'2020-01-01T00:00:00Z','updated_at':'2022-01-01T01:00:00+01:00'})
    g.add_node('unknown',node_type='event',metadata={});g.add_edge('linked-person','unknown',rel_type='PARTICIPATED_IN')
    assert c._find_affected_persons('2021-01-01T00:00:00Z')==['linked-person','updated-person']
    with pytest.raises(ValueError):c._find_affected_persons('invalid')
