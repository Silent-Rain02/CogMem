"""An identity address cannot be a substring of an unrelated word."""
import pytest
from cogmem.storage.graph_service import GraphService
from cogmem.storage.graph_store import GraphStore


def service(name, aliases=()):
    graph=GraphStore();graph._g.add_node('person_opaque',node_type='person',canonical_name=name,aliases=list(aliases))
    return GraphService(graph,None,None,use_vector_search=False)


@pytest.mark.parametrize('name,question',[('Ned','What happened during the journey?'),('Al','Who went walking yesterday?'),('Ann','Who organized the annual meeting?')])
def test_person_name_inside_another_word_does_not_create_a_lexical_identity_hit(name,question):
    assert service(name).find_anchors(question,top_k=1)==[]


def test_alias_inside_another_word_does_not_create_a_lexical_identity_hit():
    assert service('Nora',['Ann']).find_anchors('Who organized the annual meeting?',top_k=1)==[]


@pytest.mark.parametrize('name,question',[('Ned',"Where is Ned's bicycle?"),('Jo-Ann','Did Jo-Ann attend?'),('Iris Park','Where did Iris Park go?')])
def test_whole_names_and_possessives_still_match_at_existing_score(name,question):
    result=service(name).find_anchors(question,top_k=1)
    assert [(x.node_id,x.score) for x in result]==[('person_opaque',.9)]
