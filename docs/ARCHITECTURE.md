# CogMem architecture

CogMem represents dialogue with a PEC²F graph rather than a flat collection of chunks.

```text
Dialogue turn
    │ extraction + provenance
    ▼
Person ── participates in ── Event ── concerns ── Concept
  │                              │
  │ makes / holds                └── supports ── Fact
  ▼
Claim (source-attributed statement)
    │
    └── targets a Person, Event, or Concept
```

## Memory generation and evolution

1. **Ingestion** converts each document into typed nodes and edges. Nodes retain source-document and timestamp information so later answers can be grounded in the original dialogue.
2. **Consolidation** groups related episodic events, creates higher-level facts, deduplicates close concepts/facts, and reconciles claim lifecycles. Supporting episodes remain linked to synthesized facts.
3. **Persistence** stores the graph, typed FAISS indexes, chunks, and evidence metadata in an application-owned workspace (default: `.cogmem/`).

## Query-time reconstruction

The search controller classifies the question and selects bounded graph operators:

- **Anchoring:** locate relevant people, concepts, events, facts, or claims.
- **Traversal:** follow typed relations through a limited graph neighborhood.
- **Intersection:** find shared entities or events for multi-hop questions.
- **Evidence grounding:** recover source dialogue before producing an answer.

The graph schema makes claim provenance explicit: a statement such as “Bob believes the project is risky” is represented as a Claim rather than silently becoming an unattributed Fact. This distinction is central to knowledge-update and source-attribution queries.
