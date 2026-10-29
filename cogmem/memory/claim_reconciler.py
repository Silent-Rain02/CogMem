"""Claim coexistence and contradiction diagnostics without automatic supersession."""

from __future__ import annotations

from datetime import datetime
from typing import List, Dict, Set, Tuple, Optional, Literal
from collections import defaultdict

from ..storage.graph_store import GraphStore
from ..storage.schema import Claim, Relation, NodeTypeEnum, Metadata

class ClaimReconciler:
    """Preserve claims until same-context replacement is established by evidence."""
    
    # Sentiment pairs that indicate contradiction
    CONTRADICTING_SENTIMENTS = {
        ('positive', 'negative'),
        ('negative', 'positive'),
    }
    
    # Keywords that indicate opposing views
    POSITIVE_KEYWORDS = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'best', 'love', 'like', 'talented', 'smart', 'helpful'}
    NEGATIVE_KEYWORDS = {'bad', 'terrible', 'awful', 'worst', 'hate', 'dislike', 'lazy', 'stupid', 'useless', 'incompetent'}
    
    def __init__(self, graph: GraphStore):
        """Initialize ClaimReconciler
        
        Args:
            graph: Graph store instance
        """
        self.graph = graph
        
        # Statistics
        self._stats = {
            "claims_processed": 0,
            "claim_groups": 0,
            "contradictions_found": 0,
            "supersedes_edges_created": 0,
            "claims_archived": 0
        }
    
    def _get_all_claims(self) -> List[Tuple[str, Dict]]:
        """Get all Claim nodes from the graph"""
        claims = []
        for node_id, node_data in self.graph._g.nodes(data=True):
            if node_data and node_data.get('node_type') == NodeTypeEnum.CLAIM.value:
                claims.append((node_id, node_data))
        return claims
    
    def _get_claim_key(self, claim_data: Dict) -> str:
        """Generate a key from (source_id, target_id) pair"""
        source = claim_data.get('source_id', '')
        target = claim_data.get('target_id', '')
        return f"{source}|{target}"
    
    def _parse_timestamp(self, ts: Optional[str]) -> Optional[datetime]:
        """Parse timestamp string to datetime"""
        if not ts:
            return None
        
        try:
            if 'T' in ts:
                return datetime.fromisoformat(ts.replace('Z', '+00:00'))
            else:
                return datetime.strptime(ts, "%Y-%m-%d")
        except:
            return None
    
    def _get_claim_timestamp(self, claim_data: Dict) -> Optional[datetime]:
        """Get timestamp from claim data (session_time or metadata.created_at)"""
        # Try session_time first
        session_time = claim_data.get('session_time')
        if session_time:
            ts = self._parse_timestamp(session_time)
            if ts:
                return ts
        
        # Try metadata.created_at
        metadata = claim_data.get('metadata', {})
        if isinstance(metadata, dict):
            created_at = metadata.get('created_at')
            if created_at:
                ts = self._parse_timestamp(created_at)
                if ts:
                    return ts
        
        return None
    
    def _extract_sentiment(self, claim_data: Dict) -> str:
        """Extract or infer sentiment from claim data"""
        # Use explicit sentiment if available
        sentiment = claim_data.get('sentiment')
        if sentiment:
            return sentiment.lower()
        
        # Infer from content
        content = claim_data.get('content', '').lower()
        
        pos_count = sum(1 for kw in self.POSITIVE_KEYWORDS if kw in content)
        neg_count = sum(1 for kw in self.NEGATIVE_KEYWORDS if kw in content)
        
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _are_contradicting(
        self,
        claim1: Tuple[str, Dict],
        claim2: Tuple[str, Dict]
    ) -> bool:
        """Check if two claims are contradicting"""
        _, data1 = claim1
        _, data2 = claim2
        
        # Check sentiment contradiction
        sentiment1 = self._extract_sentiment(data1)
        sentiment2 = self._extract_sentiment(data2)
        
        if (sentiment1, sentiment2) in self.CONTRADICTING_SENTIMENTS:
            return True
        
        # Check content contradiction
        content1 = data1.get('content', '').lower()
        content2 = data2.get('content', '').lower()
        
        # Simple negation check
        if 'not' in content1 and 'not' not in content2:
            # Compare without 'not'
            cleaned1 = content1.replace('not ', '').replace("n't ", '')
            if cleaned1 in content2 or content2 in cleaned1:
                return True
        
        if 'not' in content2 and 'not' not in content1:
            cleaned2 = content2.replace('not ', '').replace("n't ", '')
            if cleaned2 in content1 or content1 in cleaned2:
                return True
        
        return False
    
    def group_claims_by_source_target(self) -> Dict[str, List[Tuple[str, Dict]]]:
        """Group claims by (source_id, target_id) pairs
        
        Returns:
            Dictionary mapping keys to lists of claims
        """
        claims = self._get_all_claims()
        self._stats["claims_processed"] = len(claims)
        
        if not claims:
            return {}
        
        print(f"[ClaimReconciler] Processing {len(claims)} claims...")
        
        # Group by (source, target)
        groups: Dict[str, List[Tuple[str, Dict]]] = defaultdict(list)
        
        for claim_id, claim_data in claims:
            key = self._get_claim_key(claim_data)
            groups[key].append((claim_id, claim_data))
        
        # Filter to groups with multiple claims
        multi_claim_groups = {
            key: group for key, group in groups.items()
            if len(group) > 1
        }
        
        self._stats["claim_groups"] = len(multi_claim_groups)
        print(f"[ClaimReconciler] Found {len(multi_claim_groups)} groups with multiple claims")
        
        return multi_claim_groups
    
    def find_contradictions(self) -> List[Tuple[str, str, str]]:
        """Find contradicting claim pairs
        
        Returns:
            List of (claim1_id, claim2_id, contradiction_type) tuples
        """
        groups = self.group_claims_by_source_target()
        contradictions: List[Tuple[str, str, str]] = []
        
        for key, claims in groups.items():
            n = len(claims)
            for i in range(n):
                for j in range(i + 1, n):
                    claim1, claim2 = claims[i], claims[j]
                    
                    if self._are_contradicting(claim1, claim2):
                        id1, data1 = claim1
                        id2, data2 = claim2
                        
                        # Determine contradiction type
                        sent1 = self._extract_sentiment(data1)
                        sent2 = self._extract_sentiment(data2)
                        
                        if (sent1, sent2) in self.CONTRADICTING_SENTIMENTS:
                            contra_type = 'sentiment_reversal'
                        else:
                            contra_type = 'content_conflict'
                        
                        contradictions.append((id1, id2, contra_type))
        
        self._stats["contradictions_found"] = len(contradictions)
        print(f"[ClaimReconciler] Found {len(contradictions)} contradicting claim pairs")
        
        return contradictions
    
    def _sort_claims_by_time(
        self,
        claims: List[Tuple[str, Dict]]
    ) -> List[Tuple[str, Dict]]:
        """Sort claims by timestamp (oldest first)"""
        def get_sort_key(claim: Tuple[str, Dict]) -> float:
            _, data = claim
            ts = self._get_claim_timestamp(data)
            if ts:
                return ts.timestamp()
            return 0.0
        
        return sorted(claims, key=get_sort_key)
    
    def establish_temporal_chain(self, groups, create_supersedes=True):
        """Temporal ordering alone cannot establish a same-context update."""
        if not create_supersedes:
            return {}
        raise ValueError("Claim supersession requires source-backed context validation")

    def reconcile_group(self, claims, strategy='keep_all'):
        """Coexisting claims are preserved; automatic supersession is retired."""
        if strategy != 'keep_all':
            raise ValueError("Claim supersession requires source-backed context validation")
        return claims[0][0] if len(claims) == 1 else None

    def run(
        self,
        create_supersedes: bool = True,
        strategy: Literal['newest_wins', 'keep_all', 'merge'] = 'keep_all'
    ) -> Dict[str, str]:
        """Run the full claim reconciliation process
        
        Args:
            create_supersedes: Whether to create SUPERSEDES edges
            strategy: Reconciliation strategy
        
        Returns:
            Mapping from superseded claim IDs to superseding claim IDs
        """
        print(f"[ClaimReconciler] Starting claim reconciliation")
        
        # Group claims
        groups = self.group_claims_by_source_target()
        
        if strategy == 'keep_all':
            supersede_map = {}
        elif strategy == 'newest_wins':
            supersede_map = self.establish_temporal_chain(groups, create_supersedes=create_supersedes)
        else:
            raise ValueError("Claim merging needs explicit source validation")
        
        # Find and log contradictions
        contradictions = self.find_contradictions()
        if contradictions:
            print(f"[ClaimReconciler] Warning: Found {len(contradictions)} contradicting claims")
            for id1, id2, contra_type in contradictions[:5]:
                print(f"  - {id1} vs {id2}: {contra_type}")
        
        print(f"[ClaimReconciler] Reconciliation complete:")
        print(f"  - Claims processed: {self._stats['claims_processed']}")
        print(f"  - Claim groups: {self._stats['claim_groups']}")
        print(f"  - Contradictions found: {self._stats['contradictions_found']}")
        print(f"  - SUPERSEDES edges created: {self._stats['supersedes_edges_created']}")
        print(f"  - Claims archived: {self._stats['claims_archived']}")
        
        return supersede_map
    
    def get_stats(self) -> Dict[str, int]:
        """Get reconciliation statistics"""
        return self._stats.copy()
    
    def get_claim_history(self, claim_id: str) -> List[str]:
        """Get the history of a claim (all superseded versions)
        
        Args:
            claim_id: ID of the claim to trace
        
        Returns:
            List of claim IDs in chronological order (oldest first)
        """
        history = [claim_id]
        
        # Traverse SUPERSEDES edges backwards
        current_id = claim_id
        while current_id in self.graph._g:
            # Find superseded claims (incoming SUPERSEDES edges)
            superseded = []
            for src, dst, data in self.graph._g.in_edges(current_id, data=True):
                if data.get('rel_type') == 'SUPERSEDES':
                    superseded.append(src)
            
            if not superseded:
                break
            
            # Follow the chain
            current_id = superseded[0]  # Assuming linear chain
            history.insert(0, current_id)
        
        return history
    
    def get_latest_claim(self, source_id: str, target_id: str) -> Optional[str]:
        """Get the latest (non-archived) claim from source about target
        
        Args:
            source_id: ID of the claim source
            target_id: ID of the claim target
        
        Returns:
            ID of the latest claim, or None if not found
        """
        claims = self._get_all_claims()
        
        matching = []
        for claim_id, claim_data in claims:
            if (claim_data.get('source_id') == source_id and 
                claim_data.get('target_id') == target_id and
                not claim_data.get('is_archived', False)):
                matching.append((claim_id, claim_data))
        
        if not matching:
            return None
        
        # Return the one with latest timestamp
        sorted_claims = self._sort_claims_by_time(matching)
        return sorted_claims[-1][0]
