"""Concept Manager for normalizing and managing concept IDs in the knowledge graph

This module provides the ConceptManager class which handles:
1. Normalizing concept names to standard format (lowercase with underscores)
2. Generating concept IDs with 'c_' prefix
3. Managing concept storage and deduplication
4. Resolving concept references during extraction
"""

import re
from typing import Dict, List, Optional, Set

class ConceptManager:
    """Manages concept normalization and ID generation for the knowledge graph

    The ConceptManager centralizes concept handling to ensure consistency
    across the knowledge graph. It normalizes concept names and generates
    standardized IDs.

    Examples:
        >>> manager = ConceptManager()
        >>> concept_id = manager.resolve("salsa dancing")
        >>> print(concept_id)  # "c_salsa_dancing"

        >>> concept_id2 = manager.resolve("Salsa Dancing")
        >>> print(concept_id2)  # "c_salsa_dancing" (same ID due to normalization)
    """

    def __init__(self):
        """Initialize the ConceptManager with an empty concept registry"""
        # Registry mapping concept IDs to their canonical forms and metadata
        self._concept_registry: Dict[str, Dict] = {}
        # Set of all concept IDs for quick lookup
        self._all_concepts: Set[str] = set()

    def normalize_name(self, concept_name: str) -> str:
        """Normalize a concept name to standard format

        Normalization steps:
        1. Convert to lowercase
        2. Replace spaces and hyphens with underscores
        3. Remove special characters (keep alphanumeric and underscores)
        4. Collapse multiple underscores into single underscores
        5. Remove leading/trailing underscores

        Args:
            concept_name: Raw concept name (e.g., "salsa dancing", "stress-relief")

        Returns:
            Normalized concept name (e.g., "salsa_dancing", "stress_relief")

        Examples:
            >>> normalize_name("Salsa Dancing")
            'salsa_dancing'
            >>> normalize_name("stress-relief technique")
            'stress_relief_technique'
            >>> normalize_name("  Coffee  ")
            'coffee'
        """
        if not concept_name or not isinstance(concept_name, str):
            raise ValueError(f"Concept name must be a non-empty string, got: {type(concept_name)}")

        # Convert to lowercase
        normalized = concept_name.lower().strip()

        # Replace spaces and hyphens with underscores
        normalized = normalized.replace(' ', '_').replace('-', '_')

        # Remove special characters (keep alphanumeric and underscores only)
        normalized = re.sub(r'[^a-z0-9_]', '', normalized)

        # Collapse multiple underscores into single underscores
        normalized = re.sub(r'_+', '_', normalized)

        # Remove leading/trailing underscores
        normalized = normalized.strip('_')

        if not normalized:
            raise ValueError(f"Concept name '{concept_name}' normalized to empty string")

        return normalized

    def generate_id(self, concept_name: str) -> str:
        """Generate a concept ID from a concept name

        Args:
            concept_name: Raw concept name

        Returns:
            Concept ID with 'c_' prefix (e.g., "c_salsa_dancing")

        Examples:
            >>> generate_id("salsa dancing")
            'c_salsa_dancing'
        """
        normalized = self.normalize_name(concept_name)
        return f"c_{normalized}"

    def resolve(self, concept_name: str, auto_register: bool = True) -> str:
        """Resolve a concept name to its standardized ID

        This is the main method used during extraction to convert
        raw concept mentions to standardized IDs.

        Args:
            concept_name: Raw concept name from text
            auto_register: Whether to automatically register new concepts

        Returns:
            Standardized concept ID

        Examples:
            >>> manager = ConceptManager()
            >>> manager.resolve("salsa dancing")
            'c_salsa_dancing'
            >>> manager.resolve("SALSA DANCING")  # Same ID due to normalization
            'c_salsa_dancing'
        """
        concept_id = self.generate_id(concept_name)

        if auto_register and concept_id not in self._concept_registry:
            self.register(concept_id, concept_name)

        return concept_id

    def register(self, concept_id: str, original_name: str, metadata: Optional[Dict] = None):
        """Register a concept in the registry

        Args:
            concept_id: Standardized concept ID
            original_name: Original concept name before normalization
            metadata: Optional metadata about the concept

        Examples:
            >>> manager = ConceptManager()
            >>> manager.register("c_salsa_dancing", "salsa dancing", {"category": "activity"})
        """
        if not concept_id.startswith('c_'):
            raise ValueError(f"Concept ID must start with 'c_', got: {concept_id}")

        self._concept_registry[concept_id] = {
            'id': concept_id,
            'original_name': original_name,
            'normalized_name': self.normalize_name(original_name),
            'metadata': metadata or {},
            'mention_count': 0
        }
        self._all_concepts.add(concept_id)

    def exists(self, concept_id: str) -> bool:
        """Check if a concept ID exists in the registry

        Args:
            concept_id: Concept ID to check

        Returns:
            True if the concept exists, False otherwise

        Examples:
            >>> manager = ConceptManager()
            >>> manager.resolve("coffee")
            'c_coffee'
            >>> manager.exists("c_coffee")
            True
            >>> manager.exists("c_tea")
            False
        """
        return concept_id in self._concept_registry

    def get_concept(self, concept_id: str) -> Optional[Dict]:
        """Get concept details from the registry

        Args:
            concept_id: Concept ID to retrieve

        Returns:
            Concept details dict or None if not found

        Examples:
            >>> manager = ConceptManager()
            >>> manager.register("c_coffee", "coffee", {"category": "beverage"})
            >>> concept = manager.get_concept("c_coffee")
            >>> print(concept['original_name'])
            'coffee'
        """
        return self._concept_registry.get(concept_id)

    def get_all_concepts(self) -> Dict[str, Dict]:
        """Get all registered concepts

        Returns:
            Dictionary mapping concept IDs to their details
        """
        return self._concept_registry.copy()

    def increment_mention(self, concept_id: str):
        """Increment the mention count for a concept

        Used during extraction to track how often each concept is mentioned.

        Args:
            concept_id: Concept ID to increment
        """
        if concept_id in self._concept_registry:
            self._concept_registry[concept_id]['mention_count'] += 1

    def get_top_concepts(self, limit: int = 10) -> List[Dict]:
        """Get the most frequently mentioned concepts

        Args:
            limit: Maximum number of concepts to return

        Returns:
            List of concept details sorted by mention count (descending)

        Examples:
            >>> manager = ConceptManager()
            >>> # ... register and increment some concepts ...
            >>> top_concepts = manager.get_top_concepts(5)
            >>> for concept in top_concepts:
            ...     print(f"{concept['id']}: {concept['mention_count']} mentions")
        """
        concepts = list(self._concept_registry.values())
        concepts.sort(key=lambda x: x['mention_count'], reverse=True)
        return concepts[:limit]

    def merge_concepts(self, source_id: str, target_id: str):
        """Merge two concepts (for deduplication)

        Args:
            source_id: Source concept ID to merge from
            target_id: Target concept ID to merge into

        Examples:
            >>> manager = ConceptManager()
            >>> manager.merge_concepts("c_salsa", "c_salsa_dancing")
        """
        if source_id not in self._concept_registry:
            return

        if target_id not in self._concept_registry:
            # Copy source to target
            self._concept_registry[target_id] = self._concept_registry[source_id].copy()
            self._concept_registry[target_id]['id'] = target_id
        else:
            # Merge mention counts
            self._concept_registry[target_id]['mention_count'] += self._concept_registry[source_id]['mention_count']

        # Remove source
        del self._concept_registry[source_id]
        self._all_concepts.discard(source_id)

    def clear(self):
        """Clear all registered concepts (for testing or reset)"""
        self._concept_registry.clear()
        self._all_concepts.clear()

# Global singleton instance
_default_manager = None

def get_default_manager() -> ConceptManager:
    """Get the default global ConceptManager instance

    Returns:
        The default ConceptManager singleton

    Examples:
        >>> manager = get_default_manager()
        >>> concept_id = manager.resolve("coffee")
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = ConceptManager()
    return _default_manager

def set_default_manager(manager: ConceptManager):
    """Set the default global ConceptManager instance

    Args:
        manager: ConceptManager instance to set as default

    Examples:
        >>> custom_manager = ConceptManager()
        >>> set_default_manager(custom_manager)
    """
    global _default_manager
    _default_manager = manager
