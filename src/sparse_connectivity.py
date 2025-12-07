"""Sparse connectivity matrix for efficient synapse storage and lookup.

This module provides a sparse matrix representation for synaptic connections
using Compressed Sparse Row (CSR) format, which is more efficient than a list
for large neural networks.
"""

from typing import List, Tuple, Optional
import numpy as np


class SparseConnectivityMatrix:
    """Sparse matrix representation for synaptic connections.
    
    Uses CSR (Compressed Sparse Row) format for memory-efficient storage
    and fast row-wise access patterns. This is optimal for neural network
    simulations where we frequently need to access all incoming or outgoing
    synapses for a neuron.
    
    Memory complexity: O(num_synapses) vs O(num_neurons^2) for dense matrix
    Access complexity: O(1) for row start, O(k) for k connections per neuron
    """
    
    def __init__(self, max_neuron_id: int = 1000):
        """Initialize sparse connectivity matrix.
        
        Args:
            max_neuron_id: Maximum neuron ID to support (can grow dynamically)
        """
        self._max_neuron_id = max_neuron_id
        
        # CSR format arrays
        self._pre_ids: List[int] = []  # Pre-synaptic neuron IDs
        self._post_ids: List[int] = []  # Post-synaptic neuron IDs
        self._weights: List[float] = []  # Synaptic weights
        self._delays: List[int] = []  # Synaptic delays
        self._plasticity_tags: List[float] = []  # Plasticity tags
        self._synapse_types: List[str] = []  # Synapse types
        
        # Index structures for fast lookup
        # Maps post_id -> list of synapse indices (incoming synapses)
        self._post_index: dict[int, List[int]] = {}
        # Maps pre_id -> list of synapse indices (outgoing synapses)
        self._pre_index: dict[int, List[int]] = {}
        
    def add_synapse(
        self,
        pre_id: int,
        post_id: int,
        weight: float = 0.1,
        delay: int = 1,
        plasticity_tag: float = 0.0,
        synapse_type: str = "excitatory"
    ) -> int:
        """Add a synapse to the connectivity matrix.
        
        Args:
            pre_id: Pre-synaptic neuron ID
            post_id: Post-synaptic neuron ID
            weight: Synaptic weight
            delay: Synaptic delay in time steps
            plasticity_tag: Tag for plasticity mechanisms
            synapse_type: Type of synapse ("excitatory" or "inhibitory")
            
        Returns:
            Index of the added synapse
        """
        synapse_idx = len(self._pre_ids)
        
        self._pre_ids.append(pre_id)
        self._post_ids.append(post_id)
        self._weights.append(weight)
        self._delays.append(delay)
        self._plasticity_tags.append(plasticity_tag)
        self._synapse_types.append(synapse_type)
        
        # Update indices
        if post_id not in self._post_index:
            self._post_index[post_id] = []
        self._post_index[post_id].append(synapse_idx)
        
        if pre_id not in self._pre_index:
            self._pre_index[pre_id] = []
        self._pre_index[pre_id].append(synapse_idx)
        
        return synapse_idx
    
    def get_incoming_synapses(self, post_id: int) -> List[Tuple[int, int, float, int, float, str]]:
        """Get all incoming synapses for a post-synaptic neuron.
        
        Args:
            post_id: Post-synaptic neuron ID
            
        Returns:
            List of tuples (synapse_idx, pre_id, weight, delay, plasticity_tag, synapse_type)
        """
        if post_id not in self._post_index:
            return []
        
        result = []
        for idx in self._post_index[post_id]:
            result.append((
                idx,
                self._pre_ids[idx],
                self._weights[idx],
                self._delays[idx],
                self._plasticity_tags[idx],
                self._synapse_types[idx]
            ))
        return result
    
    def get_outgoing_synapses(self, pre_id: int) -> List[Tuple[int, int, float, int, float, str]]:
        """Get all outgoing synapses for a pre-synaptic neuron.
        
        Args:
            pre_id: Pre-synaptic neuron ID
            
        Returns:
            List of tuples (synapse_idx, post_id, weight, delay, plasticity_tag, synapse_type)
        """
        if pre_id not in self._pre_index:
            return []
        
        result = []
        for idx in self._pre_index[pre_id]:
            result.append((
                idx,
                self._post_ids[idx],
                self._weights[idx],
                self._delays[idx],
                self._plasticity_tags[idx],
                self._synapse_types[idx]
            ))
        return result
    
    def get_synapse(self, synapse_idx: int) -> Optional[Tuple[int, int, float, int, float, str]]:
        """Get synapse by index.
        
        Args:
            synapse_idx: Index of the synapse
            
        Returns:
            Tuple (pre_id, post_id, weight, delay, plasticity_tag, synapse_type) or None
        """
        if 0 <= synapse_idx < len(self._pre_ids):
            return (
                self._pre_ids[synapse_idx],
                self._post_ids[synapse_idx],
                self._weights[synapse_idx],
                self._delays[synapse_idx],
                self._plasticity_tags[synapse_idx],
                self._synapse_types[synapse_idx]
            )
        return None
    
    def update_weight(self, synapse_idx: int, weight: float) -> None:
        """Update synaptic weight.
        
        Args:
            synapse_idx: Index of the synapse
            weight: New weight value
        """
        if 0 <= synapse_idx < len(self._weights):
            self._weights[synapse_idx] = weight
    
    def update_plasticity_tag(self, synapse_idx: int, tag: float) -> None:
        """Update plasticity tag.
        
        Args:
            synapse_idx: Index of the synapse
            tag: New plasticity tag value
        """
        if 0 <= synapse_idx < len(self._plasticity_tags):
            self._plasticity_tags[synapse_idx] = tag
    
    def remove_synapses_for_neuron(self, neuron_id: int) -> None:
        """Remove all synapses connected to a neuron.
        
        This is more efficient than removing individual synapses as it
        batches the removals and rebuilds indices once.
        
        Args:
            neuron_id: Neuron ID to remove synapses for
        """
        # Find indices to keep (those not connected to neuron_id)
        indices_to_keep = []
        for idx in range(len(self._pre_ids)):
            if self._pre_ids[idx] != neuron_id and self._post_ids[idx] != neuron_id:
                indices_to_keep.append(idx)
        
        # Rebuild arrays with only kept synapses
        self._pre_ids = [self._pre_ids[i] for i in indices_to_keep]
        self._post_ids = [self._post_ids[i] for i in indices_to_keep]
        self._weights = [self._weights[i] for i in indices_to_keep]
        self._delays = [self._delays[i] for i in indices_to_keep]
        self._plasticity_tags = [self._plasticity_tags[i] for i in indices_to_keep]
        self._synapse_types = [self._synapse_types[i] for i in indices_to_keep]
        
        # Rebuild indices
        self._rebuild_indices()
    
    def _rebuild_indices(self) -> None:
        """Rebuild the index structures from scratch."""
        self._post_index = {}
        self._pre_index = {}
        
        for idx in range(len(self._pre_ids)):
            pre_id = self._pre_ids[idx]
            post_id = self._post_ids[idx]
            
            if post_id not in self._post_index:
                self._post_index[post_id] = []
            self._post_index[post_id].append(idx)
            
            if pre_id not in self._pre_index:
                self._pre_index[pre_id] = []
            self._pre_index[pre_id].append(idx)
    
    def num_synapses(self) -> int:
        """Get total number of synapses."""
        return len(self._pre_ids)
    
    def to_list(self) -> List[dict]:
        """Convert to list of synapse dictionaries for compatibility.
        
        Returns:
            List of synapse dictionaries with keys: pre_id, post_id, weight,
            delay, plasticity_tag, synapse_type
        """
        result = []
        for idx in range(len(self._pre_ids)):
            result.append({
                "pre_id": self._pre_ids[idx],
                "post_id": self._post_ids[idx],
                "weight": self._weights[idx],
                "delay": self._delays[idx],
                "plasticity_tag": self._plasticity_tags[idx],
                "synapse_type": self._synapse_types[idx]
            })
        return result
    
    @classmethod
    def from_list(cls, synapses: List[dict], max_neuron_id: int = 1000) -> "SparseConnectivityMatrix":
        """Create from list of synapse dictionaries.
        
        Args:
            synapses: List of synapse dictionaries
            max_neuron_id: Maximum neuron ID to support
            
        Returns:
            New SparseConnectivityMatrix instance
        """
        matrix = cls(max_neuron_id)
        for syn in synapses:
            matrix.add_synapse(
                pre_id=syn["pre_id"],
                post_id=syn["post_id"],
                weight=syn.get("weight", 0.1),
                delay=syn.get("delay", 1),
                plasticity_tag=syn.get("plasticity_tag", 0.0),
                synapse_type=syn.get("synapse_type", "excitatory")
            )
        return matrix
