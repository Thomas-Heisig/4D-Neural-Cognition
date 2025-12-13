"""Reasoning engine for emergent cognitive capabilities.

This module implements reasoning mechanisms that emerge from the dynamics
of the 4D neural substrate, including pattern matching, inference, and
basic logical operations.
"""

from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    try:
        from ..brain_model import BrainModel
    except ImportError:
        from brain_model import BrainModel


class ReasoningEngine:
    """Emergent reasoning engine based on 4D neural dynamics.
    
    Implements reasoning capabilities that emerge from local neural rules:
    - Pattern completion and inference
    - Spatial reasoning
    - Temporal prediction
    - Simple logical operations
    """
    
    def __init__(
        self,
        model: "BrainModel",
        reasoning_area: str = "executive",
        threshold: float = 0.7
    ):
        """Initialize reasoning engine.
        
        Args:
            model: Brain model with 4D lattice
            reasoning_area: Brain area for reasoning operations
            threshold: Activation threshold for reasoning operations
        """
        self.model = model
        self.reasoning_area = reasoning_area
        self.threshold = threshold
        
        # Reasoning state
        self.active_patterns: Dict[str, np.ndarray] = {}
        self.inference_history: List[Dict[str, Any]] = []
    
    def encode_premise(
        self,
        premise_id: str,
        pattern: np.ndarray
    ) -> None:
        """Encode a premise into the reasoning system.
        
        Args:
            premise_id: Identifier for the premise
            pattern: Neural pattern representing the premise
        """
        self.active_patterns[premise_id] = pattern.copy()
    
    def pattern_completion(
        self,
        partial_pattern: np.ndarray,
        min_similarity: float = 0.5
    ) -> Tuple[Optional[np.ndarray], float]:
        """Complete a partial pattern using stored patterns.
        
        Args:
            partial_pattern: Partial input pattern
            min_similarity: Minimum similarity for pattern matching
            
        Returns:
            Tuple of (completed pattern, confidence score)
        """
        if not self.active_patterns:
            return None, 0.0
        
        best_match = None
        best_similarity = 0.0
        
        # Find best matching stored pattern
        for pattern_id, stored_pattern in self.active_patterns.items():
            # Ensure same size
            if len(partial_pattern) != len(stored_pattern):
                continue
            
            # Compute similarity (only on non-zero elements of partial)
            mask = partial_pattern != 0
            if np.sum(mask) == 0:
                continue
            
            similarity = np.corrcoef(
                partial_pattern[mask],
                stored_pattern[mask]
            )[0, 1]
            
            if not np.isnan(similarity) and similarity > best_similarity:
                best_similarity = similarity
                best_match = stored_pattern.copy()
        
        if best_similarity >= min_similarity and best_match is not None:
            return best_match, best_similarity
        
        return None, 0.0
    
    def spatial_inference(
        self,
        object_positions: List[Tuple[float, ...]],
        query_type: str = "nearest_neighbor"
    ) -> Dict[str, Any]:
        """Perform spatial reasoning on object positions.
        
        Args:
            object_positions: List of object positions in 4D space
            query_type: Type of spatial query
            
        Returns:
            Dictionary containing inference results
        """
        if not object_positions:
            return {"result": None, "confidence": 0.0}
        
        positions = np.array(object_positions)
        
        if query_type == "nearest_neighbor":
            # Find nearest neighbor pairs
            if len(positions) < 2:
                return {"result": None, "confidence": 0.0}
            
            distances = []
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    distances.append((i, j, dist))
            
            if distances:
                nearest = min(distances, key=lambda x: x[2])
                return {
                    "result": {
                        "object_1": nearest[0],
                        "object_2": nearest[1],
                        "distance": float(nearest[2])
                    },
                    "confidence": 0.9
                }
        
        elif query_type == "centroid":
            # Compute centroid of objects
            centroid = np.mean(positions, axis=0)
            return {
                "result": {
                    "centroid": centroid.tolist()
                },
                "confidence": 0.95
            }
        
        elif query_type == "bounding_box":
            # Compute bounding box
            min_coords = np.min(positions, axis=0)
            max_coords = np.max(positions, axis=0)
            return {
                "result": {
                    "min": min_coords.tolist(),
                    "max": max_coords.tolist(),
                    "volume": float(np.prod(max_coords - min_coords))
                },
                "confidence": 0.95
            }
        
        return {"result": None, "confidence": 0.0}
    
    def temporal_prediction(
        self,
        sequence: List[np.ndarray],
        steps_ahead: int = 1
    ) -> Tuple[Optional[np.ndarray], float]:
        """Predict next elements in a temporal sequence.
        
        Args:
            sequence: List of patterns in temporal order
            steps_ahead: Number of steps to predict
            
        Returns:
            Tuple of (predicted pattern, confidence)
        """
        if len(sequence) < 2:
            return None, 0.0
        
        # Simple linear extrapolation for now
        # More sophisticated prediction will use actual network dynamics
        sequence_array = np.array(sequence)
        
        # Compute velocity (change between last two states)
        velocity = sequence_array[-1] - sequence_array[-2]
        
        # Predict by extrapolating
        prediction = sequence_array[-1] + velocity * steps_ahead
        
        # Confidence based on sequence regularity
        if len(sequence) >= 3:
            # Check consistency of changes
            changes = np.diff(sequence_array, axis=0)
            consistency = 1.0 - np.std(changes) / (np.mean(np.abs(changes)) + 1e-8)
            confidence = np.clip(consistency, 0.0, 1.0)
        else:
            confidence = 0.5
        
        return prediction, float(confidence)
    
    def logical_and(
        self,
        pattern_a: np.ndarray,
        pattern_b: np.ndarray
    ) -> np.ndarray:
        """Compute logical AND of two neural patterns.
        
        Args:
            pattern_a: First pattern
            pattern_b: Second pattern
            
        Returns:
            Result pattern
        """
        # Neural AND: both patterns must be active
        return np.minimum(pattern_a, pattern_b)
    
    def logical_or(
        self,
        pattern_a: np.ndarray,
        pattern_b: np.ndarray
    ) -> np.ndarray:
        """Compute logical OR of two neural patterns.
        
        Args:
            pattern_a: First pattern
            pattern_b: Second pattern
            
        Returns:
            Result pattern
        """
        # Neural OR: either pattern active
        return np.maximum(pattern_a, pattern_b)
    
    def logical_not(
        self,
        pattern: np.ndarray,
        max_value: float = 1.0
    ) -> np.ndarray:
        """Compute logical NOT of a neural pattern.
        
        Args:
            pattern: Input pattern
            max_value: Maximum activation value
            
        Returns:
            Inverted pattern
        """
        # Neural NOT: invert activation
        return max_value - pattern
    
    def inference_step(
        self,
        premises: List[str],
        rule: str = "modus_ponens"
    ) -> Dict[str, Any]:
        """Perform one step of logical inference.
        
        Args:
            premises: List of premise IDs
            rule: Inference rule to apply
            
        Returns:
            Dictionary containing inference result
        """
        result = {
            "conclusion": None,
            "confidence": 0.0,
            "rule": rule
        }
        
        if rule == "modus_ponens":
            # If A and (A -> B), then B
            if len(premises) >= 2:
                # Simplified: AND of premises
                pattern_a = self.active_patterns.get(premises[0])
                pattern_b = self.active_patterns.get(premises[1])
                
                if pattern_a is not None and pattern_b is not None:
                    conclusion = self.logical_and(pattern_a, pattern_b)
                    result["conclusion"] = conclusion
                    result["confidence"] = 0.8
        
        elif rule == "disjunction":
            # A or B
            if len(premises) >= 2:
                pattern_a = self.active_patterns.get(premises[0])
                pattern_b = self.active_patterns.get(premises[1])
                
                if pattern_a is not None and pattern_b is not None:
                    conclusion = self.logical_or(pattern_a, pattern_b)
                    result["conclusion"] = conclusion
                    result["confidence"] = 0.9
        
        # Store in history
        self.inference_history.append(result.copy())
        
        return result
    
    def get_reasoning_statistics(self) -> Dict[str, Any]:
        """Get statistics about reasoning operations.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "active_patterns": len(self.active_patterns),
            "inference_history_length": len(self.inference_history),
            "average_confidence": np.mean([
                h["confidence"] for h in self.inference_history
                if h["confidence"] > 0
            ]) if self.inference_history else 0.0
        }


__all__ = ["ReasoningEngine"]
