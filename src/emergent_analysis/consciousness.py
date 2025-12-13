"""Metrics for consciousness and awareness emergence.

This module implements various metrics for measuring consciousness-like
properties in neural systems, based on theories like Integrated Information
Theory (IIT) and Global Workspace Theory (GWT).
"""

from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    try:
        from ..brain_model import BrainModel
    except ImportError:
        from brain_model import BrainModel


class ConsciousnessMetrics:
    """Metrics for consciousness and awareness in neural systems.
    
    Implements measures inspired by:
    - Integrated Information Theory (IIT) - Φ (phi)
    - Global Workspace Theory (GWT) - global availability
    - Recurrent Processing Theory - feedback loops
    - Higher-Order Theories - meta-representation
    """
    
    def __init__(self, model: "BrainModel"):
        """Initialize consciousness metrics.
        
        Args:
            model: Brain model to analyze
        """
        self.model = model
        self.activity_history: List[np.ndarray] = []
        self.integration_history: List[float] = []
    
    def record_activity(self, activity: np.ndarray) -> None:
        """Record neural activity for analysis.
        
        Args:
            activity: Array of neural activities
        """
        self.activity_history.append(activity.copy())
    
    def compute_phi(
        self,
        activity: np.ndarray,
        partition_size: int = 2
    ) -> float:
        """Compute integrated information (Φ).
        
        This is a simplified approximation of IIT's Φ measure.
        
        Args:
            activity: Neural activity array
            partition_size: Size of partitions for integration test
            
        Returns:
            Phi value (integrated information)
        """
        n = len(activity)
        if n < partition_size * 2:
            return 0.0
        
        # Compute mutual information across partitions
        # This is a simplified version - full IIT is much more complex
        
        # Split into partitions
        mid = n // 2
        part1 = activity[:mid]
        part2 = activity[mid:]
        
        # Compute correlation (proxy for mutual information)
        if len(part1) > 0 and len(part2) > 0:
            # Pad to same length
            min_len = min(len(part1), len(part2))
            part1 = part1[:min_len]
            part2 = part2[:min_len]
            
            corr = np.corrcoef(part1, part2)[0, 1]
            phi = abs(corr) if not np.isnan(corr) else 0.0
        else:
            phi = 0.0
        
        return float(phi)
    
    def global_workspace_availability(
        self,
        activity: np.ndarray,
        threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Measure global workspace availability.
        
        Based on Global Workspace Theory - measures how much information
        is broadly available across the network.
        
        Args:
            activity: Neural activity array
            threshold: Threshold for "active" neurons
            
        Returns:
            Dictionary with GWT metrics
        """
        if len(activity) == 0:
            return {
                "global_availability": 0.0,
                "active_fraction": 0.0,
                "broadcast_strength": 0.0
            }
        
        # Fraction of highly active neurons
        active_mask = activity > (np.max(activity) * threshold)
        active_fraction = np.mean(active_mask)
        
        # Measure of broadcast (variance of activity)
        broadcast_strength = np.std(activity) / (np.mean(activity) + 1e-8)
        
        # Global availability: combination of activation breadth and strength
        global_availability = active_fraction * np.mean(activity[active_mask]) if np.any(active_mask) else 0.0
        
        return {
            "global_availability": float(global_availability),
            "active_fraction": float(active_fraction),
            "broadcast_strength": float(broadcast_strength)
        }
    
    def recurrent_processing_index(
        self,
        activity_matrix: np.ndarray
    ) -> float:
        """Compute recurrent processing index.
        
        Measures the degree of feedback/recurrent processing.
        
        Args:
            activity_matrix: Matrix of neural activities (neurons x time)
            
        Returns:
            Recurrence index
        """
        if activity_matrix.shape[0] < 2 or activity_matrix.shape[1] < 2:
            return 0.0
        
        # Compute autocorrelation across time
        autocorrs = []
        for neuron_activity in activity_matrix:
            if len(neuron_activity) > 1:
                # Compute autocorrelation at lag 1
                corr = np.corrcoef(neuron_activity[:-1], neuron_activity[1:])[0, 1]
                if not np.isnan(corr):
                    autocorrs.append(abs(corr))
        
        if not autocorrs:
            return 0.0
        
        # Average autocorrelation as recurrence measure
        return float(np.mean(autocorrs))
    
    def meta_representation_score(
        self,
        activity: np.ndarray,
        meta_neurons: Optional[np.ndarray] = None
    ) -> float:
        """Compute meta-representation score.
        
        Based on Higher-Order Theories - measures activity in neurons
        that represent other neural states.
        
        Args:
            activity: Full neural activity array
            meta_neurons: Indices of meta-cognitive neurons
            
        Returns:
            Meta-representation score
        """
        if meta_neurons is None:
            # Assume top 10% are meta-cognitive
            n_meta = max(1, len(activity) // 10)
            meta_neurons = np.argsort(activity)[-n_meta:]
        
        if len(meta_neurons) == 0:
            return 0.0
        
        # Activity of meta neurons
        meta_activity = activity[meta_neurons]
        
        # Compare to overall activity
        overall_mean = np.mean(activity)
        meta_mean = np.mean(meta_activity)
        
        # Score based on relative activation
        score = meta_mean / (overall_mean + 1e-8)
        
        return float(np.clip(score, 0.0, 2.0))
    
    def consciousness_state_estimate(self) -> Dict[str, Any]:
        """Estimate overall consciousness state.
        
        Combines multiple metrics to estimate consciousness level.
        
        Returns:
            Dictionary with consciousness estimates
        """
        if not self.activity_history:
            return {"consciousness_level": 0.0, "state": "offline"}
        
        # Use recent activity
        recent_activity = self.activity_history[-1]
        
        # Compute multiple metrics
        phi = self.compute_phi(recent_activity)
        gwt = self.global_workspace_availability(recent_activity)
        
        # Compute recurrence if we have history
        if len(self.activity_history) >= 2:
            activity_matrix = np.array(self.activity_history[-10:]).T
            recurrence = self.recurrent_processing_index(activity_matrix)
        else:
            recurrence = 0.0
        
        meta_score = self.meta_representation_score(recent_activity)
        
        # Combine into overall consciousness estimate
        consciousness_level = (
            0.3 * phi +
            0.3 * gwt["global_availability"] +
            0.2 * recurrence +
            0.2 * meta_score
        )
        
        # Categorize state
        if consciousness_level > 0.7:
            state = "high_awareness"
        elif consciousness_level > 0.4:
            state = "moderate_awareness"
        elif consciousness_level > 0.1:
            state = "minimal_awareness"
        else:
            state = "offline"
        
        return {
            "consciousness_level": float(consciousness_level),
            "state": state,
            "phi": phi,
            "global_workspace": gwt["global_availability"],
            "recurrence": recurrence,
            "meta_representation": meta_score
        }
    
    def track_consciousness_dynamics(self) -> Dict[str, Any]:
        """Track how consciousness metrics evolve over time.
        
        Returns:
            Dictionary with temporal dynamics
        """
        if len(self.activity_history) < 5:
            return {"error": "Insufficient history"}
        
        # Compute metrics over time
        phi_values = []
        gwa_values = []
        
        for activity in self.activity_history[-20:]:
            phi_values.append(self.compute_phi(activity))
            gwt = self.global_workspace_availability(activity)
            gwa_values.append(gwt["global_availability"])
        
        return {
            "phi_trajectory": phi_values,
            "gwa_trajectory": gwa_values,
            "phi_trend": "increasing" if phi_values[-1] > phi_values[0] else "decreasing",
            "gwa_trend": "increasing" if gwa_values[-1] > gwa_values[0] else "decreasing",
            "stability": float(1.0 - np.std(phi_values) / (np.mean(phi_values) + 1e-8))
        }
    
    def get_consciousness_report(self) -> Dict[str, Any]:
        """Generate comprehensive consciousness analysis report.
        
        Returns:
            Full consciousness analysis
        """
        state = self.consciousness_state_estimate()
        dynamics = self.track_consciousness_dynamics() if len(self.activity_history) >= 5 else {}
        
        report = {
            "current_state": state,
            "dynamics": dynamics,
            "history_length": len(self.activity_history),
            "assessment": self._generate_assessment(state)
        }
        
        return report
    
    def _generate_assessment(self, state: Dict[str, Any]) -> str:
        """Generate human-readable assessment.
        
        Args:
            state: Consciousness state dictionary
            
        Returns:
            Assessment string
        """
        level = state["consciousness_level"]
        state_name = state["state"]
        
        if level > 0.7:
            return f"High consciousness-like activity detected ({state_name}). Strong integration and global availability."
        elif level > 0.4:
            return f"Moderate consciousness-like patterns ({state_name}). Some integration present."
        elif level > 0.1:
            return f"Minimal awareness indicators ({state_name}). Limited integration."
        else:
            return f"Low/no consciousness-like activity ({state_name})."


__all__ = ["ConsciousnessMetrics"]
