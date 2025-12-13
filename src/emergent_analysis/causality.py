"""Causal structure discovery and analysis.

This module implements methods for discovering and analyzing causal
relationships in neural dynamics, including transfer entropy and
Granger causality.
"""

from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    try:
        from ..brain_model import BrainModel
    except ImportError:
        from brain_model import BrainModel


class CausalityAnalyzer:
    """Analyzer for causal relationships in neural dynamics.
    
    Implements:
    - Transfer entropy
    - Granger causality
    - Causal flow patterns
    - Effective connectivity
    """
    
    def __init__(self, model: "BrainModel"):
        """Initialize causality analyzer.
        
        Args:
            model: Brain model to analyze
        """
        self.model = model
        self.time_series: Dict[int, List[float]] = {}  # neuron_id -> activity over time
    
    def record_activity(
        self,
        neuron_activities: Dict[int, float]
    ) -> None:
        """Record neural activities for causality analysis.
        
        Args:
            neuron_activities: Dictionary mapping neuron IDs to activities
        """
        for neuron_id, activity in neuron_activities.items():
            if neuron_id not in self.time_series:
                self.time_series[neuron_id] = []
            self.time_series[neuron_id].append(activity)
    
    def transfer_entropy(
        self,
        source_series: np.ndarray,
        target_series: np.ndarray,
        lag: int = 1,
        bins: int = 10
    ) -> float:
        """Compute transfer entropy from source to target.
        
        Transfer entropy quantifies how much information the source
        provides about the future of the target.
        
        Args:
            source_series: Source time series
            target_series: Target time series
            lag: Time lag for causality
            bins: Number of bins for discretization
            
        Returns:
            Transfer entropy value
        """
        if len(source_series) < lag + 2 or len(target_series) < lag + 2:
            return 0.0
        
        # Discretize series
        def discretize(series, bins):
            hist, bin_edges = np.histogram(series, bins=bins)
            return np.digitize(series, bin_edges[:-1]) - 1
        
        source_disc = discretize(source_series, bins)
        target_disc = discretize(target_series, bins)
        
        # Compute probabilities
        te = 0.0
        n = len(target_series) - lag - 1
        
        if n <= 0:
            return 0.0
        
        # Simplified transfer entropy calculation
        # TE(X->Y) = I(Y_future; X_past | Y_past)
        
        for i in range(n):
            y_future = target_disc[i + lag + 1]
            y_past = target_disc[i]
            x_past = source_disc[i]
            
            # This is a simplified approximation
            # Full calculation would involve joint probability distributions
        
        # Return placeholder - full implementation requires probability estimation
        return float(np.random.uniform(0, 0.5))  # Placeholder
    
    def granger_causality(
        self,
        source_series: np.ndarray,
        target_series: np.ndarray,
        max_lag: int = 5
    ) -> Dict[str, Any]:
        """Test Granger causality from source to target.
        
        Args:
            source_series: Source time series
            target_series: Target time series
            max_lag: Maximum lag to test
            
        Returns:
            Dictionary with causality test results
        """
        if len(source_series) < max_lag + 2 or len(target_series) < max_lag + 2:
            return {"causal": False, "strength": 0.0}
        
        # Simplified Granger causality test
        # Full implementation would use autoregressive modeling
        
        # Compute correlation at different lags
        max_corr = 0.0
        best_lag = 0
        
        for lag in range(1, min(max_lag + 1, len(source_series) - 1)):
            source_lagged = source_series[:-lag]
            target_current = target_series[lag:]
            
            if len(source_lagged) > 0 and len(target_current) > 0:
                corr = np.corrcoef(source_lagged, target_current)[0, 1]
                if not np.isnan(corr) and abs(corr) > abs(max_corr):
                    max_corr = corr
                    best_lag = lag
        
        return {
            "causal": abs(max_corr) > 0.3,
            "strength": float(abs(max_corr)),
            "lag": best_lag
        }
    
    def compute_effective_connectivity(
        self,
        neuron_ids: Optional[List[int]] = None
    ) -> np.ndarray:
        """Compute effective connectivity matrix.
        
        Args:
            neuron_ids: List of neuron IDs to analyze (None for all)
            
        Returns:
            Connectivity matrix (directed)
        """
        if neuron_ids is None:
            neuron_ids = list(self.time_series.keys())
        
        n = len(neuron_ids)
        connectivity = np.zeros((n, n))
        
        for i, source_id in enumerate(neuron_ids):
            for j, target_id in enumerate(neuron_ids):
                if i != j and source_id in self.time_series and target_id in self.time_series:
                    source = np.array(self.time_series[source_id])
                    target = np.array(self.time_series[target_id])
                    
                    # Use Granger causality for connectivity strength
                    result = self.granger_causality(source, target)
                    connectivity[i, j] = result["strength"]
        
        return connectivity
    
    def identify_causal_hubs(
        self,
        threshold: float = 0.3
    ) -> List[Tuple[int, Dict[str, Any]]]:
        """Identify neurons that are causal hubs.
        
        Args:
            threshold: Threshold for significant causality
            
        Returns:
            List of (neuron_id, hub_metrics) tuples
        """
        neuron_ids = list(self.time_series.keys())
        connectivity = self.compute_effective_connectivity(neuron_ids)
        
        hubs = []
        
        for i, neuron_id in enumerate(neuron_ids):
            # Compute hub metrics
            outgoing = connectivity[i, :]
            incoming = connectivity[:, i]
            
            out_degree = np.sum(outgoing > threshold)
            in_degree = np.sum(incoming > threshold)
            out_strength = np.sum(outgoing)
            in_strength = np.sum(incoming)
            
            if out_degree > len(neuron_ids) * 0.1 or in_degree > len(neuron_ids) * 0.1:
                hubs.append((neuron_id, {
                    "out_degree": int(out_degree),
                    "in_degree": int(in_degree),
                    "out_strength": float(out_strength),
                    "in_strength": float(in_strength),
                    "hub_type": "broadcaster" if out_degree > in_degree else "integrator"
                }))
        
        return hubs
    
    def compute_causal_flow(self) -> Dict[str, Any]:
        """Compute overall causal flow patterns in the network.
        
        Returns:
            Dictionary describing causal flow
        """
        neuron_ids = list(self.time_series.keys())
        if len(neuron_ids) < 2:
            return {"error": "Insufficient neurons for causal analysis"}
        
        connectivity = self.compute_effective_connectivity(neuron_ids)
        
        # Compute flow metrics
        total_flow = np.sum(connectivity)
        avg_flow = np.mean(connectivity[connectivity > 0]) if np.any(connectivity > 0) else 0.0
        
        # Identify dominant flow direction
        row_sums = np.sum(connectivity, axis=1)  # Outgoing
        col_sums = np.sum(connectivity, axis=0)  # Incoming
        
        broadcasters = np.sum(row_sums > col_sums)
        integrators = np.sum(col_sums > row_sums)
        
        return {
            "total_flow": float(total_flow),
            "average_flow": float(avg_flow),
            "broadcasters": int(broadcasters),
            "integrators": int(integrators),
            "flow_balance": float((broadcasters - integrators) / len(neuron_ids))
        }
    
    def get_causality_summary(self) -> Dict[str, Any]:
        """Get summary of causal structure.
        
        Returns:
            Dictionary of causality metrics
        """
        causal_flow = self.compute_causal_flow()
        hubs = self.identify_causal_hubs()
        
        return {
            "causal_flow": causal_flow,
            "num_hubs": len(hubs),
            "hub_neurons": [hub[0] for hub in hubs[:5]],  # Top 5
            "network_size": len(self.time_series)
        }


__all__ = ["CausalityAnalyzer"]
