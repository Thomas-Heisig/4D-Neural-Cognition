"""Algorithmic complexity measures for neural dynamics.

This module implements various complexity measures to quantify the
information content and computational sophistication of neural activity.
"""

from typing import Dict, List, Optional, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    try:
        from ..brain_model import BrainModel
    except ImportError:
        from brain_model import BrainModel


class ComplexityAnalyzer:
    """Analyzer for algorithmic complexity of neural dynamics.
    
    Implements various complexity measures:
    - Lempel-Ziv complexity
    - Entropy and information content
    - Neural complexity (integration/differentiation)
    - Multiscale entropy
    """
    
    def __init__(self, model: "BrainModel"):
        """Initialize complexity analyzer.
        
        Args:
            model: Brain model to analyze
        """
        self.model = model
        self.activity_history: List[np.ndarray] = []
    
    def record_activity(self, activity: np.ndarray) -> None:
        """Record neural activity for analysis.
        
        Args:
            activity: Array of neural activities
        """
        self.activity_history.append(activity.copy())
    
    def shannon_entropy(self, activity: np.ndarray, bins: int = 10) -> float:
        """Compute Shannon entropy of activity distribution.
        
        Args:
            activity: Neural activity array
            bins: Number of bins for discretization
            
        Returns:
            Shannon entropy value
        """
        if len(activity) == 0:
            return 0.0
        
        # Discretize activity
        hist, _ = np.histogram(activity, bins=bins)
        hist = hist / np.sum(hist)  # Normalize to probabilities
        
        # Compute entropy
        entropy = 0.0
        for p in hist:
            if p > 0:
                entropy -= p * np.log2(p)
        
        return float(entropy)
    
    def lempel_ziv_complexity(self, binary_sequence: np.ndarray) -> float:
        """Compute Lempel-Ziv complexity of a binary sequence.
        
        Args:
            binary_sequence: Binary array (0s and 1s)
            
        Returns:
            Normalized LZ complexity
        """
        n = len(binary_sequence)
        if n == 0:
            return 0.0
        
        # Convert to string for easier processing
        seq = ''.join(map(str, binary_sequence.astype(int)))
        
        complexity = 1
        prefix_len = 1
        i = 0
        
        while i + prefix_len <= n:
            substring = seq[i:i + prefix_len]
            prefix = seq[:i + prefix_len]
            
            # Check if substring appears in prefix
            if substring in prefix[:-prefix_len] or i + prefix_len == n:
                prefix_len += 1
            else:
                complexity += 1
                i += prefix_len
                prefix_len = 1
        
        # Normalize by theoretical maximum
        max_complexity = n / np.log2(n) if n > 1 else 1
        return complexity / max_complexity
    
    def neural_complexity(
        self,
        activity_matrix: np.ndarray
    ) -> Dict[str, float]:
        """Compute neural complexity (Tononi et al.).
        
        Measures the balance between integration and differentiation.
        
        Args:
            activity_matrix: Matrix of neural activities (neurons x time)
            
        Returns:
            Dictionary with complexity measures
        """
        if activity_matrix.shape[0] < 2 or activity_matrix.shape[1] < 2:
            return {"complexity": 0.0, "integration": 0.0, "differentiation": 0.0}
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(activity_matrix)
        corr_matrix = np.nan_to_num(corr_matrix)
        
        # Integration: average correlation
        integration = np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))
        
        # Differentiation: variance of correlations
        differentiation = np.std(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
        
        # Neural complexity: product of integration and differentiation
        complexity = integration * differentiation
        
        return {
            "complexity": float(complexity),
            "integration": float(integration),
            "differentiation": float(differentiation)
        }
    
    def multiscale_entropy(
        self,
        time_series: np.ndarray,
        max_scale: int = 5,
        bins: int = 10
    ) -> Dict[int, float]:
        """Compute multiscale entropy.
        
        Args:
            time_series: Time series of neural activity
            max_scale: Maximum scale to analyze
            bins: Number of bins for entropy calculation
            
        Returns:
            Dictionary mapping scales to entropy values
        """
        entropies = {}
        
        for scale in range(1, max_scale + 1):
            # Coarse-grain the time series
            n_points = len(time_series) // scale
            coarse_grained = np.zeros(n_points)
            
            for i in range(n_points):
                coarse_grained[i] = np.mean(
                    time_series[i * scale:(i + 1) * scale]
                )
            
            # Compute entropy at this scale
            entropy = self.shannon_entropy(coarse_grained, bins)
            entropies[scale] = entropy
        
        return entropies
    
    def compute_activity_complexity(self) -> Dict[str, Any]:
        """Compute complexity measures for recorded activity history.
        
        Returns:
            Dictionary of complexity metrics
        """
        if not self.activity_history:
            return {"error": "No activity recorded"}
        
        # Stack activity history into matrix
        activity_matrix = np.array(self.activity_history).T  # neurons x time
        
        # Compute various complexity measures
        results = {
            "shannon_entropy": {},
            "lempel_ziv": {},
            "neural_complexity": self.neural_complexity(activity_matrix),
            "multiscale_entropy": {}
        }
        
        # Shannon entropy for recent activity
        recent_activity = self.activity_history[-1]
        results["shannon_entropy"]["recent"] = self.shannon_entropy(recent_activity)
        
        # LZ complexity for binarized activity
        if len(recent_activity) > 0:
            threshold = np.median(recent_activity)
            binary = (recent_activity > threshold).astype(int)
            results["lempel_ziv"]["recent"] = self.lempel_ziv_complexity(binary)
        
        # Multiscale entropy for time series
        if len(self.activity_history) >= 10:
            # Take one neuron's time series
            time_series = activity_matrix[0, :]
            results["multiscale_entropy"] = self.multiscale_entropy(time_series)
        
        return results
    
    def get_complexity_summary(self) -> Dict[str, float]:
        """Get a summary of key complexity metrics.
        
        Returns:
            Dictionary of summarized complexity metrics
        """
        full_results = self.compute_activity_complexity()
        
        summary = {
            "shannon_entropy": full_results.get("shannon_entropy", {}).get("recent", 0.0),
            "lempel_ziv": full_results.get("lempel_ziv", {}).get("recent", 0.0),
            "neural_complexity": full_results.get("neural_complexity", {}).get("complexity", 0.0),
            "integration": full_results.get("neural_complexity", {}).get("integration", 0.0),
            "differentiation": full_results.get("neural_complexity", {}).get("differentiation", 0.0)
        }
        
        return summary


__all__ = ["ComplexityAnalyzer"]
