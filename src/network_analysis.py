"""Network analysis tools for studying neural connectivity and dynamics.

This module provides tools for:
- Network connectivity analysis (graph metrics)
- Firing pattern analysis
- Population dynamics
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Set, Optional, Any, TYPE_CHECKING
from collections import defaultdict, deque

if TYPE_CHECKING:
    try:
        from .brain_model import BrainModel
    except ImportError:
        from brain_model import BrainModel


class ConnectivityAnalyzer:
    """Analyze network connectivity using graph metrics."""

    def __init__(self, model: "BrainModel"):
        """Initialize connectivity analyzer.

        Args:
            model: Brain model to analyze.
        """
        self.model = model
        self._adjacency_cache: Optional[Dict[int, Set[int]]] = None

    def compute_degree_distribution(self) -> Dict[str, np.ndarray]:
        """Compute degree distribution of the network.

        Returns:
            Dictionary with 'in_degree' and 'out_degree' arrays.
        """
        in_degrees = defaultdict(int)
        out_degrees = defaultdict(int)

        # Count degrees from synapses
        for synapse in self.model.synapses:
            out_degrees[synapse.pre_id] += 1
            in_degrees[synapse.post_id] += 1

        # Convert to arrays
        neuron_ids = sorted(self.model.neurons.keys())
        in_degree_array = np.array([in_degrees[nid] for nid in neuron_ids])
        out_degree_array = np.array([out_degrees[nid] for nid in neuron_ids])

        return {
            'in_degree': in_degree_array,
            'out_degree': out_degree_array,
            'total_degree': in_degree_array + out_degree_array,
        }

    def compute_clustering_coefficient(self) -> float:
        """Compute average clustering coefficient of the network.

        Returns:
            Average clustering coefficient.
        """
        adjacency = self._get_adjacency_dict()
        clustering_coeffs = []

        for neuron_id in self.model.neurons.keys():
            neighbors = adjacency.get(neuron_id, set())
            k = len(neighbors)

            if k < 2:
                # Need at least 2 neighbors to compute clustering
                continue

            # Count triangles
            triangles = 0
            for n1 in neighbors:
                n1_neighbors = adjacency.get(n1, set())
                for n2 in neighbors:
                    if n1 != n2 and n2 in n1_neighbors:
                        triangles += 1

            # Clustering coefficient for this node
            possible_triangles = k * (k - 1)
            if possible_triangles > 0:
                local_clustering = triangles / possible_triangles
                clustering_coeffs.append(local_clustering)

        if not clustering_coeffs:
            return 0.0

        return float(np.mean(clustering_coeffs))

    def compute_path_lengths(self, sample_size: int = 100) -> Dict[str, float]:
        """Compute average path length using BFS sampling.

        Args:
            sample_size: Number of source nodes to sample.

        Returns:
            Dictionary with path length statistics.
        """
        adjacency = self._get_adjacency_dict()
        neuron_ids = list(self.model.neurons.keys())

        if not neuron_ids:
            return {'mean': 0.0, 'max': 0.0}

        # Sample source nodes
        sample_size = min(sample_size, len(neuron_ids))
        sampled_sources = np.random.choice(neuron_ids, size=sample_size, replace=False)

        all_distances = []

        for source in sampled_sources:
            distances = self._bfs_distances(source, adjacency)
            all_distances.extend([d for d in distances.values() if d > 0 and d < float('inf')])

        if not all_distances:
            return {'mean': 0.0, 'max': 0.0}

        return {
            'mean': float(np.mean(all_distances)),
            'max': float(np.max(all_distances)),
            'std': float(np.std(all_distances)),
        }

    def _bfs_distances(self, source: int, adjacency: Dict[int, Set[int]]) -> Dict[int, float]:
        """Compute shortest path distances from source using BFS.

        Args:
            source: Source neuron ID.
            adjacency: Adjacency dictionary.

        Returns:
            Dictionary of distances to each reachable node.
        """
        distances = {source: 0}
        queue = deque([source])

        while queue:
            current = queue.popleft()
            current_dist = distances[current]

            neighbors = adjacency.get(current, set())
            for neighbor in neighbors:
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)

        return distances

    def identify_hubs(self, top_k: int = 10) -> List[Tuple[int, int]]:
        """Identify hub neurons with highest degree.

        Args:
            top_k: Number of top hubs to return.

        Returns:
            List of (neuron_id, degree) tuples.
        """
        degrees = self.compute_degree_distribution()
        total_degrees = degrees['total_degree']

        neuron_ids = sorted(self.model.neurons.keys())
        neuron_degrees = [(nid, int(deg)) for nid, deg in zip(neuron_ids, total_degrees)]

        # Sort by degree
        neuron_degrees.sort(key=lambda x: x[1], reverse=True)

        return neuron_degrees[:top_k]

    def compute_modularity(self, communities: Dict[int, int]) -> float:
        """Compute modularity given community assignments.

        Args:
            communities: Dictionary mapping neuron_id to community_id.

        Returns:
            Modularity value.
        """
        # Count edges and degrees
        m = len(self.model.synapses)
        if m == 0:
            return 0.0

        degrees = self.compute_degree_distribution()
        total_degrees = degrees['total_degree']
        neuron_ids = sorted(self.model.neurons.keys())
        degree_dict = {nid: int(deg) for nid, deg in zip(neuron_ids, total_degrees)}

        # Compute modularity
        modularity = 0.0
        for synapse in self.model.synapses:
            pre_id = synapse.pre_id
            post_id = synapse.post_id

            if pre_id in communities and post_id in communities:
                # Delta function: 1 if same community, 0 otherwise
                delta = 1 if communities[pre_id] == communities[post_id] else 0

                # Expected edges
                ki = degree_dict.get(pre_id, 0)
                kj = degree_dict.get(post_id, 0)
                expected = (ki * kj) / (2 * m)

                modularity += delta - expected / (2 * m)

        modularity /= (2 * m)
        return float(modularity)

    def _get_adjacency_dict(self) -> Dict[int, Set[int]]:
        """Get adjacency dictionary for efficient graph operations.

        Returns:
            Dictionary mapping neuron_id to set of connected neuron_ids.
        """
        if self._adjacency_cache is not None:
            return self._adjacency_cache

        adjacency = defaultdict(set)
        for synapse in self.model.synapses:
            adjacency[synapse.pre_id].add(synapse.post_id)

        self._adjacency_cache = dict(adjacency)
        return self._adjacency_cache

    def clear_cache(self) -> None:
        """Clear cached computations."""
        self._adjacency_cache = None


class FiringPatternAnalyzer:
    """Analyze firing patterns and spike statistics."""

    def __init__(self):
        """Initialize firing pattern analyzer."""
        self.spike_history: Dict[int, List[float]] = defaultdict(list)

    def record_spikes(self, neuron_id: int, time: float) -> None:
        """Record a spike event.

        Args:
            neuron_id: ID of neuron that spiked.
            time: Time of spike.
        """
        self.spike_history[neuron_id].append(time)

    def compute_firing_rates(self, time_window: float) -> Dict[int, float]:
        """Compute firing rates for all neurons.

        Args:
            time_window: Time window for rate computation.

        Returns:
            Dictionary mapping neuron_id to firing rate.
        """
        firing_rates = {}

        for neuron_id, spike_times in self.spike_history.items():
            if not spike_times:
                firing_rates[neuron_id] = 0.0
                continue

            # Count spikes in time window
            recent_spikes = [t for t in spike_times if t >= spike_times[-1] - time_window]
            rate = len(recent_spikes) / time_window if time_window > 0 else 0.0
            firing_rates[neuron_id] = rate

        return firing_rates

    def compute_interspike_intervals(self, neuron_id: int) -> np.ndarray:
        """Compute inter-spike intervals for a neuron.

        Args:
            neuron_id: ID of neuron.

        Returns:
            Array of inter-spike intervals.
        """
        spike_times = self.spike_history.get(neuron_id, [])

        if len(spike_times) < 2:
            return np.array([])

        spike_times = sorted(spike_times)
        isis = np.diff(spike_times)
        return isis

    def compute_cv(self, neuron_id: int) -> float:
        """Compute coefficient of variation of ISIs.

        Args:
            neuron_id: ID of neuron.

        Returns:
            Coefficient of variation (CV).
        """
        isis = self.compute_interspike_intervals(neuron_id)

        if len(isis) < 2:
            return 0.0

        mean_isi = np.mean(isis)
        if mean_isi == 0:
            return 0.0

        std_isi = np.std(isis)
        cv = std_isi / mean_isi
        return float(cv)

    def detect_bursts(
        self,
        neuron_id: int,
        max_isi: float = 0.01,
        min_spikes: int = 3,
    ) -> List[List[float]]:
        """Detect burst events in spike train.

        Args:
            neuron_id: ID of neuron.
            max_isi: Maximum ISI within a burst.
            min_spikes: Minimum spikes to constitute a burst.

        Returns:
            List of bursts, each burst is a list of spike times.
        """
        spike_times = sorted(self.spike_history.get(neuron_id, []))

        if len(spike_times) < min_spikes:
            return []

        bursts = []
        current_burst = [spike_times[0]]

        for i in range(1, len(spike_times)):
            isi = spike_times[i] - spike_times[i - 1]

            if isi <= max_isi:
                current_burst.append(spike_times[i])
            else:
                if len(current_burst) >= min_spikes:
                    bursts.append(current_burst)
                current_burst = [spike_times[i]]

        # Check last burst
        if len(current_burst) >= min_spikes:
            bursts.append(current_burst)

        return bursts

    def compute_synchrony(
        self,
        neuron_ids: List[int],
        time_window: float = 0.005,
    ) -> float:
        """Compute synchrony measure for a population of neurons.

        Args:
            neuron_ids: List of neuron IDs.
            time_window: Time window for considering spikes synchronous.

        Returns:
            Synchrony measure (0 to 1).
        """
        if len(neuron_ids) < 2:
            return 0.0

        # Collect all spikes
        all_spikes = []
        for nid in neuron_ids:
            spikes = self.spike_history.get(nid, [])
            for spike_time in spikes:
                all_spikes.append((spike_time, nid))

        if len(all_spikes) < 2:
            return 0.0

        all_spikes.sort()

        # Count synchronous events
        synchronous_count = 0
        total_spikes = len(all_spikes)

        i = 0
        while i < len(all_spikes):
            time_i = all_spikes[i][0]
            sync_group = 1

            # Find all spikes within time window
            j = i + 1
            while j < len(all_spikes) and all_spikes[j][0] - time_i <= time_window:
                sync_group += 1
                j += 1

            if sync_group > 1:
                synchronous_count += sync_group

            i = j if j > i + 1 else i + 1

        synchrony = synchronous_count / total_spikes if total_spikes > 0 else 0.0
        return float(synchrony)

    def reset(self) -> None:
        """Clear spike history."""
        self.spike_history.clear()


class PopulationDynamicsAnalyzer:
    """Analyze population-level neural dynamics."""

    def __init__(self, model: "BrainModel"):
        """Initialize population dynamics analyzer.

        Args:
            model: Brain model to analyze.
        """
        self.model = model
        self.activity_history: List[np.ndarray] = []
        self.max_history = 1000

    def record_population_activity(self) -> None:
        """Record current population activity state."""
        # Collect membrane potentials
        neuron_ids = sorted(self.model.neurons.keys())
        activity = np.array([
            self.model.neurons[nid].v if hasattr(self.model.neurons[nid], 'v') else 0.0
            for nid in neuron_ids
        ])

        self.activity_history.append(activity)

        # Limit history size
        if len(self.activity_history) > self.max_history:
            self.activity_history.pop(0)

    def compute_population_rate(self, threshold: float = 0.0) -> float:
        """Compute instantaneous population firing rate.

        Args:
            threshold: Activity threshold for counting as active.

        Returns:
            Population firing rate.
        """
        if not self.activity_history:
            return 0.0

        latest_activity = self.activity_history[-1]
        active_count = np.sum(latest_activity > threshold)
        rate = active_count / len(latest_activity) if len(latest_activity) > 0 else 0.0

        return float(rate)

    def compute_mean_field(self) -> np.ndarray:
        """Compute mean field activity over recent history.

        Returns:
            Mean field activity vector.
        """
        if not self.activity_history:
            return np.array([])

        activity_matrix = np.array(self.activity_history)
        mean_field = np.mean(activity_matrix, axis=0)

        return mean_field

    def compute_variance(self) -> float:
        """Compute variance of population activity.

        Returns:
            Variance of activity.
        """
        if not self.activity_history:
            return 0.0

        latest_activity = self.activity_history[-1]
        variance = float(np.var(latest_activity))

        return variance

    def detect_oscillations(self, min_freq: float = 1.0, max_freq: float = 100.0) -> Dict[str, Any]:
        """Detect oscillatory patterns in population activity.

        Args:
            min_freq: Minimum frequency to consider (Hz).
            max_freq: Maximum frequency to consider (Hz).

        Returns:
            Dictionary with oscillation information.
        """
        if len(self.activity_history) < 10:
            return {'detected': False}

        # Compute population mean over time
        activity_matrix = np.array(self.activity_history)
        population_mean = np.mean(activity_matrix, axis=1)

        # Simple autocorrelation-based detection
        n = len(population_mean)
        autocorr = np.correlate(population_mean - np.mean(population_mean),
                               population_mean - np.mean(population_mean),
                               mode='full')
        autocorr = autocorr[n-1:]
        autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr

        # Find peaks in autocorrelation
        peaks = []
        for i in range(1, min(len(autocorr) - 1, 100)):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                if autocorr[i] > 0.3:  # Threshold for significant peak
                    peaks.append(i)

        if peaks:
            dominant_period = peaks[0]
            return {
                'detected': True,
                'period': dominant_period,
                'strength': float(autocorr[dominant_period]),
            }

        return {'detected': False}

    def compute_dimensionality(self, method: str = 'pca') -> int:
        """Estimate effective dimensionality of population activity.

        Args:
            method: Method for dimensionality estimation ('pca').

        Returns:
            Estimated dimensionality.
        """
        if len(self.activity_history) < 10:
            return 0

        activity_matrix = np.array(self.activity_history)

        # Center the data
        centered = activity_matrix - np.mean(activity_matrix, axis=0)

        # Compute covariance matrix
        cov_matrix = np.cov(centered.T)

        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[eigenvalues > 0]

        if len(eigenvalues) == 0:
            return 0

        # Use cumulative variance threshold
        total_var = np.sum(eigenvalues)
        cumsum = np.cumsum(eigenvalues)
        dimensionality = np.searchsorted(cumsum, 0.95 * total_var) + 1

        return int(dimensionality)

    def reset(self) -> None:
        """Clear activity history."""
        self.activity_history.clear()


class NetworkMotifDetector:
    """Detector for recurring connectivity patterns (network motifs).
    
    Network motifs are patterns of interconnections that recur significantly more
    often than in randomized networks. They are considered the building blocks of
    complex networks and can reveal functional circuit properties.
    
    This implementation focuses on 3-node motifs (triads) which are the smallest
    non-trivial motifs and computationally tractable.
    
    Attributes:
        model: The brain model to analyze
        motif_counts: Dictionary storing counts of each motif type
        
    Example:
        >>> detector = NetworkMotifDetector(model)
        >>> motifs = detector.detect_triadic_motifs()
        >>> print(f"Found {motifs['feedforward_chain']} feedforward chains")
    """
    
    def __init__(self, model: "BrainModel"):
        """Initialize network motif detector.
        
        Args:
            model: The brain model to analyze
        """
        self.model = model
        self.motif_counts: Dict[str, int] = {}
    
    def detect_triadic_motifs(self, sample_size: Optional[int] = None) -> Dict[str, int]:
        """Detect 3-node motifs in the network.
        
        Analyzes all or a sample of 3-node subgraphs (triads) to identify
        recurring connectivity patterns. Returns counts of each motif type.
        
        Motif types detected:
        - feedforward_chain: A→B→C (no feedback)
        - convergent: A→C, B→C (two inputs converge)
        - divergent: A→B, A→C (one input diverges)
        - feedback_loop: A→B→C→A (3-node cycle)
        - reciprocal_pair: A↔B→C (one reciprocal connection)
        - fully_connected: All 6 possible connections present
        
        Args:
            sample_size: Optional number of triads to sample (for large networks).
                        If None, analyzes all possible triads.
        
        Returns:
            Dictionary mapping motif type names to their counts
            
        Note:
            For networks with N neurons, there are N*(N-1)*(N-2)/6 possible triads.
            Sampling is recommended for networks with >1000 neurons.
        """
        self.motif_counts = {
            "feedforward_chain": 0,
            "convergent": 0,
            "divergent": 0,
            "feedback_loop": 0,
            "reciprocal_pair": 0,
            "fully_connected": 0,
            "empty": 0,  # No connections
            "other": 0,  # Other patterns
        }
        
        neuron_ids = list(self.model.neurons.keys())
        
        if len(neuron_ids) < 3:
            return self.motif_counts
        
        # Build adjacency information for fast lookup
        adjacency = self._build_adjacency_dict()
        
        # Generate all possible triads or sample
        from itertools import combinations
        all_triads = list(combinations(neuron_ids, 3))
        
        if sample_size and sample_size < len(all_triads):
            triads_to_analyze = random.sample(all_triads, sample_size)
        else:
            triads_to_analyze = all_triads
        
        # Analyze each triad
        for n1, n2, n3 in triads_to_analyze:
            motif_type = self._classify_triad_motif(n1, n2, n3, adjacency)
            self.motif_counts[motif_type] += 1
        
        return self.motif_counts.copy()
    
    def _build_adjacency_dict(self) -> Dict[Tuple[int, int], bool]:
        """Build adjacency dictionary for fast connection lookup.
        
        Returns:
            Dictionary mapping (pre_id, post_id) tuples to True
        """
        adjacency = {}
        for synapse in self.model.synapses:
            adjacency[(synapse.pre_id, synapse.post_id)] = True
        return adjacency
    
    def _classify_triad_motif(
        self,
        n1: int,
        n2: int,
        n3: int,
        adjacency: Dict[Tuple[int, int], bool]
    ) -> str:
        """Classify the connectivity pattern of a 3-node subgraph.
        
        Args:
            n1, n2, n3: The three neuron IDs
            adjacency: Adjacency dictionary for fast lookup
            
        Returns:
            String identifier for the motif type
        """
        # Check all 6 possible directed connections
        edges = {
            (n1, n2): (n1, n2) in adjacency,
            (n2, n1): (n2, n1) in adjacency,
            (n1, n3): (n1, n3) in adjacency,
            (n3, n1): (n3, n1) in adjacency,
            (n2, n3): (n2, n3) in adjacency,
            (n3, n2): (n3, n2) in adjacency,
        }
        
        edge_count = sum(edges.values())
        
        # No connections
        if edge_count == 0:
            return "empty"
        
        # Fully connected (all 6 edges)
        if edge_count == 6:
            return "fully_connected"
        
        # Check for feedback loop (3-cycle)
        if (edges[(n1, n2)] and edges[(n2, n3)] and edges[(n3, n1)]) or \
           (edges[(n1, n3)] and edges[(n3, n2)] and edges[(n2, n1)]):
            return "feedback_loop"
        
        # Check for reciprocal connections
        reciprocal_count = sum([
            edges[(n1, n2)] and edges[(n2, n1)],
            edges[(n1, n3)] and edges[(n3, n1)],
            edges[(n2, n3)] and edges[(n3, n2)],
        ])
        
        if reciprocal_count > 0:
            return "reciprocal_pair"
        
        # Check for feedforward chain (A→B→C with no feedback)
        chains = [
            (edges[(n1, n2)] and edges[(n2, n3)] and not any([
                edges[(n2, n1)], edges[(n3, n2)], edges[(n3, n1)], edges[(n1, n3)]
            ])),
            (edges[(n1, n3)] and edges[(n3, n2)] and not any([
                edges[(n3, n1)], edges[(n2, n3)], edges[(n2, n1)], edges[(n1, n2)]
            ])),
            (edges[(n2, n1)] and edges[(n1, n3)] and not any([
                edges[(n1, n2)], edges[(n3, n1)], edges[(n3, n2)], edges[(n2, n3)]
            ])),
            (edges[(n2, n3)] and edges[(n3, n1)] and not any([
                edges[(n3, n2)], edges[(n1, n3)], edges[(n1, n2)], edges[(n2, n1)]
            ])),
            (edges[(n3, n1)] and edges[(n1, n2)] and not any([
                edges[(n1, n3)], edges[(n2, n1)], edges[(n2, n3)], edges[(n3, n2)]
            ])),
            (edges[(n3, n2)] and edges[(n2, n1)] and not any([
                edges[(n2, n3)], edges[(n1, n2)], edges[(n1, n3)], edges[(n3, n1)]
            ])),
        ]
        
        if any(chains):
            return "feedforward_chain"
        
        # Check for convergent (two neurons connect to one target)
        if (edges[(n1, n3)] and edges[(n2, n3)]) or \
           (edges[(n1, n2)] and edges[(n3, n2)]) or \
           (edges[(n2, n1)] and edges[(n3, n1)]):
            return "convergent"
        
        # Check for divergent (one neuron connects to two targets)
        if (edges[(n1, n2)] and edges[(n1, n3)]) or \
           (edges[(n2, n1)] and edges[(n2, n3)]) or \
           (edges[(n3, n1)] and edges[(n3, n2)]):
            return "divergent"
        
        # Any other pattern
        return "other"
    
    def compute_motif_significance(
        self,
        n_randomizations: int = 100,
        seed: Optional[int] = None
    ) -> Dict[str, float]:
        """Compute statistical significance of motif counts via randomization.
        
        Compares observed motif counts to those in randomized networks with
        the same degree distribution. Returns z-scores for each motif type.
        
        Args:
            n_randomizations: Number of random networks to generate
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary mapping motif types to z-scores
            
        Note:
            Z-score > 2 indicates motif appears significantly more than expected.
            Z-score < -2 indicates motif appears significantly less than expected.
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Get observed motif counts
        observed = self.detect_triadic_motifs()
        
        # Store motif counts from randomized networks
        randomized_counts = {motif: [] for motif in observed.keys()}
        
        # Create randomized networks and count motifs
        for _ in range(n_randomizations):
            # Randomize by rewiring edges while preserving degree distribution
            randomized_model = self._create_degree_preserving_randomization()
            
            # Detect motifs in randomized network
            temp_detector = NetworkMotifDetector(randomized_model)
            random_motifs = temp_detector.detect_triadic_motifs()
            
            for motif, count in random_motifs.items():
                randomized_counts[motif].append(count)
        
        # Compute z-scores
        z_scores = {}
        for motif, obs_count in observed.items():
            random_counts = randomized_counts[motif]
            mean = np.mean(random_counts)
            std = np.std(random_counts)
            
            if std > 0:
                z_scores[motif] = (obs_count - mean) / std
            else:
                z_scores[motif] = 0.0
        
        return z_scores
    
    def _create_degree_preserving_randomization(self) -> "BrainModel":
        """Create randomized network preserving degree distribution.
        
        Uses edge rewiring to create random network with same in/out degrees.
        
        Returns:
            Randomized brain model
        """
        # This is a simplified implementation
        # A full implementation would use the configuration model or edge-swapping
        
        # For now, create a simple random network with same number of edges
        import copy
        randomized = copy.deepcopy(self.model)
        
        # Clear synapses
        randomized.synapses.clear()
        
        # Recreate same number of random synapses
        neuron_ids = list(randomized.neurons.keys())
        n_synapses = len(self.model.synapses)
        
        for _ in range(n_synapses):
            pre_id = np.random.choice(neuron_ids)
            post_id = np.random.choice(neuron_ids)
            
            # Avoid self-connections
            while pre_id == post_id:
                post_id = np.random.choice(neuron_ids)
            
            # Add synapse (if not duplicate)
            try:
                randomized.add_synapse(pre_id, post_id, weight=0.1)
            except (ValueError, KeyError):
                # Ignore duplicates or invalid neuron IDs
                pass
        
        return randomized


class InformationTheoryAnalyzer:
    """Analyze information theoretic properties of neural networks."""
    
    def __init__(self, model: "BrainModel"):
        """Initialize information theory analyzer.
        
        Args:
            model: Brain model to analyze
        """
        self.model = model
        self.spike_history: List[List[int]] = []
    
    def record_spikes(self, spikes: List[int]) -> None:
        """Record spike events for later analysis.
        
        Args:
            spikes: List of neuron IDs that spiked this timestep
        """
        self.spike_history.append(spikes)
    
    def calculate_entropy(self, neuron_id: int, time_window: int = 100) -> float:
        """Calculate entropy of a neuron's firing pattern.
        
        Higher entropy indicates more random/unpredictable firing.
        Lower entropy indicates more regular/predictable firing.
        
        Args:
            neuron_id: ID of neuron to analyze
            time_window: Number of recent timesteps to analyze
        
        Returns:
            Shannon entropy in bits
        """
        if len(self.spike_history) < time_window:
            time_window = len(self.spike_history)
        
        if time_window == 0:
            return 0.0
        
        # Count spikes vs no-spikes
        recent_history = self.spike_history[-time_window:]
        spike_count = sum(1 for spikes in recent_history if neuron_id in spikes)
        no_spike_count = time_window - spike_count
        
        # Calculate probabilities
        p_spike = spike_count / time_window
        p_no_spike = no_spike_count / time_window
        
        # Calculate entropy: H = -Σ p(x) log2(p(x))
        entropy = 0.0
        if p_spike > 0:
            entropy -= p_spike * np.log2(p_spike)
        if p_no_spike > 0:
            entropy -= p_no_spike * np.log2(p_no_spike)
        
        return float(entropy)
    
    def calculate_mutual_information(
        self, 
        neuron_a: int, 
        neuron_b: int,
        time_window: int = 100,
        bin_size: int = 1
    ) -> float:
        """Calculate mutual information between two neurons.
        
        Measures how much knowing one neuron's activity tells us about another.
        High MI indicates strong statistical dependence (not necessarily causal).
        
        Args:
            neuron_a: First neuron ID
            neuron_b: Second neuron ID
            time_window: Number of timesteps to analyze
            bin_size: Bin size for discretizing spike times (default: 1ms)
        
        Returns:
            Mutual information in bits
        """
        if len(self.spike_history) < time_window:
            time_window = len(self.spike_history)
        
        if time_window == 0:
            return 0.0
        
        recent_history = self.spike_history[-time_window:]
        
        # Count joint and marginal occurrences
        count_00 = 0  # Neither fires
        count_01 = 0  # Only B fires
        count_10 = 0  # Only A fires
        count_11 = 0  # Both fire
        
        for spikes in recent_history:
            a_fires = neuron_a in spikes
            b_fires = neuron_b in spikes
            
            if a_fires and b_fires:
                count_11 += 1
            elif a_fires:
                count_10 += 1
            elif b_fires:
                count_01 += 1
            else:
                count_00 += 1
        
        # Calculate probabilities
        total = time_window
        p_00 = count_00 / total
        p_01 = count_01 / total
        p_10 = count_10 / total
        p_11 = count_11 / total
        
        # Marginal probabilities
        p_a0 = p_00 + p_01  # A doesn't fire
        p_a1 = p_10 + p_11  # A fires
        p_b0 = p_00 + p_10  # B doesn't fire
        p_b1 = p_01 + p_11  # B fires
        
        # Calculate mutual information: I(A;B) = Σ p(a,b) log2(p(a,b) / (p(a)p(b)))
        mi = 0.0
        
        if p_00 > 0 and p_a0 > 0 and p_b0 > 0:
            mi += p_00 * np.log2(p_00 / (p_a0 * p_b0))
        if p_01 > 0 and p_a0 > 0 and p_b1 > 0:
            mi += p_01 * np.log2(p_01 / (p_a0 * p_b1))
        if p_10 > 0 and p_a1 > 0 and p_b0 > 0:
            mi += p_10 * np.log2(p_10 / (p_a1 * p_b0))
        if p_11 > 0 and p_a1 > 0 and p_b1 > 0:
            mi += p_11 * np.log2(p_11 / (p_a1 * p_b1))
        
        return float(mi)
    
    def calculate_transfer_entropy(
        self,
        source_neuron: int,
        target_neuron: int,
        time_window: int = 100,
        delay: int = 1
    ) -> float:
        """Calculate transfer entropy from source to target neuron.
        
        Measures directed information flow: how much source's past
        helps predict target's future, beyond target's own history.
        
        Args:
            source_neuron: Source neuron ID
            target_neuron: Target neuron ID
            time_window: Number of timesteps to analyze
            delay: Time delay to consider (in timesteps)
        
        Returns:
            Transfer entropy in bits (directional)
        """
        if len(self.spike_history) < time_window + delay:
            return 0.0
        
        # Extract histories
        recent_history = self.spike_history[-(time_window + delay):]
        
        # Count state transitions
        counts = defaultdict(int)
        
        for t in range(len(recent_history) - delay):
            # Current states
            target_past = 1 if target_neuron in recent_history[t] else 0
            source_past = 1 if source_neuron in recent_history[t] else 0
            
            # Future state (after delay)
            target_future = 1 if target_neuron in recent_history[t + delay] else 0
            
            # Count (target_future, target_past, source_past) combinations
            state = (target_future, target_past, source_past)
            counts[state] += 1
        
        # Calculate probabilities
        total = sum(counts.values())
        if total == 0:
            return 0.0
        
        probs = {state: count / total for state, count in counts.items()}
        
        # Calculate marginal probabilities
        p_tf_tp = defaultdict(float)  # P(target_future, target_past)
        p_tf_tp_sp = defaultdict(float)  # P(target_future, target_past, source_past)
        p_tp = defaultdict(float)  # P(target_past)
        p_tp_sp = defaultdict(float)  # P(target_past, source_past)
        
        for (tf, tp, sp), prob in probs.items():
            p_tf_tp[(tf, tp)] += prob
            p_tf_tp_sp[(tf, tp, sp)] = prob
            p_tp[tp] += prob
            p_tp_sp[(tp, sp)] += prob
        
        # Calculate transfer entropy
        te = 0.0
        for (tf, tp, sp), p_full in p_tf_tp_sp.items():
            if p_full > 0:
                p_cond = p_tf_tp.get((tf, tp), 0)
                p_tp_val = p_tp.get(tp, 0)
                p_tp_sp_val = p_tp_sp.get((tp, sp), 0)
                
                if p_cond > 0 and p_tp_val > 0 and p_tp_sp_val > 0:
                    te += p_full * np.log2((p_full * p_tp_val) / (p_cond * p_tp_sp_val))
        
        return float(te)
    
    def calculate_population_entropy(self, time_window: int = 100) -> float:
        """Calculate entropy of population activity patterns.
        
        Measures diversity of population states over time.
        
        Args:
            time_window: Number of timesteps to analyze
        
        Returns:
            Population entropy in bits
        """
        if len(self.spike_history) < time_window:
            time_window = len(self.spike_history)
        
        if time_window == 0:
            return 0.0
        
        recent_history = self.spike_history[-time_window:]
        
        # Convert each timestep's spike pattern to a hashable state
        # (using frozenset for order-independent comparison)
        state_counts = defaultdict(int)
        for spikes in recent_history:
            state = frozenset(spikes)
            state_counts[state] += 1
        
        # Calculate probabilities
        probs = [count / time_window for count in state_counts.values()]
        
        # Calculate entropy
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        
        return float(entropy)
    
    def get_analysis_summary(self, time_window: int = 100) -> Dict[str, Any]:
        """Get comprehensive information theory analysis summary.
        
        Args:
            time_window: Number of timesteps to analyze
        
        Returns:
            Dictionary with various information theoretic measures
        """
        neuron_ids = list(self.model.neurons.keys())
        
        # Sample neurons for pairwise analysis (avoid O(n²) for large networks)
        sample_size = min(10, len(neuron_ids))
        sample_neurons = np.random.choice(neuron_ids, size=sample_size, replace=False)
        
        # Calculate individual neuron entropies
        entropies = {}
        for nid in sample_neurons:
            entropies[nid] = self.calculate_entropy(nid, time_window)
        
        # Calculate pairwise mutual information
        mutual_info = []
        for i, nid_a in enumerate(sample_neurons):
            for nid_b in sample_neurons[i+1:]:
                mi = self.calculate_mutual_information(nid_a, nid_b, time_window)
                mutual_info.append(mi)
        
        # Population measures
        pop_entropy = self.calculate_population_entropy(time_window)
        
        return {
            'neuron_entropies': {
                'mean': float(np.mean(list(entropies.values()))) if entropies else 0.0,
                'std': float(np.std(list(entropies.values()))) if entropies else 0.0,
                'min': float(np.min(list(entropies.values()))) if entropies else 0.0,
                'max': float(np.max(list(entropies.values()))) if entropies else 0.0,
            },
            'mutual_information': {
                'mean': float(np.mean(mutual_info)) if mutual_info else 0.0,
                'std': float(np.std(mutual_info)) if mutual_info else 0.0,
                'max': float(np.max(mutual_info)) if mutual_info else 0.0,
            },
            'population_entropy': pop_entropy,
            'sample_size': sample_size,
            'time_window': time_window
        }
