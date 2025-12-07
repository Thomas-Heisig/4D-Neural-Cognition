"""Network analysis tools for studying neural connectivity and dynamics.

This module provides tools for:
- Network connectivity analysis (graph metrics)
- Firing pattern analysis
- Population dynamics
"""

import numpy as np
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
        for synapse in self.model.synapses.values():
            out_degrees[synapse.pre_neuron_id] += 1
            in_degrees[synapse.post_neuron_id] += 1

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
        for synapse in self.model.synapses.values():
            pre_id = synapse.pre_neuron_id
            post_id = synapse.post_neuron_id

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
        for synapse in self.model.synapses.values():
            adjacency[synapse.pre_neuron_id].add(synapse.post_neuron_id)

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
