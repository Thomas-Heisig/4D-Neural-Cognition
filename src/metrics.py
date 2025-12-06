"""Advanced metrics for evaluating neural network performance.

This module provides information theory metrics, network stability measures,
and other advanced evaluation tools for 4D neural cognition.
"""

import numpy as np
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from collections import Counter

if TYPE_CHECKING:
    from .brain_model import BrainModel


def calculate_entropy(spike_counts: List[int]) -> float:
    """Calculate Shannon entropy of spike distribution.
    
    Measures the uncertainty/randomness in the spike pattern.
    Higher entropy = more random, Lower entropy = more predictable
    
    Args:
        spike_counts: List of spike counts for different neurons or time bins
        
    Returns:
        Entropy value in bits (base 2)
        
    Examples:
        >>> calculate_entropy([10, 10, 10, 10])  # Uniform - high entropy
        2.0
        >>> calculate_entropy([40, 0, 0, 0])  # Concentrated - low entropy
        0.0
    """
    if not spike_counts:
        return 0.0
    
    if sum(spike_counts) == 0:
        return 0.0
    
    # Convert to probabilities
    total = sum(spike_counts)
    probabilities = [count / total for count in spike_counts if count > 0]
    
    # Calculate entropy: H(X) = -Σ p(x) * log2(p(x))
    entropy = -sum(p * np.log2(p) for p in probabilities)
    
    return float(entropy)


def calculate_mutual_information(
    x_values: List[int],
    y_values: List[int]
) -> float:
    """Calculate mutual information between two discrete variables.
    
    Measures how much knowing one variable reduces uncertainty about the other.
    Useful for measuring correlation between input patterns and neural responses.
    
    Args:
        x_values: First variable observations
        y_values: Second variable observations (same length as x_values)
        
    Returns:
        Mutual information in bits
        
    Raises:
        ValueError: If x_values and y_values have different lengths
    """
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have the same length")
    
    if len(x_values) == 0:
        return 0.0
    
    n = len(x_values)
    
    # Count joint occurrences
    joint_counts = Counter(zip(x_values, y_values))
    x_counts = Counter(x_values)
    y_counts = Counter(y_values)
    
    # Calculate mutual information: I(X;Y) = Σ p(x,y) * log2(p(x,y) / (p(x)*p(y)))
    mi = 0.0
    for (x, y), joint_count in joint_counts.items():
        p_xy = joint_count / n
        p_x = x_counts[x] / n
        p_y = y_counts[y] / n
        
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * np.log2(p_xy / (p_x * p_y))
    
    return float(mi)


def calculate_spike_rate_entropy(spike_history: List[List[int]]) -> float:
    """Calculate entropy of spike rate distribution across neurons.
    
    Args:
        spike_history: List of spike lists, each inner list contains neuron IDs
                      that spiked in that time step
                      
    Returns:
        Entropy of the spike rate distribution
    """
    if not spike_history:
        return 0.0
    
    # Count spikes per neuron
    neuron_spike_counts = Counter()
    for spikes in spike_history:
        for neuron_id in spikes:
            neuron_spike_counts[neuron_id] += 1
    
    # Get counts as list
    counts = list(neuron_spike_counts.values())
    
    return calculate_entropy(counts)


def calculate_network_stability(
    activity_history: List[float],
    window_size: int = 10
) -> Dict[str, float]:
    """Calculate stability metrics for network activity.
    
    Measures how stable the network activity is over time.
    
    Args:
        activity_history: List of activity values over time (e.g., total spikes per step)
        window_size: Size of sliding window for local stability calculation
        
    Returns:
        Dictionary with stability metrics:
        - variance: Overall variance in activity
        - cv: Coefficient of variation (std/mean)
        - local_stability: Average stability within sliding windows
        - trend: Linear trend coefficient (positive = increasing, negative = decreasing)
    """
    if len(activity_history) < 2:
        return {
            'variance': 0.0,
            'cv': 0.0,
            'local_stability': 1.0,
            'trend': 0.0
        }
    
    activity_array = np.array(activity_history)
    
    # Overall variance
    variance = float(np.var(activity_array))
    
    # Coefficient of variation
    mean_activity = np.mean(activity_array)
    if mean_activity > 0:
        cv = float(np.std(activity_array) / mean_activity)
    else:
        cv = 0.0
    
    # Local stability (inverse of local variance)
    local_stabilities = []
    for i in range(len(activity_history) - window_size + 1):
        window = activity_array[i:i + window_size]
        window_std = np.std(window)
        window_mean = np.mean(window)
        
        if window_mean > 0:
            # Inverse CV for this window (higher = more stable)
            local_stability = 1.0 / (1.0 + window_std / window_mean)
        else:
            local_stability = 1.0
        
        local_stabilities.append(local_stability)
    
    avg_local_stability = float(np.mean(local_stabilities)) if local_stabilities else 1.0
    
    # Linear trend
    if len(activity_history) > 1:
        x = np.arange(len(activity_history))
        coeffs = np.polyfit(x, activity_array, 1)
        trend = float(coeffs[0])  # Slope
    else:
        trend = 0.0
    
    return {
        'variance': variance,
        'cv': cv,
        'local_stability': avg_local_stability,
        'trend': trend
    }


def calculate_burst_metrics(spike_times: List[int], burst_threshold: int = 3) -> Dict[str, float]:
    """Calculate burst detection metrics.
    
    Identifies bursts (rapid sequences of spikes) in spike train data.
    
    Args:
        spike_times: List of time steps when spikes occurred
        burst_threshold: Minimum number of spikes in close succession to count as burst
        
    Returns:
        Dictionary with burst metrics:
        - num_bursts: Number of bursts detected
        - burst_rate: Bursts per time unit
        - avg_burst_size: Average spikes per burst
        - burst_fraction: Fraction of spikes that are part of bursts
    """
    if len(spike_times) < burst_threshold:
        return {
            'num_bursts': 0,
            'burst_rate': 0.0,
            'avg_burst_size': 0.0,
            'burst_fraction': 0.0
        }
    
    # Sort spike times
    sorted_times = sorted(spike_times)
    
    # Detect bursts (ISI < 5 time steps)
    max_isi = 5
    bursts = []
    current_burst = [sorted_times[0]]
    
    for i in range(1, len(sorted_times)):
        isi = sorted_times[i] - sorted_times[i-1]
        
        if isi <= max_isi:
            current_burst.append(sorted_times[i])
        else:
            if len(current_burst) >= burst_threshold:
                bursts.append(current_burst)
            current_burst = [sorted_times[i]]
    
    # Check last burst
    if len(current_burst) >= burst_threshold:
        bursts.append(current_burst)
    
    # Calculate metrics
    num_bursts = len(bursts)
    
    if num_bursts > 0:
        spikes_in_bursts = sum(len(burst) for burst in bursts)
        avg_burst_size = spikes_in_bursts / num_bursts
        burst_fraction = spikes_in_bursts / len(spike_times)
    else:
        avg_burst_size = 0.0
        burst_fraction = 0.0
    
    total_time = max(sorted_times) - min(sorted_times) + 1 if sorted_times else 1
    burst_rate = num_bursts / total_time
    
    return {
        'num_bursts': num_bursts,
        'burst_rate': float(burst_rate),
        'avg_burst_size': float(avg_burst_size),
        'burst_fraction': float(burst_fraction)
    }


def calculate_population_synchrony(
    spike_matrix: np.ndarray,
    time_window: int = 1
) -> float:
    """Calculate population synchrony metric.
    
    Measures how synchronized the population activity is.
    
    Args:
        spike_matrix: Binary matrix [neurons x time] indicating spikes
        time_window: Time window for synchrony calculation
        
    Returns:
        Synchrony value between 0 (no synchrony) and 1 (perfect synchrony)
    """
    if spike_matrix.size == 0:
        return 0.0
    
    # For each time bin, count how many neurons spiked
    spikes_per_bin = np.sum(spike_matrix, axis=0)
    
    # Normalize by number of neurons
    n_neurons = spike_matrix.shape[0]
    if n_neurons == 0:
        return 0.0
    
    normalized_activity = spikes_per_bin / n_neurons
    
    # Synchrony is the variance of normalized activity
    # High variance = high synchrony (many neurons spike together sometimes)
    # Low variance = low synchrony (neurons spike independently)
    synchrony = float(np.var(normalized_activity))
    
    return synchrony


def calculate_learning_curve_metrics(
    performance_history: List[float],
    convergence_threshold: float = 0.01
) -> Dict[str, Any]:
    """Calculate metrics from a learning curve.
    
    Args:
        performance_history: List of performance values over time
        convergence_threshold: Threshold for detecting convergence
        
    Returns:
        Dictionary with learning curve metrics:
        - initial_performance: First performance value
        - final_performance: Last performance value
        - improvement: Total improvement (final - initial)
        - convergence_step: Step where performance converged (or None)
        - learning_rate: Average rate of improvement
        - plateau_reached: Whether learning has plateaued
    """
    if not performance_history:
        return {
            'initial_performance': 0.0,
            'final_performance': 0.0,
            'improvement': 0.0,
            'convergence_step': None,
            'learning_rate': 0.0,
            'plateau_reached': False
        }
    
    initial_performance = performance_history[0]
    final_performance = performance_history[-1]
    improvement = final_performance - initial_performance
    
    # Detect convergence (performance change < threshold for extended period)
    convergence_step = None
    window_size = min(10, len(performance_history) // 4)
    
    if len(performance_history) > window_size:
        for i in range(window_size, len(performance_history)):
            window = performance_history[i-window_size:i]
            window_variance = np.var(window)
            
            if window_variance < convergence_threshold:
                convergence_step = i
                break
    
    # Learning rate (average change per step)
    if len(performance_history) > 1:
        learning_rate = improvement / (len(performance_history) - 1)
    else:
        learning_rate = 0.0
    
    # Plateau detection (last 20% of curve has low variance)
    plateau_reached = False
    if len(performance_history) >= 10:
        last_portion = performance_history[-len(performance_history)//5:]
        if np.var(last_portion) < convergence_threshold:
            plateau_reached = True
    
    return {
        'initial_performance': float(initial_performance),
        'final_performance': float(final_performance),
        'improvement': float(improvement),
        'convergence_step': convergence_step,
        'learning_rate': float(learning_rate),
        'plateau_reached': plateau_reached
    }


def calculate_generalization_metrics(
    train_performance: List[float],
    test_performance: List[float]
) -> Dict[str, float]:
    """Calculate generalization metrics from train/test performance.
    
    Args:
        train_performance: Performance on training data over time
        test_performance: Performance on test data over time
        
    Returns:
        Dictionary with generalization metrics:
        - final_train: Final training performance
        - final_test: Final test performance
        - generalization_gap: Difference between train and test
        - overfitting_score: Metric indicating degree of overfitting (0=none, 1=severe)
    """
    if not train_performance or not test_performance:
        return {
            'final_train': 0.0,
            'final_test': 0.0,
            'generalization_gap': 0.0,
            'overfitting_score': 0.0
        }
    
    final_train = train_performance[-1]
    final_test = test_performance[-1]
    generalization_gap = final_train - final_test
    
    # Overfitting score: normalized gap considering performance levels
    if final_train > 0:
        overfitting_score = max(0.0, generalization_gap / final_train)
    else:
        overfitting_score = 0.0
    
    return {
        'final_train': float(final_train),
        'final_test': float(final_test),
        'generalization_gap': float(generalization_gap),
        'overfitting_score': float(overfitting_score)
    }
