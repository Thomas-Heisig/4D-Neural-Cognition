"""Visualization tools for evaluation and analysis.

This module provides visualization functions for:
- Performance comparison plots
- Learning curve visualization
- Confusion matrices for classification
- Activity pattern visualization during tasks
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def plot_performance_comparison(
    results: List[Dict[str, Any]],
    metric_name: str = "accuracy",
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """Create performance comparison plot across different configurations.
    
    Args:
        results: List of result dictionaries with 'name' and metrics
        metric_name: Name of metric to plot
        save_path: Optional path to save figure
        
    Returns:
        Dictionary with plot data for rendering
    """
    if not results:
        return {"error": "No results provided"}
    
    names = [r.get("name", f"Config {i}") for i, r in enumerate(results)]
    values = [r.get(metric_name, 0.0) for r in results]
    
    plot_data = {
        "type": "bar",
        "names": names,
        "values": values,
        "metric_name": metric_name,
        "title": f"Performance Comparison - {metric_name}",
        "xlabel": "Configuration",
        "ylabel": metric_name.capitalize(),
    }
    
    if save_path:
        plot_data["save_path"] = save_path
    
    return plot_data


def plot_learning_curves(
    performance_history: List[float],
    labels: Optional[List[str]] = None,
    title: str = "Learning Curve",
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """Visualize learning curves over time.
    
    Args:
        performance_history: Single history or list of histories
        labels: Optional labels for each curve
        title: Plot title
        save_path: Optional path to save figure
        
    Returns:
        Dictionary with plot data
    """
    # Handle single or multiple curves
    if not isinstance(performance_history[0], list):
        performance_history = [performance_history]
    
    if labels is None:
        labels = [f"Curve {i+1}" for i in range(len(performance_history))]
    
    plot_data = {
        "type": "line",
        "curves": [
            {
                "label": label,
                "x": list(range(len(curve))),
                "y": curve
            }
            for label, curve in zip(labels, performance_history)
        ],
        "title": title,
        "xlabel": "Training Step",
        "ylabel": "Performance",
    }
    
    if save_path:
        plot_data["save_path"] = save_path
    
    return plot_data


def create_confusion_matrix(
    predictions: List[int],
    targets: List[int],
    num_classes: Optional[int] = None
) -> np.ndarray:
    """Create confusion matrix for classification results.
    
    Args:
        predictions: List of predicted class labels
        targets: List of true class labels
        num_classes: Number of classes (inferred if not provided)
        
    Returns:
        Confusion matrix as numpy array [num_classes x num_classes]
        
    Raises:
        ValueError: If predictions and targets have different lengths
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have same length")
    
    if num_classes is None:
        num_classes = max(max(predictions), max(targets)) + 1
    
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    
    for pred, true in zip(predictions, targets):
        if 0 <= pred < num_classes and 0 <= true < num_classes:
            confusion[true, pred] += 1
    
    return confusion


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """Visualize confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix array
        class_names: Optional names for classes
        normalize: Whether to normalize by row (true class)
        save_path: Optional path to save figure
        
    Returns:
        Dictionary with plot data
    """
    cm = confusion_matrix.copy()
    
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        cm = cm.astype(float) / row_sums
    
    num_classes = cm.shape[0]
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    plot_data = {
        "type": "heatmap",
        "matrix": cm.tolist(),
        "class_names": class_names,
        "title": "Confusion Matrix" + (" (Normalized)" if normalize else ""),
        "xlabel": "Predicted Class",
        "ylabel": "True Class",
        "colorbar_label": "Proportion" if normalize else "Count",
    }
    
    if save_path:
        plot_data["save_path"] = save_path
    
    return plot_data


def visualize_activity_patterns(
    spike_history: List[List[int]],
    time_window: Optional[Tuple[int, int]] = None,
    neuron_ids: Optional[List[int]] = None,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """Visualize neural activity patterns during task execution.
    
    Creates a raster plot showing spike times for different neurons.
    
    Args:
        spike_history: List of spike lists (each inner list has neuron IDs that spiked)
        time_window: Optional (start, end) time range to plot
        neuron_ids: Optional list of specific neurons to include
        save_path: Optional path to save figure
        
    Returns:
        Dictionary with plot data
    """
    if not spike_history:
        return {"error": "No spike history provided"}
    
    # Apply time window
    if time_window:
        start, end = time_window
        spike_history = spike_history[start:end]
        time_offset = start
    else:
        time_offset = 0
    
    # Collect all spike events
    spike_events = []
    for t, spikes in enumerate(spike_history):
        for neuron_id in spikes:
            if neuron_ids is None or neuron_id in neuron_ids:
                spike_events.append((t + time_offset, neuron_id))
    
    if not spike_events:
        return {"error": "No spikes in specified range"}
    
    plot_data = {
        "type": "raster",
        "events": spike_events,
        "title": "Neural Activity Pattern",
        "xlabel": "Time Step",
        "ylabel": "Neuron ID",
    }
    
    if save_path:
        plot_data["save_path"] = save_path
    
    return plot_data


def plot_spike_rate_histogram(
    spike_counts: Dict[int, int],
    bins: int = 20,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """Create histogram of spike rates across neurons.
    
    Args:
        spike_counts: Dictionary mapping neuron ID to spike count
        bins: Number of histogram bins
        save_path: Optional path to save figure
        
    Returns:
        Dictionary with plot data
    """
    if not spike_counts:
        return {"error": "No spike counts provided"}
    
    counts = list(spike_counts.values())
    
    # Calculate histogram
    hist, bin_edges = np.histogram(counts, bins=bins)
    
    plot_data = {
        "type": "histogram",
        "counts": hist.tolist(),
        "bin_edges": bin_edges.tolist(),
        "title": "Spike Rate Distribution",
        "xlabel": "Spike Count",
        "ylabel": "Number of Neurons",
    }
    
    if save_path:
        plot_data["save_path"] = save_path
    
    return plot_data


def create_comparison_table(
    results: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Create comparison table for multiple configurations.
    
    Args:
        results: List of result dictionaries
        metrics: Optional list of specific metrics to include
        
    Returns:
        Dictionary with table data
    """
    if not results:
        return {"error": "No results provided"}
    
    # Determine metrics to include
    if metrics is None:
        # Get all unique metrics from results
        metrics = set()
        for result in results:
            metrics.update(k for k in result.keys() if k != "name")
        metrics = sorted(list(metrics))
    
    # Build table
    rows = []
    for result in results:
        row = {
            "name": result.get("name", "Unnamed"),
            **{metric: result.get(metric, "N/A") for metric in metrics}
        }
        rows.append(row)
    
    table_data = {
        "type": "table",
        "columns": ["name"] + metrics,
        "rows": rows,
        "title": "Configuration Comparison",
    }
    
    return table_data


def plot_network_statistics(
    stats_history: List[Dict[str, Any]],
    stat_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """Plot network statistics over time.
    
    Args:
        stats_history: List of statistics dictionaries over time
        stat_names: Optional list of specific stats to plot
        save_path: Optional path to save figure
        
    Returns:
        Dictionary with plot data
    """
    if not stats_history:
        return {"error": "No statistics history provided"}
    
    # Determine which stats to plot
    if stat_names is None:
        # Use first stats dict as template
        stat_names = [k for k in stats_history[0].keys() if isinstance(stats_history[0][k], (int, float))]
    
    # Extract time series for each stat
    curves = []
    for stat_name in stat_names:
        values = [stats.get(stat_name, 0) for stats in stats_history]
        curves.append({
            "label": stat_name,
            "x": list(range(len(values))),
            "y": values
        })
    
    plot_data = {
        "type": "line",
        "curves": curves,
        "title": "Network Statistics Over Time",
        "xlabel": "Time Step",
        "ylabel": "Value",
    }
    
    if save_path:
        plot_data["save_path"] = save_path
    
    return plot_data


def calculate_accuracy_from_confusion(confusion_matrix: np.ndarray) -> float:
    """Calculate overall accuracy from confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix array
        
    Returns:
        Overall accuracy (0-1)
    """
    if confusion_matrix.size == 0:
        return 0.0
    
    correct = np.trace(confusion_matrix)
    total = np.sum(confusion_matrix)
    
    if total == 0:
        return 0.0
    
    return float(correct / total)


def calculate_class_metrics(confusion_matrix: np.ndarray) -> Dict[int, Dict[str, float]]:
    """Calculate per-class precision, recall, and F1 score.
    
    Args:
        confusion_matrix: Confusion matrix array
        
    Returns:
        Dictionary mapping class index to metrics dict
    """
    num_classes = confusion_matrix.shape[0]
    class_metrics = {}
    
    for i in range(num_classes):
        true_positives = confusion_matrix[i, i]
        false_positives = np.sum(confusion_matrix[:, i]) - true_positives
        false_negatives = np.sum(confusion_matrix[i, :]) - true_positives
        
        # Precision
        if true_positives + false_positives > 0:
            precision = true_positives / (true_positives + false_positives)
        else:
            precision = 0.0
        
        # Recall
        if true_positives + false_negatives > 0:
            recall = true_positives / (true_positives + false_negatives)
        else:
            recall = 0.0
        
        # F1 Score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        class_metrics[i] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }
    
    return class_metrics


def plot_raster(
    spike_times: Dict[int, List[int]],
    time_window: Optional[Tuple[int, int]] = None,
    neuron_ids: Optional[List[int]] = None,
    title: str = "Raster Plot",
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """Create a raster plot showing spike times for multiple neurons.
    
    A raster plot displays each neuron's spike times as vertical marks,
    with one row per neuron.
    
    Args:
        spike_times: Dictionary mapping neuron_id to list of spike times
        time_window: Optional (start_time, end_time) tuple to limit display
        neuron_ids: Optional list of neuron IDs to include (default: all)
        title: Plot title
        save_path: Optional path to save figure
        
    Returns:
        Dictionary with plot data for rendering
    """
    if not spike_times:
        return {"error": "No spike times provided"}
    
    # Filter neurons if specified
    if neuron_ids is not None:
        spike_times = {nid: times for nid, times in spike_times.items() if nid in neuron_ids}
    
    if not spike_times:
        return {"error": "No neurons in specified list"}
    
    # Prepare spike events
    events = []
    for neuron_id, times in spike_times.items():
        for t in times:
            # Filter by time window if specified
            if time_window is None or (time_window[0] <= t <= time_window[1]):
                events.append((t, neuron_id))
    
    if not events:
        return {"error": "No spikes in specified time window"}
    
    # Sort by neuron ID for better visualization
    events.sort(key=lambda x: (x[1], x[0]))
    
    plot_data = {
        "type": "raster",
        "events": events,
        "title": title,
        "xlabel": "Time (ms)",
        "ylabel": "Neuron ID",
        "num_neurons": len(spike_times),
        "time_range": time_window if time_window else (min(e[0] for e in events), max(e[0] for e in events)),
    }
    
    if save_path:
        plot_data["save_path"] = save_path
    
    return plot_data


def plot_psth(
    spike_times: Dict[int, List[int]],
    stimulus_times: List[int],
    pre_window: int = 50,
    post_window: int = 200,
    bin_size: int = 10,
    neuron_ids: Optional[List[int]] = None,
    title: str = "Peri-Stimulus Time Histogram (PSTH)",
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """Create a Peri-Stimulus Time Histogram (PSTH).
    
    PSTH shows the average firing rate of neurons aligned to stimulus onset.
    Useful for understanding neural responses to stimuli.
    
    Args:
        spike_times: Dictionary mapping neuron_id to list of spike times
        stimulus_times: List of stimulus onset times
        pre_window: Time before stimulus to include (ms)
        post_window: Time after stimulus to include (ms)
        bin_size: Bin width for histogram (ms)
        neuron_ids: Optional list of neuron IDs to include (default: all)
        title: Plot title
        save_path: Optional path to save figure
        
    Returns:
        Dictionary with plot data for rendering
    """
    if not spike_times or not stimulus_times:
        return {"error": "No spike times or stimulus times provided"}
    
    # Filter neurons if specified
    if neuron_ids is not None:
        spike_times = {nid: times for nid, times in spike_times.items() if nid in neuron_ids}
    
    if not spike_times:
        return {"error": "No neurons in specified list"}
    
    # Time bins relative to stimulus
    time_bins = np.arange(-pre_window, post_window + bin_size, bin_size)
    bin_centers = time_bins[:-1] + bin_size / 2
    
    # Collect spikes aligned to each stimulus
    all_aligned_spikes = []
    for stim_time in stimulus_times:
        for neuron_id, times in spike_times.items():
            for spike_time in times:
                relative_time = spike_time - stim_time
                if -pre_window <= relative_time < post_window:
                    all_aligned_spikes.append(relative_time)
    
    if not all_aligned_spikes:
        return {"error": "No spikes in PSTH window"}
    
    # Calculate histogram
    counts, _ = np.histogram(all_aligned_spikes, bins=time_bins)
    
    # Convert to firing rate (spikes per second per neuron per trial)
    num_neurons = len(spike_times)
    num_trials = len(stimulus_times)
    bin_size_sec = bin_size / 1000.0
    firing_rates = counts / (num_neurons * num_trials * bin_size_sec)
    
    plot_data = {
        "type": "psth",
        "bin_centers": bin_centers.tolist(),
        "firing_rates": firing_rates.tolist(),
        "title": title,
        "xlabel": "Time relative to stimulus (ms)",
        "ylabel": "Firing Rate (Hz)",
        "pre_window": pre_window,
        "post_window": post_window,
        "bin_size": bin_size,
        "num_neurons": num_neurons,
        "num_trials": num_trials,
    }
    
    if save_path:
        plot_data["save_path"] = save_path
    
    return plot_data


def plot_spike_train_correlation(
    spike_times_1: List[int],
    spike_times_2: List[int],
    max_lag: int = 100,
    bin_size: int = 1,
    title: str = "Cross-Correlation",
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """Calculate and plot cross-correlation between two spike trains.
    
    Cross-correlation measures temporal relationships between spike trains,
    useful for detecting synchrony and causal relationships.
    
    Args:
        spike_times_1: First spike train (list of spike times)
        spike_times_2: Second spike train (list of spike times)
        max_lag: Maximum time lag to compute (ms)
        bin_size: Bin size for spike binning (ms)
        title: Plot title
        save_path: Optional path to save figure
        
    Returns:
        Dictionary with plot data for rendering
    """
    if not spike_times_1 or not spike_times_2:
        return {"error": "Insufficient spike data"}
    
    # Determine time range
    all_times = spike_times_1 + spike_times_2
    min_time = min(all_times)
    max_time = max(all_times)
    
    # Create binned spike trains
    bins = np.arange(min_time, max_time + bin_size, bin_size)
    train_1, _ = np.histogram(spike_times_1, bins=bins)
    train_2, _ = np.histogram(spike_times_2, bins=bins)
    
    # Compute cross-correlation
    max_lag_bins = int(max_lag / bin_size)
    correlation = np.correlate(train_1 - np.mean(train_1), train_2 - np.mean(train_2), mode='full')
    
    # Extract relevant lag range
    center = len(correlation) // 2
    correlation = correlation[center - max_lag_bins:center + max_lag_bins + 1]
    lags = np.arange(-max_lag, max_lag + bin_size, bin_size)[:len(correlation)]
    
    # Normalize
    correlation = correlation / (np.std(train_1) * np.std(train_2) * len(train_1))
    
    plot_data = {
        "type": "cross_correlation",
        "lags": lags.tolist(),
        "correlation": correlation.tolist(),
        "title": title,
        "xlabel": "Time Lag (ms)",
        "ylabel": "Cross-Correlation",
        "max_lag": max_lag,
        "bin_size": bin_size,
    }
    
    if save_path:
        plot_data["save_path"] = save_path
    
    return plot_data


def plot_phase_space(
    state_variable_1: np.ndarray,
    state_variable_2: np.ndarray,
    state_variable_3: Optional[np.ndarray] = None,
    labels: Optional[Tuple[str, str, str]] = None,
    title: str = "Phase Space Plot",
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Create phase space plot for network dynamics.
    
    Phase space plots visualize the trajectory of a dynamical system by plotting
    state variables against each other. This is useful for understanding network
    behavior, identifying attractors, and detecting chaotic dynamics.
    
    Args:
        state_variable_1: First state variable (e.g., mean firing rate)
        state_variable_2: Second state variable (e.g., membrane potential)
        state_variable_3: Optional third state variable for 3D phase space
        labels: Tuple of labels for (x, y, z) axes
        title: Plot title
        save_path: Optional path to save figure
        
    Returns:
        Dictionary with plot data for rendering
        
    Examples:
        >>> # 2D phase space plot
        >>> firing_rates = np.array([0.1, 0.2, 0.3, 0.4, 0.3, 0.2])
        >>> membrane_potentials = np.array([-70, -65, -60, -55, -60, -65])
        >>> plot_data = plot_phase_space(firing_rates, membrane_potentials,
        ...                              labels=("Firing Rate", "V_m (mV)"))
        
        >>> # 3D phase space plot
        >>> synchrony = np.array([0.5, 0.6, 0.7, 0.8, 0.7, 0.6])
        >>> plot_data = plot_phase_space(firing_rates, membrane_potentials, synchrony,
        ...                              labels=("Firing Rate", "V_m", "Synchrony"))
    """
    if len(state_variable_1) != len(state_variable_2):
        return {"error": "State variables must have the same length"}
    
    if state_variable_3 is not None and len(state_variable_3) != len(state_variable_1):
        return {"error": "All state variables must have the same length"}
    
    # Set default labels
    if labels is None:
        if state_variable_3 is not None:
            labels = ("State Variable 1", "State Variable 2", "State Variable 3")
        else:
            labels = ("State Variable 1", "State Variable 2", None)
    
    is_3d = state_variable_3 is not None
    
    plot_data = {
        "type": "phase_space_3d" if is_3d else "phase_space_2d",
        "x": state_variable_1.tolist(),
        "y": state_variable_2.tolist(),
        "title": title,
        "xlabel": labels[0],
        "ylabel": labels[1],
    }
    
    if is_3d:
        plot_data["z"] = state_variable_3.tolist()
        plot_data["zlabel"] = labels[2]
        
        # Compute trajectory statistics for 3D
        trajectory_length = np.sum(np.sqrt(
            np.diff(state_variable_1)**2 + 
            np.diff(state_variable_2)**2 + 
            np.diff(state_variable_3)**2
        ))
    else:
        # Compute trajectory statistics for 2D
        trajectory_length = np.sum(np.sqrt(
            np.diff(state_variable_1)**2 + 
            np.diff(state_variable_2)**2
        ))
    
    plot_data["trajectory_length"] = float(trajectory_length)
    plot_data["num_points"] = len(state_variable_1)
    
    # Compute phase space metrics
    # Check for fixed points (where trajectory stays in small region)
    if len(state_variable_1) > 10:
        # Use sliding window to detect regions of low velocity
        window_size = min(10, len(state_variable_1) // 5)
        velocities = []
        
        for i in range(len(state_variable_1) - window_size):
            window_1 = state_variable_1[i:i+window_size]
            window_2 = state_variable_2[i:i+window_size]
            
            # Compute variance in window (low variance = potential fixed point)
            variance = np.var(window_1) + np.var(window_2)
            velocities.append(variance)
        
        plot_data["min_velocity_variance"] = float(np.min(velocities)) if velocities else 0.0
        plot_data["mean_velocity_variance"] = float(np.mean(velocities)) if velocities else 0.0
    
    if save_path:
        plot_data["save_path"] = save_path
    
    return plot_data


def plot_network_motifs(
    motif_counts: Dict[str, int],
    total_triplets: int,
    title: str = "Network Motif Distribution",
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Visualize distribution of network motifs.
    
    Network motifs are recurring patterns of connectivity that appear more frequently
    than expected by chance. This function visualizes the distribution of different
    motif types (e.g., feedforward, feedback, reciprocal connections).
    
    Args:
        motif_counts: Dictionary mapping motif names to their counts
        total_triplets: Total number of three-neuron subgraphs analyzed
        title: Plot title
        save_path: Optional path to save figure
        
    Returns:
        Dictionary with plot data for rendering
        
    Examples:
        >>> motifs = {
        ...     "chain": 150,
        ...     "convergent": 80,
        ...     "divergent": 75,
        ...     "feedback": 45,
        ...     "reciprocal": 30,
        ... }
        >>> plot_data = plot_network_motifs(motifs, 1000)
    """
    if not motif_counts:
        return {"error": "No motif data provided"}
    
    motif_names = list(motif_counts.keys())
    counts = list(motif_counts.values())
    
    # Calculate frequencies
    frequencies = [count / total_triplets * 100 if total_triplets > 0 else 0 
                   for count in counts]
    
    # Calculate z-scores (assuming random network as null model)
    # For a random network, expected frequency is approximately uniform
    expected_freq = 100.0 / len(motif_counts) if motif_counts else 0
    z_scores = [(freq - expected_freq) / np.sqrt(expected_freq) 
                for freq in frequencies]
    
    plot_data = {
        "type": "network_motifs",
        "motif_names": motif_names,
        "counts": counts,
        "frequencies": frequencies,
        "z_scores": z_scores,
        "total_triplets": total_triplets,
        "title": title,
        "xlabel": "Motif Type",
        "ylabel": "Count",
    }
    
    if save_path:
        plot_data["save_path"] = save_path
    
    return plot_data
