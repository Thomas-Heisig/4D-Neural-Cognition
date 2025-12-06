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
