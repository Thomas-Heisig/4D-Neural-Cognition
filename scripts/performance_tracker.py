#!/usr/bin/env python3
"""Performance tracking and regression detection for neural simulations.

This script tracks key performance metrics across different model sizes
and configurations to detect performance regressions.
"""

import sys
import os
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.brain_model import BrainModel
from src.simulation import Simulation


class PerformanceMetrics:
    """Container for performance metrics."""
    
    def __init__(self):
        """Initialize metrics."""
        self.neurons_per_second: float = 0.0
        self.memory_per_neuron: float = 0.0
        self.synapses_per_second: float = 0.0
        self.steps_per_second: float = 0.0
        self.initialization_time: float = 0.0
        self.simulation_time: float = 0.0
        self.memory_usage_mb: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'neurons_per_second': self.neurons_per_second,
            'memory_per_neuron': self.memory_per_neuron,
            'synapses_per_second': self.synapses_per_second,
            'steps_per_second': self.steps_per_second,
            'initialization_time': self.initialization_time,
            'simulation_time': self.simulation_time,
            'memory_usage_mb': self.memory_usage_mb
        }


class PerformanceTracker:
    """Track and analyze performance metrics."""
    
    def __init__(self, results_file: str = 'performance_results.json'):
        """Initialize tracker.
        
        Args:
            results_file: Path to results file
        """
        self.results_file = results_file
        self.history: List[Dict] = []
        self._load_history()
    
    def _load_history(self):
        """Load performance history from file."""
        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, 'r') as f:
                    self.history = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load history: {e}")
                self.history = []
    
    def _save_history(self):
        """Save performance history to file."""
        try:
            with open(self.results_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save history: {e}")
    
    def benchmark_configuration(
        self,
        num_neurons: int,
        connection_prob: float,
        num_steps: int = 100
    ) -> PerformanceMetrics:
        """Benchmark a specific configuration.
        
        Args:
            num_neurons: Number of neurons
            connection_prob: Connection probability
            num_steps: Number of simulation steps
            
        Returns:
            Performance metrics
        """
        print(f"\nBenchmarking: {num_neurons} neurons, {connection_prob:.3f} connectivity")
        
        metrics = PerformanceMetrics()
        
        # Create minimal config
        config = {
            'lattice_shape': [10, 10, 10, 10],
            'neuron_model': {
                'tau_m': 10.0,
                'v_rest': -65.0,
                'v_reset': -65.0,
                'v_threshold': -50.0
            },
            'cell_lifecycle': {
                'max_age': 10000,
                'health_decay_rate': 0.01,
                'reproduction_threshold': 0.8,
                'death_threshold': 0.2
            },
            'plasticity': {
                'learning_rate': 0.01,
                'weight_decay': 0.0001,
                'min_weight': -1.0,
                'max_weight': 1.0
            },
            'senses': {},
            'areas': {}
        }
        
        # Initialization
        print("  Initializing model...")
        start_time = time.time()
        
        model = BrainModel(config=config)
        sim = Simulation(model, seed=42)
        
        # Add neurons
        for i in range(num_neurons):
            x = i % 10
            y = (i // 10) % 10
            z = (i // 100) % 10
            w = (i // 1000) % 10
            model.add_neuron(x, y, z, w)
        
        # Add synapses
        num_synapses = int(num_neurons * (num_neurons - 1) * connection_prob)
        neuron_ids = list(model.neurons.keys())
        
        for _ in range(num_synapses):
            pre_id = np.random.choice(neuron_ids)
            post_id = np.random.choice(neuron_ids)
            if pre_id != post_id:
                model.add_synapse(pre_id, post_id, weight=np.random.randn() * 0.1)
        
        init_time = time.time() - start_time
        metrics.initialization_time = init_time
        print(f"  Initialization: {init_time:.3f}s")
        
        # Simulation
        print(f"  Running {num_steps} simulation steps...")
        start_time = time.time()
        
        for step in range(num_steps):
            sim.step()
        
        sim_time = time.time() - start_time
        metrics.simulation_time = sim_time
        print(f"  Simulation: {sim_time:.3f}s")
        
        # Calculate derived metrics
        metrics.neurons_per_second = num_neurons / sim_time if sim_time > 0 else 0
        metrics.synapses_per_second = len(model.synapses) / sim_time if sim_time > 0 else 0
        metrics.steps_per_second = num_steps / sim_time if sim_time > 0 else 0
        
        # Estimate memory usage (rough approximation)
        bytes_per_neuron = 200  # Approximate
        bytes_per_synapse = 50   # Approximate
        total_bytes = (num_neurons * bytes_per_neuron + 
                      len(model.synapses) * bytes_per_synapse)
        metrics.memory_usage_mb = total_bytes / (1024 * 1024)
        metrics.memory_per_neuron = total_bytes / num_neurons if num_neurons > 0 else 0
        
        print(f"  Performance:")
        print(f"    Neurons/sec: {metrics.neurons_per_second:.0f}")
        print(f"    Synapses/sec: {metrics.synapses_per_second:.0f}")
        print(f"    Steps/sec: {metrics.steps_per_second:.1f}")
        print(f"    Memory: {metrics.memory_usage_mb:.2f} MB")
        
        return metrics
    
    def run_benchmark_suite(self):
        """Run comprehensive benchmark suite."""
        print("="*60)
        print("PERFORMANCE BENCHMARK SUITE")
        print("="*60)
        
        # Benchmark configurations
        configs = [
            (100, 0.1, 100),     # Small network
            (500, 0.05, 100),    # Medium network
            (1000, 0.02, 100),   # Large network
            (2000, 0.01, 50),    # Very large network (fewer steps)
        ]
        
        results = []
        
        for num_neurons, conn_prob, num_steps in configs:
            try:
                metrics = self.benchmark_configuration(
                    num_neurons, conn_prob, num_steps
                )
                
                result = {
                    'timestamp': datetime.now().isoformat(),
                    'configuration': {
                        'num_neurons': num_neurons,
                        'connection_probability': conn_prob,
                        'num_steps': num_steps
                    },
                    'metrics': metrics.to_dict()
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"  ❌ ERROR: {e}")
                continue
        
        # Save results
        self.history.extend(results)
        self._save_history()
        
        # Analyze results
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        
        if results:
            self._print_summary(results)
            self._check_regressions(results)
        
        print("\n" + "="*60 + "\n")
    
    def _print_summary(self, results: List[Dict]):
        """Print summary of results."""
        print("\nBenchmark Results:")
        print(f"{'Config':<20} {'Neurons/s':<12} {'Steps/s':<10} {'Memory (MB)':<12}")
        print("-" * 60)
        
        for result in results:
            config = result['configuration']
            metrics = result['metrics']
            
            config_str = f"{config['num_neurons']}n @ {config['connection_probability']:.2f}"
            neurons_per_s = f"{metrics['neurons_per_second']:.0f}"
            steps_per_s = f"{metrics['steps_per_second']:.1f}"
            memory = f"{metrics['memory_usage_mb']:.2f}"
            
            print(f"{config_str:<20} {neurons_per_s:<12} {steps_per_s:<10} {memory:<12}")
    
    def _check_regressions(self, current_results: List[Dict]):
        """Check for performance regressions."""
        if len(self.history) < len(current_results) + 1:
            print("\nℹ️  Not enough historical data for regression detection")
            return
        
        print("\n🔍 Checking for performance regressions...")
        
        # Compare with previous run
        prev_start = len(self.history) - 2 * len(current_results)
        if prev_start < 0:
            return
        
        prev_results = self.history[prev_start:prev_start + len(current_results)]
        
        regressions = []
        improvements = []
        
        for curr, prev in zip(current_results, prev_results):
            curr_metrics = curr['metrics']
            prev_metrics = prev['metrics']
            
            # Compare neurons/second
            curr_nps = curr_metrics['neurons_per_second']
            prev_nps = prev_metrics['neurons_per_second']
            
            if prev_nps > 0:
                change = (curr_nps - prev_nps) / prev_nps * 100
                
                config = curr['configuration']
                config_str = f"{config['num_neurons']}n"
                
                if change < -10:  # More than 10% slower
                    regressions.append((config_str, change))
                elif change > 10:  # More than 10% faster
                    improvements.append((config_str, change))
        
        if regressions:
            print("\n⚠️  PERFORMANCE REGRESSIONS DETECTED:")
            for config, change in regressions:
                print(f"  - {config}: {abs(change):.1f}% slower")
        
        if improvements:
            print("\n✅ PERFORMANCE IMPROVEMENTS:")
            for config, change in improvements:
                print(f"  - {config}: {change:.1f}% faster")
        
        if not regressions and not improvements:
            print("  ✅ Performance stable (within ±10%)")


def main():
    """Main benchmark script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Track performance metrics and detect regressions"
    )
    parser.add_argument(
        '--results-file',
        type=str,
        default='performance_results.json',
        help='Path to results file'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick benchmark with fewer configurations'
    )
    
    args = parser.parse_args()
    
    tracker = PerformanceTracker(results_file=args.results_file)
    
    if args.quick:
        print("Running quick benchmark...")
        metrics = tracker.benchmark_configuration(100, 0.1, 50)
        print(f"\nQuick benchmark complete!")
        print(f"Neurons/sec: {metrics.neurons_per_second:.0f}")
        print(f"Steps/sec: {metrics.steps_per_second:.1f}")
    else:
        tracker.run_benchmark_suite()


if __name__ == '__main__':
    main()
