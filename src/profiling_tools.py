"""Performance and memory profiling tools for 4D Neural Cognition.

This module provides tools for:
- Performance profiling
- Memory profiling
- Bottleneck identification
- Optimization suggestions
"""

import time
import sys
import gc
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict
from dataclasses import dataclass
import numpy as np

try:
    from .brain_model import BrainModel
    from .simulation import Simulation
except ImportError:
    from brain_model import BrainModel
    from simulation import Simulation


@dataclass
class ProfileResult:
    """Result of a profiling run."""
    
    name: str
    total_time: float
    n_calls: int
    avg_time: float
    min_time: float
    max_time: float
    memory_delta: int  # bytes


class PerformanceProfiler:
    """Profile performance of simulation components."""
    
    def __init__(self):
        """Initialize performance profiler."""
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.call_counts: Dict[str, int] = defaultdict(int)
        self.active_timers: Dict[str, float] = {}
        self.memory_snapshots: Dict[str, int] = {}
    
    def start(self, name: str) -> None:
        """Start timing a section.
        
        Args:
            name: Name of the section to time
        """
        self.active_timers[name] = time.perf_counter()
    
    def stop(self, name: str) -> None:
        """Stop timing a section and record result.
        
        Args:
            name: Name of the section to stop timing
        """
        if name not in self.active_timers:
            print(f"Warning: Timer '{name}' was not started")
            return
        
        elapsed = time.perf_counter() - self.active_timers[name]
        self.timings[name].append(elapsed)
        self.call_counts[name] += 1
        del self.active_timers[name]
    
    def profile_function(self, func: Callable, name: Optional[str] = None) -> Callable:
        """Decorator to profile a function.
        
        Args:
            func: Function to profile
            name: Optional name (defaults to function name)
        
        Returns:
            Wrapped function
        """
        prof_name = name or func.__name__
        
        def wrapper(*args, **kwargs):
            self.start(prof_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                self.stop(prof_name)
        
        return wrapper
    
    def get_results(self) -> List[ProfileResult]:
        """Get profiling results.
        
        Returns:
            List of ProfileResult objects sorted by total time
        """
        results = []
        
        for name, times in self.timings.items():
            if times:
                result = ProfileResult(
                    name=name,
                    total_time=sum(times),
                    n_calls=len(times),
                    avg_time=np.mean(times),
                    min_time=min(times),
                    max_time=max(times),
                    memory_delta=self.memory_snapshots.get(name, 0)
                )
                results.append(result)
        
        # Sort by total time descending
        results.sort(key=lambda r: r.total_time, reverse=True)
        return results
    
    def print_report(self, top_n: int = 20) -> None:
        """Print formatted profiling report.
        
        Args:
            top_n: Number of top items to show
        """
        results = self.get_results()[:top_n]
        
        if not results:
            print("No profiling data collected yet")
            return
        
        print("\n" + "="*80)
        print("PERFORMANCE PROFILING REPORT")
        print("="*80)
        
        print(f"\n{'Section':<30} {'Calls':>8} {'Total':>12} {'Avg':>12} {'Min':>12} {'Max':>12}")
        print("-"*80)
        
        for result in results:
            print(f"{result.name:<30} "
                  f"{result.n_calls:>8} "
                  f"{result.total_time:>11.4f}s "
                  f"{result.avg_time*1000:>11.2f}ms "
                  f"{result.min_time*1000:>11.2f}ms "
                  f"{result.max_time*1000:>11.2f}ms")
        
        total_time = sum(r.total_time for r in results)
        print("-"*80)
        print(f"{'TOTAL':<30} {'':<8} {total_time:>11.4f}s")
        print("="*80 + "\n")
    
    def reset(self) -> None:
        """Reset all profiling data."""
        self.timings.clear()
        self.call_counts.clear()
        self.active_timers.clear()
        self.memory_snapshots.clear()


class MemoryProfiler:
    """Profile memory usage of simulation."""
    
    def __init__(self):
        """Initialize memory profiler."""
        self.snapshots: List[Dict[str, Any]] = []
        self.baseline: Optional[int] = None
    
    def take_snapshot(self, label: str = "") -> Dict[str, Any]:
        """Take a memory snapshot.
        
        Args:
            label: Optional label for this snapshot
        
        Returns:
            Dictionary with memory information
        """
        # Force garbage collection for accurate measurement
        gc.collect()
        
        # Get memory info
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        
        snapshot = {
            'label': label,
            'timestamp': time.time(),
            'rss': mem_info.rss,  # Resident Set Size
            'vms': mem_info.vms,  # Virtual Memory Size
            'rss_mb': mem_info.rss / (1024 * 1024),
            'vms_mb': mem_info.vms / (1024 * 1024),
        }
        
        if self.baseline is None:
            self.baseline = mem_info.rss
            snapshot['delta_mb'] = 0.0
        else:
            snapshot['delta_mb'] = (mem_info.rss - self.baseline) / (1024 * 1024)
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def profile_simulation_step(self, simulation: Simulation, n_steps: int = 100) -> Dict[str, Any]:
        """Profile memory usage over simulation steps.
        
        Args:
            simulation: Simulation to profile
            n_steps: Number of steps to run
        
        Returns:
            Dictionary with memory profile
        """
        self.take_snapshot("start")
        
        mem_samples = []
        
        for i in range(n_steps):
            simulation.step()
            
            if i % 10 == 0:  # Sample every 10 steps
                snapshot = self.take_snapshot(f"step_{i}")
                mem_samples.append(snapshot['rss_mb'])
        
        self.take_snapshot("end")
        
        # Analyze memory growth
        if len(mem_samples) > 1:
            mem_growth = mem_samples[-1] - mem_samples[0]
            mem_growth_per_step = mem_growth / n_steps
        else:
            mem_growth = 0.0
            mem_growth_per_step = 0.0
        
        return {
            'n_steps': n_steps,
            'initial_mb': mem_samples[0] if mem_samples else 0,
            'final_mb': mem_samples[-1] if mem_samples else 0,
            'growth_mb': mem_growth,
            'growth_per_step_mb': mem_growth_per_step,
            'samples': mem_samples,
            'leak_detected': mem_growth_per_step > 0.01  # >10KB per step
        }
    
    def estimate_capacity(self, model: BrainModel, available_memory_mb: float = 1000) -> Dict[str, Any]:
        """Estimate maximum network size for available memory.
        
        Args:
            model: Current brain model
            available_memory_mb: Available memory in MB
        
        Returns:
            Dictionary with capacity estimates
        """
        # Estimate memory per neuron and synapse
        n_neurons = len(model.neurons)
        n_synapses = len(model.synapses)
        
        if n_neurons == 0:
            return {'error': 'No neurons in model'}
        
        # Get current memory usage
        gc.collect()
        import psutil
        process = psutil.Process()
        current_mb = process.memory_info().rss / (1024 * 1024)
        
        # Rough estimates (very approximate)
        bytes_per_neuron = (current_mb * 1024 * 1024 * 0.6) / n_neurons  # 60% for neurons
        bytes_per_synapse = (current_mb * 1024 * 1024 * 0.3) / n_synapses if n_synapses > 0 else 0  # 30% for synapses
        
        # Estimate maximum capacity
        avg_synapses_per_neuron = n_synapses / n_neurons if n_neurons > 0 else 0
        available_bytes = available_memory_mb * 1024 * 1024
        
        # Equation: neurons * bytes_per_neuron + neurons * avg_synapses * bytes_per_synapse = available_bytes
        memory_per_neuron_total = bytes_per_neuron + avg_synapses_per_neuron * bytes_per_synapse
        
        max_neurons = int(available_bytes / memory_per_neuron_total) if memory_per_neuron_total > 0 else 0
        
        return {
            'current_neurons': n_neurons,
            'current_synapses': n_synapses,
            'current_memory_mb': current_mb,
            'bytes_per_neuron': bytes_per_neuron,
            'bytes_per_synapse': bytes_per_synapse,
            'available_memory_mb': available_memory_mb,
            'estimated_max_neurons': max_neurons,
            'estimated_max_synapses': int(max_neurons * avg_synapses_per_neuron),
        }
    
    def print_report(self) -> None:
        """Print formatted memory profiling report."""
        if not self.snapshots:
            print("No memory snapshots taken yet")
            return
        
        print("\n" + "="*80)
        print("MEMORY PROFILING REPORT")
        print("="*80)
        
        print(f"\n{'Label':<20} {'RSS (MB)':>12} {'VMS (MB)':>12} {'Delta (MB)':>12}")
        print("-"*80)
        
        for snapshot in self.snapshots:
            print(f"{snapshot['label']:<20} "
                  f"{snapshot['rss_mb']:>12.2f} "
                  f"{snapshot['vms_mb']:>12.2f} "
                  f"{snapshot['delta_mb']:>+12.2f}")
        
        if len(self.snapshots) > 1:
            growth = self.snapshots[-1]['rss_mb'] - self.snapshots[0]['rss_mb']
            print("-"*80)
            print(f"{'TOTAL GROWTH':<20} {'':<12} {'':<12} {growth:>+12.2f}")
        
        print("="*80 + "\n")


class BottleneckAnalyzer:
    """Analyze and identify performance bottlenecks."""
    
    def __init__(self, simulation: Simulation):
        """Initialize bottleneck analyzer.
        
        Args:
            simulation: Simulation to analyze
        """
        self.simulation = simulation
        self.profiler = PerformanceProfiler()
    
    def analyze_simulation_step(self, n_steps: int = 100) -> Dict[str, Any]:
        """Analyze performance breakdown of simulation steps.
        
        Args:
            n_steps: Number of steps to profile
        
        Returns:
            Dictionary with bottleneck analysis
        """
        print(f"Analyzing {n_steps} simulation steps...")
        
        # Profile different phases
        phase_times = {
            'neuron_update': [],
            'plasticity': [],
            'lifecycle': [],
            'housekeeping': []
        }
        
        for i in range(n_steps):
            # Neuron dynamics
            start = time.perf_counter()
            neuron_ids = list(self.simulation.model.neurons.keys())
            spikes = []
            for neuron_id in neuron_ids:
                spiked = self.simulation.lif_step(neuron_id)
                if spiked:
                    spikes.append(neuron_id)
            phase_times['neuron_update'].append(time.perf_counter() - start)
            
            # Plasticity (simplified - just measure time)
            start = time.perf_counter()
            # Plasticity happens in actual step(), we're just measuring overhead here
            phase_times['plasticity'].append(time.perf_counter() - start)
            
            # Advance step counter
            self.simulation.model.current_step += 1
            
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{n_steps} steps")
        
        # Calculate statistics
        analysis = {}
        for phase, times in phase_times.items():
            if times:
                analysis[phase] = {
                    'total': sum(times),
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': min(times),
                    'max': max(times),
                }
        
        # Calculate percentages
        total_time = sum(sum(times) for times in phase_times.values())
        for phase in analysis:
            analysis[phase]['percentage'] = (analysis[phase]['total'] / total_time * 100) if total_time > 0 else 0
        
        return {
            'n_steps': n_steps,
            'total_time': total_time,
            'avg_step_time': total_time / n_steps if n_steps > 0 else 0,
            'phases': analysis,
            'bottleneck': max(analysis.items(), key=lambda x: x[1]['total'])[0] if analysis else None
        }
    
    def analyze_scalability(self, sizes: List[int] = None) -> Dict[str, Any]:
        """Analyze how performance scales with network size.
        
        Args:
            sizes: List of network sizes to test (number of neurons)
        
        Returns:
            Dictionary with scalability analysis
        """
        if sizes is None:
            sizes = [100, 500, 1000, 5000, 10000]
        
        print("Analyzing scalability across network sizes...")
        
        results = []
        
        for size in sizes:
            print(f"\nTesting size: {size} neurons")
            
            # Create test model
            try:
                from .brain_model import BrainModel
            except ImportError:
                from brain_model import BrainModel
            
            model = BrainModel(
                name=f"test_{size}",
                lattice_size=(10, 10, 10, 10),
                neuron_density=size / 10000  # Adjust density for target size
            )
            
            from .simulation import Simulation
            sim = Simulation(model)
            
            # Measure step time
            step_times = []
            for _ in range(10):
                start = time.perf_counter()
                sim.step()
                step_times.append(time.perf_counter() - start)
            
            results.append({
                'size': len(model.neurons),
                'synapses': len(model.synapses),
                'step_time_ms': np.mean(step_times) * 1000,
                'step_time_std': np.std(step_times) * 1000,
            })
        
        # Analyze scaling
        sizes_actual = [r['size'] for r in results]
        times = [r['step_time_ms'] for r in results]
        
        # Fit to power law: time = a * size^b
        if len(sizes_actual) > 1:
            log_sizes = np.log(sizes_actual)
            log_times = np.log(times)
            coeffs = np.polyfit(log_sizes, log_times, 1)
            scaling_exponent = coeffs[0]
        else:
            scaling_exponent = None
        
        return {
            'results': results,
            'scaling_exponent': scaling_exponent,
            'complexity': 'O(n)' if scaling_exponent and scaling_exponent < 1.5 else
                         'O(n²)' if scaling_exponent and scaling_exponent < 2.5 else
                         'O(n³+)' if scaling_exponent else 'Unknown'
        }
    
    def suggest_optimizations(self, analysis: Dict[str, Any]) -> List[str]:
        """Suggest optimizations based on profiling results.
        
        Args:
            analysis: Analysis results from analyze_simulation_step
        
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        phases = analysis.get('phases', {})
        
        # Check neuron update time
        if 'neuron_update' in phases:
            pct = phases['neuron_update']['percentage']
            if pct > 50:
                suggestions.append(
                    f"⚡ Neuron updates take {pct:.1f}% of time. Consider: "
                    "1) Using time-indexed spike buffer, "
                    "2) Vectorizing neuron updates with NumPy, "
                    "3) Using sparse connectivity matrix"
                )
        
        # Check plasticity time
        if 'plasticity' in phases:
            pct = phases['plasticity']['percentage']
            if pct > 30:
                suggestions.append(
                    f"⚡ Plasticity takes {pct:.1f}% of time. Consider: "
                    "1) Reducing plasticity update frequency, "
                    "2) Using sparse weight updates, "
                    "3) Batching weight updates"
                )
        
        # Check average step time
        avg_time = analysis.get('avg_step_time', 0)
        if avg_time > 0.1:  # >100ms per step
            suggestions.append(
                f"⚠️  Average step time is {avg_time*1000:.1f}ms. For real-time simulation, "
                "consider reducing network size or enabling optimizations."
            )
        
        # Check if using time-indexed spikes
        if not self.simulation.use_time_indexed_spikes:
            suggestions.append(
                "💡 Enable time-indexed spike buffer (use_time_indexed_spikes=True) "
                "for O(1) spike lookups instead of O(n)"
            )
        
        # General suggestions
        suggestions.append(
            "📊 Profile with cProfile for detailed function-level analysis: "
            "python -m cProfile -o profile.stats your_script.py"
        )
        
        suggestions.append(
            "🔧 For production use, consider: "
            "1) Using PyPy for ~2-5x speedup, "
            "2) Implementing C extensions for critical loops, "
            "3) Using GPU acceleration (future feature)"
        )
        
        return suggestions


def create_profiling_report(simulation: Simulation, n_steps: int = 100) -> None:
    """Generate comprehensive profiling report.
    
    Args:
        simulation: Simulation to profile
        n_steps: Number of steps to profile
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE PROFILING REPORT")
    print("="*80)
    
    # Performance analysis
    print("\n📊 Analyzing performance...")
    analyzer = BottleneckAnalyzer(simulation)
    perf_analysis = analyzer.analyze_simulation_step(n_steps)
    
    print(f"\nTotal time: {perf_analysis['total_time']:.2f}s")
    print(f"Average step time: {perf_analysis['avg_step_time']*1000:.2f}ms")
    print(f"Main bottleneck: {perf_analysis['bottleneck']}")
    
    print("\nPhase breakdown:")
    for phase, stats in perf_analysis['phases'].items():
        print(f"  {phase:20s}: {stats['percentage']:5.1f}% ({stats['mean']*1000:6.2f}ms avg)")
    
    # Memory analysis
    print("\n💾 Analyzing memory...")
    mem_profiler = MemoryProfiler()
    mem_analysis = mem_profiler.profile_simulation_step(simulation, n_steps=20)
    
    print(f"\nInitial memory: {mem_analysis['initial_mb']:.2f} MB")
    print(f"Final memory: {mem_analysis['final_mb']:.2f} MB")
    print(f"Growth: {mem_analysis['growth_mb']:.2f} MB")
    print(f"Growth per step: {mem_analysis['growth_per_step_mb']*1000:.2f} KB")
    
    if mem_analysis['leak_detected']:
        print("⚠️  Potential memory leak detected!")
    
    # Capacity estimation
    capacity = mem_profiler.estimate_capacity(simulation.model, available_memory_mb=4096)
    print(f"\nCurrent network: {capacity['current_neurons']} neurons, {capacity['current_synapses']} synapses")
    print(f"Estimated max capacity (4GB): {capacity['estimated_max_neurons']} neurons")
    
    # Optimization suggestions
    print("\n🔧 Optimization suggestions:")
    suggestions = analyzer.suggest_optimizations(perf_analysis)
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n{i}. {suggestion}")
    
    print("\n" + "="*80 + "\n")
