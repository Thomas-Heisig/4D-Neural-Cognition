"""Debugging and inspection tools for 4D Neural Cognition.

This module provides tools for:
- Neuron state inspection
- Synapse tracing
- Activity monitoring
- Simulation debugging
"""

import numpy as np
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import defaultdict, deque

try:
    from .brain_model import BrainModel, Neuron, Synapse
    from .simulation import Simulation
except ImportError:
    from brain_model import BrainModel, Neuron, Synapse
    from simulation import Simulation


class NeuronInspector:
    """Interactive neuron state inspector for debugging."""
    
    def __init__(self, model: BrainModel):
        """Initialize neuron inspector.
        
        Args:
            model: Brain model to inspect
        """
        self.model = model
        self.watch_list: Set[int] = set()
        self.history: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        self.max_history = 1000  # Keep last 1000 timesteps
    
    def add_watch(self, neuron_id: int) -> None:
        """Add neuron to watch list for detailed tracking.
        
        Args:
            neuron_id: ID of neuron to watch
        """
        if neuron_id not in self.model.neurons:
            raise ValueError(f"Neuron {neuron_id} not found in model")
        self.watch_list.add(neuron_id)
    
    def remove_watch(self, neuron_id: int) -> None:
        """Remove neuron from watch list.
        
        Args:
            neuron_id: ID of neuron to stop watching
        """
        self.watch_list.discard(neuron_id)
    
    def clear_watches(self) -> None:
        """Clear all watches."""
        self.watch_list.clear()
    
    def record_state(self, neuron_id: int) -> None:
        """Record current state of a watched neuron.
        
        Args:
            neuron_id: ID of neuron to record
        """
        if neuron_id not in self.watch_list:
            return
        
        neuron = self.model.neurons.get(neuron_id)
        if neuron is None:
            return
        
        state = {
            'step': self.model.current_step,
            'v_membrane': neuron.v_membrane,
            'external_input': neuron.external_input,
            'last_spike_time': neuron.last_spike_time,
            'age': neuron.age,
            'health': neuron.health,
            'position': neuron.position,
            'area': neuron.area
        }
        
        self.history[neuron_id].append(state)
        
        # Trim history if too long
        if len(self.history[neuron_id]) > self.max_history:
            self.history[neuron_id] = self.history[neuron_id][-self.max_history:]
    
    def inspect(self, neuron_id: int) -> Dict[str, Any]:
        """Get detailed inspection report for a neuron.
        
        Args:
            neuron_id: ID of neuron to inspect
        
        Returns:
            Dictionary with neuron details and connectivity
        """
        neuron = self.model.neurons.get(neuron_id)
        if neuron is None:
            return {'error': f'Neuron {neuron_id} not found'}
        
        # Get connectivity info
        incoming = self.model.get_synapses_for_neuron(neuron_id, direction='post')
        outgoing = self.model.get_synapses_for_neuron(neuron_id, direction='pre')
        
        # Calculate statistics
        incoming_weights = [s.weight for s in incoming]
        outgoing_weights = [s.weight for s in outgoing]
        
        report = {
            'id': neuron_id,
            'state': {
                'v_membrane': neuron.v_membrane,
                'external_input': neuron.external_input,
                'last_spike_time': neuron.last_spike_time,
                'time_since_spike': self.model.current_step - neuron.last_spike_time,
            },
            'lifecycle': {
                'age': neuron.age,
                'health': neuron.health,
                'alive': neuron.health > 0,
            },
            'position': {
                'x': neuron.position[0],
                'y': neuron.position[1],
                'z': neuron.position[2],
                'w': neuron.position[3],
                'area': neuron.area,
            },
            'connectivity': {
                'n_incoming': len(incoming),
                'n_outgoing': len(outgoing),
                'incoming_weights': {
                    'mean': float(np.mean(incoming_weights)) if incoming_weights else 0.0,
                    'std': float(np.std(incoming_weights)) if incoming_weights else 0.0,
                    'min': float(np.min(incoming_weights)) if incoming_weights else 0.0,
                    'max': float(np.max(incoming_weights)) if incoming_weights else 0.0,
                },
                'outgoing_weights': {
                    'mean': float(np.mean(outgoing_weights)) if outgoing_weights else 0.0,
                    'std': float(np.std(outgoing_weights)) if outgoing_weights else 0.0,
                    'min': float(np.min(outgoing_weights)) if outgoing_weights else 0.0,
                    'max': float(np.max(outgoing_weights)) if outgoing_weights else 0.0,
                },
            },
            'parameters': neuron.params,
        }
        
        # Add history if available
        if neuron_id in self.history:
            history = self.history[neuron_id]
            if history:
                report['history'] = {
                    'length': len(history),
                    'v_membrane': [h['v_membrane'] for h in history[-100:]],
                    'spike_times': [h['step'] for h in history 
                                   if h['step'] == h['last_spike_time']],
                }
        
        return report
    
    def print_inspection(self, neuron_id: int) -> None:
        """Print formatted inspection report.
        
        Args:
            neuron_id: ID of neuron to inspect
        """
        report = self.inspect(neuron_id)
        
        if 'error' in report:
            print(f"ERROR: {report['error']}")
            return
        
        print(f"\n{'='*60}")
        print(f"NEURON INSPECTION: ID {neuron_id}")
        print(f"{'='*60}")
        
        print(f"\nSTATE:")
        print(f"  Membrane potential: {report['state']['v_membrane']:.2f} mV")
        print(f"  External input: {report['state']['external_input']:.2f}")
        print(f"  Last spike: step {report['state']['last_spike_time']}")
        print(f"  Time since spike: {report['state']['time_since_spike']} steps")
        
        print(f"\nLIFECYCLE:")
        print(f"  Age: {report['lifecycle']['age']} steps")
        print(f"  Health: {report['lifecycle']['health']:.2f}")
        print(f"  Status: {'Alive' if report['lifecycle']['alive'] else 'Dead'}")
        
        print(f"\nPOSITION:")
        pos = report['position']
        print(f"  4D coordinates: ({pos['x']:.2f}, {pos['y']:.2f}, {pos['z']:.2f}, {pos['w']:.2f})")
        print(f"  Brain area: {pos['area']}")
        
        print(f"\nCONNECTIVITY:")
        conn = report['connectivity']
        print(f"  Incoming synapses: {conn['n_incoming']}")
        if conn['n_incoming'] > 0:
            print(f"    Weight mean±std: {conn['incoming_weights']['mean']:.3f}±{conn['incoming_weights']['std']:.3f}")
            print(f"    Weight range: [{conn['incoming_weights']['min']:.3f}, {conn['incoming_weights']['max']:.3f}]")
        print(f"  Outgoing synapses: {conn['n_outgoing']}")
        if conn['n_outgoing'] > 0:
            print(f"    Weight mean±std: {conn['outgoing_weights']['mean']:.3f}±{conn['outgoing_weights']['std']:.3f}")
            print(f"    Weight range: [{conn['outgoing_weights']['min']:.3f}, {conn['outgoing_weights']['max']:.3f}]")
        
        if 'history' in report:
            print(f"\nHISTORY:")
            hist = report['history']
            print(f"  Recorded timesteps: {hist['length']}")
            if hist['spike_times']:
                print(f"  Recent spikes: {hist['spike_times'][-10:]}")
                firing_rate = len(hist['spike_times']) / hist['length']
                print(f"  Firing rate: {firing_rate:.3f} spikes/step")
        
        print(f"\n{'='*60}\n")
    
    def compare_neurons(self, neuron_ids: List[int]) -> Dict[str, Any]:
        """Compare multiple neurons side by side.
        
        Args:
            neuron_ids: List of neuron IDs to compare
        
        Returns:
            Comparison dictionary
        """
        comparisons = {}
        for nid in neuron_ids:
            comparisons[nid] = self.inspect(nid)
        
        return {
            'neurons': comparisons,
            'summary': {
                'n_neurons': len(neuron_ids),
                'avg_health': float(np.mean([c['lifecycle']['health'] 
                                           for c in comparisons.values() 
                                           if 'error' not in c])),
                'avg_incoming': float(np.mean([c['connectivity']['n_incoming'] 
                                              for c in comparisons.values() 
                                              if 'error' not in c])),
                'avg_outgoing': float(np.mean([c['connectivity']['n_outgoing'] 
                                              for c in comparisons.values() 
                                              if 'error' not in c])),
            }
        }


class SynapseTracer:
    """Trace synaptic pathways and signal flow through network."""
    
    def __init__(self, model: BrainModel):
        """Initialize synapse tracer.
        
        Args:
            model: Brain model to trace
        """
        self.model = model
        self._adjacency_cache: Optional[Dict[int, List[int]]] = None
    
    def _build_adjacency(self) -> None:
        """Build adjacency list for fast path finding."""
        self._adjacency_cache = defaultdict(list)
        for synapse in self.model.synapses:
            self._adjacency_cache[synapse.pre_id].append(synapse.post_id)
    
    def trace_forward(self, start_neuron: int, max_depth: int = 5) -> Dict[str, Any]:
        """Trace forward connections from a neuron.
        
        Uses breadth-first search to find all neurons reachable
        from the starting neuron.
        
        Args:
            start_neuron: Starting neuron ID
            max_depth: Maximum path length to trace
        
        Returns:
            Dictionary with path information
        """
        if self._adjacency_cache is None:
            self._build_adjacency()
        
        visited = {start_neuron}
        paths = {start_neuron: []}  # neuron_id -> path to reach it
        queue = deque([(start_neuron, 0)])  # (neuron_id, depth)
        
        by_depth = defaultdict(list)
        by_depth[0] = [start_neuron]
        
        while queue:
            current, depth = queue.popleft()
            
            if depth >= max_depth:
                continue
            
            # Get all postsynaptic neurons
            for next_neuron in self._adjacency_cache.get(current, []):
                if next_neuron not in visited:
                    visited.add(next_neuron)
                    paths[next_neuron] = paths[current] + [current]
                    by_depth[depth + 1].append(next_neuron)
                    queue.append((next_neuron, depth + 1))
        
        return {
            'start': start_neuron,
            'reachable': list(visited),
            'n_reachable': len(visited),
            'max_depth': max(by_depth.keys()),
            'by_depth': dict(by_depth),
            'paths': paths,
        }
    
    def trace_backward(self, target_neuron: int, max_depth: int = 5) -> Dict[str, Any]:
        """Trace backward connections to a neuron.
        
        Finds all neurons that can influence the target neuron.
        
        Args:
            target_neuron: Target neuron ID
            max_depth: Maximum path length to trace
        
        Returns:
            Dictionary with path information
        """
        # Build reverse adjacency
        reverse_adj = defaultdict(list)
        for synapse in self.model.synapses:
            reverse_adj[synapse.post_id].append(synapse.pre_id)
        
        visited = {target_neuron}
        paths = {target_neuron: []}
        queue = deque([(target_neuron, 0)])
        
        by_depth = defaultdict(list)
        by_depth[0] = [target_neuron]
        
        while queue:
            current, depth = queue.popleft()
            
            if depth >= max_depth:
                continue
            
            for prev_neuron in reverse_adj.get(current, []):
                if prev_neuron not in visited:
                    visited.add(prev_neuron)
                    paths[prev_neuron] = paths[current] + [current]
                    by_depth[depth + 1].append(prev_neuron)
                    queue.append((prev_neuron, depth + 1))
        
        return {
            'target': target_neuron,
            'sources': list(visited),
            'n_sources': len(visited),
            'max_depth': max(by_depth.keys()),
            'by_depth': dict(by_depth),
            'paths': paths,
        }
    
    def find_path(self, source: int, target: int, max_depth: int = 10) -> Optional[List[int]]:
        """Find shortest path between two neurons.
        
        Args:
            source: Source neuron ID
            target: Target neuron ID
            max_depth: Maximum path length to search
        
        Returns:
            List of neuron IDs in path, or None if no path exists
        """
        if self._adjacency_cache is None:
            self._build_adjacency()
        
        if source == target:
            return [source]
        
        visited = {source}
        queue = deque([(source, [source], 0)])
        
        while queue:
            current, path, depth = queue.popleft()
            
            if depth >= max_depth:
                continue
            
            for next_neuron in self._adjacency_cache.get(current, []):
                if next_neuron == target:
                    return path + [next_neuron]
                
                if next_neuron not in visited:
                    visited.add(next_neuron)
                    queue.append((next_neuron, path + [next_neuron], depth + 1))
        
        return None
    
    def trace_synapse(self, pre_id: int, post_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific synapse.
        
        Args:
            pre_id: Presynaptic neuron ID
            post_id: Postsynaptic neuron ID
        
        Returns:
            Synapse information dictionary, or None if not found
        """
        for synapse in self.model.synapses:
            if synapse.pre_id == pre_id and synapse.post_id == post_id:
                pre_neuron = self.model.neurons.get(pre_id)
                post_neuron = self.model.neurons.get(post_id)
                
                return {
                    'pre_id': pre_id,
                    'post_id': post_id,
                    'weight': synapse.weight,
                    'delay': synapse.delay,
                    'pre_area': pre_neuron.area if pre_neuron else None,
                    'post_area': post_neuron.area if post_neuron else None,
                    'distance': float(np.linalg.norm(
                        np.array(pre_neuron.position) - np.array(post_neuron.position)
                    )) if pre_neuron and post_neuron else None,
                }
        
        return None
    
    def find_loops(self, max_length: int = 5) -> List[List[int]]:
        """Find feedback loops in the network.
        
        Args:
            max_length: Maximum loop length to search
        
        Returns:
            List of loops, each represented as list of neuron IDs
        """
        if self._adjacency_cache is None:
            self._build_adjacency()
        
        loops = []
        
        # Check each neuron as potential loop start
        for start_neuron in list(self.model.neurons.keys())[:100]:  # Limit search for performance
            # Do DFS to find loops back to start
            stack = [(start_neuron, [start_neuron])]
            visited_in_path = {start_neuron}
            
            while stack:
                current, path = stack.pop()
                
                if len(path) > max_length:
                    continue
                
                for next_neuron in self._adjacency_cache.get(current, []):
                    if next_neuron == start_neuron and len(path) > 1:
                        # Found a loop!
                        loops.append(path + [next_neuron])
                    elif next_neuron not in visited_in_path and len(path) < max_length:
                        stack.append((next_neuron, path + [next_neuron]))
        
        return loops


class ActivityMonitor:
    """Monitor and log network activity over time."""
    
    def __init__(self, model: BrainModel):
        """Initialize activity monitor.
        
        Args:
            model: Brain model to monitor
        """
        self.model = model
        self.activity_log: List[Dict[str, Any]] = []
        self.alert_rules: List[Tuple[str, callable, str]] = []
        self.alerts: List[Dict[str, Any]] = []
    
    def record_step(self, stats: Dict[str, Any]) -> None:
        """Record activity from a simulation step.
        
        Args:
            stats: Statistics dictionary from simulation step
        """
        activity = {
            'step': stats['step'],
            'n_spikes': len(stats['spikes']),
            'spike_rate': len(stats['spikes']) / len(self.model.neurons) if self.model.neurons else 0,
            'deaths': stats.get('deaths', 0),
            'births': stats.get('births', 0),
            'n_neurons': len(self.model.neurons),
            'n_synapses': len(self.model.synapses),
        }
        
        self.activity_log.append(activity)
        
        # Check alert rules
        self._check_alerts(activity)
    
    def add_alert_rule(self, name: str, condition: callable, message: str) -> None:
        """Add an alert rule to monitor for problems.
        
        Args:
            name: Name of the alert
            condition: Function that takes activity dict and returns True if alert should fire
            message: Message to display when alert fires
        """
        self.alert_rules.append((name, condition, message))
    
    def _check_alerts(self, activity: Dict[str, Any]) -> None:
        """Check if any alert rules are triggered.
        
        Args:
            activity: Current activity dictionary
        """
        for name, condition, message in self.alert_rules:
            try:
                if condition(activity):
                    alert = {
                        'step': activity['step'],
                        'name': name,
                        'message': message,
                        'activity': activity.copy()
                    }
                    self.alerts.append(alert)
                    print(f"⚠️  ALERT [{name}]: {message} (step {activity['step']})")
            except Exception as e:
                print(f"Error in alert rule '{name}': {e}")
    
    def get_summary(self, last_n_steps: Optional[int] = None) -> Dict[str, Any]:
        """Get summary of monitored activity.
        
        Args:
            last_n_steps: Summarize only last N steps (None = all)
        
        Returns:
            Summary dictionary
        """
        if not self.activity_log:
            return {'error': 'No activity recorded yet'}
        
        log = self.activity_log if last_n_steps is None else self.activity_log[-last_n_steps:]
        
        spike_rates = [a['spike_rate'] for a in log]
        n_neurons = [a['n_neurons'] for a in log]
        
        return {
            'n_steps': len(log),
            'spike_rate': {
                'mean': float(np.mean(spike_rates)),
                'std': float(np.std(spike_rates)),
                'min': float(np.min(spike_rates)),
                'max': float(np.max(spike_rates)),
            },
            'network_size': {
                'neurons_mean': float(np.mean(n_neurons)),
                'neurons_current': n_neurons[-1] if n_neurons else 0,
                'synapses_current': log[-1]['n_synapses'] if log else 0,
            },
            'lifecycle': {
                'total_deaths': sum(a['deaths'] for a in log),
                'total_births': sum(a['births'] for a in log),
            },
            'alerts': {
                'n_alerts': len(self.alerts),
                'recent_alerts': self.alerts[-10:] if self.alerts else [],
            }
        }
    
    def setup_default_alerts(self) -> None:
        """Set up commonly useful alert rules."""
        # Alert on very low activity
        self.add_alert_rule(
            "low_activity",
            lambda a: a['spike_rate'] < 0.001,
            "Network activity very low - possible silent network"
        )
        
        # Alert on very high activity
        self.add_alert_rule(
            "high_activity",
            lambda a: a['spike_rate'] > 0.9,
            "Network activity very high - possible runaway excitation"
        )
        
        # Alert on sudden death spike
        self.add_alert_rule(
            "mass_death",
            lambda a: a['deaths'] > 0.1 * a['n_neurons'],
            "Sudden mass neuron death detected"
        )
        
        # Alert on network shrinkage
        self.add_alert_rule(
            "network_shrinking",
            lambda a: a['n_neurons'] < 100,
            "Network has fewer than 100 neurons"
        )
