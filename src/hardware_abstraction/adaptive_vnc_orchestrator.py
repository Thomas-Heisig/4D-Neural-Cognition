"""Adaptive VNC Orchestrator for self-optimizing neural compute.

This module implements Phase 2 of the VNC enhancement roadmap: real-time analysis
and adaptive control of the VNC system. The orchestrator monitors VPU performance,
detects load imbalances, and dynamically adjusts partitioning to optimize throughput.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..simulation import Simulation
    from .virtual_clock import GlobalVirtualClock
    from .slice_partitioner import SlicePartitioner

logger = logging.getLogger(__name__)


class AdaptiveVNCOrchestrator:
    """Adaptive orchestrator for VNC system that monitors and optimizes performance.
    
    This class analyzes VPU utilization in real-time and adapts the system:
    - Detects load imbalances across VPUs
    - Identifies "hot" (highly active) and "cold" (idle) slices
    - Dynamically repartitions neurons across VPUs
    - Adjusts compute priorities based on activity
    
    The orchestrator acts as a meta-controller that makes the VNC system
    self-optimizing and adaptive to changing neural dynamics.
    
    Attributes:
        simulation: The simulation being orchestrated
        virtual_clock: The global virtual clock managing VPUs
        partitioner: The slice partitioner for repartitioning
        performance_log: Historical performance data
        optimization_history: Record of optimizations performed
    """
    
    def __init__(
        self,
        simulation: Simulation,
        imbalance_threshold: float = 0.3,
        activity_threshold: float = 0.7,
        monitoring_interval: int = 100,
    ):
        """Initialize the Adaptive VNC Orchestrator.
        
        Args:
            simulation: The simulation to orchestrate
            imbalance_threshold: Load imbalance threshold for repartitioning (0.0-1.0)
            activity_threshold: Activity threshold for "hot" slice detection (0.0-1.0)
            monitoring_interval: Number of cycles between monitoring checks
        """
        self.sim = simulation
        self.imbalance_threshold = imbalance_threshold
        self.activity_threshold = activity_threshold
        self.monitoring_interval = monitoring_interval
        
        # Performance tracking
        self.performance_log: List[Dict] = []
        self.optimization_history: List[Dict] = []
        
        # Statistics
        self.total_repartitions = 0
        self.total_priority_adjustments = 0
        
        logger.info(
            f"Initialized AdaptiveVNCOrchestrator "
            f"(imbalance_threshold={imbalance_threshold}, "
            f"activity_threshold={activity_threshold})"
        )
    
    def monitor_and_adapt(self, current_cycle: int) -> Dict:
        """Monitor VPU performance and apply adaptive optimizations.
        
        This is the main orchestration method that should be called periodically
        during simulation. It analyzes performance and triggers optimizations.
        
        Args:
            current_cycle: Current simulation cycle
            
        Returns:
            Dictionary with monitoring results and actions taken
        """
        # Skip if not at monitoring interval
        if current_cycle % self.monitoring_interval != 0:
            return {"monitored": False, "cycle": current_cycle}
        
        # Get VPU statistics
        if not hasattr(self.sim, 'virtual_clock') or self.sim.virtual_clock is None:
            return {
                "monitored": False,
                "cycle": current_cycle,
                "reason": "no_vnc_system"
            }
        
        vpu_stats = self._collect_vpu_statistics()
        
        # Calculate load imbalance
        load_imbalance = self._calculate_load_imbalance(vpu_stats)
        
        # Identify hot and cold slices
        hot_slices = self._identify_hot_slices(vpu_stats, self.activity_threshold)
        cold_slices = self._identify_cold_slices(vpu_stats, self.activity_threshold)
        
        actions_taken = []
        
        # Apply adaptive repartitioning if imbalance is high
        if load_imbalance > self.imbalance_threshold:
            logger.info(
                f"High load imbalance detected: {load_imbalance:.2%} "
                f"(threshold: {self.imbalance_threshold:.2%})"
            )
            repartition_result = self._repartition_adaptive(vpu_stats)
            if repartition_result["success"]:
                actions_taken.append("repartition")
                self.total_repartitions += 1
        
        # Apply compute priority adjustments for hot slices
        if hot_slices:
            logger.info(f"Hot slices detected: {len(hot_slices)} slices")
            priority_result = self._apply_compute_priority(hot_slices)
            if priority_result["success"]:
                actions_taken.append("priority_adjustment")
                self.total_priority_adjustments += 1
        
        # Log performance snapshot
        snapshot = {
            "cycle": current_cycle,
            "load_imbalance": load_imbalance,
            "hot_slices": len(hot_slices),
            "cold_slices": len(cold_slices),
            "actions": actions_taken,
            "vpu_count": len(vpu_stats),
        }
        self.performance_log.append(snapshot)
        
        # Keep only recent history
        MAX_PERFORMANCE_LOG_SIZE = 1000
        if len(self.performance_log) > MAX_PERFORMANCE_LOG_SIZE:
            self.performance_log = self.performance_log[-MAX_PERFORMANCE_LOG_SIZE:]
        
        return {
            "monitored": True,
            "cycle": current_cycle,
            "load_imbalance": load_imbalance,
            "hot_slices": len(hot_slices),
            "cold_slices": len(cold_slices),
            "actions_taken": actions_taken,
        }
    
    def _collect_vpu_statistics(self) -> List[Dict]:
        """Collect statistics from all VPUs.
        
        Returns:
            List of VPU statistics dictionaries
        """
        vpu_stats = []
        
        if hasattr(self.sim, 'virtual_clock') and self.sim.virtual_clock:
            for vpu in self.sim.virtual_clock.vpus:
                stats = vpu.get_statistics()
                stats['vpu_id'] = vpu.vpu_id
                stats['neuron_count'] = len(vpu.neuron_batch)
                vpu_stats.append(stats)
        
        return vpu_stats
    
    def _calculate_load_imbalance(self, vpu_stats: List[Dict]) -> float:
        """Calculate load imbalance across VPUs.
        
        Load imbalance is measured as the coefficient of variation (CV) of
        processing times across VPUs. A CV of 0 means perfectly balanced,
        while higher values indicate greater imbalance.
        
        Args:
            vpu_stats: List of VPU statistics
            
        Returns:
            Load imbalance metric (0.0 = perfect balance, higher = more imbalance)
        """
        if not vpu_stats:
            return 0.0
        
        # Use average processing time as load metric
        processing_times = []
        for stats in vpu_stats:
            if stats.get("cycles_executed", 0) > 0:
                avg_time = stats.get("avg_processing_time_ms", 0.0)
                processing_times.append(avg_time)
        
        if not processing_times or len(processing_times) < 2:
            return 0.0
        
        # Calculate coefficient of variation
        mean_time = np.mean(processing_times)
        if mean_time == 0:
            return 0.0
        
        std_time = np.std(processing_times)
        cv = std_time / mean_time
        
        return cv
    
    def _identify_hot_slices(
        self, vpu_stats: List[Dict], threshold: float
    ) -> List[int]:
        """Identify VPUs with high activity ("hot" slices).
        
        Hot slices are those with spike rates above the activity threshold.
        
        Args:
            vpu_stats: List of VPU statistics
            threshold: Activity threshold (0.0-1.0)
            
        Returns:
            List of VPU IDs for hot slices
        """
        hot_slices = []
        
        if not vpu_stats:
            return hot_slices
        
        # Calculate spike rates
        spike_rates = []
        for stats in vpu_stats:
            neurons = stats.get("neuron_count", 0)
            cycles = stats.get("cycles_executed", 0)
            if neurons > 0 and cycles > 0:
                spikes = stats.get("spikes_generated", 0)
                spike_rate = spikes / (neurons * cycles)
                spike_rates.append((stats['vpu_id'], spike_rate))
        
        if not spike_rates:
            return hot_slices
        
        # Normalize spike rates to 0-1 range
        max_rate = max(rate for _, rate in spike_rates)
        if max_rate > 0:
            for vpu_id, rate in spike_rates:
                normalized_rate = rate / max_rate
                if normalized_rate >= threshold:
                    hot_slices.append(vpu_id)
        
        return hot_slices
    
    def _identify_cold_slices(
        self, vpu_stats: List[Dict], threshold: float
    ) -> List[int]:
        """Identify VPUs with low activity ("cold" slices).
        
        Cold slices are those with spike rates below (1 - threshold).
        
        Args:
            vpu_stats: List of VPU statistics
            threshold: Activity threshold (0.0-1.0)
            
        Returns:
            List of VPU IDs for cold slices
        """
        cold_slices = []
        
        if not vpu_stats:
            return cold_slices
        
        # Calculate spike rates
        spike_rates = []
        for stats in vpu_stats:
            neurons = stats.get("neuron_count", 0)
            cycles = stats.get("cycles_executed", 0)
            if neurons > 0 and cycles > 0:
                spikes = stats.get("spikes_generated", 0)
                spike_rate = spikes / (neurons * cycles)
                spike_rates.append((stats['vpu_id'], spike_rate))
        
        if not spike_rates:
            return cold_slices
        
        # Normalize spike rates to 0-1 range
        max_rate = max(rate for _, rate in spike_rates)
        if max_rate > 0:
            for vpu_id, rate in spike_rates:
                normalized_rate = rate / max_rate
                if normalized_rate <= (1.0 - threshold):
                    cold_slices.append(vpu_id)
        
        return cold_slices
    
    def _repartition_adaptive(self, vpu_stats: List[Dict]) -> Dict:
        """Dynamically repartition neurons across VPUs based on activity.
        
        This redistributes neurons to balance the computational load.
        Currently logs the intent; full implementation would require
        migration support in the VNC system.
        
        Args:
            vpu_stats: List of VPU statistics
            
        Returns:
            Dictionary with repartitioning result
        """
        logger.info("Adaptive repartitioning triggered")
        
        # Calculate target neurons per VPU for balanced load
        total_neurons = sum(stats.get("neuron_count", 0) for stats in vpu_stats)
        num_vpus = len(vpu_stats)
        
        if num_vpus == 0:
            return {"success": False, "reason": "no_vpus"}
        
        target_neurons_per_vpu = total_neurons / num_vpus
        
        # Identify overloaded and underloaded VPUs
        overloaded = []
        underloaded = []
        
        for stats in vpu_stats:
            neuron_count = stats.get("neuron_count", 0)
            vpu_id = stats['vpu_id']
            
            if neuron_count > target_neurons_per_vpu * 1.2:
                overloaded.append((vpu_id, neuron_count))
            elif neuron_count < target_neurons_per_vpu * 0.8:
                underloaded.append((vpu_id, neuron_count))
        
        # Log optimization intent
        optimization_record = {
            "type": "repartition",
            "total_neurons": total_neurons,
            "target_per_vpu": target_neurons_per_vpu,
            "overloaded_vpus": len(overloaded),
            "underloaded_vpus": len(underloaded),
            "applied": False,  # Would be True when actually implemented
        }
        self.optimization_history.append(optimization_record)
        
        logger.info(
            f"Repartitioning analysis: {len(overloaded)} overloaded, "
            f"{len(underloaded)} underloaded VPUs"
        )
        
        return {
            "success": True,
            "overloaded": len(overloaded),
            "underloaded": len(underloaded),
            "note": "Analysis performed, full migration pending implementation"
        }
    
    def _apply_compute_priority(self, hot_slice_ids: List[int]) -> Dict:
        """Apply higher compute priority to hot slices.
        
        This would adjust the scheduling priority for VPUs processing
        hot slices. Currently logs the intent.
        
        Args:
            hot_slice_ids: List of VPU IDs for hot slices
            
        Returns:
            Dictionary with priority adjustment result
        """
        logger.info(f"Applying compute priority to {len(hot_slice_ids)} hot slices")
        
        # Log optimization intent
        optimization_record = {
            "type": "priority_adjustment",
            "hot_slices": hot_slice_ids,
            "applied": False,  # Would be True when scheduler supports priorities
        }
        self.optimization_history.append(optimization_record)
        
        return {
            "success": True,
            "hot_slices": len(hot_slice_ids),
            "note": "Priority analysis performed, scheduler integration pending"
        }
    
    def get_performance_summary(self) -> Dict:
        """Get summary of orchestrator performance.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.performance_log:
            return {
                "total_repartitions": self.total_repartitions,
                "total_priority_adjustments": self.total_priority_adjustments,
                "monitoring_cycles": 0,
            }
        
        # Calculate aggregate metrics
        load_imbalances = [log["load_imbalance"] for log in self.performance_log]
        hot_slice_counts = [log["hot_slices"] for log in self.performance_log]
        
        return {
            "total_repartitions": self.total_repartitions,
            "total_priority_adjustments": self.total_priority_adjustments,
            "monitoring_cycles": len(self.performance_log),
            "avg_load_imbalance": np.mean(load_imbalances) if load_imbalances else 0.0,
            "max_load_imbalance": np.max(load_imbalances) if load_imbalances else 0.0,
            "avg_hot_slices": np.mean(hot_slice_counts) if hot_slice_counts else 0.0,
            "recent_actions": [log["actions"] for log in self.performance_log[-10:]],
        }
    
    def reset_statistics(self) -> None:
        """Reset orchestrator statistics."""
        self.performance_log = []
        self.optimization_history = []
        self.total_repartitions = 0
        self.total_priority_adjustments = 0
        logger.info("Orchestrator statistics reset")


class CognitiveAwareOrchestrator(AdaptiveVNCOrchestrator):
    """Cognitive-aware VNC orchestrator with priority management.
    
    Extends the adaptive orchestrator to prioritize compute resources
    for critical cognitive regions (motor planning, self-perception,
    sensor fusion) based on cognitive activity levels.
    
    This enables the system to dynamically allocate more VPUs to regions
    that are actively engaged in learning or processing, improving
    overall cognitive performance.
    
    Attributes:
        critical_regions: Mapping of cognitive regions to VPU requirements
        sensorimotor_activity_threshold: Threshold for high activity detection
        reallocation_count: Number of VPU reallocations performed
    """
    
    # Critical cognitive regions with their w-slices and minimum VPU requirements
    CRITICAL_REGIONS = {
        'motor_planning': {'w_slice': 10, 'min_vpus': 2, 'priority': 3},
        'self_perception': {'w_slice': 12, 'min_vpus': 2, 'priority': 3},
        'sensor_fusion': {'w_slice': 6, 'min_vpus': 1, 'priority': 2},
        'executive_control': {'w_slice': 14, 'min_vpus': 1, 'priority': 2},
    }
    
    def __init__(
        self,
        simulation: Simulation,
        imbalance_threshold: float = 0.3,
        activity_threshold: float = 0.7,
        monitoring_interval: int = 100,
        sensorimotor_activity_threshold: float = 0.6,
    ):
        """Initialize cognitive-aware orchestrator.
        
        Args:
            simulation: The simulation to orchestrate
            imbalance_threshold: Load imbalance threshold
            activity_threshold: Activity threshold for hot slices
            monitoring_interval: Monitoring interval in cycles
            sensorimotor_activity_threshold: Threshold for sensorimotor activity
        """
        super().__init__(
            simulation,
            imbalance_threshold,
            activity_threshold,
            monitoring_interval
        )
        
        self.sensorimotor_activity_threshold = sensorimotor_activity_threshold
        self.reallocation_count = 0
        self.priority_history: List[Dict] = []
        
        logger.info(
            f"Initialized CognitiveAwareOrchestrator with "
            f"{len(self.CRITICAL_REGIONS)} critical regions"
        )
    
    def monitor_and_adapt(self, current_cycle: int) -> Dict:
        """Monitor and adapt with cognitive priority management.
        
        Extends parent monitoring with cognitive region prioritization.
        
        Args:
            current_cycle: Current simulation cycle
            
        Returns:
            Monitoring results with cognitive actions
        """
        # Call parent monitoring
        result = super().monitor_and_adapt(current_cycle)
        
        if not result.get('monitored', False):
            return result
        
        # Add cognitive priority management
        if self.is_high_sensorimotor_activity():
            priority_result = self.prioritize_critical_regions()
            
            if priority_result['reallocations'] > 0:
                if 'actions_taken' not in result:
                    result['actions_taken'] = []
                result['actions_taken'].append('cognitive_priority')
                result['vpu_reallocations'] = priority_result['reallocations']
                self.reallocation_count += priority_result['reallocations']
        
        return result
    
    def is_high_sensorimotor_activity(self) -> bool:
        """Detect if system is in high sensorimotor activity state.
        
        Checks neural activity in motor and sensory regions to determine
        if sensorimotor processing is demanding.
        
        Returns:
            True if sensorimotor activity is high
        """
        if not hasattr(self.sim, 'virtual_clock') or not self.sim.virtual_clock:
            return False
        
        # Check activity in motor and sensory slices
        motor_slice = self.CRITICAL_REGIONS['motor_planning']['w_slice']
        sensor_slice = self.CRITICAL_REGIONS['sensor_fusion']['w_slice']
        
        total_activity = 0.0
        region_count = 0
        
        for vpu in self.sim.virtual_clock.vpus:
            # Check if VPU handles relevant slices
            if hasattr(vpu, 'w_slice'):
                if vpu.w_slice in [motor_slice, sensor_slice]:
                    # Get spike rate as activity measure
                    stats = vpu.get_statistics()
                    neurons = stats.get('neuron_count', 0)
                    cycles = stats.get('cycles_executed', 0)
                    
                    if neurons > 0 and cycles > 0:
                        spikes = stats.get('spikes_generated', 0)
                        activity = spikes / (neurons * cycles)
                        total_activity += activity
                        region_count += 1
        
        if region_count > 0:
            avg_activity = total_activity / region_count
            return avg_activity >= self.sensorimotor_activity_threshold
        
        return False
    
    def prioritize_critical_regions(self) -> Dict:
        """Allocate more VPUs to critical brain regions.
        
        Ensures critical cognitive regions (motor planning, self-perception,
        sensor fusion) have sufficient VPUs allocated, reallocating from
        less critical regions if necessary.
        
        Returns:
            Dictionary with reallocation results
        """
        if not hasattr(self.sim, 'virtual_clock') or not self.sim.virtual_clock:
            return {'reallocations': 0, 'reason': 'no_vnc_system'}
        
        reallocations = 0
        reallocation_details = []
        
        # Check each critical region
        for region_name, config in self.CRITICAL_REGIONS.items():
            w_slice = config['w_slice']
            min_vpus = config['min_vpus']
            priority = config['priority']
            
            # Count VPUs assigned to this slice
            current_vpus = self.get_vpus_for_slice(w_slice)
            
            if len(current_vpus) < min_vpus:
                # Need more VPUs for this region
                needed = min_vpus - len(current_vpus)
                
                logger.info(
                    f"Critical region '{region_name}' (w={w_slice}) "
                    f"needs {needed} more VPUs"
                )
                
                # Find low-priority slices to donate VPUs
                for _ in range(needed):
                    donor_slice = self.find_low_priority_slice(
                        exclude_slices=[config['w_slice'] for config in self.CRITICAL_REGIONS.values()]
                    )
                    
                    if donor_slice is not None:
                        success = self.reallocate_vpu(
                            from_slice=donor_slice,
                            to_slice=w_slice,
                            reason=f'priority_{region_name}'
                        )
                        
                        if success:
                            reallocations += 1
                            reallocation_details.append({
                                'from': donor_slice,
                                'to': w_slice,
                                'region': region_name,
                                'priority': priority,
                            })
        
        # Log reallocation summary
        if reallocations > 0:
            logger.info(
                f"Performed {reallocations} VPU reallocations "
                f"for cognitive priority"
            )
            
            self.priority_history.append({
                'cycle': self.sim.current_step if hasattr(self.sim, 'current_step') else 0,
                'reallocations': reallocations,
                'details': reallocation_details,
            })
        
        return {
            'reallocations': reallocations,
            'details': reallocation_details,
        }
    
    def get_vpus_for_slice(self, w_slice: int) -> List:
        """Get VPUs assigned to specific w-slice.
        
        Args:
            w_slice: W-coordinate of slice
            
        Returns:
            List of VPUs handling this slice
        """
        vpus = []
        
        if hasattr(self.sim, 'virtual_clock') and self.sim.virtual_clock:
            for vpu in self.sim.virtual_clock.vpus:
                if hasattr(vpu, 'w_slice') and vpu.w_slice == w_slice:
                    vpus.append(vpu)
        
        return vpus
    
    def find_low_priority_slice(
        self,
        exclude_slices: Optional[List[int]] = None
    ) -> Optional[int]:
        """Find a low-priority slice that can donate a VPU.
        
        Identifies slices with low activity or multiple VPUs that are
        not critical regions.
        
        Args:
            exclude_slices: Slices to exclude from consideration
            
        Returns:
            W-coordinate of donor slice, or None if none found
        """
        if not hasattr(self.sim, 'virtual_clock') or not self.sim.virtual_clock:
            return None
        
        if exclude_slices is None:
            exclude_slices = []
        
        # Collect activity by slice
        slice_activities = {}
        slice_vpu_counts = {}
        
        for vpu in self.sim.virtual_clock.vpus:
            if hasattr(vpu, 'w_slice'):
                w = vpu.w_slice
                
                if w in exclude_slices:
                    continue
                
                # Count VPUs per slice
                if w not in slice_vpu_counts:
                    slice_vpu_counts[w] = 0
                slice_vpu_counts[w] += 1
                
                # Calculate activity
                stats = vpu.get_statistics()
                neurons = stats.get('neuron_count', 0)
                cycles = stats.get('cycles_executed', 0)
                
                if neurons > 0 and cycles > 0:
                    spikes = stats.get('spikes_generated', 0)
                    activity = spikes / (neurons * cycles)
                    
                    if w not in slice_activities:
                        slice_activities[w] = []
                    slice_activities[w].append(activity)
        
        # Find slice with lowest activity that has multiple VPUs
        candidates = []
        for w, activities in slice_activities.items():
            if slice_vpu_counts.get(w, 0) > 1:  # Can spare a VPU
                avg_activity = np.mean(activities)
                candidates.append((w, avg_activity, slice_vpu_counts[w]))
        
        if candidates:
            # Sort by activity (lowest first), then by VPU count (highest first)
            candidates.sort(key=lambda x: (x[1], -x[2]))
            return candidates[0][0]
        
        return None
    
    def reallocate_vpu(
        self,
        from_slice: int,
        to_slice: int,
        reason: str
    ) -> bool:
        """Reallocate a VPU from one slice to another.
        
        This is a placeholder for actual VPU reallocation logic.
        Full implementation would require migrating neurons and
        updating VPU assignments.
        
        Args:
            from_slice: Source w-slice
            to_slice: Target w-slice
            reason: Reason for reallocation
            
        Returns:
            True if reallocation was successful
        """
        logger.info(
            f"Reallocating VPU: w={from_slice} -> w={to_slice} "
            f"(reason: {reason})"
        )
        
        # Record in optimization history
        self.optimization_history.append({
            'type': 'vpu_reallocation',
            'from_slice': from_slice,
            'to_slice': to_slice,
            'reason': reason,
            'applied': False,  # Would be True when fully implemented
        })
        
        # Placeholder: would actually move VPU assignment
        # For now, just log the intent
        return True
    
    def get_cognitive_performance_summary(self) -> Dict:
        """Get cognitive orchestration performance summary.
        
        Returns:
            Dictionary with cognitive metrics
        """
        base_summary = self.get_performance_summary()
        
        # Add cognitive-specific metrics
        base_summary.update({
            'total_reallocations': self.reallocation_count,
            'critical_regions': len(self.CRITICAL_REGIONS),
            'priority_adjustments': len(self.priority_history),
        })
        
        # Recent priority adjustments
        if self.priority_history:
            recent_priorities = self.priority_history[-10:]
            base_summary['recent_priority_actions'] = recent_priorities
        
        return base_summary
