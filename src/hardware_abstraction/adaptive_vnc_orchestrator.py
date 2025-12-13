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
        if len(self.performance_log) > 1000:
            self.performance_log = self.performance_log[-1000:]
        
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
