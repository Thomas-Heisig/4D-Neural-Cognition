"""Global Virtual Clock for synchronizing virtual processing units."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .virtual_processing_unit import VirtualProcessingUnit

logger = logging.getLogger(__name__)


class GlobalVirtualClock:
    """Global virtual clock that synchronizes all Virtual Processing Units.
    
    This class implements a virtual clock that runs at a specified frequency
    (e.g., 20 MHz) and coordinates the execution of all VPUs. Each clock cycle,
    all VPUs process their assigned neurons in parallel, then synchronize at
    a barrier before proceeding to the next cycle.
    
    This architecture emulates neuromorphic hardware behavior where all neurons
    update synchronously at a fixed clock rate.
    
    Attributes:
        frequency: Clock frequency in Hz (e.g., 20e6 for 20 MHz)
        current_cycle: Current clock cycle number
        vpus: List of Virtual Processing Units managed by this clock
        is_running: Whether the clock is currently running
    """
    
    def __init__(self, frequency_hz: float = 20e6, max_workers: Optional[int] = None):
        """Initialize the Global Virtual Clock.
        
        Args:
            frequency_hz: Clock frequency in Hz (default: 20 MHz)
            max_workers: Maximum number of worker threads (default: number of VPUs)
        """
        self.frequency = frequency_hz
        self.current_cycle = 0
        self.vpus: List[VirtualProcessingUnit] = []
        self.is_running = False
        self.max_workers = max_workers
        
        # Synchronization
        self._cycle_barrier: Optional[threading.Barrier] = None
        self._lock = threading.Lock()
        
        # Statistics
        self.statistics = {
            "total_cycles": 0,
            "total_neurons_processed": 0,
            "total_spikes": 0,
            "total_time_ms": 0.0,
        }
        
        logger.info(f"Initialized GlobalVirtualClock at {frequency_hz/1e6:.1f} MHz")
    
    def add_vpu(self, vpu: VirtualProcessingUnit) -> None:
        """Add a Virtual Processing Unit to be managed by this clock.
        
        Args:
            vpu: The VPU to add
        """
        with self._lock:
            self.vpus.append(vpu)
            # Recreate barrier with new VPU count
            if len(self.vpus) > 0:
                self._cycle_barrier = threading.Barrier(len(self.vpus))
            logger.info(f"Added VPU {vpu.vpu_id}, total VPUs: {len(self.vpus)}")
    
    def remove_vpu(self, vpu_id: int) -> bool:
        """Remove a Virtual Processing Unit.
        
        Args:
            vpu_id: ID of the VPU to remove
            
        Returns:
            True if VPU was found and removed, False otherwise
        """
        with self._lock:
            for i, vpu in enumerate(self.vpus):
                if vpu.vpu_id == vpu_id:
                    self.vpus.pop(i)
                    # Recreate barrier with new VPU count
                    if len(self.vpus) > 0:
                        self._cycle_barrier = threading.Barrier(len(self.vpus))
                    else:
                        self._cycle_barrier = None
                    logger.info(f"Removed VPU {vpu_id}, total VPUs: {len(self.vpus)}")
                    return True
            return False
    
    def run_cycle(self) -> dict:
        """Execute one complete clock cycle across all VPUs.
        
        This method:
        1. Starts all VPUs in parallel using a thread pool
        2. Waits for all VPUs to complete their processing
        3. Synchronizes at a barrier
        4. Increments the global clock cycle
        5. Collects and returns statistics
        
        Returns:
            Dictionary with cycle statistics including neurons processed and timing
        """
        import time
        
        if len(self.vpus) == 0:
            logger.warning("No VPUs available for processing")
            return {
                "cycle": self.current_cycle,
                "vpus": 0,
                "neurons_processed": 0,
                "spikes": 0,
                "time_ms": 0.0,
            }
        
        start_time = time.time()
        cycle_results = []
        
        # Step 1: Execute all VPUs in parallel
        workers = self.max_workers if self.max_workers is not None else len(self.vpus)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all VPU processing tasks
            futures = {
                executor.submit(vpu.process_cycle, self.current_cycle): vpu
                for vpu in self.vpus
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    result = future.result()
                    cycle_results.append(result)
                except Exception as e:
                    vpu = futures[future]
                    logger.error(f"VPU {vpu.vpu_id} failed during cycle {self.current_cycle}: {e}")
        
        # Step 2: Synchronization point (implicit through ThreadPoolExecutor completion)
        # In a more advanced implementation, we would use the barrier here
        
        # Step 3: Increment global cycle counter
        with self._lock:
            self.current_cycle += 1
        
        # Step 4: Collect statistics
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        total_neurons = sum(r["neurons_processed"] for r in cycle_results)
        total_spikes = sum(r["spikes"] for r in cycle_results)
        
        self.statistics["total_cycles"] += 1
        self.statistics["total_neurons_processed"] += total_neurons
        self.statistics["total_spikes"] += total_spikes
        self.statistics["total_time_ms"] += processing_time
        
        return {
            "cycle": self.current_cycle - 1,  # Return the cycle we just completed
            "vpus": len(self.vpus),
            "neurons_processed": total_neurons,
            "spikes": total_spikes,
            "time_ms": processing_time,
            "vpu_results": cycle_results,
        }
    
    def run_cycles(self, num_cycles: int, callback=None) -> List[dict]:
        """Run multiple clock cycles.
        
        Args:
            num_cycles: Number of cycles to run
            callback: Optional callback function called after each cycle
                     with signature: callback(cycle_result)
        
        Returns:
            List of cycle result dictionaries
        """
        self.is_running = True
        results = []
        
        try:
            for i in range(num_cycles):
                if not self.is_running:
                    logger.info(f"Clock stopped after {i} cycles")
                    break
                
                result = self.run_cycle()
                results.append(result)
                
                if callback is not None:
                    callback(result)
                
                # Log progress periodically
                if (i + 1) % 100 == 0:
                    logger.info(
                        f"Cycle {result['cycle']}: "
                        f"{result['neurons_processed']} neurons, "
                        f"{result['spikes']} spikes, "
                        f"{result['time_ms']:.2f}ms"
                    )
        finally:
            self.is_running = False
        
        return results
    
    def stop(self) -> None:
        """Stop the clock after the current cycle completes."""
        self.is_running = False
        logger.info("Clock stop requested")
    
    def reset(self) -> None:
        """Reset the clock to cycle 0 and clear statistics."""
        with self._lock:
            self.current_cycle = 0
            self.statistics = {
                "total_cycles": 0,
                "total_neurons_processed": 0,
                "total_spikes": 0,
                "total_time_ms": 0.0,
            }
            for vpu in self.vpus:
                vpu.reset_statistics()
        logger.info("Clock reset to cycle 0")
    
    def get_statistics(self) -> dict:
        """Get global clock statistics.
        
        Returns:
            Dictionary with overall performance metrics
        """
        stats = self.statistics.copy()
        
        # Calculate derived metrics
        if stats["total_cycles"] > 0:
            stats["avg_neurons_per_cycle"] = (
                stats["total_neurons_processed"] / stats["total_cycles"]
            )
            stats["avg_spikes_per_cycle"] = (
                stats["total_spikes"] / stats["total_cycles"]
            )
            stats["avg_time_per_cycle_ms"] = (
                stats["total_time_ms"] / stats["total_cycles"]
            )
        else:
            stats["avg_neurons_per_cycle"] = 0.0
            stats["avg_spikes_per_cycle"] = 0.0
            stats["avg_time_per_cycle_ms"] = 0.0
        
        if stats["total_time_ms"] > 0:
            stats["neurons_per_second"] = (
                stats["total_neurons_processed"] / (stats["total_time_ms"] / 1000.0)
            )
            stats["spikes_per_second"] = (
                stats["total_spikes"] / (stats["total_time_ms"] / 1000.0)
            )
            # Virtual clock rate vs actual
            stats["effective_clock_hz"] = (
                stats["total_cycles"] / (stats["total_time_ms"] / 1000.0)
            )
        else:
            stats["neurons_per_second"] = 0.0
            stats["spikes_per_second"] = 0.0
            stats["effective_clock_hz"] = 0.0
        
        stats["configured_clock_hz"] = self.frequency
        stats["num_vpus"] = len(self.vpus)
        
        return stats
    
    def get_vpu_statistics(self) -> List[dict]:
        """Get statistics for all VPUs.
        
        Returns:
            List of VPU statistics dictionaries
        """
        return [vpu.get_statistics() for vpu in self.vpus]
    
    def rebalance_partitions(self) -> None:
        """Rebalance neuron distribution across VPUs for load balancing.
        
        This is a placeholder for future implementation of adaptive load balancing.
        The algorithm would analyze VPU processing times and redistribute neurons
        to balance the load across all VPUs.
        """
        # TODO: Implement adaptive load balancing
        # 1. Analyze VPU processing times
        # 2. Identify overloaded and underloaded VPUs
        # 3. Redistribute neuron slices to balance load
        logger.debug("Partition rebalancing not yet implemented")
