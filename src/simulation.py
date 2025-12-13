"""Main simulation loop for 4D Neural Cognition."""

from typing import Callable, Optional, Union

import numpy as np

try:
    from .brain_model import BrainModel
    from .cell_lifecycle import maybe_kill_and_reproduce, update_health_and_age
    from .plasticity import apply_weight_decay, hebbian_update
    from .time_indexed_spikes import TimeIndexedSpikeBuffer, SpikeHistoryAdapter
    from .hardware_abstraction.virtual_clock import GlobalVirtualClock
    from .hardware_abstraction.virtual_processing_unit import VirtualProcessingUnit
    from .hardware_abstraction.slice_partitioner import SlicePartitioner
except ImportError:
    from brain_model import BrainModel
    from cell_lifecycle import maybe_kill_and_reproduce, update_health_and_age
    from plasticity import apply_weight_decay, hebbian_update
    from time_indexed_spikes import TimeIndexedSpikeBuffer, SpikeHistoryAdapter
    try:
        from hardware_abstraction.virtual_clock import GlobalVirtualClock
        from hardware_abstraction.virtual_processing_unit import VirtualProcessingUnit
        from hardware_abstraction.slice_partitioner import SlicePartitioner
    except ImportError:
        # VNC not available, will be None
        GlobalVirtualClock = None
        VirtualProcessingUnit = None
        SlicePartitioner = None


class Simulation:
    """Main simulation class for running the 4D brain model."""

    def __init__(
        self,
        model: BrainModel,
        seed: int = None,
        use_time_indexed_spikes: bool = False,
        use_vnc: bool = False,
        vnc_clock_frequency: float = 20e6,
    ):
        """Initialize the simulation.

        Args:
            model: The brain model to simulate.
            seed: Random seed for reproducibility.
            use_time_indexed_spikes: If True, use time-indexed spike buffer for O(1) lookups
            use_vnc: If True, use Virtual Neuromorphic Clock for parallel processing
            vnc_clock_frequency: Virtual clock frequency in Hz (default: 20 MHz)
        """
        self.model = model
        self.rng = np.random.default_rng(seed)
        self.use_time_indexed_spikes = use_time_indexed_spikes
        self.use_vnc = use_vnc
        
        # Initialize spike storage
        if use_time_indexed_spikes:
            self._spike_buffer = TimeIndexedSpikeBuffer(window_size=100)
            self.spike_history = SpikeHistoryAdapter(self._spike_buffer)
        else:
            self._spike_buffer = None
            self.spike_history: dict[int, list[int]] = {}
        
        self._callbacks: list[Callable] = []
        
        # Virtual Neuromorphic Clock system
        self.virtual_clock: Optional[GlobalVirtualClock] = None
        self.vnc_clock_frequency = vnc_clock_frequency
        if use_vnc and GlobalVirtualClock is not None:
            self._initialize_vnc()
        elif use_vnc:
            import logging
            logging.warning("VNC requested but hardware_abstraction module not available")

    def add_callback(self, callback: Callable) -> None:
        """Add a callback function to be called each step.

        Args:
            callback: Function taking (simulation, step) as arguments.
        """
        self._callbacks.append(callback)

    def initialize_neurons(
        self,
        area_names: list[str] = None,
        density: float = 1.0,
    ) -> None:
        """Initialize neurons in specified areas.

        Creates neurons at random positions within brain areas based on
        specified density. This populates the 4D lattice with neurons.

        Args:
            area_names: List of area names to initialize. If None, all areas.
            density: Fraction of positions to fill with neurons (0-1).

        Raises:
            ValueError: If density is not in valid range [0, 1] or if
                       specified area names don't exist.
        """
        # Validate density parameter
        if not 0.0 <= density <= 1.0:
            raise ValueError(f"Density must be between 0 and 1, got {density}")

        areas = self.model.get_areas()

        # Validate area names if specified
        if area_names is not None:
            available_areas = {a["name"] for a in areas}
            invalid_areas = set(area_names) - available_areas
            if invalid_areas:
                raise ValueError(f"Unknown area names: {invalid_areas}. " f"Available areas: {available_areas}")
            areas = [a for a in areas if a["name"] in area_names]

        # Create neurons in each specified area
        for area in areas:
            ranges = area["coord_ranges"]
            # Iterate through all positions in 4D space
            for w in range(ranges["w"][0], ranges["w"][1] + 1):
                for z in range(ranges["z"][0], ranges["z"][1] + 1):
                    for y in range(ranges["y"][0], ranges["y"][1] + 1):
                        for x in range(ranges["x"][0], ranges["x"][1] + 1):
                            # Probabilistically create neuron based on density
                            if self.rng.random() <= density:
                                self.model.add_neuron(x, y, z, w)

    def initialize_random_synapses(
        self,
        connection_probability: float = 0.01,
        weight_mean: float = 0.1,
        weight_std: float = 0.05,
    ) -> None:
        """Create random synaptic connections between neurons.

        Generates a random connectivity matrix where each pair of distinct
        neurons has a probability of being connected. Weights are drawn from
        a normal distribution to add biological variability.

        Note: This creates O(n²) potential connections. For large networks,
        consider using a more scalable initialization method.

        Args:
            connection_probability: Probability of connection between any two
                                   neurons (0-1). Default: 0.01.
            weight_mean: Mean initial synaptic weight. Default: 0.1.
            weight_std: Standard deviation of initial weights. Default: 0.05.

        Raises:
            ValueError: If connection_probability is not in [0, 1] or if
                       weight_std is negative.
        """
        # Validate parameters
        if not 0.0 <= connection_probability <= 1.0:
            raise ValueError(f"connection_probability must be between 0 and 1, " f"got {connection_probability}")

        if weight_std < 0:
            raise ValueError(f"weight_std must be non-negative, got {weight_std}")

        neuron_ids = list(self.model.neurons.keys())

        # Generate random connections between all pairs of neurons
        for pre_id in neuron_ids:
            for post_id in neuron_ids:
                # No self-connections (autapses)
                if pre_id != post_id:
                    # Probabilistically create synapse
                    if self.rng.random() < connection_probability:
                        # Sample weight from normal distribution
                        weight = self.rng.normal(weight_mean, weight_std)
                        self.model.add_synapse(pre_id, post_id, weight)

    def lif_step(self, neuron_id: int, dt: float = 1.0) -> bool:
        """Perform one Leaky Integrate-and-Fire step for a neuron.

        Implements the LIF neuron model which consists of:
        1. Refractory period check (neuron cannot spike immediately after last spike)
        2. Synaptic input integration (weighted sum from presynaptic neurons)
        3. Leaky membrane integration (exponential decay toward rest potential)
        4. Spike threshold detection and reset

        The membrane potential equation:
            dV/dt = (-(V - V_rest) + I_total) / tau_m

        Args:
            neuron_id: ID of the neuron to update.
            dt: Time step in ms (default: 1.0).

        Returns:
            True if the neuron spiked this step, False otherwise.
        """
        neuron = self.model.neurons.get(neuron_id)
        if neuron is None:
            return False

        # Extract neuron parameters from config
        params = neuron.params
        tau_m = params.get("tau_m", 20.0)  # Membrane time constant (ms)
        v_rest = params.get("v_rest", -65.0)  # Resting potential (mV)
        v_reset = params.get("v_reset", -70.0)  # Reset potential after spike (mV)
        v_threshold = params.get("v_threshold", -50.0)  # Spike threshold (mV)
        refractory_period = params.get("refractory_period", 5.0)  # Refractory period (ms)

        current_step = self.model.current_step

        # Phase 1: Check refractory period
        # During refractory period, neuron cannot spike and ignores input
        time_since_spike = current_step - neuron.last_spike_time
        if time_since_spike < refractory_period:
            neuron.external_input = 0.0
            return False

        # Phase 2: Calculate synaptic input from all presynaptic neurons
        # Sum weighted contributions from neurons that spiked at the right time
        # considering synaptic delay
        synaptic_input = 0.0
        
        if self.use_time_indexed_spikes:
            # Optimized O(1) lookup using time-indexed buffer
            for synapse in self.model.get_synapses_for_neuron(neuron_id, direction="post"):
                pre_neuron = self.model.neurons.get(synapse.pre_id)
                if pre_neuron is not None:
                    # Direct O(1) check if neuron spiked at the right time
                    spike_time = current_step - synapse.delay
                    if self._spike_buffer.did_spike_at(synapse.pre_id, spike_time):
                        synaptic_input += synapse.weight
        else:
            # Original O(n) lookup with list iteration
            for synapse in self.model.get_synapses_for_neuron(neuron_id, direction="post"):
                pre_neuron = self.model.neurons.get(synapse.pre_id)
                if pre_neuron is not None:
                    # Check if presynaptic neuron spiked at time matching the delay
                    pre_spike_times = self.spike_history.get(synapse.pre_id, [])
                    for spike_time in pre_spike_times:
                        # Synaptic delay: spike arrives after 'delay' time steps
                        if current_step - spike_time == synapse.delay:
                            synaptic_input += synapse.weight
                            break

        # Phase 3: Compute total input current
        # Combines synaptic input and external sensory input
        total_input = synaptic_input + neuron.external_input

        # Phase 4: Leaky integration step
        # Membrane potential decays exponentially toward rest potential
        # while being driven by input current
        dv = (-(neuron.v_membrane - v_rest) + total_input) / tau_m * dt
        neuron.v_membrane += dv

        # Check for NaN/Inf and reset to safe value if needed
        # This prevents numerical instability from propagating
        if np.isnan(neuron.v_membrane) or np.isinf(neuron.v_membrane):
            neuron.v_membrane = v_rest

        # Reset external input after processing (one-time input)
        neuron.external_input = 0.0

        # Phase 5: Check for spike and reset
        if neuron.v_membrane >= v_threshold:
            # Spike occurs: reset membrane potential
            neuron.v_membrane = v_reset
            neuron.last_spike_time = current_step

            # Record spike in history for synaptic transmission
            if self.use_time_indexed_spikes:
                self._spike_buffer.add_spike(neuron_id, current_step)
            else:
                if neuron_id not in self.spike_history:
                    self.spike_history[neuron_id] = []
                self.spike_history[neuron_id].append(current_step)

            return True

        return False

    def step(self) -> dict:
        """Run one simulation step.

        Executes a complete simulation cycle including:
        1. Neuron dynamics update (LIF model)
        2. Synaptic plasticity (Hebbian learning)
        3. Cell lifecycle (aging, death, reproduction)
        4. Housekeeping (spike history cleanup, callbacks)

        Returns:
            Dictionary with step statistics including:
                - step: Current simulation step number
                - spikes: List of neuron IDs that spiked
                - deaths: Number of neurons that died
                - births: Number of new neurons born
        """
        stats = {
            "step": self.model.current_step,
            "spikes": [],
            "deaths": 0,
            "births": 0,
        }

        # Phase 1: Neural dynamics - Update membrane potentials and detect spikes
        # Must snapshot neuron IDs to avoid modification during iteration
        neuron_ids = list(self.model.neurons.keys())
        spikes = []

        for neuron_id in neuron_ids:
            spiked = self.lif_step(neuron_id)
            if spiked:
                spikes.append(neuron_id)

        stats["spikes"] = spikes

        # Phase 2: Synaptic plasticity - Update connection strengths
        # Hebbian rule: "Cells that fire together, wire together"
        # Weight decay prevents unbounded weight growth
        # Optimization: Use set for O(1) lookup instead of O(n) list membership check
        spike_set = set(spikes)
        for synapse in self.model.synapses:
            pre_spiked = synapse.pre_id in spike_set
            post_spiked = synapse.post_id in spike_set
            hebbian_update(synapse, pre_spiked, post_spiked, self.model)
            apply_weight_decay(synapse, self.model)

        # Phase 3: Cell lifecycle - Aging, death, and reproduction with mutation
        # Neurons can die due to old age or low health
        # Dead neurons are replaced by offspring with inherited properties
        neuron_ids = list(self.model.neurons.keys())  # Re-snapshot after potential changes
        for neuron_id in neuron_ids:
            neuron = self.model.neurons.get(neuron_id)
            if neuron is None:  # Neuron may have been removed
                continue

            # Age the neuron and decay its health
            update_health_and_age(neuron, self.model)

            # Check if neuron should die and reproduce
            old_id = neuron.id
            new_neuron = maybe_kill_and_reproduce(neuron, self.model, self.rng)

            # Track lifecycle events
            if new_neuron is None:
                # Neuron died without reproduction
                stats["deaths"] += 1
            elif new_neuron.id != old_id:
                # Neuron died and was replaced by offspring
                stats["deaths"] += 1
                stats["births"] += 1

        # Phase 4: Housekeeping - Memory management and callbacks
        # Clean up old spike history to prevent unbounded memory growth
        # Only keep spikes from recent past (needed for synaptic transmission)
        current_step = self.model.current_step
        
        if self.use_time_indexed_spikes:
            # Time-indexed buffer handles cleanup automatically
            self._spike_buffer.advance_time(current_step)
        else:
            # Manual cleanup for dict-based history
            max_history = 100  # Keep last 100 time steps
            for neuron_id in list(self.spike_history.keys()):
                self.spike_history[neuron_id] = [t for t in self.spike_history[neuron_id] if current_step - t < max_history]
                # Remove empty histories to free memory
                if not self.spike_history[neuron_id]:
                    del self.spike_history[neuron_id]

        # Execute registered callbacks (e.g., logging, visualization)
        for callback in self._callbacks:
            callback(self, self.model.current_step)

        # Advance simulation time
        self.model.current_step += 1
        return stats

    def run(self, n_steps: int, verbose: bool = False) -> list[dict]:
        """Run multiple simulation steps.

        Args:
            n_steps: Number of steps to run.
            verbose: Whether to print progress.

        Returns:
            List of statistics dictionaries for each step.
        """
        # Use VNC mode if enabled
        if self.use_vnc and self.virtual_clock is not None:
            return self._run_vnc(n_steps, verbose)
        
        # Standard mode
        all_stats = []

        for i in range(n_steps):
            stats = self.step()
            all_stats.append(stats)

            if verbose and (i + 1) % 100 == 0:
                print(
                    f"Step {i + 1}/{n_steps}: "
                    f"{len(stats['spikes'])} spikes, "
                    f"{stats['deaths']} deaths, "
                    f"{stats['births']} births"
                )

        return all_stats
    
    def _initialize_vnc(self) -> None:
        """Initialize the Virtual Neuromorphic Clock system."""
        import logging
        logger = logging.getLogger(__name__)
        
        if GlobalVirtualClock is None or SlicePartitioner is None or VirtualProcessingUnit is None:
            logger.error("VNC modules not available")
            return
        
        # Create virtual clock
        self.virtual_clock = GlobalVirtualClock(
            frequency_hz=self.vnc_clock_frequency
        )
        
        # Determine lattice shape from model
        lattice_shape = self._get_lattice_shape()
        
        # Partition the lattice into slices
        partitions = SlicePartitioner.partition_by_w_slice(lattice_shape)
        
        # Create VPUs for each partition
        for i, partition in enumerate(partitions):
            vpu = VirtualProcessingUnit(vpu_id=i, clock_speed_hz=self.vnc_clock_frequency)
            vpu.assign_slice(partition)
            vpu.initialize_batch(self.model, self)
            self.virtual_clock.add_vpu(vpu)
        
        logger.info(
            f"Initialized VNC with {len(partitions)} VPUs at "
            f"{self.vnc_clock_frequency/1e6:.1f} MHz"
        )
    
    def _get_lattice_shape(self) -> tuple:
        """Get the shape of the 4D lattice from the model.
        
        Returns:
            4-tuple (x_size, y_size, z_size, w_size)
        """
        if not self.model.neurons:
            # Default shape if no neurons
            return (10, 10, 10, 10)
        
        # Find bounds from existing neurons
        positions = [neuron.position for neuron in self.model.neurons.values()]
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]
        z_coords = [p[2] for p in positions]
        w_coords = [p[3] for p in positions]
        
        return (
            max(x_coords) + 1 if x_coords else 10,
            max(y_coords) + 1 if y_coords else 10,
            max(z_coords) + 1 if z_coords else 10,
            max(w_coords) + 1 if w_coords else 10,
        )
    
    def step_vnc(self) -> dict:
        """Run one simulation step using the Virtual Neuromorphic Clock.
        
        This method uses the VNC system for parallel processing of neurons.
        All VPUs execute in parallel, synchronized by the global clock.
        
        Returns:
            Dictionary with step statistics
        """
        if self.virtual_clock is None:
            raise RuntimeError("VNC not initialized. Set use_vnc=True in constructor.")
        
        # Run one global clock cycle
        cycle_result = self.virtual_clock.run_cycle()
        
        # Apply plasticity and cell lifecycle (done globally after VPU processing)
        stats = {
            "step": self.model.current_step,
            "spikes": [],
            "deaths": 0,
            "births": 0,
            "vnc_stats": cycle_result,
        }
        
        # Collect spikes from spike history
        if self.use_time_indexed_spikes:
            # Get spikes from this cycle
            for neuron_id in self.model.neurons.keys():
                if self._spike_buffer.did_spike_at(neuron_id, self.model.current_step):
                    stats["spikes"].append(neuron_id)
        else:
            # Get spikes from this step
            for neuron_id, spike_times in self.spike_history.items():
                if self.model.current_step in spike_times:
                    stats["spikes"].append(neuron_id)
        
        # Phase 2: Synaptic plasticity
        spike_set = set(stats["spikes"])
        for synapse in self.model.synapses:
            pre_spiked = synapse.pre_id in spike_set
            post_spiked = synapse.post_id in spike_set
            hebbian_update(synapse, pre_spiked, post_spiked, self.model)
            apply_weight_decay(synapse, self.model)
        
        # Phase 3: Cell lifecycle
        neuron_ids = list(self.model.neurons.keys())
        for neuron_id in neuron_ids:
            neuron = self.model.neurons.get(neuron_id)
            if neuron is None:
                continue
            
            update_health_and_age(neuron, self.model)
            old_id = neuron.id
            new_neuron = maybe_kill_and_reproduce(neuron, self.model, self.rng)
            
            if new_neuron is None:
                stats["deaths"] += 1
            elif new_neuron.id != old_id:
                stats["deaths"] += 1
                stats["births"] += 1
        
        # Execute callbacks
        for callback in self._callbacks:
            callback(self, self.model.current_step)
        
        # Advance simulation time
        self.model.current_step += 1
        
        # Adaptive load balancing every 1000 cycles
        if cycle_result["cycle"] > 0 and cycle_result["cycle"] % 1000 == 0:
            self.virtual_clock.rebalance_partitions()
        
        return stats
    
    def _run_vnc(self, n_steps: int, verbose: bool = False) -> list[dict]:
        """Run multiple simulation steps using VNC mode.
        
        Args:
            n_steps: Number of steps to run
            verbose: Whether to print progress
            
        Returns:
            List of statistics dictionaries for each step
        """
        all_stats = []
        
        def progress_callback(cycle_result):
            if verbose and (cycle_result["cycle"] + 1) % 100 == 0:
                print(
                    f"VNC Cycle {cycle_result['cycle'] + 1}/{n_steps}: "
                    f"{cycle_result['neurons_processed']} neurons, "
                    f"{cycle_result['spikes']} spikes, "
                    f"{cycle_result['time_ms']:.2f}ms"
                )
        
        for i in range(n_steps):
            stats = self.step_vnc()
            all_stats.append(stats)
            
            if verbose and (i + 1) % 100 == 0:
                vnc_stats = stats.get("vnc_stats", {})
                print(
                    f"Step {i + 1}/{n_steps}: "
                    f"{len(stats['spikes'])} spikes, "
                    f"{stats['deaths']} deaths, "
                    f"{stats['births']} births "
                    f"[VNC: {vnc_stats.get('neurons_processed', 0)} neurons, "
                    f"{vnc_stats.get('time_ms', 0):.2f}ms]"
                )
        
        return all_stats
    
    def get_vnc_statistics(self) -> Optional[dict]:
        """Get Virtual Neuromorphic Clock statistics.
        
        Returns:
            Dictionary with VNC performance metrics, or None if VNC not enabled
        """
        if self.virtual_clock is None:
            return None
        
        return self.virtual_clock.get_statistics()
    
    def get_vpu_statistics(self) -> Optional[list]:
        """Get statistics for all Virtual Processing Units.
        
        Returns:
            List of VPU statistics dictionaries, or None if VNC not enabled
        """
        if self.virtual_clock is None:
            return None
        
        return self.virtual_clock.get_vpu_statistics()
