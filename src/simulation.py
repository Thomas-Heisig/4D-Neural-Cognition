"""Main simulation loop for 4D Neural Cognition."""

import numpy as np
from typing import Callable

try:
    from .brain_model import BrainModel
    from .cell_lifecycle import maybe_kill_and_reproduce, update_health_and_age
    from .plasticity import hebbian_update, apply_weight_decay
except ImportError:
    from brain_model import BrainModel
    from cell_lifecycle import maybe_kill_and_reproduce, update_health_and_age
    from plasticity import hebbian_update, apply_weight_decay


class Simulation:
    """Main simulation class for running the 4D brain model."""

    def __init__(
        self,
        model: BrainModel,
        seed: int = None,
    ):
        """Initialize the simulation.

        Args:
            model: The brain model to simulate.
            seed: Random seed for reproducibility.
        """
        self.model = model
        self.rng = np.random.default_rng(seed)
        self.spike_history: dict[int, list[int]] = {}
        self._callbacks: list[Callable] = []

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

        Args:
            area_names: List of area names to initialize. If None, all areas.
            density: Fraction of positions to fill with neurons (0-1).
        """
        areas = self.model.get_areas()
        if area_names is not None:
            areas = [a for a in areas if a["name"] in area_names]

        for area in areas:
            ranges = area["coord_ranges"]
            for w in range(ranges["w"][0], ranges["w"][1] + 1):
                for z in range(ranges["z"][0], ranges["z"][1] + 1):
                    for y in range(ranges["y"][0], ranges["y"][1] + 1):
                        for x in range(ranges["x"][0], ranges["x"][1] + 1):
                            if self.rng.random() <= density:
                                self.model.add_neuron(x, y, z, w)

    def initialize_random_synapses(
        self,
        connection_probability: float = 0.01,
        weight_mean: float = 0.1,
        weight_std: float = 0.05,
    ) -> None:
        """Create random synaptic connections between neurons.

        Args:
            connection_probability: Probability of connection between any two neurons.
            weight_mean: Mean initial weight.
            weight_std: Standard deviation of initial weights.
        """
        neuron_ids = list(self.model.neurons.keys())

        for pre_id in neuron_ids:
            for post_id in neuron_ids:
                if pre_id != post_id:
                    if self.rng.random() < connection_probability:
                        weight = self.rng.normal(weight_mean, weight_std)
                        self.model.add_synapse(pre_id, post_id, weight)

    def lif_step(self, neuron_id: int, dt: float = 1.0) -> bool:
        """Perform one Leaky Integrate-and-Fire step for a neuron.

        Args:
            neuron_id: ID of the neuron to update.
            dt: Time step in ms.

        Returns:
            True if the neuron spiked, False otherwise.
        """
        neuron = self.model.neurons.get(neuron_id)
        if neuron is None:
            return False

        params = neuron.params
        tau_m = params.get("tau_m", 20.0)
        v_rest = params.get("v_rest", -65.0)
        v_reset = params.get("v_reset", -70.0)
        v_threshold = params.get("v_threshold", -50.0)
        refractory_period = params.get("refractory_period", 5.0)

        current_step = self.model.current_step

        # Check refractory period
        time_since_spike = current_step - neuron.last_spike_time
        if time_since_spike < refractory_period:
            neuron.external_input = 0.0
            return False

        # Calculate synaptic input
        synaptic_input = 0.0
        for synapse in self.model.get_synapses_for_neuron(neuron_id, direction="post"):
            pre_neuron = self.model.neurons.get(synapse.pre_id)
            if pre_neuron is not None:
                # Check if presynaptic neuron spiked recently
                pre_spike_times = self.spike_history.get(synapse.pre_id, [])
                for spike_time in pre_spike_times:
                    if current_step - spike_time == synapse.delay:
                        synaptic_input += synapse.weight
                        break

        # Total input
        total_input = synaptic_input + neuron.external_input

        # Leaky integration
        dv = (-( neuron.v_membrane - v_rest) + total_input) / tau_m * dt
        neuron.v_membrane += dv

        # Reset external input
        neuron.external_input = 0.0

        # Check for spike
        if neuron.v_membrane >= v_threshold:
            neuron.v_membrane = v_reset
            neuron.last_spike_time = current_step

            # Record spike
            if neuron_id not in self.spike_history:
                self.spike_history[neuron_id] = []
            self.spike_history[neuron_id].append(current_step)

            return True

        return False

    def step(self) -> dict:
        """Run one simulation step.

        Returns:
            Dictionary with step statistics.
        """
        stats = {
            "step": self.model.current_step,
            "spikes": [],
            "deaths": 0,
            "births": 0,
        }

        # Update each neuron
        neuron_ids = list(self.model.neurons.keys())
        spikes = []

        for neuron_id in neuron_ids:
            spiked = self.lif_step(neuron_id)
            if spiked:
                spikes.append(neuron_id)

        stats["spikes"] = spikes

        # Apply plasticity
        for synapse in self.model.synapses:
            pre_spiked = synapse.pre_id in spikes
            post_spiked = synapse.post_id in spikes
            hebbian_update(synapse, pre_spiked, post_spiked, self.model)
            apply_weight_decay(synapse, self.model)

        # Update health and age, check for death/reproduction
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

        # Clean up old spike history (keep only recent spikes)
        max_history = 100
        current_step = self.model.current_step
        for neuron_id in list(self.spike_history.keys()):
            self.spike_history[neuron_id] = [
                t
                for t in self.spike_history[neuron_id]
                if current_step - t < max_history
            ]
            if not self.spike_history[neuron_id]:
                del self.spike_history[neuron_id]

        # Call callbacks
        for callback in self._callbacks:
            callback(self, self.model.current_step)

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
