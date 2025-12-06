"""Cell lifecycle management - death and reproduction with inheritance."""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    try:
        from .brain_model import BrainModel, Neuron
    except ImportError:
        from brain_model import BrainModel, Neuron


def mutate_params(params: dict, rng: np.random.Generator, std: float = 0.05) -> dict:
    """Apply random mutations to neuron parameters.

    Args:
        params: Original neuron parameters.
        rng: Random number generator.
        std: Standard deviation for mutations (relative).

    Returns:
        Mutated parameters dictionary.
    """
    mutated = {}
    for key, value in params.items():
        if isinstance(value, (int, float)):
            # Apply Gaussian mutation
            mutation_factor = 1.0 + rng.normal(0, std)
            mutated[key] = value * mutation_factor
        else:
            mutated[key] = value
    return mutated


def mutate_weight(weight: float, rng: np.random.Generator, std: float = 0.02) -> float:
    """Apply random mutation to a synapse weight.

    Args:
        weight: Original weight value.
        rng: Random number generator.
        std: Standard deviation for mutation.

    Returns:
        Mutated weight value.
    """
    return weight + rng.normal(0, std)


def maybe_kill_and_reproduce(
    neuron: "Neuron",
    model: "BrainModel",
    rng: np.random.Generator,
) -> "Neuron":
    """Check if a neuron should die and reproduce.

    If the neuron's health is below threshold or age exceeds max_age,
    it dies and a new neuron is created at the same position with
    inherited (mutated) properties.

    Args:
        neuron: The neuron to check.
        model: The brain model containing neurons and synapses.
        rng: Random number generator.

    Returns:
        The new neuron if reproduction occurred, otherwise the original neuron.
    """
    lifecycle = model.get_lifecycle_config()

    if not lifecycle["enable_death"]:
        return neuron

    death_threshold = 0.1
    max_age = lifecycle["max_age"]

    # Check death conditions
    if neuron.health >= death_threshold and neuron.age <= max_age:
        return neuron

    if not lifecycle["enable_reproduction"]:
        # Just remove the neuron without reproduction
        model.remove_neuron(neuron.id)
        return None

    parent_id = neuron.id
    old_position = neuron.position()

    # Get synapses connected to the dying neuron
    old_synapses = []
    for synapse in model.synapses:
        if synapse.pre_id == parent_id or synapse.post_id == parent_id:
            old_synapses.append(synapse)

    # Create new neuron with inherited properties
    mutation_std_params = lifecycle["mutation_std_params"]
    mutation_std_weights = lifecycle["mutation_std_weights"]

    mutated_params = mutate_params(neuron.params, rng, mutation_std_params)

    new_neuron = model.add_neuron(
        x=old_position[0],
        y=old_position[1],
        z=old_position[2],
        w=old_position[3],
        generation=neuron.generation + 1,
        parent_id=parent_id,
        health=1.0,
        params=mutated_params,
    )

    # Transfer and mutate synapses
    new_synapses_data = []
    for synapse in old_synapses:
        new_pre = new_neuron.id if synapse.pre_id == parent_id else synapse.pre_id
        new_post = new_neuron.id if synapse.post_id == parent_id else synapse.post_id
        new_weight = mutate_weight(synapse.weight, rng, mutation_std_weights)

        new_synapses_data.append((new_pre, new_post, new_weight, synapse.delay))

    # Remove old neuron and its synapses
    model.remove_neuron(parent_id)

    # Add new synapses
    for pre_id, post_id, weight, delay in new_synapses_data:
        # Only add synapse if both endpoints exist
        if pre_id in model.neurons and post_id in model.neurons:
            model.add_synapse(pre_id, post_id, weight, delay)

    return new_neuron


def update_health_and_age(neuron: "Neuron", model: "BrainModel") -> None:
    """Update neuron health and age for one time step.

    Args:
        neuron: The neuron to update.
        model: The brain model containing configuration.
    """
    lifecycle = model.get_lifecycle_config()
    neuron.age += 1
    neuron.health -= lifecycle["health_decay_per_step"]
    neuron.health = max(0.0, neuron.health)
