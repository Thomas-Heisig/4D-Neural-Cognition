"""Hebbian plasticity rules for synapse learning."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    try:
        from .brain_model import BrainModel, Synapse
    except ImportError:
        from brain_model import BrainModel, Synapse


def hebbian_update(
    synapse: "Synapse",
    pre_active: bool,
    post_active: bool,
    model: "BrainModel",
) -> None:
    """Apply Hebbian learning rule to update synapse weight.

    "Cells that fire together, wire together."

    Args:
        synapse: The synapse to update.
        pre_active: Whether the presynaptic neuron is active.
        post_active: Whether the postsynaptic neuron is active.
        model: The brain model containing plasticity configuration.
    """
    plasticity = model.get_plasticity_config()
    learning_rate = plasticity["learning_rate"]
    weight_min = plasticity["weight_min"]
    weight_max = plasticity["weight_max"]

    if pre_active and post_active:
        # Strengthen connection when both neurons fire together
        delta = learning_rate
    elif pre_active and not post_active:
        # Weaken connection when pre fires but post doesn't
        delta = -learning_rate * 0.5
    else:
        # No change
        delta = 0.0

    synapse.weight += delta
    synapse.weight = max(weight_min, min(weight_max, synapse.weight))


def apply_weight_decay(synapse: "Synapse", model: "BrainModel") -> None:
    """Apply weight decay to a synapse.

    Args:
        synapse: The synapse to apply decay to.
        model: The brain model containing plasticity configuration.
    """
    plasticity = model.get_plasticity_config()
    decay = plasticity["weight_decay"]

    # Decay towards zero
    if synapse.weight > 0:
        synapse.weight = max(0, synapse.weight - decay)
    elif synapse.weight < 0:
        synapse.weight = min(0, synapse.weight + decay)


def spike_timing_dependent_plasticity(
    synapse: "Synapse",
    pre_spike_time: int,
    post_spike_time: int,
    model: "BrainModel",
    tau_plus: float = 20.0,
    tau_minus: float = 20.0,
    a_plus: float = 0.01,
    a_minus: float = 0.012,
) -> None:
    """Apply STDP learning rule based on spike timing.

    If pre fires before post: potentiation (LTP)
    If post fires before pre: depression (LTD)

    Args:
        synapse: The synapse to update.
        pre_spike_time: Time step of presynaptic spike.
        post_spike_time: Time step of postsynaptic spike.
        model: The brain model.
        tau_plus: Time constant for potentiation.
        tau_minus: Time constant for depression.
        a_plus: Learning rate for potentiation.
        a_minus: Learning rate for depression.
    """
    import math

    plasticity = model.get_plasticity_config()
    weight_min = plasticity["weight_min"]
    weight_max = plasticity["weight_max"]

    delta_t = post_spike_time - pre_spike_time

    if delta_t > 0:
        # Pre fires before post: potentiation
        delta_w = a_plus * math.exp(-delta_t / tau_plus)
    elif delta_t < 0:
        # Post fires before pre: depression
        delta_w = -a_minus * math.exp(delta_t / tau_minus)
    else:
        delta_w = 0.0

    synapse.weight += delta_w
    synapse.weight = max(weight_min, min(weight_max, synapse.weight))
