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

    Implements the classic Hebbian learning principle:
    "Cells that fire together, wire together."

    This creates a correlation-based learning rule where:
    - Correlated activity strengthens connections (Long-Term Potentiation)
    - Uncorrelated activity weakens connections (Long-Term Depression)

    Learning rules:
    1. Both fire together → Strengthen (LTP): Δw = +η
    2. Pre fires, post doesn't → Weaken (LTD): Δw = -0.5η
    3. Neither fire or only post fires → No change: Δw = 0

    Args:
        synapse: The synapse to update.
        pre_active: Whether the presynaptic neuron spiked this step.
        post_active: Whether the postsynaptic neuron spiked this step.
        model: The brain model containing plasticity configuration.
    """
    plasticity = model.get_plasticity_config()
    learning_rate = plasticity["learning_rate"]
    weight_min = plasticity["weight_min"]
    weight_max = plasticity["weight_max"]

    # Determine weight change based on firing patterns
    if pre_active and post_active:
        # Case 1: Correlated activity (both neurons spike together)
        # Strengthen the connection - this is Long-Term Potentiation (LTP)
        delta = learning_rate
    elif pre_active and not post_active:
        # Case 2: Pre fires but fails to trigger post
        # Weaken the connection - this is Long-Term Depression (LTD)
        # Use smaller magnitude to create asymmetric learning
        delta = -learning_rate * 0.5
    else:
        # Case 3: No presynaptic activity or only postsynaptic
        # No weight change (post-only activity doesn't affect this synapse)
        delta = 0.0

    # Apply weight update with bounds to prevent extreme values
    synapse.weight += delta

    # Check for NaN/Inf values and clip to valid range
    # This prevents numerical instability from propagating
    import math

    if math.isnan(synapse.weight) or math.isinf(synapse.weight):
        # Reset to a safe middle value if weight becomes invalid
        synapse.weight = (weight_min + weight_max) / 2.0
    else:
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
    """Apply Spike-Timing-Dependent Plasticity (STDP) learning rule.

    STDP is a biologically realistic learning rule where the weight change
    depends on the precise timing between pre- and postsynaptic spikes.

    Temporal learning rules:
    - Δt > 0 (pre before post): Potentiation - Δw = A+ * exp(-Δt/τ+)
    - Δt < 0 (post before pre): Depression - Δw = -A- * exp(Δt/τ-)
    - Δt = 0 (simultaneous): No change

    The exponential decay creates a temporal learning window where
    closely-timed spikes have stronger effects than distant ones.
    This implements causality: pre→post strengthens, post→pre weakens.

    Args:
        synapse: The synapse to update.
        pre_spike_time: Time step when presynaptic neuron spiked.
        post_spike_time: Time step when postsynaptic neuron spiked.
        model: The brain model containing weight bounds.
        tau_plus: Time constant for potentiation window (ms, default: 20).
        tau_minus: Time constant for depression window (ms, default: 20).
        a_plus: Maximum potentiation amplitude (default: 0.01).
        a_minus: Maximum depression amplitude (default: 0.012).
    """
    import math

    plasticity = model.get_plasticity_config()
    weight_min = plasticity["weight_min"]
    weight_max = plasticity["weight_max"]

    # Calculate spike time difference (Δt)
    # Positive: pre fired before post (causal relationship)
    # Negative: post fired before pre (acausal relationship)
    delta_t = post_spike_time - pre_spike_time

    if delta_t > 0:
        # Causal timing: Pre→Post suggests synapse contributed to post firing
        # Apply Long-Term Potentiation (LTP) with exponential decay
        # Closer spikes in time → stronger potentiation
        delta_w = a_plus * math.exp(-delta_t / tau_plus)
    elif delta_t < 0:
        # Acausal timing: Post fired before pre (or pre ineffective)
        # Apply Long-Term Depression (LTD) with exponential decay
        # Note: delta_t is negative, so exp(delta_t/tau_minus) decays as |delta_t| increases
        delta_w = -a_minus * math.exp(delta_t / tau_minus)
    else:
        # Simultaneous spikes: no temporal information
        delta_w = 0.0

    # Apply weight change with hard bounds
    synapse.weight += delta_w

    # Check for NaN/Inf values and clip to valid range
    # This prevents numerical instability from propagating
    if math.isnan(synapse.weight) or math.isinf(synapse.weight):
        # Reset to a safe middle value if weight becomes invalid
        synapse.weight = (weight_min + weight_max) / 2.0
    else:
        synapse.weight = max(weight_min, min(weight_max, synapse.weight))
