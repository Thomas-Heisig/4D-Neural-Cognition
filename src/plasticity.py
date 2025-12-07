"""Hebbian plasticity rules for synapse learning.

Extended with advanced plasticity mechanisms:
- Homeostatic plasticity (synaptic scaling)
- Metaplasticity (BCM-like threshold modulation)
- Short-term plasticity (facilitation and depression)
"""

from typing import TYPE_CHECKING, Dict
from dataclasses import dataclass, field

if TYPE_CHECKING:
    try:
        from .brain_model import BrainModel, Synapse, Neuron
    except ImportError:
        from brain_model import BrainModel, Synapse, Neuron


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


# ============================================================================
# Homeostatic Plasticity
# ============================================================================


def homeostatic_scaling(
    neurons: Dict[int, "Neuron"],
    synapses: list["Synapse"],
    target_rate: float = 5.0,
    time_window: int = 1000,
    scaling_rate: float = 0.01,
    model: "BrainModel" = None,
) -> None:
    """Apply homeostatic synaptic scaling to maintain target firing rates.
    
    Homeostatic plasticity stabilizes network activity by globally scaling
    synaptic weights to maintain neurons near a target firing rate. This
    prevents runaway excitation or silencing after Hebbian changes.
    
    Implementation:
    - Multiplicative scaling of all incoming synapses to each neuron
    - Scaling factor based on deviation from target firing rate
    - Slow adjustment to avoid disrupting learned patterns
    
    Args:
        neurons: Dictionary of neurons
        synapses: List of synapses
        target_rate: Target firing rate in Hz (default: 5.0)
        time_window: Time window for rate calculation (default: 1000 steps)
        scaling_rate: Rate of homeostatic adjustment (default: 0.01)
        model: Brain model for weight bounds
    """
    # Get weight bounds
    if model is not None:
        plasticity = model.get_plasticity_config()
        weight_min = plasticity["weight_min"]
        weight_max = plasticity["weight_max"]
    else:
        weight_min = 0.0
        weight_max = 1.0
    
    # Calculate firing rate for each neuron
    firing_rates = {}
    for neuron_id, neuron in neurons.items():
        # Estimate firing rate from recent activity
        # In a real implementation, this would use spike history
        # Here we use a placeholder based on membrane potential
        rate_estimate = max(0.0, (neuron.v_membrane + 65.0) / 10.0)
        firing_rates[neuron_id] = rate_estimate
    
    # Group synapses by postsynaptic neuron
    synapses_by_post = {}
    for synapse in synapses:
        if synapse.post_id not in synapses_by_post:
            synapses_by_post[synapse.post_id] = []
        synapses_by_post[synapse.post_id].append(synapse)
    
    # Apply scaling to each neuron's incoming synapses
    for neuron_id, incoming_synapses in synapses_by_post.items():
        if neuron_id not in firing_rates:
            continue
        
        current_rate = firing_rates[neuron_id]
        
        # Calculate scaling factor
        # If firing too much, scale down; if too little, scale up
        rate_ratio = target_rate / max(current_rate, 0.1)
        scaling_factor = 1.0 + (rate_ratio - 1.0) * scaling_rate
        
        # Apply multiplicative scaling to all incoming synapses
        for synapse in incoming_synapses:
            synapse.weight *= scaling_factor
            # Clip to bounds
            synapse.weight = max(weight_min, min(weight_max, synapse.weight))


# ============================================================================
# Metaplasticity
# ============================================================================


@dataclass
class BCMThreshold:
    """BCM (Bienenstock-Cooper-Munro) sliding threshold for metaplasticity.
    
    The threshold for LTP/LTD induction slides based on recent postsynaptic
    activity, implementing metaplasticity: the plasticity of plasticity.
    """
    
    theta: float = 0.5  # Current threshold
    target_rate: float = 5.0  # Target average activity
    tau: float = 1000.0  # Time constant for threshold adjustment
    theta_min: float = 0.1  # Minimum threshold
    theta_max: float = 2.0  # Maximum threshold
    
    def update(self, postsynaptic_rate: float, dt: float = 1.0) -> None:
        """Update threshold based on recent postsynaptic activity.
        
        Args:
            postsynaptic_rate: Recent firing rate of postsynaptic neuron
            dt: Time step
        """
        # Threshold moves toward squared activity (BCM rule)
        # Higher activity -> higher threshold (harder to potentiate)
        target_theta = postsynaptic_rate ** 2 / self.target_rate
        
        # Slow exponential approach to target
        self.theta += (target_theta - self.theta) * (dt / self.tau)
        
        # Clip to bounds
        self.theta = max(self.theta_min, min(self.theta_max, self.theta))
    
    def get_threshold(self) -> float:
        """Get current threshold value."""
        return self.theta


def bcm_plasticity(
    synapse: "Synapse",
    pre_rate: float,
    post_rate: float,
    bcm_threshold: BCMThreshold,
    learning_rate: float = 0.01,
    model: "BrainModel" = None,
) -> None:
    """Apply BCM (Bienenstock-Cooper-Munro) metaplasticity rule.
    
    BCM implements metaplasticity where the threshold for LTP/LTD slides
    based on postsynaptic activity history. This stabilizes learning while
    maintaining plasticity.
    
    Learning rule:
    - Δw = η * pre_rate * post_rate * (post_rate - θ)
    - When post_rate > θ: LTP (strengthening)
    - When post_rate < θ: LTD (weakening)
    - θ adapts to maintain target activity
    
    Args:
        synapse: The synapse to update
        pre_rate: Presynaptic firing rate
        post_rate: Postsynaptic firing rate
        bcm_threshold: BCM threshold object
        learning_rate: Learning rate
        model: Brain model for weight bounds
    """
    import math
    
    # Get weight bounds
    if model is not None:
        plasticity = model.get_plasticity_config()
        weight_min = plasticity["weight_min"]
        weight_max = plasticity["weight_max"]
    else:
        weight_min = 0.0
        weight_max = 1.0
    
    # BCM learning rule
    theta = bcm_threshold.get_threshold()
    delta_w = learning_rate * pre_rate * post_rate * (post_rate - theta)
    
    # Apply weight change
    synapse.weight += delta_w
    
    # Check for NaN/Inf and clip
    if math.isnan(synapse.weight) or math.isinf(synapse.weight):
        synapse.weight = (weight_min + weight_max) / 2.0
    else:
        synapse.weight = max(weight_min, min(weight_max, synapse.weight))


# ============================================================================
# Short-Term Plasticity
# ============================================================================


@dataclass
class ShortTermPlasticityState:
    """State variables for short-term synaptic plasticity.
    
    Implements the Tsodyks-Markram model with facilitation and depression.
    """
    
    u: float = 0.5  # Utilization of synaptic efficacy (facilitation variable)
    x: float = 1.0  # Fraction of available resources (depression variable)
    
    # Parameters
    U: float = 0.5  # Baseline utilization probability
    tau_facil: float = 50.0  # Facilitation time constant (ms)
    tau_rec: float = 800.0  # Recovery time constant for depression (ms)
    
    # Facilitation-dominant: U low (~0.1), tau_facil < tau_rec
    # Depression-dominant: U high (~0.5), tau_facil > tau_rec
    
    def reset(self) -> None:
        """Reset to baseline state."""
        self.u = self.U
        self.x = 1.0
    
    def update_on_spike(self) -> float:
        """Update state when presynaptic spike occurs.
        
        Returns:
            Effective synaptic weight multiplier
        """
        # Update utilization (facilitation)
        # u increases toward 1 with each spike
        self.u = self.u + self.U * (1 - self.u)
        
        # Calculate effective release before depression
        effective_release = self.u * self.x
        
        # Apply depression: decrease available resources  
        # x decreases with each release
        self.x = self.x * (1 - self.u)
        
        return effective_release
    
    def decay(self, dt: float = 1.0) -> None:
        """Decay toward baseline between spikes.
        
        Args:
            dt: Time step
        """
        # Facilitation decays toward baseline U
        self.u = self.u + (self.U - self.u) * (dt / self.tau_facil)
        
        # Resources recover toward 1.0
        self.x = self.x + (1.0 - self.x) * (dt / self.tau_rec)


def apply_short_term_plasticity(
    synapse: "Synapse",
    stp_state: ShortTermPlasticityState,
    presynaptic_spike: bool,
    dt: float = 1.0,
) -> float:
    """Apply short-term plasticity to compute effective synaptic weight.
    
    Args:
        synapse: The synapse
        stp_state: Short-term plasticity state
        presynaptic_spike: Whether presynaptic neuron spiked
        dt: Time step
        
    Returns:
        Effective weight considering short-term plasticity
    """
    if presynaptic_spike:
        # On spike: update facilitation/depression
        multiplier = stp_state.update_on_spike()
    else:
        # Between spikes: decay toward baseline
        stp_state.decay(dt)
        multiplier = stp_state.u * stp_state.x
    
    # Return effective weight
    return synapse.weight * multiplier


def create_facilitating_synapse() -> ShortTermPlasticityState:
    """Create a facilitation-dominant synapse (e.g., for temporal integration).
    
    Returns:
        STP state configured for facilitation
    """
    return ShortTermPlasticityState(
        U=0.1,  # Low baseline utilization
        tau_facil=50.0,  # Fast facilitation
        tau_rec=800.0,  # Slow recovery
    )


def create_depressing_synapse() -> ShortTermPlasticityState:
    """Create a depression-dominant synapse (e.g., for novelty detection).
    
    Returns:
        STP state configured for depression
    """
    return ShortTermPlasticityState(
        U=0.5,  # High baseline utilization
        tau_facil=800.0,  # Slow facilitation
        tau_rec=100.0,  # Fast recovery
    )
