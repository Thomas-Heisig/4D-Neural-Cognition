"""Different neuron model implementations for 4D Neural Cognition.

This module provides:
- Leaky Integrate-and-Fire (LIF) model
- Izhikevich model with multiple neuron types
- Support for excitatory and inhibitory neurons
"""

from typing import Dict, Tuple

import numpy as np

try:
    from .brain_model import Neuron
except ImportError:
    from brain_model import Neuron


def update_lif_neuron(
    neuron: Neuron,
    synaptic_input: float,
    dt: float = 1.0
) -> Tuple[bool, float]:
    """Update neuron using Leaky Integrate-and-Fire (LIF) model.
    
    Args:
        neuron: Neuron to update
        synaptic_input: Total synaptic input current
        dt: Time step size
        
    Returns:
        Tuple of (spiked: bool, new_membrane_potential: float)
    """
    params = neuron.params
    
    # LIF parameters
    v_rest = params.get("v_rest", -65.0)
    v_threshold = params.get("v_threshold", -50.0)
    v_reset = params.get("v_reset", -65.0)
    tau_membrane = params.get("tau_membrane", 10.0)
    
    # Leak current
    leak_current = (v_rest - neuron.v_membrane) / tau_membrane
    
    # Total current
    total_current = leak_current + synaptic_input + neuron.external_input
    
    # Update membrane potential
    new_v = neuron.v_membrane + dt * total_current
    
    # Check for spike
    spiked = new_v >= v_threshold
    
    if spiked:
        new_v = v_reset
    
    return spiked, new_v


def update_izhikevich_neuron(
    neuron: Neuron,
    synaptic_input: float,
    dt: float = 1.0
) -> Tuple[bool, float, float]:
    """Update neuron using Izhikevich model.
    
    The Izhikevich model can reproduce various neuron types including:
    - Regular spiking (RS)
    - Fast spiking (FS) 
    - Bursting
    - Chattering
    
    Args:
        neuron: Neuron to update
        synaptic_input: Total synaptic input current
        dt: Time step size
        
    Returns:
        Tuple of (spiked: bool, new_v: float, new_u: float)
    """
    params = neuron.params
    
    # Get neuron type-specific parameters
    a, b, c, d = get_izhikevich_parameters(neuron.neuron_type, params)
    
    v = neuron.v_membrane
    u = neuron.u_recovery
    
    # Izhikevich equations
    # dv/dt = 0.04*v^2 + 5*v + 140 - u + I
    # du/dt = a*(b*v - u)
    
    total_input = synaptic_input + neuron.external_input
    
    # Update using Euler method
    dv = (0.04 * v * v + 5 * v + 140 - u + total_input) * dt
    du = a * (b * v - u) * dt
    
    new_v = v + dv
    new_u = u + du
    
    # Check for spike
    spiked = new_v >= 30.0
    
    if spiked:
        new_v = c  # Reset voltage
        new_u = new_u + d  # Reset recovery
    
    return spiked, new_v, new_u


def get_izhikevich_parameters(
    neuron_type: str,
    params: Dict = None
) -> Tuple[float, float, float, float]:
    """Get Izhikevich model parameters for different neuron types.
    
    Args:
        neuron_type: Type of neuron
        params: Optional parameter overrides
        
    Returns:
        Tuple of (a, b, c, d) parameters
    """
    if params is None:
        params = {}
    
    # Default parameters for different neuron types
    if neuron_type == "regular_spiking":
        # Regular spiking (RS) - excitatory cortical neurons
        defaults = {
            "a": 0.02,
            "b": 0.2,
            "c": -65.0,
            "d": 8.0
        }
    elif neuron_type == "fast_spiking":
        # Fast spiking (FS) - inhibitory interneurons
        defaults = {
            "a": 0.1,
            "b": 0.2,
            "c": -65.0,
            "d": 2.0
        }
    elif neuron_type == "bursting":
        # Intrinsically bursting (IB)
        defaults = {
            "a": 0.02,
            "b": 0.2,
            "c": -55.0,
            "d": 4.0
        }
    elif neuron_type == "inhibitory":
        # Generic inhibitory neuron
        defaults = {
            "a": 0.1,
            "b": 0.2,
            "c": -65.0,
            "d": 2.0
        }
    else:
        # Default excitatory neuron
        defaults = {
            "a": 0.02,
            "b": 0.2,
            "c": -65.0,
            "d": 8.0
        }
    
    # Allow parameter overrides
    a = params.get("izh_a", defaults["a"])
    b = params.get("izh_b", defaults["b"])
    c = params.get("izh_c", defaults["c"])
    d = params.get("izh_d", defaults["d"])
    
    return a, b, c, d


def update_neuron(
    neuron: Neuron,
    synaptic_input: float,
    dt: float = 1.0
) -> bool:
    """Update neuron using its specified model type.
    
    Args:
        neuron: Neuron to update
        synaptic_input: Total synaptic input
        dt: Time step
        
    Returns:
        True if neuron spiked
    """
    if neuron.model_type == "izhikevich":
        spiked, new_v, new_u = update_izhikevich_neuron(neuron, synaptic_input, dt)
        neuron.v_membrane = new_v
        neuron.u_recovery = new_u
    else:
        # Default to LIF
        spiked, new_v = update_lif_neuron(neuron, synaptic_input, dt)
        neuron.v_membrane = new_v
    
    return spiked


def create_balanced_network_types(
    num_neurons: int,
    inhibitory_fraction: float = 0.2,
    rng: np.random.Generator = None
) -> list[str]:
    """Create balanced mix of neuron types for network initialization.
    
    Args:
        num_neurons: Total number of neurons
        inhibitory_fraction: Fraction of inhibitory neurons (typically 0.2)
        rng: Random number generator
        
    Returns:
        List of neuron types
    """
    if rng is None:
        rng = np.random.default_rng()
    
    num_inhibitory = int(num_neurons * inhibitory_fraction)
    num_excitatory = num_neurons - num_inhibitory
    
    # Distribute excitatory types
    # ~80% regular spiking, ~20% bursting
    num_bursting = int(num_excitatory * 0.2)
    num_regular = num_excitatory - num_bursting
    
    types = (
        ["regular_spiking"] * num_regular +
        ["bursting"] * num_bursting +
        ["fast_spiking"] * num_inhibitory
    )
    
    # Shuffle to randomize positions
    rng.shuffle(types)
    
    return types


def get_synapse_type_from_presynaptic(pre_neuron: Neuron) -> str:
    """Determine synapse type based on presynaptic neuron.
    
    Dale's principle: A neuron releases the same neurotransmitters at all synapses.
    
    Args:
        pre_neuron: Presynaptic neuron
        
    Returns:
        "excitatory" or "inhibitory"
    """
    if pre_neuron.is_inhibitory():
        return "inhibitory"
    return "excitatory"


def calculate_excitation_inhibition_balance(
    excitatory_input: float,
    inhibitory_input: float
) -> Dict[str, float]:
    """Calculate excitation-inhibition (E-I) balance metrics.
    
    Args:
        excitatory_input: Total excitatory input
        inhibitory_input: Total inhibitory input (positive value)
        
    Returns:
        Dictionary with E-I balance metrics
    """
    total = excitatory_input + inhibitory_input
    
    if total > 0:
        e_fraction = excitatory_input / total
        i_fraction = inhibitory_input / total
        balance_ratio = excitatory_input / max(inhibitory_input, 0.01)
    else:
        e_fraction = 0.0
        i_fraction = 0.0
        balance_ratio = 0.0
    
    return {
        "excitatory_input": excitatory_input,
        "inhibitory_input": inhibitory_input,
        "e_fraction": e_fraction,
        "i_fraction": i_fraction,
        "balance_ratio": balance_ratio,
    }
