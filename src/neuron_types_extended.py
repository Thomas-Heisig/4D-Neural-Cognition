"""Extended neuron types for 4D Neural Cognition.

This module implements specialized neuron types:
- Cortical interneurons (Martinotti, Chandelier, Bipolar, Double-bouquet)
- Layer-specific neurons (Spiny stellate, Betz pyramidal, etc.)
- Subcortical neurons (Thalamic, Basal ganglia, Cerebellar, Hippocampal)

References:
- Markram, H., et al. (2004). Interneurons of the neocortical inhibitory system
- DeFelipe, J., et al. (2013). New insights into the classification of cortical GABAergic interneurons
- Shepherd, G.M. (2013). Corticostriatal connectivity and its role in disease
"""

from dataclasses import dataclass
from typing import Dict, Tuple


# ============================================================================
# SPECIALIZED CORTICAL INTERNEURONS
# ============================================================================

def get_martinotti_cell_params() -> Dict:
    """Somatostatin-positive (SOM+) Martinotti cells.
    
    Function: Feedforward inhibition to apical dendrites
    Target: Layer 1 (apical tufts of pyramidal cells)
    Properties:
    - Low-threshold spiking (LTS)
    - Adapting firing pattern
    - Vertical axonal projection
    """
    return {
        "neuron_type": "martinotti",
        "marker": "SOM+",
        "izh_a": 0.02,
        "izh_b": 0.25,
        "izh_c": -65.0,
        "izh_d": 2.0,
        "tau_membrane": 20.0,
        "v_threshold": -50.0,
        "inhibitory": True,
        "target_layer": [1],  # Projects to layer 1
        "firing_pattern": "adapting",
    }


def get_chandelier_cell_params() -> Dict:
    """Parvalbumin-positive (PV+) Chandelier cells (Axo-axonic cells).
    
    Function: Axo-axonic inhibition (controls spike output)
    Target: Axon initial segment of pyramidal cells
    Properties:
    - Fast spiking
    - Precise timing
    - Vertical axonal "candles"
    """
    return {
        "neuron_type": "chandelier",
        "marker": "PV+",
        "izh_a": 0.1,
        "izh_b": 0.2,
        "izh_c": -65.0,
        "izh_d": 2.0,
        "tau_membrane": 5.0,
        "v_threshold": -45.0,
        "inhibitory": True,
        "target": "axon_initial_segment",
        "firing_pattern": "fast_spiking",
    }


def get_bipolar_cell_params() -> Dict:
    """Vasoactive intestinal peptide-positive (VIP+) Bipolar cells.
    
    Function: Disinhibition (inhibits other inhibitory neurons)
    Target: Other interneurons (especially SOM+ cells)
    Properties:
    - Irregular spiking
    - Burst firing
    - Disinhibitory motif
    """
    return {
        "neuron_type": "bipolar_vip",
        "marker": "VIP+",
        "izh_a": 0.02,
        "izh_b": 0.2,
        "izh_c": -55.0,
        "izh_d": 4.0,
        "tau_membrane": 15.0,
        "v_threshold": -50.0,
        "inhibitory": True,
        "target_cell_type": ["martinotti", "basket"],
        "firing_pattern": "irregular_burst",
    }


def get_double_bouquet_cell_params() -> Dict:
    """Calretinin-positive (CR+) Double-bouquet cells.
    
    Function: Columnar inhibition
    Target: Pyramidal and stellate cells in same column
    Properties:
    - Narrow columnar axonal spread
    - Bitufted dendrites
    - Adapting firing
    """
    return {
        "neuron_type": "double_bouquet",
        "marker": "CR+",
        "izh_a": 0.02,
        "izh_b": 0.2,
        "izh_c": -65.0,
        "izh_d": 6.0,
        "tau_membrane": 18.0,
        "v_threshold": -52.0,
        "inhibitory": True,
        "axonal_spread": "columnar",
        "firing_pattern": "adapting",
    }


def get_basket_cell_params() -> Dict:
    """Parvalbumin-positive (PV+) Basket cells.
    
    Function: Perisomatic inhibition of pyramidal cells
    Target: Soma and proximal dendrites
    Properties:
    - Fast spiking
    - Broad axonal arbor
    - Strong inhibition
    """
    return {
        "neuron_type": "basket",
        "marker": "PV+",
        "izh_a": 0.1,
        "izh_b": 0.2,
        "izh_c": -65.0,
        "izh_d": 2.0,
        "tau_membrane": 5.0,
        "v_threshold": -45.0,
        "inhibitory": True,
        "target": "perisomatic",
        "firing_pattern": "fast_spiking",
    }


# ============================================================================
# LAYER-SPECIFIC CORTICAL NEURONS
# ============================================================================

def get_cajal_retzius_cell_params() -> Dict:
    """Cajal-Retzius cells in Layer 1.
    
    Function: Development, guidance, reelin secretion
    Mostly present during development
    Properties:
    - Horizontal axonal spread
    - Secretes reelin (migration guidance)
    """
    return {
        "neuron_type": "cajal_retzius",
        "layer": 1,
        "izh_a": 0.02,
        "izh_b": 0.2,
        "izh_c": -65.0,
        "izh_d": 8.0,
        "developmental": True,
        "secretes": "reelin",
        "inhibitory": False,
    }


def get_spiny_stellate_params() -> Dict:
    """Spiny stellate cells in Layer 4.
    
    Function: Thalamic input processing
    Target: Layer 2/3 pyramidal cells
    Properties:
    - Regular spiking
    - Star-shaped dendrites
    - Main target of thalamocortical input
    """
    return {
        "neuron_type": "spiny_stellate",
        "layer": 4,
        "izh_a": 0.02,
        "izh_b": 0.2,
        "izh_c": -65.0,
        "izh_d": 8.0,
        "tau_membrane": 12.0,
        "v_threshold": -50.0,
        "inhibitory": False,
        "input_source": "thalamus",
        "firing_pattern": "regular_spiking",
    }


def get_betz_pyramidal_params() -> Dict:
    """Betz pyramidal cells in Layer 5B (primary motor cortex).
    
    Function: Motor output to spinal cord
    Target: Corticospinal tract
    Properties:
    - Very large soma (up to 100 μm)
    - Long-range projection
    - Thick apical dendrite
    """
    return {
        "neuron_type": "betz_pyramidal",
        "layer": 5,
        "sublayer": "5B",
        "izh_a": 0.02,
        "izh_b": 0.2,
        "izh_c": -50.0,
        "izh_d": 2.0,
        "tau_membrane": 20.0,
        "v_threshold": -50.0,
        "inhibitory": False,
        "soma_diameter": 100.0,  # μm
        "projection_target": "spinal_cord",
        "firing_pattern": "regular_spiking",
    }


def get_meynert_cell_params() -> Dict:
    """Meynert cells (giant pyramidal cells).
    
    Function: Large corticocortical projections
    Found in various cortical areas
    Properties:
    - Very large pyramidal neurons
    - Long-range connections
    - High metabolic demand
    """
    return {
        "neuron_type": "meynert",
        "layer": 5,
        "izh_a": 0.02,
        "izh_b": 0.2,
        "izh_c": -55.0,
        "izh_d": 4.0,
        "tau_membrane": 20.0,
        "v_threshold": -50.0,
        "inhibitory": False,
        "soma_diameter": 80.0,
        "projection_type": "corticocortical",
        "firing_pattern": "bursting",
    }


# ============================================================================
# THALAMIC NEURONS
# ============================================================================

def get_thalamocortical_relay_params() -> Dict:
    """Thalamocortical (TC) relay neurons.
    
    Function: Relay sensory information to cortex
    Properties:
    - Tonic and burst firing modes
    - T-type Ca2+ channels
    - Receives sensory input and cortical feedback
    """
    return {
        "neuron_type": "thalamocortical_relay",
        "region": "thalamus",
        "izh_a": 0.02,
        "izh_b": 0.25,
        "izh_c": -65.0,
        "izh_d": 0.05,
        "firing_modes": ["tonic", "burst"],
        "ca_channel_type": "T",
        "inhibitory": False,
    }


def get_reticular_nucleus_params() -> Dict:
    """Thalamic reticular nucleus (TRN) neurons.
    
    Function: Gating and attention (inhibits thalamic relay cells)
    Properties:
    - GABAergic
    - Burst firing
    - Reciprocal connections with relay cells
    """
    return {
        "neuron_type": "reticular",
        "region": "thalamus",
        "izh_a": 0.1,
        "izh_b": 0.25,
        "izh_c": -65.0,
        "izh_d": 2.0,
        "inhibitory": True,
        "firing_pattern": "burst",
        "function": "attentional_gating",
    }


# ============================================================================
# BASAL GANGLIA NEURONS
# ============================================================================

def get_msn_d1_params() -> Dict:
    """Medium Spiny Neuron (MSN) with D1 dopamine receptors.
    
    Function: Direct pathway (Go signal)
    Properties:
    - GABAergic projection to GPi/SNr
    - Facilitated by dopamine
    - Up-state/down-state dynamics
    """
    return {
        "neuron_type": "msn_d1",
        "region": "striatum",
        "izh_a": 0.01,
        "izh_b": -20.0,  # Special MSN dynamics
        "izh_c": -65.0,
        "izh_d": 8.0,
        "dopamine_receptor": "D1",
        "pathway": "direct",
        "inhibitory": True,
        "projection_target": ["GPi", "SNr"],
    }


def get_msn_d2_params() -> Dict:
    """Medium Spiny Neuron (MSN) with D2 dopamine receptors.
    
    Function: Indirect pathway (NoGo signal)
    Properties:
    - GABAergic projection to GPe
    - Inhibited by dopamine
    - Up-state/down-state dynamics
    """
    return {
        "neuron_type": "msn_d2",
        "region": "striatum",
        "izh_a": 0.01,
        "izh_b": -20.0,
        "izh_c": -65.0,
        "izh_d": 8.0,
        "dopamine_receptor": "D2",
        "pathway": "indirect",
        "inhibitory": True,
        "projection_target": "GPe",
    }


def get_stn_params() -> Dict:
    """Subthalamic nucleus (STN) neurons.
    
    Function: Hyperdirect pathway, action inhibition
    Properties:
    - Glutamatergic
    - High-frequency firing
    - Receives cortical input
    """
    return {
        "neuron_type": "stn",
        "region": "basal_ganglia",
        "izh_a": 0.005,
        "izh_b": 0.265,
        "izh_c": -65.0,
        "izh_d": 2.0,
        "inhibitory": False,
        "firing_rate": "high",
        "pathway": "hyperdirect",
    }


def get_snc_dopamine_params() -> Dict:
    """Substantia nigra pars compacta (SNc) dopamine neurons.
    
    Function: Reward prediction error signaling
    Properties:
    - Dopaminergic
    - Pacemaker activity
    - Modulated by reward
    """
    return {
        "neuron_type": "snc_dopamine",
        "region": "substantia_nigra",
        "izh_a": 0.001,
        "izh_b": 0.2,
        "izh_c": -65.0,
        "izh_d": 8.0,
        "neurotransmitter": "dopamine",
        "firing_pattern": "pacemaker",
        "function": "reward_prediction_error",
    }


# ============================================================================
# CEREBELLAR NEURONS
# ============================================================================

def get_purkinje_cell_params() -> Dict:
    """Cerebellar Purkinje cells.
    
    Function: Sole output of cerebellar cortex
    Properties:
    - Elaborate dendritic tree (planar)
    - Complex spikes and simple spikes
    - GABAergic to deep cerebellar nuclei
    """
    return {
        "neuron_type": "purkinje",
        "region": "cerebellum",
        "izh_a": 0.02,
        "izh_b": 0.2,
        "izh_c": -65.0,
        "izh_d": 4.0,
        "inhibitory": True,
        "spike_types": ["simple", "complex"],
        "dendritic_spines": 100000,  # Extensive
    }


def get_granule_cell_params() -> Dict:
    """Cerebellar granule cells.
    
    Function: Expansion recoding of mossy fiber input
    Properties:
    - Smallest neurons in brain
    - Most numerous neurons
    - Parallel fiber axons
    """
    return {
        "neuron_type": "granule",
        "region": "cerebellum",
        "izh_a": 0.02,
        "izh_b": 0.2,
        "izh_c": -65.0,
        "izh_d": 8.0,
        "inhibitory": False,
        "soma_diameter": 5.0,  # Very small
        "axon_type": "parallel_fiber",
    }


def get_golgi_cell_params() -> Dict:
    """Cerebellar Golgi cells.
    
    Function: Feedback inhibition to granule cells
    Properties:
    - Large interneurons
    - GABAergic
    - Regulate granule cell activity
    """
    return {
        "neuron_type": "golgi",
        "region": "cerebellum",
        "izh_a": 0.1,
        "izh_b": 0.2,
        "izh_c": -65.0,
        "izh_d": 2.0,
        "inhibitory": True,
        "target": "granule_cells",
        "firing_pattern": "regular",
    }


# ============================================================================
# HIPPOCAMPAL NEURONS
# ============================================================================

def get_ca1_pyramidal_params() -> Dict:
    """CA1 pyramidal neurons.
    
    Function: Spatial memory, place cells
    Properties:
    - Regular spiking
    - Long apical dendrite
    - Theta rhythm modulation
    """
    return {
        "neuron_type": "ca1_pyramidal",
        "region": "hippocampus",
        "subregion": "CA1",
        "izh_a": 0.02,
        "izh_b": 0.2,
        "izh_c": -65.0,
        "izh_d": 8.0,
        "inhibitory": False,
        "function": "place_cell",
        "rhythm": "theta",
    }


def get_ca3_pyramidal_params() -> Dict:
    """CA3 pyramidal neurons.
    
    Function: Pattern completion, associative memory
    Properties:
    - Recurrent collaterals
    - Bursting
    - Auto-associative network
    """
    return {
        "neuron_type": "ca3_pyramidal",
        "region": "hippocampus",
        "subregion": "CA3",
        "izh_a": 0.02,
        "izh_b": 0.2,
        "izh_c": -50.0,
        "izh_d": 2.0,
        "inhibitory": False,
        "firing_pattern": "bursting",
        "recurrent_collaterals": True,
    }


def get_dentate_granule_params() -> Dict:
    """Dentate gyrus granule cells.
    
    Function: Pattern separation
    Properties:
    - Sparse activity
    - Neurogenesis throughout life
    - Gate to hippocampus
    """
    return {
        "neuron_type": "dentate_granule",
        "region": "hippocampus",
        "subregion": "dentate_gyrus",
        "izh_a": 0.02,
        "izh_b": 0.2,
        "izh_c": -65.0,
        "izh_d": 8.0,
        "inhibitory": False,
        "neurogenesis": True,
        "activity_level": "sparse",
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_neuron_type_params(neuron_type: str) -> Dict:
    """Get parameters for a specific neuron type.
    
    Args:
        neuron_type: Name of neuron type
        
    Returns:
        Dictionary of neuron parameters
    """
    type_map = {
        # Cortical interneurons
        "martinotti": get_martinotti_cell_params,
        "chandelier": get_chandelier_cell_params,
        "bipolar_vip": get_bipolar_cell_params,
        "double_bouquet": get_double_bouquet_cell_params,
        "basket": get_basket_cell_params,
        
        # Layer-specific
        "cajal_retzius": get_cajal_retzius_cell_params,
        "spiny_stellate": get_spiny_stellate_params,
        "betz_pyramidal": get_betz_pyramidal_params,
        "meynert": get_meynert_cell_params,
        
        # Thalamus
        "thalamocortical_relay": get_thalamocortical_relay_params,
        "reticular": get_reticular_nucleus_params,
        
        # Basal ganglia
        "msn_d1": get_msn_d1_params,
        "msn_d2": get_msn_d2_params,
        "stn": get_stn_params,
        "snc_dopamine": get_snc_dopamine_params,
        
        # Cerebellum
        "purkinje": get_purkinje_cell_params,
        "granule": get_granule_cell_params,
        "golgi": get_golgi_cell_params,
        
        # Hippocampus
        "ca1_pyramidal": get_ca1_pyramidal_params,
        "ca3_pyramidal": get_ca3_pyramidal_params,
        "dentate_granule": get_dentate_granule_params,
    }
    
    if neuron_type in type_map:
        return type_map[neuron_type]()
    else:
        # Return default excitatory parameters
        return {
            "neuron_type": neuron_type,
            "izh_a": 0.02,
            "izh_b": 0.2,
            "izh_c": -65.0,
            "izh_d": 8.0,
            "inhibitory": False,
        }


def create_cortical_column_neurons(num_neurons: int) -> list[Dict]:
    """Create a realistic distribution of neuron types for a cortical column.
    
    Args:
        num_neurons: Total number of neurons
        
    Returns:
        List of neuron parameter dictionaries
    """
    # Realistic cortical distribution:
    # ~80% excitatory pyramidal neurons
    # ~20% inhibitory interneurons
    #   - ~40% PV+ (basket, chandelier)
    #   - ~30% SOM+ (martinotti)
    #   - ~30% VIP+ and others (bipolar, double-bouquet)
    
    neurons = []
    
    # Excitatory neurons (80%)
    num_excitatory = int(num_neurons * 0.8)
    for _ in range(num_excitatory):
        neurons.append({
            "neuron_type": "regular_spiking",
            "izh_a": 0.02,
            "izh_b": 0.2,
            "izh_c": -65.0,
            "izh_d": 8.0,
            "inhibitory": False,
        })
    
    # Inhibitory neurons (20%)
    num_inhibitory = num_neurons - num_excitatory
    
    # PV+ cells (40% of inhibitory)
    num_pv = int(num_inhibitory * 0.4)
    for _ in range(num_pv // 2):
        neurons.append(get_basket_cell_params())
    for _ in range(num_pv - num_pv // 2):
        neurons.append(get_chandelier_cell_params())
    
    # SOM+ cells (30% of inhibitory)
    num_som = int(num_inhibitory * 0.3)
    for _ in range(num_som):
        neurons.append(get_martinotti_cell_params())
    
    # VIP+ and others (30% of inhibitory)
    num_other = num_inhibitory - num_pv - num_som
    for _ in range(num_other // 2):
        neurons.append(get_bipolar_cell_params())
    for _ in range(num_other - num_other // 2):
        neurons.append(get_double_bouquet_cell_params())
    
    return neurons
