"""Cortical layer organization for 4D Neural Cognition.

This module implements the 6-layer laminar structure of neocortex:
- Layer 1 (Molecular): Apical dendrites, horizontal connections
- Layer 2/3: Cortico-cortical connections (associative)
- Layer 4: Thalamic input (sensory)
- Layer 5: Output to subcortical structures
- Layer 6: Feedback to thalamus

Also includes columnar and minicolumnar organization.

References:
- Douglas, R.J., & Martin, K.A. (2004). Neuronal circuits of the neocortex
- Harris, K.D., & Shepherd, G.M. (2015). The neocortical circuit
- Mountcastle, V.B. (1997). The columnar organization of the neocortex
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class CorticalLayer:
    """Represents a cortical layer with its characteristic properties."""
    
    layer_id: int  # 1, 2, 3, 4, 5, 6
    name: str
    
    # Thickness (relative)
    thickness: float = 1.0
    
    # Cell density (cells per unit volume)
    cell_density: float = 1.0
    
    # Cell type distribution
    pyramidal_fraction: float = 0.8
    interneuron_fraction: float = 0.2
    
    # Connectivity patterns
    afferent_sources: List[str] = field(default_factory=list)  # Where inputs come from
    efferent_targets: List[str] = field(default_factory=list)  # Where outputs go
    
    # Neuron types predominant in this layer
    predominant_neuron_types: List[str] = field(default_factory=list)
    
    # Functional role
    functional_role: str = ""


def create_layer_1() -> CorticalLayer:
    """Layer 1 (Molecular layer).
    
    Characteristics:
    - Few cell bodies (mostly Cajal-Retzius cells)
    - Dense neuropil (dendrites and axons)
    - Apical dendrites from deeper layers
    - Horizontal connections for integration
    """
    return CorticalLayer(
        layer_id=1,
        name="Layer 1 (Molecular)",
        thickness=0.5,
        cell_density=0.1,  # Very sparse
        pyramidal_fraction=0.0,
        interneuron_fraction=1.0,
        afferent_sources=["thalamus_matrix", "layer_2_3", "layer_5"],
        efferent_targets=["layer_2_3_apical", "layer_5_apical"],
        predominant_neuron_types=["cajal_retzius"],
        functional_role="Distal integration of feedback and feedforward signals"
    )


def create_layer_2_3() -> CorticalLayer:
    """Layer 2/3 (Supragranular).
    
    Characteristics:
    - Small to medium pyramidal cells
    - Dense local connections
    - Cortico-cortical projections
    - Associative processing
    """
    return CorticalLayer(
        layer_id=2,  # Combined 2 and 3
        name="Layer 2/3 (Supragranular)",
        thickness=1.5,
        cell_density=1.2,
        pyramidal_fraction=0.8,
        interneuron_fraction=0.2,
        afferent_sources=["layer_4", "other_cortical_areas", "layer_5"],
        efferent_targets=["other_cortical_areas", "layer_5"],
        predominant_neuron_types=["regular_spiking", "basket", "martinotti"],
        functional_role="Cortico-cortical associative processing"
    )


def create_layer_4() -> CorticalLayer:
    """Layer 4 (Granular - Internal granular).
    
    Characteristics:
    - Dense layer of small neurons
    - Spiny stellate cells (in sensory cortex)
    - Main target of thalamic input
    - Input layer
    """
    return CorticalLayer(
        layer_id=4,
        name="Layer 4 (Granular)",
        thickness=1.0,
        cell_density=1.5,  # Very dense
        pyramidal_fraction=0.6,  # More stellate cells
        interneuron_fraction=0.4,
        afferent_sources=["thalamus_core", "layer_6"],
        efferent_targets=["layer_2_3", "layer_5"],
        predominant_neuron_types=["spiny_stellate", "basket", "chandelier"],
        functional_role="Thalamic input processing and distribution"
    )


def create_layer_5() -> CorticalLayer:
    """Layer 5 (Internal pyramidal).
    
    Characteristics:
    - Large pyramidal cells (including Betz cells in M1)
    - Long-range subcortical projections
    - Thick and thin tufted cells
    - Output layer to subcortex
    """
    return CorticalLayer(
        layer_id=5,
        name="Layer 5 (Internal pyramidal)",
        thickness=1.5,
        cell_density=1.0,
        pyramidal_fraction=0.85,
        interneuron_fraction=0.15,
        afferent_sources=["layer_2_3", "layer_4", "thalamus"],
        efferent_targets=["subcortical", "brainstem", "spinal_cord", "layer_1"],
        predominant_neuron_types=["bursting", "betz_pyramidal", "martinotti"],
        functional_role="Output to subcortical structures and brainstem"
    )


def create_layer_6() -> CorticalLayer:
    """Layer 6 (Multiform).
    
    Characteristics:
    - Diverse cell types
    - Corticothalamic feedback
    - Modulates thalamic activity
    - Interface with white matter
    """
    return CorticalLayer(
        layer_id=6,
        name="Layer 6 (Multiform)",
        thickness=1.2,
        cell_density=0.9,
        pyramidal_fraction=0.75,
        interneuron_fraction=0.25,
        afferent_sources=["layer_4", "layer_5", "thalamus"],
        efferent_targets=["thalamus", "layer_4"],
        predominant_neuron_types=["regular_spiking", "martinotti"],
        functional_role="Corticothalamic feedback and thalamic modulation"
    )


@dataclass
class MiniColumn:
    """Minicolumn - basic computational unit (~50-80 μm diameter).
    
    Minicolumns are vertical arrangements of neurons sharing similar
    receptive field properties.
    """
    
    id: int
    center_x: float
    center_y: float
    
    # Neurons in this minicolumn (by layer)
    neurons_by_layer: Dict[int, List[int]] = field(default_factory=dict)
    
    # Minicolumn parameters
    diameter: float = 50.0  # μm
    vertical_connectivity: float = 0.8  # Strong within-column
    
    def add_neuron(self, neuron_id: int, layer: int) -> None:
        """Add a neuron to this minicolumn.
        
        Args:
            neuron_id: Neuron ID
            layer: Cortical layer (1-6)
        """
        if layer not in self.neurons_by_layer:
            self.neurons_by_layer[layer] = []
        self.neurons_by_layer[layer].append(neuron_id)
    
    def get_vertical_connections(self) -> List[Tuple[int, int]]:
        """Get vertical connections within minicolumn.
        
        Returns:
            List of (pre_id, post_id) tuples
        """
        connections = []
        
        # Layer 4 -> Layer 2/3
        if 4 in self.neurons_by_layer and 2 in self.neurons_by_layer:
            for pre in self.neurons_by_layer[4]:
                for post in self.neurons_by_layer[2]:
                    if np.random.random() < self.vertical_connectivity:
                        connections.append((pre, post))
        
        # Layer 2/3 -> Layer 5
        if 2 in self.neurons_by_layer and 5 in self.neurons_by_layer:
            for pre in self.neurons_by_layer[2]:
                for post in self.neurons_by_layer[5]:
                    if np.random.random() < self.vertical_connectivity * 0.6:
                        connections.append((pre, post))
        
        # Layer 5 -> Layer 6
        if 5 in self.neurons_by_layer and 6 in self.neurons_by_layer:
            for pre in self.neurons_by_layer[5]:
                for post in self.neurons_by_layer[6]:
                    if np.random.random() < self.vertical_connectivity * 0.4:
                        connections.append((pre, post))
        
        return connections


@dataclass
class MacroColumn:
    """Macrocolumn - functional column (~1 mm diameter).
    
    Macrocolumns contain multiple minicolumns and process related
    information (e.g., orientation columns in V1).
    """
    
    id: int
    center_x: float
    center_y: float
    
    # Minicolumns in this macrocolumn
    minicolumns: List[MiniColumn] = field(default_factory=list)
    
    # Macrocolumn parameters
    diameter: float = 1000.0  # μm (1 mm)
    
    # Functional property (e.g., preferred orientation in V1)
    feature_selectivity: Optional[Dict] = None
    
    def add_minicolumn(self, minicolumn: MiniColumn) -> None:
        """Add a minicolumn to this macrocolumn.
        
        Args:
            minicolumn: MiniColumn to add
        """
        self.minicolumns.append(minicolumn)
    
    def get_lateral_connections(self, connection_probability: float = 0.1) -> List[Tuple[int, int]]:
        """Get lateral connections within macrocolumn.
        
        Args:
            connection_probability: Probability of connection between minicolumns
            
        Returns:
            List of (pre_id, post_id) tuples
        """
        connections = []
        
        # Connect neurons in Layer 2/3 across minicolumns (lateral)
        for i, mini1 in enumerate(self.minicolumns):
            for mini2 in self.minicolumns[i+1:]:
                if 2 in mini1.neurons_by_layer and 2 in mini2.neurons_by_layer:
                    for pre in mini1.neurons_by_layer[2]:
                        for post in mini2.neurons_by_layer[2]:
                            if np.random.random() < connection_probability:
                                connections.append((pre, post))
                                # Reciprocal connection
                                if np.random.random() < 0.5:
                                    connections.append((post, pre))
        
        return connections


@dataclass
class CorticalArea:
    """Cortical area with laminar structure.
    
    Represents a functionally-defined cortical region (e.g., V1, M1, A1)
    with 6-layer organization and columnar structure.
    """
    
    name: str
    area_type: str  # "sensory", "motor", "association"
    
    # Layers
    layers: Dict[int, CorticalLayer] = field(default_factory=dict)
    
    # Columnar organization
    macrocolumns: List[MacroColumn] = field(default_factory=list)
    
    # Area boundaries (4D coordinates)
    x_range: Tuple[int, int] = (0, 100)
    y_range: Tuple[int, int] = (0, 100)
    z_range: Tuple[int, int] = (0, 30)
    w_range: Tuple[int, int] = (0, 10)
    
    def __post_init__(self):
        """Initialize layers."""
        if not self.layers:
            self.layers = {
                1: create_layer_1(),
                2: create_layer_2_3(),
                4: create_layer_4(),
                5: create_layer_5(),
                6: create_layer_6(),
            }
    
    def get_layer_for_z(self, z: float) -> int:
        """Determine cortical layer from z-coordinate.
        
        Args:
            z: Z-coordinate within area
            
        Returns:
            Layer number (1-6)
        """
        z_min, z_max = self.z_range
        z_normalized = (z - z_min) / (z_max - z_min)
        
        # Layer boundaries (approximate)
        if z_normalized < 0.1:
            return 1
        elif z_normalized < 0.35:
            return 2  # 2/3
        elif z_normalized < 0.5:
            return 4
        elif z_normalized < 0.75:
            return 5
        else:
            return 6
    
    def assign_neuron_to_layer(
        self,
        neuron_id: int,
        x: int,
        y: int,
        z: int
    ) -> Tuple[int, Optional[MiniColumn]]:
        """Assign neuron to layer and minicolumn.
        
        Args:
            neuron_id: Neuron ID
            x, y, z: 3D coordinates
            
        Returns:
            Tuple of (layer, minicolumn)
        """
        layer = self.get_layer_for_z(z)
        
        # Find or create minicolumn
        minicolumn = self.find_or_create_minicolumn(x, y)
        minicolumn.add_neuron(neuron_id, layer)
        
        return layer, minicolumn
    
    def find_or_create_minicolumn(self, x: float, y: float) -> MiniColumn:
        """Find or create minicolumn at location.
        
        Args:
            x, y: Location
            
        Returns:
            MiniColumn at that location
        """
        # Check existing minicolumns
        for macro in self.macrocolumns:
            for mini in macro.minicolumns:
                dist = np.sqrt((mini.center_x - x)**2 + (mini.center_y - y)**2)
                if dist < mini.diameter / 2:
                    return mini
        
        # Create new minicolumn
        mini = MiniColumn(
            id=sum(len(m.minicolumns) for m in self.macrocolumns),
            center_x=x,
            center_y=y
        )
        
        # Find or create macrocolumn
        macro = self.find_or_create_macrocolumn(x, y)
        macro.add_minicolumn(mini)
        
        return mini
    
    def find_or_create_macrocolumn(self, x: float, y: float) -> MacroColumn:
        """Find or create macrocolumn at location.
        
        Args:
            x, y: Location
            
        Returns:
            MacroColumn at that location
        """
        # Check existing macrocolumns
        for macro in self.macrocolumns:
            dist = np.sqrt((macro.center_x - x)**2 + (macro.center_y - y)**2)
            if dist < macro.diameter / 2:
                return macro
        
        # Create new macrocolumn
        macro = MacroColumn(
            id=len(self.macrocolumns),
            center_x=x,
            center_y=y
        )
        self.macrocolumns.append(macro)
        
        return macro
    
    def get_connectivity_pattern(
        self,
        source_layer: int,
        target_layer: int
    ) -> float:
        """Get connectivity probability between layers.
        
        Args:
            source_layer: Source layer (1-6)
            target_layer: Target layer (1-6)
            
        Returns:
            Connection probability
        """
        # Canonical cortical connectivity
        connectivity_matrix = {
            # From Layer 4
            (4, 2): 0.6,  # Strong to L2/3
            (4, 3): 0.6,
            (4, 5): 0.2,
            (4, 6): 0.1,
            
            # From Layer 2/3
            (2, 2): 0.3,  # Lateral
            (2, 5): 0.4,
            (2, 6): 0.1,
            
            # From Layer 5
            (5, 2): 0.2,  # Feedback
            (5, 1): 0.3,
            (5, 6): 0.2,
            
            # From Layer 6
            (6, 4): 0.3,  # Feedback to input
            (6, 5): 0.1,
        }
        
        # Normalize layer 2/3
        if source_layer == 3:
            source_layer = 2
        if target_layer == 3:
            target_layer = 2
        
        return connectivity_matrix.get((source_layer, target_layer), 0.05)
    
    def generate_laminar_connections(
        self,
        neurons_by_layer: Dict[int, List[int]]
    ) -> List[Tuple[int, int, float]]:
        """Generate connections following laminar patterns.
        
        Args:
            neurons_by_layer: Dictionary mapping layer to neuron IDs
            
        Returns:
            List of (pre_id, post_id, weight) tuples
        """
        connections = []
        
        for source_layer in neurons_by_layer:
            for target_layer in neurons_by_layer:
                conn_prob = self.get_connectivity_pattern(source_layer, target_layer)
                
                if conn_prob > 0:
                    for pre_id in neurons_by_layer[source_layer]:
                        for post_id in neurons_by_layer[target_layer]:
                            if pre_id != post_id and np.random.random() < conn_prob:
                                # Weight depends on layer pair
                                weight = 0.1 * conn_prob
                                connections.append((pre_id, post_id, weight))
        
        return connections


def create_sensory_cortex(name: str) -> CorticalArea:
    """Create a sensory cortical area (e.g., V1, A1, S1).
    
    Sensory cortex has:
    - Prominent Layer 4 (thalamic input)
    - Columnar organization for feature maps
    
    Args:
        name: Area name
        
    Returns:
        Configured CorticalArea
    """
    area = CorticalArea(
        name=name,
        area_type="sensory"
    )
    
    # Enhance Layer 4
    area.layers[4].thickness = 1.5
    area.layers[4].cell_density = 2.0
    
    return area


def create_motor_cortex() -> CorticalArea:
    """Create motor cortex (M1).
    
    Motor cortex has:
    - Reduced or absent Layer 4
    - Large Layer 5 pyramidal cells (Betz cells)
    - Output to spinal cord
    
    Returns:
        Configured CorticalArea
    """
    area = CorticalArea(
        name="M1",
        area_type="motor"
    )
    
    # Reduce Layer 4
    area.layers[4].thickness = 0.3
    area.layers[4].cell_density = 0.5
    
    # Enhance Layer 5
    area.layers[5].thickness = 2.0
    area.layers[5].predominant_neuron_types.append("betz_pyramidal")
    
    return area


def create_association_cortex(name: str) -> CorticalArea:
    """Create association cortex (e.g., PFC, parietal).
    
    Association cortex has:
    - Prominent Layer 2/3 for cortico-cortical connections
    - Complex integration
    
    Args:
        name: Area name
        
    Returns:
        Configured CorticalArea
    """
    area = CorticalArea(
        name=name,
        area_type="association"
    )
    
    # Enhance Layer 2/3
    area.layers[2].thickness = 2.0
    
    return area
