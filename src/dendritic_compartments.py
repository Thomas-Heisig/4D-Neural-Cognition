"""Dendritic compartments and spines for 4D Neural Cognition.

This module implements:
- Dendritic spines (thin, stubby, mushroom, filopodia)
- Dendritic spikes (calcium, sodium, NMDA)
- Active zones and vesicle pools
- Compartmental modeling

References:
- Yuste, R., & Denk, W. (1995). Dendritic spines as basic functional units
- Larkum, M.E., et al. (2009). Synaptic integration in tuft dendrites
- Harnett, M.T., et al. (2012). Nonlinear dendritic integration
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np


@dataclass
class DendriticSpine:
    """Dendritic spine as a postsynaptic site.
    
    Spines are small protrusions from dendrites that receive most
    excitatory synaptic input. They:
    - Compartmentalize calcium
    - Undergo structural plasticity
    - Act as biochemical computation units
    
    Spine types:
    - Thin: learning spines (dynamic)
    - Stubby: immature or transitional
    - Mushroom: memory spines (stable)
    - Filopodia: exploratory (development)
    """
    
    id: int
    parent_neuron_id: int
    
    # Morphology
    spine_type: str = "thin"  # thin, stubby, mushroom, filopodia
    neck_length: float = 1.0  # μm
    neck_diameter: float = 0.1  # μm
    head_volume: float = 0.1  # μm³
    
    # Spine resistance (affects electrical coupling)
    neck_resistance: float = 100.0  # MΩ
    
    # Calcium compartmentalization
    calcium_level: float = 0.1  # μM (resting ~100 nM)
    calcium_threshold_ltp: float = 1.0  # μM
    calcium_threshold_ltd: float = 0.3  # μM
    
    # Associated synapse
    synapse_id: Optional[int] = None
    synaptic_weight: float = 0.1
    
    # AMPA and NMDA receptors
    ampa_receptors: int = 50
    nmda_receptors: int = 10
    
    # Structural plasticity
    age: int = 0
    stability: float = 0.5  # 0=unstable, 1=stable
    growth_signal: float = 0.0
    
    # Actin cytoskeleton dynamics
    actin_polymerization: float = 0.0
    
    def get_electrical_coupling(self) -> float:
        """Get electrical coupling to parent dendrite.
        
        Returns:
            Coupling coefficient (0-1)
        """
        # Higher neck resistance = lower coupling
        return 1.0 / (1.0 + self.neck_resistance / 50.0)
    
    def update_calcium(self, nmda_current: float, vgcc_current: float, dt: float) -> None:
        """Update spine calcium concentration.
        
        Args:
            nmda_current: NMDA receptor current (Ca2+ influx)
            vgcc_current: Voltage-gated Ca2+ channel current
            dt: Time step (ms)
        """
        # Calcium influx
        ca_influx = (-nmda_current - vgcc_current) * 0.01  # Conversion factor
        
        # Buffering and extrusion
        buffer_capacity = 20.0
        extrusion_rate = 0.5  # 1/ms
        
        # Update concentration
        self.calcium_level += ca_influx / (1.0 + buffer_capacity)
        self.calcium_level -= (self.calcium_level - 0.1) * extrusion_rate * dt
        
        # Ensure non-negative
        self.calcium_level = max(0.0, self.calcium_level)
    
    def detect_plasticity_signal(self) -> str:
        """Detect plasticity signal based on calcium level.
        
        Returns:
            "LTP", "LTD", or "none"
        """
        if self.calcium_level > self.calcium_threshold_ltp:
            return "LTP"
        elif self.calcium_level > self.calcium_threshold_ltd:
            return "LTD"
        return "none"
    
    def structural_plasticity(self, dt: float) -> Optional[str]:
        """Apply structural plasticity to spine.
        
        Args:
            dt: Time step
            
        Returns:
            "grow", "shrink", "stabilize", or None
        """
        plasticity_signal = self.detect_plasticity_signal()
        
        if plasticity_signal == "LTP":
            # LTP promotes spine growth and stabilization
            self.growth_signal += 0.1 * dt
            self.actin_polymerization += 0.1 * dt
            self.stability += 0.01 * dt
            
            # Transition to mushroom spine
            if self.spine_type == "thin" and self.growth_signal > 1.0:
                self.spine_type = "mushroom"
                self.head_volume *= 1.5
                self.stability = 0.9
                return "stabilize"
            
            return "grow"
        
        elif plasticity_signal == "LTD":
            # LTD promotes spine shrinkage
            self.growth_signal -= 0.1 * dt
            self.actin_polymerization -= 0.1 * dt
            self.stability -= 0.01 * dt
            
            # Risk of spine elimination
            if self.growth_signal < -1.0:
                return "shrink"
        
        # Decay growth signal
        self.growth_signal *= 0.99
        self.actin_polymerization *= 0.99
        
        return None
    
    def insert_receptors(self, ampa: int = 0, nmda: int = 0) -> None:
        """Insert receptors (e.g., unsilencing).
        
        Args:
            ampa: Number of AMPA receptors to add
            nmda: Number of NMDA receptors to add
        """
        self.ampa_receptors += ampa
        self.nmda_receptors += nmda
    
    def remove_receptors(self, ampa: int = 0, nmda: int = 0) -> None:
        """Remove receptors (e.g., during LTD).
        
        Args:
            ampa: Number of AMPA receptors to remove
            nmda: Number of NMDA receptors to remove
        """
        self.ampa_receptors = max(0, self.ampa_receptors - ampa)
        self.nmda_receptors = max(0, self.nmda_receptors - nmda)


@dataclass
class DendriticBranch:
    """Dendritic branch with compartmental modeling.
    
    Implements active dendritic properties:
    - Dendritic spikes (Ca2+, Na+, NMDA)
    - Non-linear integration
    - Branch-specific plasticity
    """
    
    id: int
    parent_neuron_id: int
    
    # Branch location
    branch_order: int = 1  # 1=primary, 2=secondary, etc.
    distance_from_soma: float = 100.0  # μm
    
    # Compartments along branch
    num_compartments: int = 10
    compartment_voltages: List[float] = field(default_factory=lambda: [-65.0] * 10)
    
    # Active conductances
    na_conductance: float = 0.1  # For Na+ spikes
    ca_conductance: float = 0.5  # For Ca2+ spikes
    nmda_conductance: float = 0.3  # For NMDA spikes
    
    # Spines on this branch
    spines: List[DendriticSpine] = field(default_factory=list)
    
    # Branch excitability
    spike_threshold: float = -40.0  # mV
    last_spike_time: int = -1000
    
    def add_spine(self, spine: DendriticSpine) -> None:
        """Add a spine to this branch.
        
        Args:
            spine: DendriticSpine to add
        """
        self.spines.append(spine)
    
    def detect_dendritic_spike(self, compartment_idx: int) -> Optional[str]:
        """Detect dendritic spike in a compartment.
        
        Args:
            compartment_idx: Compartment index
            
        Returns:
            Spike type: "calcium", "sodium", "nmda", or None
        """
        if compartment_idx >= len(self.compartment_voltages):
            return None
        
        v = self.compartment_voltages[compartment_idx]
        
        if v > self.spike_threshold:
            # Determine spike type based on voltage and location
            if self.distance_from_soma > 200.0:
                # Distal: Ca2+ spikes more likely
                return "calcium"
            elif v > 0:
                # High amplitude: Na+ spike
                return "sodium"
            else:
                # Intermediate: NMDA spike
                return "nmda"
        
        return None
    
    def propagate_spike(
        self,
        spike_type: str,
        origin_compartment: int
    ) -> List[int]:
        """Propagate dendritic spike along branch.
        
        Args:
            spike_type: Type of spike
            origin_compartment: Where spike originated
            
        Returns:
            List of affected compartments
        """
        affected = [origin_compartment]
        
        # Spike propagation characteristics
        if spike_type == "calcium":
            # Ca2+ spikes: slow, decremental
            amplitude = 30.0
            spread = 2
        elif spike_type == "sodium":
            # Na+ spikes: fast, propagate well
            amplitude = 50.0
            spread = 5
        else:  # NMDA
            # NMDA spikes: local
            amplitude = 20.0
            spread = 1
        
        # Propagate to neighboring compartments
        for i in range(max(0, origin_compartment - spread),
                      min(self.num_compartments, origin_compartment + spread + 1)):
            if i != origin_compartment:
                # Distance-dependent attenuation
                distance = abs(i - origin_compartment)
                attenuation = np.exp(-distance / spread)
                self.compartment_voltages[i] += amplitude * attenuation
                affected.append(i)
        
        return affected
    
    def integrate_inputs(
        self,
        synaptic_inputs: List[tuple[int, float]]
    ) -> float:
        """Integrate synaptic inputs across compartments.
        
        Args:
            synaptic_inputs: List of (compartment_idx, current) tuples
            
        Returns:
            Integrated current at soma
        """
        total_current = 0.0
        
        # Add inputs to compartments
        for comp_idx, current in synaptic_inputs:
            if comp_idx < len(self.compartment_voltages):
                self.compartment_voltages[comp_idx] += current
                
                # Check for dendritic spikes
                spike_type = self.detect_dendritic_spike(comp_idx)
                if spike_type:
                    # Amplify and propagate
                    self.propagate_spike(spike_type, comp_idx)
        
        # Sum currents with distance-dependent attenuation
        for i, v in enumerate(self.compartment_voltages):
            # Attenuation increases with distance from soma
            attenuation = np.exp(-i * 0.1)
            total_current += v * attenuation
        
        # Passive decay of voltages
        self.compartment_voltages = [
            v * 0.9 - 65.0 * 0.1  # Decay toward rest
            for v in self.compartment_voltages
        ]
        
        return total_current


@dataclass
class ActiveZone:
    """Presynaptic active zone with vesicle pools.
    
    Active zones are specialized presynaptic sites where:
    - Vesicles dock and fuse
    - Ca2+ channels cluster
    - Release machinery is organized
    """
    
    id: int
    presynaptic_neuron_id: int
    
    # Vesicle pools
    readily_releasable_pool: int = 10  # RRP
    recycling_pool: int = 50
    reserve_pool: int = 200
    
    # Release parameters
    release_probability: float = 0.3
    calcium_sensitivity: float = 1.0
    
    # Calcium concentration at active zone
    local_calcium: float = 0.1  # μM
    
    # Release sites
    num_release_sites: int = 5
    
    def release_vesicles(self, presynaptic_spike: bool, ca_influx: float) -> int:
        """Release neurotransmitter vesicles.
        
        Args:
            presynaptic_spike: Whether presynaptic neuron spiked
            ca_influx: Calcium influx at active zone
            
        Returns:
            Number of vesicles released
        """
        if not presynaptic_spike:
            return 0
        
        # Update local calcium
        self.local_calcium += ca_influx
        
        # Calcium-dependent release probability
        ca_factor = (self.local_calcium / (self.local_calcium + 1.0)) ** 4
        effective_p = self.release_probability * ca_factor * self.calcium_sensitivity
        
        # Binomial release from RRP
        released = 0
        for _ in range(min(self.readily_releasable_pool, self.num_release_sites)):
            if np.random.random() < effective_p:
                released += 1
        
        # Deplete RRP
        self.readily_releasable_pool -= released
        
        # Decay calcium
        self.local_calcium *= 0.5
        
        return released
    
    def refill_rrp(self, dt: float = 1.0) -> None:
        """Refill readily releasable pool from recycling pool.
        
        Args:
            dt: Time step (ms)
        """
        # Refill rate depends on recycling pool
        refill_rate = 0.1  # 1/ms
        max_refill = 10
        
        if self.readily_releasable_pool < max_refill:
            needed = max_refill - self.readily_releasable_pool
            available = min(needed, self.recycling_pool)
            transfer = int(available * refill_rate * dt)
            
            self.readily_releasable_pool += transfer
            self.recycling_pool -= transfer
    
    def recycle_vesicles(self, dt: float = 1.0) -> None:
        """Recycle vesicles from reserve to recycling pool.
        
        Args:
            dt: Time step (ms)
        """
        recycle_rate = 0.01  # 1/ms (slow)
        max_recycling = 50
        
        if self.recycling_pool < max_recycling:
            needed = max_recycling - self.recycling_pool
            available = min(needed, self.reserve_pool)
            transfer = int(available * recycle_rate * dt)
            
            self.recycling_pool += transfer
            self.reserve_pool -= transfer


@dataclass
class CompartmentalNeuron:
    """Neuron with compartmental dendritic structure."""
    
    neuron_id: int
    
    # Soma
    soma_voltage: float = -65.0
    soma_conductances: Dict[str, float] = field(default_factory=dict)
    
    # Dendrites
    basal_dendrites: List[DendriticBranch] = field(default_factory=list)
    apical_dendrites: List[DendriticBranch] = field(default_factory=list)
    
    # Axon
    axon_initial_segment_voltage: float = -65.0
    spike_threshold: float = -50.0
    
    # Active zones
    active_zones: List[ActiveZone] = field(default_factory=list)
    
    def add_dendritic_branch(
        self,
        branch_type: str,
        **kwargs
    ) -> DendriticBranch:
        """Add a dendritic branch.
        
        Args:
            branch_type: "basal" or "apical"
            **kwargs: Branch parameters
            
        Returns:
            Created DendriticBranch
        """
        branch_id = len(self.basal_dendrites) + len(self.apical_dendrites)
        branch = DendriticBranch(
            id=branch_id,
            parent_neuron_id=self.neuron_id,
            **kwargs
        )
        
        if branch_type == "basal":
            self.basal_dendrites.append(branch)
        else:
            self.apical_dendrites.append(branch)
        
        return branch
    
    def integrate_dendritic_inputs(
        self,
        basal_inputs: List[tuple[int, int, float]],  # (branch, compartment, current)
        apical_inputs: List[tuple[int, int, float]]
    ) -> float:
        """Integrate inputs from all dendrites.
        
        Args:
            basal_inputs: Inputs to basal dendrites
            apical_inputs: Inputs to apical dendrites
            
        Returns:
            Total current at soma
        """
        total = 0.0
        
        # Integrate basal dendrites
        for branch_idx, comp_idx, current in basal_inputs:
            if branch_idx < len(self.basal_dendrites):
                branch = self.basal_dendrites[branch_idx]
                total += branch.integrate_inputs([(comp_idx, current)])
        
        # Integrate apical dendrites (often with different weighting)
        for branch_idx, comp_idx, current in apical_inputs:
            if branch_idx < len(self.apical_dendrites):
                branch = self.apical_dendrites[branch_idx]
                # Apical inputs might have different integration rules
                total += branch.integrate_inputs([(comp_idx, current)]) * 0.8
        
        return total
    
    def update_soma(self, dendritic_current: float, dt: float) -> bool:
        """Update somatic voltage and check for spike.
        
        Args:
            dendritic_current: Current from dendrites
            dt: Time step
            
        Returns:
            True if neuron spiked
        """
        # Simple integration
        leak = (-65.0 - self.soma_voltage) * 0.1
        self.soma_voltage += (dendritic_current + leak) * dt
        
        # Check for spike at AIS
        if self.soma_voltage > self.spike_threshold:
            self.soma_voltage = -65.0  # Reset
            return True
        
        return False
