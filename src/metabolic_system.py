"""Metabolic system for 4D Neural Cognition.

This module implements brain metabolism and energetics:
- Energy budget (ATP consumption)
- Glucose metabolism and lactate shuttle
- Oxygen consumption
- Blood flow coupling (neurovascular unit)
- BOLD signal simulation

References:
- Attwell, D., & Laughlin, S.B. (2001). An energy budget for signaling in the grey matter
- Magistretti, P.J., & Allaman, I. (2015). A cellular perspective on brain energy metabolism
- Buxton, R.B., et al. (2004). Modeling the hemodynamic response
"""

from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np


@dataclass
class ATPBudget:
    """ATP energy budget for neural activity.
    
    Brain energy consumption breakdown:
    - ~50% for action potentials (Na+/K+ ATPase)
    - ~30% for synaptic transmission (vesicle cycling)
    - ~20% for resting potential maintenance
    
    Total: ~20% of body's energy at rest
    """
    
    # Total ATP available (arbitrary units)
    total_atp: float = 1000.0
    current_atp: float = 1000.0
    
    # Costs (ATP per event)
    spike_cost: float = 0.1  # Per action potential
    synapse_cost: float = 0.05  # Per synaptic transmission
    resting_cost: float = 0.001  # Per neuron per step
    
    # Production rate
    atp_production_rate: float = 1.0  # Per step
    
    # Metabolic state
    is_hypoxic: bool = False
    hypoxia_threshold: float = 0.3  # Fraction of total ATP
    
    def consume_spike(self, n_spikes: int = 1) -> bool:
        """Consume ATP for action potentials.
        
        Args:
            n_spikes: Number of spikes
            
        Returns:
            True if sufficient ATP available
        """
        cost = self.spike_cost * n_spikes
        if self.current_atp >= cost:
            self.current_atp -= cost
            return True
        else:
            # Insufficient energy -> hypoxia
            self.is_hypoxic = True
            return False
    
    def consume_synapse(self, n_transmissions: int = 1) -> bool:
        """Consume ATP for synaptic transmission.
        
        Args:
            n_transmissions: Number of synaptic events
            
        Returns:
            True if sufficient ATP available
        """
        cost = self.synapse_cost * n_transmissions
        if self.current_atp >= cost:
            self.current_atp -= cost
            return True
        else:
            self.is_hypoxic = True
            return False
    
    def consume_resting(self, n_neurons: int) -> None:
        """Consume ATP for resting potential.
        
        Args:
            n_neurons: Number of neurons
        """
        cost = self.resting_cost * n_neurons
        self.current_atp = max(0.0, self.current_atp - cost)
    
    def produce_atp(self, glucose: float, oxygen: float) -> float:
        """Produce ATP from glucose and oxygen.
        
        Args:
            glucose: Glucose availability (0-1)
            oxygen: Oxygen availability (0-1)
            
        Returns:
            ATP produced
        """
        # Aerobic metabolism: glucose + O2 -> ATP + CO2 + H2O
        # 1 glucose -> ~30-32 ATP (oxidative phosphorylation)
        aerobic_production = glucose * oxygen * self.atp_production_rate * 30.0
        
        # Anaerobic glycolysis: glucose -> lactate + ATP
        # 1 glucose -> 2 ATP (less efficient)
        anaerobic_production = glucose * (1.0 - oxygen) * self.atp_production_rate * 2.0
        
        total_production = aerobic_production + anaerobic_production
        
        self.current_atp = min(self.total_atp, self.current_atp + total_production)
        
        # Check hypoxia
        if self.current_atp < self.total_atp * self.hypoxia_threshold:
            self.is_hypoxic = True
        else:
            self.is_hypoxic = False
        
        return total_production
    
    def get_energy_state(self) -> Dict[str, float]:
        """Get current energy state.
        
        Returns:
            Dictionary with energy metrics
        """
        return {
            "atp_level": self.current_atp,
            "atp_fraction": self.current_atp / self.total_atp,
            "is_hypoxic": self.is_hypoxic,
        }


@dataclass
class MetabolitePool:
    """Pool of metabolites in brain tissue.
    
    Key metabolites:
    - Glucose: Primary energy source
    - Lactate: Astrocyte-neuron lactate shuttle (ANLS)
    - Oxygen: For oxidative metabolism
    - CO2: Metabolic byproduct
    """
    
    # Concentrations (arbitrary units)
    glucose: float = 5.0  # ~5 mM in blood
    lactate: float = 1.0  # ~1 mM baseline
    oxygen: float = 100.0  # pO2 in mmHg
    co2: float = 40.0  # pCO2 in mmHg
    
    # Consumption rates
    glucose_consumption_rate: float = 0.1
    oxygen_consumption_rate: float = 0.5
    
    # Production rates
    lactate_production_rate: float = 0.05
    co2_production_rate: float = 0.5
    
    def consume_glucose(self, neural_activity: float) -> float:
        """Consume glucose based on neural activity.
        
        Args:
            neural_activity: Activity level (0-1)
            
        Returns:
            Glucose consumed
        """
        consumption = self.glucose_consumption_rate * neural_activity
        self.glucose = max(0.0, self.glucose - consumption)
        return consumption
    
    def consume_oxygen(self, neural_activity: float) -> float:
        """Consume oxygen based on neural activity.
        
        Args:
            neural_activity: Activity level (0-1)
            
        Returns:
            Oxygen consumed
        """
        consumption = self.oxygen_consumption_rate * neural_activity
        self.oxygen = max(0.0, self.oxygen - consumption)
        return consumption
    
    def produce_lactate(self, glucose_consumed: float, oxygen_available: float) -> float:
        """Produce lactate via glycolysis.
        
        When oxygen is limited, more lactate is produced.
        Astrocytes produce lactate that neurons can use.
        
        Args:
            glucose_consumed: Amount of glucose consumed
            oxygen_available: Oxygen availability (0-1)
            
        Returns:
            Lactate produced
        """
        # More lactate when oxygen is low (anaerobic glycolysis)
        production = glucose_consumed * (1.0 - oxygen_available) * 2.0
        self.lactate += production
        return production
    
    def produce_co2(self, oxygen_consumed: float) -> float:
        """Produce CO2 from oxidative metabolism.
        
        Args:
            oxygen_consumed: Amount of oxygen consumed
            
        Returns:
            CO2 produced
        """
        production = oxygen_consumed * self.co2_production_rate
        self.co2 += production
        return production
    
    def astrocyte_neuron_lactate_shuttle(self) -> float:
        """Transfer lactate from astrocytes to neurons.
        
        Returns:
            Lactate available for neurons
        """
        # Astrocytes take up glucose and produce lactate
        # Neurons preferentially use lactate over glucose
        available_lactate = min(self.lactate, 0.5)
        self.lactate -= available_lactate
        return available_lactate


@dataclass
class BloodVessel:
    """Blood vessel for oxygen and glucose delivery.
    
    Models:
    - Blood flow
    - Oxygen delivery
    - Glucose delivery
    - CO2 removal
    """
    
    id: int
    position: tuple
    
    # Flow
    baseline_flow: float = 1.0
    current_flow: float = 1.0
    
    # Capacity
    max_flow_increase: float = 3.0  # Can increase 3x
    
    # Delivery
    oxygen_delivery_rate: float = 10.0  # Per unit flow
    glucose_delivery_rate: float = 1.0
    
    # Removal
    co2_removal_rate: float = 5.0
    
    def dilate(self, signal: float) -> None:
        """Dilate vessel in response to metabolic demand.
        
        Args:
            signal: Vasodilation signal (0-1)
        """
        # Increase flow up to maximum
        target_flow = self.baseline_flow * (1.0 + signal * (self.max_flow_increase - 1.0))
        
        # Smooth transition
        self.current_flow += (target_flow - self.current_flow) * 0.1
    
    def deliver_oxygen(self, metabolites: MetabolitePool) -> None:
        """Deliver oxygen to tissue.
        
        Args:
            metabolites: Metabolite pool to deliver to
        """
        delivery = self.oxygen_delivery_rate * self.current_flow
        metabolites.oxygen = min(metabolites.oxygen + delivery, 150.0)  # Cap at high pO2
    
    def deliver_glucose(self, metabolites: MetabolitePool) -> None:
        """Deliver glucose to tissue.
        
        Args:
            metabolites: Metabolite pool to deliver to
        """
        delivery = self.glucose_delivery_rate * self.current_flow
        metabolites.glucose = min(metabolites.glucose + delivery, 10.0)  # Cap at high concentration
    
    def remove_co2(self, metabolites: MetabolitePool) -> None:
        """Remove CO2 from tissue.
        
        Args:
            metabolites: Metabolite pool to remove from
        """
        removal = self.co2_removal_rate * self.current_flow
        metabolites.co2 = max(metabolites.co2 - removal, 35.0)  # Floor at low pCO2


@dataclass
class NeurovascularUnit:
    """Neurovascular unit coupling neural activity to blood flow.
    
    Components:
    - Neurons: Generate activity and metabolic demand
    - Astrocytes: Sense activity and release vasoactive factors
    - Blood vessels: Respond to signals by dilating
    
    Implements neurovascular coupling for BOLD signal.
    """
    
    # Components
    blood_vessels: List[BloodVessel] = field(default_factory=list)
    metabolites: MetabolitePool = field(default_factory=MetabolitePool)
    
    # Coupling parameters
    activity_to_flow_gain: float = 2.0
    flow_response_time: float = 2.0  # seconds (here in steps)
    
    # Vasoactive factors
    nitric_oxide: float = 0.0  # Vasodilator
    adenosine: float = 0.0  # Vasodilator
    k_plus: float = 3.0  # Extracellular K+, vasodilator
    
    # BOLD signal components
    cbf: float = 1.0  # Cerebral blood flow
    cbv: float = 1.0  # Cerebral blood volume
    cmro2: float = 1.0  # Cerebral metabolic rate of O2
    deoxyhemoglobin: float = 0.5  # Affects BOLD signal
    
    def sense_neural_activity(self, activity_level: float) -> None:
        """Sense neural activity and release vasoactive factors.
        
        Args:
            activity_level: Neural activity level (0-1)
        """
        # High activity -> increase vasodilators
        self.nitric_oxide = activity_level * 0.5
        
        # Adenosine accumulates with metabolism
        self.adenosine += activity_level * 0.1
        self.adenosine *= 0.9  # Decay
        
        # K+ released during spikes
        self.k_plus = 3.0 + activity_level * 2.0
    
    def calculate_vasodilation_signal(self) -> float:
        """Calculate vasodilation signal.
        
        Returns:
            Vasodilation signal (0-1)
        """
        # Combine vasoactive factors
        signal = (
            self.nitric_oxide +
            self.adenosine +
            (self.k_plus - 3.0) / 2.0  # Normalize K+
        )
        
        return np.clip(signal, 0.0, 1.0)
    
    def update_blood_flow(self, activity_level: float) -> None:
        """Update blood flow based on activity.
        
        Args:
            activity_level: Neural activity level (0-1)
        """
        # Sense activity
        self.sense_neural_activity(activity_level)
        
        # Calculate vasodilation
        vasodilation = self.calculate_vasodilation_signal()
        
        # Update all vessels
        for vessel in self.blood_vessels:
            vessel.dilate(vasodilation)
            vessel.deliver_oxygen(self.metabolites)
            vessel.deliver_glucose(self.metabolites)
            vessel.remove_co2(self.metabolites)
        
        # Update CBF
        avg_flow = np.mean([v.current_flow for v in self.blood_vessels]) if self.blood_vessels else 1.0
        self.cbf += (avg_flow - self.cbf) / self.flow_response_time
        
        # CBV follows CBF (Grubb's relationship: CBV ∝ CBF^0.38)
        self.cbv = self.cbf ** 0.38
        
        # CMRO2 increases with activity (but less than CBF)
        self.cmro2 = 1.0 + activity_level * 0.5
    
    def calculate_bold_signal(self) -> float:
        """Calculate BOLD signal.
        
        The BOLD signal depends on:
        - Increased CBF (positive contribution)
        - Increased CBV (negative contribution)
        - Increased CMRO2 (negative contribution via deoxyhemoglobin)
        
        Returns:
            BOLD signal (percent change from baseline)
        """
        # Simplified Balloon model
        # Deoxyhemoglobin changes with CMRO2 and flow
        self.deoxyhemoglobin = (self.cmro2 / self.cbf) * self.cbv
        
        # BOLD signal (inverse relationship with deoxyhemoglobin)
        # Positive BOLD = decreased deoxyhemoglobin
        bold = 100.0 * (1.0 - self.deoxyhemoglobin)
        
        return bold


@dataclass
class MetabolicSystem:
    """Complete metabolic system for brain simulation."""
    
    # Energy
    atp_budget: ATPBudget = field(default_factory=ATPBudget)
    
    # Metabolites
    metabolites: MetabolitePool = field(default_factory=MetabolitePool)
    
    # Neurovascular coupling
    neurovascular: NeurovascularUnit = field(default_factory=NeurovascularUnit)
    
    # Statistics
    total_glucose_consumed: float = 0.0
    total_oxygen_consumed: float = 0.0
    total_atp_produced: float = 0.0
    
    def process_neural_activity(
        self,
        n_spikes: int,
        n_synapses: int,
        n_neurons: int,
        activity_level: float
    ) -> Dict[str, float]:
        """Process metabolic demands of neural activity.
        
        Args:
            n_spikes: Number of action potentials
            n_synapses: Number of synaptic transmissions
            n_neurons: Total number of neurons
            activity_level: Overall activity level (0-1)
            
        Returns:
            Dictionary with metabolic state
        """
        # Consume ATP
        spikes_ok = self.atp_budget.consume_spike(n_spikes)
        synapses_ok = self.atp_budget.consume_synapse(n_synapses)
        self.atp_budget.consume_resting(n_neurons)
        
        # Consume metabolites
        glucose_consumed = self.metabolites.consume_glucose(activity_level)
        oxygen_consumed = self.metabolites.consume_oxygen(activity_level)
        
        self.total_glucose_consumed += glucose_consumed
        self.total_oxygen_consumed += oxygen_consumed
        
        # Produce byproducts
        oxygen_availability = self.metabolites.oxygen / 100.0
        self.metabolites.produce_lactate(glucose_consumed, oxygen_availability)
        self.metabolites.produce_co2(oxygen_consumed)
        
        # Lactate shuttle (astrocyte to neuron)
        lactate_available = self.metabolites.astrocyte_neuron_lactate_shuttle()
        
        # Produce ATP
        glucose_available = (self.metabolites.glucose / 5.0) + lactate_available
        atp_produced = self.atp_budget.produce_atp(glucose_available, oxygen_availability)
        self.total_atp_produced += atp_produced
        
        # Update blood flow (neurovascular coupling)
        self.neurovascular.metabolites = self.metabolites
        self.neurovascular.update_blood_flow(activity_level)
        
        # Calculate BOLD signal
        bold = self.neurovascular.calculate_bold_signal()
        
        return {
            "atp_level": self.atp_budget.current_atp,
            "glucose": self.metabolites.glucose,
            "oxygen": self.metabolites.oxygen,
            "lactate": self.metabolites.lactate,
            "cbf": self.neurovascular.cbf,
            "bold_signal": bold,
            "is_hypoxic": self.atp_budget.is_hypoxic,
            "energy_sufficient": spikes_ok and synapses_ok,
        }
    
    def get_statistics(self) -> Dict[str, float]:
        """Get metabolic statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "total_glucose_consumed": self.total_glucose_consumed,
            "total_oxygen_consumed": self.total_oxygen_consumed,
            "total_atp_produced": self.total_atp_produced,
            "current_atp": self.atp_budget.current_atp,
            "atp_fraction": self.atp_budget.current_atp / self.atp_budget.total_atp,
        }
