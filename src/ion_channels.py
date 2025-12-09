"""Ion channel implementations for 4D Neural Cognition.

This module implements biophysically realistic ion channels:
- Voltage-gated channels (Na, K, Ca, HCN)
- Ligand-gated channels (AMPA, NMDA, GABA-A, GABA-B, mGluR)
- Leak channels (K, Cl)
- Calcium dynamics (buffering, CICR via IP3)

References:
- Hodgkin, A.L., & Huxley, A.F. (1952). A quantitative description of membrane current
- Johnston, D., & Wu, S.M. (1995). Foundations of Cellular Neurophysiology
- Dayan, P., & Abbott, L.F. (2001). Theoretical Neuroscience
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np


@dataclass
class VoltageGatedChannel:
    """Base class for voltage-gated ion channels."""
    
    # Conductance
    g_max: float = 1.0  # Maximum conductance (mS/cm²)
    
    # Reversal potential
    E_rev: float = 0.0  # Reversal potential (mV)
    
    # Gating variables
    activation: float = 0.0  # m gate (0-1)
    inactivation: float = 1.0  # h gate (0-1)
    
    def current(self, v_membrane: float) -> float:
        """Calculate ionic current.
        
        Args:
            v_membrane: Membrane potential (mV)
            
        Returns:
            Ionic current (μA/cm²)
        """
        raise NotImplementedError
    
    def update_gates(self, v_membrane: float, dt: float) -> None:
        """Update gating variables.
        
        Args:
            v_membrane: Membrane potential (mV)
            dt: Time step (ms)
        """
        raise NotImplementedError


@dataclass
class SodiumChannel(VoltageGatedChannel):
    """Fast voltage-gated sodium channel (Nav) for action potentials.
    
    Implements Nav1.x channels responsible for:
    - Fast depolarization during action potential
    - Spike initiation at axon initial segment
    """
    
    g_max: float = 120.0  # mS/cm²
    E_rev: float = 50.0  # mV (ENa)
    
    def alpha_m(self, v: float) -> float:
        """Activation rate constant."""
        if abs(v + 40.0) < 1e-6:
            return 1.0
        return 0.1 * (v + 40.0) / (1.0 - np.exp(-(v + 40.0) / 10.0))
    
    def beta_m(self, v: float) -> float:
        """Deactivation rate constant."""
        return 4.0 * np.exp(-(v + 65.0) / 18.0)
    
    def alpha_h(self, v: float) -> float:
        """Inactivation rate constant."""
        return 0.07 * np.exp(-(v + 65.0) / 20.0)
    
    def beta_h(self, v: float) -> float:
        """Deinactivation rate constant."""
        return 1.0 / (1.0 + np.exp(-(v + 35.0) / 10.0))
    
    def update_gates(self, v_membrane: float, dt: float) -> None:
        """Update m and h gates."""
        v = v_membrane
        
        # Update activation (m)
        alpha = self.alpha_m(v)
        beta = self.beta_m(v)
        dm = (alpha * (1 - self.activation) - beta * self.activation) * dt
        self.activation = np.clip(self.activation + dm, 0.0, 1.0)
        
        # Update inactivation (h)
        alpha = self.alpha_h(v)
        beta = self.beta_h(v)
        dh = (alpha * (1 - self.inactivation) - beta * self.inactivation) * dt
        self.inactivation = np.clip(self.inactivation + dh, 0.0, 1.0)
    
    def current(self, v_membrane: float) -> float:
        """Calculate sodium current."""
        return self.g_max * (self.activation ** 3) * self.inactivation * (v_membrane - self.E_rev)


@dataclass
class PotassiumChannel(VoltageGatedChannel):
    """Delayed rectifier potassium channel (Kv) for repolarization.
    
    Implements Kv channels responsible for:
    - Repolarization after action potential
    - Regulation of firing frequency
    """
    
    g_max: float = 36.0  # mS/cm²
    E_rev: float = -77.0  # mV (EK)
    
    def alpha_n(self, v: float) -> float:
        """Activation rate constant."""
        if abs(v + 55.0) < 1e-6:
            return 0.1
        return 0.01 * (v + 55.0) / (1.0 - np.exp(-(v + 55.0) / 10.0))
    
    def beta_n(self, v: float) -> float:
        """Deactivation rate constant."""
        return 0.125 * np.exp(-(v + 65.0) / 80.0)
    
    def update_gates(self, v_membrane: float, dt: float) -> None:
        """Update n gate."""
        v = v_membrane
        
        alpha = self.alpha_n(v)
        beta = self.beta_n(v)
        dn = (alpha * (1 - self.activation) - beta * self.activation) * dt
        self.activation = np.clip(self.activation + dn, 0.0, 1.0)
        
        # No inactivation for delayed rectifier
        self.inactivation = 1.0
    
    def current(self, v_membrane: float) -> float:
        """Calculate potassium current."""
        return self.g_max * (self.activation ** 4) * (v_membrane - self.E_rev)


@dataclass
class CalciumChannel(VoltageGatedChannel):
    """Voltage-gated calcium channel (Cav) for calcium influx.
    
    Implements multiple types:
    - L-type: High-voltage activated, long-lasting
    - N-type: Presynaptic neurotransmitter release
    - P/Q-type: Cerebellar Purkinje cells
    - T-type: Low-voltage activated, transient
    """
    
    channel_type: str = "L"  # L, N, P, Q, R, or T
    g_max: float = 0.3  # mS/cm²
    E_rev: float = 120.0  # mV (ECa)
    
    def __post_init__(self):
        """Initialize channel-type specific parameters."""
        if self.channel_type == "L":
            self.v_half = -10.0  # Half-activation voltage
            self.tau = 10.0  # Activation time constant
        elif self.channel_type == "T":
            self.v_half = -50.0  # Lower threshold (LVA)
            self.tau = 5.0  # Faster
        else:  # N, P, Q, R
            self.v_half = -20.0
            self.tau = 8.0
    
    def steady_state(self, v: float) -> float:
        """Steady-state activation."""
        return 1.0 / (1.0 + np.exp(-(v - self.v_half) / 5.0))
    
    def update_gates(self, v_membrane: float, dt: float) -> None:
        """Update activation gate."""
        m_inf = self.steady_state(v_membrane)
        dm = (m_inf - self.activation) / self.tau * dt
        self.activation = np.clip(self.activation + dm, 0.0, 1.0)
        
        # Voltage-dependent inactivation for some types
        if self.channel_type == "T":
            h_inf = 1.0 / (1.0 + np.exp((v_membrane + 70.0) / 5.0))
            dh = (h_inf - self.inactivation) / 20.0 * dt
            self.inactivation = np.clip(self.inactivation + dh, 0.0, 1.0)
        else:
            self.inactivation = 1.0
    
    def current(self, v_membrane: float) -> float:
        """Calculate calcium current."""
        return self.g_max * (self.activation ** 2) * self.inactivation * (v_membrane - self.E_rev)


@dataclass
class HCNChannel(VoltageGatedChannel):
    """Hyperpolarization-activated cyclic nucleotide-gated (HCN) channel.
    
    Responsible for:
    - Ih current (hyperpolarization-activated inward current)
    - Rhythmic activity and pacemaking
    - Dendritic integration
    """
    
    g_max: float = 0.1  # mS/cm²
    E_rev: float = -40.0  # mV (mixed Na+/K+ current)
    
    def steady_state(self, v: float) -> float:
        """Steady-state activation (activated by hyperpolarization)."""
        return 1.0 / (1.0 + np.exp((v + 80.0) / 10.0))
    
    def update_gates(self, v_membrane: float, dt: float) -> None:
        """Update activation gate."""
        m_inf = self.steady_state(v_membrane)
        tau_h = 100.0  # Slow activation
        dm = (m_inf - self.activation) / tau_h * dt
        self.activation = np.clip(self.activation + dm, 0.0, 1.0)
        self.inactivation = 1.0  # No inactivation
    
    def current(self, v_membrane: float) -> float:
        """Calculate HCN current."""
        return self.g_max * self.activation * (v_membrane - self.E_rev)


@dataclass
class LigandGatedChannel:
    """Base class for ligand-gated ion channels (ionotropic receptors)."""
    
    # Conductance
    g_max: float = 1.0  # Maximum conductance
    
    # Reversal potential
    E_rev: float = 0.0  # mV
    
    # Binding
    open_fraction: float = 0.0  # Fraction of open channels
    
    # Kinetics
    rise_time: float = 1.0  # ms
    decay_time: float = 5.0  # ms
    
    def bind(self, neurotransmitter_concentration: float) -> None:
        """Bind neurotransmitter and open channels.
        
        Args:
            neurotransmitter_concentration: Concentration in synaptic cleft
        """
        raise NotImplementedError
    
    def update(self, dt: float) -> None:
        """Update channel state.
        
        Args:
            dt: Time step (ms)
        """
        # Exponential decay of open fraction
        decay = np.exp(-dt / self.decay_time)
        self.open_fraction *= decay
    
    def current(self, v_membrane: float) -> float:
        """Calculate ionic current.
        
        Args:
            v_membrane: Membrane potential (mV)
            
        Returns:
            Ionic current
        """
        return self.g_max * self.open_fraction * (v_membrane - self.E_rev)


@dataclass
class AMPAReceptor(LigandGatedChannel):
    """AMPA receptor for fast excitatory transmission.
    
    Properties:
    - Fast kinetics (rise ~1ms, decay ~5ms)
    - Mediates basal excitatory transmission
    - Permeable to Na+ and K+
    """
    
    g_max: float = 1.0
    E_rev: float = 0.0  # mV
    rise_time: float = 1.0  # ms
    decay_time: float = 5.0  # ms
    
    def bind(self, glutamate_concentration: float) -> None:
        """Bind glutamate."""
        # Instantaneous rise, exponential decay
        self.open_fraction = min(1.0, self.open_fraction + glutamate_concentration)


@dataclass
class NMDAReceptor(LigandGatedChannel):
    """NMDA receptor for coincidence detection and plasticity.
    
    Properties:
    - Slower kinetics (rise ~5ms, decay ~50ms)
    - Mg2+ block at hyperpolarized potentials
    - Permeable to Ca2+ (important for plasticity)
    - Requires glutamate + co-agonist (glycine/D-serine)
    """
    
    g_max: float = 0.5
    E_rev: float = 0.0  # mV
    rise_time: float = 5.0  # ms
    decay_time: float = 50.0  # ms
    
    # Mg2+ block parameters
    mg_concentration: float = 1.0  # mM
    
    def mg_block(self, v_membrane: float) -> float:
        """Voltage-dependent Mg2+ block.
        
        Args:
            v_membrane: Membrane potential (mV)
            
        Returns:
            Block factor (0-1, 0=full block, 1=no block)
        """
        return 1.0 / (1.0 + (self.mg_concentration / 3.57) * np.exp(-0.062 * v_membrane))
    
    def bind(self, glutamate_concentration: float, coagonist: float = 1.0) -> None:
        """Bind glutamate and co-agonist.
        
        Args:
            glutamate_concentration: Glutamate concentration
            coagonist: D-serine or glycine concentration
        """
        # Requires both glutamate and co-agonist
        binding = glutamate_concentration * coagonist
        self.open_fraction = min(1.0, self.open_fraction + binding * 0.5)
    
    def current(self, v_membrane: float) -> float:
        """Calculate NMDA current with Mg2+ block."""
        unblocked_current = self.g_max * self.open_fraction * (v_membrane - self.E_rev)
        return unblocked_current * self.mg_block(v_membrane)


@dataclass
class GABAaReceptor(LigandGatedChannel):
    """GABA-A receptor for fast inhibitory transmission.
    
    Properties:
    - Fast kinetics
    - Cl- permeable (hyperpolarizing)
    - Mediate phasic inhibition
    """
    
    g_max: float = 1.0
    E_rev: float = -70.0  # mV (ECl)
    rise_time: float = 1.0  # ms
    decay_time: float = 10.0  # ms
    
    def bind(self, gaba_concentration: float) -> None:
        """Bind GABA."""
        self.open_fraction = min(1.0, self.open_fraction + gaba_concentration)


@dataclass
class GABAbReceptor:
    """GABA-B receptor for slow inhibitory modulation (metabotropic).
    
    Properties:
    - Slow G-protein coupled response
    - Activates K+ channels (GIRK)
    - Inhibits Ca2+ channels
    - Mediates tonic inhibition
    """
    
    activation: float = 0.0
    rise_time: float = 50.0  # ms (slow)
    decay_time: float = 200.0  # ms (very slow)
    
    # Effector channel conductance
    g_k_max: float = 0.5  # GIRK channel conductance
    E_k: float = -90.0  # mV
    
    def bind(self, gaba_concentration: float) -> None:
        """Bind GABA and activate G-protein cascade.
        
        Args:
            gaba_concentration: GABA concentration
        """
        self.activation = min(1.0, self.activation + gaba_concentration * 0.1)
    
    def update(self, dt: float) -> None:
        """Update activation state.
        
        Args:
            dt: Time step (ms)
        """
        decay = np.exp(-dt / self.decay_time)
        self.activation *= decay
    
    def k_current(self, v_membrane: float) -> float:
        """Calculate K+ current via GIRK channels.
        
        Args:
            v_membrane: Membrane potential (mV)
            
        Returns:
            K+ current (hyperpolarizing)
        """
        return self.g_k_max * self.activation * (v_membrane - self.E_k)


@dataclass
class CalciumDynamics:
    """Intracellular calcium dynamics.
    
    Implements:
    - Calcium influx via voltage-gated channels
    - Calcium buffering
    - Calcium extrusion (pumps, exchangers)
    - Calcium-induced calcium release (CICR) via IP3 receptors
    """
    
    # Calcium concentration (μM)
    ca_concentration: float = 0.1  # Resting ~100 nM
    ca_rest: float = 0.1
    ca_external: float = 2000.0  # 2 mM outside
    
    # Buffering
    buffer_capacity: float = 20.0  # κ in Helmchen et al.
    
    # Extrusion
    extrusion_rate: float = 0.5  # 1/ms
    
    # Internal stores (ER)
    ca_store: float = 400.0  # μM in ER
    
    # IP3 receptor (CICR)
    ip3_concentration: float = 0.0  # μM
    ip3_threshold: float = 0.1  # μM
    
    def influx(self, i_ca: float, dt: float, volume: float = 1.0) -> None:
        """Add calcium from influx current.
        
        Args:
            i_ca: Calcium current (negative = influx)
            dt: Time step (ms)
            volume: Cell volume factor
        """
        # Convert current to concentration change
        # Using Faraday's constant and conversion factors
        ca_influx = -i_ca * dt / (2.0 * 96485.0 * volume) * 1e6  # μM
        
        # Account for buffering
        delta_ca = ca_influx / (1.0 + self.buffer_capacity)
        
        self.ca_concentration += delta_ca
    
    def extrude(self, dt: float) -> None:
        """Remove calcium via pumps and exchangers.
        
        Args:
            dt: Time step (ms)
        """
        # Exponential decay to resting level
        decay = (self.ca_concentration - self.ca_rest) * self.extrusion_rate * dt
        self.ca_concentration -= decay
    
    def cicr(self, dt: float) -> None:
        """Calcium-induced calcium release via IP3 receptors.
        
        Args:
            dt: Time step (ms)
        """
        if self.ip3_concentration > self.ip3_threshold:
            # CICR depends on [Ca2+], [IP3], and store content
            release_rate = (
                self.ip3_concentration *
                (self.ca_concentration / (self.ca_concentration + 0.3)) *
                (self.ca_store / 400.0)
            )
            
            release = release_rate * dt * 10.0
            
            # Transfer from store to cytoplasm
            self.ca_store -= release
            self.ca_concentration += release / (1.0 + self.buffer_capacity)
    
    def refill_stores(self, dt: float) -> None:
        """Refill ER calcium stores via SERCA pumps.
        
        Args:
            dt: Time step (ms)
        """
        refill_rate = 0.1  # 1/ms
        if self.ca_store < 400.0:
            refill = refill_rate * (400.0 - self.ca_store) * dt
            self.ca_store += refill
    
    def update(self, i_ca: float, dt: float) -> None:
        """Update calcium dynamics for one time step.
        
        Args:
            i_ca: Calcium current
            dt: Time step (ms)
        """
        self.influx(i_ca, dt)
        self.extrude(dt)
        self.cicr(dt)
        self.refill_stores(dt)
        
        # Ensure non-negative
        self.ca_concentration = max(0.0, self.ca_concentration)
        self.ca_store = max(0.0, self.ca_store)
    
    def get_concentration(self) -> float:
        """Get current calcium concentration.
        
        Returns:
            Calcium concentration (μM)
        """
        return self.ca_concentration


@dataclass
class IonChannelComplement:
    """Complete complement of ion channels for a neuron."""
    
    # Voltage-gated channels
    na_channel: Optional[SodiumChannel] = field(default_factory=lambda: SodiumChannel())
    k_channel: Optional[PotassiumChannel] = field(default_factory=lambda: PotassiumChannel())
    ca_channel: Optional[CalciumChannel] = field(default_factory=lambda: CalciumChannel(channel_type="L"))
    hcn_channel: Optional[HCNChannel] = field(default_factory=lambda: HCNChannel())
    
    # Ligand-gated channels
    ampa_receptors: List[AMPAReceptor] = field(default_factory=list)
    nmda_receptors: List[NMDAReceptor] = field(default_factory=list)
    gabaa_receptors: List[GABAaReceptor] = field(default_factory=list)
    gabab_receptors: List[GABAbReceptor] = field(default_factory=list)
    
    # Leak channels
    g_leak: float = 0.3  # mS/cm²
    E_leak: float = -54.4  # mV
    
    # Calcium dynamics
    ca_dynamics: CalciumDynamics = field(default_factory=CalciumDynamics)
    
    def update_voltage_gated(self, v_membrane: float, dt: float) -> Dict[str, float]:
        """Update all voltage-gated channels and return currents.
        
        Args:
            v_membrane: Membrane potential (mV)
            dt: Time step (ms)
            
        Returns:
            Dictionary of ionic currents
        """
        currents = {}
        
        if self.na_channel:
            self.na_channel.update_gates(v_membrane, dt)
            currents["I_Na"] = self.na_channel.current(v_membrane)
        
        if self.k_channel:
            self.k_channel.update_gates(v_membrane, dt)
            currents["I_K"] = self.k_channel.current(v_membrane)
        
        if self.ca_channel:
            self.ca_channel.update_gates(v_membrane, dt)
            currents["I_Ca"] = self.ca_channel.current(v_membrane)
        
        if self.hcn_channel:
            self.hcn_channel.update_gates(v_membrane, dt)
            currents["I_h"] = self.hcn_channel.current(v_membrane)
        
        # Leak current
        currents["I_leak"] = self.g_leak * (v_membrane - self.E_leak)
        
        # Update calcium dynamics
        if self.ca_channel:
            self.ca_dynamics.update(currents.get("I_Ca", 0.0), dt)
        
        return currents
    
    def update_ligand_gated(self, v_membrane: float, dt: float) -> float:
        """Update all ligand-gated channels and return total synaptic current.
        
        Args:
            v_membrane: Membrane potential (mV)
            dt: Time step (ms)
            
        Returns:
            Total synaptic current
        """
        total_current = 0.0
        
        # Update AMPA receptors
        for ampa in self.ampa_receptors:
            ampa.update(dt)
            total_current += ampa.current(v_membrane)
        
        # Update NMDA receptors
        for nmda in self.nmda_receptors:
            nmda.update(dt)
            total_current += nmda.current(v_membrane)
        
        # Update GABA-A receptors
        for gabaa in self.gabaa_receptors:
            gabaa.update(dt)
            total_current += gabaa.current(v_membrane)
        
        # Update GABA-B receptors
        for gabab in self.gabab_receptors:
            gabab.update(dt)
            total_current += gabab.k_current(v_membrane)
        
        return total_current
