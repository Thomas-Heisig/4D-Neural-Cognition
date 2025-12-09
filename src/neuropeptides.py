"""Neuropeptide systems for 4D Neural Cognition.

This module implements neuropeptides that modulate neural function:
- Substance P: Pain transmission and inflammatory response
- Neuropeptide Y: Appetite regulation and stress response
- Oxytocin: Social bonding and trust
- Vasopressin: Water balance, social behavior, and aggression
- Endorphins: Pain relief and reward

References:
- Hökfelt, T., et al. (2000). Neuropeptides--an overview
- Martel, F.L., et al. (2012). Oxytocin modulation of neural function
- Grinevich, V., & Neumann, I.D. (2021). Brain oxytocin
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SubstancePSystem:
    """Substance P neuropeptide system for pain and inflammation.
    
    Substance P is involved in:
    - Nociception (pain transmission)
    - Inflammatory response
    - Stress response
    - Neurogenic inflammation
    """
    
    level: float = 0.0  # Current level (0-1)
    baseline: float = 0.1
    decay_rate: float = 0.05
    
    # Pain modulation
    pain_sensitivity: float = 1.0
    pain_threshold: float = 0.5
    
    # Inflammatory response
    inflammation_level: float = 0.0
    
    def release(self, pain_signal: float, tissue_damage: float = 0.0) -> None:
        """Release Substance P in response to pain or damage.
        
        Args:
            pain_signal: Intensity of pain signal (0-1)
            tissue_damage: Level of tissue damage (0-1)
        """
        # Release proportional to pain and damage
        release_amount = (pain_signal + tissue_damage) * 0.5
        self.level = min(1.0, self.level + release_amount)
        
        # Also triggers inflammation
        self.inflammation_level += tissue_damage * 0.3
    
    def modulate_pain(self, base_pain: float) -> float:
        """Modulate pain perception based on Substance P level.
        
        Args:
            base_pain: Base pain level
            
        Returns:
            Modulated pain level
        """
        # High Substance P increases pain sensitivity
        return base_pain * (1.0 + self.level * self.pain_sensitivity)
    
    def decay(self) -> None:
        """Decay Substance P level toward baseline."""
        if self.level > self.baseline:
            self.level -= self.decay_rate
        self.inflammation_level *= 0.9


@dataclass
class NeuropeptideYSystem:
    """Neuropeptide Y (NPY) system for appetite and stress.
    
    NPY is involved in:
    - Appetite stimulation
    - Energy homeostasis
    - Anxiety reduction (anxiolytic)
    - Stress resilience
    """
    
    level: float = 0.5  # Current level (0-1)
    baseline: float = 0.5
    decay_rate: float = 0.05
    
    # Appetite regulation
    hunger_signal: float = 0.0
    satiety_threshold: float = 0.3
    
    # Stress buffering
    stress_resistance: float = 1.0
    
    def update(self, hunger: float = 0.0, stress: float = 0.0) -> None:
        """Update NPY level based on hunger and stress.
        
        Args:
            hunger: Hunger level (0-1)
            stress: Stress level (0-1)
        """
        # High hunger increases NPY
        self.level = min(1.0, self.level + hunger * 0.3)
        
        # NPY released during stress (stress buffer)
        if stress > 0.5:
            self.level = min(1.0, self.level + 0.1)
        
        self.hunger_signal = hunger
    
    def stimulate_feeding(self) -> float:
        """Get feeding drive based on NPY level.
        
        Returns:
            Feeding motivation (0-1)
        """
        # High NPY promotes eating
        if self.level > self.satiety_threshold:
            return (self.level - self.satiety_threshold) / (1.0 - self.satiety_threshold)
        return 0.0
    
    def buffer_stress(self, stress: float) -> float:
        """Buffer stress response via NPY.
        
        Args:
            stress: Incoming stress level
            
        Returns:
            Buffered stress level
        """
        # High NPY reduces stress impact
        buffering = self.level * self.stress_resistance
        return stress * (1.0 - buffering * 0.5)
    
    def decay(self) -> None:
        """Decay NPY level toward baseline."""
        if self.level > self.baseline:
            self.level -= self.decay_rate
        elif self.level < self.baseline:
            self.level += self.decay_rate


@dataclass
class OxytocinSystem:
    """Oxytocin neuropeptide system for social bonding.
    
    Oxytocin is involved in:
    - Social bonding and attachment
    - Trust and cooperation
    - Maternal behavior
    - Stress reduction
    - Empathy
    """
    
    level: float = 0.3  # Current level (0-1)
    baseline: float = 0.3
    decay_rate: float = 0.05
    
    # Social bonding
    bond_strength: Dict[str, float] = field(default_factory=dict)
    trust_level: float = 0.5
    
    # Effects
    prosocial_bias: float = 1.5  # Increases prosocial behavior
    
    def release(self, social_positive: float, physical_contact: float = 0.0) -> None:
        """Release oxytocin in response to positive social interactions.
        
        Args:
            social_positive: Positive social signal (0-1)
            physical_contact: Physical contact/proximity (0-1)
        """
        # Release during positive social interactions
        release_amount = (social_positive + physical_contact) * 0.3
        self.level = min(1.0, self.level + release_amount)
    
    def form_bond(self, agent_id: str, interaction_quality: float) -> None:
        """Form or strengthen social bond.
        
        Args:
            agent_id: ID of social partner
            interaction_quality: Quality of interaction (0-1)
        """
        if agent_id not in self.bond_strength:
            self.bond_strength[agent_id] = 0.0
        
        # Oxytocin enhances bond formation
        bonding_rate = interaction_quality * self.level
        self.bond_strength[agent_id] = min(1.0, self.bond_strength[agent_id] + bonding_rate * 0.1)
    
    def modulate_trust(self, base_trust: float) -> float:
        """Modulate trust based on oxytocin level.
        
        Args:
            base_trust: Base trust level
            
        Returns:
            Modulated trust level
        """
        # High oxytocin increases trust
        return base_trust * (1.0 + self.level * 0.5)
    
    def promote_prosocial(self) -> float:
        """Get prosocial behavior tendency.
        
        Returns:
            Prosocial motivation (0-1)
        """
        return self.level * self.prosocial_bias
    
    def decay(self) -> None:
        """Decay oxytocin level toward baseline."""
        if self.level > self.baseline:
            self.level -= self.decay_rate
        
        # Bonds also decay slowly without reinforcement
        for agent_id in list(self.bond_strength.keys()):
            self.bond_strength[agent_id] *= 0.99
            if self.bond_strength[agent_id] < 0.01:
                del self.bond_strength[agent_id]


@dataclass
class VasopressinSystem:
    """Vasopressin (ADH) neuropeptide system for social behavior.
    
    Vasopressin is involved in:
    - Social behavior (especially in males)
    - Pair bonding
    - Territorial behavior and aggression
    - Water balance (peripheral effect)
    """
    
    level: float = 0.4  # Current level (0-1)
    baseline: float = 0.4
    decay_rate: float = 0.05
    
    # Social/territorial behavior
    territorial_drive: float = 0.0
    aggression_threshold: float = 0.6
    
    # Pair bonding (complementary to oxytocin)
    pair_bond_strength: float = 0.0
    
    def update(self, social_threat: float = 0.0, territorial_challenge: float = 0.0) -> None:
        """Update vasopressin level.
        
        Args:
            social_threat: Threat to social status (0-1)
            territorial_challenge: Challenge to territory (0-1)
        """
        # Increases with social threats
        threat_signal = social_threat + territorial_challenge
        self.level = min(1.0, self.level + threat_signal * 0.3)
        
        self.territorial_drive = territorial_challenge
    
    def modulate_aggression(self, base_aggression: float) -> float:
        """Modulate aggression based on vasopressin level.
        
        Args:
            base_aggression: Base aggression level
            
        Returns:
            Modulated aggression level
        """
        # High vasopressin increases territorial aggression
        if self.level > self.aggression_threshold:
            aggression_boost = (self.level - self.aggression_threshold) * 2.0
            return base_aggression * (1.0 + aggression_boost)
        return base_aggression
    
    def promote_pair_bonding(self, partner_quality: float) -> None:
        """Promote pair bonding (monogamous attachment).
        
        Args:
            partner_quality: Quality of partner (0-1)
        """
        # Vasopressin promotes selective pair bonding
        self.pair_bond_strength = min(1.0, self.pair_bond_strength + partner_quality * self.level * 0.1)
    
    def decay(self) -> None:
        """Decay vasopressin level toward baseline."""
        if self.level > self.baseline:
            self.level -= self.decay_rate
        elif self.level < self.baseline:
            self.level += self.decay_rate


@dataclass
class EndorphinSystem:
    """Endorphin (endogenous opioid) system for pain relief and reward.
    
    Endorphins include:
    - β-endorphin (main)
    - Enkephalins
    - Dynorphins
    
    Functions:
    - Pain relief (analgesia)
    - Reward and euphoria
    - Stress-induced analgesia
    - Runner's high
    """
    
    level: float = 0.2  # Current level (0-1)
    baseline: float = 0.2
    decay_rate: float = 0.1  # Relatively fast decay
    
    # Pain modulation
    analgesia_strength: float = 2.0
    
    # Reward enhancement
    reward_enhancement: float = 1.5
    
    def release(
        self,
        pain: float = 0.0,
        stress: float = 0.0,
        exercise: float = 0.0,
        pleasure: float = 0.0
    ) -> None:
        """Release endorphins in response to various stimuli.
        
        Args:
            pain: Pain level (triggers release)
            stress: Stress level
            exercise: Exercise intensity
            pleasure: Pleasurable activity
        """
        # Released in response to pain (stress-induced analgesia)
        if pain > 0.5:
            self.level = min(1.0, self.level + pain * 0.3)
        
        # Released during intense exercise
        if exercise > 0.6:
            self.level = min(1.0, self.level + exercise * 0.2)
        
        # Released during pleasure/reward
        self.level = min(1.0, self.level + pleasure * 0.1)
    
    def modulate_pain(self, pain_signal: float) -> float:
        """Modulate pain through analgesia.
        
        Args:
            pain_signal: Incoming pain signal
            
        Returns:
            Reduced pain signal
        """
        # High endorphins reduce pain (analgesia)
        pain_reduction = self.level * self.analgesia_strength
        return pain_signal * (1.0 - min(0.9, pain_reduction))
    
    def enhance_reward(self, base_reward: float) -> float:
        """Enhance reward sensation.
        
        Args:
            base_reward: Base reward signal
            
        Returns:
            Enhanced reward signal
        """
        # Endorphins contribute to euphoria
        return base_reward * (1.0 + self.level * self.reward_enhancement)
    
    def decay(self) -> None:
        """Decay endorphin level toward baseline (relatively fast)."""
        if self.level > self.baseline:
            self.level -= self.decay_rate
        elif self.level < self.baseline:
            self.level += self.decay_rate * 0.5


@dataclass
class NeuropeptideSystem:
    """Integrated neuropeptide system."""
    
    substance_p: SubstancePSystem = field(default_factory=SubstancePSystem)
    neuropeptide_y: NeuropeptideYSystem = field(default_factory=NeuropeptideYSystem)
    oxytocin: OxytocinSystem = field(default_factory=OxytocinSystem)
    vasopressin: VasopressinSystem = field(default_factory=VasopressinSystem)
    endorphins: EndorphinSystem = field(default_factory=EndorphinSystem)
    
    def step(self) -> None:
        """Update all neuropeptide systems."""
        self.substance_p.decay()
        self.neuropeptide_y.decay()
        self.oxytocin.decay()
        self.vasopressin.decay()
        self.endorphins.decay()
    
    def get_state(self) -> Dict[str, float]:
        """Get current state of all neuropeptides.
        
        Returns:
            Dictionary with neuropeptide levels
        """
        return {
            "substance_p": self.substance_p.level,
            "neuropeptide_y": self.neuropeptide_y.level,
            "oxytocin": self.oxytocin.level,
            "vasopressin": self.vasopressin.level,
            "endorphins": self.endorphins.level,
        }
    
    def process_pain(self, pain_signal: float, tissue_damage: float = 0.0) -> float:
        """Process pain through neuropeptide systems.
        
        Args:
            pain_signal: Base pain signal
            tissue_damage: Tissue damage level
            
        Returns:
            Final pain perception
        """
        # Substance P increases pain
        pain = self.substance_p.modulate_pain(pain_signal)
        self.substance_p.release(pain_signal, tissue_damage)
        
        # Endorphins reduce pain
        pain = self.endorphins.modulate_pain(pain)
        
        # Release endorphins if pain is high
        self.endorphins.release(pain=pain)
        
        return pain
    
    def process_social_interaction(
        self,
        interaction_quality: float,
        agent_id: str = "other",
        physical_contact: float = 0.0
    ) -> Dict[str, float]:
        """Process social interaction through neuropeptide systems.
        
        Args:
            interaction_quality: Quality of interaction (-1 to 1)
            agent_id: ID of social partner
            physical_contact: Level of physical contact
            
        Returns:
            Dictionary of social effects
        """
        effects = {}
        
        if interaction_quality > 0:
            # Positive interaction -> oxytocin
            self.oxytocin.release(interaction_quality, physical_contact)
            self.oxytocin.form_bond(agent_id, interaction_quality)
            effects["trust"] = self.oxytocin.modulate_trust(0.5)
            effects["prosocial"] = self.oxytocin.promote_prosocial()
        else:
            # Negative interaction -> vasopressin (threat response)
            threat = -interaction_quality
            self.vasopressin.update(social_threat=threat)
            effects["aggression"] = self.vasopressin.modulate_aggression(0.3)
        
        return effects
    
    def process_stress(self, stress_level: float) -> float:
        """Process stress through neuropeptide systems.
        
        Args:
            stress_level: Stress level (0-1)
            
        Returns:
            Buffered stress level
        """
        # NPY buffers stress
        buffered_stress = self.neuropeptide_y.buffer_stress(stress_level)
        self.neuropeptide_y.update(stress=stress_level)
        
        # High stress releases endorphins
        self.endorphins.release(stress=stress_level)
        
        return buffered_stress
