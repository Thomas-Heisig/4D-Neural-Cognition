"""Brain states and global dynamics for 4D Neural Cognition.

This module implements global brain states:
- Sleep stages (NREM N1-N3, REM)
- Arousal states (alert, drowsy, unconscious)
- Epileptiform activity (interictal, ictal)
- Anesthesia levels

References:
- Steriade, M., et al. (2001). Natural waking and sleep states
- Brown, E.N., et al. (2010). General anesthesia, sleep, and coma
- Saper, C.B., et al. (2005). Sleep state switching
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import numpy as np


class SleepStage(Enum):
    """Sleep stages."""
    WAKE = "wake"
    N1 = "n1"  # Light sleep, theta waves
    N2 = "n2"  # Sleep spindles, K-complexes
    N3 = "n3"  # Slow wave sleep, delta waves
    REM = "rem"  # Rapid eye movement, dreaming


class ArousalLevel(Enum):
    """Arousal/consciousness levels."""
    ALERT = "alert"
    DROWSY = "drowsy"
    LIGHT_SLEEP = "light_sleep"
    DEEP_SLEEP = "deep_sleep"
    UNCONSCIOUS = "unconscious"
    COMA = "coma"


@dataclass
class SleepWakeRegulation:
    """Sleep-wake regulation system.
    
    Implements:
    - Circadian rhythm (24h cycle)
    - Homeostatic sleep pressure (Process S)
    - Flip-flop switch (wake vs sleep promoting nuclei)
    """
    
    # Current state
    current_stage: SleepStage = SleepStage.WAKE
    time_in_stage: int = 0
    
    # Circadian rhythm (Process C)
    circadian_phase: float = 0.5  # 0=night, 1=day
    circadian_period: int = 24000  # Steps per cycle (24h)
    
    # Homeostatic sleep pressure (Process S)
    sleep_pressure: float = 0.0
    wake_accumulation_rate: float = 0.0001  # Increases during wake
    sleep_dissipation_rate: float = 0.0002  # Decreases during sleep
    
    # Sleep/wake promoting signals
    wake_promoting: float = 0.5  # Orexin, NE, ACh, histamine
    sleep_promoting: float = 0.5  # GABA, adenosine
    
    # Transition thresholds (configurable)
    wake_to_sleep_threshold: float = 0.7
    sleep_to_wake_threshold: float = 0.3
    
    # Configuration constants
    SLEEP_PRESSURE_THRESHOLD: float = 0.5  # For N2->N3 transition
    
    # REM regulation
    rem_pressure: float = 0.0
    rem_accumulation_rate: float = 0.0001
    
    def update_circadian(self, dt: int = 1) -> None:
        """Update circadian rhythm.
        
        Args:
            dt: Time step
        """
        # Advance circadian phase
        self.circadian_phase += dt / self.circadian_period
        self.circadian_phase = self.circadian_phase % 1.0
        
        # Wake-promoting signal peaks during day
        self.wake_promoting = 0.5 + 0.5 * np.sin(2 * np.pi * self.circadian_phase)
    
    def update_homeostatic(self, dt: int = 1) -> None:
        """Update homeostatic sleep pressure.
        
        Args:
            dt: Time step
        """
        if self.current_stage == SleepStage.WAKE:
            # Accumulate sleep pressure during wake
            self.sleep_pressure += self.wake_accumulation_rate * dt
            self.sleep_pressure = min(1.0, self.sleep_pressure)
        else:
            # Dissipate sleep pressure during sleep
            self.sleep_pressure -= self.sleep_dissipation_rate * dt
            self.sleep_pressure = max(0.0, self.sleep_pressure)
        
        # Sleep pressure opposes wake-promoting
        self.sleep_promoting = self.sleep_pressure
    
    def update_rem_pressure(self, dt: int = 1) -> None:
        """Update REM pressure.
        
        Args:
            dt: Time step
        """
        if self.current_stage in [SleepStage.N2, SleepStage.N3]:
            # Accumulate REM pressure during NREM
            self.rem_pressure += self.rem_accumulation_rate * dt
            self.rem_pressure = min(1.0, self.rem_pressure)
        elif self.current_stage == SleepStage.REM:
            # Dissipate during REM
            self.rem_pressure -= 0.001 * dt
            self.rem_pressure = max(0.0, self.rem_pressure)
    
    def transition_stage(self) -> Optional[SleepStage]:
        """Determine if stage transition should occur.
        
        Returns:
            New stage if transition occurs, None otherwise
        """
        # Wake/sleep flip-flop
        wake_drive = self.wake_promoting - self.sleep_promoting
        
        if self.current_stage == SleepStage.WAKE:
            if wake_drive < -self.wake_to_sleep_threshold:
                return SleepStage.N1
        
        elif self.current_stage == SleepStage.N1:
            if self.time_in_stage > 500:  # ~5 min
                return SleepStage.N2
            if wake_drive > self.sleep_to_wake_threshold:
                return SleepStage.WAKE
        
        elif self.current_stage == SleepStage.N2:
            if self.time_in_stage > 1000:
                if self.sleep_pressure > self.SLEEP_PRESSURE_THRESHOLD:
                    return SleepStage.N3
                elif self.rem_pressure > 0.7:
                    return SleepStage.REM
            if wake_drive > self.sleep_to_wake_threshold:
                return SleepStage.WAKE
        
        elif self.current_stage == SleepStage.N3:
            if self.sleep_pressure < 0.3 or self.time_in_stage > 3000:
                return SleepStage.N2
        
        elif self.current_stage == SleepStage.REM:
            if self.rem_pressure < 0.2 or self.time_in_stage > 2000:
                return SleepStage.N2
            if wake_drive > 0.5:  # Easier to wake from REM
                return SleepStage.WAKE
        
        return None
    
    def step(self, dt: int = 1) -> SleepStage:
        """Update sleep-wake regulation.
        
        Args:
            dt: Time step
            
        Returns:
            Current sleep stage
        """
        self.update_circadian(dt)
        self.update_homeostatic(dt)
        self.update_rem_pressure(dt)
        
        # Check for stage transition
        new_stage = self.transition_stage()
        if new_stage is not None:
            self.current_stage = new_stage
            self.time_in_stage = 0
        else:
            self.time_in_stage += dt
        
        return self.current_stage


@dataclass
class BrainRhythm:
    """Neural oscillation/rhythm generator.
    
    Generates characteristic brain rhythms:
    - Delta (0.5-4 Hz): Deep sleep
    - Theta (4-8 Hz): Light sleep, meditation
    - Alpha (8-13 Hz): Relaxed wakefulness
    - Beta (13-30 Hz): Active thinking
    - Gamma (30-100 Hz): Attention, binding
    """
    
    frequency: float  # Hz
    amplitude: float = 1.0
    phase: float = 0.0
    
    def generate(self, dt: float = 0.001) -> float:
        """Generate oscillation value.
        
        Args:
            dt: Time step (in seconds)
            
        Returns:
            Oscillation amplitude
        """
        self.phase += 2 * np.pi * self.frequency * dt
        self.phase = self.phase % (2 * np.pi)
        return self.amplitude * np.sin(self.phase)


@dataclass
class SleepSpindle:
    """Sleep spindle generator (12-16 Hz bursts during N2).
    
    Sleep spindles are:
    - 12-16 Hz oscillations
    - 0.5-2 second duration
    - Generated by thalamic reticular nucleus
    - Important for memory consolidation
    """
    
    frequency: float = 14.0  # Hz
    duration: int = 1000  # ms
    active: bool = False
    time_active: int = 0
    
    def start(self) -> None:
        """Start a spindle."""
        self.active = True
        self.time_active = 0
    
    def update(self, dt: int = 1) -> Optional[float]:
        """Update spindle.
        
        Args:
            dt: Time step (ms)
            
        Returns:
            Spindle amplitude if active, None otherwise
        """
        if not self.active:
            return None
        
        self.time_active += dt
        
        # End spindle after duration
        if self.time_active >= self.duration:
            self.active = False
            return None
        
        # Envelope: rise, sustain, fall
        progress = self.time_active / self.duration
        if progress < 0.2:
            envelope = progress / 0.2
        elif progress > 0.8:
            envelope = (1.0 - progress) / 0.2
        else:
            envelope = 1.0
        
        # Oscillation
        phase = 2 * np.pi * self.frequency * self.time_active / 1000.0
        return envelope * np.sin(phase)


@dataclass
class KComplex:
    """K-complex generator (sharp wave during N2).
    
    K-complexes are:
    - Large amplitude, sharp waveforms
    - Occur spontaneously or in response to stimuli
    - Mark transition to deeper sleep
    - May protect sleep from disturbances
    """
    
    duration: int = 500  # ms
    amplitude: float = 2.0
    active: bool = False
    time_active: int = 0
    
    def trigger(self) -> None:
        """Trigger a K-complex."""
        self.active = True
        self.time_active = 0
    
    def update(self, dt: int = 1) -> Optional[float]:
        """Update K-complex.
        
        Args:
            dt: Time step (ms)
            
        Returns:
            K-complex amplitude if active, None otherwise
        """
        if not self.active:
            return None
        
        self.time_active += dt
        
        if self.time_active >= self.duration:
            self.active = False
            return None
        
        # Biphasic waveform (negative then positive)
        t = self.time_active / self.duration
        waveform = self.amplitude * (np.sin(2 * np.pi * t) - 0.5 * np.sin(4 * np.pi * t))
        
        return waveform


@dataclass
class BrainStateManager:
    """Manager for global brain states."""
    
    # Sleep-wake regulation
    sleep_wake: SleepWakeRegulation = field(default_factory=SleepWakeRegulation)
    
    # Arousal level
    arousal: ArousalLevel = ArousalLevel.ALERT
    
    # Brain rhythms
    delta: BrainRhythm = field(default_factory=lambda: BrainRhythm(frequency=2.0))
    theta: BrainRhythm = field(default_factory=lambda: BrainRhythm(frequency=6.0))
    alpha: BrainRhythm = field(default_factory=lambda: BrainRhythm(frequency=10.0))
    beta: BrainRhythm = field(default_factory=lambda: BrainRhythm(frequency=20.0))
    gamma: BrainRhythm = field(default_factory=lambda: BrainRhythm(frequency=40.0))
    
    # Sleep-specific features
    spindle: SleepSpindle = field(default_factory=SleepSpindle)
    k_complex: KComplex = field(default_factory=KComplex)
    
    # Spontaneous event probabilities
    spindle_probability: float = 0.001  # Per step in N2
    k_complex_probability: float = 0.0005  # Per step in N2
    
    def update_rhythms(self, stage: SleepStage, dt: float = 0.001) -> Dict[str, float]:
        """Update brain rhythms based on sleep stage.
        
        Args:
            stage: Current sleep stage
            dt: Time step (seconds)
            
        Returns:
            Dictionary of rhythm amplitudes
        """
        rhythms = {}
        
        # Set amplitudes based on stage
        if stage == SleepStage.WAKE:
            self.alpha.amplitude = 0.5
            self.beta.amplitude = 1.0
            self.gamma.amplitude = 0.8
            self.delta.amplitude = 0.1
            self.theta.amplitude = 0.2
        
        elif stage == SleepStage.N1:
            self.alpha.amplitude = 0.3
            self.theta.amplitude = 1.0
            self.beta.amplitude = 0.3
            self.gamma.amplitude = 0.2
            self.delta.amplitude = 0.3
        
        elif stage == SleepStage.N2:
            self.theta.amplitude = 0.7
            self.delta.amplitude = 0.8
            self.alpha.amplitude = 0.1
            self.beta.amplitude = 0.1
            self.gamma.amplitude = 0.1
        
        elif stage == SleepStage.N3:
            self.delta.amplitude = 2.0  # Dominant
            self.theta.amplitude = 0.3
            self.alpha.amplitude = 0.0
            self.beta.amplitude = 0.0
            self.gamma.amplitude = 0.0
        
        elif stage == SleepStage.REM:
            # Similar to wake but with theta
            self.theta.amplitude = 1.0
            self.alpha.amplitude = 0.4
            self.beta.amplitude = 0.6
            self.gamma.amplitude = 0.7
            self.delta.amplitude = 0.1
        
        # Generate rhythms
        rhythms["delta"] = self.delta.generate(dt)
        rhythms["theta"] = self.theta.generate(dt)
        rhythms["alpha"] = self.alpha.generate(dt)
        rhythms["beta"] = self.beta.generate(dt)
        rhythms["gamma"] = self.gamma.generate(dt)
        
        # Add sleep-specific features
        if stage == SleepStage.N2:
            # Spontaneous spindles
            if not self.spindle.active and np.random.random() < self.spindle_probability:
                self.spindle.start()
            
            spindle_amp = self.spindle.update()
            if spindle_amp is not None:
                rhythms["spindle"] = spindle_amp
            
            # Spontaneous K-complexes
            if not self.k_complex.active and np.random.random() < self.k_complex_probability:
                self.k_complex.trigger()
            
            k_amp = self.k_complex.update()
            if k_amp is not None:
                rhythms["k_complex"] = k_amp
        
        return rhythms
    
    def step(self, dt: int = 1) -> Dict[str, any]:
        """Update brain state.
        
        Args:
            dt: Time step
            
        Returns:
            Dictionary of current state information
        """
        # Update sleep-wake regulation
        stage = self.sleep_wake.step(dt)
        
        # Update arousal based on stage
        if stage == SleepStage.WAKE:
            self.arousal = ArousalLevel.ALERT
        elif stage == SleepStage.N1:
            self.arousal = ArousalLevel.DROWSY
        elif stage == SleepStage.N2:
            self.arousal = ArousalLevel.LIGHT_SLEEP
        elif stage == SleepStage.N3:
            self.arousal = ArousalLevel.DEEP_SLEEP
        elif stage == SleepStage.REM:
            self.arousal = ArousalLevel.LIGHT_SLEEP
        
        # Update rhythms
        rhythms = self.update_rhythms(stage, dt / 1000.0)
        
        return {
            "stage": stage,
            "arousal": self.arousal,
            "circadian_phase": self.sleep_wake.circadian_phase,
            "sleep_pressure": self.sleep_wake.sleep_pressure,
            "rhythms": rhythms,
        }
    
    def get_neural_modulation(self) -> Dict[str, float]:
        """Get neural modulation factors based on brain state.
        
        Returns:
            Dictionary of modulation factors
        """
        stage = self.sleep_wake.current_stage
        
        if stage == SleepStage.WAKE:
            return {
                "excitability": 1.0,
                "plasticity": 1.0,
                "noise": 0.5,
            }
        elif stage in [SleepStage.N1, SleepStage.N2]:
            return {
                "excitability": 0.7,
                "plasticity": 1.2,  # Enhanced consolidation
                "noise": 0.3,
            }
        elif stage == SleepStage.N3:
            return {
                "excitability": 0.4,
                "plasticity": 1.5,  # Maximum consolidation
                "noise": 0.2,
            }
        elif stage == SleepStage.REM:
            return {
                "excitability": 0.8,
                "plasticity": 1.3,  # Consolidation with high activity
                "noise": 0.6,  # High variability
            }
        else:
            return {
                "excitability": 1.0,
                "plasticity": 1.0,
                "noise": 0.5,
            }


@dataclass
class EpileptiformActivity:
    """Epileptiform activity patterns.
    
    Models:
    - Interictal spikes (between seizures)
    - Ictal activity (seizures)
    - Postictal depression
    """
    
    # State
    in_seizure: bool = False
    seizure_duration: int = 0
    time_since_seizure: int = 10000
    
    # Seizure parameters
    seizure_threshold: float = 0.9  # Hyperexcitability threshold
    typical_duration: int = 5000  # ms
    
    # Interictal activity
    interictal_spike_rate: float = 0.001  # Spikes per step
    
    def update(
        self,
        network_excitability: float,
        dt: int = 1
    ) -> Dict[str, any]:
        """Update epileptiform activity.
        
        Args:
            network_excitability: Current network excitability (0-1)
            dt: Time step
            
        Returns:
            Dictionary with seizure state
        """
        if not self.in_seizure:
            self.time_since_seizure += dt
            
            # Check for seizure onset
            if network_excitability > self.seizure_threshold:
                self.in_seizure = True
                self.seizure_duration = 0
                return {"state": "ictal_onset", "duration": 0}
            
            # Interictal spikes
            if np.random.random() < self.interictal_spike_rate:
                return {"state": "interictal_spike", "amplitude": 2.0}
        
        else:
            # In seizure
            self.seizure_duration += dt
            
            # Check for termination
            if self.seizure_duration >= self.typical_duration:
                self.in_seizure = False
                self.time_since_seizure = 0
                return {"state": "postictal", "duration": self.seizure_duration}
            
            return {"state": "ictal", "duration": self.seizure_duration}
        
        return {"state": "normal"}
