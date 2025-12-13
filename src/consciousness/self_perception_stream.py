"""Self-Perception Stream for continuous self-awareness.

This module implements a continuous stream of self-perception data that
integrates all aspects of the agent's self-model.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Dict, List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SelfPerceptionStream:
    """Continuous stream of self-perception integrating all self-modalities.
    
    This class maintains a continuous buffer of self-perception data that
    includes proprioception, self-generated sounds, visual self-image,
    motor intentions, and internal states. It enables temporal integration
    of self-information and tracks self-consistency over time.
    
    Attributes:
        stream: Circular buffer of self-perception snapshots
        update_frequency_hz: Rate of updates (default: 100 Hz)
        buffer_size: Maximum number of snapshots to keep
        integrated_self_model: Current integrated self-representation
    """
    
    def __init__(
        self,
        update_frequency_hz: float = 100.0,
        buffer_duration_seconds: float = 10.0,
    ):
        """Initialize self-perception stream.
        
        Args:
            update_frequency_hz: Frequency of self-perception updates
            buffer_duration_seconds: How long to keep historical data
        """
        self.update_frequency_hz = update_frequency_hz
        self.buffer_size = int(update_frequency_hz * buffer_duration_seconds)
        
        # Circular buffer for self-perception history
        self.stream: deque = deque(maxlen=self.buffer_size)
        
        # Integrated self-model (current state)
        self.integrated_self_model: Dict = {}
        
        # Statistics
        self.update_count = 0
        self.start_time = time.time()
        
        logger.info(
            f"Initialized SelfPerceptionStream "
            f"({update_frequency_hz} Hz, {buffer_duration_seconds}s buffer)"
        )
    
    def update(
        self,
        sensor_data: Dict,
        motor_commands: Dict,
        internal_state: Dict,
        timestamp_ns: Optional[int] = None,
    ) -> None:
        """Update self-perception with latest data.
        
        Integrates all sources of self-information into a coherent
        self-model snapshot.
        
        Args:
            sensor_data: Sensory feedback about self
            motor_commands: Motor intentions and executed commands
            internal_state: Internal physiological/metabolic state
            timestamp_ns: Optional timestamp (nanoseconds), defaults to current time
        """
        if timestamp_ns is None:
            timestamp_ns = time.time_ns()
        
        # Create self-model snapshot
        self_model_update = {
            "timestamp_ns": timestamp_ns,
            "timestamp_s": timestamp_ns / 1e9,
            
            # Body state (proprioception)
            "body_state": sensor_data.get('proprioception', {}),
            
            # Self-generated audio
            "own_voice": sensor_data.get('audio_self', {}),
            
            # Visual self-perception
            "self_image": sensor_data.get('visual_self', {}),
            
            # Motor intentions
            "intentions": motor_commands.get('planned', {}),
            
            # Executed motor commands
            "executed_actions": motor_commands.get('executed', {}),
            
            # Internal state
            "vital_signs": internal_state.get('metabolic', {}),
            
            # Higher-level cognitive state
            "attention_focus": internal_state.get('attention', {}),
            "emotional_state": internal_state.get('emotion', {}),
        }
        
        # Add to stream
        self.stream.append(self_model_update)
        
        # Update integrated self-model
        self.integrated_self_model = self_model_update
        
        # Update self-other boundary
        self._update_self_other_boundary(self_model_update)
        
        self.update_count += 1
    
    def _update_self_other_boundary(self, self_model_update: Dict) -> None:
        """Update the boundary between self and environment.
        
        Analyzes consistency between motor intentions and sensory feedback
        to maintain self-other distinction.
        
        Args:
            self_model_update: Latest self-perception snapshot
        """
        # This is a placeholder for self-other boundary detection
        # Full implementation would use agency detection, sensorimotor
        # contingencies, and prediction error analysis
        
        intentions = self_model_update.get('intentions', {})
        executed = self_model_update.get('executed_actions', {})
        
        # Simple consistency check
        if intentions and executed:
            # Calculate how well intentions match execution
            # (simplified metric)
            pass
    
    def get_self_awareness_metric(self) -> Dict:
        """Quantify degree of self-awareness.
        
        Measures:
        1. Temporal consistency: How stable is the self-model over time
        2. Cross-modal integration: How well different self-aspects align
        3. Agency detection: How well motor intentions predict outcomes
        
        Returns:
            Dictionary with self-awareness metrics
        """
        if len(self.stream) < 2:
            return {
                "self_consistency": 0.0,
                "integration": 0.0,
                "agency_score": 0.0,
                "data_points": len(self.stream),
            }
        
        # Calculate temporal consistency
        consistency = self._calculate_temporal_consistency()
        
        # Calculate cross-modal integration
        integration = self._calculate_cross_modal_integration()
        
        # Calculate agency score
        agency = self._calculate_agency_score()
        
        return {
            "self_consistency": consistency,
            "integration": integration,
            "agency_score": agency,
            "data_points": len(self.stream),
            "update_frequency_actual": self._calculate_actual_frequency(),
        }
    
    def _calculate_temporal_consistency(self) -> float:
        """Calculate temporal consistency of self-model.
        
        Measures how stable the self-model is over time. High consistency
        indicates a coherent sense of self.
        
        Returns:
            Consistency score (0-1, higher is more consistent)
        """
        if len(self.stream) < 10:
            return 0.0
        
        # Sample recent snapshots
        recent = list(self.stream)[-10:]
        
        # Calculate variance in key self-attributes
        # (simplified - just checks if data exists and varies smoothly)
        has_body_state = sum(1 for s in recent if s.get('body_state')) / len(recent)
        has_intentions = sum(1 for s in recent if s.get('intentions')) / len(recent)
        has_self_image = sum(1 for s in recent if s.get('self_image')) / len(recent)
        
        # Average presence of self-information
        consistency = (has_body_state + has_intentions + has_self_image) / 3.0
        
        return consistency
    
    def _calculate_cross_modal_integration(self) -> float:
        """Calculate cross-modal integration score.
        
        Measures how well different modalities of self-information are
        integrated into a coherent whole.
        
        Returns:
            Integration score (0-1, higher is better integrated)
        """
        if not self.integrated_self_model:
            return 0.0
        
        # Count how many self-modalities are present
        modalities = [
            'body_state',
            'own_voice',
            'self_image',
            'intentions',
            'executed_actions',
            'vital_signs',
        ]
        
        present_modalities = sum(
            1 for mod in modalities
            if self.integrated_self_model.get(mod)
        )
        
        integration = present_modalities / len(modalities)
        
        return integration
    
    def _calculate_agency_score(self) -> float:
        """Calculate sense of agency score.
        
        Measures how well motor intentions predict sensory outcomes,
        indicating a sense of control and agency.
        
        Returns:
            Agency score (0-1, higher indicates stronger sense of agency)
        """
        if len(self.stream) < 5:
            return 0.0
        
        # Sample recent snapshots
        recent = list(self.stream)[-5:]
        
        # Check consistency between intentions and executed actions
        matches = 0
        for snapshot in recent:
            intentions = snapshot.get('intentions', {})
            executed = snapshot.get('executed_actions', {})
            
            if intentions and executed:
                # Simple check: if both present, assume some agency
                matches += 1
        
        agency = matches / len(recent)
        
        return agency
    
    def _calculate_actual_frequency(self) -> float:
        """Calculate actual update frequency.
        
        Returns:
            Updates per second
        """
        if self.update_count == 0:
            return 0.0
        
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return 0.0
        
        return self.update_count / elapsed
    
    def get_recent_history(self, duration_seconds: float = 1.0) -> List[Dict]:
        """Get recent self-perception history.
        
        Args:
            duration_seconds: How far back to retrieve
            
        Returns:
            List of self-perception snapshots
        """
        if not self.stream:
            return []
        
        current_time = time.time()
        cutoff_time = current_time - duration_seconds
        
        recent = [
            snapshot for snapshot in self.stream
            if snapshot['timestamp_s'] >= cutoff_time
        ]
        
        return recent
    
    def get_self_trajectory(
        self,
        attribute: str,
        duration_seconds: float = 5.0,
    ) -> List[float]:
        """Get temporal trajectory of a specific self-attribute.
        
        Args:
            attribute: Name of attribute to track (e.g., 'body_state.velocity')
            duration_seconds: How far back to track
            
        Returns:
            List of values over time
        """
        history = self.get_recent_history(duration_seconds)
        
        trajectory = []
        for snapshot in history:
            # Navigate nested dictionary
            value = snapshot
            for key in attribute.split('.'):
                value = value.get(key, {})
                if not isinstance(value, dict):
                    break
            
            if value is not None and not isinstance(value, dict):
                trajectory.append(value)
        
        return trajectory
    
    def reset(self) -> None:
        """Reset self-perception stream."""
        self.stream.clear()
        self.integrated_self_model = {}
        self.update_count = 0
        self.start_time = time.time()
        
        logger.info("Self-perception stream reset")
    
    def get_statistics(self) -> Dict:
        """Get stream statistics.
        
        Returns:
            Dictionary with stream metrics
        """
        return {
            "buffer_size": self.buffer_size,
            "current_size": len(self.stream),
            "update_count": self.update_count,
            "configured_frequency_hz": self.update_frequency_hz,
            "actual_frequency_hz": self._calculate_actual_frequency(),
            "uptime_seconds": time.time() - self.start_time,
        }
