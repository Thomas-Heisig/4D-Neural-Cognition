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
        # TODO: Implement full self-other boundary detection
        # This would use:
        # 1. Agency detection (comparing motor intentions with sensory outcomes)
        # 2. Sensorimotor contingencies (predictable sensory changes from actions)
        # 3. Prediction error analysis (unexpected vs expected sensory feedback)
        # 4. Temporal binding (linking actions to effects across time)
        
        intentions = self_model_update.get('intentions', {})
        executed = self_model_update.get('executed_actions', {})
        
        # Placeholder: Simple consistency check
        # Future: Calculate prediction error between intentions and outcomes
        if intentions and executed:
            # Would calculate: error = |intended - actual|
            # Track error distribution to identify self vs environment
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
    
    def detect_self_consistency_anomalies(self) -> List[Dict]:
        """Detect discrepancies between self-model and reality.
        
        Analyzes prediction errors and cross-modal inconsistencies to
        identify when the self-model doesn't match actual sensory feedback.
        This is critical for maintaining an accurate body schema and
        detecting external perturbations.
        
        Returns:
            List of detected anomalies with type and severity
        """
        anomalies = []
        
        if len(self.stream) < 2:
            return anomalies
        
        # 1. Predictive self-perception: Expectation vs. Reality
        predicted_feedback = self.predict_next_proprioception()
        actual_feedback = self.integrated_self_model.get('body_state', {})
        
        if predicted_feedback and actual_feedback:
            discrepancy = self.discrepancy(predicted_feedback, actual_feedback)
            
            if discrepancy > 0.2:
                anomalies.append({
                    'type': 'motor_prediction_error',
                    'severity': 'high' if discrepancy > 0.5 else 'medium',
                    'value': discrepancy,
                    'implication': 'Body model inaccurate or external force',
                    'timestamp': time.time(),
                })
        
        # 2. Audiovisual self-consistency: Own voice + lip movement
        audio_self = self.integrated_self_model.get('own_voice', {})
        visual_self = self.integrated_self_model.get('self_image', {})
        
        if audio_self and visual_self:
            # Check for self-vocalization
            last_vocalization = audio_self.get('last_self_vocalization')
            mouth_movement = visual_self.get('self_mouth_movement')
            
            if last_vocalization and mouth_movement:
                sync_score = self.check_audio_visual_synchronization(
                    last_vocalization, mouth_movement
                )
                
                if sync_score < 0.7:
                    anomalies.append({
                        'type': 'av_sync_error',
                        'severity': 'medium',
                        'value': sync_score,
                        'implication': 'Self-voice recognition issue',
                        'timestamp': time.time(),
                    })
        
        # 3. Agency detection: Intentions vs. Outcomes
        intentions = self.integrated_self_model.get('intentions', {})
        executed = self.integrated_self_model.get('executed_actions', {})
        body_state = self.integrated_self_model.get('body_state', {})
        
        if intentions and executed and body_state:
            agency_score = self._calculate_agency_match(
                intentions, executed, body_state
            )
            
            if agency_score < 0.5:
                anomalies.append({
                    'type': 'agency_error',
                    'severity': 'high' if agency_score < 0.3 else 'medium',
                    'value': agency_score,
                    'implication': 'Loss of control or external manipulation',
                    'timestamp': time.time(),
                })
        
        return anomalies
    
    def predict_next_proprioception(self) -> Dict:
        """Predict next proprioceptive state based on motor intentions.
        
        Uses recent history and motor commands to predict expected
        sensory feedback (forward model).
        
        Returns:
            Predicted proprioceptive state
        """
        if len(self.stream) < 5:
            return {}
        
        # Get recent history
        recent = list(self.stream)[-5:]
        
        # Extract velocity/trend from body state
        predicted = {}
        
        # Simple linear extrapolation (placeholder for learned model)
        if recent[-1].get('body_state'):
            current_state = recent[-1]['body_state']
            
            # Predict based on motor intentions
            intentions = self.integrated_self_model.get('intentions', {})
            
            if intentions:
                # Apply intended changes to current state
                predicted = current_state.copy()
                
                # Modify based on intentions (simplified)
                for key, value in intentions.items():
                    if key in predicted:
                        predicted[key] = predicted[key] + value * 0.1
            else:
                # No intentions: maintain current state
                predicted = current_state.copy()
        
        return predicted
    
    def discrepancy(self, predicted: Dict, actual: Dict) -> float:
        """Calculate discrepancy between predicted and actual state.
        
        Args:
            predicted: Predicted state
            actual: Actual state
            
        Returns:
            Discrepancy score (0-1, higher = more discrepant)
        """
        if not predicted or not actual:
            return 0.0
        
        # Compare all matching keys
        errors = []
        
        for key in predicted:
            if key in actual:
                pred_val = predicted[key]
                actual_val = actual[key]
                
                # Handle nested dicts (like joint_angles)
                if isinstance(pred_val, dict) and isinstance(actual_val, dict):
                    for subkey in pred_val:
                        if subkey in actual_val:
                            error = abs(pred_val[subkey] - actual_val[subkey])
                            errors.append(error)
                elif isinstance(pred_val, (int, float)) and isinstance(actual_val, (int, float)):
                    error = abs(pred_val - actual_val)
                    errors.append(error)
        
        if errors:
            return float(np.mean(errors))
        return 0.0
    
    def check_audio_visual_synchronization(
        self,
        audio_data: Dict,
        visual_data: Dict
    ) -> float:
        """Check synchronization between audio and visual self-signals.
        
        Args:
            audio_data: Audio self-signal data
            visual_data: Visual self-signal data
            
        Returns:
            Synchronization score (0-1, higher = better sync)
        """
        # Extract timestamps
        audio_time = audio_data.get('timestamp', 0)
        visual_time = visual_data.get('timestamp', 0)
        
        # Calculate temporal offset
        offset_ms = abs(audio_time - visual_time) * 1000
        
        # Good sync within 50ms, degraded up to 200ms
        if offset_ms < 50:
            sync_score = 1.0
        elif offset_ms < 200:
            sync_score = 1.0 - (offset_ms - 50) / 150
        else:
            sync_score = 0.0
        
        # Also check magnitude correlation if available
        audio_magnitude = audio_data.get('magnitude', 0)
        visual_magnitude = visual_data.get('magnitude', 0)
        
        if audio_magnitude > 0 and visual_magnitude > 0:
            magnitude_correlation = min(audio_magnitude, visual_magnitude) / max(audio_magnitude, visual_magnitude)
            sync_score = (sync_score + magnitude_correlation) / 2.0
        
        return sync_score
    
    def _calculate_agency_match(
        self,
        intentions: Dict,
        executed: Dict,
        outcomes: Dict
    ) -> float:
        """Calculate how well intentions match outcomes (sense of agency).
        
        Args:
            intentions: Intended actions
            executed: Executed actions
            outcomes: Resulting sensory feedback
            
        Returns:
            Agency score (0-1, higher = stronger sense of agency)
        """
        if not intentions or not outcomes:
            return 0.5  # Neutral
        
        # Check if executed actions match intentions
        execution_match = 0.0
        if executed:
            matches = sum(
                1 for key in intentions
                if key in executed and abs(intentions[key] - executed.get(key, 0)) < 0.1
            )
            if intentions:
                execution_match = matches / len(intentions)
        
        # Check if outcomes are consistent with intentions
        # (simplified: would use learned forward model)
        outcome_match = 0.5  # Placeholder
        
        # Combined agency score
        agency_score = (execution_match + outcome_match) / 2.0
        
        return agency_score
    
    def update_self_model_based_on_anomalies(
        self,
        anomalies: List[Dict]
    ) -> Dict:
        """Adapt self-model when inconsistencies are detected.
        
        Recalibrates internal models based on prediction errors and
        inconsistencies. This maintains an accurate self-representation
        even as the body changes or external forces are applied.
        
        Args:
            anomalies: List of detected anomalies
            
        Returns:
            Dictionary with recalibration results
        """
        recalibration_results = {
            'anomalies_processed': len(anomalies),
            'recalibrations': [],
        }
        
        for anomaly in anomalies:
            anomaly_type = anomaly['type']
            severity = anomaly['severity']
            
            # Determine learning rate based on severity
            if severity == 'high':
                learning_rate = 0.05
            elif severity == 'medium':
                learning_rate = 0.02
            else:
                learning_rate = 0.01
            
            if anomaly_type == 'motor_prediction_error':
                # Recalibrate proprioceptive model
                result = self._recalibrate_body_model(learning_rate)
                recalibration_results['recalibrations'].append({
                    'type': 'body_model',
                    'learning_rate': learning_rate,
                    'result': result,
                })
            
            elif anomaly_type == 'av_sync_error':
                # Adjust audio-visual delay compensation
                result = self._adjust_av_sync_delay(learning_rate)
                recalibration_results['recalibrations'].append({
                    'type': 'av_sync_delay',
                    'adjustment': result,
                })
            
            elif anomaly_type == 'agency_error':
                # Update motor-sensory mapping
                result = self._update_agency_model(learning_rate)
                recalibration_results['recalibrations'].append({
                    'type': 'agency_model',
                    'learning_rate': learning_rate,
                    'result': result,
                })
        
        return recalibration_results
    
    def _recalibrate_body_model(self, learning_rate: float) -> Dict:
        """Recalibrate proprioceptive body model.
        
        Args:
            learning_rate: Rate of adaptation
            
        Returns:
            Recalibration result
        """
        # Placeholder: would update internal forward model
        # For now, track that recalibration occurred
        if not hasattr(self, 'body_model_calibration'):
            self.body_model_calibration = 1.0
        
        # Adjust calibration factor
        self.body_model_calibration *= (1.0 - learning_rate * 0.1)
        
        return {
            'calibration_factor': self.body_model_calibration,
            'updated': True,
        }
    
    def _adjust_av_sync_delay(self, learning_rate: float) -> Dict:
        """Adjust audio-visual synchronization delay.
        
        Args:
            learning_rate: Rate of adjustment
            
        Returns:
            Adjustment result
        """
        if not hasattr(self, 'av_sync_delay'):
            self.av_sync_delay = 0  # ms
        
        # Adjust delay (simplified)
        self.av_sync_delay += 5 * learning_rate  # Add 5ms per adjustment
        
        return {
            'new_delay_ms': self.av_sync_delay,
            'updated': True,
        }
    
    def _update_agency_model(self, learning_rate: float) -> Dict:
        """Update motor-sensory agency model.
        
        Args:
            learning_rate: Rate of update
            
        Returns:
            Update result
        """
        # Placeholder: would update agency forward model
        if not hasattr(self, 'agency_model_confidence'):
            self.agency_model_confidence = 1.0
        
        # Reduce confidence when agency errors occur
        self.agency_model_confidence *= (1.0 - learning_rate)
        
        return {
            'confidence': self.agency_model_confidence,
            'updated': True,
        }
    
    def reset(self) -> None:
        """Reset self-perception stream."""
        self.stream.clear()
        self.integrated_self_model = {}
        self.update_count = 0
        self.start_time = time.time()
        
        # Reset calibration state
        if hasattr(self, 'body_model_calibration'):
            self.body_model_calibration = 1.0
        if hasattr(self, 'av_sync_delay'):
            self.av_sync_delay = 0
        if hasattr(self, 'agency_model_confidence'):
            self.agency_model_confidence = 1.0
        
        logger.info("Self-perception stream reset")
    
    def get_statistics(self) -> Dict:
        """Get stream statistics.
        
        Returns:
            Dictionary with stream metrics
        """
        stats = {
            "buffer_size": self.buffer_size,
            "current_size": len(self.stream),
            "update_count": self.update_count,
            "configured_frequency_hz": self.update_frequency_hz,
            "actual_frequency_hz": self._calculate_actual_frequency(),
            "uptime_seconds": time.time() - self.start_time,
        }
        
        # Add calibration state if available
        if hasattr(self, 'body_model_calibration'):
            stats['body_model_calibration'] = self.body_model_calibration
        if hasattr(self, 'av_sync_delay'):
            stats['av_sync_delay_ms'] = self.av_sync_delay
        if hasattr(self, 'agency_model_confidence'):
            stats['agency_confidence'] = self.agency_model_confidence
        
        return stats
