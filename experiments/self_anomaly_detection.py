"""Experiment: Self-Anomaly Detection and Model Recalibration.

This experiment tests the self-perception stream's ability to detect
anomalies (like external perturbations) and recalibrate the self-model
in response.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from consciousness.self_perception_stream import SelfPerceptionStream
from embodiment.virtual_body import VirtualBody

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerturbationSimulator:
    """Simulates external perturbations to the body.
    
    Can apply various types of external forces or disturbances
    to test anomaly detection.
    
    Attributes:
        perturbation_type: Type of perturbation to apply
        magnitude: Strength of perturbation
        duration_steps: How long perturbation lasts
    """
    
    def __init__(
        self,
        perturbation_type: str = "external_push",
        magnitude: float = 1.0,
        duration_steps: int = 5,
    ):
        """Initialize perturbation simulator.
        
        Args:
            perturbation_type: Type of perturbation
            magnitude: Perturbation strength
            duration_steps: Duration in steps
        """
        self.perturbation_type = perturbation_type
        self.magnitude = magnitude
        self.duration_steps = duration_steps
        
        self.active = False
        self.steps_remaining = 0
        
        logger.info(
            f"Initialized PerturbationSimulator: "
            f"type={perturbation_type}, magnitude={magnitude}"
        )
    
    def trigger(self) -> None:
        """Trigger the perturbation."""
        self.active = True
        self.steps_remaining = self.duration_steps
        logger.info(f"Perturbation triggered: {self.perturbation_type}")
    
    def apply(self, body_state: Dict) -> Dict:
        """Apply perturbation to body state.
        
        Args:
            body_state: Current body state
            
        Returns:
            Modified body state
        """
        if not self.active:
            return body_state
        
        # Decrement duration
        self.steps_remaining -= 1
        if self.steps_remaining <= 0:
            self.active = False
            logger.info("Perturbation ended")
        
        # Apply perturbation based on type
        perturbed_state = body_state.copy()
        
        if self.perturbation_type == "external_push":
            # Add unexpected displacement to position
            if 'position' in perturbed_state:
                push_vector = np.random.randn(3) * self.magnitude
                perturbed_state['position'] = (
                    np.array(perturbed_state['position']) + push_vector
                )
            
            # Modify joint angles unexpectedly
            if 'joint_angles' in perturbed_state:
                for joint_id in perturbed_state['joint_angles']:
                    perturbed_state['joint_angles'][joint_id] += (
                        np.random.randn() * self.magnitude * 0.1
                    )
        
        elif self.perturbation_type == "sensor_noise":
            # Add noise to all sensors
            if 'joint_angles' in perturbed_state:
                for joint_id in perturbed_state['joint_angles']:
                    perturbed_state['joint_angles'][joint_id] += (
                        np.random.randn() * self.magnitude * 0.5
                    )
        
        elif self.perturbation_type == "motor_failure":
            # Simulate motor command not executing properly
            # (would need motor command context)
            pass
        
        return perturbed_state


def run_experiment(
    perturbation: str = "external_push",
    magnitude: float = 0.5,
    num_trials: int = 10,
    steps_per_trial: int = 100,
    perturbation_timing: str = "middle",
    output_file: str = None,
) -> Dict:
    """Run self-anomaly detection experiment.
    
    Args:
        perturbation: Type of perturbation to test
        magnitude: Perturbation magnitude
        num_trials: Number of trials to run
        steps_per_trial: Steps per trial
        perturbation_timing: When to apply perturbation ('early', 'middle', 'late')
        output_file: Optional output file for results
        
    Returns:
        Dictionary with experiment results
    """
    logger.info(
        f"Starting self-anomaly detection experiment: "
        f"perturbation={perturbation}, magnitude={magnitude}, "
        f"trials={num_trials}"
    )
    
    # Create components
    body = VirtualBody(body_type="humanoid", num_joints=8)
    perception_stream = SelfPerceptionStream(
        update_frequency_hz=100.0,
        buffer_duration_seconds=2.0
    )
    perturbator = PerturbationSimulator(
        perturbation_type=perturbation,
        magnitude=magnitude,
        duration_steps=5
    )
    
    # Results storage
    trial_results = []
    
    for trial in range(num_trials):
        logger.info(f"\nTrial {trial + 1}/{num_trials}")
        
        # Reset
        body.reset()
        perception_stream.reset()
        
        # Determine perturbation timing
        if perturbation_timing == "early":
            perturbation_step = steps_per_trial // 4
        elif perturbation_timing == "late":
            perturbation_step = 3 * steps_per_trial // 4
        else:  # middle
            perturbation_step = steps_per_trial // 2
        
        trial_data = {
            'trial': trial,
            'anomalies_detected': [],
            'prediction_errors': [],
            'recalibrations': [],
            'detection_latency': None,
            'perturbation_step': perturbation_step,
        }
        
        perturbation_triggered = False
        detection_occurred = False
        
        # Run trial
        for step in range(steps_per_trial):
            # Trigger perturbation at designated time
            if step == perturbation_step:
                perturbator.trigger()
                perturbation_triggered = True
            
            # Generate motor command (random movement)
            motor_command = {
                'planned': {f'joint_{i}': np.random.rand() * 0.3 for i in range(4)},
                'executed': {f'joint_{i}': np.random.rand() * 0.3 for i in range(4)},
            }
            
            # Execute motor command
            feedback = body.execute_motor_command({
                'motor_neurons': {i: np.random.rand() * 0.5 for i in range(4)}
            })
            
            # Apply perturbation if active
            feedback = perturbator.apply(feedback)
            
            # Update self-perception stream
            perception_stream.update(
                sensor_data={'proprioception': feedback},
                motor_commands=motor_command,
                internal_state={'metabolic': {}, 'attention': {}, 'emotion': {}},
            )
            
            # Detect anomalies
            anomalies = perception_stream.detect_self_consistency_anomalies()
            
            if anomalies:
                trial_data['anomalies_detected'].extend(anomalies)
                
                # Track detection latency (time from perturbation to detection)
                if perturbation_triggered and not detection_occurred:
                    detection_occurred = True
                    trial_data['detection_latency'] = step - perturbation_step
                    logger.info(
                        f"  Anomaly detected at step {step} "
                        f"(latency: {trial_data['detection_latency']} steps)"
                    )
                
                # Update self-model based on anomalies
                recalibration = perception_stream.update_self_model_based_on_anomalies(
                    anomalies
                )
                trial_data['recalibrations'].append({
                    'step': step,
                    'result': recalibration,
                })
            
            # Track prediction errors
            if hasattr(perception_stream, 'integrated_self_model'):
                # Calculate prediction error (simplified)
                predicted = perception_stream.predict_next_proprioception()
                actual = feedback
                error = perception_stream.discrepancy(predicted, actual)
                trial_data['prediction_errors'].append({
                    'step': step,
                    'error': error,
                    'perturbation_active': perturbator.active,
                })
        
        # Trial summary
        num_anomalies = len(trial_data['anomalies_detected'])
        avg_error_before = np.mean([
            pe['error'] for pe in trial_data['prediction_errors']
            if pe['step'] < perturbation_step
        ]) if trial_data['prediction_errors'] else 0.0
        
        avg_error_during = np.mean([
            pe['error'] for pe in trial_data['prediction_errors']
            if perturbation_step <= pe['step'] < perturbation_step + 10
        ]) if trial_data['prediction_errors'] else 0.0
        
        avg_error_after = np.mean([
            pe['error'] for pe in trial_data['prediction_errors']
            if pe['step'] >= perturbation_step + 10
        ]) if trial_data['prediction_errors'] else 0.0
        
        trial_data['summary'] = {
            'num_anomalies': num_anomalies,
            'detection_success': detection_occurred,
            'avg_error_before': avg_error_before,
            'avg_error_during': avg_error_during,
            'avg_error_after': avg_error_after,
            'recalibration_count': len(trial_data['recalibrations']),
        }
        
        trial_results.append(trial_data)
        
        logger.info(f"  Anomalies detected: {num_anomalies}")
        logger.info(f"  Detection success: {detection_occurred}")
        logger.info(f"  Recalibrations: {len(trial_data['recalibrations'])}")
    
    # Aggregate results
    detection_rate = sum(
        1 for t in trial_results if t['summary']['detection_success']
    ) / num_trials
    
    avg_detection_latency = np.mean([
        t['detection_latency'] for t in trial_results
        if t['detection_latency'] is not None
    ]) if any(t['detection_latency'] is not None for t in trial_results) else None
    
    avg_anomalies_per_trial = np.mean([
        t['summary']['num_anomalies'] for t in trial_results
    ])
    
    avg_recalibrations_per_trial = np.mean([
        t['summary']['recalibration_count'] for t in trial_results
    ])
    
    results = {
        'config': {
            'perturbation': perturbation,
            'magnitude': magnitude,
            'num_trials': num_trials,
            'steps_per_trial': steps_per_trial,
            'perturbation_timing': perturbation_timing,
        },
        'aggregate_metrics': {
            'detection_rate': detection_rate,
            'avg_detection_latency_steps': avg_detection_latency,
            'avg_anomalies_per_trial': avg_anomalies_per_trial,
            'avg_recalibrations_per_trial': avg_recalibrations_per_trial,
        },
        'trial_results': trial_results,
    }
    
    # Save results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*60)
    logger.info(f"Perturbation type: {perturbation}")
    logger.info(f"Detection rate: {detection_rate:.1%}")
    if avg_detection_latency is not None:
        logger.info(f"Avg detection latency: {avg_detection_latency:.1f} steps")
    logger.info(f"Avg anomalies per trial: {avg_anomalies_per_trial:.1f}")
    logger.info(f"Avg recalibrations per trial: {avg_recalibrations_per_trial:.1f}")
    logger.info("="*60)
    
    return results


def main():
    """Main entry point for experiment."""
    parser = argparse.ArgumentParser(
        description='Run self-anomaly detection experiment'
    )
    parser.add_argument(
        '--perturbation',
        type=str,
        default='external_push',
        choices=['external_push', 'sensor_noise', 'motor_failure'],
        help='Type of perturbation to test'
    )
    parser.add_argument(
        '--magnitude',
        type=float,
        default=0.5,
        help='Perturbation magnitude'
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=10,
        help='Number of trials'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=100,
        help='Steps per trial'
    )
    parser.add_argument(
        '--timing',
        type=str,
        default='middle',
        choices=['early', 'middle', 'late'],
        help='When to apply perturbation'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for results (JSON)'
    )
    
    args = parser.parse_args()
    
    # Run experiment
    results = run_experiment(
        perturbation=args.perturbation,
        magnitude=args.magnitude,
        num_trials=args.trials,
        steps_per_trial=args.steps,
        perturbation_timing=args.timing,
        output_file=args.output,
    )
    
    return results


if __name__ == '__main__':
    main()
