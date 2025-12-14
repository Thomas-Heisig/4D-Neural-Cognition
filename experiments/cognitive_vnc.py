"""Experiment: Cognitive-Aware VNC Orchestration.

This experiment tests the cognitive-aware VNC orchestrator's ability
to dynamically allocate VPUs to critical brain regions during motor
learning, and measures the impact on learning speed.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from brain_model import BrainModel
from simulation import Simulation
from hardware_abstraction.adaptive_vnc_orchestrator import CognitiveAwareOrchestrator
from hardware_abstraction.virtual_clock import GlobalVirtualClock
from hardware_abstraction.vectorized_vpu import VectorizedVPU

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_brain_model() -> BrainModel:
    """Create brain model for VNC testing.
    
    Returns:
        BrainModel instance
    """
    config = {
        "lattice_shape": [20, 20, 10, 15],
        "neuron_model": {
            "type": "lif",
            "params_default": {
                "threshold": -50.0,
                "reset_potential": -65.0,
                "tau_membrane": 20.0,
                "refractory_period": 2,
            }
        },
        "cell_lifecycle": {
            "neurogenesis_rate": 0.0,
            "apoptosis_threshold": 0.0,
        },
        "plasticity": {
            "stdp_enabled": True,
            "learning_rate": 0.01,
        },
        "senses": {
            "digital": {"areal": "V1"}
        },
        "areas": [
            {
                "name": "M1",  # Motor cortex (w=10)
                "bounds": {"x": [0, 20], "y": [0, 20], "z": [0, 10], "w": [10, 10]},
                "neuron_type": "excitatory",
            },
            {
                "name": "S1",  # Somatosensory (w=6)
                "bounds": {"x": [0, 20], "y": [0, 20], "z": [0, 10], "w": [6, 6]},
                "neuron_type": "excitatory",
            },
            {
                "name": "PFC",  # Prefrontal (w=14)
                "bounds": {"x": [0, 20], "y": [0, 20], "z": [0, 10], "w": [14, 14]},
                "neuron_type": "excitatory",
            },
        ]
    }
    
    return BrainModel(config=config)


def simulate_motor_learning_activity(
    sim: Simulation,
    intensity: float = 0.8,
    duration_cycles: int = 50,
) -> Dict:
    """Simulate motor learning activity in the brain.
    
    Args:
        sim: Simulation instance
        intensity: Activity intensity (0-1)
        duration_cycles: How long to simulate
        
    Returns:
        Activity statistics
    """
    spikes_generated = 0
    
    # Get motor neurons (w=10)
    motor_neurons = [
        nid for nid, n in sim.model.neurons.items()
        if n.w == 10
    ]
    
    # Get sensory neurons (w=6)
    sensory_neurons = [
        nid for nid, n in sim.model.neurons.items()
        if n.w == 6
    ]
    
    for cycle in range(duration_cycles):
        # Stimulate motor and sensory neurons
        for nid in motor_neurons[:int(len(motor_neurons) * intensity)]:
            if nid in sim.model.neurons:
                neuron = sim.model.neurons[nid]
                neuron.external_input = 50.0 * intensity
        
        for nid in sensory_neurons[:int(len(sensory_neurons) * intensity * 0.5)]:
            if nid in sim.model.neurons:
                neuron = sim.model.neurons[nid]
                neuron.external_input = 30.0 * intensity
        
        # Step simulation
        sim.step()
        
        # Count spikes
        if hasattr(sim, 'spike_history') and sim.spike_history:
            spikes_generated += len(sim.spike_history[-1])
    
    return {
        'total_spikes': spikes_generated,
        'cycles': duration_cycles,
        'avg_spikes_per_cycle': spikes_generated / duration_cycles if duration_cycles > 0 else 0,
    }


def run_experiment(
    scenario: str = "motor_learning",
    use_cognitive_orchestrator: bool = True,
    num_vpus: int = 4,
    learning_duration_cycles: int = 500,
    measure_interval: int = 50,
    output_file: str = None,
) -> Dict:
    """Run cognitive VNC orchestration experiment.
    
    Args:
        scenario: Scenario to test
        use_cognitive_orchestrator: Whether to use cognitive orchestrator
        num_vpus: Number of VPUs
        learning_duration_cycles: Duration of learning simulation
        measure_interval: Measurement interval
        output_file: Optional output file
        
    Returns:
        Experiment results
    """
    logger.info(
        f"Starting cognitive VNC experiment: "
        f"scenario={scenario}, "
        f"cognitive_orchestrator={use_cognitive_orchestrator}, "
        f"vpus={num_vpus}"
    )
    
    # Create brain model
    brain = create_test_brain_model()
    
    # Add some neurons
    logger.info("Creating neurons...")
    neuron_count = 0
    for x in range(0, 20, 2):
        for y in range(0, 20, 2):
            for z in range(0, 10, 2):
                for w in [6, 10, 14]:  # Sensory, Motor, Executive
                    brain.add_neuron(x, y, z, w)
                    neuron_count += 1
    
    logger.info(f"Created {neuron_count} neurons")
    
    # Create simulation
    sim = Simulation(brain)
    
    # Create VNC system
    logger.info("Setting up VNC system...")
    virtual_clock = GlobalVirtualClock(frequency_mhz=20.0)
    
    # Create VPUs with w-slice partitioning
    w_slices = [6, 10, 14]  # One VPU per critical slice initially
    for i, w in enumerate(w_slices):
        neurons_in_slice = [
            (nid, n) for nid, n in brain.neurons.items()
            if n.w == w
        ]
        
        if neurons_in_slice:
            vpu = VectorizedVPU(
                vpu_id=i,
                neuron_batch=dict(neurons_in_slice)
            )
            vpu.w_slice = w  # Tag with w-slice
            virtual_clock.register_vpu(vpu)
    
    # Add extra VPUs if requested
    for i in range(len(w_slices), num_vpus):
        # Distribute extra VPUs across slices
        w = w_slices[i % len(w_slices)]
        neurons_in_slice = [
            (nid, n) for nid, n in brain.neurons.items()
            if n.w == w
        ]
        
        if neurons_in_slice:
            # Split neurons for extra VPU
            split_point = len(neurons_in_slice) // 2
            vpu = VectorizedVPU(
                vpu_id=i,
                neuron_batch=dict(neurons_in_slice[split_point:])
            )
            vpu.w_slice = w
            virtual_clock.register_vpu(vpu)
    
    sim.virtual_clock = virtual_clock
    
    logger.info(f"Created {len(virtual_clock.vpus)} VPUs")
    
    # Create orchestrator
    if use_cognitive_orchestrator:
        orchestrator = CognitiveAwareOrchestrator(
            simulation=sim,
            monitoring_interval=measure_interval,
        )
        logger.info("Using CognitiveAwareOrchestrator")
    else:
        orchestrator = None
        logger.info("No orchestrator (baseline)")
    
    # Run learning simulation
    logger.info(f"Running {learning_duration_cycles} cycles of motor learning...")
    
    measurements = []
    
    for cycle in range(0, learning_duration_cycles, measure_interval):
        # Simulate motor learning activity
        activity_stats = simulate_motor_learning_activity(
            sim=sim,
            intensity=0.7,
            duration_cycles=measure_interval,
        )
        
        # Monitor and adapt with orchestrator
        if orchestrator:
            orchestrator_result = orchestrator.monitor_and_adapt(sim.current_step)
        else:
            orchestrator_result = None
        
        # Collect VPU statistics
        vpu_stats = []
        for vpu in virtual_clock.vpus:
            stats = vpu.get_statistics()
            stats['vpu_id'] = vpu.vpu_id
            stats['w_slice'] = getattr(vpu, 'w_slice', None)
            vpu_stats.append(stats)
        
        measurement = {
            'cycle_range': (cycle, cycle + measure_interval),
            'activity': activity_stats,
            'vpu_stats': vpu_stats,
            'orchestrator_result': orchestrator_result,
        }
        
        measurements.append(measurement)
        
        if (cycle // measure_interval + 1) % 5 == 0:
            logger.info(
                f"  Cycle {cycle}/{learning_duration_cycles}: "
                f"spikes={activity_stats['total_spikes']}"
            )
    
    # Analyze results
    logger.info("Analyzing results...")
    
    # Calculate average VPU allocation per region
    region_vpu_counts = {6: [], 10: [], 14: []}
    
    for measurement in measurements:
        for w_slice in [6, 10, 14]:
            count = sum(
                1 for vpu_stat in measurement['vpu_stats']
                if vpu_stat.get('w_slice') == w_slice
            )
            region_vpu_counts[w_slice].append(count)
    
    avg_vpus_per_region = {
        w: np.mean(counts) if counts else 0
        for w, counts in region_vpu_counts.items()
    }
    
    # Calculate learning speed proxy (total spikes / time)
    total_spikes = sum(
        m['activity']['total_spikes'] for m in measurements
    )
    learning_speed = total_spikes / learning_duration_cycles
    
    # Get orchestrator performance if used
    orchestrator_performance = None
    if orchestrator:
        orchestrator_performance = orchestrator.get_cognitive_performance_summary()
    
    results = {
        'config': {
            'scenario': scenario,
            'use_cognitive_orchestrator': use_cognitive_orchestrator,
            'num_vpus': num_vpus,
            'learning_duration_cycles': learning_duration_cycles,
        },
        'metrics': {
            'total_spikes': total_spikes,
            'learning_speed_spikes_per_cycle': learning_speed,
            'avg_vpus_per_region': avg_vpus_per_region,
        },
        'orchestrator_performance': orchestrator_performance,
        'measurements': measurements,
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
    logger.info(f"Scenario: {scenario}")
    logger.info(f"Cognitive orchestrator: {use_cognitive_orchestrator}")
    logger.info(f"Total spikes: {total_spikes}")
    logger.info(f"Learning speed: {learning_speed:.2f} spikes/cycle")
    logger.info("Average VPUs per region:")
    for w, count in avg_vpus_per_region.items():
        region_name = {6: "Sensory", 10: "Motor", 14: "Executive"}[w]
        logger.info(f"  {region_name} (w={w}): {count:.2f}")
    
    if orchestrator_performance:
        logger.info(f"Total reallocations: {orchestrator_performance['total_reallocations']}")
    
    logger.info("="*60)
    
    return results


def main():
    """Main entry point for experiment."""
    parser = argparse.ArgumentParser(
        description='Run cognitive VNC orchestration experiment'
    )
    parser.add_argument(
        '--scenario',
        type=str,
        default='motor_learning',
        choices=['motor_learning'],
        help='Scenario to test'
    )
    parser.add_argument(
        '--cognitive',
        action='store_true',
        help='Use cognitive-aware orchestrator'
    )
    parser.add_argument(
        '--baseline',
        action='store_true',
        help='Run baseline without orchestrator'
    )
    parser.add_argument(
        '--vpus',
        type=int,
        default=4,
        help='Number of VPUs'
    )
    parser.add_argument(
        '--cycles',
        type=int,
        default=500,
        help='Learning duration in cycles'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for results (JSON)'
    )
    
    args = parser.parse_args()
    
    # Determine if using orchestrator
    use_orchestrator = args.cognitive or not args.baseline
    
    # Run experiment
    results = run_experiment(
        scenario=args.scenario,
        use_cognitive_orchestrator=use_orchestrator,
        num_vpus=args.vpus,
        learning_duration_cycles=args.cycles,
        output_file=args.output,
    )
    
    return results


if __name__ == '__main__':
    main()
