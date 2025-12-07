#!/usr/bin/env python3
"""Advanced Multi-Modal Integration Example

This example demonstrates:
- Integration of vision and digital senses
- Motor output generation
- Network connectivity analysis
- Population dynamics during multi-modal tasks
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from brain_model import BrainModel
from simulation import Simulation
from senses import feed_sense_input
from vision_processing import EdgeDetector, ColorProcessor
from digital_processing import NLPProcessor, TimeSeriesProcessor
from motor_output import extract_motor_commands, ActionSelector, ContinuousController
from network_analysis import ConnectivityAnalyzer, FiringPatternAnalyzer, PopulationDynamicsAnalyzer


def create_multimodal_stimuli():
    """Create paired visual and text stimuli."""
    stimuli = []
    
    # Stimulus 1: Circle with "round" text
    circle = np.zeros((15, 15))
    y, x = np.ogrid[-7:8, -7:8]
    mask = x**2 + y**2 <= 25
    circle[mask] = 1.0
    stimuli.append({
        'name': 'round_object',
        'visual': circle,
        'text': 'round circular shape',
        'expected_action': 'grab'
    })
    
    # Stimulus 2: Square with "box" text
    square = np.zeros((15, 15))
    square[3:12, 3:12] = 1.0
    stimuli.append({
        'name': 'box_object',
        'visual': square,
        'text': 'square box container',
        'expected_action': 'push'
    })
    
    # Stimulus 3: Horizontal line with "path" text
    line = np.zeros((15, 15))
    line[7, :] = 1.0
    stimuli.append({
        'name': 'path_object',
        'visual': line,
        'text': 'straight path line',
        'expected_action': 'follow'
    })
    
    return stimuli


def demonstrate_connectivity_analysis(sim):
    """Analyze network connectivity."""
    print("\n" + "="*60)
    print("NETWORK CONNECTIVITY ANALYSIS")
    print("="*60)
    
    analyzer = ConnectivityAnalyzer(sim.model)
    
    # Degree distribution
    print("\nDegree Distribution:")
    degrees = analyzer.compute_degree_distribution()
    print(f"  Mean in-degree: {np.mean(degrees['in_degree']):.2f}")
    print(f"  Mean out-degree: {np.mean(degrees['out_degree']):.2f}")
    print(f"  Max degree: {np.max(degrees['total_degree'])}")
    
    # Clustering coefficient
    print("\nClustering Analysis:")
    clustering = analyzer.compute_clustering_coefficient()
    print(f"  Clustering coefficient: {clustering:.4f}")
    
    # Hub identification
    print("\nHub Neurons (Top 5):")
    hubs = analyzer.identify_hubs(top_k=5)
    for i, (neuron_id, degree) in enumerate(hubs, 1):
        print(f"  {i}. Neuron {neuron_id}: degree = {degree}")
    
    # Path lengths
    print("\nPath Length Statistics:")
    path_stats = analyzer.compute_path_lengths(sample_size=50)
    print(f"  Mean path length: {path_stats['mean']:.2f}")
    print(f"  Max path length: {path_stats['max']:.0f}")


def demonstrate_firing_patterns(sim, stimuli):
    """Analyze firing patterns during stimulation."""
    print("\n" + "="*60)
    print("FIRING PATTERN ANALYSIS")
    print("="*60)
    
    analyzer = FiringPatternAnalyzer()
    
    for stimulus in stimuli:
        print(f"\n{stimulus['name']}:")
        
        # Present visual stimulus
        visual = EdgeDetector.sobel_edge_detection(stimulus['visual'])
        visual = ColorProcessor.normalize_channel(visual, 0.0, 1.0)
        feed_sense_input(sim.model, sense_name="vision", input_matrix=visual)
        
        # Present text stimulus
        text_vector = NLPProcessor.text_to_vector(stimulus['text'], vocab_size=100)
        text_2d = np.tile(text_vector[:10], (10, 1))
        feed_sense_input(sim.model, sense_name="digital", input_matrix=text_2d)
        
        # Record spikes
        for step in range(30):
            spiked = sim.step()
            current_time = step * 0.001  # Assuming 1ms per step
            for neuron_id in spiked:
                analyzer.record_spikes(neuron_id, current_time)
        
        # Clear inputs
        for step in range(10):
            sim.step()
    
    # Compute firing rates
    print("\nFiring Rate Statistics:")
    rates = analyzer.compute_firing_rates(time_window=0.05)
    if rates:
        rate_values = list(rates.values())
        print(f"  Mean rate: {np.mean(rate_values):.1f} Hz")
        print(f"  Max rate: {np.max(rate_values):.1f} Hz")
        print(f"  Active neurons: {len(rate_values)}")
    
    # Check synchrony
    neuron_ids = list(rates.keys())[:10]  # Sample 10 neurons
    if len(neuron_ids) >= 2:
        synchrony = analyzer.compute_synchrony(neuron_ids, time_window=0.005)
        print(f"\nSynchrony measure: {synchrony:.3f}")


def demonstrate_population_dynamics(sim, num_steps=100):
    """Analyze population-level dynamics."""
    print("\n" + "="*60)
    print("POPULATION DYNAMICS ANALYSIS")
    print("="*60)
    
    analyzer = PopulationDynamicsAnalyzer(sim.model)
    
    # Record activity over time
    print("\nRecording population activity...")
    for step in range(num_steps):
        # Add some periodic input
        if step % 20 < 10:
            # Create simple input pattern
            input_pattern = np.random.rand(15, 15) * 0.5
            feed_sense_input(sim.model, sense_name="vision", input_matrix=input_pattern)
        
        sim.step()
        analyzer.record_population_activity()
        
        if step % 25 == 0:
            rate = analyzer.compute_population_rate(threshold=0.0)
            print(f"  Step {step}: population rate = {rate:.3f}")
    
    # Compute statistics
    print("\nPopulation Statistics:")
    mean_field = analyzer.compute_mean_field()
    variance = analyzer.compute_variance()
    print(f"  Mean field activity: {np.mean(mean_field):.3f}")
    print(f"  Activity variance: {variance:.3f}")
    
    # Detect oscillations
    print("\nOscillation Detection:")
    osc_info = analyzer.detect_oscillations(min_freq=1.0, max_freq=50.0)
    if osc_info['detected']:
        print(f"  Oscillation detected!")
        print(f"    Period: {osc_info['period']} steps")
        print(f"    Strength: {osc_info['strength']:.3f}")
    else:
        print("  No significant oscillations detected")
    
    # Estimate dimensionality
    dim = analyzer.compute_dimensionality()
    print(f"\nEffective dimensionality: {dim}")


def demonstrate_motor_output(sim, stimuli):
    """Demonstrate motor command generation."""
    print("\n" + "="*60)
    print("MOTOR OUTPUT GENERATION")
    print("="*60)
    
    # Create action selector
    action_selector = ActionSelector(num_actions=4, selection_method='softmax')
    action_names = ['grab', 'push', 'follow', 'wait']
    
    # Create continuous controller
    continuous_controller = ContinuousController(output_dim=2, output_range=(-1.0, 1.0))
    
    for stimulus in stimuli:
        print(f"\n{stimulus['name']}:")
        
        # Present multi-modal input
        visual = EdgeDetector.sobel_edge_detection(stimulus['visual'])
        visual = ColorProcessor.normalize_channel(visual, 0.0, 1.0)
        feed_sense_input(sim.model, sense_name="vision", input_matrix=visual)
        
        text_vector = NLPProcessor.text_to_vector(stimulus['text'], vocab_size=100)
        text_2d = np.tile(text_vector[:10], (10, 1))
        feed_sense_input(sim.model, sense_name="digital", input_matrix=text_2d)
        
        # Let network process
        for step in range(20):
            sim.step()
        
        # Extract motor commands (discrete)
        motor_output = extract_motor_commands(
            sim.model,
            motor_area_name="V1_like",  # Using available area
            control_type="discrete",
            num_actions=4
        )
        
        # Select action
        if len(motor_output) == 4:
            selected_action = action_selector.select_action(motor_output, temperature=1.0)
            action_probs = action_selector.get_action_probabilities(motor_output, temperature=1.0)
            
            print(f"  Expected action: {stimulus['expected_action']}")
            print(f"  Selected action: {action_names[selected_action]}")
            print(f"  Action probabilities:")
            for name, prob in zip(action_names, action_probs):
                print(f"    {name}: {prob:.3f}")
        
        # Extract continuous control
        continuous_output = extract_motor_commands(
            sim.model,
            motor_area_name="V1_like",
            control_type="continuous",
            num_actions=2
        )
        
        print(f"  Continuous control: [{continuous_output[0]:.3f}, {continuous_output[1]:.3f}]")
        
        # Clear
        for step in range(10):
            sim.step()


def train_multimodal_association(sim, stimuli, epochs=3):
    """Train network on multi-modal associations."""
    print("\n" + "="*60)
    print("MULTI-MODAL ASSOCIATION TRAINING")
    print("="*60)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        for stimulus in stimuli:
            # Present both modalities simultaneously
            visual = EdgeDetector.sobel_edge_detection(stimulus['visual'])
            visual = ColorProcessor.normalize_channel(visual, 0.0, 1.0)
            feed_sense_input(sim.model, sense_name="vision", input_matrix=visual)
            
            text_vector = NLPProcessor.text_to_vector(stimulus['text'], vocab_size=100)
            text_2d = np.tile(text_vector[:10], (10, 1))
            feed_sense_input(sim.model, sense_name="digital", input_matrix=text_2d)
            
            # Train (plasticity applied automatically in step())
            for step in range(25):
                sim.step()
            
            # Rest
            for step in range(10):
                sim.step()
        
        # Check synaptic weights
        weights = [s.weight for s in sim.model.synapses.values()]
        print(f"  Mean weight: {np.mean(weights):.3f} (std: {np.std(weights):.3f})")


def main():
    """Run the advanced multi-modal integration example."""
    print("="*60)
    print("ADVANCED MULTI-MODAL INTEGRATION")
    print("Vision + Text + Motor Output")
    print("="*60)
    
    # Setup
    print("\n" + "="*60)
    print("NEURAL NETWORK SETUP")
    print("="*60)
    
    model = BrainModel(config_path=str(Path(__file__).parent.parent / "brain_base_model.json"))
    sim = Simulation(model, seed=42)
    
    print("\nInitializing multi-area network...")
    sim.initialize_neurons(area_names=["V1_like", "digital_area"], density=0.1)
    sim.initialize_random_synapses(
        connection_probability=0.15,
        weight_mean=0.5,
        weight_std=0.15
    )
    
    print(f"  Neurons: {len(sim.model.neurons)}")
    print(f"  Synapses: {len(sim.model.synapses)}")
    
    # Create stimuli
    stimuli = create_multimodal_stimuli()
    print(f"\nCreated {len(stimuli)} multi-modal stimuli")
    
    # Network analysis before training
    demonstrate_connectivity_analysis(sim)
    
    # Train
    train_multimodal_association(sim, stimuli, epochs=3)
    
    # Analyze after training
    demonstrate_firing_patterns(sim, stimuli)
    demonstrate_population_dynamics(sim, num_steps=100)
    demonstrate_motor_output(sim, stimuli)
    
    # Final summary
    print("\n" + "="*60)
    print("EXAMPLE COMPLETE")
    print("="*60)
    print("\nKey Features Demonstrated:")
    print("  ✓ Multi-modal sensory integration (vision + text)")
    print("  ✓ Network connectivity analysis")
    print("  ✓ Firing pattern analysis")
    print("  ✓ Population dynamics")
    print("  ✓ Motor output generation")
    print("  ✓ Discrete and continuous control")
    
    print("\nNext Steps:")
    print("  • Add more sensory modalities")
    print("  • Implement attention mechanisms")
    print("  • Create closed-loop sensorimotor tasks")
    print("  • Add reinforcement learning for motor control")


if __name__ == "__main__":
    main()
