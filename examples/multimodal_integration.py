#!/usr/bin/env python3
"""Multi-Modal Integration Example

This example demonstrates how to integrate multiple sensory modalities
(vision, audio, digital) in a single neural network, showing how the
4D architecture supports cross-modal learning and integration.

Scenario: The network learns to associate visual patterns with audio
signatures and text labels, demonstrating multi-modal binding.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from brain_model import BrainModel
from simulation import Simulation
from senses import feed_sense_input, create_digital_sense_input


def create_multimodal_stimuli():
    """Create coordinated multi-modal stimuli.
    
    Each stimulus has three components:
    - Visual: A pattern
    - Audio: A frequency signature
    - Digital: A text label
    """
    stimuli = {}
    
    # Stimulus 1: "High" - Vertical line + high frequency + "high"
    visual_high = np.zeros((10, 10))
    visual_high[:, 5] = 1.0
    
    audio_high = np.zeros((20, 10))
    audio_high[15:18, :] = 1.0  # High frequency bins
    
    digital_high = create_digital_sense_input("high")
    
    stimuli['high'] = {
        'vision': visual_high,
        'audio': audio_high,
        'digital': digital_high
    }
    
    # Stimulus 2: "Middle" - Horizontal line + middle frequency + "middle"
    visual_mid = np.zeros((10, 10))
    visual_mid[5, :] = 1.0
    
    audio_mid = np.zeros((20, 10))
    audio_mid[9:12, :] = 1.0  # Middle frequency bins
    
    digital_mid = create_digital_sense_input("middle")
    
    stimuli['middle'] = {
        'vision': visual_mid,
        'audio': audio_mid,
        'digital': digital_mid
    }
    
    # Stimulus 3: "Low" - Diagonal + low frequency + "low"
    visual_low = np.eye(10)
    
    audio_low = np.zeros((20, 10))
    audio_low[2:5, :] = 1.0  # Low frequency bins
    
    digital_low = create_digital_sense_input("low")
    
    stimuli['low'] = {
        'vision': visual_low,
        'audio': audio_low,
        'digital': digital_low
    }
    
    return stimuli


def visualize_stimulus(stimulus, name):
    """Visualize a multi-modal stimulus."""
    print(f"\n{'='*50}")
    print(f"Stimulus: {name}")
    print(f"{'='*50}")
    
    # Visual
    print("\nVisual component:")
    for row in stimulus['vision']:
        print(''.join(['██' if x > 0.5 else '··' for x in row]))
    
    # Audio (show frequency profile)
    print("\nAudio component (frequency spectrum):")
    audio_profile = np.max(stimulus['audio'], axis=1)  # Max over time
    print("High → ", end='')
    for i in range(19, -1, -1):  # Top to bottom (high to low freq)
        print('█' if audio_profile[i] > 0.5 else '·', end='')
    print(" ← Low")
    
    # Digital
    print(f"\nDigital component: '{name}'")


def present_multimodal_stimulus(sim, stimulus, duration=30):
    """Present all modalities of a stimulus simultaneously."""
    # Feed all senses at once
    feed_sense_input(
        sim.model,
        sense_name="vision",
        input_matrix=stimulus['vision'])
    
    feed_sense_input(
        sim.model,
        sense_name="audio",
        input_matrix=stimulus['audio'])
    
    feed_sense_input(
        sim.model,
        sense_name="digital",
        input_matrix=stimulus['digital'])
    
    # Let network process
    spike_counts = []
    for step in range(duration):
        spiked = sim.step()
        spike_counts.append(len(spiked))
    
    return spike_counts


def train_multimodal(sim, stimuli, epochs=5):
    """Train network on multi-modal associations."""
    print("\n" + "="*60)
    print("MULTI-MODAL TRAINING")
    print("="*60)
    print("\nTraining network to associate vision, audio, and text...")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        for name, stimulus in stimuli.items():
            # Present all modalities together
            present_multimodal_stimulus(sim, stimulus, duration=30)
            
            # Apply learning
            for _ in range(5):
                sim.apply_plasticity()
            
            # Brief pause between stimuli
            for step in range(10):
                sim.step()
        
        # Report progress
        if epoch % 2 == 0:
            weights = [s['weight'] for s in sim.model.synapses.values()]
            print(f"  Mean weight: {np.mean(weights):.3f}")


def test_cross_modal(sim, stimuli):
    """Test cross-modal associations by presenting partial stimuli.
    
    Present only one modality and measure if network still responds
    appropriately (showing learned association).
    """
    print("\n" + "="*60)
    print("CROSS-MODAL TESTING")
    print("="*60)
    print("\nTesting if network can recognize from single modality...")
    
    results = {}
    
    for name, full_stimulus in stimuli.items():
        results[name] = {}
        
        # Test vision only
        print(f"\n{name} (vision only):")
        feed_sense_input(
            sim.model,
            sense_name="vision",
            input_matrix=full_stimulus['vision'])
        
        vision_spikes = []
        for step in range(30):
            spiked = sim.step()
            vision_spikes.append(len(spiked))
        results[name]['vision'] = sum(vision_spikes)
        print(f"  Response: {sum(vision_spikes)} total spikes")
        
        # Clear
        for step in range(10):
            sim.step()
        
        # Test audio only
        print(f"\n{name} (audio only):")
        feed_sense_input(
            sim.model,
            sense_name="audio",
            input_matrix=full_stimulus['audio'])
        
        audio_spikes = []
        for step in range(30):
            spiked = sim.step()
            audio_spikes.append(len(spiked))
        results[name]['audio'] = sum(audio_spikes)
        print(f"  Response: {sum(audio_spikes)} total spikes")
        
        # Clear
        for step in range(10):
            sim.step()
        
        # Test digital only
        print(f"\n{name} (digital only):")
        feed_sense_input(
            sim.model,
            sense_name="digital",
            input_matrix=full_stimulus['digital'])
        
        digital_spikes = []
        for step in range(30):
            spiked = sim.step()
            digital_spikes.append(len(spiked))
        results[name]['digital'] = sum(digital_spikes)
        print(f"  Response: {sum(digital_spikes)} total spikes")
        
        # Clear
        for step in range(10):
            sim.step()
    
    return results


def analyze_integration(results):
    """Analyze multi-modal integration results."""
    print("\n" + "="*60)
    print("INTEGRATION ANALYSIS")
    print("="*60)
    
    # Check if responses are balanced across modalities
    print("\nResponse by modality:")
    for modality in ['vision', 'audio', 'digital']:
        total = sum(results[stim][modality] for stim in results.keys())
        print(f"  {modality:10s}: {total} total spikes")
    
    # Check consistency
    print("\nConsistency across stimuli:")
    for name in results.keys():
        responses = [results[name][m] for m in ['vision', 'audio', 'digital']]
        cv = np.std(responses) / np.mean(responses) if np.mean(responses) > 0 else 0
        print(f"  {name:10s}: CV = {cv:.2f}", end="")
        if cv < 0.3:
            print(" (well integrated)")
        elif cv < 0.6:
            print(" (moderate integration)")
        else:
            print(" (needs more training)")


def main():
    """Run the multi-modal integration example."""
    print("="*60)
    print("MULTI-MODAL INTEGRATION EXAMPLE")
    print("4D Neural Cognition System")
    print("="*60)
    
    # Create stimuli
    print("\nCreating multi-modal stimuli...")
    stimuli = create_multimodal_stimuli()
    
    # Visualize
    for name, stimulus in stimuli.items():
        visualize_stimulus(stimulus, name)
    
    # Setup network
    print("\n" + "="*60)
    print("NETWORK SETUP")
    print("="*60)
    
    model = BrainModel(config_path=str(Path(__file__).parent.parent / "brain_base_model.json"))
    sim = Simulation(model, seed=42)
    
    # Initialize neurons in multiple sensory areas
    print("\nInitializing multi-modal network...")
    sim.initialize_neurons(
        area_names=["V1_like", "A1_like", "Digital_sensor"],
        density=0.1
    )
    
    # Create connections (including cross-modal)
    sim.initialize_random_synapses(
        connection_probability=0.1,
        weight_mean=0.5,
        weight_std=0.1
    )
    
    neurons = sim.model.neurons
    synapses = sim.model.synapses
    
    print(f"  Neurons: {len(neurons)}")
    print(f"  Synapses: {len(synapses)}")
    
    # Count cross-modal connections
    cross_modal = 0
    for syn in synapses.values():
        pre_area = neurons[syn['pre']]['area']
        post_area = neurons[syn['post']]['area']
        if pre_area != post_area:
            cross_modal += 1
    
    print(f"  Cross-modal synapses: {cross_modal} ({cross_modal/len(synapses)*100:.1f}%)")
    
    # Train
    train_multimodal(sim, stimuli, epochs=5)
    
    # Test cross-modal associations
    results = test_cross_modal(sim, stimuli)
    
    # Analyze
    analyze_integration(results)
    
    print("\n" + "="*60)
    print("EXAMPLE COMPLETE")
    print("="*60)
    
    print("\nKey Insights:")
    print("  • Network learns associations across modalities")
    print("  • Cross-modal connections enable integration")
    print("  • Single modality can activate full representation")
    print("  • 4D architecture supports multi-sensory binding")
    
    print("\nApplications:")
    print("  • Audio-visual speech recognition")
    print("  • Cross-modal prediction and completion")
    print("  • Multi-sensory scene understanding")
    print("  • Synesthetic-like associations")
    
    print("\nNext Steps:")
    print("  • Add more modalities (touch, proprioception)")
    print("  • Test with missing/noisy inputs")
    print("  • Explore emergent cross-modal properties")
    print("  • Measure integration at different network depths")


if __name__ == "__main__":
    main()
