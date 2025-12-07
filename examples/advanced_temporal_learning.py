#!/usr/bin/env python3
"""Advanced Temporal Sequence Learning Example

This example demonstrates:
- Time-series pattern learning
- Sequence prediction
- Working memory for temporal patterns
- Attractor networks for sequence storage
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from brain_model import BrainModel
from simulation import Simulation
from senses import feed_sense_input
from digital_processing import TimeSeriesProcessor
from working_memory import (
    WorkingMemoryBuffer,
    AttractorNetwork,
    PersistentActivityManager
)


def generate_sine_sequence(length=50, frequency=0.1, amplitude=1.0):
    """Generate a sine wave sequence."""
    t = np.arange(length)
    sequence = amplitude * np.sin(2 * np.pi * frequency * t)
    return sequence


def generate_sawtooth_sequence(length=50, period=10):
    """Generate a sawtooth wave sequence."""
    sequence = np.array([i % period / period for i in range(length)])
    return sequence


def generate_step_sequence(length=50, step_size=10):
    """Generate a step function sequence."""
    sequence = np.array([i // step_size for i in range(length)], dtype=float)
    return sequence / np.max(sequence) if np.max(sequence) > 0 else sequence


def demonstrate_timeseries_features():
    """Demonstrate time-series feature extraction."""
    print("\n" + "="*60)
    print("TIME-SERIES FEATURE EXTRACTION")
    print("="*60)
    
    sequences = {
        'sine': generate_sine_sequence(),
        'sawtooth': generate_sawtooth_sequence(),
        'step': generate_step_sequence(),
    }
    
    for name, sequence in sequences.items():
        print(f"\n{name}:")
        
        # Extract features
        features = TimeSeriesProcessor.extract_features(sequence, window_size=10)
        print(f"  Mean: {features['mean']:.3f}")
        print(f"  Std: {features['std']:.3f}")
        print(f"  Min: {features['min']:.3f}")
        print(f"  Max: {features['max']:.3f}")
        print(f"  Trend: {features['trend']:.3f}")
        
        # Detect anomalies
        anomalies = TimeSeriesProcessor.detect_anomalies(sequence, threshold=2.0)
        num_anomalies = np.sum(anomalies)
        print(f"  Anomalies detected: {num_anomalies}")


def demonstrate_working_memory():
    """Demonstrate working memory buffer."""
    print("\n" + "="*60)
    print("WORKING MEMORY BUFFER DEMO")
    print("="*60)
    
    # Create memory buffer
    buffer = WorkingMemoryBuffer(num_slots=5, slot_size=10)
    
    # Store sequences
    sequences = [
        np.random.rand(10) for _ in range(3)
    ]
    
    print("\nStoring sequences in working memory...")
    for i, seq in enumerate(sequences):
        slot = buffer.store(seq)
        print(f"  Sequence {i+1} stored in slot {slot}")
    
    print(f"\nMemory occupancy: {buffer.get_occupancy()*100:.0f}%")
    
    # Retrieve sequences
    print("\nRetrieving sequences...")
    for i in range(3):
        retrieved = buffer.retrieve(i)
        if retrieved is not None:
            print(f"  Slot {i}: retrieved sequence mean = {np.mean(retrieved):.3f}")
    
    # Content-based search
    print("\nContent-based search...")
    query = sequences[0] + 0.1 * np.random.randn(10)  # Noisy version
    matches = buffer.search_content(query, top_k=2)
    print(f"  Query matches:")
    for slot, similarity in matches:
        print(f"    Slot {slot}: similarity = {similarity:.3f}")


def demonstrate_attractor_network():
    """Demonstrate attractor network for pattern completion."""
    print("\n" + "="*60)
    print("ATTRACTOR NETWORK DEMO")
    print("="*60)
    
    # Create attractor network
    network = AttractorNetwork(size=20, num_attractors=3)
    
    # Create and store patterns
    patterns = [
        np.array([1 if i < 10 else -1 for i in range(20)]),  # First half
        np.array([1 if i % 2 == 0 else -1 for i in range(20)]),  # Alternating
        np.array([1 if i > 10 else -1 for i in range(20)]),  # Second half
    ]
    
    print("\nStoring patterns...")
    for i, pattern in enumerate(patterns):
        network.store_pattern(pattern)
        print(f"  Pattern {i+1} stored")
    
    # Test pattern recall with noise
    print("\nRecalling patterns from noisy cues...")
    for i, pattern in enumerate(patterns):
        # Add noise
        noisy = pattern.copy()
        flip_indices = np.random.choice(20, size=5, replace=False)
        noisy[flip_indices] *= -1
        
        # Recall
        recalled, converged = network.recall(noisy, max_iterations=50)
        
        # Check accuracy
        accuracy = np.mean(recalled == pattern)
        print(f"  Pattern {i+1}: {accuracy*100:.0f}% accuracy, converged={converged}")


def train_sequence_predictor(sim, sequences, sequence_length=20):
    """Train network to predict sequences.
    
    Args:
        sim: Simulation instance
        sequences: Dictionary of sequence_name -> sequence_array
        sequence_length: Length of sequences to process
    """
    print("\n" + "="*60)
    print("TRAINING SEQUENCE PREDICTOR")
    print("="*60)
    
    for epoch in range(3):
        print(f"\nEpoch {epoch+1}/3")
        
        for name, sequence in sequences.items():
            # Use sliding windows
            windows = TimeSeriesProcessor.sliding_window(
                sequence[:sequence_length],
                window_size=5,
                stride=2
            )
            
            for window in windows:
                # Normalize window
                normalized = TimeSeriesProcessor.normalize_timeseries(window)
                
                # Convert to 2D input (reshape for vision area)
                input_2d = np.tile(normalized, (5, 1))
                
                # Feed to network
                feed_sense_input(sim.model, sense_name="digital", input_matrix=input_2d)
                
                # Process
                for step in range(10):
                    sim.step()
                    if step % 3 == 0:
                        sim.apply_plasticity()
                
                # Brief rest
                for step in range(3):
                    sim.step()


def test_sequence_prediction(sim, test_sequences):
    """Test sequence prediction capabilities."""
    print("\n" + "="*60)
    print("TESTING SEQUENCE PREDICTION")
    print("="*60)
    
    results = {}
    
    for name, sequence in test_sequences.items():
        # Present first half
        half_length = len(sequence) // 2
        first_half = sequence[:half_length]
        
        # Normalize and present
        normalized = TimeSeriesProcessor.normalize_timeseries(first_half)
        input_2d = np.tile(normalized, (5, 1))
        
        feed_sense_input(sim.model, sense_name="digital", input_matrix=input_2d)
        
        # Measure response
        spike_counts = []
        for step in range(20):
            spiked = sim.step()
            spike_counts.append(len(spiked))
        
        # Calculate response metrics
        total_spikes = sum(spike_counts)
        response_pattern = np.array(spike_counts)
        
        # Check if response has temporal structure
        autocorr = np.correlate(response_pattern, response_pattern, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        has_structure = np.std(autocorr[1:6]) > 0.1
        
        results[name] = {
            'total_spikes': total_spikes,
            'has_structure': has_structure,
            'response_var': np.var(spike_counts),
        }
        
        print(f"\n{name}:")
        print(f"  Total spikes: {total_spikes}")
        print(f"  Temporal structure: {'Yes' if has_structure else 'No'}")
        print(f"  Response variance: {np.var(spike_counts):.2f}")
        
        # Clear
        for step in range(5):
            sim.step()
    
    return results


def demonstrate_persistent_activity(sim):
    """Demonstrate persistent activity patterns."""
    print("\n" + "="*60)
    print("PERSISTENT ACTIVITY DEMO")
    print("="*60)
    
    # Create persistent activity manager
    manager = PersistentActivityManager(
        sim.model,
        memory_area="V1_like",  # Using available area
        maintenance_current=0.3
    )
    
    # Encode a pattern
    print("\nEncoding pattern into persistent activity...")
    pattern = np.array([1.0, 0.5, 0.8, 0.3, 0.6])
    manager.encode_pattern("test_pattern", pattern)
    
    # Maintain and monitor
    print("\nMaintaining activity over time...")
    for step in range(5):
        manager.maintain_activity("test_pattern")
        
        # Run simulation
        for _ in range(10):
            sim.step()
        
        # Check pattern state
        retrieved = manager.retrieve_pattern("test_pattern")
        if retrieved is not None:
            print(f"  Step {step+1}: mean activity = {np.mean(retrieved):.3f}")
    
    # Test decay
    print("\nTesting activity decay (no maintenance)...")
    for step in range(5):
        manager.decay_activity("test_pattern")
        
        for _ in range(10):
            sim.step()
        
        retrieved = manager.retrieve_pattern("test_pattern")
        if retrieved is not None:
            print(f"  Step {step+1}: mean activity = {np.mean(retrieved):.3f}")


def main():
    """Run the advanced temporal learning example."""
    print("="*60)
    print("ADVANCED TEMPORAL SEQUENCE LEARNING")
    print("Enhanced Time-Series Processing")
    print("="*60)
    
    # Demonstrate components
    demonstrate_timeseries_features()
    demonstrate_working_memory()
    demonstrate_attractor_network()
    
    # Setup neural network
    print("\n" + "="*60)
    print("NEURAL NETWORK SETUP")
    print("="*60)
    
    model = BrainModel(config_path=str(Path(__file__).parent.parent / "brain_base_model.json"))
    sim = Simulation(model, seed=42)
    
    print("\nInitializing network...")
    sim.initialize_neurons(area_names=["V1_like", "digital_area"], density=0.08)
    sim.initialize_random_synapses(
        connection_probability=0.12,
        weight_mean=0.4,
        weight_std=0.1
    )
    
    print(f"  Neurons: {len(sim.model.neurons)}")
    print(f"  Synapses: {len(sim.model.synapses)}")
    
    # Create temporal sequences
    sequences = {
        'sine': generate_sine_sequence(length=40),
        'sawtooth': generate_sawtooth_sequence(length=40),
        'step': generate_step_sequence(length=40),
    }
    
    # Train on sequences
    train_sequence_predictor(sim, sequences, sequence_length=30)
    
    # Test prediction
    test_sequences = {
        'sine_test': generate_sine_sequence(length=40, frequency=0.12),
        'sawtooth_test': generate_sawtooth_sequence(length=40, period=12),
    }
    results = test_sequence_prediction(sim, test_sequences)
    
    # Demonstrate persistent activity
    demonstrate_persistent_activity(sim)
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    print("\nSequence Recognition Results:")
    for name, result in results.items():
        print(f"  {name}:")
        print(f"    Response strength: {result['total_spikes']}")
        print(f"    Temporal structure: {'Yes' if result['has_structure'] else 'No'}")
    
    print("\n" + "="*60)
    print("EXAMPLE COMPLETE")
    print("="*60)
    print("\nKey Features Demonstrated:")
    print("  ✓ Time-series feature extraction")
    print("  ✓ Working memory buffer with content search")
    print("  ✓ Attractor networks for pattern completion")
    print("  ✓ Persistent activity patterns")
    print("  ✓ Sequence prediction and learning")
    
    print("\nNext Steps:")
    print("  • Train on longer sequences")
    print("  • Implement predictive coding")
    print("  • Add hierarchical temporal memory")
    print("  • Combine with motor output for closed-loop control")


if __name__ == "__main__":
    main()
