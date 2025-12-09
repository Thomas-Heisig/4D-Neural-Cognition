"""Advanced example demonstrating long-term memory, attention, and analysis features.

This example shows how to use:
- Long-term memory consolidation
- Memory replay mechanisms
- Sleep-like states
- Attention mechanisms
- Phase space analysis
- Network motif detection

Run with: python examples/advanced_memory_example.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from brain_model import BrainModel
from simulation import Simulation
from longterm_memory import MemoryConsolidation, MemoryReplay, SleepLikeState
from working_memory import AttentionMechanism
from visualization import plot_phase_space, plot_network_motifs
from network_analysis import NetworkMotifDetector


def main():
    """Main example function."""
    print("=" * 60)
    print("Advanced Memory & Attention Example")
    print("=" * 60)
    
    # 1. Create brain model
    print("\n1. Creating brain model...")
    # Use robust path resolution
    config_path = os.path.join(os.path.dirname(__file__), "..", "brain_base_model.json")
    model = BrainModel(config_path=config_path)
    sim = Simulation(model, seed=42)
    
    # Initialize neurons in specific areas
    sim.initialize_neurons(area_names=["hippocampus", "temporal_cortex", "prefrontal_cortex"], density=0.3)
    sim.initialize_random_synapses(connection_probability=0.02)
    
    print(f"   Created model with {len(model.neurons)} neurons and {len(model.synapses)} synapses")
    
    # 2. Set up memory systems
    print("\n2. Setting up memory systems...")
    consolidator = MemoryConsolidation(
        model,
        short_term_area="hippocampus",
        long_term_area="temporal_cortex"
    )
    
    replay = MemoryReplay(model, replay_area="hippocampus")
    sleep = SleepLikeState(model, consolidator, replay)
    
    print("   Memory systems initialized")
    
    # 3. Set up attention mechanism
    print("\n3. Setting up attention mechanism...")
    attention = AttentionMechanism(model)
    
    # Apply top-down attention to prefrontal cortex
    attention.apply_topdown_attention("prefrontal_cortex", strength=2.0)
    print("   Attention focused on prefrontal cortex")
    
    # 4. Store patterns in working memory
    print("\n4. Storing patterns in short-term memory...")
    patterns = []
    for i in range(5):
        # Create a pattern
        pattern = np.random.rand(20) * 2.0
        patterns.append(pattern)
        
        # Store in short-term memory
        consolidator.store_pattern(pattern, f"pattern_{i}")
        
        # Record for replay
        replay.record_pattern(pattern, f"pattern_{i}", importance=1.0 + i * 0.2)
    
    print(f"   Stored {len(patterns)} patterns")
    
    # 5. Run simulation with attention
    print("\n5. Running simulation with attention...")
    firing_rates = []
    membrane_potentials = []
    
    for step in range(20):
        # Apply attention modulation
        attention.apply_attention_modulation()
        
        # Run simulation step
        stats = sim.step()
        
        # Track neural activity for phase space analysis
        if len(model.neurons) > 0:
            neuron_list = list(model.neurons.values())
            avg_firing = len(stats["spikes"]) / len(model.neurons) if len(model.neurons) > 0 else 0
            avg_potential = np.mean([n.membrane_potential for n in neuron_list])
            
            firing_rates.append(avg_firing)
            membrane_potentials.append(avg_potential)
    
    print(f"   Ran 20 simulation steps")
    
    # 6. Consolidate memories
    print("\n6. Consolidating memories...")
    n_consolidated = 0
    for i in range(5):
        if consolidator.consolidate(f"pattern_{i}", strength_multiplier=1.5):
            n_consolidated += 1
    
    print(f"   Consolidated {n_consolidated} patterns to long-term memory")
    
    # 7. Enter sleep state and perform offline learning
    print("\n7. Entering sleep state...")
    sleep.enter_sleep(depth=0.8)
    
    sleep_stats = []
    for _ in range(10):
        stats = sleep.sleep_step()
        sleep_stats.append(stats)
    
    exit_stats = sleep.exit_sleep()
    
    total_replays = sum(s.get("replays", 0) for s in sleep_stats)
    total_consolidations = sum(s.get("consolidations", 0) for s in sleep_stats)
    
    print(f"   Sleep cycle complete:")
    print(f"   - Duration: {exit_stats['total_sleep_duration']} steps")
    print(f"   - Total replays: {total_replays}")
    print(f"   - Total consolidations: {total_consolidations}")
    
    # 8. Memory replay
    print("\n8. Performing memory replay...")
    n_replayed = replay.prioritized_replay(n_patterns=3, temperature=0.5)
    print(f"   Replayed {n_replayed} patterns")
    
    replay_stats = replay.get_replay_stats()
    print(f"   Total patterns stored: {replay_stats['num_patterns']}")
    print(f"   Total replays performed: {replay_stats['total_replays']}")
    
    # 9. Phase space analysis
    print("\n9. Performing phase space analysis...")
    if len(firing_rates) > 2:
        plot_data = plot_phase_space(
            np.array(firing_rates),
            np.array(membrane_potentials),
            labels=("Firing Rate", "Membrane Potential (mV)"),
            title="Network Dynamics Phase Space"
        )
        
        print(f"   Phase space trajectory computed:")
        print(f"   - Trajectory length: {plot_data['trajectory_length']:.2f}")
        print(f"   - Number of points: {plot_data['num_points']}")
    
    # 10. Network motif detection
    print("\n10. Detecting network motifs...")
    if len(model.synapses) > 0:
        detector = NetworkMotifDetector(model)
        
        # Sample 100 triads for efficiency
        motifs = detector.detect_triadic_motifs(sample_size=100)
        
        print(f"   Motifs detected:")
        for motif_type, count in sorted(motifs.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"   - {motif_type}: {count}")
        
        # Visualize motifs
        if sum(motifs.values()) > 0:
            plot_data = plot_network_motifs(
                motifs,
                total_triplets=100,
                title="Network Motif Distribution"
            )
            print(f"   Motif visualization data generated")
    
    # 11. Bottom-up saliency
    print("\n11. Computing bottom-up saliency...")
    saliency = attention.compute_saliency("prefrontal_cortex", use_temporal_change=False)
    
    if saliency:
        top_salient = sorted(saliency.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"   Top 3 salient neurons:")
        for neuron_id, score in top_salient:
            print(f"   - Neuron {neuron_id}: {score:.3f}")
    
    # 12. Winner-take-all
    print("\n12. Applying winner-take-all selection...")
    winners = attention.winner_take_all("prefrontal_cortex", top_k=3)
    print(f"   Selected {len(winners)} winner neurons: {winners}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("Example Complete!")
    print("=" * 60)
    print(f"\nFinal state:")
    print(f"  - Neurons: {len(model.neurons)}")
    print(f"  - Synapses: {len(model.synapses)}")
    print(f"  - Patterns stored: {len(consolidator.short_term_patterns)}")
    print(f"  - Consolidations performed: {len(consolidator.consolidation_history)}")
    print(f"  - Replay patterns: {replay.replay_count}")
    print("\nThis example demonstrated:")
    print("  ✓ Long-term memory consolidation")
    print("  ✓ Memory replay mechanisms")
    print("  ✓ Sleep-like states for offline learning")
    print("  ✓ Top-down and bottom-up attention")
    print("  ✓ Winner-take-all circuits")
    print("  ✓ Phase space analysis")
    print("  ✓ Network motif detection")


if __name__ == "__main__":
    main()
