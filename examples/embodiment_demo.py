#!/usr/bin/env python3
"""Demo: Embodied AI with Sensorimotor Learning and Self-Awareness.

This script demonstrates the complete embodied learning system:
1. Virtual body with proprioception
2. Sensorimotor reinforcement learning
3. Self-perception stream with anomaly detection
4. Multimodal self-recognition
5. Cognitive-aware VNC orchestration

Run with:
    python examples/embodiment_demo.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from brain_model import BrainModel
from simulation import Simulation
from embodiment.virtual_body import VirtualBody
from embodiment.sensorimotor_learner import SensorimotorReinforcementLearner
from consciousness.self_perception_stream import SelfPerceptionStream
from multimodal_integration import MultimodalIntegrationSystem
from hardware_abstraction.adaptive_vnc_orchestrator import CognitiveAwareOrchestrator

print("="*70)
print("🤖 4D Neural Cognition - Embodied AI Demo")
print("="*70)

# 1. Create brain model
print("\n[1/6] Creating brain model...")
config = {
    "lattice_shape": [10, 10, 5, 15],
    "neuron_model": {
        "type": "lif",
        "params_default": {
            "threshold": -50.0,
            "reset_potential": -65.0,
            "tau_membrane": 20.0,
            "refractory_period": 2,
        }
    },
    "cell_lifecycle": {"neurogenesis_rate": 0.0, "apoptosis_threshold": 0.0},
    "plasticity": {"stdp_enabled": True, "learning_rate": 0.01},
    "senses": {"digital": {"areal": "V1"}},
    "areas": [
        {"name": "M1", "bounds": {"x": [0, 10], "y": [0, 10], "z": [0, 5], "w": [10, 10]}, "neuron_type": "excitatory"},
        {"name": "S1", "bounds": {"x": [0, 10], "y": [0, 10], "z": [0, 5], "w": [6, 6]}, "neuron_type": "excitatory"},
    ]
}
brain = BrainModel(config=config)

# Add some neurons
for x in range(0, 10, 2):
    for y in range(0, 10, 2):
        for w in [6, 10]:  # Sensory and Motor
            brain.add_neuron(x, y, 0, w)

print(f"✓ Created brain with {len(brain.neurons)} neurons")

# 2. Create virtual body
print("\n[2/6] Creating virtual body...")
body = VirtualBody(body_type="humanoid", num_joints=8)
print(f"✓ Created humanoid body with {len(body.muscles)} muscles")

# 3. Create sensorimotor learner
print("\n[3/6] Initializing sensorimotor learner...")
learner = SensorimotorReinforcementLearner(
    virtual_body=body,
    brain_model=brain,
    learning_rate=0.01,
    dopamine_modulation_strength=0.1
)
print("✓ Learner initialized with STDP and intrinsic motivation")

# 4. Create self-perception stream
print("\n[4/6] Setting up self-perception stream...")
perception = SelfPerceptionStream(
    update_frequency_hz=100.0,
    buffer_duration_seconds=5.0
)
print("✓ Self-perception stream active (100 Hz)")

# 5. Create multimodal integration system
print("\n[5/6] Initializing multimodal integration...")
multimodal = MultimodalIntegrationSystem()
print("✓ Multimodal system ready (vision, audio, touch)")

# 6. Create simulation with cognitive VNC
print("\n[6/6] Setting up cognitive VNC orchestrator...")
sim = Simulation(brain)
orchestrator = CognitiveAwareOrchestrator(
    simulation=sim,
    monitoring_interval=10
)
print("✓ Cognitive-aware VNC orchestrator online")

print("\n" + "="*70)
print("🎮 Running Embodied Learning Demo (20 steps)")
print("="*70)

# Run demo
learner.start_episode()

for step in range(20):
    print(f"\n--- Step {step + 1}/20 ---")
    
    # 1. Generate motor command (random exploration)
    motor_command = {
        'planned': {f'joint_{i}': np.random.rand() * 0.3 for i in range(4)},
        'executed': {f'joint_{i}': np.random.rand() * 0.3 for i in range(4)},
    }
    
    # 2. Execute motor command on body
    feedback = body.execute_motor_command({
        'motor_neurons': {i: np.random.rand() * 0.5 for i in range(4)}
    })
    print(f"  Body position: {feedback['position'][:2]} (x, y)")
    print(f"  Active joints: {len(feedback['joint_angles'])}")
    
    # 3. Learn from interaction
    reward = np.random.rand() * 0.5  # Simulated task reward
    learning_result = learner.learn_from_interaction(
        action=motor_command,
        resulting_feedback=feedback,
        external_reward=reward
    )
    print(f"  Learning: reward={learning_result['total_reward']:.3f}, " +
          f"error={learning_result['prediction_error']:.3f}")
    
    # 4. Update self-perception
    perception.update(
        sensor_data={'proprioception': feedback},
        motor_commands=motor_command,
        internal_state={'metabolic': {}, 'attention': {}, 'emotion': {}}
    )
    
    # 5. Detect anomalies
    anomalies = perception.detect_self_consistency_anomalies()
    if anomalies:
        print(f"  ⚠️  Anomaly detected: {anomalies[0]['type']}")
        recalibration = perception.update_self_model_based_on_anomalies(anomalies)
        print(f"  🔧 Recalibrated: {len(recalibration['recalibrations'])} adjustments")
    
    # 6. Multimodal self-recognition
    fusion_result = multimodal.fuse_modalities_for_self_recognition(feedback)
    print(f"  Self-confidence: {fusion_result['self_confidence']:.2%}")
    
    # 7. VNC orchestration (every 10 steps)
    if step % 10 == 0:
        vnc_result = orchestrator.monitor_and_adapt(step)
        if vnc_result.get('monitored'):
            print(f"  VNC: load_imbalance={vnc_result.get('load_imbalance', 0):.2%}")
    
    # Simulate some computation
    sim.step()

# End episode
episode_summary = learner.end_episode()

print("\n" + "="*70)
print("📊 Episode Summary")
print("="*70)
print(f"Episodes completed: {episode_summary['episodes']}")
print(f"Average reward: {episode_summary['avg_reward']:.3f}")
print(f"Reward trend: {episode_summary['reward_trend']:.3f}")
print(f"Prediction error: {episode_summary['avg_prediction_error']:.3f}")
print(f"Error reduction: {episode_summary['error_reduction']:.3f}")

# Self-awareness metrics
awareness_metrics = perception.get_self_awareness_metric()
print(f"\nSelf-Awareness:")
print(f"  Consistency: {awareness_metrics['self_consistency']:.2%}")
print(f"  Integration: {awareness_metrics['integration']:.2%}")
print(f"  Agency: {awareness_metrics['agency_score']:.2%}")

# Multimodal stats
mm_stats = multimodal.get_statistics()
print(f"\nMultimodal Integration:")
print(f"  Fusion count: {mm_stats['fusion_count']}")
print(f"  Current confidence: {mm_stats['current_confidence']:.2%}")
print(f"  Modality weights: V={mm_stats['modality_weights']['vision']:.2f}, " +
      f"A={mm_stats['modality_weights']['audio']:.2f}, " +
      f"P={mm_stats['modality_weights']['proprio']:.2f}")

# VNC stats
vnc_summary = orchestrator.get_cognitive_performance_summary()
print(f"\nCognitive VNC:")
print(f"  Total repartitions: {vnc_summary['total_repartitions']}")
print(f"  Priority adjustments: {vnc_summary['total_priority_adjustments']}")
print(f"  Monitoring cycles: {vnc_summary['monitoring_cycles']}")

print("\n" + "="*70)
print("✅ Demo completed successfully!")
print("="*70)
print("\nNext steps:")
print("  1. Run experiments: python -m experiments.sensorimotor_learning")
print("  2. Test anomaly detection: python -m experiments.self_anomaly_detection")
print("  3. Analyze VNC: python -m experiments.cognitive_vnc")
print("  4. Open dashboard: python app.py and visit http://localhost:5000/dashboard")
print("="*70)
