"""Demonstration of integrated learning systems.

This example showcases the integration of biological/psychological learning
systems and machine learning approaches within the 4D Neural Cognition framework.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.learning_systems import (
    LearningContext,
    AssociativeLearning,
    NonAssociativeLearning,
    OperantConditioning,
    SupervisedLearning,
    UnsupervisedLearning,
    ReinforcementLearning,
    TransferLearning,
    MetaLearning,
    LearningSystemManager,
    create_default_learning_systems,
)


def demonstrate_biological_learning():
    """Demonstrate biological/psychological learning systems."""
    print("=" * 80)
    print("BIOLOGICAL/PSYCHOLOGICAL LEARNING SYSTEMS")
    print("=" * 80)
    
    # Associative Learning
    print("\n1. Associative Learning - Pavlovian Conditioning")
    print("-" * 80)
    assoc = AssociativeLearning()
    context = LearningContext(timestep=1)
    
    # Classical conditioning: bell + food
    for i in range(5):
        data = {
            "stimulus_a": "bell",
            "stimulus_b": "food",
            "strength": 1.0
        }
        result = assoc.learn(context, data)
        print(f"  Trial {i+1}: {result.feedback}, strength={result.metrics['association_strength']:.3f}")
    
    # Non-Associative Learning
    print("\n2. Non-Associative Learning - Habituation")
    print("-" * 80)
    non_assoc = NonAssociativeLearning()
    
    print("  Repeated exposure to loud noise:")
    for i in range(5):
        data = {
            "stimulus": "loud_noise",
            "type": "habituation"
        }
        result = non_assoc.learn(context, data)
        print(f"  Exposure {i+1}: response={result.metrics['response_strength']:.3f}")
    
    # Operant Conditioning
    print("\n3. Operant Conditioning - Reward/Punishment")
    print("-" * 80)
    operant = OperantConditioning()
    
    # Reinforce lever pressing
    print("  Learning to press lever (positive reinforcement):")
    for i in range(5):
        data = {
            "behavior": "press_lever",
            "reward": 1.0 if np.random.random() > 0.3 else 0.0
        }
        result = operant.learn(context, data)
        print(f"  Trial {i+1}: value={result.metrics['value']:.3f}")
    
    print("\n" + "=" * 80)


def demonstrate_machine_learning():
    """Demonstrate machine learning systems."""
    print("\n" + "=" * 80)
    print("MACHINE LEARNING SYSTEMS")
    print("=" * 80)
    
    # Supervised Learning
    print("\n1. Supervised Learning - Classification")
    print("-" * 80)
    supervised = SupervisedLearning()
    context = LearningContext()
    
    print("  Training on labeled examples:")
    for i in range(5):
        data = {
            "input": [np.random.random() for _ in range(3)],
            "label": i % 2,
            "error": 0.5 * (1 - i/10)  # Decreasing error
        }
        result = supervised.learn(context, data)
        print(f"  Sample {i+1}: samples={result.metrics['samples']}, error={result.metrics['error']:.3f}")
    
    # Unsupervised Learning
    print("\n2. Unsupervised Learning - Clustering")
    print("-" * 80)
    unsupervised = UnsupervisedLearning()
    
    print("  Discovering patterns in unlabeled data:")
    for i in range(5):
        data = {
            "input": [np.random.random() * 10 for _ in range(3)]
        }
        result = unsupervised.learn(context, data)
        print(f"  Pattern {i+1}: {result.feedback}, total={result.metrics['total_patterns']}")
    
    # Reinforcement Learning
    print("\n3. Reinforcement Learning - Q-Learning")
    print("-" * 80)
    rl = ReinforcementLearning()
    
    print("  Agent learning optimal policy:")
    states = ["start", "middle", "goal"]
    actions = ["left", "right", "stay"]
    
    for i in range(10):
        state = np.random.choice(states)
        action = np.random.choice(actions)
        reward = 1.0 if state == "goal" else -0.1
        
        data = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": np.random.choice(states)
        }
        result = rl.learn(context, data)
        if i % 3 == 0:
            print(f"  Update {i+1}: Q({state},{action})={result.metrics['q_value']:.3f}")
    
    # Transfer Learning
    print("\n4. Transfer Learning - Domain Adaptation")
    print("-" * 80)
    transfer = TransferLearning()
    
    transfers = [
        ("images", "sketches", 0.8),
        ("english", "spanish", 0.6),
        ("chess", "go", 0.4),
    ]
    
    print("  Transferring knowledge across domains:")
    for source, target, similarity in transfers:
        data = {
            "source_domain": source,
            "target_domain": target,
            "domain_similarity": similarity
        }
        result = transfer.learn(context, data)
        status = "✓" if result.success else "✗"
        print(f"  {status} {source} → {target}: similarity={similarity:.2f}")
    
    # Meta-Learning
    print("\n5. Meta-Learning - Learning to Learn")
    print("-" * 80)
    meta = MetaLearning()
    
    strategies = [
        ("gradient_descent", 0.7),
        ("adam", 0.85),
        ("sgd", 0.65),
        ("adam", 0.88),  # Test same strategy again
        ("adam", 0.90),
    ]
    
    print("  Testing learning strategies:")
    for strategy, performance in strategies:
        data = {
            "strategy": strategy,
            "performance": performance
        }
        result = meta.learn(context, data)
        print(f"  {strategy}: performance={performance:.2f}, "
              f"avg={result.metrics['avg_performance']:.2f}")
    
    print("\n" + "=" * 80)


def demonstrate_integrated_learning():
    """Demonstrate integrated learning with multiple systems."""
    print("\n" + "=" * 80)
    print("INTEGRATED LEARNING SYSTEMS")
    print("=" * 80)
    
    # Create manager with all systems
    manager = create_default_learning_systems()
    
    print(f"\nRegistered {len(manager.systems)} learning systems:")
    bio_systems = manager.get_biological_systems()
    ml_systems = manager.get_machine_systems()
    print(f"  - {len(bio_systems)} biological/psychological systems")
    print(f"  - {len(ml_systems)} machine learning systems")
    
    # Activate selected systems
    print("\nActivating systems:")
    systems_to_activate = [
        "Associative Learning",
        "Operant Conditioning",
        "Supervised Learning",
        "Reinforcement Learning"
    ]
    
    for system_name in systems_to_activate:
        manager.activate_system(system_name)
        print(f"  ✓ {system_name}")
    
    # Perform integrated learning
    print("\nPerforming integrated learning across multiple systems:")
    context = LearningContext(timestep=1)
    
    # Scenario: Learning to respond to a signal
    for trial in range(3):
        print(f"\n  Trial {trial + 1}:")
        
        data = {
            "Associative Learning": {
                "stimulus_a": "signal",
                "stimulus_b": "action",
                "strength": 1.0
            },
            "Operant Conditioning": {
                "behavior": "respond_to_signal",
                "reward": 1.0 if trial > 0 else 0.5
            },
            "Supervised Learning": {
                "input": [trial, trial * 0.5, trial * 0.25],
                "label": 1,
                "error": 0.3 / (trial + 1)
            },
            "Reinforcement Learning": {
                "state": f"state_{trial}",
                "action": "respond",
                "reward": 1.0,
                "next_state": f"state_{trial + 1}"
            }
        }
        
        results = manager.learn(context, data)
        
        for system_name, result in results.items():
            status = "✓" if result.success else "✗"
            print(f"    {status} {system_name}: delta={result.learning_delta:.3f}")
    
    # Show metrics
    print("\n" + "-" * 80)
    print("Final Learning Metrics:")
    print("-" * 80)
    all_metrics = manager.get_all_metrics()
    
    for system_name in systems_to_activate:
        if system_name in all_metrics:
            metrics = all_metrics[system_name]
            if metrics:
                print(f"\n  {system_name}:")
                for metric_name, value in metrics.items():
                    print(f"    - {metric_name}: {value:.3f}")
    
    print("\n" + "=" * 80)


def main():
    """Main demonstration function."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "4D NEURAL COGNITION - LEARNING SYSTEMS DEMO" + " " * 20 + "║")
    print("╚" + "═" * 78 + "╝")
    
    print("\nThis demonstration showcases the integration of biological/psychological")
    print("learning systems and machine learning approaches.")
    print("\nThe fundamental difference:")
    print("  • Biological learning: consciousness-capable, flexible, context-dependent")
    print("  • Machine learning: statistical pattern recognition based on algorithms")
    
    # Run demonstrations
    demonstrate_biological_learning()
    demonstrate_machine_learning()
    demonstrate_integrated_learning()
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nThe learning systems are now integrated into the 4D Neural Cognition framework.")
    print("They can be used independently or in combination for complex learning tasks.")
    print()


if __name__ == "__main__":
    main()
