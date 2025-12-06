#!/usr/bin/env python3
"""Example script demonstrating the benchmark and evaluation system.

This script shows how to:
1. Create benchmark configurations
2. Run benchmark suites
3. Compare configurations
4. Use the knowledge database for pre-training
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from evaluation import (
    BenchmarkConfig,
    BenchmarkSuite,
    run_configuration_comparison,
    create_standard_benchmark_suite
)
from tasks import PatternClassificationTask, TemporalSequenceTask
from knowledge_db import KnowledgeDatabase, populate_sample_knowledge, KnowledgeBasedTrainer
from brain_model import BrainModel
from simulation import Simulation


def example_single_benchmark():
    """Example: Run a single benchmark configuration."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Benchmark Run")
    print("="*80 + "\n")
    
    # Create configuration
    config = BenchmarkConfig(
        name="baseline",
        description="Baseline configuration with standard parameters",
        config_path="../brain_base_model.json",
        seed=42,
        initialization_params={
            'area_names': ['V1_like', 'Digital_sensor'],
            'density': 0.1,
            'connection_probability': 0.01,
            'weight_mean': 0.1,
            'weight_std': 0.05
        }
    )
    
    # Create benchmark suite
    suite = create_standard_benchmark_suite()
    
    # Run benchmark
    results = suite.run(config, output_dir=Path("../benchmark_results"))
    
    print(f"\nCompleted {len(results)} tasks")
    for result in results:
        print(f"\n{result.task_name}:")
        print(f"  Accuracy: {result.task_result.accuracy:.4f}")
        print(f"  Reward: {result.task_result.reward:.4f}")


def example_configuration_comparison():
    """Example: Compare multiple configurations."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Configuration Comparison")
    print("="*80 + "\n")
    
    # Create multiple configurations to compare
    configs = [
        BenchmarkConfig(
            name="baseline",
            description="Baseline with 0.1 density",
            config_path="../brain_base_model.json",
            seed=42,
            initialization_params={
                'area_names': ['V1_like', 'Digital_sensor'],
                'density': 0.1,
                'connection_probability': 0.01,
                'weight_mean': 0.1,
                'weight_std': 0.05
            }
        ),
        BenchmarkConfig(
            name="dense_network",
            description="Denser network with 0.2 density",
            config_path="../brain_base_model.json",
            seed=42,
            initialization_params={
                'area_names': ['V1_like', 'Digital_sensor'],
                'density': 0.2,
                'connection_probability': 0.02,
                'weight_mean': 0.1,
                'weight_std': 0.05
            }
        ),
        BenchmarkConfig(
            name="stronger_weights",
            description="Stronger initial weights",
            config_path="../brain_base_model.json",
            seed=42,
            initialization_params={
                'area_names': ['V1_like', 'Digital_sensor'],
                'density': 0.1,
                'connection_probability': 0.01,
                'weight_mean': 0.2,
                'weight_std': 0.1
            }
        ),
    ]
    
    # Run comparison
    report = run_configuration_comparison(
        configs=configs,
        output_dir=Path("../benchmark_results")
    )
    
    print("\nComparison completed!")


def example_knowledge_database():
    """Example: Using knowledge database for pre-training."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Knowledge Database Pre-training")
    print("="*80 + "\n")
    
    db_path = "../knowledge.db"
    
    # Populate database with sample data if it doesn't exist
    if not Path(db_path).exists():
        print("Creating and populating knowledge database...")
        populate_sample_knowledge(db_path)
    else:
        print(f"Using existing database: {db_path}")
    
    # Open database
    db = KnowledgeDatabase(db_path)
    
    # Show database contents
    print(f"\nDatabase statistics:")
    print(f"  Total entries: {db.count()}")
    print(f"  Pattern recognition: {db.count(category='pattern_recognition')}")
    print(f"  Sequence learning: {db.count(category='sequence_learning')}")
    
    # Create a simulation
    print("\nInitializing neural network...")
    model = BrainModel(config_path='../brain_base_model.json')
    sim = Simulation(model, seed=42)
    sim.initialize_neurons(area_names=['V1_like', 'Digital_sensor'], density=0.1)
    sim.initialize_random_synapses(connection_probability=0.01)
    
    # Create trainer
    trainer = KnowledgeBasedTrainer(sim, db)
    
    # Pre-train on pattern recognition
    print("\nPre-training on pattern recognition...")
    stats = trainer.pretrain(
        category='pattern_recognition',
        num_samples=20,
        steps_per_sample=30
    )
    
    print(f"\nPre-training statistics:")
    print(f"  Samples processed: {stats['samples_processed']}")
    print(f"  Total steps: {stats['total_steps']}")
    print(f"  Average activity: {stats['avg_activity']:.2f} spikes/step")
    
    db.close()


def example_custom_task():
    """Example: Creating and running a custom task."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Custom Task Creation")
    print("="*80 + "\n")
    
    # Create custom benchmark suite
    suite = BenchmarkSuite(
        name="Custom Vision Suite",
        description="Custom suite focusing on vision tasks"
    )
    
    # Add multiple pattern classification tasks with different parameters
    suite.add_task(PatternClassificationTask(
        num_classes=2,
        pattern_size=(20, 20),
        noise_level=0.05,
        seed=42
    ))
    
    suite.add_task(PatternClassificationTask(
        num_classes=4,
        pattern_size=(20, 20),
        noise_level=0.1,
        seed=42
    ))
    
    suite.add_task(PatternClassificationTask(
        num_classes=4,
        pattern_size=(20, 20),
        noise_level=0.2,
        seed=42
    ))
    
    # Create configuration
    config = BenchmarkConfig(
        name="vision_optimized",
        description="Configuration optimized for vision tasks",
        config_path="../brain_base_model.json",
        seed=42,
        initialization_params={
            'area_names': ['V1_like'],  # Only vision area
            'density': 0.15,
            'connection_probability': 0.015,
            'weight_mean': 0.12,
            'weight_std': 0.06
        }
    )
    
    # Run custom suite
    results = suite.run(config, output_dir=Path("../benchmark_results"))
    
    print(f"\nCompleted custom suite with {len(results)} tasks")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("4D NEURAL COGNITION - BENCHMARK & EVALUATION EXAMPLES")
    print("="*80)
    
    # Create output directory
    Path("../benchmark_results").mkdir(exist_ok=True)
    
    try:
        # Run examples
        example_single_benchmark()
        input("\nPress Enter to continue to configuration comparison...")
        
        example_configuration_comparison()
        input("\nPress Enter to continue to knowledge database example...")
        
        example_knowledge_database()
        input("\nPress Enter to continue to custom task example...")
        
        example_custom_task()
        
        print("\n" + "="*80)
        print("All examples completed successfully!")
        print("="*80 + "\n")
        
        print("Results saved to: ../benchmark_results/")
        print("Knowledge database: ../knowledge.db")
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
