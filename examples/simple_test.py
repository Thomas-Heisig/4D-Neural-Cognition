#!/usr/bin/env python3
"""Simple test to verify the Tasks & Evaluation framework works with actual simulations."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from brain_model import BrainModel
from simulation import Simulation
from tasks import PatternClassificationTask
from evaluation import BenchmarkConfig, BenchmarkSuite
from knowledge_db import populate_sample_knowledge, KnowledgeDatabase, KnowledgeBasedTrainer


def test_simple_benchmark():
    """Test a simple benchmark run."""
    print("\n" + "="*70)
    print("TEST: Simple Benchmark Run")
    print("="*70 + "\n")
    
    # Create configuration
    config = BenchmarkConfig(
        name="test_baseline",
        description="Simple test configuration",
        config_path="../brain_base_model.json",
        seed=42,
        initialization_params={
            'area_names': ['V1_like'],
            'density': 0.05,  # Small for speed
            'connection_probability': 0.01,
            'weight_mean': 0.1,
            'weight_std': 0.05
        }
    )
    
    # Create simple suite
    suite = BenchmarkSuite(name="Simple Test Suite")
    suite.add_task(PatternClassificationTask(
        num_classes=2,  # Simple binary classification
        pattern_size=(20, 20),
        noise_level=0.1,
        seed=42
    ))
    
    # Run benchmark
    print("Running benchmark (this may take a minute)...")
    results = suite.run(config)
    
    # Check results
    assert len(results) == 1, "Should have 1 result"
    result = results[0]
    
    print(f"\n✓ Test completed successfully!")
    print(f"  Task: {result.task_name}")
    print(f"  Accuracy: {result.task_result.accuracy:.4f}")
    print(f"  Execution time: {result.execution_time:.2f}s")
    
    return True


def test_knowledge_database():
    """Test knowledge database functionality."""
    print("\n" + "="*70)
    print("TEST: Knowledge Database")
    print("="*70 + "\n")
    
    db_path = "/tmp/test_knowledge_simple.db"
    
    # Create and populate database
    print("Populating database...")
    populate_sample_knowledge(db_path)
    
    # Open and verify
    db = KnowledgeDatabase(db_path)
    
    total = db.count()
    patterns = db.count(category='pattern_recognition')
    sequences = db.count(category='sequence_learning')
    
    print(f"\n✓ Database created successfully!")
    print(f"  Total entries: {total}")
    print(f"  Pattern recognition: {patterns}")
    print(f"  Sequence learning: {sequences}")
    
    assert total > 0, "Database should have entries"
    assert patterns > 0, "Should have pattern entries"
    assert sequences > 0, "Should have sequence entries"
    
    db.close()
    return True


def test_pretraining():
    """Test pre-training from knowledge database."""
    print("\n" + "="*70)
    print("TEST: Pre-training from Knowledge Database")
    print("="*70 + "\n")
    
    db_path = "/tmp/test_knowledge_simple.db"
    
    # Create simulation
    print("Creating neural network...")
    model = BrainModel(config_path='../brain_base_model.json')
    sim = Simulation(model, seed=42)
    sim.initialize_neurons(area_names=['V1_like'], density=0.05)
    sim.initialize_random_synapses(connection_probability=0.01)
    
    # Pre-train
    db = KnowledgeDatabase(db_path)
    trainer = KnowledgeBasedTrainer(sim, db)
    
    print("Pre-training (this may take a minute)...")
    stats = trainer.pretrain(
        category='pattern_recognition',
        num_samples=10,  # Small number for speed
        steps_per_sample=20
    )
    
    print(f"\n✓ Pre-training completed!")
    print(f"  Samples processed: {stats['samples_processed']}")
    print(f"  Total steps: {stats['total_steps']}")
    print(f"  Average activity: {stats['avg_activity']:.2f} spikes/step")
    
    assert stats['samples_processed'] == 10, "Should process 10 samples"
    assert stats['total_steps'] == 200, "Should have 200 total steps"
    
    db.close()
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("4D NEURAL COGNITION - SIMPLE INTEGRATION TESTS")
    print("="*70)
    
    tests = [
        ("Knowledge Database", test_knowledge_database),
        ("Pre-training", test_pretraining),
        ("Simple Benchmark", test_simple_benchmark),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*70}")
            print(f"Running: {test_name}")
            print(f"{'='*70}")
            
            if test_func():
                passed += 1
                print(f"\n✓ {test_name} PASSED")
            else:
                failed += 1
                print(f"\n✗ {test_name} FAILED")
                
        except Exception as e:
            failed += 1
            print(f"\n✗ {test_name} FAILED with error:")
            print(f"  {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n✗ {failed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
