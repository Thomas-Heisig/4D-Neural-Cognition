# Testing Documentation

## Overview

The 4D Neural Cognition project has a comprehensive test suite with 408 tests covering unit tests, integration tests, performance benchmarks, and advanced metrics validation.

## Test Structure

```
tests/
├── conftest.py                 # Shared fixtures and configuration
├── test_brain_model.py        # Brain model unit tests (41 tests)
├── test_simulation.py         # Simulation unit tests (28 tests)
├── test_cell_lifecycle.py     # Cell lifecycle unit tests (15 tests)
├── test_plasticity.py         # Plasticity unit tests (16 tests)
├── test_senses.py             # Sensory input unit tests (17 tests)
├── test_storage.py            # Storage unit tests (15 tests)
├── test_integration.py        # Integration tests (12 tests)
├── test_performance.py        # Performance benchmarks (16 tests)
└── test_metrics.py            # Advanced metrics tests (35 tests)
```

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test Files
```bash
pytest tests/test_brain_model.py
pytest tests/test_integration.py
pytest tests/test_performance.py
```

### Run with Coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

### Run with Verbose Output
```bash
pytest tests/ -v
```

### Run Performance Benchmarks Only
```bash
pytest tests/test_performance.py -v -s
```

## Test Categories

### Unit Tests (126 tests)

Unit tests validate individual components in isolation:

- **brain_model.py** (41 tests): Neuron/synapse creation, serialization, configuration
- **simulation.py** (28 tests): Simulation step logic, initialization
- **cell_lifecycle.py** (15 tests): Aging, death, reproduction, mutation
- **plasticity.py** (16 tests): Hebbian learning, weight decay, LTP/LTD
- **senses.py** (17 tests): Coordinate conversion, sensory input feeding
- **storage.py** (15 tests): JSON/HDF5 save/load, roundtrip consistency
- **metrics.py** (35 tests): Information theory, network stability, learning curves

### Integration Tests (12 tests)

Integration tests validate complete workflows:

- **Basic simulation runs**: End-to-end simulation with neurons and synapses
- **Sensory input processing**: Input feeding and neuron activation
- **Plasticity integration**: Weight changes during learning
- **Cell lifecycle**: Aging, death, and reproduction during simulation
- **Multi-area simulation**: Multiple brain areas working together
- **Save/load workflows**: Checkpoint and resume simulation
- **Error recovery**: Graceful handling of edge cases
- **Reproducibility**: Seeded simulations produce consistent results

### Performance Benchmarks (16 tests)

Performance tests measure and validate performance characteristics:

#### Neuron Operations
- Single neuron creation: < 1ms
- 100 neurons: < 100ms
- 1000 neurons: ~5ms

#### Synapse Operations
- Single synapse creation: < 1ms
- 100 synapses: < 100ms

#### Simulation Steps
- Small network (50 neurons): < 50ms per step
- Medium network (100+ neurons): < 200ms per step
- 100 steps: < 10s

#### Storage Operations
- JSON save: < 1s for medium network
- JSON load: < 1s
- HDF5 save: ~4ms (3x faster than JSON)
- HDF5 load: ~4ms

#### Scalability
- Performance scales roughly linearly with neuron count
- Synapse count has modest impact on step time

## Coverage

Current test coverage: **50%**

Coverage by module:
- brain_model.py: 91%
- cell_lifecycle.py: 100%
- metrics.py: 95%
- simulation.py: 98%
- storage.py: 94%
- plasticity.py: 64%
- senses.py: 73%

Untested modules (planned for future):
- evaluation.py: 0% (requires task implementation)
- knowledge_db.py: 0% (database testing)
- tasks.py: 0% (environment testing)

## Test Fixtures

Common fixtures are defined in `conftest.py`:

- **minimal_config**: Basic valid configuration
- **brain_model**: Initialized brain model
- **populated_model**: Model with neurons and synapses
- **simulation**: Simulation instance
- **sample_neuron**: Example neuron
- **sample_synapse**: Example synapse
- **rng**: Seeded random number generator
- **temp_dir**: Temporary directory for file operations

## Writing New Tests

### Unit Test Example

```python
def test_add_neuron(brain_model):
    """Test adding a neuron to the model."""
    neuron = brain_model.add_neuron(1, 2, 3, 0)
    
    assert neuron.x == 1
    assert neuron.y == 2
    assert neuron.z == 3
    assert neuron.w == 0
    assert len(brain_model.neurons) == 1
```

### Integration Test Example

```python
def test_simulation_with_input(full_config):
    """Test complete simulation with sensory input."""
    model = BrainModel(config=full_config)
    sim = Simulation(model, seed=42)
    
    # Setup
    sim.initialize_neurons(area_names=["V1_like"], density=0.2)
    sim.initialize_random_synapses(connection_probability=0.01)
    
    # Run with input
    input_pattern = np.random.rand(10, 10)
    for i in range(20):
        if i % 5 == 0:
            feed_sense_input(model, "vision", input_pattern)
        stats = sim.step()
    
    # Validate
    assert model.current_step == 20
    assert len(stats['spikes']) >= 0
```

### Performance Test Example

```python
def test_operation_performance(benchmark_config):
    """Measure operation performance."""
    model = BrainModel(config=benchmark_config)
    
    start_time = time.perf_counter()
    for i in range(1000):
        model.add_neuron(i % 20, i % 20, i % 20, 0)
    elapsed = time.perf_counter() - start_time
    
    assert elapsed < 1.0  # Should complete in < 1s
    print(f"Operation took {elapsed*1000:.2f}ms")
```

## Continuous Integration

Tests are run automatically on:
- Every pull request
- Every commit to main branch
- Nightly builds

CI runs:
- All unit tests
- All integration tests
- Performance benchmarks (informational)
- Code coverage reporting

## Best Practices

1. **Write tests first**: Follow TDD when adding new features
2. **Keep tests independent**: Each test should run standalone
3. **Use descriptive names**: Test names should explain what they test
4. **Test edge cases**: Include boundary conditions and error cases
5. **Mock external dependencies**: Use fixtures and mocks appropriately
6. **Keep tests fast**: Unit tests should run in milliseconds
7. **Document complex tests**: Add docstrings explaining test purpose
8. **Validate assumptions**: Assert preconditions and postconditions
9. **Clean up resources**: Use fixtures and context managers
10. **Check coverage**: Aim for >80% coverage on new code

## Troubleshooting

### Tests Fail with Import Errors
```bash
# Ensure src is in Python path
export PYTHONPATH="${PYTHONPATH}:./src"
pytest tests/
```

### Tests Fail with Missing Dependencies
```bash
# Install development dependencies
pip install -r requirements.txt
```

### Performance Tests Fail on Slow Hardware
```bash
# Skip performance tests
pytest tests/ -k "not performance"
```

### Coverage Report Not Generated
```bash
# Install coverage plugin
pip install pytest-cov

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Future Work

Planned test improvements:
- [ ] Increase coverage to >80%
- [ ] Add tests for evaluation.py
- [ ] Add tests for knowledge_db.py
- [ ] Add tests for tasks.py
- [ ] Add visual regression tests for UI
- [ ] Add property-based testing with Hypothesis
- [ ] Add mutation testing
- [ ] Set up continuous performance monitoring

## Related Documentation

- [CONTRIBUTING.md](../CONTRIBUTING.md): Contribution guidelines
- [API.md](api/API.md): API documentation
- [ARCHITECTURE.md](ARCHITECTURE.md): System architecture
