# Autonomous Learning Loop Guide

## Overview

The autonomous learning loop represents the final integration piece that transforms the 4D Neural Cognition system from a reactive to a truly autonomous intelligent agent. This system closes the loop between perception, action, and adaptive learning, enabling the agent to:

1. **Set intrinsic goals** based on curiosity and competence
2. **Plan actions** using an internal world model
3. **Execute behaviors** through embodied interaction
4. **Learn from prediction errors** to improve both behavior and self-model
5. **Adapt learning strategies** based on success/failure patterns

This is the critical transition from **reactive intelligence** to **autonomous intelligence** - a system that doesn't just respond, but actively constructs its own learning trajectory.

---

## Architecture

### Core Components

#### 1. Intrinsic Motivation Engine (`IntrinsicMotivationEngine`)

**Purpose**: Generates autonomous goals without external instruction.

**Goal Types**:
- **Curiosity** (`REDUCE_PREDICTION_ERROR`): Drive to understand sensorimotor contingencies
- **Exploration** (`MAXIMIZE_NOVEL_SENSATIONS`): Seek new experiences
- **Competence** (`MASTER_MOTOR_SKILL`): Practice and refine motor skills
- **Homeostasis** (`MAINTAIN_BODY_INTEGRITY`): Maintain body health and avoid damage

**Key Features**:
- Balances multiple motivational drives based on configurable weights
- Tracks state visitation for novelty detection
- Monitors prediction errors and learning progress
- Dynamically generates context-appropriate goals

**Example Usage**:
```python
from src.autonomous_learning_loop import IntrinsicMotivationEngine

# Create engine with custom weights
engine = IntrinsicMotivationEngine(
    curiosity_weight=0.3,
    exploration_weight=0.3,
    competence_weight=0.3,
    homeostasis_weight=0.1,
)

# Generate a goal
current_state = {'position': np.array([1, 1, 1])}
prediction_error = 0.5
body_health = 0.9

goal = engine.generate_goal(current_state, prediction_error, body_health)
print(f"New goal: {goal['description']}")

# Check if goal achieved
metrics = {'body_health': 0.95, 'prediction_error': 0.2}
if engine.goal_achieved(goal, metrics):
    print("Goal achieved!")
```

#### 2. Predictive World Model (`PredictiveWorldModel`)

**Purpose**: Internal model for simulating "what-if" scenarios before action execution.

**Capabilities**:
- Learns state transition dynamics from experience
- Simulates action sequences mentally
- Provides prediction errors for learning
- Continuously improves through online learning

**Key Features**:
- Simple linear transition model (extensible to more complex models)
- Online learning from prediction errors
- Accuracy tracking over time
- Uncertainty estimation through noise injection

**Example Usage**:
```python
from src.autonomous_learning_loop import PredictiveWorldModel

# Create world model
model = PredictiveWorldModel(state_dim=10, action_dim=6)

# Simulate action sequence
initial_state = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
actions = [
    np.array([0.1, 0, 0, 0, 0, 0]),
    np.array([0, 0.1, 0, 0, 0, 0]),
]

predicted_states = model.simulate(initial_state, actions)
print(f"Predicted outcome: {predicted_states[-1]}")

# Learn from experience
actual_next_state = np.array([1.05, 0.1, 0, 0, 0, 0, 0, 0, 0, 0])
error = model.update_from_experience(
    initial_state, 
    actions[0], 
    actual_next_state,
    learning_rate=0.01
)
print(f"Prediction error: {error:.3f}")
print(f"Model accuracy: {model.get_accuracy():.3f}")
```

#### 3. Meta-Learning Controller (`MetaLearningController`)

**Purpose**: Adapts the learning strategy itself based on success/failure patterns.

**Learning Strategies**:
- **Explore**: Random exploration (default starting strategy)
- **Exploit**: Use learned policies
- **Imitate**: Learn from demonstrations (when failing repeatedly)
- **Curious**: Seek novelty
- **Consolidate**: Practice and refine mastered skills

**Key Features**:
- Monitors recent performance trends
- Switches strategies based on success rates and reward trends
- Prevents too-frequent strategy changes
- Tracks strategy performance history

**Strategy Transition Rules**:
- Explore → Exploit: When making progress (reward trend > 0, success > 30%)
- Exploit → Explore: When performance plateaus (reward trend < -0.1, success < 20%)
- Exploit → Consolidate: When high performance (success > 70%)
- Consolidate → Curious: When mastered (success > 90%)
- Any → Imitate: After repeated failures (> 15 consecutive failures)

**Example Usage**:
```python
from src.autonomous_learning_loop import MetaLearningController, LearningStrategy

# Create controller
controller = MetaLearningController()

# Adapt strategy based on performance
learning_history = [
    {'reward': 0.3, 'success': False},
    {'reward': 0.5, 'success': True},
    {'reward': 0.7, 'success': True},
    # ... more history
]

current_performance = {'reward': 0.8, 'success': True}

strategy = controller.adapt_strategy(learning_history, current_performance)
print(f"Current strategy: {strategy.value}")
```

#### 4. Autonomous Learning Agent (`AutonomousLearningAgent`)

**Purpose**: Integrates all components into a complete autonomous learning system.

**The Autonomous Cycle**:
1. **Goal Setting**: Generate or check current goal
2. **Planning**: Simulate action candidates in world model
3. **Execution**: Execute best action through embodiment
4. **Learning**: Update from prediction errors
5. **Self-Model Adaptation**: Adjust body model when errors are high
6. **Meta-Learning**: Adapt learning strategy based on progress

**Example Usage**:
```python
from src.autonomous_learning_loop import AutonomousLearningAgent
from src.embodiment.virtual_body import VirtualBody
from src.embodiment.sensorimotor_learner import SensorimotorReinforcementLearner
from src.consciousness.self_perception_stream import SelfPerceptionStream
from src.brain_model import BrainModel

# Create components
brain = BrainModel(config=brain_config)
body = VirtualBody(body_type="humanoid", num_joints=6)
self_stream = SelfPerceptionStream()
learner = SensorimotorReinforcementLearner(
    virtual_body=body,
    brain_model=brain,
    learning_rate=0.01,
)

# Create autonomous agent
agent = AutonomousLearningAgent(
    embodiment=body,
    brain=brain,
    self_stream=self_stream,
    learner=learner,
    state_dim=10,
    action_dim=6,
)

# Run autonomous cycles
for cycle in range(1000):
    environment_context = {
        'position': current_position,
        'velocity': current_velocity,
        'body_health': 1.0,
        'timestamp': cycle,
    }
    
    result = agent.run_autonomous_cycle(environment_context)
    
    # Result contains:
    # - goal: Current goal being pursued
    # - strategy: Current learning strategy
    # - prediction_error: Latest prediction error
    # - world_model_accuracy: Model accuracy
    # - learning_result: Detailed learning metrics

# Get statistics
stats = agent.get_statistics()
print(f"Total cycles: {stats['total_cycles']}")
print(f"Goals pursued: {stats['goal_history_length']}")
print(f"Strategy changes: {stats['strategy_changes']}")
print(f"World model accuracy: {stats['world_model_accuracy']:.3f}")
```

---

## Experiments

### 1. Autonomous Exploration

**Purpose**: Test the agent's ability to explore an environment without instructions.

**Success Criterion**: Discover 80% of objects in 1000 cycles.

**Run Experiment**:
```bash
# Basic run
python -m experiments.autonomous_exploration \
  --environment="simple_room" \
  --duration=1000 \
  --motivation="curiosity_and_competence" \
  --output="results/autonomy_metrics.json"

# Shorter test run
python -m experiments.autonomous_exploration \
  --duration=100 \
  --output="/tmp/quick_test.json"
```

**Key Metrics**:
- **Discovery rate**: Percentage of objects discovered
- **Goal distribution**: Types of goals pursued
- **Strategy evolution**: How learning strategy changed over time
- **World model accuracy**: Prediction accuracy improvement

**Example Output**:
```
============================================================
AUTONOMOUS EXPLORATION SUMMARY
============================================================
Total cycles: 1000
Elapsed time: 0.7s (1349.6 cycles/s)
Objects discovered: 8/10 (80.0%)
Goals pursued: 12
Strategy changes: 3
Final world model accuracy: 0.756
Current strategy: consolidate
============================================================
SUCCESS CRITERION: ✓ PASSED (target: 80% discovery rate)
```

### 2. Learning Trajectory Analysis

**Purpose**: Analyze learning patterns, phase transitions, and developmental stages.

**Run Analysis**:
```bash
# Analyze experiment results
python -m analysis.learning_trajectory \
  --input="results/autonomy_metrics.json" \
  --output="results/trajectory_analysis.json"

# With plotting
python -m analysis.learning_trajectory \
  --input="results/autonomy_metrics.json" \
  --plot="phase_transitions" \
  --output="results/trajectory_analysis.json"
```

**Analysis Features**:
- **Phase Transitions**: Detects significant changes in behavior (e.g., random → goal-directed)
- **Strategy Evolution**: Tracks learning strategy switches and durations
- **Learning Progress**: Measures prediction error reduction and skill acquisition
- **Developmental Stages**: Identifies emergent stages similar to infant development

**Example Output**:
```
============================================================
LEARNING TRAJECTORY ANALYSIS
============================================================
Phase transitions detected: 5
Strategy changes: 3
Dominant strategy: exploit

Prediction error reduction: 0.423
Total discoveries: 8

Developmental stages (5):
  Cycles 0-200: Random Exploration
  Cycles 200-400: Active Discovery
  Cycles 400-600: Skill Refinement
  Cycles 600-800: Mastery & Consolidation
  Cycles 800-1000: Transitional
============================================================
```

---

## Scientific Context

### From Reactive to Autonomous Intelligence

**Traditional AI Systems** (Reactive):
- Wait for external input/commands
- Execute pre-programmed responses
- Learn from external supervision

**Autonomous Learning Agent**:
- Generates own goals from intrinsic motivation
- Actively explores and tests hypotheses
- Learns from self-generated experience
- Adapts both behavior AND learning strategy

This represents a fundamental shift in AI architecture: from systems that respond to commands to systems that construct their own learning trajectories.

### Biological Inspiration

The autonomous learning loop is inspired by infant sensorimotor development:

1. **Reflexive Stage** (0-1 months): Random movements, basic reflexes
2. **Primary Circular Reactions** (1-4 months): Repeating interesting actions
3. **Secondary Circular Reactions** (4-8 months): Goal-directed actions emerge
4. **Coordination of Reactions** (8-12 months): Intentional problem-solving
5. **Tertiary Circular Reactions** (12-18 months): Active experimentation

Our system exhibits similar developmental progressions through:
- Initial random exploration (EXPLORE strategy)
- Discovery of interesting patterns (CURIOUS strategy)
- Goal-directed skill refinement (EXPLOIT strategy)
- Mastery and consolidation (CONSOLIDATE strategy)

### Key Research Questions

The autonomous learning loop enables investigation of:

1. **Emergent Learning Trajectories**: Do purely intrinsic motivations lead to structured developmental stages?
2. **Phase Transitions**: What triggers transitions from exploration to exploitation?
3. **Self-Calibration**: How effectively can agents adjust their self-models from prediction errors?
4. **Meta-Learning**: Can agents learn to switch strategies optimally?
5. **Cognitive Development**: Do neuromorphic architectures exhibit cognitive development milestones?

---

## Integration with Existing Systems

### Embodiment Layer
- Uses `VirtualBody` for physical interaction
- Integrates with `SensorimotorReinforcementLearner` for motor learning
- Leverages proprioception and motor feedback

### Consciousness Layer
- Updates `SelfPerceptionStream` with learning progress
- Detects anomalies and triggers self-model updates
- Maintains temporal buffer of self-awareness

### Brain Architecture
- Motor commands decoded to neural patterns
- Sensory feedback encoded as neural activity
- STDP and neuromodulation for learning

### Learning Systems
- Extends the Learning Systems Framework
- Operates at meta-level (learning how to learn)
- Complements existing plasticity mechanisms

---

## Performance Considerations

### Computational Efficiency
- **World model**: O(n²) for state dim n (linear model)
- **Goal generation**: O(1) per cycle
- **Strategy adaptation**: O(h) for history length h
- **Full cycle**: ~0.7ms per cycle (1350 cycles/s on typical hardware)

### Scalability
- State and action dimensions configurable
- World model can be replaced with more sophisticated models (neural networks, GP)
- Goal types easily extensible
- Strategy rules customizable

### Memory Usage
- Minimal: ~10MB for typical configuration
- History buffers configurable size
- Can operate with limited memory footprint

---

## Future Extensions

### Planned Enhancements

1. **Skill Acquisition Experiment**: Multi-step task learning (reach → grasp → manipulate)
2. **Self-Model Calibration**: Adaptation to body changes or perturbations
3. **Meta-Learning Switch**: Automatic strategy optimization
4. **Social Learning**: Imitation learning from demonstrations
5. **Hierarchical Goals**: Goal hierarchies and sub-goal decomposition

### Extensibility Points

1. **World Model**: Replace with neural network, Gaussian process, or physics engine
2. **Goal Types**: Add new intrinsic motivations (social, creative, etc.)
3. **Learning Strategies**: Implement new strategies (curriculum, self-play, etc.)
4. **Action Planning**: Integrate with planning algorithms (MCTS, A*, etc.)
5. **Reward Shaping**: Add external rewards for specific tasks

---

## Testing

### Unit Tests
```bash
# Run autonomous learning tests
python -m pytest tests/test_autonomous_learning.py -v

# Run with coverage
python -m pytest tests/test_autonomous_learning.py --cov=src.autonomous_learning_loop
```

**Test Coverage**: 15 tests covering:
- Intrinsic motivation engine
- Predictive world model
- Meta-learning controller
- Autonomous learning agent
- Integration tests

### Integration Tests
```bash
# Quick integration test
python -m experiments.autonomous_exploration --duration=50

# Full experiment
python -m experiments.autonomous_exploration --duration=1000 --output=results/test.json

# Analyze results
python -m analysis.learning_trajectory --input=results/test.json
```

---

## Troubleshooting

### Common Issues

**Issue**: Agent not discovering objects
- **Solution**: Check movement generation, increase exploration weight
- **Debug**: Monitor agent position and object distances

**Issue**: Strategy not changing
- **Solution**: Verify `min_steps_before_change` parameter, check performance metrics
- **Debug**: Print `steps_since_strategy_change` counter

**Issue**: High prediction errors
- **Solution**: Increase world model learning rate, check state encoding
- **Debug**: Monitor `world_model.accuracy_history`

**Issue**: No goal generation
- **Solution**: Check goal priority weights, verify state format
- **Debug**: Print urgency calculations in `_calculate_goal_urgencies`

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Access internal state:
```python
# Check current goal
print(agent.current_goal)

# Check strategy history
print(agent.meta_controller.strategy_history)

# Check world model accuracy
print(agent.world_model.get_accuracy())

# Check motivation state
print(agent.motivation_engine.visited_states)
```

---

## Citation

If you use the autonomous learning loop in your research, please cite:

```bibtex
@software{4d_neural_cognition_autonomous,
  title = {Autonomous Learning Loop for 4D Neural Cognition},
  author = {Thomas Heisig and contributors},
  year = {2025},
  url = {https://github.com/Thomas-Heisig/4D-Neural-Cognition}
}
```

---

## Contributing

Contributions are welcome! Areas of particular interest:

1. **World Model Improvements**: Neural networks, Gaussian processes
2. **New Goal Types**: Social goals, creative goals
3. **Learning Strategies**: Curriculum learning, self-play
4. **Experimental Paradigms**: New tests for autonomous learning
5. **Analysis Tools**: Visualization, trajectory analysis

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

This implementation is inspired by:
- Predictive Processing frameworks (Friston, Clark)
- Intrinsic Motivation in robotics (Oudeyer, Schmidhuber)
- Developmental robotics and sensorimotor learning
- Active Inference and Free Energy Principle

The autonomous learning loop represents the culmination of integrating neuromorphic architecture, embodied cognition, self-perception, and autonomous learning into a unified system capable of self-directed cognitive development.
