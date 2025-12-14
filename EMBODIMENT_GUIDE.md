# Embodied AI & Self-Awareness System Guide

This guide explains the new embodiment system for the 4D Neural Cognition framework, which integrates sensorimotor learning, self-perception, and cognitive resource management.

## Table of Contents
- [Overview](#overview)
- [Core Components](#core-components)
- [Getting Started](#getting-started)
- [Experiments](#experiments)
- [Dashboard](#dashboard)
- [API Reference](#api-reference)
- [Scientific Background](#scientific-background)

## Overview

The embodiment system enables the neural network to:
1. **Control a virtual body** with proprioceptive feedback
2. **Learn motor skills** through reinforcement learning with STDP
3. **Maintain self-awareness** through continuous self-perception
4. **Detect anomalies** in self-model and recalibrate
5. **Recognize itself** across multiple sensory modalities
6. **Optimize compute resources** based on cognitive demands

## Core Components

### 1. Sensorimotor Reinforcement Learner

**File**: `src/embodiment/sensorimotor_learner.py`

Implements the critical learning loop connecting action, perception, and reward:

```python
from embodiment.sensorimotor_learner import SensorimotorReinforcementLearner

learner = SensorimotorReinforcementLearner(
    virtual_body=body,
    brain_model=brain,
    learning_rate=0.01,
    dopamine_modulation_strength=0.1
)

# Learn from interaction
result = learner.learn_from_interaction(
    action=motor_command,
    resulting_feedback=proprioceptive_feedback,
    external_reward=task_reward
)
```

**Key Features**:
- **Intrinsic Motivation**: Novelty-seeking and competence-driven rewards
- **STDP Integration**: Spike-timing-dependent plasticity for synaptic updates
- **Dopamine Modulation**: Neuromodulation based on reward signals
- **Learning Progress Tracking**: Episode-by-episode metrics

### 2. Self-Perception Stream

**File**: `src/consciousness/self_perception_stream.py` (extended)

Maintains continuous self-awareness with anomaly detection:

```python
from consciousness.self_perception_stream import SelfPerceptionStream

perception = SelfPerceptionStream(
    update_frequency_hz=100.0,
    buffer_duration_seconds=10.0
)

# Update perception
perception.update(
    sensor_data={'proprioception': feedback},
    motor_commands=commands,
    internal_state=state
)

# Detect anomalies
anomalies = perception.detect_self_consistency_anomalies()

# Recalibrate if needed
if anomalies:
    recalibration = perception.update_self_model_based_on_anomalies(anomalies)
```

**Anomaly Types**:
- **Motor Prediction Error**: Mismatch between predicted and actual feedback
- **AV Sync Error**: Audio-visual synchronization issues
- **Agency Error**: Loss of control or external manipulation

### 3. Cognitive-Aware VNC Orchestrator

**File**: `src/hardware_abstraction/adaptive_vnc_orchestrator.py` (extended)

Dynamically allocates VPUs to critical brain regions during learning:

```python
from hardware_abstraction.adaptive_vnc_orchestrator import CognitiveAwareOrchestrator

orchestrator = CognitiveAwareOrchestrator(
    simulation=sim,
    monitoring_interval=100,
    sensorimotor_activity_threshold=0.6
)

# Monitor and adapt
result = orchestrator.monitor_and_adapt(current_cycle)

# Check prioritization
if result.get('vpu_reallocations'):
    print(f"Reallocated {result['vpu_reallocations']} VPUs")
```

**Critical Regions** (auto-prioritized):
- **Motor Planning** (w=10): Minimum 2 VPUs
- **Self-Perception** (w=12): Minimum 2 VPUs
- **Sensor Fusion** (w=6): Minimum 1 VPU
- **Executive Control** (w=14): Minimum 1 VPU

### 4. Multimodal Integration System

**File**: `src/multimodal_integration.py`

Fuses visual, auditory, and proprioceptive signals for self-recognition:

```python
from multimodal_integration import MultimodalIntegrationSystem

multimodal = MultimodalIntegrationSystem()

# Fuse modalities
result = multimodal.fuse_modalities_for_self_recognition(
    proprioception_data=body_feedback
)

print(f"Self-confidence: {result['self_confidence']:.2%}")
print(f"Modality contributions: {result['modality_contributions']}")
```

**Features**:
- **Bayesian Fusion**: Confidence-weighted integration
- **Cross-Modal Correlation**: Movement matching across modalities
- **Adaptive Weights**: Learns reliability of each modality
- **Self-Other Boundary**: Distinguishes self from environment

## Getting Started

### Quick Demo

Run the interactive demo to see all components working together:

```bash
python examples/embodiment_demo.py
```

This demonstrates:
- Virtual body control
- Sensorimotor learning loop
- Anomaly detection and recalibration
- Multimodal self-recognition
- VNC cognitive prioritization

### Minimal Example

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from brain_model import BrainModel
from embodiment.virtual_body import VirtualBody
from embodiment.sensorimotor_learner import SensorimotorReinforcementLearner

# Create brain
brain = BrainModel(config={"lattice_shape": [10, 10, 5, 15], ...})

# Create body
body = VirtualBody(body_type="humanoid", num_joints=8)

# Create learner
learner = SensorimotorReinforcementLearner(body, brain)

# Learning loop
learner.start_episode()
for step in range(100):
    # Generate action
    action = generate_motor_command()
    
    # Execute on body
    feedback = body.execute_motor_command(action)
    
    # Learn
    result = learner.learn_from_interaction(action, feedback, reward=0.5)
    
learner.end_episode()
```

## Experiments

### 1. Sensorimotor Learning

Test motor skill learning with reaching tasks:

```bash
# Basic usage
python -m experiments.sensorimotor_learning

# Custom configuration
python -m experiments.sensorimotor_learning \
    --body_type humanoid \
    --episodes 200 \
    --learning_algorithm stdp_plus_reward \
    --output results.json
```

**Metrics**:
- Average reward per episode
- Success rate (reaching target)
- Learning speed (episodes to criterion)
- Prediction error reduction

### 2. Self-Anomaly Detection

Test anomaly detection and self-model recalibration:

```bash
# Basic usage
python -m experiments.self_anomaly_detection

# Test external push
python -m experiments.self_anomaly_detection \
    --perturbation external_push \
    --magnitude 0.8 \
    --trials 20 \
    --output anomaly_results.json
```

**Measured**:
- Detection rate (% of perturbations detected)
- Detection latency (steps from perturbation to detection)
- Recalibration effectiveness
- Prediction error before/during/after perturbation

### 3. Cognitive VNC

Test VPU prioritization during motor learning:

```bash
# With cognitive orchestrator
python -m experiments.cognitive_vnc --cognitive

# Baseline (no orchestrator)
python -m experiments.cognitive_vnc --baseline

# Compare results
python -m experiments.cognitive_vnc \
    --scenario motor_learning \
    --vpus 6 \
    --cycles 1000 \
    --output vnc_results.json
```

**Compared**:
- Learning speed (spikes per cycle)
- VPU allocation across regions
- Reallocation frequency
- Resource efficiency

## Dashboard

### Digital Twin Section

Access at `http://localhost:5000/dashboard` → "Digital Twin"

**Features**:
- **3D Body Visualization**: Stick-figure with adjustable joints
- **Proprioception Display**: Real-time joint angles, muscle tensions
- **Motor Commands**: Interactive sliders for manual control
- **Multimodal Confidence**: Bars showing self-recognition confidence

### Self-Awareness Section

**Features**:
- **Anomaly Alerts**: Real-time detection with severity levels
- **Self-Model Calibration**: Current calibration state
- **Consistency Metrics**: Temporal and cross-modal integration
- **Perception History**: Time-series charts

### Motor Learning Section

**Features**:
- **Learning Progress**: Episode rewards and trends
- **Reward System**: External, intrinsic, and total rewards
- **STDP Visualization**: Spike-timing curve
- **VNC Priority Heatmap**: VPU allocation to critical regions
- **Neuromodulation Status**: Dopamine levels, plasticity tags

## API Reference

### SensorimotorReinforcementLearner

```python
class SensorimotorReinforcementLearner:
    def __init__(
        self,
        virtual_body: VirtualBody,
        brain_model: BrainModel,
        learning_rate: float = 0.01,
        discount_factor: float = 0.95,
        dopamine_modulation_strength: float = 0.1
    )
    
    def learn_from_interaction(
        self,
        action: Dict,
        resulting_feedback: Dict,
        external_reward: float = 0.0
    ) -> Dict
    
    def start_episode() -> None
    def end_episode() -> Dict
    def calculate_learning_progress() -> Dict
    def get_statistics() -> Dict
```

### SelfPerceptionStream (Extended)

```python
class SelfPerceptionStream:
    def detect_self_consistency_anomalies() -> List[Dict]
    def predict_next_proprioception() -> Dict
    def discrepancy(predicted: Dict, actual: Dict) -> float
    def update_self_model_based_on_anomalies(anomalies: List[Dict]) -> Dict
```

### CognitiveAwareOrchestrator

```python
class CognitiveAwareOrchestrator(AdaptiveVNCOrchestrator):
    CRITICAL_REGIONS = {
        'motor_planning': {'w_slice': 10, 'min_vpus': 2, 'priority': 3},
        'self_perception': {'w_slice': 12, 'min_vpus': 2, 'priority': 3},
        'sensor_fusion': {'w_slice': 6, 'min_vpus': 1, 'priority': 2},
        'executive_control': {'w_slice': 14, 'min_vpus': 1, 'priority': 2},
    }
    
    def is_high_sensorimotor_activity() -> bool
    def prioritize_critical_regions() -> Dict
    def get_cognitive_performance_summary() -> Dict
```

### MultimodalIntegrationSystem

```python
class MultimodalIntegrationSystem:
    def __init__(self)
    
    def fuse_modalities_for_self_recognition(
        self,
        proprioception_data: Optional[Dict] = None
    ) -> Dict
    
    def bayesian_fusion(
        self,
        modality_evidences: List[Tuple[str, float]],
        prior: float = 0.5
    ) -> Dict
    
    def update_modality_weights(feedback: Dict[str, float]) -> None
    def get_statistics() -> Dict
```

## Scientific Background

### Embodied Cognition

The embodiment system is grounded in embodied cognition theory:
- **Body Schema**: Internal model of body configuration (proprioception)
- **Forward Models**: Predict sensory consequences of actions
- **Sense of Agency**: Feeling of control over actions and outcomes
- **Self-Other Distinction**: Distinguishing self from environment

### Reinforcement Learning

Implements biologically-inspired RL:
- **STDP**: Synaptic plasticity based on spike timing
- **Dopamine Modulation**: Reward signals modulate plasticity
- **Intrinsic Motivation**: Curiosity and competence drive exploration
- **Prediction Error**: Learning signal from expectation violations

### Multimodal Integration

Based on multisensory integration research:
- **Bayesian Fusion**: Optimal combination of uncertain information
- **Cross-Modal Correlation**: Self-identification through movement
- **Temporal Binding**: Linking actions to sensory effects
- **Adaptive Weighting**: Learning reliability of each modality

### Cognitive Resource Management

Inspired by biological attention and resource allocation:
- **Priority Mapping**: Critical regions get more resources
- **Activity-Based Allocation**: Compute follows cognitive demands
- **Dynamic Reallocation**: Adapt to changing task requirements

## References

### Key Papers

1. **Embodied Cognition**: Shapiro, L. (2019). Embodied Cognition. Routledge.
2. **Forward Models**: Wolpert, D.M., & Kawato, M. (1998). Multiple paired forward and inverse models for motor control. Neural Networks, 11(7-8), 1317-1329.
3. **Sense of Agency**: Haggard, P. (2017). Sense of agency in the human brain. Nature Reviews Neuroscience, 18(4), 196-207.
4. **STDP**: Markram, H., Lübke, J., Frotscher, M., & Sakmann, B. (1997). Regulation of synaptic efficacy by coincidence of postsynaptic APs and EPSPs. Science, 275(5297), 213-215.
5. **Multimodal Integration**: Ernst, M.O., & Banks, M.S. (2002). Humans integrate visual and haptic information in a statistically optimal fashion. Nature, 415(6870), 429-433.

### Implementation Details

- **STDP Window**: τ+ = τ- = 20ms (standard values)
- **Dopamine Strength**: 0.1 (tunable, affects learning rate)
- **Self-Perception Frequency**: 100 Hz (fast enough for motor control)
- **Anomaly Threshold**: 0.2 prediction error (20% discrepancy)
- **VPU Reallocation**: Based on activity > 0.6 (60% spike rate)

## Troubleshooting

### Common Issues

**Issue**: Body doesn't move
- Check that neurons exist in motor cortex (w=10)
- Verify motor commands are being generated
- Ensure VPUs are allocated to motor slice

**Issue**: No anomalies detected
- Increase perturbation magnitude
- Check that self-perception stream is updating
- Verify prediction model is initialized

**Issue**: VNC not reallocating VPUs
- Confirm sensorimotor activity is above threshold (0.6)
- Check that multiple VPUs exist
- Verify monitoring interval is reached

**Issue**: Low self-recognition confidence
- Allow more time for modality fusion
- Check that proprioception data is valid
- Tune Bayesian fusion weights

## Future Enhancements

Potential additions:
- [ ] Full physics engine integration (PyBullet/MuJoCo)
- [ ] Vision-based self-recognition (camera feed)
- [ ] Actual audio processing (microphone input)
- [ ] Real robot control interfaces
- [ ] Advanced forward model learning
- [ ] Social cognition (recognizing others)
- [ ] Tool use and affordance learning
- [ ] Emotion-body coupling

## License

This code is part of the 4D Neural Cognition project and follows the project's main license (see LICENSE file).

## Contact

For questions or contributions, please open an issue on GitHub.

---

*Last updated: December 2025*
