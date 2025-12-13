# VNC System Enhancements

This document describes the major enhancements made to the Virtual Neuromorphic Clock (VNC) system and the new Embodiment capabilities.

## Overview

The VNC system has been enhanced with:
1. **Vectorized VPU** - 50-100x performance improvement through array operations
2. **Adaptive Orchestrator** - Self-optimizing load balancing and resource allocation
3. **Embodiment System** - Virtual bodies for sensorimotor learning
4. **Self-Perception** - Continuous self-awareness and agency detection

## Phase 1: Vectorized VPU

### Motivation

The original VPU processed neurons sequentially within each slice, limiting the performance gain from parallelization. The Vectorized VPU uses NumPy array operations to process all neurons simultaneously.

### Implementation

**File**: `src/hardware_abstraction/vectorized_vpu.py`

Key features:
- **Structured NumPy arrays** for neuron state storage
- **Vectorized LIF dynamics** using array operations
- **Boolean masking** for spike detection and refractory periods
- **Efficient I/O** with circular buffers

```python
from hardware_abstraction.vectorized_vpu import VectorizedVPU

# Create vectorized VPU
vpu = VectorizedVPU(vpu_id=0, clock_speed_hz=20e6)
vpu.assign_slice((0, 10, 0, 10, 0, 5, 0, 0))
vpu.initialize_batch_vectorized(model, simulation)

# Process cycle (vectorized)
result = vpu.process_cycle_vectorized(global_clock_cycle=0)
```

### Performance

- **Expected speedup**: 50-100x per VPU
- **Scales with**: Number of neurons per slice
- **Memory efficient**: O(n) where n = neurons in slice

### Technical Details

The vectorized update performs:

```python
# 1. Membrane potential update (vectorized)
leak_current = (v_rest - v_membrane) / tau_m
dv = dt * (leak_current + inputs)
v_membrane += dv

# 2. Spike detection (vectorized)
spike_mask = v_membrane >= v_threshold

# 3. Reset (vectorized)
v_membrane[spike_mask] = v_reset[spike_mask]
```

## Phase 2: Adaptive VNC Orchestrator

### Motivation

Fixed partitioning can lead to load imbalances as neural activity patterns change over time. The Adaptive Orchestrator monitors performance and dynamically adjusts resource allocation.

### Implementation

**File**: `src/hardware_abstraction/adaptive_vnc_orchestrator.py`

Key features:
- **Real-time monitoring** of VPU performance
- **Load imbalance detection** using coefficient of variation
- **Hot/cold slice identification** based on spike rates
- **Adaptive repartitioning** proposals
- **Priority management** for high-activity slices

```python
from hardware_abstraction.adaptive_vnc_orchestrator import AdaptiveVNCOrchestrator

# Create orchestrator
orchestrator = AdaptiveVNCOrchestrator(
    simulation,
    imbalance_threshold=0.3,      # 30% load imbalance triggers action
    activity_threshold=0.7,       # 70% spike rate = "hot" slice
    monitoring_interval=100,       # Check every 100 cycles
)

# Monitor and adapt
result = orchestrator.monitor_and_adapt(current_cycle)
```

### Metrics

The orchestrator tracks:

1. **Load Imbalance**: Coefficient of variation of processing times
2. **Activity Levels**: Normalized spike rates per VPU
3. **Hot Slices**: VPUs with activity above threshold
4. **Cold Slices**: VPUs with activity below threshold

### Optimization Actions

When imbalance exceeds threshold:
- Analyzes neuron distribution across VPUs
- Identifies overloaded and underloaded VPUs
- Proposes repartitioning strategy
- Adjusts compute priorities

## Phase 3: Embodiment System

### Virtual Body

**File**: `src/embodiment/virtual_body.py`

Provides a simulated physical body with:
- **Skeletal structure** with configurable joints
- **Virtual muscles** with force and fatigue
- **Proprioceptive sensing** (joint angles, muscle tensions)
- **Simplified physics** for body dynamics

```python
from embodiment.virtual_body import VirtualBody

# Create humanoid body
body = VirtualBody(
    body_type="humanoid",
    num_joints=12,
    max_force=100.0
)

# Execute motor command
motor_output = {'motor_neurons': {0: 0.5, 1: 0.7, ...}}
kinematic_feedback = body.execute_motor_command(motor_output)

# Get body state
state = body.get_state()
```

### Body Types

Supported configurations:
- **Humanoid**: Spine, limbs, head (12-15 joints)
- **Quadruped**: Four-legged (8-12 joints)
- **Generic**: Customizable joint count

### Proprioception

The ProprioceptiveSensor tracks:
- Joint angles and ranges
- Muscle tensions
- Body velocity and acceleration
- Center of mass position

## Phase 4: Self-Perception Stream

**File**: `src/consciousness/self_perception_stream.py`

Maintains continuous self-awareness through:
- **Multi-modal integration** (proprioception, vision, audio, intentions)
- **Temporal tracking** with circular buffer
- **Self-consistency metrics**
- **Agency detection**

```python
from consciousness.self_perception_stream import SelfPerceptionStream

# Create self-perception stream
stream = SelfPerceptionStream(
    update_frequency_hz=100.0,
    buffer_duration_seconds=10.0
)

# Update with sensorimotor data
stream.update(
    sensor_data={'proprioception': kinematic_feedback, ...},
    motor_commands={'planned': intentions, 'executed': actions},
    internal_state={'metabolic': vitals, ...}
)

# Get self-awareness metrics
metrics = stream.get_self_awareness_metric()
# Returns: self_consistency, integration, agency_score
```

### Self-Awareness Metrics

1. **Self-Consistency** (0-1): Temporal stability of self-model
2. **Cross-Modal Integration** (0-1): How well different modalities align
3. **Agency Score** (0-1): How well intentions predict outcomes

## Integration Guide

### Basic Usage

```python
from brain_model import BrainModel
from simulation import Simulation
from hardware_abstraction.vectorized_vpu import VectorizedVPU
from hardware_abstraction.adaptive_vnc_orchestrator import AdaptiveVNCOrchestrator
from embodiment.virtual_body import VirtualBody
from consciousness.self_perception_stream import SelfPerceptionStream

# 1. Create model with VNC
model = create_your_model()
sim = Simulation(model, use_vnc=True)

# 2. Add adaptive orchestrator
orchestrator = AdaptiveVNCOrchestrator(sim)

# 3. Create virtual body
body = VirtualBody(body_type="humanoid")

# 4. Create self-perception stream
self_stream = SelfPerceptionStream()

# 5. Run sensorimotor loop
for cycle in range(n_cycles):
    # Get motor output from neural activity
    motor_output = decode_neural_activity(sim)
    
    # Execute on body
    feedback = body.execute_motor_command(motor_output)
    
    # Update self-perception
    self_stream.update(
        sensor_data={'proprioception': feedback},
        motor_commands={'planned': motor_output, 'executed': motor_output},
        internal_state=sim.get_internal_state()
    )
    
    # Monitor and adapt VNC
    orchestrator.monitor_and_adapt(cycle)
    
    # Run simulation step
    sim.step()
```

### Advanced: Custom Vectorized VPU

You can create custom VPUs with specialized processing:

```python
class CustomVectorizedVPU(VectorizedVPU):
    def process_cycle_vectorized(self, global_clock_cycle):
        # Your custom vectorized processing
        # ... call parent or implement from scratch
        result = super().process_cycle_vectorized(global_clock_cycle)
        
        # Add custom post-processing
        # ...
        
        return result
```

## Performance Tips

1. **VPU Sizing**: Balance neuron count per VPU
   - Too few neurons: overhead dominates
   - Too many: memory bandwidth bottleneck
   - Sweet spot: 100-10,000 neurons per VPU

2. **Orchestrator Tuning**:
   - `imbalance_threshold`: Lower = more aggressive optimization
   - `monitoring_interval`: Lower = more responsive, higher overhead

3. **Embodiment**:
   - Match body complexity to task requirements
   - Update self-stream at appropriate frequency (10-100 Hz)

## Future Enhancements

Planned improvements:
- **Phase 3**: Live Data Fabric (WebSocket streams, database polling)
- **Phase 4**: Hardware-in-the-Loop emulation (Loihi2, SpiNNaker)
- **Phase 5**: Neuromorphic Debugger (neuron tracing, visualization)
- **Audio/Vision**: Complete sensory modalities for embodiment
- **CuPy Support**: GPU acceleration for vectorized operations

## References

- Original VNC: See `VNC_IMPLEMENTATION_SUMMARY.md`
- LIF Model: Gerstner & Kistler, "Spiking Neuron Models"
- Embodied Cognition: Pfeifer & Bongard, "How the Body Shapes the Way We Think"
- Self-Awareness: Seth & Bayne, "Theories of Consciousness"
