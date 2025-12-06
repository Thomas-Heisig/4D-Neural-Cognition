# Sensory Input Tutorial

Learn how to provide sensory input to your neural network, including vision, audio, and the unique digital sense.

## Table of Contents

1. [Overview of Senses](#overview-of-senses)
2. [Vision Input](#vision-input)
3. [Audio Input](#audio-input)
4. [Digital Sense](#digital-sense)
5. [Other Senses](#other-senses)
6. [Advanced Techniques](#advanced-techniques)

---

## Overview of Senses

The 4D Neural Cognition system supports seven sensory modalities:

| Sense | Description | Input Type |
|-------|-------------|------------|
| **vision** | Visual information (images) | 2D array (height × width) |
| **audio** | Auditory information (sound) | 2D array (frequency × time) |
| **touch** | Tactile/somatosensory | 2D array (body map) |
| **taste** | Gustatory information | 2D array (taste space) |
| **smell** | Olfactory information | 2D array (odor space) |
| **proprioception** | Body position/movement | 2D array (joint angles) |
| **digital** | Text/data/symbolic | Text string or structured data |

Each sense is mapped to specific areas in the 4D brain lattice.

---

## Vision Input

### Basic Vision Example

```python
import numpy as np
from senses import feed_sense_input

# Create a simple 10×10 black and white image
# 1 = white/active, 0 = black/inactive
image = np.zeros((10, 10))
image[3:7, 3:7] = 1.0  # White square in center

# Feed to visual area
feed_sense_input(
    model=sim.model
    sense_name="vision"
    input_data=image
    z_layer=0,        # Which z-layer to target
         # Input strength
)

# Run simulation to process the input
sim.run(steps=50)
```

### Understanding Vision Parameters

**Input Data**:
- 2D numpy array
- Values typically 0 to 1 (normalized)
- Shape can vary, automatically scaled to target area

**Z-Layer**:
- Which depth layer in the visual area to stimulate
- Different layers can process different aspects
- Default: 0

**Intensity**:
- Multiplier for input strength
- Higher = stronger neural activation
- Typical range: 1.0 to 10.0

### Pattern Examples

```python
# Vertical line
pattern = np.zeros((10, 10))
pattern[:, 5] = 1.0

# Horizontal line
pattern = np.zeros((10, 10))
pattern[5, :] = 1.0

# Diagonal
pattern = np.eye(10)

# Checkerboard
x, y = np.meshgrid(range(10), range(10))
pattern = ((x + y) % 2).astype(float)

# Circle
x, y = np.meshgrid(range(10), range(10))
center_x, center_y = 5, 5
radius = 3
pattern = ((x - center_x)**2 + (y - center_y)**2 <= radius**2).astype(float)
```

### Grayscale Images

```python
# Create grayscale gradient
gradient = np.linspace(0, 1, 100).reshape(10, 10)

feed_sense_input(
    model=sim.model
    sense_name="vision"
    input_data=gradient
    
)
```

### Real Image Loading

```python
from PIL import Image

# Load and preprocess image
img = Image.open('photo.jpg')
img = img.convert('L')  # Convert to grayscale
img = img.resize((50, 50))  # Resize

# Convert to numpy array and normalize
img_array = np.array(img) / 255.0

# Feed to network
feed_sense_input(
    model=sim.model
    sense_name="vision"
    input_data=img_array
    
)
```

### Sequential Visual Input

```python
# Show a sequence of patterns
patterns = [
    np.eye(10),                    # Diagonal
    np.rot90(np.eye(10)),          # Other diagonal
    np.ones((10, 10)) * 0.5,       # Uniform gray
    np.random.rand(10, 10)         # Random noise
]

for i, pattern in enumerate(patterns):
    print(f"Showing pattern {i+1}...")
    
    # Feed pattern
    feed_sense_input(
        model=sim.model
        sense_name="vision"
        input_data=pattern
        
    )
    
    # Let network process
    sim.run(steps=25)
    
    # Optional: Clear input between patterns
    # (neurons will decay naturally)
```

---

## Audio Input

Audio is represented as a 2D array: frequency × time

### Simple Tone

```python
# Create a simple tone (frequency over time)
# Rows = frequency bins, Columns = time steps
audio = np.zeros((20, 10))  # 20 frequencies, 10 time steps

# Activate frequency bin 10 for all time
audio[10, :] = 1.0

feed_sense_input(
    model=sim.model
    sense_name="audio"
    input_data=audio
    
)
```

### Frequency Sweep

```python
# Rising pitch
audio = np.zeros((20, 20))
for t in range(20):
    freq = int(t)  # Frequency increases with time
    audio[freq, t] = 1.0

feed_sense_input(
    model=sim.model
    sense_name="audio"
    input_data=audio
    
)
```

### Chord (Multiple Frequencies)

```python
# Three frequency components (a chord)
audio = np.zeros((20, 10))
audio[5, :] = 1.0   # Base note
audio[8, :] = 0.7   # Third
audio[12, :] = 0.5  # Fifth

feed_sense_input(
    model=sim.model
    sense_name="audio"
    input_data=audio
    
)
```

### Realistic Audio Processing

```python
import scipy.signal as signal

# Load audio file (requires scipy and librosa)
# This is a conceptual example
sample_rate = 16000
duration = 1.0  # seconds

# Create spectrogram
frequencies, times, spectrogram = signal.spectrogram(
    audio_signal
    fs=sample_rate
    nperseg=256
)

# Normalize and resize to fit network
spectrogram = spectrogram / np.max(spectrogram)
spectrogram_resized = resize_array(spectrogram, (20, 20))

feed_sense_input(
    model=sim.model
    sense_name="audio"
    input_data=spectrogram_resized
    
)
```

---

## Digital Sense

The digital sense is unique: it processes text and symbolic information.

### Basic Text Input

```python
from senses import create_digital_sense_input, feed_sense_input

# Create input from text
text = "Hello, neural network!"
digital_input = create_digital_sense_input(text)

# Feed to network
feed_sense_input(
    model=sim.model
    sense_name="digital"
    input_data=digital_input
    
)
```

### How It Works

The `create_digital_sense_input` function:
1. Converts text to character codes
2. Normalizes to 0-1 range
3. Creates a 2D representation
4. Maps to the digital sensor area

### Structured Data

```python
# Numbers and data
data_string = "123.456 789.012"
digital_input = create_digital_sense_input(data_string)

feed_sense_input(
    model=sim.model
    sense_name="digital"
    input_data=digital_input
    
)
```

### Sequential Text Processing

```python
# Process words one at a time
sentence = "The quick brown fox jumps"
words = sentence.split()

for word in words:
    print(f"Processing: {word}")
    
    # Create input
    digital_input = create_digital_sense_input(word)
    
    # Feed to network
    feed_sense_input(
        model=sim.model
        sense_name="digital"
        input_data=digital_input
        
    )
    
    # Process
    sim.run(steps=10)
```

### Symbolic Patterns

```python
# Process mathematical expressions
expressions = [
    "1+1=2"
    "2*3=6"
    "5-2=3"
]

for expr in expressions:
    digital_input = create_digital_sense_input(expr)
    feed_sense_input(
        model=sim.model
        sense_name="digital"
        input_data=digital_input
        
    )
    sim.run(steps=20)
```

---

## Other Senses

### Touch/Somatosensory

```python
# Simulate touch on a body map
# Example: 10×10 body surface
touch_map = np.zeros((10, 10))
touch_map[2:4, 5:7] = 1.0  # Stimulus at specific location

feed_sense_input(
    model=sim.model
    sense_name="touch"
    input_data=touch_map
    
)
```

### Taste

```python
# 5 basic tastes: sweet, salty, sour, bitter, umami
# Represented as 2D space
taste_input = np.zeros((5, 5))
taste_input[0, 0] = 1.0  # Sweet
taste_input[1, 0] = 0.5  # Bit of salty

feed_sense_input(
    model=sim.model
    sense_name="taste"
    input_data=taste_input
    
)
```

### Smell

```python
# Odor space representation
# Different dimensions for different odor qualities
smell_input = np.random.rand(10, 10) * 0.5  # Complex odor
smell_input[5, 5] = 1.0  # Strong component

feed_sense_input(
    model=sim.model
    sense_name="smell"
    input_data=smell_input
    
)
```

### Proprioception

```python
# Body position and movement
# Each position represents a joint angle or body part position
proprioception = np.zeros((8, 8))
proprioception[0, :] = 0.3  # Left arm
proprioception[1, :] = 0.7  # Right arm
proprioception[2, :] = 0.5  # Left leg
proprioception[3, :] = 0.5  # Right leg

feed_sense_input(
    model=sim.model
    sense_name="proprioception"
    input_data=proprioception
    
)
```

---

## Advanced Techniques

### Multi-Modal Input

Combine multiple senses:

```python
# Create visual and audio input simultaneously
visual = np.random.rand(10, 10)
audio = np.random.rand(20, 10)

# Feed both
feed_sense_input(sim.model, "vision", visual)
feed_sense_input(sim.model, "audio", audio)

# Process
sim.run(steps=50)
```

### Timed Sequences

```python
# Create a timed sequence of inputs
def timed_sequence(sim, inputs, steps_per_input=25):
    """
    inputs: list of (sense_name, data) tuples
    """
    for sense_name, data in inputs:
        print(f"Feeding {sense_name}...")
        feed_sense_input(
            model=sim.model
            sense_name=sense_name
            input_data=data
            
        )
        sim.run(steps=steps_per_input)

# Use it
sequence = [
    ("vision", np.eye(10))
    ("audio", np.ones((20, 10)) * 0.5)
    ("digital", create_digital_sense_input("test"))
]
timed_sequence(sim, sequence)
```

### Continuous Input Streams

```python
# Simulate continuous sensory stream
def continuous_input(sim, duration=100):
    """Generate random continuous input."""
    for step in range(duration):
        # Generate new visual frame
        visual = np.random.rand(10, 10) * 0.5
        
        feed_sense_input(
            sim.model
            "vision"
            visual
            
        )
        
        sim.step()
        
        if step % 20 == 0:
            print(f"Step {step}/{duration}")

continuous_input(sim, duration=100)
```

### Intensity Modulation

```python
# Vary input intensity over time
base_pattern = np.ones((10, 10))

for intensity in np.linspace(1, 10, 10):
    print(f"Intensity: {intensity:.1f}")
    
    feed_sense_input(
        sim.model
        "vision"
        base_pattern
        intensity
    )
    
    sim.run(steps=10)
```

### Input with Noise

```python
# Add noise to inputs for robustness
clean_pattern = np.eye(10)

# Add Gaussian noise
noisy_pattern = clean_pattern + np.random.normal(0, 0.1, clean_pattern.shape)
noisy_pattern = np.clip(noisy_pattern, 0, 1)  # Keep in valid range

feed_sense_input(
    sim.model
    "vision"
    noisy_pattern
    
)
```

### Accumulating Input

```python
# Accumulate input over time instead of replacing
# (useful for integration tasks)

# First input
feed_sense_input(
    sim.model
    "vision"
    np.ones((10, 10)) * 0.3
    
)
sim.run(steps=10)

# Second input (adds to existing)
feed_sense_input(
    sim.model
    "vision"
    np.ones((10, 10)) * 0.3
    
)
sim.run(steps=10)

# Result: neurons receive combined input
```

---

## Complete Example: Multi-Modal Learning

```python
#!/usr/bin/env python3
"""Multi-modal sensory input example."""

import numpy as np
from brain_model import BrainModel
from simulation import Simulation
from senses import feed_sense_input, create_digital_sense_input

def main():
    # Setup
    model = BrainModel(config_path="brain_base_model.json")
    sim = Simulation(model, seed=42)
    
    # Initialize multiple sensory areas
    sim.initialize_neurons(
        areas=["V1_like", "A1_like", "Digital_sensor"]
        density=0.1
    )
    sim.initialize_random_synapses(connection_prob=0.1)
    
    print("=== Multi-Modal Input Demo ===\n")
    
    # Trial 1: Visual + Audio
    print("Trial 1: Visual + Audio")
    visual = np.eye(10)
    audio = np.ones((20, 10)) * 0.5
    
    feed_sense_input(sim.model, "vision", visual)
    feed_sense_input(sim.model, "audio", audio)
    sim.run(steps=30)
    
    # Trial 2: Visual + Digital
    print("\nTrial 2: Visual + Digital")
    visual = np.ones((10, 10))
    digital = create_digital_sense_input("cat")
    
    feed_sense_input(sim.model, "vision", visual)
    feed_sense_input(sim.model, "digital", digital)
    sim.run(steps=30)
    
    # Trial 3: All three
    print("\nTrial 3: All Modalities")
    visual = np.random.rand(10, 10)
    audio = np.random.rand(20, 10)
    digital = create_digital_sense_input("multimodal")
    
    feed_sense_input(sim.model, "vision", visual)
    feed_sense_input(sim.model, "audio", audio)
    feed_sense_input(sim.model, "digital", digital)
    sim.run(steps=30)
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    main()
```

---

## Tips and Best Practices

1. **Start Simple**: Begin with small, simple patterns before complex inputs
2. **Normalize Data**: Keep input values in 0-1 range
3. **Appropriate Intensity**: Too high = saturation, too low = no response
4. **Process Time**: Give network time to process (20-50 steps per input)
5. **Monitor Activity**: Check spike counts to ensure input is being processed
6. **Reproducibility**: Use same patterns and parameters for experiments

---

## Next Steps

- **[Plasticity Tutorial](PLASTICITY.md)** - Learn about learning mechanisms
- **[API Documentation](../api/API.md)** - Full API reference
- **[Examples](../../examples/)** - More complex demonstrations

---

*Last Updated: December 2025*  
*Part of the 4D Neural Cognition Documentation*
