# Video Export for Simulations

This document describes the video export functionality for recording neural network simulations.

## Overview

The `video_export.py` module provides tools to capture simulation frames and export them as video files. Requires OpenCV (`cv2`).

## Installation

```bash
pip install opencv-python
```

## Basic Usage

### Simple Video Recording

```python
from src.video_export import VideoExporter
import numpy as np

# Create video exporter
with VideoExporter("simulation.mp4", fps=30, resolution=(1920, 1080)) as exporter:
    for frame_idx in range(300):  # 10 seconds at 30 fps
        # Create or capture frame (RGB or BGR format)
        frame = create_frame()  # Returns numpy array (height, width, 3)
        
        # Add to video
        exporter.add_frame(frame)

print("Video saved to simulation.mp4")
```

### Recording Neural Network Simulation

```python
from src.video_export import SimulationRecorder
from src.brain_model import BrainModel

# Create brain model
model = BrainModel(lattice_size=(20, 20, 10, 5))

# Record simulation
with SimulationRecorder(
    model=model,
    output_path="brain_simulation.mp4",
    fps=30,
    resolution=(1920, 1080)
) as recorder:
    # Record 1000 simulation steps (captures every step by default)
    recorder.record_simulation(num_steps=1000, steps_per_frame=10)

print("Simulation video saved!")
```

## Advanced Features

### Custom Visualization

```python
from src.video_export import SimulationRecorder
import cv2
import numpy as np

def custom_visualization(model, step):
    """Custom visualization function."""
    # Create frame
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    # Draw neural activity
    for neuron in model.neurons.values():
        x = int(neuron.x * 50)
        y = int(neuron.y * 50)
        activity = int(neuron.v_membrane + 70)  # Normalize to 0-255
        color = (0, activity, 255 - activity)
        cv2.circle(frame, (x, y), 3, color, -1)
    
    # Add text overlay
    cv2.putText(frame, f"Step: {step}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame

# Use custom visualization
with SimulationRecorder(
    model=model,
    output_path="custom_viz.mp4",
    visualization_func=custom_visualization
) as recorder:
    recorder.record_simulation(num_steps=500)
```

### Activity Heatmap Export

```python
from src.video_export import export_activity_heatmap_video

# Export heatmap with specific 2D projection
export_activity_heatmap_video(
    model=model,
    output_path="heatmap_xy.mp4",
    num_steps=1000,
    fps=30,
    resolution=(1920, 1080),
    projection="xy"  # Options: 'xy', 'xz', 'yz', 'xw', etc.
)
```

### Side-by-Side Model Comparison

```python
from src.video_export import create_comparison_video

# Create multiple models
model1 = BrainModel(lattice_size=(10, 10, 10, 5))
model2 = BrainModel(lattice_size=(20, 20, 10, 5))
model3 = BrainModel(lattice_size=(30, 30, 10, 5))

# Record comparison video
create_comparison_video(
    models=[model1, model2, model3],
    model_names=["Small", "Medium", "Large"],
    output_path="comparison.mp4",
    num_steps=500,
    fps=30,
    resolution=(1920, 1080)
)
```

### Progress Tracking

```python
def progress_callback(current, total):
    """Display progress during recording."""
    percent = (current / total) * 100
    print(f"Recording: {percent:.1f}% ({current}/{total})", end='\r')

with SimulationRecorder(model, "output.mp4") as recorder:
    recorder.record_simulation(
        num_steps=1000,
        steps_per_frame=5,
        progress_callback=progress_callback
    )
```

## Video Configuration

### Quality Settings

```python
# High quality
exporter = VideoExporter(
    "high_quality.mp4",
    fps=60,
    resolution=(3840, 2160),  # 4K
    codec="mp4v"
)

# Low quality / fast rendering
exporter = VideoExporter(
    "low_quality.mp4",
    fps=15,
    resolution=(1280, 720),  # 720p
    codec="mp4v"
)
```

### Codec Options

- `"mp4v"` - MPEG-4 (default, good compatibility)
- `"avc1"` - H.264 (better compression)
- `"XVID"` - Xvid MPEG-4 (alternative)

## Metadata

Video metadata is automatically saved as JSON:

```python
# Recording saves metadata to simulation.json
exporter = VideoExporter("simulation.mp4", fps=30, resolution=(1920, 1080))
# ... add frames ...
exporter.finalize()

# Load metadata
import json
with open("simulation.json") as f:
    metadata = json.load(f)
    print(f"Total frames: {metadata['total_frames']}")
    print(f"FPS: {metadata['fps']}")
```

## Performance Tips

1. **Frame Rate**: Lower FPS = faster rendering, smaller files
2. **Resolution**: Lower resolution = faster processing
3. **Steps per Frame**: Skip simulation steps to speed up recording
4. **Batch Processing**: Record in segments for long simulations

```python
# Efficient long simulation recording
num_segments = 10
steps_per_segment = 1000

for segment in range(num_segments):
    output_file = f"segment_{segment}.mp4"
    
    with SimulationRecorder(model, output_file, fps=30) as recorder:
        recorder.record_simulation(
            num_steps=steps_per_segment,
            steps_per_frame=10  # Only capture every 10th step
        )
```

## Troubleshooting

### ImportError: No module named 'cv2'

Install OpenCV:
```bash
pip install opencv-python
```

### Video file is empty or corrupted

Ensure you call `finalize()` or use context manager:
```python
# Good - with context manager
with VideoExporter("output.mp4") as exporter:
    exporter.add_frame(frame)

# Good - manual finalize
exporter = VideoExporter("output.mp4")
exporter.add_frame(frame)
exporter.finalize()
```

### Frame size mismatch

Ensure frames match the specified resolution:
```python
# Frames are automatically resized, but for efficiency:
frame = np.zeros((1080, 1920, 3), dtype=np.uint8)  # Match resolution
```

## Complete Example

```python
from src.video_export import SimulationRecorder
from src.brain_model import BrainModel
from src.senses import feed_sense_input

# Create model
model = BrainModel(lattice_size=(20, 20, 10, 5))

# Setup video recording
with SimulationRecorder(
    model=model,
    output_path="training_visualization.mp4",
    fps=30,
    resolution=(1920, 1080)
) as recorder:
    
    def training_callback(step, total):
        # Optional: feed inputs during recording
        if step % 50 == 0:
            feed_sense_input(model, "vision", pattern=create_pattern())
        
        print(f"Recording: {step}/{total}", end='\r')
    
    # Record training session
    recorder.record_simulation(
        num_steps=3000,
        steps_per_frame=10,  # 300 frames total
        progress_callback=training_callback
    )

print("\nVideo recording complete!")
```

## See Also

- [Visualization Tools](../user-guide/VISUALIZATION.md)
- [Model Comparison](MODEL_COMPARISON.md)
- [Brain Model API](../api/BRAIN_MODEL.md)
