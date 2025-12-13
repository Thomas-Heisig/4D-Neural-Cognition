"""Video export functionality for neural network simulations.

This module provides tools to capture simulation frames and export them as videos.
Supports MP4 format with configurable quality and frame rates.
"""

from typing import Optional, List, Tuple, Callable, Any, TYPE_CHECKING
import numpy as np
from pathlib import Path
import json

if TYPE_CHECKING:
    try:
        from .brain_model import BrainModel
    except ImportError:
        from brain_model import BrainModel

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    import logging
    logging.debug(
        "OpenCV (cv2) not available. Video export features will be unavailable. "
        "Install with: pip install opencv-python"
    )


class VideoExporter:
    """Export neural network simulations as video files."""
    
    def __init__(
        self,
        output_path: str,
        fps: int = 30,
        resolution: Tuple[int, int] = (1920, 1080),
        codec: str = "mp4v"
    ):
        """Initialize video exporter.
        
        Args:
            output_path: Path to output video file (.mp4).
            fps: Frames per second for video.
            resolution: Video resolution as (width, height).
            codec: Video codec (default 'mp4v' for MP4).
            
        Raises:
            ImportError: If OpenCV (cv2) is not installed.
        """
        if not CV2_AVAILABLE:
            raise ImportError(
                "OpenCV (cv2) is required for video export. "
                "Install with: pip install opencv-python"
            )
        
        self.output_path = Path(output_path)
        self.fps = fps
        self.resolution = resolution
        self.codec = codec
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            fps,
            resolution
        )
        
        self.frame_count = 0
        self.metadata = {
            "fps": fps,
            "resolution": resolution,
            "codec": codec,
            "frames": []
        }
    
    def add_frame(self, frame: np.ndarray, color_format: str = "RGB") -> None:
        """Add a frame to the video.
        
        Args:
            frame: Frame as numpy array (height, width, 3).
            color_format: Input color format, either "RGB" or "BGR" (default: "RGB").
        """
        if frame.shape[:2][::-1] != self.resolution:
            # Resize frame to match resolution
            frame = cv2.resize(frame, self.resolution)
        
        # Ensure frame is in BGR format (OpenCV default)
        if frame.shape[2] == 3 and color_format == "RGB":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        self.writer.write(frame)
        self.frame_count += 1
    
    def finalize(self) -> None:
        """Finalize and close the video file."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        
        # Save metadata
        self.metadata["total_frames"] = self.frame_count
        metadata_path = self.output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finalize()


class SimulationRecorder:
    """Record neural network simulation as video with visualization."""
    
    def __init__(
        self,
        model: "BrainModel",
        output_path: str,
        fps: int = 30,
        resolution: Tuple[int, int] = (1920, 1080),
        visualization_func: Optional[Callable] = None
    ):
        """Initialize simulation recorder.
        
        Args:
            model: Brain model to record.
            output_path: Path to output video file.
            fps: Frames per second.
            resolution: Video resolution.
            visualization_func: Optional custom visualization function that takes
                               (model, step) and returns RGB frame as numpy array.
        """
        if not CV2_AVAILABLE:
            raise ImportError(
                "OpenCV (cv2) is required for video recording. "
                "Install with: pip install opencv-python"
            )
        
        self.model = model
        self.exporter = VideoExporter(output_path, fps, resolution)
        self.visualization_func = visualization_func or self._default_visualization
    
    def _default_visualization(self, model: "BrainModel", step: int) -> np.ndarray:
        """Default visualization: 2D heatmap of neural activity.
        
        Args:
            model: Brain model.
            step: Current simulation step.
            
        Returns:
            RGB frame as numpy array.
        """
        # Create a 2D projection of 4D space
        width, height = self.exporter.resolution
        
        # Create blank frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Get neuron positions and activities
        if not model.neurons:
            return frame
        
        # Project 4D coordinates to 2D
        max_coords = {'x': 1, 'y': 1, 'z': 1, 'w': 1}
        for neuron in model.neurons.values():
            max_coords['x'] = max(max_coords['x'], neuron.x)
            max_coords['y'] = max(max_coords['y'], neuron.y)
            max_coords['z'] = max(max_coords['z'], neuron.z)
            max_coords['w'] = max(max_coords['w'], neuron.w)
        
        for neuron in model.neurons.values():
            # Project 4D to 2D: use (x + z*0.5, y + w*0.5)
            proj_x = (neuron.x + neuron.z * 0.5) / (max_coords['x'] + max_coords['z'] * 0.5)
            proj_y = (neuron.y + neuron.w * 0.5) / (max_coords['y'] + max_coords['w'] * 0.5)
            
            pixel_x = int(proj_x * (width - 1))
            pixel_y = int(proj_y * (height - 1))
            
            # Get neuron activity (membrane potential)
            activity = getattr(neuron, 'v_membrane', 0.0)
            
            # Normalize activity to 0-255
            activity_normalized = max(0, min(255, int((activity + 70) / 140 * 255)))
            
            # Color based on activity (blue = low, red = high)
            if activity_normalized < 128:
                # Blue to green
                color = (0, activity_normalized * 2, 255 - activity_normalized * 2)
            else:
                # Green to red
                color = ((activity_normalized - 128) * 2, 255 - (activity_normalized - 128) * 2, 0)
            
            # Draw neuron as small circle
            cv2.circle(frame, (pixel_x, pixel_y), 2, color, -1)
        
        # Add text overlay with step number
        cv2.putText(
            frame,
            f"Step: {step}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        
        return frame
    
    def record_simulation(
        self,
        num_steps: int,
        steps_per_frame: int = 1,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> None:
        """Record simulation for specified number of steps.
        
        Args:
            num_steps: Number of simulation steps to record.
            steps_per_frame: Number of simulation steps per video frame.
            progress_callback: Optional callback(current_step, total_steps).
        """
        for step in range(num_steps):
            # Step simulation
            if hasattr(self.model, 'step'):
                self.model.step()
            
            # Capture frame every N steps
            if step % steps_per_frame == 0:
                frame = self.visualization_func(self.model, step)
                self.exporter.add_frame(frame)
            
            # Progress callback
            if progress_callback:
                progress_callback(step + 1, num_steps)
        
        # Finalize video
        self.exporter.finalize()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.exporter.finalize()


def export_activity_heatmap_video(
    model: "BrainModel",
    output_path: str,
    num_steps: int,
    fps: int = 30,
    resolution: Tuple[int, int] = (1920, 1080),
    projection: str = "xy"
) -> None:
    """Export simulation as heatmap video with specific projection.
    
    Args:
        model: Brain model to visualize.
        output_path: Output video file path.
        num_steps: Number of simulation steps.
        fps: Frames per second.
        resolution: Video resolution.
        projection: Projection type ('xy', 'xz', 'yz', 'xw', etc.).
    """
    def projection_visualization(model: "BrainModel", step: int) -> np.ndarray:
        """Create heatmap visualization with specified projection."""
        width, height = resolution
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        if not model.neurons:
            return frame
        
        # Determine projection axes
        if len(projection) != 2:
            projection_axes = "xy"
        else:
            projection_axes = projection.lower()
        
        axis_map = {'x': 0, 'y': 1, 'z': 2, 'w': 3}
        ax1_name = projection_axes[0]
        ax2_name = projection_axes[1]
        
        # Get coordinates
        coords_1 = []
        coords_2 = []
        activities = []
        
        for neuron in model.neurons.values():
            coord_tuple = (neuron.x, neuron.y, neuron.z, neuron.w)
            
            c1 = coord_tuple[axis_map.get(ax1_name, 0)]
            c2 = coord_tuple[axis_map.get(ax2_name, 1)]
            activity = getattr(neuron, 'v_membrane', 0.0)
            
            coords_1.append(c1)
            coords_2.append(c2)
            activities.append(activity)
        
        if not coords_1:
            return frame
        
        # Normalize coordinates
        min_c1, max_c1 = min(coords_1), max(coords_1)
        min_c2, max_c2 = min(coords_2), max(coords_2)
        
        range_c1 = max_c1 - min_c1 if max_c1 > min_c1 else 1
        range_c2 = max_c2 - min_c2 if max_c2 > min_c2 else 1
        
        # Draw neurons
        for c1, c2, activity in zip(coords_1, coords_2, activities):
            norm_c1 = (c1 - min_c1) / range_c1
            norm_c2 = (c2 - min_c2) / range_c2
            
            pixel_x = int(norm_c1 * (width - 20) + 10)
            pixel_y = int(norm_c2 * (height - 60) + 30)
            
            # Normalize activity
            activity_norm = max(0, min(255, int((activity + 70) / 140 * 255)))
            
            # Color gradient
            if activity_norm < 128:
                color = (0, activity_norm * 2, 255 - activity_norm * 2)
            else:
                color = ((activity_norm - 128) * 2, 255 - (activity_norm - 128) * 2, 0)
            
            cv2.circle(frame, (pixel_x, pixel_y), 3, color, -1)
        
        # Add labels
        cv2.putText(frame, f"Step: {step}", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Projection: {projection_axes}", (10, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    # Record with custom visualization
    with SimulationRecorder(model, output_path, fps, resolution, projection_visualization) as recorder:
        recorder.record_simulation(num_steps)


def create_comparison_video(
    models: List["BrainModel"],
    model_names: List[str],
    output_path: str,
    num_steps: int,
    fps: int = 30,
    resolution: Tuple[int, int] = (1920, 1080)
) -> None:
    """Create side-by-side comparison video of multiple models.
    
    Args:
        models: List of brain models to compare.
        model_names: Names for each model.
        output_path: Output video file path.
        num_steps: Number of simulation steps.
        fps: Frames per second.
        resolution: Total video resolution.
    """
    if len(models) != len(model_names):
        raise ValueError("Number of models must match number of names")
    
    if not models:
        raise ValueError("At least one model required")
    
    # Calculate sub-frame size
    n_models = len(models)
    if n_models == 1:
        grid = (1, 1)
    elif n_models == 2:
        grid = (2, 1)
    elif n_models <= 4:
        grid = (2, 2)
    elif n_models <= 6:
        grid = (3, 2)
    else:
        grid = (3, 3)
    
    sub_width = resolution[0] // grid[0]
    sub_height = resolution[1] // grid[1]
    
    with VideoExporter(output_path, fps, resolution) as exporter:
        for step in range(num_steps):
            # Create combined frame
            combined_frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
            
            for idx, (model, name) in enumerate(zip(models, model_names)):
                # Step model
                if hasattr(model, 'step'):
                    model.step()
                
                # Create sub-frame (simplified visualization)
                sub_frame = np.zeros((sub_height, sub_width, 3), dtype=np.uint8)
                
                # Add model name
                cv2.putText(sub_frame, name, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Place in grid
                grid_x = idx % grid[0]
                grid_y = idx // grid[0]
                
                x_start = grid_x * sub_width
                y_start = grid_y * sub_height
                
                combined_frame[y_start:y_start + sub_height, 
                             x_start:x_start + sub_width] = sub_frame
            
            exporter.add_frame(combined_frame)
