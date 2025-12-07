#!/usr/bin/env python3
"""Advanced Pattern Recognition Example with Enhanced Vision Processing

This example demonstrates:
- Edge detection preprocessing
- Multi-scale processing
- Motion detection
- Pattern classification with the enhanced vision module
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from brain_model import BrainModel
from simulation import Simulation
from senses import feed_sense_input
from vision_processing import (
    EdgeDetector,
    ColorProcessor,
    MotionDetector,
    MultiScaleProcessor,
    preprocess_vision_input
)


def create_test_patterns():
    """Create diverse test patterns for recognition."""
    patterns = {}
    
    # Simple shapes
    circle = np.zeros((20, 20))
    y, x = np.ogrid[-10:10, -10:10]
    mask = x**2 + y**2 <= 36
    circle[mask] = 1.0
    patterns['circle'] = circle
    
    # Square
    square = np.zeros((20, 20))
    square[5:15, 5:15] = 1.0
    patterns['square'] = square
    
    # Triangle (simple approximation)
    triangle = np.zeros((20, 20))
    for i in range(10):
        start = 10 - i
        end = 10 + i
        triangle[15-i, start:end+1] = 1.0
    patterns['triangle'] = triangle
    
    # Cross pattern
    cross = np.zeros((20, 20))
    cross[10, :] = 1.0
    cross[:, 10] = 1.0
    patterns['cross'] = cross
    
    # Random noise
    noise = np.random.rand(20, 20)
    patterns['noise'] = noise
    
    return patterns


def create_rgb_patterns():
    """Create RGB colored patterns."""
    patterns = {}
    
    # Red square
    red = np.zeros((20, 20, 3))
    red[5:15, 5:15, 0] = 1.0
    patterns['red_square'] = red
    
    # Green circle
    green = np.zeros((20, 20, 3))
    y, x = np.ogrid[-10:10, -10:10]
    mask = x**2 + y**2 <= 36
    green[mask, 1] = 1.0
    patterns['green_circle'] = green
    
    # Blue triangle
    blue = np.zeros((20, 20, 3))
    for i in range(10):
        start = 10 - i
        end = 10 + i
        blue[15-i, start:end+1, 2] = 1.0
    patterns['blue_triangle'] = blue
    
    return patterns


def demonstrate_edge_detection():
    """Demonstrate edge detection capabilities."""
    print("\n" + "="*60)
    print("EDGE DETECTION DEMO")
    print("="*60)
    
    patterns = create_test_patterns()
    
    for name, pattern in patterns.items():
        print(f"\n{name}:")
        
        # Apply Sobel edge detection
        edges_sobel = EdgeDetector.sobel_edge_detection(pattern)
        edge_strength = np.mean(edges_sobel)
        print(f"  Sobel edge strength: {edge_strength:.3f}")
        
        # Apply Laplacian edge detection
        edges_laplacian = EdgeDetector.laplacian_edge_detection(pattern)
        laplacian_strength = np.mean(edges_laplacian)
        print(f"  Laplacian edge strength: {laplacian_strength:.3f}")


def demonstrate_color_processing():
    """Demonstrate RGB color processing."""
    print("\n" + "="*60)
    print("COLOR PROCESSING DEMO")
    print("="*60)
    
    patterns = create_rgb_patterns()
    
    for name, rgb_pattern in patterns.items():
        print(f"\n{name}:")
        
        # Split channels
        r, g, b = ColorProcessor.split_rgb_channels(rgb_pattern)
        print(f"  Red channel mean: {np.mean(r):.3f}")
        print(f"  Green channel mean: {np.mean(g):.3f}")
        print(f"  Blue channel mean: {np.mean(b):.3f}")
        
        # Convert to grayscale
        gray = ColorProcessor.rgb_to_grayscale(rgb_pattern)
        print(f"  Grayscale mean: {np.mean(gray):.3f}")


def demonstrate_motion_detection():
    """Demonstrate motion detection across frames."""
    print("\n" + "="*60)
    print("MOTION DETECTION DEMO")
    print("="*60)
    
    detector = MotionDetector(history_size=3)
    
    # Create moving square
    frames = []
    for i in range(5):
        frame = np.zeros((20, 20))
        x_pos = 5 + i * 2
        frame[5:10, x_pos:x_pos+5] = 1.0
        frames.append(frame)
    
    print("\nDetecting motion in moving square...")
    for i, frame in enumerate(frames):
        detector.add_frame(frame)
        motion = detector.detect_motion()
        
        if motion is not None:
            motion_magnitude = np.mean(motion)
            print(f"  Frame {i}: motion magnitude = {motion_magnitude:.3f}")
    
    # Optical flow
    if len(detector.frame_history) >= 2:
        flow_x, flow_y = detector.detect_optical_flow_simple()
        print(f"\nOptical flow detected:")
        print(f"  Mean flow X: {np.mean(flow_x):.3f}")
        print(f"  Mean flow Y: {np.mean(flow_y):.3f}")


def demonstrate_multiscale_processing():
    """Demonstrate multi-scale pyramid processing."""
    print("\n" + "="*60)
    print("MULTI-SCALE PROCESSING DEMO")
    print("="*60)
    
    # Create a complex pattern
    pattern = np.zeros((32, 32))
    pattern[10:22, 10:22] = 1.0
    pattern[14:18, 14:18] = 0.0
    
    # Gaussian pyramid
    print("\nGaussian Pyramid:")
    gaussian_pyramid = MultiScaleProcessor.create_gaussian_pyramid(pattern, levels=4)
    for i, level in enumerate(gaussian_pyramid):
        print(f"  Level {i}: shape {level.shape}, mean {np.mean(level):.3f}")
    
    # Laplacian pyramid
    print("\nLaplacian Pyramid:")
    laplacian_pyramid = MultiScaleProcessor.create_laplacian_pyramid(pattern, levels=4)
    for i, level in enumerate(laplacian_pyramid):
        print(f"  Level {i}: shape {level.shape}, mean {np.mean(level):.3f}")


def train_with_enhanced_vision(sim, patterns, use_edge_detection=True):
    """Train network using enhanced vision preprocessing.
    
    Args:
        sim: Simulation instance
        patterns: Dictionary of patterns
        use_edge_detection: Whether to apply edge detection
    """
    print("\n" + "="*60)
    print("TRAINING WITH ENHANCED VISION")
    print("="*60)
    
    for epoch in range(3):
        print(f"\nEpoch {epoch+1}/3")
        
        for name, pattern in patterns.items():
            # Preprocess with vision enhancements
            if pattern.ndim == 3:
                # RGB image
                processed = ColorProcessor.rgb_to_grayscale(pattern)
            else:
                processed = pattern
            
            if use_edge_detection:
                processed = EdgeDetector.sobel_edge_detection(processed)
            
            # Normalize
            processed = ColorProcessor.normalize_channel(processed, 0.0, 1.0)
            
            # Feed to network
            feed_sense_input(sim.model, sense_name="vision", input_matrix=processed)
            
            # Process and learn
            for step in range(20):
                sim.step()
                if step % 5 == 0:
                    sim.apply_plasticity()
            
            # Rest period
            for step in range(5):
                sim.step()


def test_enhanced_recognition(sim, patterns, use_edge_detection=True):
    """Test pattern recognition with enhanced preprocessing."""
    print("\n" + "="*60)
    print("TESTING WITH ENHANCED VISION")
    print("="*60)
    
    results = {}
    
    for name, pattern in patterns.items():
        # Preprocess
        if pattern.ndim == 3:
            processed = ColorProcessor.rgb_to_grayscale(pattern)
        else:
            processed = pattern
        
        if use_edge_detection:
            processed = EdgeDetector.sobel_edge_detection(processed)
        
        processed = ColorProcessor.normalize_channel(processed, 0.0, 1.0)
        
        # Present to network
        feed_sense_input(sim.model, sense_name="vision", input_matrix=processed)
        
        # Measure response
        spike_counts = []
        for step in range(20):
            spiked = sim.step()
            spike_counts.append(len(spiked))
        
        total_spikes = sum(spike_counts)
        results[name] = total_spikes
        
        print(f"\n{name}: {total_spikes} total spikes")
        
        # Clear
        for step in range(5):
            sim.step()
    
    return results


def main():
    """Run the advanced pattern recognition example."""
    print("="*60)
    print("ADVANCED PATTERN RECOGNITION")
    print("Enhanced Vision Processing Demo")
    print("="*60)
    
    # Demonstrate individual components
    demonstrate_edge_detection()
    demonstrate_color_processing()
    demonstrate_motion_detection()
    demonstrate_multiscale_processing()
    
    # Setup neural network
    print("\n" + "="*60)
    print("NEURAL NETWORK SETUP")
    print("="*60)
    
    model = BrainModel(config_path=str(Path(__file__).parent.parent / "brain_base_model.json"))
    sim = Simulation(model, seed=42)
    
    print("\nInitializing network...")
    sim.initialize_neurons(area_names=["V1_like"], density=0.1)
    sim.initialize_random_synapses(
        connection_probability=0.1,
        weight_mean=0.5,
        weight_std=0.1
    )
    
    print(f"  Neurons: {len(sim.model.neurons)}")
    print(f"  Synapses: {len(sim.model.synapses)}")
    
    # Create test patterns
    patterns = create_test_patterns()
    
    # Train and test
    train_with_enhanced_vision(sim, patterns, use_edge_detection=True)
    results = test_enhanced_recognition(sim, patterns, use_edge_detection=True)
    
    # Analysis
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    print("\nPatterns ranked by neural response:")
    for i, (name, spikes) in enumerate(sorted_results, 1):
        print(f"  {i}. {name}: {spikes} spikes")
    
    print("\n" + "="*60)
    print("EXAMPLE COMPLETE")
    print("="*60)
    print("\nKey Features Demonstrated:")
    print("  ✓ Edge detection preprocessing")
    print("  ✓ RGB color processing")
    print("  ✓ Motion detection")
    print("  ✓ Multi-scale pyramid processing")
    print("  ✓ Enhanced pattern recognition")
    
    print("\nNext Steps:")
    print("  • Experiment with different preprocessing combinations")
    print("  • Try real images instead of synthetic patterns")
    print("  • Combine color and edge information")
    print("  • Use motion detection for temporal learning")


if __name__ == "__main__":
    main()
