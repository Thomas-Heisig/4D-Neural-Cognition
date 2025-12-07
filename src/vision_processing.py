"""Advanced vision processing module for enhanced sensory input.

This module provides sophisticated vision processing capabilities including:
- Edge detection preprocessing
- Color processing (RGB channels)
- Motion detection
- Multi-scale processing
"""

import numpy as np
from typing import Tuple, Optional, List


class EdgeDetector:
    """Edge detection preprocessing for visual input."""

    @staticmethod
    def sobel_edge_detection(image: np.ndarray) -> np.ndarray:
        """Apply Sobel edge detection to an image.

        Args:
            image: 2D numpy array representing grayscale image.

        Returns:
            Edge-detected image as 2D numpy array.
        """
        if image.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {image.shape}")

        # Sobel kernels for x and y directions
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # Apply convolution
        grad_x = EdgeDetector._convolve2d(image, sobel_x)
        grad_y = EdgeDetector._convolve2d(image, sobel_y)

        # Compute gradient magnitude
        edges = np.sqrt(grad_x**2 + grad_y**2)
        return edges

    @staticmethod
    def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Simple 2D convolution implementation.

        Args:
            image: Input 2D array.
            kernel: Convolution kernel.

        Returns:
            Convolved image.
        """
        kh, kw = kernel.shape
        ih, iw = image.shape
        pad_h, pad_w = kh // 2, kw // 2

        # Pad image
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

        # Initialize output
        output = np.zeros_like(image)

        # Perform convolution
        for i in range(ih):
            for j in range(iw):
                output[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)

        return output

    @staticmethod
    def laplacian_edge_detection(image: np.ndarray) -> np.ndarray:
        """Apply Laplacian edge detection to an image.

        Args:
            image: 2D numpy array representing grayscale image.

        Returns:
            Edge-detected image as 2D numpy array.
        """
        if image.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {image.shape}")

        # Laplacian kernel
        laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

        # Apply convolution
        edges = EdgeDetector._convolve2d(image, laplacian)
        return np.abs(edges)


class ColorProcessor:
    """RGB color processing for vision input."""

    @staticmethod
    def split_rgb_channels(rgb_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split RGB image into separate channels.

        Args:
            rgb_image: 3D numpy array with shape (height, width, 3).

        Returns:
            Tuple of (red_channel, green_channel, blue_channel).
        """
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape (H, W, 3), got {rgb_image.shape}")

        return rgb_image[:, :, 0], rgb_image[:, :, 1], rgb_image[:, :, 2]

    @staticmethod
    def rgb_to_grayscale(rgb_image: np.ndarray) -> np.ndarray:
        """Convert RGB image to grayscale using standard weights.

        Args:
            rgb_image: 3D numpy array with shape (height, width, 3).

        Returns:
            Grayscale image as 2D numpy array.
        """
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape (H, W, 3), got {rgb_image.shape}")

        # Standard grayscale conversion weights
        weights = np.array([0.299, 0.587, 0.114])
        grayscale = np.dot(rgb_image, weights)
        return grayscale

    @staticmethod
    def normalize_channel(channel: np.ndarray, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
        """Normalize channel values to specified range.

        Args:
            channel: 2D numpy array representing image channel.
            min_val: Minimum value for normalization.
            max_val: Maximum value for normalization.

        Returns:
            Normalized channel.
        """
        channel_min = np.min(channel)
        channel_max = np.max(channel)

        if channel_max == channel_min:
            return np.full_like(channel, min_val)

        normalized = (channel - channel_min) / (channel_max - channel_min)
        normalized = normalized * (max_val - min_val) + min_val
        return normalized


class MotionDetector:
    """Motion detection for temporal visual input."""

    def __init__(self, history_size: int = 3):
        """Initialize motion detector.

        Args:
            history_size: Number of frames to keep in history.
        """
        self.history_size = history_size
        self.frame_history: List[np.ndarray] = []

    def add_frame(self, frame: np.ndarray) -> None:
        """Add a new frame to the history.

        Args:
            frame: 2D numpy array representing a video frame.
        """
        if frame.ndim != 2:
            raise ValueError(f"Expected 2D frame, got shape {frame.shape}")

        self.frame_history.append(frame.copy())

        # Keep only recent frames
        if len(self.frame_history) > self.history_size:
            self.frame_history.pop(0)

    def detect_motion(self) -> Optional[np.ndarray]:
        """Detect motion between recent frames.

        Returns:
            Motion map as 2D numpy array, or None if insufficient history.
        """
        if len(self.frame_history) < 2:
            return None

        # Simple frame differencing
        current = self.frame_history[-1]
        previous = self.frame_history[-2]

        motion = np.abs(current - previous)
        return motion

    def detect_optical_flow_simple(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Simple optical flow estimation using frame differences.

        Returns:
            Tuple of (flow_x, flow_y) arrays, or None if insufficient history.
        """
        if len(self.frame_history) < 2:
            return None

        current = self.frame_history[-1]
        previous = self.frame_history[-2]

        # Simple gradient-based flow estimation
        dt = current - previous
        dx = np.gradient(current, axis=1)
        dy = np.gradient(current, axis=0)

        # Avoid division by zero
        denominator = dx**2 + dy**2 + 1e-8

        flow_x = -(dx * dt) / denominator
        flow_y = -(dy * dt) / denominator

        return flow_x, flow_y

    def reset(self) -> None:
        """Clear frame history."""
        self.frame_history.clear()


class MultiScaleProcessor:
    """Multi-scale image processing for hierarchical feature extraction."""

    @staticmethod
    def create_gaussian_pyramid(image: np.ndarray, levels: int = 3) -> List[np.ndarray]:
        """Create Gaussian pyramid for multi-scale processing.

        Args:
            image: 2D numpy array representing image.
            levels: Number of pyramid levels.

        Returns:
            List of images at different scales.
        """
        if image.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {image.shape}")

        if levels < 1:
            raise ValueError(f"levels must be >= 1, got {levels}")

        pyramid = [image]

        for _ in range(levels - 1):
            # Apply Gaussian blur
            blurred = MultiScaleProcessor._gaussian_blur(pyramid[-1])
            # Downsample by factor of 2
            downsampled = blurred[::2, ::2]
            pyramid.append(downsampled)

        return pyramid

    @staticmethod
    def _gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Apply Gaussian blur to image.

        Args:
            image: 2D numpy array.
            kernel_size: Size of Gaussian kernel.

        Returns:
            Blurred image.
        """
        # Create 1D Gaussian kernel
        sigma = kernel_size / 6.0
        x = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
        kernel_1d = np.exp(-(x**2) / (2 * sigma**2))
        kernel_1d = kernel_1d / np.sum(kernel_1d)

        # Create 2D kernel from outer product
        kernel_2d = np.outer(kernel_1d, kernel_1d)

        # Apply convolution
        return EdgeDetector._convolve2d(image, kernel_2d)

    @staticmethod
    def create_laplacian_pyramid(image: np.ndarray, levels: int = 3) -> List[np.ndarray]:
        """Create Laplacian pyramid for multi-scale processing.

        Args:
            image: 2D numpy array representing image.
            levels: Number of pyramid levels.

        Returns:
            List of Laplacian images at different scales.
        """
        gaussian_pyramid = MultiScaleProcessor.create_gaussian_pyramid(image, levels)
        laplacian_pyramid = []

        for i in range(len(gaussian_pyramid) - 1):
            # Compute Laplacian as difference between consecutive Gaussian levels
            current = gaussian_pyramid[i]
            next_level = gaussian_pyramid[i + 1]

            # Upsample next level to match current size
            upsampled = MultiScaleProcessor._upsample(next_level, current.shape)

            # Compute Laplacian
            laplacian = current - upsampled
            laplacian_pyramid.append(laplacian)

        # Add the smallest Gaussian level
        laplacian_pyramid.append(gaussian_pyramid[-1])

        return laplacian_pyramid

    @staticmethod
    def _upsample(image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Upsample image to target shape using nearest neighbor interpolation.

        Args:
            image: 2D numpy array.
            target_shape: Desired output shape (height, width).

        Returns:
            Upsampled image.
        """
        h_src, w_src = image.shape
        h_tgt, w_tgt = target_shape

        upsampled = np.zeros(target_shape)

        for i in range(h_tgt):
            for j in range(w_tgt):
                src_i = min(i // 2, h_src - 1)
                src_j = min(j // 2, w_src - 1)
                upsampled[i, j] = image[src_i, src_j]

        return upsampled


def preprocess_vision_input(
    image: np.ndarray,
    apply_edge_detection: bool = False,
    apply_motion_detection: bool = False,
    motion_detector: Optional[MotionDetector] = None,
    multi_scale_levels: Optional[int] = None,
) -> np.ndarray:
    """Preprocess visual input with various enhancement techniques.

    Args:
        image: Input image as 2D or 3D numpy array.
        apply_edge_detection: Whether to apply edge detection.
        apply_motion_detection: Whether to apply motion detection.
        motion_detector: MotionDetector instance for motion detection.
        multi_scale_levels: If set, extract features at multiple scales.

    Returns:
        Preprocessed image ready for neural input.
    """
    # Convert RGB to grayscale if needed
    if image.ndim == 3 and image.shape[2] == 3:
        processed = ColorProcessor.rgb_to_grayscale(image)
    else:
        processed = image.copy()

    # Apply edge detection
    if apply_edge_detection:
        processed = EdgeDetector.sobel_edge_detection(processed)

    # Apply motion detection
    if apply_motion_detection and motion_detector is not None:
        motion_detector.add_frame(processed)
        motion = motion_detector.detect_motion()
        if motion is not None:
            processed = motion

    # Multi-scale processing
    if multi_scale_levels is not None and multi_scale_levels > 1:
        pyramid = MultiScaleProcessor.create_gaussian_pyramid(processed, multi_scale_levels)
        # Return the middle level as a compromise
        processed = pyramid[len(pyramid) // 2]

    # Normalize to [0, 1] range
    processed = ColorProcessor.normalize_channel(processed, 0.0, 1.0)

    return processed
