"""Tests for vision processing module."""

import numpy as np
import pytest

from src.vision_processing import (
    EdgeDetector,
    ColorProcessor,
    MotionDetector,
    MultiScaleProcessor,
    preprocess_vision_input,
)


class TestEdgeDetector:
    """Tests for EdgeDetector class."""
    
    def test_sobel_edge_detection_basic(self):
        """Test basic Sobel edge detection."""
        # Create a simple test image with a vertical edge
        image = np.zeros((10, 10))
        image[:, 5:] = 1.0
        
        edges = EdgeDetector.sobel_edge_detection(image)
        
        assert edges.shape == image.shape
        # Edges should be stronger around the boundary
        assert np.max(edges) > 0
    
    def test_sobel_edge_detection_wrong_dimensions(self):
        """Test Sobel with wrong dimensions."""
        image_3d = np.zeros((10, 10, 3))
        
        with pytest.raises(ValueError, match="Expected 2D"):
            EdgeDetector.sobel_edge_detection(image_3d)
    
    def test_sobel_edge_detection_uniform(self):
        """Test Sobel on uniform image."""
        image = np.ones((10, 10))
        
        edges = EdgeDetector.sobel_edge_detection(image)
        
        # Uniform image should have minimal edges
        assert np.max(edges) < 0.1
    
    def test_laplacian_edge_detection_basic(self):
        """Test basic Laplacian edge detection."""
        # Create a simple test image
        image = np.zeros((10, 10))
        image[4:6, 4:6] = 1.0
        
        edges = EdgeDetector.laplacian_edge_detection(image)
        
        assert edges.shape == image.shape
        assert np.max(edges) > 0
    
    def test_laplacian_edge_detection_wrong_dimensions(self):
        """Test Laplacian with wrong dimensions."""
        image_3d = np.zeros((10, 10, 3))
        
        with pytest.raises(ValueError, match="Expected 2D"):
            EdgeDetector.laplacian_edge_detection(image_3d)
    
    def test_convolve2d_identity(self):
        """Test convolution with identity kernel."""
        image = np.random.rand(5, 5)
        identity_kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        
        result = EdgeDetector._convolve2d(image, identity_kernel)
        
        # Should be approximately the same as input
        assert np.allclose(result, image, atol=0.01)
    
    def test_convolve2d_small_image(self):
        """Test convolution on small image."""
        image = np.ones((3, 3))
        kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
        
        result = EdgeDetector._convolve2d(image, kernel)
        
        assert result.shape == image.shape


class TestColorProcessor:
    """Tests for ColorProcessor class."""
    
    def test_split_rgb_channels(self):
        """Test splitting RGB channels."""
        rgb_image = np.random.rand(10, 10, 3)
        
        r, g, b = ColorProcessor.split_rgb_channels(rgb_image)
        
        assert r.shape == (10, 10)
        assert g.shape == (10, 10)
        assert b.shape == (10, 10)
        assert np.array_equal(r, rgb_image[:, :, 0])
        assert np.array_equal(g, rgb_image[:, :, 1])
        assert np.array_equal(b, rgb_image[:, :, 2])
    
    def test_split_rgb_channels_wrong_shape(self):
        """Test splitting with wrong shape."""
        # Wrong number of channels
        image = np.random.rand(10, 10, 4)
        
        with pytest.raises(ValueError, match="Expected RGB"):
            ColorProcessor.split_rgb_channels(image)
        
        # 2D image
        image_2d = np.random.rand(10, 10)
        
        with pytest.raises(ValueError, match="Expected RGB"):
            ColorProcessor.split_rgb_channels(image_2d)
    
    def test_rgb_to_grayscale(self):
        """Test RGB to grayscale conversion."""
        # Create a pure red image
        rgb_image = np.zeros((10, 10, 3))
        rgb_image[:, :, 0] = 1.0
        
        grayscale = ColorProcessor.rgb_to_grayscale(rgb_image)
        
        assert grayscale.shape == (10, 10)
        # Red channel weight is 0.299
        assert np.allclose(grayscale, 0.299, atol=0.01)
    
    def test_rgb_to_grayscale_wrong_shape(self):
        """Test grayscale conversion with wrong shape."""
        image = np.random.rand(10, 10)
        
        with pytest.raises(ValueError, match="Expected RGB"):
            ColorProcessor.rgb_to_grayscale(image)
    
    def test_normalize_channel_basic(self):
        """Test basic channel normalization."""
        channel = np.array([[0, 50, 100], [150, 200, 255]], dtype=float)
        
        normalized = ColorProcessor.normalize_channel(channel, 0.0, 1.0)
        
        assert np.min(normalized) >= 0.0
        assert np.max(normalized) <= 1.0
        assert np.isclose(np.min(normalized), 0.0)
        assert np.isclose(np.max(normalized), 1.0)
    
    def test_normalize_channel_custom_range(self):
        """Test normalization to custom range."""
        channel = np.array([[0, 50, 100]], dtype=float)
        
        normalized = ColorProcessor.normalize_channel(channel, -1.0, 1.0)
        
        assert np.min(normalized) >= -1.0
        assert np.max(normalized) <= 1.0
    
    def test_normalize_channel_uniform(self):
        """Test normalizing uniform channel."""
        channel = np.ones((5, 5)) * 42.0
        
        normalized = ColorProcessor.normalize_channel(channel, 0.0, 1.0)
        
        # All values should be min_val
        assert np.allclose(normalized, 0.0)


class TestMotionDetector:
    """Tests for MotionDetector class."""
    
    def test_init(self):
        """Test initialization."""
        detector = MotionDetector(history_size=5)
        
        assert detector.history_size == 5
        assert len(detector.frame_history) == 0
    
    def test_add_frame(self):
        """Test adding frames."""
        detector = MotionDetector(history_size=3)
        frame = np.random.rand(10, 10)
        
        detector.add_frame(frame)
        
        assert len(detector.frame_history) == 1
        assert np.array_equal(detector.frame_history[0], frame)
    
    def test_add_frame_wrong_dimensions(self):
        """Test adding frame with wrong dimensions."""
        detector = MotionDetector()
        frame_3d = np.random.rand(10, 10, 3)
        
        with pytest.raises(ValueError, match="Expected 2D"):
            detector.add_frame(frame_3d)
    
    def test_add_frame_history_limit(self):
        """Test that history is limited to history_size."""
        detector = MotionDetector(history_size=2)
        
        for i in range(5):
            frame = np.ones((5, 5)) * i
            detector.add_frame(frame)
        
        assert len(detector.frame_history) == 2
        # Should keep only the last 2 frames (3 and 4)
        assert np.allclose(detector.frame_history[0], 3.0)
        assert np.allclose(detector.frame_history[1], 4.0)
    
    def test_detect_motion_insufficient_history(self):
        """Test motion detection with insufficient history."""
        detector = MotionDetector()
        
        motion = detector.detect_motion()
        
        assert motion is None
    
    def test_detect_motion_basic(self):
        """Test basic motion detection."""
        detector = MotionDetector()
        
        # Add two frames with different content
        frame1 = np.zeros((10, 10))
        frame2 = np.ones((10, 10))
        
        detector.add_frame(frame1)
        detector.add_frame(frame2)
        
        motion = detector.detect_motion()
        
        assert motion is not None
        assert motion.shape == (10, 10)
        # Motion should be detected (difference is 1.0)
        assert np.allclose(motion, 1.0)
    
    def test_detect_motion_no_motion(self):
        """Test motion detection with no motion."""
        detector = MotionDetector()
        
        frame = np.ones((10, 10))
        detector.add_frame(frame)
        detector.add_frame(frame)
        
        motion = detector.detect_motion()
        
        assert motion is not None
        # No motion should be detected
        assert np.allclose(motion, 0.0)
    
    def test_detect_optical_flow_insufficient_history(self):
        """Test optical flow with insufficient history."""
        detector = MotionDetector()
        
        flow = detector.detect_optical_flow_simple()
        
        assert flow is None
    
    def test_detect_optical_flow_basic(self):
        """Test basic optical flow detection."""
        detector = MotionDetector()
        
        frame1 = np.random.rand(10, 10)
        frame2 = np.random.rand(10, 10)
        
        detector.add_frame(frame1)
        detector.add_frame(frame2)
        
        flow = detector.detect_optical_flow_simple()
        
        assert flow is not None
        flow_x, flow_y = flow
        assert flow_x.shape == (10, 10)
        assert flow_y.shape == (10, 10)
    
    def test_reset(self):
        """Test resetting detector."""
        detector = MotionDetector()
        
        detector.add_frame(np.random.rand(5, 5))
        detector.add_frame(np.random.rand(5, 5))
        
        detector.reset()
        
        assert len(detector.frame_history) == 0


class TestMultiScaleProcessor:
    """Tests for MultiScaleProcessor class."""
    
    def test_create_gaussian_pyramid_basic(self):
        """Test basic Gaussian pyramid creation."""
        image = np.random.rand(32, 32)
        
        pyramid = MultiScaleProcessor.create_gaussian_pyramid(image, levels=3)
        
        assert len(pyramid) == 3
        assert pyramid[0].shape == (32, 32)
        assert pyramid[1].shape == (16, 16)
        assert pyramid[2].shape == (8, 8)
    
    def test_create_gaussian_pyramid_wrong_dimensions(self):
        """Test pyramid with wrong dimensions."""
        image_3d = np.random.rand(10, 10, 3)
        
        with pytest.raises(ValueError, match="Expected 2D"):
            MultiScaleProcessor.create_gaussian_pyramid(image_3d, levels=3)
    
    def test_create_gaussian_pyramid_invalid_levels(self):
        """Test pyramid with invalid levels."""
        image = np.random.rand(10, 10)
        
        with pytest.raises(ValueError, match="levels must be >= 1"):
            MultiScaleProcessor.create_gaussian_pyramid(image, levels=0)
    
    def test_create_gaussian_pyramid_single_level(self):
        """Test pyramid with single level."""
        image = np.random.rand(10, 10)
        
        pyramid = MultiScaleProcessor.create_gaussian_pyramid(image, levels=1)
        
        assert len(pyramid) == 1
        assert np.array_equal(pyramid[0], image)
    
    def test_gaussian_blur(self):
        """Test Gaussian blur."""
        image = np.zeros((10, 10))
        image[5, 5] = 1.0  # Single bright point
        
        blurred = MultiScaleProcessor._gaussian_blur(image)
        
        assert blurred.shape == image.shape
        # Center should still be brightest
        assert blurred[5, 5] == np.max(blurred)
        # Blur should spread to neighbors
        assert blurred[4, 5] > 0
        assert blurred[6, 5] > 0
    
    def test_create_laplacian_pyramid_basic(self):
        """Test basic Laplacian pyramid creation."""
        image = np.random.rand(32, 32)
        
        pyramid = MultiScaleProcessor.create_laplacian_pyramid(image, levels=3)
        
        assert len(pyramid) == 3
        # First levels are differences, last is the smallest Gaussian
        assert pyramid[0].shape == (32, 32)
        assert pyramid[1].shape == (16, 16)
    
    def test_upsample_basic(self):
        """Test basic upsampling."""
        small_image = np.ones((4, 4))
        
        upsampled = MultiScaleProcessor._upsample(small_image, (8, 8))
        
        assert upsampled.shape == (8, 8)
        # Should maintain values (nearest neighbor)
        assert np.allclose(upsampled, 1.0)
    
    def test_upsample_to_same_size(self):
        """Test upsampling to same size."""
        image = np.random.rand(5, 5)
        
        upsampled = MultiScaleProcessor._upsample(image, (5, 5))
        
        # Should have correct shape
        assert upsampled.shape == (5, 5)
        # With nearest neighbor upsampling from same size, values should be replicated
        assert upsampled[0, 0] == image[0, 0]


class TestPreprocessVisionInput:
    """Tests for preprocess_vision_input function."""
    
    def test_preprocess_grayscale_basic(self):
        """Test preprocessing grayscale image."""
        image = np.random.rand(10, 10)
        
        processed = preprocess_vision_input(image)
        
        assert processed.shape == image.shape
        # Should be normalized to [0, 1]
        assert np.min(processed) >= 0.0
        assert np.max(processed) <= 1.0
    
    def test_preprocess_rgb_to_grayscale(self):
        """Test preprocessing RGB image."""
        rgb_image = np.random.rand(10, 10, 3)
        
        processed = preprocess_vision_input(rgb_image)
        
        # Should be converted to grayscale
        assert processed.ndim == 2
        assert processed.shape == (10, 10)
    
    def test_preprocess_with_edge_detection(self):
        """Test preprocessing with edge detection."""
        image = np.zeros((10, 10))
        image[:, 5:] = 1.0
        
        processed = preprocess_vision_input(image, apply_edge_detection=True)
        
        assert processed.shape == image.shape
        # Edges should be detected
        assert np.max(processed) > 0
    
    def test_preprocess_with_motion_detection(self):
        """Test preprocessing with motion detection."""
        detector = MotionDetector()
        image1 = np.zeros((10, 10))
        image2 = np.ones((10, 10))
        
        # First frame
        preprocess_vision_input(image1, apply_motion_detection=True, motion_detector=detector)
        
        # Second frame (with motion)
        processed = preprocess_vision_input(image2, apply_motion_detection=True, motion_detector=detector)
        
        assert processed.shape == image2.shape
    
    def test_preprocess_with_motion_detection_no_detector(self):
        """Test motion detection without detector."""
        image = np.random.rand(10, 10)
        
        # Should not crash
        processed = preprocess_vision_input(image, apply_motion_detection=True, motion_detector=None)
        
        assert processed.shape == image.shape
    
    def test_preprocess_with_multi_scale(self):
        """Test preprocessing with multi-scale."""
        image = np.random.rand(32, 32)
        
        processed = preprocess_vision_input(image, multi_scale_levels=3)
        
        # Should return middle level (16x16 for 3 levels)
        assert processed.shape == (16, 16)
    
    def test_preprocess_all_features(self):
        """Test preprocessing with all features enabled."""
        detector = MotionDetector()
        rgb_image = np.random.rand(32, 32, 3)
        
        # Add frame to detector first
        gray = ColorProcessor.rgb_to_grayscale(rgb_image)
        detector.add_frame(gray)
        
        processed = preprocess_vision_input(
            rgb_image,
            apply_edge_detection=True,
            apply_motion_detection=True,
            motion_detector=detector,
            multi_scale_levels=2
        )
        
        assert processed.ndim == 2
        assert np.min(processed) >= 0.0
        assert np.max(processed) <= 1.0
