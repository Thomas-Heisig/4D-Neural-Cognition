"""Tests for information theory metrics."""

import pytest
import numpy as np
from src.metrics import (
    calculate_conditional_entropy,
    calculate_transfer_entropy,
    calculate_information_gain,
    calculate_joint_entropy
)


class TestConditionalEntropy:
    """Test conditional entropy calculations."""

    def test_basic_conditional_entropy(self):
        """Test basic conditional entropy calculation."""
        x = [0, 0, 1, 1]
        y = [0, 0, 1, 1]
        
        h_y_given_x = calculate_conditional_entropy(x, y)
        
        # Y is fully determined by X, so H(Y|X) should be 0
        assert h_y_given_x == 0.0

    def test_independent_variables(self):
        """Test conditional entropy with independent variables."""
        x = [0, 1, 0, 1, 0, 1, 0, 1]
        y = [0, 0, 1, 1, 0, 0, 1, 1]
        
        h_y_given_x = calculate_conditional_entropy(x, y)
        
        # Independent variables: H(Y|X) = H(Y)
        assert h_y_given_x > 0

    def test_empty_input(self):
        """Test with empty input."""
        result = calculate_conditional_entropy([], [])
        assert result == 0.0

    def test_mismatched_length(self):
        """Test with mismatched lengths."""
        x = [0, 1, 0]
        y = [0, 1]
        
        result = calculate_conditional_entropy(x, y)
        assert result == 0.0


class TestTransferEntropy:
    """Test transfer entropy calculations."""

    def test_basic_transfer_entropy(self):
        """Test basic transfer entropy."""
        source = [0, 1, 1, 0, 1, 1, 0, 0]
        target = [0, 0, 1, 1, 0, 1, 1, 0]
        
        te = calculate_transfer_entropy(source, target, target_history=1)
        
        # Should return a non-negative value
        assert te >= 0

    def test_no_transfer(self):
        """Test when there's no information transfer."""
        source = [0, 0, 0, 0, 0, 0, 0, 0]
        target = [1, 0, 1, 0, 1, 0, 1, 0]
        
        te = calculate_transfer_entropy(source, target, target_history=1)
        
        # No transfer expected
        assert te >= 0

    def test_perfect_transfer(self):
        """Test with perfect information transfer."""
        source = [0, 1, 0, 1, 0, 1, 0, 1]
        target = [0, 0, 1, 0, 1, 0, 1, 0]  # Delayed copy
        
        te = calculate_transfer_entropy(source, target, target_history=1)
        
        # Should detect strong transfer
        assert te >= 0

    def test_insufficient_data(self):
        """Test with insufficient data."""
        source = [0, 1]
        target = [1, 0]
        
        te = calculate_transfer_entropy(source, target, target_history=1)
        assert te == 0.0

    def test_different_history_lengths(self):
        """Test with different history lengths."""
        source = [0, 1, 1, 0, 1, 1, 0, 0, 1, 0]
        target = [0, 0, 1, 1, 0, 1, 1, 0, 0, 1]
        
        te1 = calculate_transfer_entropy(source, target, target_history=1)
        te2 = calculate_transfer_entropy(source, target, target_history=2)
        
        # Both should be non-negative
        assert te1 >= 0
        assert te2 >= 0


class TestInformationGain:
    """Test information gain calculations."""

    def test_perfect_split(self):
        """Test information gain with perfect split."""
        prior = [50, 50]  # Balanced classes
        posterior = [[50, 0], [0, 50]]  # Perfect separation
        
        ig = calculate_information_gain(prior, posterior)
        
        # Should gain 1 bit of information
        assert ig > 0.9

    def test_no_split(self):
        """Test information gain with no split."""
        prior = [50, 50]
        posterior = [[25, 25], [25, 25]]  # No separation
        
        ig = calculate_information_gain(prior, posterior)
        
        # Should gain no information
        assert abs(ig) < 0.01

    def test_partial_split(self):
        """Test information gain with partial split."""
        prior = [60, 40]
        posterior = [[40, 10], [20, 30]]
        
        ig = calculate_information_gain(prior, posterior)
        
        # Should gain some information
        assert ig > 0

    def test_empty_prior(self):
        """Test with empty prior."""
        prior = [0, 0]
        posterior = [[0, 0], [0, 0]]
        
        ig = calculate_information_gain(prior, posterior)
        assert ig == 0.0


class TestJointEntropy:
    """Test joint entropy calculations."""

    def test_basic_joint_entropy(self):
        """Test basic joint entropy."""
        x = [0, 0, 1, 1]
        y = [0, 1, 0, 1]
        
        h_xy = calculate_joint_entropy(x, y)
        
        # Should be 2 bits for 4 unique pairs
        assert abs(h_xy - 2.0) < 0.01

    def test_identical_variables(self):
        """Test joint entropy with identical variables."""
        x = [0, 1, 0, 1, 0, 1]
        y = [0, 1, 0, 1, 0, 1]
        
        h_xy = calculate_joint_entropy(x, y)
        
        # H(X,X) = H(X)
        assert h_xy > 0

    def test_empty_input(self):
        """Test with empty input."""
        result = calculate_joint_entropy([], [])
        assert result == 0.0

    def test_mismatched_length(self):
        """Test with mismatched lengths."""
        x = [0, 1, 0]
        y = [0, 1]
        
        result = calculate_joint_entropy(x, y)
        assert result == 0.0


class TestInformationTheoryRelationships:
    """Test relationships between information theory metrics."""

    def test_mutual_information_relationship(self):
        """Test I(X;Y) = H(X) + H(Y) - H(X,Y)."""
        from src.metrics import calculate_entropy, calculate_mutual_information
        
        x = [0, 0, 1, 1, 0, 1, 0, 1]
        y = [0, 1, 0, 1, 0, 0, 1, 1]
        
        # Calculate components
        h_x = calculate_entropy([x.count(0), x.count(1)])
        h_y = calculate_entropy([y.count(0), y.count(1)])
        h_xy = calculate_joint_entropy(x, y)
        mi = calculate_mutual_information(x, y)
        
        # Check relationship: I(X;Y) = H(X) + H(Y) - H(X,Y)
        expected_mi = h_x + h_y - h_xy
        assert abs(mi - expected_mi) < 0.01

    def test_conditional_entropy_relationship(self):
        """Test H(Y|X) = H(X,Y) - H(X)."""
        from src.metrics import calculate_entropy
        
        x = [0, 0, 1, 1, 0, 1]
        y = [0, 1, 0, 1, 0, 0]
        
        h_x = calculate_entropy([x.count(0), x.count(1)])
        h_xy = calculate_joint_entropy(x, y)
        h_y_given_x = calculate_conditional_entropy(x, y)
        
        # Check relationship
        expected = h_xy - h_x
        assert abs(h_y_given_x - expected) < 0.01

    def test_chain_rule(self):
        """Test chain rule: H(X,Y) = H(X) + H(Y|X)."""
        from src.metrics import calculate_entropy
        
        x = [0, 1, 0, 1, 0, 1, 0, 1]
        y = [0, 0, 1, 1, 0, 0, 1, 1]
        
        h_x = calculate_entropy([x.count(0), x.count(1)])
        h_y_given_x = calculate_conditional_entropy(x, y)
        h_xy = calculate_joint_entropy(x, y)
        
        # Check chain rule
        expected = h_x + h_y_given_x
        assert abs(h_xy - expected) < 0.01


class TestEdgeCases:
    """Test edge cases for information theory metrics."""

    def test_single_value(self):
        """Test with single value."""
        x = [0]
        y = [1]
        
        h_xy = calculate_joint_entropy(x, y)
        assert h_xy == 0.0

    def test_all_same_value(self):
        """Test with all same values."""
        x = [0, 0, 0, 0]
        y = [1, 1, 1, 1]
        
        h_xy = calculate_joint_entropy(x, y)
        # Only one unique pair (0,1)
        assert h_xy == 0.0

    def test_large_values(self):
        """Test with large integer values."""
        x = [1000, 2000, 1000, 2000]
        y = [3000, 3000, 4000, 4000]
        
        h_xy = calculate_joint_entropy(x, y)
        assert h_xy >= 0

    def test_negative_values(self):
        """Test with negative values."""
        x = [-1, -2, -1, -2]
        y = [1, 2, 1, 2]
        
        h_xy = calculate_joint_entropy(x, y)
        assert h_xy >= 0
