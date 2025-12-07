"""Tests for learning systems module."""

import pytest
import numpy as np
from src.learning_systems import (
    LearningCategory,
    LearningContext,
    LearningResult,
    AssociativeLearning,
    NonAssociativeLearning,
    OperantConditioning,
    SupervisedLearning,
    UnsupervisedLearning,
    ReinforcementLearning,
    TransferLearning,
    MetaLearning,
    LearningSystemManager,
    create_default_learning_systems,
)


class TestLearningContext:
    """Tests for LearningContext dataclass."""
    
    def test_default_initialization(self):
        """Test default initialization of LearningContext."""
        context = LearningContext()
        assert context.timestep == 0
        assert context.environment_state == {}
        assert context.internal_state == {}
        assert context.metadata == {}
    
    def test_custom_initialization(self):
        """Test custom initialization of LearningContext."""
        context = LearningContext(
            timestep=10,
            environment_state={"temp": 25},
            internal_state={"energy": 0.8},
            metadata={"session": "test"}
        )
        assert context.timestep == 10
        assert context.environment_state == {"temp": 25}
        assert context.internal_state == {"energy": 0.8}
        assert context.metadata == {"session": "test"}


class TestLearningResult:
    """Tests for LearningResult dataclass."""
    
    def test_default_initialization(self):
        """Test default initialization of LearningResult."""
        result = LearningResult()
        assert result.success is False
        assert result.learning_delta == 0.0
        assert result.updated_parameters == {}
        assert result.metrics == {}
        assert result.feedback == ""
    
    def test_custom_initialization(self):
        """Test custom initialization of LearningResult."""
        result = LearningResult(
            success=True,
            learning_delta=0.5,
            updated_parameters={"weights": [1, 2, 3]},
            metrics={"accuracy": 0.9},
            feedback="Learning successful"
        )
        assert result.success is True
        assert result.learning_delta == 0.5
        assert result.updated_parameters == {"weights": [1, 2, 3]}
        assert result.metrics == {"accuracy": 0.9}
        assert result.feedback == "Learning successful"


class TestAssociativeLearning:
    """Tests for AssociativeLearning system."""
    
    def test_initialization(self):
        """Test initialization of AssociativeLearning."""
        system = AssociativeLearning()
        assert system.name == "Associative Learning"
        assert system.category == LearningCategory.BIOLOGICAL
        assert not system.is_active
        assert len(system.associations) == 0
    
    def test_learn_valid_association(self):
        """Test learning a valid association."""
        system = AssociativeLearning()
        context = LearningContext(timestep=1)
        data = {
            "stimulus_a": "bell",
            "stimulus_b": "food",
            "strength": 1.0
        }
        
        result = system.learn(context, data)
        
        assert result.success is True
        assert result.learning_delta > 0
        assert ("bell", "food") in system.associations
        assert system.associations[("bell", "food")] > 0
    
    def test_learn_invalid_data(self):
        """Test learning with invalid data."""
        system = AssociativeLearning()
        context = LearningContext()
        data = {"stimulus_a": "bell"}  # Missing stimulus_b
        
        result = system.learn(context, data)
        
        assert result.success is False
        assert "Invalid data format" in result.feedback
    
    def test_association_strengthening(self):
        """Test that repeated associations strengthen."""
        system = AssociativeLearning()
        context = LearningContext()
        data = {
            "stimulus_a": "bell",
            "stimulus_b": "food",
            "strength": 1.0
        }
        
        result1 = system.learn(context, data)
        strength1 = system.associations[("bell", "food")]
        
        result2 = system.learn(context, data)
        strength2 = system.associations[("bell", "food")]
        
        assert strength2 > strength1
    
    def test_get_description(self):
        """Test get_description method."""
        system = AssociativeLearning()
        description = system.get_description()
        assert isinstance(description, str)
        assert len(description) > 0


class TestNonAssociativeLearning:
    """Tests for NonAssociativeLearning system."""
    
    def test_initialization(self):
        """Test initialization of NonAssociativeLearning."""
        system = NonAssociativeLearning()
        assert system.name == "Non-Associative Learning"
        assert system.category == LearningCategory.BIOLOGICAL
        assert len(system.stimulus_responses) == 0
    
    def test_habituation(self):
        """Test habituation (decreased response)."""
        system = NonAssociativeLearning()
        context = LearningContext()
        data = {
            "stimulus": "loud_noise",
            "type": "habituation"
        }
        
        initial_response = 1.0
        result = system.learn(context, data)
        
        assert result.success is True
        assert system.stimulus_responses["loud_noise"] < initial_response
    
    def test_sensitization(self):
        """Test sensitization (increased response)."""
        system = NonAssociativeLearning()
        context = LearningContext()
        
        # Set initial response
        system.stimulus_responses["pain"] = 0.5
        
        data = {
            "stimulus": "pain",
            "type": "sensitization"
        }
        
        result = system.learn(context, data)
        
        assert result.success is True
        assert system.stimulus_responses["pain"] > 0.5


class TestOperantConditioning:
    """Tests for OperantConditioning system."""
    
    def test_initialization(self):
        """Test initialization of OperantConditioning."""
        system = OperantConditioning()
        assert system.name == "Operant Conditioning"
        assert system.category == LearningCategory.BIOLOGICAL
        assert len(system.behavior_values) == 0
    
    def test_positive_reinforcement(self):
        """Test positive reinforcement increases behavior value."""
        system = OperantConditioning()
        context = LearningContext()
        data = {
            "behavior": "press_lever",
            "reward": 1.0
        }
        
        result = system.learn(context, data)
        
        assert result.success is True
        assert system.behavior_values["press_lever"] > 0
    
    def test_punishment(self):
        """Test punishment decreases behavior value."""
        system = OperantConditioning()
        context = LearningContext()
        
        # Set initial positive value
        system.behavior_values["bad_action"] = 0.5
        
        data = {
            "behavior": "bad_action",
            "reward": -1.0
        }
        
        result = system.learn(context, data)
        
        assert result.success is True
        assert system.behavior_values["bad_action"] < 0.5


class TestSupervisedLearning:
    """Tests for SupervisedLearning system."""
    
    def test_initialization(self):
        """Test initialization of SupervisedLearning."""
        system = SupervisedLearning()
        assert system.name == "Supervised Learning"
        assert system.category == LearningCategory.MACHINE
        assert system.training_samples == 0
    
    def test_learn_from_labeled_data(self):
        """Test learning from labeled data."""
        system = SupervisedLearning()
        context = LearningContext()
        data = {
            "input": [1, 2, 3],
            "label": 1,
            "error": 0.3
        }
        
        result = system.learn(context, data)
        
        assert result.success is True
        assert system.training_samples == 1
        assert result.learning_delta > 0
    
    def test_multiple_samples(self):
        """Test learning from multiple samples."""
        system = SupervisedLearning()
        context = LearningContext()
        
        for i in range(5):
            data = {
                "input": [i, i+1, i+2],
                "label": i,
                "error": 0.1
            }
            system.learn(context, data)
        
        assert system.training_samples == 5


class TestUnsupervisedLearning:
    """Tests for UnsupervisedLearning system."""
    
    def test_initialization(self):
        """Test initialization of UnsupervisedLearning."""
        system = UnsupervisedLearning()
        assert system.name == "Unsupervised Learning"
        assert system.category == LearningCategory.MACHINE
        assert len(system.clusters) == 0
    
    def test_clustering(self):
        """Test pattern clustering."""
        system = UnsupervisedLearning()
        context = LearningContext()
        
        data1 = {"input": [1, 2, 3]}
        result1 = system.learn(context, data1)
        
        assert result1.success is True
        assert len(system.clusters) > 0
        
        data2 = {"input": [4, 5, 6]}
        result2 = system.learn(context, data2)
        
        assert result2.success is True


class TestReinforcementLearning:
    """Tests for ReinforcementLearning system."""
    
    def test_initialization(self):
        """Test initialization of ReinforcementLearning."""
        system = ReinforcementLearning()
        assert system.name == "Reinforcement Learning"
        assert system.category == LearningCategory.MACHINE
        assert len(system.q_values) == 0
    
    def test_q_learning_update(self):
        """Test Q-learning update."""
        system = ReinforcementLearning()
        context = LearningContext()
        data = {
            "state": "s1",
            "action": "a1",
            "reward": 1.0,
            "next_state": "s2"
        }
        
        result = system.learn(context, data)
        
        assert result.success is True
        assert ("s1", "a1") in system.q_values
        assert result.learning_delta > 0
    
    def test_q_value_improvement(self):
        """Test that Q-values improve with positive rewards."""
        system = ReinforcementLearning()
        context = LearningContext()
        
        # First update
        data = {
            "state": "s1",
            "action": "a1",
            "reward": 1.0,
            "next_state": "s2"
        }
        system.learn(context, data)
        q1 = system.q_values[("s1", "a1")]
        
        # Second update with positive reward
        system.learn(context, data)
        q2 = system.q_values[("s1", "a1")]
        
        assert q2 > q1


class TestTransferLearning:
    """Tests for TransferLearning system."""
    
    def test_initialization(self):
        """Test initialization of TransferLearning."""
        system = TransferLearning()
        assert system.name == "Transfer Learning"
        assert system.category == LearningCategory.MACHINE
        assert len(system.target_adaptations) == 0
    
    def test_knowledge_transfer(self):
        """Test transferring knowledge between domains."""
        system = TransferLearning()
        context = LearningContext()
        data = {
            "source_domain": "images",
            "target_domain": "sketches",
            "domain_similarity": 0.7
        }
        
        result = system.learn(context, data)
        
        assert result.success is True
        assert "images_to_sketches" in system.target_adaptations
    
    def test_low_similarity_transfer(self):
        """Test transfer with low domain similarity."""
        system = TransferLearning()
        context = LearningContext()
        data = {
            "source_domain": "images",
            "target_domain": "audio",
            "domain_similarity": 0.2
        }
        
        result = system.learn(context, data)
        
        assert result.success is False  # Too dissimilar


class TestMetaLearning:
    """Tests for MetaLearning system."""
    
    def test_initialization(self):
        """Test initialization of MetaLearning."""
        system = MetaLearning()
        assert system.name == "Meta-Learning"
        assert system.category == LearningCategory.MACHINE
        assert len(system.learning_strategies) == 0
    
    def test_strategy_learning(self):
        """Test learning about learning strategies."""
        system = MetaLearning()
        context = LearningContext()
        data = {
            "strategy": "gradient_descent",
            "performance": 0.8
        }
        
        result = system.learn(context, data)
        
        assert result.success is True
        assert "gradient_descent" in system.learning_strategies
    
    def test_strategy_performance_tracking(self):
        """Test tracking strategy performance over multiple tasks."""
        system = MetaLearning()
        context = LearningContext()
        
        # Test strategy multiple times
        performances = [0.6, 0.7, 0.8]
        for perf in performances:
            data = {
                "strategy": "sgd",
                "performance": perf
            }
            system.learn(context, data)
        
        assert len(system.strategy_performance["sgd"]) == 3
        assert system.learning_strategies["sgd"]["avg_performance"] == np.mean(performances)


class TestLearningSystemManager:
    """Tests for LearningSystemManager."""
    
    def test_initialization(self):
        """Test initialization of LearningSystemManager."""
        manager = LearningSystemManager()
        assert len(manager.systems) == 0
        assert len(manager.active_systems) == 0
    
    def test_register_system(self):
        """Test registering a learning system."""
        manager = LearningSystemManager()
        system = AssociativeLearning()
        
        manager.register_system(system)
        
        assert "Associative Learning" in manager.systems
        assert manager.systems["Associative Learning"] == system
    
    def test_activate_system(self):
        """Test activating a learning system."""
        manager = LearningSystemManager()
        system = AssociativeLearning()
        manager.register_system(system)
        
        manager.activate_system("Associative Learning")
        
        assert system.is_active
        assert "Associative Learning" in manager.active_systems
    
    def test_deactivate_system(self):
        """Test deactivating a learning system."""
        manager = LearningSystemManager()
        system = AssociativeLearning()
        manager.register_system(system)
        manager.activate_system("Associative Learning")
        
        manager.deactivate_system("Associative Learning")
        
        assert not system.is_active
        assert "Associative Learning" not in manager.active_systems
    
    def test_learn_with_active_systems(self):
        """Test learning across multiple active systems."""
        manager = LearningSystemManager()
        
        assoc = AssociativeLearning()
        operant = OperantConditioning()
        
        manager.register_system(assoc)
        manager.register_system(operant)
        
        manager.activate_system("Associative Learning")
        manager.activate_system("Operant Conditioning")
        
        context = LearningContext()
        data = {
            "Associative Learning": {
                "stimulus_a": "bell",
                "stimulus_b": "food",
                "strength": 1.0
            },
            "Operant Conditioning": {
                "behavior": "press_lever",
                "reward": 1.0
            }
        }
        
        results = manager.learn(context, data)
        
        assert len(results) == 2
        assert "Associative Learning" in results
        assert "Operant Conditioning" in results
        assert results["Associative Learning"].success
        assert results["Operant Conditioning"].success
    
    def test_get_all_metrics(self):
        """Test getting metrics from all systems."""
        manager = LearningSystemManager()
        system = AssociativeLearning()
        manager.register_system(system)
        
        # Perform some learning
        context = LearningContext()
        data = {
            "stimulus_a": "bell",
            "stimulus_b": "food",
            "strength": 1.0
        }
        system.learn(context, data)
        
        metrics = manager.get_all_metrics()
        
        assert "Associative Learning" in metrics
        assert "success_rate" in metrics["Associative Learning"]
    
    def test_get_biological_systems(self):
        """Test getting biological systems."""
        manager = LearningSystemManager()
        manager.register_system(AssociativeLearning())
        manager.register_system(SupervisedLearning())
        
        bio_systems = manager.get_biological_systems()
        
        assert len(bio_systems) == 1
        assert bio_systems[0].category == LearningCategory.BIOLOGICAL
    
    def test_get_machine_systems(self):
        """Test getting machine learning systems."""
        manager = LearningSystemManager()
        manager.register_system(AssociativeLearning())
        manager.register_system(SupervisedLearning())
        
        ml_systems = manager.get_machine_systems()
        
        assert len(ml_systems) == 1
        assert ml_systems[0].category == LearningCategory.MACHINE


class TestCreateDefaultLearningSystems:
    """Tests for create_default_learning_systems function."""
    
    def test_creates_manager_with_systems(self):
        """Test that default systems are created and registered."""
        manager = create_default_learning_systems()
        
        assert len(manager.systems) > 0
        assert "Associative Learning" in manager.systems
        assert "Supervised Learning" in manager.systems
    
    def test_biological_systems_registered(self):
        """Test that biological systems are registered."""
        manager = create_default_learning_systems()
        bio_systems = manager.get_biological_systems()
        
        assert len(bio_systems) >= 3
        bio_names = [s.name for s in bio_systems]
        assert "Associative Learning" in bio_names
        assert "Non-Associative Learning" in bio_names
        assert "Operant Conditioning" in bio_names
    
    def test_machine_systems_registered(self):
        """Test that machine learning systems are registered."""
        manager = create_default_learning_systems()
        ml_systems = manager.get_machine_systems()
        
        assert len(ml_systems) >= 5
        ml_names = [s.name for s in ml_systems]
        assert "Supervised Learning" in ml_names
        assert "Unsupervised Learning" in ml_names
        assert "Reinforcement Learning" in ml_names
        assert "Transfer Learning" in ml_names
        assert "Meta-Learning" in ml_names
