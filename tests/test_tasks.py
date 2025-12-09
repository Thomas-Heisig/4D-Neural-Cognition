"""Tests for tasks module."""

import pytest
import numpy as np
from src.tasks import (
    TaskResult,
    Environment,
    Task,
    PatternClassificationTask,
    PatternClassificationEnvironment,
    TemporalSequenceTask,
    TemporalSequenceEnvironment,
    SensorimotorControlTask,
    SensorimotorControlEnvironment,
    MultiModalIntegrationTask,
    MultiModalIntegrationEnvironment,
    ContinuousLearningTask,
    ContinuousLearningEnvironment,
    TransferLearningTask,
    TransferLearningEnvironment,
)


class TestTaskResult:
    """Test TaskResult dataclass."""

    def test_task_result_initialization(self):
        """Test TaskResult initialization."""
        result = TaskResult(
            accuracy=0.85,
            reward=0.9,
            reaction_time=50.0,
            stability=0.95
        )
        assert result.accuracy == 0.85
        assert result.reward == 0.9
        assert result.reaction_time == 50.0
        assert result.stability == 0.95

    def test_task_result_defaults(self):
        """Test TaskResult with defaults."""
        result = TaskResult()
        assert result.accuracy == 0.0
        assert result.reward == 0.0
        assert result.reaction_time == 0.0
        assert result.stability == 0.0
        assert result.additional_metrics == {}

    def test_task_result_with_additional_metrics(self):
        """Test TaskResult with additional metrics."""
        metrics = {'precision': 0.9, 'recall': 0.85}
        result = TaskResult(accuracy=0.87, additional_metrics=metrics)
        assert result.additional_metrics['precision'] == 0.9
        assert result.additional_metrics['recall'] == 0.85


class TestPatternClassificationEnvironment:
    """Test pattern classification environment."""

    def test_environment_initialization(self):
        """Test environment initialization."""
        env = PatternClassificationEnvironment(
            num_classes=4,
            pattern_size=(20, 20),
            noise_level=0.1,
            seed=42
        )
        assert env.num_classes == 4
        assert env.pattern_size == (20, 20)
        assert env.noise_level == 0.1

    def test_generate_base_patterns(self):
        """Test base pattern generation."""
        env = PatternClassificationEnvironment(num_classes=4, seed=42)
        assert env.base_patterns.shape[0] == 4
        assert env.base_patterns.shape[1:] == (20, 20)

    def test_reset(self):
        """Test environment reset."""
        env = PatternClassificationEnvironment(num_classes=4, seed=42)
        observation, info = env.reset()
        
        assert 'vision' in observation
        assert 'target_class' in info
        assert observation['vision'].shape == (20, 20)
        assert 0 <= info['target_class'] < 4

    def test_reset_reproducibility(self):
        """Test reset produces different patterns."""
        env = PatternClassificationEnvironment(num_classes=4, seed=42)
        obs1, info1 = env.reset()
        obs2, info2 = env.reset()
        
        # Different patterns (due to noise and/or different class)
        assert not np.array_equal(obs1['vision'], obs2['vision'])

    def test_step(self):
        """Test environment step."""
        env = PatternClassificationEnvironment(num_classes=4, seed=42)
        observation, info = env.reset()
        
        observation, reward, done, info = env.step()
        assert done == True  # Single-step task
        assert 'target_class' in info

    def test_render(self):
        """Test environment rendering."""
        env = PatternClassificationEnvironment(num_classes=4, seed=42)
        env.reset()
        rendered = env.render()
        assert rendered is not None
        assert rendered.shape == (20, 20)


class TestPatternClassificationTask:
    """Test pattern classification task."""

    def test_task_initialization(self):
        """Test task initialization."""
        task = PatternClassificationTask(
            num_classes=4,
            pattern_size=(20, 20),
            noise_level=0.1,
            seed=42
        )
        assert task.num_classes == 4
        assert task.pattern_size == (20, 20)
        assert task.env is not None

    def test_get_name(self):
        """Test task name."""
        task = PatternClassificationTask(num_classes=4, seed=42)
        name = task.get_name()
        assert 'PatternClassification' in name
        assert '4class' in name

    def test_get_description(self):
        """Test task description."""
        task = PatternClassificationTask(num_classes=4, seed=42)
        desc = task.get_description()
        assert isinstance(desc, str)
        assert '4' in desc

    def test_get_metrics(self):
        """Test metric descriptions."""
        task = PatternClassificationTask(num_classes=4, seed=42)
        metrics = task.get_metrics()
        
        assert 'accuracy' in metrics
        assert 'reward' in metrics
        assert 'reaction_time' in metrics
        assert 'stability' in metrics

    def test_evaluate_returns_result(self, simulation):
        """Test that evaluate returns TaskResult."""
        task = PatternClassificationTask(num_classes=4, seed=42)
        result = task.evaluate(simulation, num_episodes=2, max_steps=10)
        
        assert isinstance(result, TaskResult)
        assert 0.0 <= result.accuracy <= 1.0
        assert 0.0 <= result.reward <= 1.0


class TestTemporalSequenceEnvironment:
    """Test temporal sequence environment."""

    def test_environment_initialization(self):
        """Test environment initialization."""
        env = TemporalSequenceEnvironment(
            sequence_length=5,
            vocabulary_size=8,
            seed=42
        )
        assert env.sequence_length == 5
        assert env.vocabulary_size == 8

    def test_reset(self):
        """Test environment reset."""
        env = TemporalSequenceEnvironment(sequence_length=5, vocabulary_size=8, seed=42)
        observation, info = env.reset()
        
        assert 'digital' in observation
        assert 'sequence' in info
        assert 'position' in info
        assert len(info['sequence']) == 5

    def test_step(self):
        """Test environment step."""
        env = TemporalSequenceEnvironment(sequence_length=5, vocabulary_size=8, seed=42)
        observation, info = env.reset()
        
        observation, reward, done, info = env.step()
        assert 'position' in info
        assert info['position'] == 1

    def test_step_to_completion(self):
        """Test stepping through entire sequence."""
        env = TemporalSequenceEnvironment(sequence_length=5, vocabulary_size=8, seed=42)
        env.reset()
        
        done = False
        steps = 0
        while not done and steps < 10:
            _, _, done, _ = env.step()
            steps += 1
        
        assert done == True


class TestTemporalSequenceTask:
    """Test temporal sequence task."""

    def test_task_initialization(self):
        """Test task initialization."""
        task = TemporalSequenceTask(sequence_length=5, vocabulary_size=8, seed=42)
        assert task.sequence_length == 5
        assert task.vocabulary_size == 8

    def test_get_name(self):
        """Test task name."""
        task = TemporalSequenceTask(sequence_length=5, vocabulary_size=8, seed=42)
        name = task.get_name()
        assert 'TemporalSequence' in name
        assert 'L5' in name
        assert 'V8' in name

    def test_get_description(self):
        """Test task description."""
        task = TemporalSequenceTask(sequence_length=5, vocabulary_size=8, seed=42)
        desc = task.get_description()
        assert '5' in desc
        assert '8' in desc

    def test_evaluate_returns_result(self, simulation):
        """Test that evaluate returns TaskResult."""
        task = TemporalSequenceTask(sequence_length=5, vocabulary_size=8, seed=42)
        result = task.evaluate(simulation, num_episodes=2, max_steps=50)
        
        assert isinstance(result, TaskResult)
        assert 0.0 <= result.accuracy <= 1.0


class TestSensorimotorControlEnvironment:
    """Test sensorimotor control environment."""

    def test_environment_initialization(self):
        """Test environment initialization."""
        env = SensorimotorControlEnvironment(max_angle=np.pi/4, seed=42)
        assert env.max_angle == np.pi/4
        assert env.gravity == 9.81
        assert env.length == 1.0

    def test_reset(self):
        """Test environment reset."""
        env = SensorimotorControlEnvironment(max_angle=np.pi/4, seed=42)
        observation, info = env.reset()
        
        assert 'vision' in observation
        assert 'angle' in info
        assert 'velocity' in info
        assert observation['vision'].shape == (20, 20)

    def test_step_with_action(self):
        """Test environment step with action."""
        env = SensorimotorControlEnvironment(max_angle=np.pi/4, seed=42)
        env.reset()
        
        action = np.array([0.5])
        observation, reward, done, info = env.step(action)
        
        assert 'angle' in info
        assert 'velocity' in info
        assert 'balanced' in info

    def test_step_without_action(self):
        """Test environment step without action."""
        env = SensorimotorControlEnvironment(max_angle=np.pi/4, seed=42)
        env.reset()
        
        observation, reward, done, info = env.step()
        assert isinstance(reward, float)

    def test_physics_update(self):
        """Test that physics updates state."""
        env = SensorimotorControlEnvironment(max_angle=np.pi/4, seed=42)
        observation, info = env.reset()
        initial_angle = env.angle
        
        # Apply force
        env.step(np.array([1.0]))
        
        # Angle should have changed
        assert env.angle != initial_angle


class TestSensorimotorControlTask:
    """Test sensorimotor control task."""

    def test_task_initialization(self):
        """Test task initialization."""
        task = SensorimotorControlTask(max_angle=np.pi/4, control_interval=10, seed=42)
        assert task.max_angle == np.pi/4
        assert task.control_interval == 10

    def test_get_name(self):
        """Test task name."""
        task = SensorimotorControlTask(seed=42)
        name = task.get_name()
        assert 'SensorimotorControl' in name

    def test_evaluate_returns_result(self, simulation):
        """Test that evaluate returns TaskResult."""
        task = SensorimotorControlTask(max_angle=np.pi/4, control_interval=10, seed=42)
        result = task.evaluate(simulation, num_episodes=2, max_steps=50)
        
        assert isinstance(result, TaskResult)
        assert 0.0 <= result.accuracy <= 1.0
        assert 0.0 <= result.stability <= 1.0


class TestMultiModalIntegrationEnvironment:
    """Test multi-modal integration environment."""

    def test_environment_initialization(self):
        """Test environment initialization."""
        env = MultiModalIntegrationEnvironment(num_classes=4, modality_noise=0.2, seed=42)
        assert env.num_classes == 4
        assert env.modality_noise == 0.2

    def test_reset(self):
        """Test environment reset."""
        env = MultiModalIntegrationEnvironment(num_classes=4, seed=42)
        observation, info = env.reset()
        
        assert 'vision' in observation
        assert 'audio' in observation
        assert 'target_class' in info
        assert observation['vision'].shape == (20, 20)
        assert observation['audio'].shape == (20, 20)

    def test_step(self):
        """Test environment step."""
        env = MultiModalIntegrationEnvironment(num_classes=4, seed=42)
        env.reset()
        
        observation, reward, done, info = env.step()
        assert done == True  # Single-step task


class TestMultiModalIntegrationTask:
    """Test multi-modal integration task."""

    def test_task_initialization(self):
        """Test task initialization."""
        task = MultiModalIntegrationTask(num_classes=4, modality_noise=0.2, seed=42)
        assert task.num_classes == 4
        assert task.modality_noise == 0.2

    def test_get_name(self):
        """Test task name."""
        task = MultiModalIntegrationTask(num_classes=4, seed=42)
        name = task.get_name()
        assert 'MultiModalIntegration' in name
        assert '4class' in name

    def test_evaluate_returns_result(self, simulation):
        """Test that evaluate returns TaskResult."""
        task = MultiModalIntegrationTask(num_classes=4, seed=42)
        result = task.evaluate(simulation, num_episodes=2, max_steps=20)
        
        assert isinstance(result, TaskResult)


class TestContinuousLearningEnvironment:
    """Test continuous learning environment."""

    def test_environment_initialization(self):
        """Test environment initialization."""
        env = ContinuousLearningEnvironment(num_phases=3, steps_per_phase=100, seed=42)
        assert env.num_phases == 3
        assert env.steps_per_phase == 100

    def test_reset(self):
        """Test environment reset."""
        env = ContinuousLearningEnvironment(num_phases=3, seed=42)
        observation, info = env.reset()
        
        assert 'vision' in observation
        assert 'current_phase' in info
        assert 'phase_step' in info
        assert info['current_phase'] == 0

    def test_step_phase_progression(self):
        """Test that phases progress."""
        env = ContinuousLearningEnvironment(num_phases=3, steps_per_phase=5, seed=42)
        env.reset()
        
        # Step through first phase
        for _ in range(6):
            _, _, _, info = env.step()
        
        # Should be in phase 1 now
        assert env.current_phase == 1

    def test_generate_phase_input(self):
        """Test phase-specific input generation."""
        env = ContinuousLearningEnvironment(num_phases=3, seed=42)
        env.reset()
        
        phase0 = env._generate_phase_input(0)
        phase1 = env._generate_phase_input(1)
        
        # Different phases should have different patterns
        assert not np.array_equal(phase0['vision'], phase1['vision'])


class TestContinuousLearningTask:
    """Test continuous learning task."""

    def test_task_initialization(self):
        """Test task initialization."""
        task = ContinuousLearningTask(num_phases=3, steps_per_phase=100, seed=42)
        assert task.num_phases == 3
        assert task.steps_per_phase == 100

    def test_get_name(self):
        """Test task name."""
        task = ContinuousLearningTask(num_phases=3, seed=42)
        name = task.get_name()
        assert 'ContinuousLearning' in name
        assert '3phases' in name

    def test_evaluate_returns_result(self, simulation):
        """Test that evaluate returns TaskResult."""
        task = ContinuousLearningTask(num_phases=2, steps_per_phase=20, seed=42)
        result = task.evaluate(simulation, num_episodes=1, max_steps=50)
        
        assert isinstance(result, TaskResult)
        assert 'phase_accuracies' in result.additional_metrics
        assert 'forgetting' in result.additional_metrics


class TestTransferLearningEnvironment:
    """Test transfer learning environment."""

    def test_environment_initialization(self):
        """Test environment initialization."""
        env = TransferLearningEnvironment(
            source_classes=4,
            target_classes=4,
            similarity=0.7,
            seed=42
        )
        assert env.source_classes == 4
        assert env.target_classes == 4
        assert env.similarity == 0.7

    def test_reset_source_task(self):
        """Test reset with source task."""
        env = TransferLearningEnvironment(source_classes=4, target_classes=4, seed=42)
        observation, info = env.reset(target_task=False)
        
        assert 'vision' in observation
        assert 'target_class' in info
        assert 'is_target_task' in info
        assert info['is_target_task'] == False

    def test_reset_target_task(self):
        """Test reset with target task."""
        env = TransferLearningEnvironment(source_classes=4, target_classes=4, seed=42)
        observation, info = env.reset(target_task=True)
        
        assert 'vision' in observation
        assert 'target_class' in info
        assert info['is_target_task'] == True

    def test_generate_patterns(self):
        """Test pattern generation."""
        env = TransferLearningEnvironment(source_classes=4, target_classes=4, seed=42)
        assert env.source_patterns.shape[0] == 4
        assert env.target_patterns.shape[0] == 4

    def test_generate_related_patterns(self):
        """Test related pattern generation."""
        env = TransferLearningEnvironment(
            source_classes=4,
            target_classes=4,
            similarity=1.0,  # High similarity
            seed=42
        )
        # With high similarity, patterns should be more similar
        assert env.source_patterns.shape == env.target_patterns.shape


class TestTransferLearningTask:
    """Test transfer learning task."""

    def test_task_initialization(self):
        """Test task initialization."""
        task = TransferLearningTask(
            source_classes=4,
            target_classes=4,
            similarity=0.7,
            seed=42
        )
        assert task.source_classes == 4
        assert task.target_classes == 4
        assert task.similarity == 0.7

    def test_get_name(self):
        """Test task name."""
        task = TransferLearningTask(source_classes=4, target_classes=4, seed=42)
        name = task.get_name()
        assert 'TransferLearning' in name
        assert 'S4' in name
        assert 'T4' in name

    def test_get_description(self):
        """Test task description."""
        task = TransferLearningTask(source_classes=4, target_classes=4, seed=42)
        desc = task.get_description()
        assert '4' in desc

    def test_evaluate_returns_result(self, simulation):
        """Test that evaluate returns TaskResult."""
        task = TransferLearningTask(source_classes=4, target_classes=4, seed=42)
        result = task.evaluate(simulation, num_episodes=10, max_steps=20)
        
        assert isinstance(result, TaskResult)
        assert 'source_accuracy' in result.additional_metrics
        assert 'target_accuracy' in result.additional_metrics
        assert 'transfer_gain' in result.additional_metrics
