"""Tests for enhanced experiment tracking and database functionality."""

import pytest
import tempfile
import os
import sqlite3
from src.experiment_management import (
    ExperimentDatabase,
    get_git_commit,
    get_git_status,
    ExperimentConfig,
    ExperimentResult
)


class TestGitTracking:
    """Tests for git tracking functionality."""
    
    def test_get_git_commit(self):
        """Test getting git commit hash."""
        commit = get_git_commit()
        # In a git repo, should return a hash
        # Outside git repo, should return None
        if commit is not None:
            assert len(commit) == 40  # Full SHA-1 hash
            assert all(c in '0123456789abcdef' for c in commit)
    
    def test_get_git_status(self):
        """Test getting git status."""
        status = get_git_status()
        # In a git repo, should return status dict
        # Outside git repo, should return None
        if status is not None:
            assert 'commit' in status
            assert 'branch' in status
            assert 'has_uncommitted_changes' in status
            assert isinstance(status['has_uncommitted_changes'], bool)


class TestExperimentDatabase:
    """Tests for experiment database functionality."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            yield db_path
    
    def test_database_creation(self, temp_db):
        """Test that database and tables are created."""
        db = ExperimentDatabase(temp_db)
        
        # Check that database file exists
        assert os.path.exists(temp_db)
        
        # Check that tables exist
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        assert 'experiments' in tables
        assert 'runs' in tables
        assert 'metrics' in tables
    
    def test_add_experiment(self, temp_db):
        """Test adding experiment to database."""
        db = ExperimentDatabase(temp_db)
        
        config = {
            'lattice_shape': [10, 10, 10, 1],
            'learning_rate': 0.01
        }
        
        db.add_experiment(
            exp_id='test_exp_1',
            name='Test Experiment',
            config=config,
            description='A test experiment',
            author='Test User'
        )
        
        # Retrieve and verify
        exp = db.get_experiment('test_exp_1')
        assert exp is not None
        assert exp['name'] == 'Test Experiment'
        assert exp['description'] == 'A test experiment'
        assert exp['author'] == 'Test User'
        assert exp['config'] == config
        assert 'timestamp' in exp
    
    def test_add_run(self, temp_db):
        """Test adding run to database."""
        db = ExperimentDatabase(temp_db)
        
        # Create experiment first
        db.add_experiment(
            exp_id='test_exp_1',
            name='Test Experiment',
            config={'lattice_shape': [10, 10, 10, 1]}
        )
        
        # Add run
        run_config = {
            'lattice_shape': [10, 10, 10, 1],
            'seed': 42
        }
        
        db.add_run(
            run_id='run_1',
            experiment_id='test_exp_1',
            config=run_config,
            seed=42
        )
        
        # Retrieve and verify
        runs = db.get_runs(experiment_id='test_exp_1')
        assert len(runs) == 1
        assert runs[0]['id'] == 'run_1'
        assert runs[0]['seed'] == 42
        assert runs[0]['status'] == 'running'
        assert runs[0]['config'] == run_config
    
    def test_update_run(self, temp_db):
        """Test updating run information."""
        db = ExperimentDatabase(temp_db)
        
        # Create experiment and run
        db.add_experiment(
            exp_id='test_exp_1',
            name='Test Experiment',
            config={}
        )
        db.add_run(
            run_id='run_1',
            experiment_id='test_exp_1',
            config={},
            seed=42
        )
        
        # Update run
        db.update_run(
            run_id='run_1',
            status='completed',
            duration=123.45,
            metrics={'accuracy': 0.95, 'loss': 0.05}
        )
        
        # Verify update
        runs = db.get_runs(experiment_id='test_exp_1')
        assert len(runs) == 1
        assert runs[0]['status'] == 'completed'
        assert runs[0]['duration_seconds'] == 123.45
        assert runs[0]['metrics']['accuracy'] == 0.95
    
    def test_add_metrics(self, temp_db):
        """Test adding time-series metrics."""
        db = ExperimentDatabase(temp_db)
        
        # Create experiment and run
        db.add_experiment(
            exp_id='test_exp_1',
            name='Test Experiment',
            config={}
        )
        db.add_run(
            run_id='run_1',
            experiment_id='test_exp_1',
            config={},
            seed=42
        )
        
        # Add metrics at different steps
        for step in range(10):
            db.add_metric(
                run_id='run_1',
                step=step,
                metric_name='spike_count',
                metric_value=float(step * 10)
            )
        
        # Retrieve metrics
        metrics = db.get_metrics(run_id='run_1', metric_name='spike_count')
        assert len(metrics) == 10
        assert metrics[0]['step'] == 0
        assert metrics[0]['metric_value'] == 0.0
        assert metrics[9]['step'] == 9
        assert metrics[9]['metric_value'] == 90.0
    
    def test_filter_runs_by_status(self, temp_db):
        """Test filtering runs by status."""
        db = ExperimentDatabase(temp_db)
        
        # Create experiment
        db.add_experiment(
            exp_id='test_exp_1',
            name='Test Experiment',
            config={}
        )
        
        # Add multiple runs with different statuses
        db.add_run(run_id='run_1', experiment_id='test_exp_1', config={}, seed=1)
        db.add_run(run_id='run_2', experiment_id='test_exp_1', config={}, seed=2)
        db.add_run(run_id='run_3', experiment_id='test_exp_1', config={}, seed=3)
        
        db.update_run(run_id='run_1', status='completed')
        db.update_run(run_id='run_2', status='completed')
        db.update_run(run_id='run_3', status='failed')
        
        # Filter by status
        completed_runs = db.get_runs(experiment_id='test_exp_1', status='completed')
        failed_runs = db.get_runs(experiment_id='test_exp_1', status='failed')
        
        assert len(completed_runs) == 2
        assert len(failed_runs) == 1
        assert failed_runs[0]['id'] == 'run_3'
    
    def test_multiple_experiments(self, temp_db):
        """Test managing multiple experiments."""
        db = ExperimentDatabase(temp_db)
        
        # Create multiple experiments
        for i in range(3):
            db.add_experiment(
                exp_id=f'exp_{i}',
                name=f'Experiment {i}',
                config={'id': i}
            )
            
            # Add runs for each experiment
            for j in range(2):
                db.add_run(
                    run_id=f'run_{i}_{j}',
                    experiment_id=f'exp_{i}',
                    config={'run': j},
                    seed=i * 10 + j
                )
        
        # Verify we can retrieve runs for specific experiments
        exp_0_runs = db.get_runs(experiment_id='exp_0')
        exp_1_runs = db.get_runs(experiment_id='exp_1')
        
        assert len(exp_0_runs) == 2
        assert len(exp_1_runs) == 2
        assert all(run['experiment_id'] == 'exp_0' for run in exp_0_runs)
        assert all(run['experiment_id'] == 'exp_1' for run in exp_1_runs)
    
    def test_git_tracking_in_database(self, temp_db):
        """Test that git information is stored in database."""
        db = ExperimentDatabase(temp_db)
        
        db.add_experiment(
            exp_id='test_exp_1',
            name='Test Experiment',
            config={}
        )
        
        exp = db.get_experiment('test_exp_1')
        
        # Git fields should exist (may be None if not in a git repo)
        assert 'git_commit' in exp
        assert 'git_branch' in exp
        assert 'has_uncommitted_changes' in exp
        
        # If we're in a git repo, these should have values
        git_status = get_git_status()
        if git_status:
            assert exp['git_commit'] == git_status['commit']
            assert exp['git_branch'] == git_status['branch']


class TestExperimentConfigWithTracking:
    """Test ExperimentConfig with enhanced tracking."""
    
    def test_config_with_git_info(self):
        """Test that ExperimentConfig can store git information."""
        config = ExperimentConfig(
            name="Test Experiment",
            description="Test with git tracking"
        )
        
        # Add git information
        git_status = get_git_status()
        if git_status:
            config_dict = config.to_dict()
            config_dict['git_commit'] = git_status['commit']
            config_dict['git_branch'] = git_status['branch']
            
            # Should be serializable
            import json
            json_str = json.dumps(config_dict)
            assert json_str is not None
