"""Unified configuration and experiment management system.

This module provides tools for managing experiments, configurations,
parameter sweeps, and reproducible research workflows.
"""

import json
import yaml
import datetime
import hashlib
import os
import subprocess
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import itertools
import copy


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    
    name: str
    description: str = ""
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    version: str = "1.0"
    
    # Network configuration
    network_config: Dict[str, Any] = field(default_factory=dict)
    
    # Variables to sweep
    variables: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metrics to track
    metrics: List[str] = field(default_factory=list)
    
    # Export settings
    export_formats: List[str] = field(default_factory=lambda: ["json"])
    
    # Reproducibility
    seed: Optional[int] = None
    
    # Metadata
    author: str = ""
    tags: List[str] = field(default_factory=list)
    
    def get_hash(self) -> str:
        """Get unique hash for this configuration."""
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'ExperimentConfig':
        """Load from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Extract experiment section if it exists
        if 'experiment' in data:
            data = data['experiment']
        
        return cls.from_dict(data)
    
    def to_yaml(self, path: str) -> None:
        """Save to YAML file."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump({'experiment': self.to_dict()}, f, default_flow_style=False)
    
    @classmethod
    def from_json(cls, path: str) -> 'ExperimentConfig':
        """Load from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def to_json(self, path: str) -> None:
        """Save to JSON file."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    
    experiment_name: str
    run_id: str
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    
    # Configuration used
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Metrics collected
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Status
    status: str = "running"  # running, completed, failed
    error: Optional[str] = None
    
    # Timing
    duration_seconds: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentResult':
        """Create from dictionary."""
        return cls(**data)


class ParameterSweep:
    """Manage parameter sweeps for experiments."""
    
    def __init__(self, base_config: Dict[str, Any]):
        """Initialize parameter sweep.
        
        Args:
            base_config: Base configuration to modify
        """
        self.base_config = copy.deepcopy(base_config)
        self.sweep_params: List[Dict[str, Any]] = []
    
    def add_parameter(
        self,
        param_path: str,
        values: List[Any]
    ) -> None:
        """Add a parameter to sweep.
        
        Args:
            param_path: Dot-separated path to parameter (e.g., 'network.tau_m')
            values: List of values to try
        """
        self.sweep_params.append({
            'path': param_path,
            'values': values
        })
    
    def generate_configs(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Generate all configuration combinations.
        
        Returns:
            List of (config_id, config_dict) tuples
        """
        if not self.sweep_params:
            return [("base", self.base_config)]
        
        # Get all parameter combinations
        param_names = [p['path'] for p in self.sweep_params]
        param_values = [p['values'] for p in self.sweep_params]
        
        configs = []
        for i, combination in enumerate(itertools.product(*param_values)):
            # Create config for this combination
            config = copy.deepcopy(self.base_config)
            
            # Set parameters
            config_id_parts = []
            for param_path, value in zip(param_names, combination):
                self._set_nested_value(config, param_path, value)
                config_id_parts.append(f"{param_path.split('.')[-1]}={value}")
            
            config_id = f"sweep_{i}_" + "_".join(config_id_parts)
            configs.append((config_id, config))
        
        return configs
    
    def _set_nested_value(self, d: Dict, path: str, value: Any) -> None:
        """Set a nested dictionary value using dot notation."""
        keys = path.split('.')
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value
    
    def get_num_configs(self) -> int:
        """Get total number of configurations in sweep."""
        if not self.sweep_params:
            return 1
        
        count = 1
        for param in self.sweep_params:
            count *= len(param['values'])
        return count


class ExperimentManager:
    """Manage experiments and results."""
    
    def __init__(self, experiments_dir: str = "experiments"):
        """Initialize experiment manager.
        
        Args:
            experiments_dir: Directory to store experiments and results
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.configs_dir = self.experiments_dir / "configs"
        self.results_dir = self.experiments_dir / "results"
        self.configs_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
    
    def create_experiment(
        self,
        name: str,
        config: ExperimentConfig
    ) -> str:
        """Create a new experiment.
        
        Args:
            name: Experiment name
            config: Experiment configuration
            
        Returns:
            Experiment ID
        """
        config.name = name
        exp_id = f"{name}_{config.get_hash()}"
        
        # Save config
        config_path = self.configs_dir / f"{exp_id}.yaml"
        config.to_yaml(str(config_path))
        
        return exp_id
    
    def load_experiment(self, exp_id: str) -> ExperimentConfig:
        """Load an experiment configuration.
        
        Args:
            exp_id: Experiment ID
            
        Returns:
            Experiment configuration
        """
        config_path = self.configs_dir / f"{exp_id}.yaml"
        if not config_path.exists():
            # Try JSON
            config_path = self.configs_dir / f"{exp_id}.json"
        
        if config_path.suffix == '.yaml':
            return ExperimentConfig.from_yaml(str(config_path))
        else:
            return ExperimentConfig.from_json(str(config_path))
    
    def save_result(
        self,
        result: ExperimentResult
    ) -> None:
        """Save experiment result.
        
        Args:
            result: Experiment result to save
        """
        result_path = self.results_dir / f"{result.run_id}.json"
        with open(result_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
    
    def load_results(
        self,
        experiment_name: Optional[str] = None
    ) -> List[ExperimentResult]:
        """Load experiment results.
        
        Args:
            experiment_name: If specified, only load results for this experiment
            
        Returns:
            List of experiment results
        """
        results = []
        
        for result_file in self.results_dir.glob("*.json"):
            with open(result_file, 'r') as f:
                result_data = json.load(f)
                result = ExperimentResult.from_dict(result_data)
                
                if experiment_name is None or result.experiment_name == experiment_name:
                    results.append(result)
        
        return results
    
    def list_experiments(self) -> List[str]:
        """List all experiments.
        
        Returns:
            List of experiment IDs
        """
        experiments = []
        
        for config_file in self.configs_dir.glob("*"):
            if config_file.suffix in ['.yaml', '.json']:
                experiments.append(config_file.stem)
        
        return experiments
    
    def compare_results(
        self,
        exp_ids: List[str],
        metric: str
    ) -> Dict[str, List[float]]:
        """Compare results across experiments.
        
        Args:
            exp_ids: List of experiment IDs to compare
            metric: Metric to compare
            
        Returns:
            Dictionary mapping experiment IDs to metric values
        """
        comparison = {}
        
        for exp_id in exp_ids:
            results = self.load_results(exp_id)
            metric_values = []
            
            for result in results:
                if metric in result.metrics:
                    value = result.metrics[metric]
                    if isinstance(value, (int, float)):
                        metric_values.append(value)
                    elif isinstance(value, list):
                        metric_values.extend(value)
            
            comparison[exp_id] = metric_values
        
        return comparison
    
    def export_experiment(
        self,
        exp_id: str,
        output_dir: str,
        formats: Optional[List[str]] = None
    ) -> List[str]:
        """Export experiment data in various formats.
        
        Args:
            exp_id: Experiment ID
            output_dir: Output directory
            formats: List of formats to export ('json', 'csv', 'hdf5')
            
        Returns:
            List of exported file paths
        """
        if formats is None:
            formats = ['json']
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exported_files = []
        
        # Load experiment and results
        config = self.load_experiment(exp_id)
        results = self.load_results(config.name)
        
        # Export in requested formats
        if 'json' in formats:
            json_path = output_path / f"{exp_id}.json"
            export_data = {
                'config': config.to_dict(),
                'results': [r.to_dict() for r in results]
            }
            with open(json_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            exported_files.append(str(json_path))
        
        if 'csv' in formats:
            csv_path = output_path / f"{exp_id}.csv"
            self._export_to_csv(config, results, str(csv_path))
            exported_files.append(str(csv_path))
        
        return exported_files
    
    def _export_to_csv(
        self,
        config: ExperimentConfig,
        results: List[ExperimentResult],
        csv_path: str
    ) -> None:
        """Export results to CSV format."""
        import csv
        
        if not results:
            return
        
        # Collect all metric names
        all_metrics = set()
        for result in results:
            all_metrics.update(result.metrics.keys())
        
        # Write CSV
        with open(csv_path, 'w', newline='') as f:
            fieldnames = ['run_id', 'timestamp', 'status'] + sorted(all_metrics)
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                row = {
                    'run_id': result.run_id,
                    'timestamp': result.timestamp,
                    'status': result.status
                }
                row.update(result.metrics)
                writer.writerow(row)


def create_experiment_from_yaml(yaml_path: str) -> ExperimentConfig:
    """Create experiment configuration from YAML file.
    
    Args:
        yaml_path: Path to YAML configuration file
        
    Returns:
        ExperimentConfig instance
    """
    return ExperimentConfig.from_yaml(yaml_path)


def generate_sweep_configs(
    base_config: Dict[str, Any],
    sweep_params: Dict[str, List[Any]]
) -> List[Tuple[str, Dict[str, Any]]]:
    """Generate parameter sweep configurations.
    
    Args:
        base_config: Base configuration
        sweep_params: Dictionary mapping parameter paths to value lists
        
    Returns:
        List of (config_id, config_dict) tuples
    """
    sweep = ParameterSweep(base_config)
    
    for param_path, values in sweep_params.items():
        sweep.add_parameter(param_path, values)
    
    return sweep.generate_configs()


def get_git_commit() -> Optional[str]:
    """Get current git commit hash.
    
    Returns:
        Git commit hash or None if not in a git repository
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def get_git_status() -> Optional[Dict[str, Any]]:
    """Get git repository status.
    
    Returns:
        Dictionary with git status information or None
    """
    try:
        # Get commit hash
        commit = get_git_commit()
        if commit is None:
            return None
        
        # Check for uncommitted changes
        result = subprocess.run(
            ['git', 'diff', '--quiet'],
            timeout=5
        )
        has_uncommitted = result.returncode != 0
        
        # Get branch name
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )
        branch = result.stdout.strip()
        
        return {
            'commit': commit,
            'branch': branch,
            'has_uncommitted_changes': has_uncommitted
        }
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


class ExperimentDatabase:
    """SQLite database for experiment tracking."""
    
    def __init__(self, db_path: str = "experiments/experiments.db"):
        """Initialize experiment database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Experiments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                timestamp TEXT NOT NULL,
                git_commit TEXT,
                git_branch TEXT,
                has_uncommitted_changes INTEGER,
                author TEXT,
                config_json TEXT NOT NULL
            )
        """)
        
        # Runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                seed INTEGER,
                status TEXT,
                duration_seconds REAL,
                error TEXT,
                config_json TEXT NOT NULL,
                metrics_json TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
        """)
        
        # Metrics table (for time-series data)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                step INTEGER,
                metric_name TEXT NOT NULL,
                metric_value REAL,
                timestamp TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def add_experiment(
        self,
        exp_id: str,
        name: str,
        config: Dict[str, Any],
        description: str = "",
        author: str = ""
    ) -> None:
        """Add a new experiment to database.
        
        Args:
            exp_id: Unique experiment ID
            name: Experiment name
            config: Experiment configuration
            description: Experiment description
            author: Author name
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get git information
        git_status = get_git_status()
        
        cursor.execute("""
            INSERT INTO experiments 
            (id, name, description, timestamp, git_commit, git_branch, 
             has_uncommitted_changes, author, config_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            exp_id,
            name,
            description,
            datetime.datetime.now().isoformat(),
            git_status['commit'] if git_status else None,
            git_status['branch'] if git_status else None,
            git_status['has_uncommitted_changes'] if git_status else None,
            author,
            json.dumps(config)
        ))
        
        conn.commit()
        conn.close()
    
    def add_run(
        self,
        run_id: str,
        experiment_id: str,
        config: Dict[str, Any],
        seed: Optional[int] = None
    ) -> None:
        """Add a new run to database.
        
        Args:
            run_id: Unique run ID
            experiment_id: Associated experiment ID
            config: Run configuration
            seed: Random seed used
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO runs 
            (id, experiment_id, timestamp, seed, status, config_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            experiment_id,
            datetime.datetime.now().isoformat(),
            seed,
            "running",
            json.dumps(config)
        ))
        
        conn.commit()
        conn.close()
    
    def update_run(
        self,
        run_id: str,
        status: str = None,
        duration: float = None,
        error: str = None,
        metrics: Dict[str, Any] = None
    ) -> None:
        """Update run information.
        
        Args:
            run_id: Run ID to update
            status: New status
            duration: Duration in seconds
            error: Error message if failed
            metrics: Final metrics dictionary
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        updates = []
        params = []
        
        if status is not None:
            updates.append("status = ?")
            params.append(status)
        if duration is not None:
            updates.append("duration_seconds = ?")
            params.append(duration)
        if error is not None:
            updates.append("error = ?")
            params.append(error)
        if metrics is not None:
            updates.append("metrics_json = ?")
            params.append(json.dumps(metrics))
        
        if updates:
            params.append(run_id)
            query = f"UPDATE runs SET {', '.join(updates)} WHERE id = ?"
            cursor.execute(query, params)
        
        conn.commit()
        conn.close()
    
    def add_metric(
        self,
        run_id: str,
        step: int,
        metric_name: str,
        metric_value: float
    ) -> None:
        """Add a metric data point.
        
        Args:
            run_id: Run ID
            step: Simulation step
            metric_name: Name of metric
            metric_value: Value of metric
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO metrics (run_id, step, metric_name, metric_value, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (
            run_id,
            step,
            metric_name,
            metric_value,
            datetime.datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_experiment(self, exp_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment information.
        
        Args:
            exp_id: Experiment ID
            
        Returns:
            Dictionary with experiment info or None
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            result = dict(row)
            result['config'] = json.loads(result['config_json'])
            del result['config_json']
            return result
        return None
    
    def get_runs(
        self,
        experiment_id: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get runs, optionally filtered.
        
        Args:
            experiment_id: Filter by experiment
            status: Filter by status
            
        Returns:
            List of run dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM runs WHERE 1=1"
        params = []
        
        if experiment_id:
            query += " AND experiment_id = ?"
            params.append(experiment_id)
        if status:
            query += " AND status = ?"
            params.append(status)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            result = dict(row)
            result['config'] = json.loads(result['config_json'])
            del result['config_json']
            if result['metrics_json']:
                result['metrics'] = json.loads(result['metrics_json'])
                del result['metrics_json']
            results.append(result)
        
        return results
    
    def get_metrics(
        self,
        run_id: str,
        metric_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get metric time series data.
        
        Args:
            run_id: Run ID
            metric_name: Optional metric name filter
            
        Returns:
            List of metric data points
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM metrics WHERE run_id = ?"
        params = [run_id]
        
        if metric_name:
            query += " AND metric_name = ?"
            params.append(metric_name)
        
        query += " ORDER BY step"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
