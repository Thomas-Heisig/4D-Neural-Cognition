"""Tests for Adaptive VNC Orchestrator."""

import pytest
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from brain_model import BrainModel
from simulation import Simulation
from hardware_abstraction.adaptive_vnc_orchestrator import AdaptiveVNCOrchestrator
from hardware_abstraction.virtual_clock import GlobalVirtualClock
from hardware_abstraction.virtual_processing_unit import VirtualProcessingUnit


class TestAdaptiveVNCOrchestrator:
    """Tests for AdaptiveVNCOrchestrator class."""
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        config = {
            "lattice_shape": [10, 10, 5, 4],
            "neuron_model": {
                "type": "LIF",
                "params_default": {
                    "tau_m": 20.0,
                    "v_rest": -65.0,
                    "v_reset": -70.0,
                    "v_threshold": -50.0,
                    "refractory_period": 5.0,
                }
            },
            "cell_lifecycle": {
                "enable_death": False,
                "enable_reproduction": False,
                "max_age": 1000,
            },
            "plasticity": {"learning_rate": 0.01},
            "senses": {},
            "areas": []
        }
        
        model = BrainModel(config=config)
        sim = Simulation(model, use_vnc=False)
        
        orchestrator = AdaptiveVNCOrchestrator(
            sim,
            imbalance_threshold=0.3,
            activity_threshold=0.7,
            monitoring_interval=100,
        )
        
        assert orchestrator.sim == sim
        assert orchestrator.imbalance_threshold == 0.3
        assert orchestrator.activity_threshold == 0.7
        assert orchestrator.monitoring_interval == 100
        assert len(orchestrator.performance_log) == 0
        assert orchestrator.total_repartitions == 0
    
    def test_monitor_without_vnc(self):
        """Test monitoring when VNC is not enabled."""
        config = {
            "lattice_shape": [10, 10, 5, 4],
            "neuron_model": {
                "type": "LIF",
                "params_default": {
                    "tau_m": 20.0,
                    "v_rest": -65.0,
                    "v_reset": -70.0,
                    "v_threshold": -50.0,
                    "refractory_period": 5.0,
                }
            },
            "cell_lifecycle": {
                "enable_death": False,
                "enable_reproduction": False,
                "max_age": 1000,
            },
            "plasticity": {"learning_rate": 0.01},
            "senses": {},
            "areas": []
        }
        
        model = BrainModel(config=config)
        sim = Simulation(model, use_vnc=False)
        
        orchestrator = AdaptiveVNCOrchestrator(sim)
        
        # Monitor at interval
        result = orchestrator.monitor_and_adapt(current_cycle=100)
        
        assert result["monitored"] is False
        assert result["reason"] == "no_vnc_system"
    
    def test_monitor_skip_non_interval(self):
        """Test that monitoring is skipped when not at interval."""
        config = {
            "lattice_shape": [10, 10, 5, 4],
            "neuron_model": {
                "type": "LIF",
                "params_default": {
                    "tau_m": 20.0,
                    "v_rest": -65.0,
                    "v_reset": -70.0,
                    "v_threshold": -50.0,
                    "refractory_period": 5.0,
                }
            },
            "cell_lifecycle": {
                "enable_death": False,
                "enable_reproduction": False,
                "max_age": 1000,
            },
            "plasticity": {"learning_rate": 0.01},
            "senses": {},
            "areas": []
        }
        
        model = BrainModel(config=config)
        sim = Simulation(model, use_vnc=False)
        
        orchestrator = AdaptiveVNCOrchestrator(sim, monitoring_interval=100)
        
        # Monitor at non-interval cycle
        result = orchestrator.monitor_and_adapt(current_cycle=50)
        
        assert result["monitored"] is False
        assert result["cycle"] == 50
    
    def test_monitor_with_vnc(self):
        """Test monitoring with VNC system."""
        config = {
            "lattice_shape": [10, 10, 5, 4],
            "neuron_model": {
                "type": "LIF",
                "params_default": {
                    "tau_m": 20.0,
                    "v_rest": -65.0,
                    "v_reset": -70.0,
                    "v_threshold": -50.0,
                    "refractory_period": 5.0,
                }
            },
            "cell_lifecycle": {
                "enable_death": False,
                "enable_reproduction": False,
                "max_age": 1000,
            },
            "plasticity": {"learning_rate": 0.01},
            "senses": {},
            "areas": []
        }
        
        model = BrainModel(config=config)
        
        # Add some neurons before creating simulation
        for i in range(20):
            model.add_neuron(i % 10, (i // 10) % 10, 0, 0)
        
        sim = Simulation(model, use_vnc=True)
        
        orchestrator = AdaptiveVNCOrchestrator(sim, monitoring_interval=10)
        
        # Run some cycles to generate statistics
        for cycle in range(10):
            if hasattr(sim, 'virtual_clock'):
                for vpu in sim.virtual_clock.vpus:
                    vpu.process_cycle(cycle)
        
        # Monitor at interval
        result = orchestrator.monitor_and_adapt(current_cycle=10)
        
        assert result["monitored"] is True
        assert result["cycle"] == 10
        assert "load_imbalance" in result
        assert "hot_slices" in result
        assert "actions_taken" in result
    
    def test_collect_vpu_statistics(self):
        """Test collecting VPU statistics."""
        config = {
            "lattice_shape": [10, 10, 5, 4],
            "neuron_model": {
                "type": "LIF",
                "params_default": {
                    "tau_m": 20.0,
                    "v_rest": -65.0,
                    "v_reset": -70.0,
                    "v_threshold": -50.0,
                    "refractory_period": 5.0,
                }
            },
            "cell_lifecycle": {
                "enable_death": False,
                "enable_reproduction": False,
                "max_age": 1000,
            },
            "plasticity": {"learning_rate": 0.01},
            "senses": {},
            "areas": []
        }
        
        model = BrainModel(config=config)
        
        # Add some neurons across multiple w-slices
        for i in range(20):
            model.add_neuron(i % 10, (i // 10) % 10, 0, i % 4)
        
        sim = Simulation(model, use_vnc=True)
        
        orchestrator = AdaptiveVNCOrchestrator(sim)
        
        stats = orchestrator._collect_vpu_statistics()
        
        assert len(stats) >= 1  # At least one VPU should be created
        assert all("vpu_id" in s for s in stats)
        assert all("neuron_count" in s for s in stats)
    
    def test_calculate_load_imbalance_empty(self):
        """Test load imbalance calculation with no VPUs."""
        config = {
            "lattice_shape": [10, 10, 5, 4],
            "neuron_model": {
                "type": "LIF",
                "params_default": {
                    "tau_m": 20.0,
                    "v_rest": -65.0,
                    "v_reset": -70.0,
                    "v_threshold": -50.0,
                    "refractory_period": 5.0,
                }
            },
            "cell_lifecycle": {
                "enable_death": False,
                "enable_reproduction": False,
                "max_age": 1000,
            },
            "plasticity": {"learning_rate": 0.01},
            "senses": {},
            "areas": []
        }
        
        model = BrainModel(config=config)
        sim = Simulation(model, use_vnc=False)
        
        orchestrator = AdaptiveVNCOrchestrator(sim)
        
        imbalance = orchestrator._calculate_load_imbalance([])
        
        assert imbalance == 0.0
    
    def test_calculate_load_imbalance_with_stats(self):
        """Test load imbalance calculation with VPU stats."""
        config = {
            "lattice_shape": [10, 10, 5, 4],
            "neuron_model": {
                "type": "LIF",
                "params_default": {
                    "tau_m": 20.0,
                    "v_rest": -65.0,
                    "v_reset": -70.0,
                    "v_threshold": -50.0,
                    "refractory_period": 5.0,
                }
            },
            "cell_lifecycle": {
                "enable_death": False,
                "enable_reproduction": False,
                "max_age": 1000,
            },
            "plasticity": {"learning_rate": 0.01},
            "senses": {},
            "areas": []
        }
        
        model = BrainModel(config=config)
        sim = Simulation(model, use_vnc=False)
        
        orchestrator = AdaptiveVNCOrchestrator(sim)
        
        # Mock VPU statistics with varying processing times
        vpu_stats = [
            {"vpu_id": 0, "cycles_executed": 10, "avg_processing_time_ms": 1.0},
            {"vpu_id": 1, "cycles_executed": 10, "avg_processing_time_ms": 2.0},
            {"vpu_id": 2, "cycles_executed": 10, "avg_processing_time_ms": 3.0},
        ]
        
        imbalance = orchestrator._calculate_load_imbalance(vpu_stats)
        
        assert imbalance > 0.0  # Should have some imbalance
    
    def test_identify_hot_slices(self):
        """Test identification of hot slices."""
        config = {
            "lattice_shape": [10, 10, 5, 4],
            "neuron_model": {
                "type": "LIF",
                "params_default": {
                    "tau_m": 20.0,
                    "v_rest": -65.0,
                    "v_reset": -70.0,
                    "v_threshold": -50.0,
                    "refractory_period": 5.0,
                }
            },
            "cell_lifecycle": {
                "enable_death": False,
                "enable_reproduction": False,
                "max_age": 1000,
            },
            "plasticity": {"learning_rate": 0.01},
            "senses": {},
            "areas": []
        }
        
        model = BrainModel(config=config)
        sim = Simulation(model, use_vnc=False)
        
        orchestrator = AdaptiveVNCOrchestrator(sim, activity_threshold=0.7)
        
        # Mock VPU statistics with varying activity
        vpu_stats = [
            {"vpu_id": 0, "neuron_count": 10, "cycles_executed": 10, "spikes_generated": 100},  # High activity
            {"vpu_id": 1, "neuron_count": 10, "cycles_executed": 10, "spikes_generated": 10},   # Low activity
            {"vpu_id": 2, "neuron_count": 10, "cycles_executed": 10, "spikes_generated": 90},   # High activity
        ]
        
        hot_slices = orchestrator._identify_hot_slices(vpu_stats, threshold=0.7)
        
        assert len(hot_slices) >= 1  # At least one hot slice
        assert 0 in hot_slices or 2 in hot_slices  # VPU 0 or 2 should be hot
    
    def test_identify_cold_slices(self):
        """Test identification of cold slices."""
        config = {
            "lattice_shape": [10, 10, 5, 4],
            "neuron_model": {
                "type": "LIF",
                "params_default": {
                    "tau_m": 20.0,
                    "v_rest": -65.0,
                    "v_reset": -70.0,
                    "v_threshold": -50.0,
                    "refractory_period": 5.0,
                }
            },
            "cell_lifecycle": {
                "enable_death": False,
                "enable_reproduction": False,
                "max_age": 1000,
            },
            "plasticity": {"learning_rate": 0.01},
            "senses": {},
            "areas": []
        }
        
        model = BrainModel(config=config)
        sim = Simulation(model, use_vnc=False)
        
        orchestrator = AdaptiveVNCOrchestrator(sim, activity_threshold=0.7)
        
        # Mock VPU statistics with varying activity
        vpu_stats = [
            {"vpu_id": 0, "neuron_count": 10, "cycles_executed": 10, "spikes_generated": 100},  # High activity
            {"vpu_id": 1, "neuron_count": 10, "cycles_executed": 10, "spikes_generated": 5},    # Low activity
            {"vpu_id": 2, "neuron_count": 10, "cycles_executed": 10, "spikes_generated": 90},   # High activity
        ]
        
        cold_slices = orchestrator._identify_cold_slices(vpu_stats, threshold=0.7)
        
        assert len(cold_slices) >= 1  # At least one cold slice
        assert 1 in cold_slices  # VPU 1 should be cold
    
    def test_repartition_adaptive(self):
        """Test adaptive repartitioning logic."""
        config = {
            "lattice_shape": [10, 10, 5, 4],
            "neuron_model": {
                "type": "LIF",
                "params_default": {
                    "tau_m": 20.0,
                    "v_rest": -65.0,
                    "v_reset": -70.0,
                    "v_threshold": -50.0,
                    "refractory_period": 5.0,
                }
            },
            "cell_lifecycle": {
                "enable_death": False,
                "enable_reproduction": False,
                "max_age": 1000,
            },
            "plasticity": {"learning_rate": 0.01},
            "senses": {},
            "areas": []
        }
        
        model = BrainModel(config=config)
        sim = Simulation(model, use_vnc=False)
        
        orchestrator = AdaptiveVNCOrchestrator(sim)
        
        # Mock VPU statistics with imbalanced neuron counts
        vpu_stats = [
            {"vpu_id": 0, "neuron_count": 50},  # Overloaded
            {"vpu_id": 1, "neuron_count": 10},  # Underloaded
            {"vpu_id": 2, "neuron_count": 15},  # Normal
        ]
        
        result = orchestrator._repartition_adaptive(vpu_stats)
        
        assert result["success"] is True
        assert "overloaded" in result
        assert "underloaded" in result
    
    def test_apply_compute_priority(self):
        """Test compute priority application."""
        config = {
            "lattice_shape": [10, 10, 5, 4],
            "neuron_model": {
                "type": "LIF",
                "params_default": {
                    "tau_m": 20.0,
                    "v_rest": -65.0,
                    "v_reset": -70.0,
                    "v_threshold": -50.0,
                    "refractory_period": 5.0,
                }
            },
            "cell_lifecycle": {
                "enable_death": False,
                "enable_reproduction": False,
                "max_age": 1000,
            },
            "plasticity": {"learning_rate": 0.01},
            "senses": {},
            "areas": []
        }
        
        model = BrainModel(config=config)
        sim = Simulation(model, use_vnc=False)
        
        orchestrator = AdaptiveVNCOrchestrator(sim)
        
        hot_slices = [0, 2, 5]
        result = orchestrator._apply_compute_priority(hot_slices)
        
        assert result["success"] is True
        assert result["hot_slices"] == 3
    
    def test_get_performance_summary(self):
        """Test getting performance summary."""
        config = {
            "lattice_shape": [10, 10, 5, 4],
            "neuron_model": {
                "type": "LIF",
                "params_default": {
                    "tau_m": 20.0,
                    "v_rest": -65.0,
                    "v_reset": -70.0,
                    "v_threshold": -50.0,
                    "refractory_period": 5.0,
                }
            },
            "cell_lifecycle": {
                "enable_death": False,
                "enable_reproduction": False,
                "max_age": 1000,
            },
            "plasticity": {"learning_rate": 0.01},
            "senses": {},
            "areas": []
        }
        
        model = BrainModel(config=config)
        sim = Simulation(model, use_vnc=False)
        
        orchestrator = AdaptiveVNCOrchestrator(sim)
        
        # Add some performance log entries
        orchestrator.performance_log.append({
            "cycle": 0,
            "load_imbalance": 0.1,
            "hot_slices": 2,
            "cold_slices": 1,
            "actions": []
        })
        orchestrator.performance_log.append({
            "cycle": 100,
            "load_imbalance": 0.3,
            "hot_slices": 3,
            "cold_slices": 0,
            "actions": ["repartition"]
        })
        
        summary = orchestrator.get_performance_summary()
        
        assert summary["monitoring_cycles"] == 2
        assert "avg_load_imbalance" in summary
        assert "max_load_imbalance" in summary
        assert "recent_actions" in summary
    
    def test_reset_statistics(self):
        """Test resetting orchestrator statistics."""
        config = {
            "lattice_shape": [10, 10, 5, 4],
            "neuron_model": {
                "type": "LIF",
                "params_default": {
                    "tau_m": 20.0,
                    "v_rest": -65.0,
                    "v_reset": -70.0,
                    "v_threshold": -50.0,
                    "refractory_period": 5.0,
                }
            },
            "cell_lifecycle": {
                "enable_death": False,
                "enable_reproduction": False,
                "max_age": 1000,
            },
            "plasticity": {"learning_rate": 0.01},
            "senses": {},
            "areas": []
        }
        
        model = BrainModel(config=config)
        sim = Simulation(model, use_vnc=False)
        
        orchestrator = AdaptiveVNCOrchestrator(sim)
        
        # Add some data
        orchestrator.performance_log.append({"test": "data"})
        orchestrator.optimization_history.append({"test": "opt"})
        orchestrator.total_repartitions = 5
        orchestrator.total_priority_adjustments = 3
        
        # Reset
        orchestrator.reset_statistics()
        
        assert len(orchestrator.performance_log) == 0
        assert len(orchestrator.optimization_history) == 0
        assert orchestrator.total_repartitions == 0
        assert orchestrator.total_priority_adjustments == 0
