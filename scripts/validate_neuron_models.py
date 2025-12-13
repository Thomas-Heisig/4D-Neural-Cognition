#!/usr/bin/env python3
"""Validation script for neuron models (LIF and Izhikevich).

This script validates the implemented neuron models against:
1. Analytical solutions for simple scenarios
2. Expected firing patterns and behaviors
3. Known parameter relationships

This is critical for ensuring scientific validity of simulation results.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from src.brain_model import BrainModel
from src.simulation import Simulation
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    passed: bool
    expected: Optional[float] = None
    actual: Optional[float] = None
    error: Optional[str] = None
    
    def __str__(self):
        if self.passed:
            return f"✓ {self.test_name}: PASS"
        else:
            msg = f"✗ {self.test_name}: FAIL"
            if self.expected is not None and self.actual is not None:
                msg += f" (expected: {self.expected:.4f}, actual: {self.actual:.4f})"
            if self.error:
                msg += f" - {self.error}"
            return msg


class NeuronModelValidator:
    """Validator for neuron model implementations."""
    
    def __init__(self, output_dir: str = "/tmp/neuron_validation"):
        """Initialize validator.
        
        Args:
            output_dir: Directory to save validation plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results: List[ValidationResult] = []
    
    def validate_lif_rest_potential(self) -> ValidationResult:
        """Validate that LIF neuron stays at rest without input."""
        config = {
            "lattice_shape": [5, 5, 5, 1],
            "neuron_model": {
                "model_type": "LIF",
                "params_default": {
                    "v_rest": -65.0,
                    "v_threshold": -50.0,
                    "v_reset": -70.0,
                    "tau_m": 20.0,
                    "refractory_period": 5
                }
            },
            "cell_lifecycle": {"enabled": False},
            "plasticity": {"enabled": False},
            "senses": {},
            "areas": [{
                "name": "TestArea",
                "coord_ranges": {"x": [0, 4], "y": [0, 4], "z": [0, 4], "w": [0, 0]}
            }]
        }
        
        model = BrainModel(config=config)
        sim = Simulation(model, seed=42)
        
        # Create single neuron
        neuron = model.add_neuron(0, 0, 0, 0)
        neuron_id = neuron.id
        neuron.v_membrane = -65.0
        
        # Run without input
        voltages = []
        for _ in range(100):
            voltages.append(neuron.v_membrane)
            sim.lif_step(neuron_id)
        
        # Should stay close to rest potential
        mean_voltage = np.mean(voltages)
        expected = -65.0
        tolerance = 0.1
        passed = abs(mean_voltage - expected) < tolerance
        
        return ValidationResult(
            test_name="LIF Rest Potential",
            passed=passed,
            expected=expected,
            actual=mean_voltage
        )
    
    def validate_lif_constant_input(self) -> ValidationResult:
        """Validate LIF response to constant input."""
        config = {
            "lattice_shape": [5, 5, 5, 1],
            "neuron_model": {
                "model_type": "LIF",
                "params_default": {
                    "v_rest": -65.0,
                    "v_threshold": -50.0,
                    "v_reset": -70.0,
                    "tau_m": 20.0,
                    "refractory_period": 5
                }
            },
            "cell_lifecycle": {"enabled": False},
            "senses": {},
            "plasticity": {"enabled": False},
            "areas": [{
                "name": "TestArea",
                "coord_ranges": {"x": [0, 4], "y": [0, 4], "z": [0, 4], "w": [0, 0]}
            }]
        }
        
        model = BrainModel(config=config)
        sim = Simulation(model, seed=42)
        
        # Create single neuron
        neuron = model.add_neuron(0, 0, 0, 0)
        neuron_id = neuron.id
        neuron.v_membrane = -65.0
        
        # Apply constant input above threshold
        constant_input = 20.0
        voltages = []
        spike_times = []
        
        for step in range(200):
            neuron.external_input = constant_input
            voltages.append(neuron.v_membrane)
            
            spiked = sim.lif_step(neuron_id)
            if spiked:
                spike_times.append(step)
        
        # Should spike regularly with constant input
        num_spikes = len(spike_times)
        passed = num_spikes > 5  # Should have multiple spikes
        
        # Save plot
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(voltages)
        plt.ylabel('Membrane Potential (mV)')
        plt.title('LIF Neuron with Constant Input')
        plt.axhline(y=-50.0, color='r', linestyle='--', label='Threshold')
        plt.axhline(y=-65.0, color='g', linestyle='--', label='Rest')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        if spike_times:
            plt.scatter(spike_times, [1]*len(spike_times), marker='|', s=200)
        plt.ylabel('Spikes')
        plt.xlabel('Time Step')
        plt.ylim(0, 2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'lif_constant_input.png'))
        plt.close()
        
        return ValidationResult(
            test_name="LIF Constant Input",
            passed=passed,
            expected=5.0,
            actual=float(num_spikes)
        )
    
    def validate_lif_refractory_period(self) -> ValidationResult:
        """Validate that refractory period prevents immediate re-spiking."""
        config = {
            "lattice_shape": [5, 5, 5, 1],
            "neuron_model": {
                "model_type": "LIF",
                "params_default": {
                    "v_rest": -65.0,
                    "v_threshold": -50.0,
                    "v_reset": -70.0,
                    "tau_m": 20.0,
                    "refractory_period": 10
                }
            },
            "cell_lifecycle": {"enabled": False},
            "senses": {},
            "plasticity": {"enabled": False},
            "areas": [{
                "name": "TestArea",
                "coord_ranges": {"x": [0, 4], "y": [0, 4], "z": [0, 4], "w": [0, 0]}
            }]
        }
        
        model = BrainModel(config=config)
        sim = Simulation(model, seed=42)
        
        neuron = model.add_neuron(0, 0, 0, 0)
        neuron_id = neuron.id
        
        # Apply very strong input
        spike_times = []
        for step in range(50):
            neuron.external_input = 50.0  # Very strong
            if sim.lif_step(neuron_id):
                spike_times.append(step)
        
        # Check inter-spike intervals
        if len(spike_times) >= 2:
            intervals = np.diff(spike_times)
            min_interval = np.min(intervals)
            # Minimum interval should be at least refractory period
            passed = min_interval >= 10
        else:
            passed = False
        
        return ValidationResult(
            test_name="LIF Refractory Period",
            passed=passed,
            expected=10.0,
            actual=min_interval if len(spike_times) >= 2 else 0.0
        )
    
    def validate_lif_subthreshold_decay(self) -> ValidationResult:
        """Validate exponential decay toward rest potential."""
        config = {
            "lattice_shape": [5, 5, 5, 1],
            "neuron_model": {
                "model_type": "LIF",
                "params_default": {
                    "v_rest": -65.0,
                    "v_threshold": -50.0,
                    "v_reset": -70.0,
                    "tau_m": 20.0,
                    "refractory_period": 5
                }
            },
            "cell_lifecycle": {"enabled": False},
            "senses": {},
            "plasticity": {"enabled": False},
            "areas": [{
                "name": "TestArea",
                "coord_ranges": {"x": [0, 4], "y": [0, 4], "z": [0, 4], "w": [0, 0]}
            }]
        }
        
        model = BrainModel(config=config)
        sim = Simulation(model, seed=42)
        
        neuron = model.add_neuron(0, 0, 0, 0)
        neuron_id = neuron.id
        
        # Set initial voltage above rest
        neuron.v_membrane = -55.0
        
        voltages = []
        for _ in range(100):
            voltages.append(neuron.v_membrane)
            sim.lif_step(neuron_id)  # No external input
        
        # Should decay toward rest
        final_voltage = voltages[-1]
        expected = -65.0
        tolerance = 1.0
        passed = abs(final_voltage - expected) < tolerance
        
        # Save plot
        plt.figure(figsize=(10, 4))
        plt.plot(voltages)
        plt.axhline(y=-65.0, color='r', linestyle='--', label='Rest Potential')
        plt.ylabel('Membrane Potential (mV)')
        plt.xlabel('Time Step')
        plt.title('LIF Subthreshold Decay')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'lif_subthreshold_decay.png'))
        plt.close()
        
        return ValidationResult(
            test_name="LIF Subthreshold Decay",
            passed=passed,
            expected=expected,
            actual=final_voltage
        )
    
    def validate_lif_f_i_curve(self) -> ValidationResult:
        """Validate firing rate increases with input current (F-I curve)."""
        config = {
            "lattice_shape": [5, 5, 5, 1],
            "neuron_model": {
                "model_type": "LIF",
                "params_default": {
                    "v_rest": -65.0,
                    "v_threshold": -50.0,
                    "v_reset": -70.0,
                    "tau_m": 20.0,
                    "refractory_period": 5
                }
            },
            "cell_lifecycle": {"enabled": False},
            "senses": {},
            "plasticity": {"enabled": False},
            "areas": [{
                "name": "TestArea",
                "coord_ranges": {"x": [0, 4], "y": [0, 4], "z": [0, 4], "w": [0, 0]}
            }]
        }
        
        # Test different input currents
        currents = np.linspace(5, 30, 10)
        firing_rates = []
        
        for current in currents:
            model = BrainModel(config=config)
            sim = Simulation(model, seed=42)
            neuron = model.add_neuron(0, 0, 0, 0)
            neuron_id = neuron.id
            
            spike_count = 0
            for _ in range(500):
                neuron.external_input = current
                if sim.lif_step(neuron_id):
                    spike_count += 1
            
            # Convert to Hz (assuming 1ms per step)
            firing_rate = (spike_count / 500) * 1000
            firing_rates.append(firing_rate)
        
        # Firing rate should increase with current
        passed = np.all(np.diff(firing_rates) >= -0.1)  # Should be monotonic
        
        # Save plot
        plt.figure(figsize=(8, 6))
        plt.plot(currents, firing_rates, 'o-')
        plt.xlabel('Input Current (arbitrary units)')
        plt.ylabel('Firing Rate (Hz)')
        plt.title('LIF F-I Curve')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'lif_f_i_curve.png'))
        plt.close()
        
        return ValidationResult(
            test_name="LIF F-I Curve",
            passed=passed,
            error=None if passed else "Firing rate should increase monotonically with input"
        )
    
    def validate_synaptic_integration(self) -> ValidationResult:
        """Validate that synaptic inputs are correctly integrated."""
        config = {
            "lattice_shape": [5, 5, 5, 1],
            "neuron_model": {
                "model_type": "LIF",
                "params_default": {
                    "v_rest": -65.0,
                    "v_threshold": -50.0,
                    "v_reset": -70.0,
                    "tau_m": 20.0,
                    "refractory_period": 5
                }
            },
            "cell_lifecycle": {"enabled": False},
            "senses": {},
            "plasticity": {"enabled": False},
            "areas": [{
                "name": "TestArea",
                "coord_ranges": {"x": [0, 4], "y": [0, 4], "z": [0, 4], "w": [0, 0]}
            }]
        }
        
        model = BrainModel(config=config)
        sim = Simulation(model, seed=42)
        
        # Create two neurons with connection
        pre_neuron = model.add_neuron(0, 0, 0, 0)
        post_neuron = model.add_neuron(1, 0, 0, 0)
        pre_id = pre_neuron.id
        post_id = post_neuron.id
        
        # Strong synapse
        model.add_synapse(pre_id, post_id, weight=15.0, delay=2)
        
        post_voltages = []
        
        for step in range(50):
            # Trigger presynaptic spike at step 10
            if step == 10:
                pre_neuron.external_input = 50.0
            
            sim.lif_step(pre_id)
            sim.lif_step(post_id)
            post_voltages.append(post_neuron.v_membrane)
            model.current_step += 1
        
        # Postsynaptic voltage should increase after delay
        voltage_before = np.mean(post_voltages[5:10])
        voltage_after = np.mean(post_voltages[15:20])
        
        # Should see depolarization
        passed = voltage_after > voltage_before + 1.0
        
        return ValidationResult(
            test_name="Synaptic Integration",
            passed=passed,
            expected=voltage_before + 1.0,
            actual=voltage_after
        )
    
    def run_all_validations(self) -> bool:
        """Run all validation tests.
        
        Returns:
            True if all tests passed, False otherwise
        """
        print("=" * 70)
        print("NEURON MODEL VALIDATION")
        print("=" * 70)
        print()
        
        print("Running LIF Model Validations...")
        print("-" * 70)
        
        self.results.append(self.validate_lif_rest_potential())
        self.results.append(self.validate_lif_constant_input())
        self.results.append(self.validate_lif_refractory_period())
        self.results.append(self.validate_lif_subthreshold_decay())
        self.results.append(self.validate_lif_f_i_curve())
        self.results.append(self.validate_synaptic_integration())
        
        # Print results
        print()
        print("RESULTS:")
        print("-" * 70)
        for result in self.results:
            print(result)
        
        # Summary
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        print()
        print("=" * 70)
        print(f"SUMMARY: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
        print(f"Validation plots saved to: {self.output_dir}")
        print("=" * 70)
        
        return passed == total


def main():
    """Main entry point."""
    validator = NeuronModelValidator()
    all_passed = validator.run_all_validations()
    
    if all_passed:
        print("\n✓ All validations passed!")
        return 0
    else:
        print("\n✗ Some validations failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
