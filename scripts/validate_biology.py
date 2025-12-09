#!/usr/bin/env python3
"""Validation script for biological plausibility of neural models.

This script checks that the neural network implementation follows
biological constraints and principles.
"""

import sys
import os
from typing import Dict, List, Tuple
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.brain_model import BrainModel


class BiologicalValidator:
    """Validator for biological plausibility checks."""
    
    def __init__(self, model: BrainModel):
        """Initialize validator.
        
        Args:
            model: BrainModel to validate
        """
        self.model = model
        self.violations: List[str] = []
        self.warnings: List[str] = []
    
    def validate_dales_law(self) -> bool:
        """Validate Dale's Law: neurons are either excitatory or inhibitory.
        
        Returns:
            True if Dale's Law is followed
        """
        print("Checking Dale's Law...")
        
        # Check that each neuron has consistent synapse types
        neuron_synapse_types: Dict[int, set] = {}
        
        for syn in self.model.synapses:
            pre_id = syn.pre_id
            syn_type = syn.synapse_type
            
            if pre_id not in neuron_synapse_types:
                neuron_synapse_types[pre_id] = set()
            
            neuron_synapse_types[pre_id].add(syn_type)
        
        # Check for violations
        violations = []
        for neuron_id, types in neuron_synapse_types.items():
            if len(types) > 1:
                violations.append(f"Neuron {neuron_id} has mixed synapse types: {types}")
        
        if violations:
            self.violations.extend(violations)
            print(f"  ❌ FAILED: {len(violations)} violations found")
            for v in violations[:5]:  # Show first 5
                print(f"     - {v}")
            if len(violations) > 5:
                print(f"     ... and {len(violations) - 5} more")
            return False
        
        print("  ✅ PASSED: All neurons follow Dale's Law")
        return True
    
    def validate_connection_probabilities(self) -> bool:
        """Validate that connection probabilities are biologically realistic.
        
        Returns:
            True if connection probabilities are plausible
        """
        print("Checking connection probabilities...")
        
        num_neurons = len(self.model.neurons)
        num_synapses = len(self.model.synapses)
        
        if num_neurons == 0:
            self.warnings.append("No neurons in model")
            print("  ⚠️  WARNING: No neurons in model")
            return True
        
        # Calculate connection probability
        max_connections = num_neurons * (num_neurons - 1)
        connection_prob = num_synapses / max_connections if max_connections > 0 else 0
        
        print(f"  Connection probability: {connection_prob:.4f}")
        
        # Biological cortex typically has 0.01-0.1 connection probability
        if connection_prob > 0.5:
            self.warnings.append(
                f"Connection probability ({connection_prob:.4f}) is higher than typical cortex (0.01-0.1)"
            )
            print(f"  ⚠️  WARNING: High connection probability")
        elif connection_prob < 0.001:
            self.warnings.append(
                f"Connection probability ({connection_prob:.6f}) is very sparse"
            )
            print(f"  ⚠️  WARNING: Very sparse connectivity")
        else:
            print(f"  ✅ PASSED: Connection probability is reasonable")
        
        return True
    
    def validate_firing_rates(
        self,
        spike_counts: Dict[int, int],
        simulation_time: float
    ) -> bool:
        """Validate that firing rates are biologically realistic.
        
        Args:
            spike_counts: Dictionary mapping neuron IDs to spike counts
            simulation_time: Total simulation time in ms
            
        Returns:
            True if firing rates are plausible
        """
        print("Checking firing rates...")
        
        if not spike_counts or simulation_time <= 0:
            self.warnings.append("No spike data available")
            print("  ⚠️  WARNING: No spike data to validate")
            return True
        
        # Calculate firing rates (Hz)
        firing_rates = []
        for neuron_id, count in spike_counts.items():
            rate = (count / simulation_time) * 1000  # Convert to Hz
            firing_rates.append(rate)
        
        if not firing_rates:
            return True
        
        mean_rate = np.mean(firing_rates)
        max_rate = np.max(firing_rates)
        
        print(f"  Mean firing rate: {mean_rate:.2f} Hz")
        print(f"  Max firing rate: {max_rate:.2f} Hz")
        
        # Biological cortical neurons typically fire at 1-100 Hz
        if mean_rate > 200:
            self.violations.append(
                f"Mean firing rate ({mean_rate:.2f} Hz) is unrealistically high (typical: 1-100 Hz)"
            )
            print(f"  ❌ FAILED: Mean firing rate too high")
            return False
        elif mean_rate < 0.1:
            self.warnings.append(
                f"Mean firing rate ({mean_rate:.2f} Hz) is very low"
            )
            print(f"  ⚠️  WARNING: Mean firing rate very low")
        
        if max_rate > 1000:
            self.violations.append(
                f"Max firing rate ({max_rate:.2f} Hz) exceeds biological maximum (~1000 Hz)"
            )
            print(f"  ❌ FAILED: Max firing rate too high")
            return False
        
        print("  ✅ PASSED: Firing rates are reasonable")
        return True
    
    def validate_synaptic_weights(self) -> bool:
        """Validate that synaptic weights are in reasonable range.
        
        Returns:
            True if weights are plausible
        """
        print("Checking synaptic weights...")
        
        if not self.model.synapses:
            self.warnings.append("No synapses in model")
            print("  ⚠️  WARNING: No synapses to validate")
            return True
        
        weights = [syn.weight for syn in self.model.synapses]
        
        mean_weight = np.mean(np.abs(weights))
        max_weight = np.max(np.abs(weights))
        min_weight = np.min(np.abs(weights))
        
        print(f"  Mean |weight|: {mean_weight:.4f}")
        print(f"  Max |weight|: {max_weight:.4f}")
        print(f"  Min |weight|: {min_weight:.4f}")
        
        # Check for unreasonable values
        if max_weight > 100:
            self.violations.append(
                f"Max synaptic weight ({max_weight:.2f}) is unrealistically high"
            )
            print(f"  ❌ FAILED: Weights too large")
            return False
        
        if mean_weight < 0.001:
            self.warnings.append(
                f"Mean synaptic weight ({mean_weight:.6f}) is very small"
            )
            print(f"  ⚠️  WARNING: Weights very small")
        
        print("  ✅ PASSED: Synaptic weights are reasonable")
        return True
    
    def validate_excitatory_inhibitory_balance(self) -> bool:
        """Validate E/I balance (typically ~80% excitatory, ~20% inhibitory).
        
        Returns:
            True if balance is reasonable
        """
        print("Checking excitatory/inhibitory balance...")
        
        if not self.model.neurons:
            self.warnings.append("No neurons in model")
            print("  ⚠️  WARNING: No neurons to validate")
            return True
        
        excitatory_count = 0
        inhibitory_count = 0
        
        for neuron in self.model.neurons.values():
            if neuron.is_excitatory():
                excitatory_count += 1
            elif neuron.is_inhibitory():
                inhibitory_count += 1
        
        total = excitatory_count + inhibitory_count
        
        if total == 0:
            self.warnings.append("No typed neurons (all unspecified)")
            print("  ⚠️  WARNING: No typed neurons")
            return True
        
        exc_ratio = excitatory_count / total
        inh_ratio = inhibitory_count / total
        
        print(f"  Excitatory: {exc_ratio*100:.1f}% ({excitatory_count} neurons)")
        print(f"  Inhibitory: {inh_ratio*100:.1f}% ({inhibitory_count} neurons)")
        
        # Biological cortex is typically 70-85% excitatory
        if exc_ratio < 0.6 or exc_ratio > 0.95:
            self.warnings.append(
                f"E/I ratio ({exc_ratio*100:.1f}% excitatory) differs from typical cortex (70-85%)"
            )
            print(f"  ⚠️  WARNING: E/I balance differs from typical cortex")
        else:
            print("  ✅ PASSED: E/I balance is reasonable")
        
        return True
    
    def validate_neuron_parameters(self) -> bool:
        """Validate neuron model parameters are biologically plausible.
        
        Returns:
            True if parameters are plausible
        """
        print("Checking neuron parameters...")
        
        config = self.model.config
        
        # Check LIF parameters if present
        if 'neuron_model' in config:
            nm = config['neuron_model']
            
            issues = []
            
            # Membrane time constant (typical: 5-30 ms)
            if 'tau_m' in nm:
                tau_m = nm['tau_m']
                if tau_m < 1 or tau_m > 100:
                    issues.append(f"tau_m={tau_m} outside typical range (5-30 ms)")
            
            # Resting potential (typical: -70 to -60 mV)
            if 'v_rest' in nm:
                v_rest = nm['v_rest']
                if v_rest < -80 or v_rest > -55:
                    issues.append(f"v_rest={v_rest} outside typical range (-70 to -60 mV)")
            
            # Threshold (typical: -55 to -45 mV)
            if 'v_threshold' in nm:
                v_thresh = nm['v_threshold']
                if v_thresh < -60 or v_thresh > -40:
                    issues.append(f"v_threshold={v_thresh} outside typical range (-55 to -45 mV)")
            
            if issues:
                self.warnings.extend(issues)
                print(f"  ⚠️  WARNING: Some parameters outside typical ranges")
                for issue in issues:
                    print(f"     - {issue}")
            else:
                print("  ✅ PASSED: Neuron parameters are reasonable")
        
        return True
    
    def run_all_validations(
        self,
        spike_counts: Dict[int, int] = None,
        simulation_time: float = None
    ) -> Tuple[bool, List[str], List[str]]:
        """Run all validation checks.
        
        Args:
            spike_counts: Optional spike count data
            simulation_time: Optional simulation time
            
        Returns:
            Tuple of (all_passed, violations, warnings)
        """
        print("\n" + "="*60)
        print("BIOLOGICAL PLAUSIBILITY VALIDATION")
        print("="*60 + "\n")
        
        checks = [
            self.validate_dales_law(),
            self.validate_connection_probabilities(),
            self.validate_synaptic_weights(),
            self.validate_excitatory_inhibitory_balance(),
            self.validate_neuron_parameters()
        ]
        
        if spike_counts and simulation_time:
            checks.append(self.validate_firing_rates(spike_counts, simulation_time))
        
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        all_passed = all(checks)
        
        if all_passed:
            print("✅ ALL CHECKS PASSED")
        else:
            print("❌ SOME CHECKS FAILED")
        
        if self.violations:
            print(f"\n🔴 {len(self.violations)} VIOLATION(S):")
            for v in self.violations:
                print(f"  - {v}")
        
        if self.warnings:
            print(f"\n⚠️  {len(self.warnings)} WARNING(S):")
            for w in self.warnings:
                print(f"  - {w}")
        
        print("\n" + "="*60 + "\n")
        
        return all_passed, self.violations, self.warnings


def main():
    """Main validation script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate biological plausibility of brain model"
    )
    parser.add_argument(
        'config_path',
        type=str,
        default='brain_base_model.json',
        nargs='?',
        help='Path to brain model configuration'
    )
    
    args = parser.parse_args()
    
    # Load model
    try:
        model = BrainModel(config_path=args.config_path)
        print(f"Loaded model from: {args.config_path}")
        print(f"Neurons: {len(model.neurons)}")
        print(f"Synapses: {len(model.synapses)}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Run validation
    validator = BiologicalValidator(model)
    all_passed, violations, warnings = validator.run_all_validations()
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
