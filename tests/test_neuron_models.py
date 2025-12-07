"""Unit tests for neuron_models module."""

import pytest
import numpy as np
from src.brain_model import Neuron
from src.neuron_models import (
    update_lif_neuron,
    update_izhikevich_neuron,
    get_izhikevich_parameters,
    update_neuron,
    create_balanced_network_types,
    get_synapse_type_from_presynaptic,
    calculate_excitation_inhibition_balance,
)


class TestLIFNeuron:
    """Test Leaky Integrate-and-Fire neuron model."""

    def test_update_lif_no_input(self):
        """Test LIF neuron with no input decays to rest."""
        neuron = Neuron(id=0, x=0, y=0, z=0, w=0)
        neuron.v_membrane = -60.0  # Above rest
        neuron.external_input = 0.0
        neuron.params = {"v_rest": -65.0, "tau_membrane": 10.0}

        spiked, new_v = update_lif_neuron(neuron, synaptic_input=0.0, dt=1.0)

        assert not spiked
        assert new_v < -60.0  # Should decay toward rest
        assert new_v > -65.0  # But not reach it in one step

    def test_update_lif_with_input_spike(self):
        """Test LIF neuron spikes with sufficient input."""
        neuron = Neuron(id=0, x=0, y=0, z=0, w=0)
        neuron.v_membrane = -55.0  # Close to threshold
        neuron.external_input = 0.0
        neuron.params = {
            "v_rest": -65.0,
            "v_threshold": -50.0,
            "v_reset": -65.0,
            "tau_membrane": 10.0,
        }

        # Provide strong input
        spiked, new_v = update_lif_neuron(neuron, synaptic_input=10.0, dt=1.0)

        assert spiked
        assert new_v == -65.0  # Should reset

    def test_update_lif_no_spike(self):
        """Test LIF neuron doesn't spike with weak input."""
        neuron = Neuron(id=0, x=0, y=0, z=0, w=0)
        neuron.v_membrane = -65.0
        neuron.external_input = 0.0
        neuron.params = {"v_threshold": -50.0}

        spiked, new_v = update_lif_neuron(neuron, synaptic_input=1.0, dt=1.0)

        assert not spiked
        assert new_v > -65.0  # Should increase
        assert new_v < -50.0  # But not reach threshold

    def test_update_lif_external_input(self):
        """Test LIF neuron with external input."""
        neuron = Neuron(id=0, x=0, y=0, z=0, w=0)
        neuron.v_membrane = -65.0
        neuron.external_input = 5.0
        neuron.params = {}

        spiked, new_v = update_lif_neuron(neuron, synaptic_input=0.0, dt=1.0)

        assert not spiked
        assert new_v > -65.0  # Should increase due to external input


class TestIzhikevichNeuron:
    """Test Izhikevich neuron model."""

    def test_update_izhikevich_regular_spiking(self):
        """Test Izhikevich neuron with regular spiking parameters."""
        neuron = Neuron(id=0, x=0, y=0, z=0, w=0)
        neuron.neuron_type = "regular_spiking"
        neuron.v_membrane = 25.0  # Start near threshold
        neuron.u_recovery = -13.0
        neuron.external_input = 0.0
        neuron.params = {}

        # Provide input to cause spike
        spiked, new_v, new_u = update_izhikevich_neuron(
            neuron, synaptic_input=20.0, dt=1.0
        )

        assert spiked
        assert new_v == -65.0  # Reset voltage (c parameter)
        assert new_u > -13.0  # Recovery variable should increase

    def test_update_izhikevich_fast_spiking(self):
        """Test Izhikevich neuron with fast spiking parameters."""
        neuron = Neuron(id=0, x=0, y=0, z=0, w=0)
        neuron.neuron_type = "fast_spiking"
        neuron.v_membrane = -65.0
        neuron.u_recovery = -13.0
        neuron.external_input = 0.0
        neuron.params = {}

        # Update without spike
        spiked, new_v, new_u = update_izhikevich_neuron(
            neuron, synaptic_input=10.0, dt=1.0
        )

        assert not spiked
        assert new_v > -65.0

    def test_update_izhikevich_bursting(self):
        """Test Izhikevich neuron with bursting parameters."""
        neuron = Neuron(id=0, x=0, y=0, z=0, w=0)
        neuron.neuron_type = "bursting"
        neuron.v_membrane = -65.0
        neuron.u_recovery = -13.0
        neuron.external_input = 0.0
        neuron.params = {}

        spiked, new_v, new_u = update_izhikevich_neuron(
            neuron, synaptic_input=20.0, dt=1.0
        )

        # Should be able to update without error
        assert isinstance(spiked, bool)
        assert isinstance(new_v, float)
        assert isinstance(new_u, float)

    def test_update_izhikevich_no_spike(self):
        """Test Izhikevich neuron without spike."""
        neuron = Neuron(id=0, x=0, y=0, z=0, w=0)
        neuron.neuron_type = "regular_spiking"
        neuron.v_membrane = -65.0
        neuron.u_recovery = -13.0
        neuron.external_input = 0.0
        neuron.params = {}

        spiked, new_v, new_u = update_izhikevich_neuron(
            neuron, synaptic_input=0.0, dt=1.0
        )

        assert not spiked
        assert new_v < 30.0  # Below spike threshold


class TestGetIzhikevichParameters:
    """Test Izhikevich parameter selection."""

    def test_regular_spiking_params(self):
        """Test regular spiking parameters."""
        a, b, c, d = get_izhikevich_parameters("regular_spiking")

        assert a == 0.02
        assert b == 0.2
        assert c == -65.0
        assert d == 8.0

    def test_fast_spiking_params(self):
        """Test fast spiking parameters."""
        a, b, c, d = get_izhikevich_parameters("fast_spiking")

        assert a == 0.1
        assert b == 0.2
        assert c == -65.0
        assert d == 2.0

    def test_bursting_params(self):
        """Test bursting parameters."""
        a, b, c, d = get_izhikevich_parameters("bursting")

        assert a == 0.02
        assert b == 0.2
        assert c == -55.0
        assert d == 4.0

    def test_inhibitory_params(self):
        """Test inhibitory neuron parameters."""
        a, b, c, d = get_izhikevich_parameters("inhibitory")

        assert a == 0.1
        assert b == 0.2
        assert c == -65.0
        assert d == 2.0

    def test_default_params(self):
        """Test default parameters for unknown type."""
        a, b, c, d = get_izhikevich_parameters("unknown_type")

        # Should return default excitatory neuron params
        assert a == 0.02
        assert b == 0.2
        assert c == -65.0
        assert d == 8.0

    def test_parameter_override(self):
        """Test parameter override."""
        params = {"izh_a": 0.05, "izh_c": -70.0}
        a, b, c, d = get_izhikevich_parameters("regular_spiking", params)

        assert a == 0.05  # Overridden
        assert b == 0.2  # Default
        assert c == -70.0  # Overridden
        assert d == 8.0  # Default


class TestUpdateNeuron:
    """Test generic neuron update function."""

    def test_update_neuron_lif_default(self):
        """Test neuron update with LIF model (default)."""
        neuron = Neuron(id=0, x=0, y=0, z=0, w=0)
        neuron.v_membrane = -65.0
        neuron.external_input = 0.0
        neuron.model_type = "lif"

        spiked = update_neuron(neuron, synaptic_input=10.0, dt=1.0)

        assert isinstance(spiked, bool)
        assert neuron.v_membrane > -65.0  # Should have increased

    def test_update_neuron_izhikevich(self):
        """Test neuron update with Izhikevich model."""
        neuron = Neuron(id=0, x=0, y=0, z=0, w=0)
        neuron.model_type = "izhikevich"
        neuron.neuron_type = "regular_spiking"
        neuron.v_membrane = -65.0
        neuron.u_recovery = -13.0
        neuron.external_input = 0.0

        spiked = update_neuron(neuron, synaptic_input=10.0, dt=1.0)

        assert isinstance(spiked, bool)
        assert hasattr(neuron, "u_recovery")  # Should update recovery variable

    def test_update_neuron_updates_membrane_potential(self):
        """Test that neuron update modifies membrane potential."""
        neuron = Neuron(id=0, x=0, y=0, z=0, w=0)
        neuron.v_membrane = -65.0
        initial_v = neuron.v_membrane

        update_neuron(neuron, synaptic_input=5.0, dt=1.0)

        assert neuron.v_membrane != initial_v


class TestCreateBalancedNetworkTypes:
    """Test balanced network type creation."""

    def test_create_balanced_network_default_fraction(self):
        """Test creating balanced network with default inhibitory fraction."""
        num_neurons = 100
        types = create_balanced_network_types(num_neurons)

        assert len(types) == num_neurons

        # Count types
        num_inhibitory = sum(1 for t in types if t == "fast_spiking")
        num_excitatory = num_neurons - num_inhibitory

        # Should have ~20% inhibitory
        assert 15 <= num_inhibitory <= 25  # Allow some variation

    def test_create_balanced_network_custom_fraction(self):
        """Test creating balanced network with custom inhibitory fraction."""
        num_neurons = 100
        types = create_balanced_network_types(num_neurons, inhibitory_fraction=0.3)

        num_inhibitory = sum(1 for t in types if t == "fast_spiking")

        # Should have ~30% inhibitory
        assert 25 <= num_inhibitory <= 35

    def test_create_balanced_network_has_bursting(self):
        """Test that network includes bursting neurons."""
        num_neurons = 100
        types = create_balanced_network_types(num_neurons)

        num_bursting = sum(1 for t in types if t == "bursting")

        # Should have some bursting neurons
        assert num_bursting > 0

    def test_create_balanced_network_reproducible(self):
        """Test that network creation is reproducible with seed."""
        num_neurons = 50
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        types1 = create_balanced_network_types(num_neurons, rng=rng1)
        types2 = create_balanced_network_types(num_neurons, rng=rng2)

        assert types1 == types2

    def test_create_balanced_network_small(self):
        """Test creating balanced network with small number of neurons."""
        num_neurons = 10
        types = create_balanced_network_types(num_neurons)

        assert len(types) == num_neurons
        # Should still work with small numbers


class TestGetSynapseType:
    """Test synapse type determination."""

    def test_synapse_type_excitatory(self):
        """Test synapse type from excitatory neuron."""
        neuron = Neuron(id=0, x=0, y=0, z=0, w=0)
        neuron.neuron_type = "regular_spiking"

        synapse_type = get_synapse_type_from_presynaptic(neuron)

        assert synapse_type == "excitatory"

    def test_synapse_type_inhibitory(self):
        """Test synapse type from inhibitory neuron."""
        neuron = Neuron(id=0, x=0, y=0, z=0, w=0)
        neuron.neuron_type = "inhibitory"

        synapse_type = get_synapse_type_from_presynaptic(neuron)

        assert synapse_type == "inhibitory"

    def test_synapse_type_fast_spiking(self):
        """Test synapse type from fast spiking neuron."""
        neuron = Neuron(id=0, x=0, y=0, z=0, w=0)
        neuron.neuron_type = "fast_spiking"

        synapse_type = get_synapse_type_from_presynaptic(neuron)

        assert synapse_type == "inhibitory"


class TestExcitationInhibitionBalance:
    """Test E-I balance calculation."""

    def test_balanced_input(self):
        """Test calculation with balanced input."""
        result = calculate_excitation_inhibition_balance(
            excitatory_input=50.0, inhibitory_input=50.0
        )

        assert result["excitatory_input"] == 50.0
        assert result["inhibitory_input"] == 50.0
        assert result["e_fraction"] == pytest.approx(0.5)
        assert result["i_fraction"] == pytest.approx(0.5)
        assert result["balance_ratio"] == pytest.approx(1.0)

    def test_excitation_dominant(self):
        """Test calculation with excitation dominant."""
        result = calculate_excitation_inhibition_balance(
            excitatory_input=80.0, inhibitory_input=20.0
        )

        assert result["e_fraction"] == pytest.approx(0.8)
        assert result["i_fraction"] == pytest.approx(0.2)
        assert result["balance_ratio"] == pytest.approx(4.0)

    def test_inhibition_dominant(self):
        """Test calculation with inhibition dominant."""
        result = calculate_excitation_inhibition_balance(
            excitatory_input=20.0, inhibitory_input=80.0
        )

        assert result["e_fraction"] == pytest.approx(0.2)
        assert result["i_fraction"] == pytest.approx(0.8)
        assert result["balance_ratio"] == pytest.approx(0.25)

    def test_zero_input(self):
        """Test calculation with zero input."""
        result = calculate_excitation_inhibition_balance(
            excitatory_input=0.0, inhibitory_input=0.0
        )

        assert result["e_fraction"] == 0.0
        assert result["i_fraction"] == 0.0
        assert result["balance_ratio"] == 0.0

    def test_only_excitatory(self):
        """Test calculation with only excitatory input."""
        result = calculate_excitation_inhibition_balance(
            excitatory_input=100.0, inhibitory_input=0.0
        )

        assert result["e_fraction"] == pytest.approx(1.0)
        assert result["i_fraction"] == pytest.approx(0.0)
        # Balance ratio should be high but not infinite
        assert result["balance_ratio"] > 0.0

    def test_only_inhibitory(self):
        """Test calculation with only inhibitory input."""
        result = calculate_excitation_inhibition_balance(
            excitatory_input=0.0, inhibitory_input=100.0
        )

        assert result["e_fraction"] == pytest.approx(0.0)
        assert result["i_fraction"] == pytest.approx(1.0)
        assert result["balance_ratio"] == pytest.approx(0.0)
