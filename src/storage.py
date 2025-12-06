"""HDF5 storage for brain model persistence (upgraded from HDF4)."""

import json
import numpy as np
import h5py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    try:
        from .brain_model import BrainModel
    except ImportError:
        from brain_model import BrainModel


def save_to_hdf5(model: "BrainModel", filepath: str) -> None:
    """Save brain model to HDF5 file.

    Stores:
    - meta_json: JSON string with full configuration
    - neurons: 2D array of neuron data
    - synapses: 2D array of synapse data

    Args:
        model: The brain model to save.
        filepath: Path to the HDF5 file to create.
    """
    with h5py.File(filepath, 'w') as hdf:
        # 1) Save meta_json (configuration as JSON string)
        meta_dict = model.to_dict()
        meta_str = json.dumps(meta_dict, ensure_ascii=False)
        
        # Store as string dataset
        hdf.create_dataset('meta_json', data=meta_str, dtype=h5py.string_dtype())

        # 2) Save neurons as structured array
        # Columns: id, x, y, z, w, generation, parent_id, health, age, v_membrane
        n_neurons = len(model.neurons)
        if n_neurons > 0:
            neurons_data = np.zeros((n_neurons, 10), dtype=np.float32)
            for i, neuron in enumerate(model.neurons.values()):
                neurons_data[i] = [
                    neuron.id,
                    neuron.x,
                    neuron.y,
                    neuron.z,
                    neuron.w,
                    neuron.generation,
                    neuron.parent_id,
                    neuron.health,
                    neuron.age,
                    neuron.v_membrane,
                ]

            hdf.create_dataset('neurons', data=neurons_data, compression='gzip')

        # 3) Save synapses as structured array
        # Columns: pre_id, post_id, weight, delay, plasticity_tag
        n_synapses = len(model.synapses)
        if n_synapses > 0:
            synapses_data = np.zeros((n_synapses, 5), dtype=np.float32)
            for i, synapse in enumerate(model.synapses):
                synapses_data[i] = [
                    synapse.pre_id,
                    synapse.post_id,
                    synapse.weight,
                    synapse.delay,
                    synapse.plasticity_tag,
                ]

            hdf.create_dataset('synapses', data=synapses_data, compression='gzip')


def load_from_hdf5(filepath: str) -> "BrainModel":
    """Load brain model from HDF5 file.

    Args:
        filepath: Path to the HDF5 file.

    Returns:
        Loaded BrainModel instance.
    """
    try:
        from .brain_model import BrainModel, Neuron, Synapse
    except ImportError:
        from brain_model import BrainModel, Neuron, Synapse

    with h5py.File(filepath, 'r') as hdf:
        # 1) Load meta_json
        meta_str = hdf['meta_json'][()].decode('utf-8')
        meta_dict = json.loads(meta_str)

        # Create model from the loaded data
        model = BrainModel.from_dict(meta_dict)

        # 2) Load neurons (if the array was stored, it updates existing data)
        if 'neurons' in hdf:
            neurons_data = hdf['neurons'][:]

            # Rebuild neurons from array data
            model.neurons.clear()
            for row in neurons_data:
                neuron_id = int(row[0])
                # Find corresponding params from meta_dict
                neuron_meta = next(
                    (n for n in meta_dict.get("neurons", []) if n["id"] == neuron_id),
                    None,
                )
                params = (
                    neuron_meta["params"]
                    if neuron_meta
                    else model.get_neuron_model_params()
                )

                neuron = Neuron(
                    id=neuron_id,
                    x=int(row[1]),
                    y=int(row[2]),
                    z=int(row[3]),
                    w=int(row[4]),
                    generation=int(row[5]),
                    parent_id=int(row[6]),
                    health=float(row[7]),
                    age=int(row[8]),
                    v_membrane=float(row[9]),
                    params=params,
                )
                model.neurons[neuron_id] = neuron

        # 3) Load synapses
        if 'synapses' in hdf:
            synapses_data = hdf['synapses'][:]

            # Rebuild synapses from array data
            model.synapses.clear()
            for row in synapses_data:
                synapse = Synapse(
                    pre_id=int(row[0]),
                    post_id=int(row[1]),
                    weight=float(row[2]),
                    delay=int(row[3]),
                    plasticity_tag=float(row[4]),
                )
                model.synapses.append(synapse)

        return model


def save_to_json(model: "BrainModel", filepath: str) -> None:
    """Save brain model to JSON file (alternative to HDF4).

    Args:
        model: The brain model to save.
        filepath: Path to the JSON file.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(model.to_dict(), f, ensure_ascii=False, indent=2)


def load_from_json(filepath: str) -> "BrainModel":
    """Load brain model from JSON file.

    Args:
        filepath: Path to the JSON file.

    Returns:
        Loaded BrainModel instance.
    """
    try:
        from .brain_model import BrainModel
    except ImportError:
        from brain_model import BrainModel

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return BrainModel.from_dict(data)
