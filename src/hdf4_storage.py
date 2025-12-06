"""HDF4 storage for brain model persistence."""

import json
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    try:
        from .brain_model import BrainModel
    except ImportError:
        from brain_model import BrainModel


def save_to_hdf4(model: "BrainModel", filepath: str) -> None:
    """Save brain model to HDF4 file.

    Stores:
    - meta_json: JSON string with full configuration
    - neurons: 2D array of neuron data
    - synapses: 2D array of synapse data

    Args:
        model: The brain model to save.
        filepath: Path to the HDF4 file to create.
    """
    from pyhdf.SD import SD, SDC

    # Create HDF4 file
    hdf = SD(filepath, SDC.WRITE | SDC.CREATE)

    try:
        # 1) Save meta_json (configuration as JSON string)
        meta_dict = model.to_dict()
        meta_str = json.dumps(meta_dict, ensure_ascii=False)
        meta_bytes = np.frombuffer(meta_str.encode("utf-8"), dtype=np.uint8)

        meta_ds = hdf.create("meta_json", SDC.UINT8, (meta_bytes.shape[0],))
        meta_ds[:] = meta_bytes
        meta_ds.endaccess()

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

            neurons_ds = hdf.create(
                "neurons", SDC.FLOAT32, (n_neurons, 10)
            )
            neurons_ds[:] = neurons_data
            neurons_ds.endaccess()

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

            synapses_ds = hdf.create(
                "synapses", SDC.FLOAT32, (n_synapses, 5)
            )
            synapses_ds[:] = synapses_data
            synapses_ds.endaccess()

    finally:
        hdf.end()


def load_from_hdf4(filepath: str) -> "BrainModel":
    """Load brain model from HDF4 file.

    Args:
        filepath: Path to the HDF4 file.

    Returns:
        Loaded BrainModel instance.
    """
    from pyhdf.SD import SD, SDC
    try:
        from .brain_model import BrainModel, Neuron, Synapse
    except ImportError:
        from brain_model import BrainModel, Neuron, Synapse

    hdf = SD(filepath, SDC.READ)

    try:
        # 1) Load meta_json
        meta_ds = hdf.select("meta_json")
        meta_bytes = meta_ds[:]
        meta_ds.endaccess()

        meta_str = bytes(meta_bytes).decode("utf-8")
        meta_dict = json.loads(meta_str)

        # Create model from the loaded data
        model = BrainModel.from_dict(meta_dict)

        # 2) Load neurons (if the array was stored, it updates existing data)
        try:
            neurons_ds = hdf.select("neurons")
            neurons_data = neurons_ds[:]
            neurons_ds.endaccess()

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
        except Exception:
            pass  # No neurons dataset

        # 3) Load synapses
        try:
            synapses_ds = hdf.select("synapses")
            synapses_data = synapses_ds[:]
            synapses_ds.endaccess()

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
        except Exception:
            pass  # No synapses dataset

        return model

    finally:
        hdf.end()


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
