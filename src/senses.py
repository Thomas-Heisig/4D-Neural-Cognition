"""Sense input processing and brain area mapping."""

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    try:
        from .brain_model import BrainModel
    except ImportError:
        from brain_model import BrainModel


def coord_to_id(x: int, y: int, z: int, w: int, lattice_shape: tuple) -> int:
    """Convert 4D coordinates to a linear ID.

    Args:
        x, y, z, w: 4D coordinates.
        lattice_shape: Shape of the lattice (x_size, y_size, z_size, w_size).

    Returns:
        Linear index corresponding to the coordinates.
    """
    return (
        w * lattice_shape[0] * lattice_shape[1] * lattice_shape[2]
        + z * lattice_shape[0] * lattice_shape[1]
        + y * lattice_shape[0]
        + x
    )


def id_to_coord(neuron_id: int, lattice_shape: tuple) -> tuple:
    """Convert a linear ID back to 4D coordinates.

    Args:
        neuron_id: Linear index.
        lattice_shape: Shape of the lattice (x_size, y_size, z_size, w_size).

    Returns:
        Tuple (x, y, z, w) of coordinates.
    """
    x_size, y_size, z_size, w_size = lattice_shape

    w = neuron_id // (x_size * y_size * z_size)
    remainder = neuron_id % (x_size * y_size * z_size)

    z = remainder // (x_size * y_size)
    remainder = remainder % (x_size * y_size)

    y = remainder // x_size
    x = remainder % x_size

    return (x, y, z, w)


def feed_sense_input(
    model: "BrainModel",
    sense_name: str,
    input_matrix: np.ndarray,
    z_layer: int = 0,
) -> None:
    """Feed sensory input to neurons in the corresponding brain area.

    Maps input values from a 2D matrix to neurons in the area assigned
    to the given sense. Validates input dimensions and provides clear
    error messages for mismatches.

    Args:
        model: The brain model.
        sense_name: Name of the sense (e.g., "vision", "digital").
        input_matrix: 2D numpy array of input values.
        z_layer: Which z-layer to project the input to (default: 0).

    Raises:
        ValueError: If sense_name is unknown, area not found, or input
                   dimensions are invalid.
        TypeError: If input_matrix is not a numpy array.
    """
    # Validate input type
    if not isinstance(input_matrix, np.ndarray):
        raise TypeError(f"input_matrix must be a numpy array, got {type(input_matrix).__name__}")

    # Validate input is 2D
    if input_matrix.ndim != 2:
        raise ValueError(
            f"input_matrix must be 2D, got shape {input_matrix.shape} " f"with {input_matrix.ndim} dimensions"
        )

    senses = model.get_senses()
    areas = model.get_areas()

    if sense_name not in senses:
        available_senses = ", ".join(senses.keys())
        raise ValueError(f"Unknown sense: '{sense_name}'. Available senses: {available_senses}")

    sense = senses[sense_name]
    area_name = sense["areal"]
    area = next((a for a in areas if a["name"] == area_name), None)

    if area is None:
        raise ValueError(
            f"Area '{area_name}' not found for sense '{sense_name}'. "
            f"Check configuration for missing area definition."
        )

    # Get coordinate ranges for the sensory area
    ranges = area["coord_ranges"]
    x_range = ranges["x"]
    y_range = ranges["y"]
    z_range = ranges["z"]
    w_range = ranges["w"]

    # Calculate expected input dimensions based on area size
    expected_x_size = x_range[1] - x_range[0] + 1
    expected_y_size = y_range[1] - y_range[0] + 1

    # Validate z_layer parameter
    z_depth = z_range[1] - z_range[0] + 1
    if z_layer < 0 or z_layer >= z_depth:
        raise ValueError(
            f"z_layer {z_layer} out of range for sense '{sense_name}'. " f"Valid range: [0, {z_depth - 1}]"
        )

    # Use fixed z and w coordinates for 2D input projection
    z_fixed = z_range[0] + z_layer
    w_fixed = w_range[0]

    # Validate input dimensions match area dimensions
    # Provide helpful error message if dimensions don't match
    if input_matrix.shape[0] != expected_x_size or input_matrix.shape[1] != expected_y_size:
        # Log warning but allow partial mapping for flexibility
        warnings.warn(
            f"Input dimension mismatch for sense '{sense_name}': "
            f"expected ({expected_x_size}, {expected_y_size}), "
            f"got {input_matrix.shape}. Will map overlapping region only.",
            UserWarning,
        )

    # Create coordinate to neuron ID mapping for efficient lookup
    coord_map = model.coord_to_id_map()

    # Map input values to neurons in the sensory area
    # Only map the overlapping region if sizes don't match
    x_size = min(input_matrix.shape[0], expected_x_size)
    y_size = min(input_matrix.shape[1], expected_y_size)

    for ix in range(x_size):
        for iy in range(y_size):
            # Calculate absolute neuron coordinates
            x = x_range[0] + ix
            y = y_range[0] + iy
            coord = (x, y, z_fixed, w_fixed)

            # Inject input current if neuron exists at this position
            if coord in coord_map:
                neuron_id = coord_map[coord]
                if neuron_id in model.neurons:
                    model.neurons[neuron_id].external_input += input_matrix[ix, iy]


def get_area_input_neurons(
    model: "BrainModel",
    sense_name: str,
    z_layer: int = 0,
) -> list[Any]:
    """Get the neurons in the input layer of a sensory area.

    Args:
        model: The brain model.
        sense_name: Name of the sense.
        z_layer: Which z-layer to get neurons from.

    Returns:
        List of neurons in the input layer.
    """
    senses = model.get_senses()
    areas = model.get_areas()

    if sense_name not in senses:
        return []

    sense = senses[sense_name]
    area_name = sense["areal"]
    area = next((a for a in areas if a["name"] == area_name), None)

    if area is None:
        return []

    ranges = area["coord_ranges"]
    z_fixed = ranges["z"][0] + z_layer
    w_fixed = ranges["w"][0]

    if z_fixed > ranges["z"][1]:
        z_fixed = ranges["z"][1]

    neurons = []
    for neuron in model.neurons.values():
        if (
            ranges["x"][0] <= neuron.x <= ranges["x"][1]
            and ranges["y"][0] <= neuron.y <= ranges["y"][1]
            and neuron.z == z_fixed
            and neuron.w == w_fixed
        ):
            neurons.append(neuron)

    return neurons


def create_digital_sense_input(
    data: bytes | str | list,
    target_shape: tuple = (20, 20),
) -> np.ndarray:
    """Convert digital data to a format suitable for the digital sense.

    Args:
        data: Input data (bytes, string, or list of values).
        target_shape: Desired output shape.

    Returns:
        2D numpy array of normalized values.
    """
    if isinstance(data, str):
        data = data.encode("utf-8")

    if isinstance(data, bytes):
        # Convert bytes to float array normalized to [0, 1]
        values = np.array([b / 255.0 for b in data])
    elif isinstance(data, list):
        values = np.array(data, dtype=float)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    # Resize to target shape
    total_size = target_shape[0] * target_shape[1]
    if len(values) < total_size:
        # Pad with zeros
        padded = np.zeros(total_size)
        padded[: len(values)] = values
        values = padded
    else:
        # Truncate
        values = values[:total_size]

    return values.reshape(target_shape)
