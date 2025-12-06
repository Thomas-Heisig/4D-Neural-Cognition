"""Brain model data structures for 4D Neural Cognition."""

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Neuron:
    """Represents a single neuron in the 4D lattice."""

    id: int
    x: int
    y: int
    z: int
    w: int
    generation: int = 0
    parent_id: int = -1
    health: float = 1.0
    age: int = 0
    v_membrane: float = -65.0
    external_input: float = 0.0
    last_spike_time: int = -1000
    params: dict = field(default_factory=dict)

    def position(self) -> tuple:
        """Return the 4D position tuple."""
        return (self.x, self.y, self.z, self.w)


@dataclass
class Synapse:
    """Represents a synaptic connection between two neurons."""

    pre_id: int
    post_id: int
    weight: float = 0.1
    delay: int = 1
    plasticity_tag: float = 0.0


class BrainModel:
    """Main brain model class managing neurons and synapses."""

    def __init__(self, config_path: str = None, config: dict = None):
        """Initialize brain model from config file or dict."""
        if config is not None:
            self.config = config
        elif config_path is not None:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            raise ValueError("Either config_path or config must be provided")

        self.lattice_shape = tuple(self.config["lattice_shape"])
        self.neurons: dict[int, Neuron] = {}
        self.synapses: list[Synapse] = []
        self.current_step: int = 0
        self._next_neuron_id: int = 0

    def get_neuron_model_params(self) -> dict:
        """Get default neuron model parameters."""
        return self.config["neuron_model"]["params_default"].copy()

    def get_lifecycle_config(self) -> dict:
        """Get cell lifecycle configuration."""
        return self.config["cell_lifecycle"]

    def get_plasticity_config(self) -> dict:
        """Get plasticity configuration."""
        return self.config["plasticity"]

    def get_senses(self) -> dict:
        """Get senses configuration."""
        return self.config["senses"]

    def get_areas(self) -> list:
        """Get brain areas configuration."""
        return self.config["areas"]

    def add_neuron(
        self,
        x: int,
        y: int,
        z: int,
        w: int,
        generation: int = 0,
        parent_id: int = -1,
        health: float = 1.0,
        params: dict = None,
    ) -> Neuron:
        """Add a new neuron to the model."""
        neuron = Neuron(
            id=self._next_neuron_id,
            x=x,
            y=y,
            z=z,
            w=w,
            generation=generation,
            parent_id=parent_id,
            health=health,
            v_membrane=self.get_neuron_model_params().get("v_rest", -65.0),
            params=params if params else self.get_neuron_model_params(),
        )
        self.neurons[neuron.id] = neuron
        self._next_neuron_id += 1
        return neuron

    def remove_neuron(self, neuron_id: int) -> None:
        """Remove a neuron from the model."""
        if neuron_id in self.neurons:
            del self.neurons[neuron_id]
            # Remove associated synapses
            self.synapses = [
                s
                for s in self.synapses
                if s.pre_id != neuron_id and s.post_id != neuron_id
            ]

    def add_synapse(
        self,
        pre_id: int,
        post_id: int,
        weight: float = 0.1,
        delay: int = 1,
    ) -> Synapse:
        """Add a synapse between two neurons."""
        synapse = Synapse(pre_id=pre_id, post_id=post_id, weight=weight, delay=delay)
        self.synapses.append(synapse)
        return synapse

    def get_synapses_for_neuron(
        self, neuron_id: int, direction: str = "both"
    ) -> list[Synapse]:
        """Get synapses connected to a neuron.

        Args:
            neuron_id: The neuron ID to get synapses for.
            direction: "pre" for outgoing, "post" for incoming, "both" for all.
        """
        result = []
        for s in self.synapses:
            if direction in ("pre", "both") and s.pre_id == neuron_id:
                result.append(s)
            elif direction in ("post", "both") and s.post_id == neuron_id:
                result.append(s)
        return result

    def coord_to_id_map(self) -> dict[tuple, int]:
        """Create a mapping from 4D coordinates to neuron IDs."""
        return {n.position(): n.id for n in self.neurons.values()}

    def get_area_neurons(self, area_name: str) -> list[Neuron]:
        """Get all neurons belonging to a specific brain area."""
        area = next(
            (a for a in self.get_areas() if a["name"] == area_name), None
        )
        if area is None:
            return []

        ranges = area["coord_ranges"]
        neurons = []
        for neuron in self.neurons.values():
            if (
                ranges["x"][0] <= neuron.x <= ranges["x"][1]
                and ranges["y"][0] <= neuron.y <= ranges["y"][1]
                and ranges["z"][0] <= neuron.z <= ranges["z"][1]
                and ranges["w"][0] <= neuron.w <= ranges["w"][1]
            ):
                neurons.append(neuron)
        return neurons

    def to_dict(self) -> dict:
        """Convert model to dictionary for serialization."""
        return {
            "config": self.config,
            "current_step": self.current_step,
            "next_neuron_id": self._next_neuron_id,
            "neurons": [
                {
                    "id": n.id,
                    "x": n.x,
                    "y": n.y,
                    "z": n.z,
                    "w": n.w,
                    "generation": n.generation,
                    "parent_id": n.parent_id,
                    "health": n.health,
                    "age": n.age,
                    "v_membrane": n.v_membrane,
                    "params": n.params,
                }
                for n in self.neurons.values()
            ],
            "synapses": [
                {
                    "pre_id": s.pre_id,
                    "post_id": s.post_id,
                    "weight": s.weight,
                    "delay": s.delay,
                    "plasticity_tag": s.plasticity_tag,
                }
                for s in self.synapses
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BrainModel":
        """Create model from dictionary."""
        model = cls(config=data["config"])
        model.current_step = data["current_step"]
        model._next_neuron_id = data["next_neuron_id"]

        for n_data in data["neurons"]:
            neuron = Neuron(
                id=n_data["id"],
                x=n_data["x"],
                y=n_data["y"],
                z=n_data["z"],
                w=n_data["w"],
                generation=n_data["generation"],
                parent_id=n_data["parent_id"],
                health=n_data["health"],
                age=n_data["age"],
                v_membrane=n_data["v_membrane"],
                params=n_data["params"],
            )
            model.neurons[neuron.id] = neuron

        for s_data in data["synapses"]:
            synapse = Synapse(
                pre_id=s_data["pre_id"],
                post_id=s_data["post_id"],
                weight=s_data["weight"],
                delay=s_data["delay"],
                plasticity_tag=s_data["plasticity_tag"],
            )
            model.synapses.append(synapse)

        return model
