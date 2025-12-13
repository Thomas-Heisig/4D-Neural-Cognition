"""4D Neural Cognition - Brain Model Package."""

from .brain_model import BrainModel, Neuron
from .cell_lifecycle import maybe_kill_and_reproduce, mutate_params, mutate_weight
from .plasticity import apply_weight_decay, hebbian_update
from .senses import coord_to_id, feed_sense_input, id_to_coord
from .simulation import Simulation
from .storage import load_from_hdf5, load_from_json, save_to_hdf5, save_to_json
from .learning_systems import (
    LearningCategory,
    LearningContext,
    LearningResult,
    LearningSystem,
    LearningSystemManager,
    AssociativeLearning,
    NonAssociativeLearning,
    OperantConditioning,
    SupervisedLearning,
    UnsupervisedLearning,
    ReinforcementLearning,
    TransferLearning,
    MetaLearning,
    create_default_learning_systems,
)

# Import new modules (optional imports to avoid hard dependencies)
try:
    from .model_comparison import ModelComparator, AblationStudy, ModelResult
    _model_comparison_available = True
except ImportError:
    _model_comparison_available = False

try:
    from .video_export import VideoExporter, SimulationRecorder
    _video_export_available = True
except ImportError:
    _video_export_available = False

__all__ = [
    "BrainModel",
    "Neuron",
    "maybe_kill_and_reproduce",
    "mutate_params",
    "mutate_weight",
    "hebbian_update",
    "apply_weight_decay",
    "feed_sense_input",
    "coord_to_id",
    "id_to_coord",
    "save_to_hdf5",
    "load_from_hdf5",
    "save_to_json",
    "load_from_json",
    "Simulation",
    "LearningCategory",
    "LearningContext",
    "LearningResult",
    "LearningSystem",
    "LearningSystemManager",
    "AssociativeLearning",
    "NonAssociativeLearning",
    "OperantConditioning",
    "SupervisedLearning",
    "UnsupervisedLearning",
    "ReinforcementLearning",
    "TransferLearning",
    "MetaLearning",
    "create_default_learning_systems",
]

# Add optional exports if available
if _model_comparison_available:
    __all__.extend(["ModelComparator", "AblationStudy", "ModelResult"])

if _video_export_available:
    __all__.extend(["VideoExporter", "SimulationRecorder"])
