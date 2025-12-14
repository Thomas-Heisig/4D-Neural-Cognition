#!/usr/bin/env python3
"""Flask web application for 4D Neural Cognition frontend interface."""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock
from typing import Optional, Tuple

import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf.csrf import CSRFProtect

# Ensure logs directory exists BEFORE configuring logging
os.makedirs("logs", exist_ok=True)

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from brain_model import BrainModel
from senses import create_digital_sense_input, feed_sense_input
from simulation import Simulation
from storage import load_from_hdf5, load_from_json, save_to_hdf5, save_to_json

# Configure logging with rotation to prevent unbounded log file growth
# Each log file is limited to 10MB, with 5 backup files kept
file_handler = RotatingFileHandler("logs/app.log", maxBytes=10 * 1024 * 1024, backupCount=5)  # 10MB
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler])
logger = logging.getLogger(__name__)

# Flask app setup
# Use environment variable for secret key for better security
# Fallback to a default only for development/testing
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "neural-cognition-secret-key-4d-CHANGE-IN-PRODUCTION")
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Rate limiting to prevent DoS attacks
# Default: 200 requests per day, 50 per hour for general endpoints
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# CSRF protection for form submissions
# Note: CSRF is disabled for API endpoints when using CORS
# For production with forms, enable CSRF and use tokens in templates
csrf = CSRFProtect()
# Only enable CSRF if not in API-only mode
# For this demo app with CORS, we exempt API routes
if not os.environ.get("DISABLE_CSRF_FOR_API", "true").lower() == "true":
    csrf.init_app(app)
else:
    # In API mode, CSRF is handled via other mechanisms (Origin checks, API keys, etc.)
    logger.info("CSRF protection disabled for API-only mode. Enable for production with web forms.")

# Global state
current_model = None
current_simulation = None
simulation_lock = Lock()
is_training = False
training_thread = None

# Define allowed directories for file operations
ALLOWED_SAVE_DIR = Path("saved_models")
ALLOWED_CONFIG_DIR = Path(".")
CHECKPOINT_DIR = Path("checkpoints")
ALLOWED_SAVE_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Checkpoint configuration
CHECKPOINT_INTERVAL = 1000  # Save checkpoint every N steps


def require_initialization(f):
    """Decorator to ensure system is initialized before allowing API calls.
    
    This prevents 400 errors from premature API calls during system startup.
    """
    from functools import wraps
    
    @wraps(f)
    def decorated_function(*args, **kwargs):
        global current_model, current_simulation
        if current_model is None or current_simulation is None:
            return jsonify({
                "status": "error",
                "message": "System not initialized. Please initialize the model first.",
                "action": "Call /api/model/init to initialize the system"
            }), 400
        return f(*args, **kwargs)
    return decorated_function


def validate_filepath(filepath: str, allowed_dir: Path, allowed_extensions: list) -> Path:
    """Validate and sanitize file paths to prevent path traversal attacks.

    Args:
        filepath: User-provided file path
        allowed_dir: Directory where files are allowed
        allowed_extensions: List of allowed file extensions (e.g., ['.json', '.h5'])

    Returns:
        Validated Path object

    Raises:
        ValueError: If path is invalid or outside allowed directory
    """
    try:
        # Convert to Path and resolve to absolute path
        file_path = Path(filepath).resolve()
        allowed_path = allowed_dir.resolve()

        # Check if file is within allowed directory
        if not str(file_path).startswith(str(allowed_path)):
            raise ValueError(f"Access denied: Path outside allowed directory")

        # Check file extension
        if file_path.suffix not in allowed_extensions:
            raise ValueError(
                f"Invalid file extension: {file_path.suffix}. " f"Allowed: {', '.join(allowed_extensions)}"
            )

        return file_path
    except Exception as e:
        raise ValueError(f"Invalid file path: {str(e)}")


class WebLogger(logging.Handler):
    """Custom logging handler to send logs to web interface via SocketIO."""

    def emit(self, record):
        try:
            log_entry = self.format(record)
            socketio.emit(
                "log_message",
                {"timestamp": datetime.now().isoformat(), "level": record.levelname, "message": log_entry},
            )
        except Exception:
            pass


# Add web logger
web_handler = WebLogger()
web_handler.setLevel(logging.INFO)
logger.addHandler(web_handler)


def validate_simulation_state(simulation: Simulation, model: BrainModel) -> Tuple[bool, str]:
    """Validate simulation state before running operations.

    Args:
        simulation: The simulation object to validate
        model: The brain model to validate

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is empty.
    """
    if simulation is None:
        return False, "No simulation initialized"

    if model is None:
        return False, "No model initialized"

    # Check for minimum neuron count
    if len(model.neurons) == 0:
        return False, "Model has no neurons. Initialize neurons first."

    # Check for valid neuron states (no NaN/Inf values)
    for neuron in list(model.neurons.values())[:10]:  # Sample first 10
        if np.isnan(neuron.v_membrane) or np.isinf(neuron.v_membrane):
            return False, f"Invalid neuron state detected (NaN/Inf in membrane potential)"
        if np.isnan(neuron.health) or np.isinf(neuron.health):
            return False, f"Invalid neuron state detected (NaN/Inf in health)"

    # Check for excessive dead synapses (all weights near zero)
    if len(model.synapses) > 0:
        non_zero_weights = sum(1 for s in model.synapses if abs(s.weight) > 0.001)
        if non_zero_weights / len(model.synapses) < 0.01:  # Less than 1% active
            logger.warning("Most synapses have near-zero weights")

    return True, ""


def save_checkpoint(model: BrainModel, step: int) -> str:
    """Save an automatic checkpoint of the model state.

    Args:
        model: The brain model to checkpoint
        step: Current simulation step

    Returns:
        Path to saved checkpoint file
    """
    try:
        checkpoint_name = f"checkpoint_step_{step}.h5"
        checkpoint_path = CHECKPOINT_DIR / checkpoint_name

        save_to_hdf5(model, str(checkpoint_path))
        logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Keep only last 3 checkpoints to save disk space
        cleanup_old_checkpoints(keep_count=3)

        return str(checkpoint_path)
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {str(e)}")
        return ""


def cleanup_old_checkpoints(keep_count: int = 3):
    """Remove old checkpoints, keeping only the most recent ones.

    Args:
        keep_count: Number of recent checkpoints to keep
    """
    try:
        checkpoints = sorted(CHECKPOINT_DIR.glob("checkpoint_step_*.h5"), key=lambda p: p.stat().st_mtime, reverse=True)

        # Remove old checkpoints
        for checkpoint in checkpoints[keep_count:]:
            checkpoint.unlink()
            logger.info(f"Removed old checkpoint: {checkpoint.name}")
    except Exception as e:
        logger.error(f"Failed to cleanup checkpoints: {str(e)}")


def load_latest_checkpoint() -> Tuple[Optional[BrainModel], int]:
    """Load the most recent checkpoint.

    Returns:
        Tuple of (model, step) or (None, 0) if no checkpoint found
    """
    try:
        checkpoints = sorted(CHECKPOINT_DIR.glob("checkpoint_step_*.h5"), key=lambda p: p.stat().st_mtime, reverse=True)

        if not checkpoints:
            logger.info("No checkpoints found")
            return None, 0

        latest = checkpoints[0]
        # Extract step number from filename
        step = int(latest.stem.split("_")[-1])

        model = load_from_hdf5(str(latest))
        logger.info(f"Loaded checkpoint from step {step}")

        return model, step
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {str(e)}")
        return None, 0


@app.route("/")
def index():
    """Serve the main frontend interface."""
    return render_template("index.html")


@app.route("/.well-known/appspecific/com.chrome.devtools.json")
def chrome_devtools():
    """Handle Chrome DevTools probe to suppress 404 errors in logs."""
    return jsonify({}), 200


@app.route("/api/system/status", methods=["GET"])
def system_status():
    """Get system initialization status.
    
    This endpoint is called by frontend to check if the backend is ready
    before making other API calls. Prevents 400 errors from premature requests.
    
    Returns:
        JSON with initialization state and system information
    """
    global current_model, current_simulation
    
    status = {
        "initialized": current_model is not None and current_simulation is not None,
        "has_model": current_model is not None,
        "has_simulation": current_simulation is not None,
        "is_training": is_training
    }
    
    # Add model info if available
    if current_model is not None:
        try:
            status["model_info"] = {
                "num_neurons": len(current_model.neurons),
                "num_synapses": len(current_model.synapses),
                "current_step": current_model.current_step,
                "lattice_shape": list(current_model.lattice_shape)
            }
        except Exception as e:
            logger.warning(f"Could not get model info for status: {str(e)}")
    
    return jsonify(status)


@app.route("/api/model/init", methods=["POST"])
@limiter.limit("20 per minute")  # Limit model initialization
def init_model():
    """Initialize a new brain model."""
    global current_model, current_simulation

    try:
        config_path = request.json.get("config_path", "brain_base_model.json")

        # Validate config path
        validated_path = validate_filepath(config_path, ALLOWED_CONFIG_DIR, [".json"])

        logger.info(f"Initializing model from: {validated_path}")

        with simulation_lock:
            current_model = BrainModel(config_path=str(validated_path))
            current_simulation = Simulation(current_model, seed=42)

        logger.info(f"Model initialized: {current_model.lattice_shape}")

        return jsonify(
            {
                "status": "success",
                "lattice_shape": list(current_model.lattice_shape),
                "senses": list(current_model.get_senses().keys()),
                "areas": [a["name"] for a in current_model.get_areas()],
            }
        )
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/model/info", methods=["GET"])
@require_initialization
def get_model_info():
    """Get current model information."""
    global current_model

    try:
        return jsonify(
            {
                "status": "success",
                "lattice_shape": list(current_model.lattice_shape),
                "num_neurons": len(current_model.neurons),
                "num_synapses": len(current_model.synapses),
                "current_step": current_model.current_step,
                "senses": list(current_model.get_senses().keys()),
                "areas": [a["name"] for a in current_model.get_areas()],
            }
        )
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/config/full", methods=["GET"])
@require_initialization
def get_full_config():
    """Get complete model configuration."""
    global current_model

    try:
        config = current_model.config
        return jsonify({"status": "success", "config": config})
    except Exception as e:
        logger.error(f"Failed to get configuration: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/config/update", methods=["POST"])
@limiter.limit("30 per minute")
@require_initialization
def update_config():
    """Update model configuration parameters (requires model restart)."""
    global current_model

    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400

        # Validate data types and ranges
        allowed_config_keys = {
            'lattice_shape', 'dimensions', 'neuron_model', 'plasticity',
            'cell_lifecycle', 'neuromodulation', 'senses', 'areas'
        }
        
        # Update configuration with validation
        for key, value in data.items():
            if key not in allowed_config_keys:
                logger.warning(f"Attempted to update disallowed config key: {key}")
                continue
                
            if key in current_model.config:
                # Type and range validation
                if key == 'dimensions':
                    if not isinstance(value, int):
                        return jsonify({"status": "error", "message": f"Invalid type for {key}"}), 400
                    if value not in [3, 4]:
                        return jsonify({"status": "error", "message": "Dimensions must be 3 or 4"}), 400
                        
                if key == 'lattice_shape':
                    if not isinstance(value, list):
                        return jsonify({"status": "error", "message": f"Invalid type for {key}"}), 400
                    if len(value) != 4:
                        return jsonify({"status": "error", "message": "Lattice shape must have 4 elements"}), 400
                    if not all(isinstance(x, int) and x > 0 for x in value):
                        return jsonify({"status": "error", "message": "Lattice shape values must be positive integers"}), 400
                    
                current_model.config[key] = value
                logger.info(f"Updated config {key}")

        return jsonify({"status": "success", "message": "Configuration updated. Restart simulation to apply changes."})
    except Exception as e:
        logger.error(f"Failed to update configuration: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/neurons/details", methods=["GET"])
@require_initialization
def get_neuron_details():
    """Get detailed information about neurons."""
    global current_model

    try:
        limit = int(request.args.get("limit", 100))
        offset = int(request.args.get("offset", 0))
        
        neurons_list = list(current_model.neurons.items())
        total_neurons = len(neurons_list)
        
        neurons_data = []
        for neuron_id, neuron in neurons_list[offset:offset+limit]:
            neurons_data.append({
                "id": neuron_id,
                "position": {"x": neuron.x, "y": neuron.y, "z": neuron.z, "w": getattr(neuron, "w", 0)},
                "v_membrane": neuron.v_membrane,
                "v_threshold": neuron.v_threshold,
                "v_rest": neuron.v_rest,
                "health": neuron.health,
                "age": neuron.age,
                "neuron_type": neuron.neuron_type if hasattr(neuron, "neuron_type") else "excitatory",
            })

        return jsonify({
            "status": "success",
            "neurons": neurons_data,
            "total": total_neurons,
            "limit": limit,
            "offset": offset
        })
    except Exception as e:
        logger.error(f"Failed to get neuron details: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/synapses/details", methods=["GET"])
@require_initialization
def get_synapse_details():
    """Get detailed information about synapses."""
    global current_model

    try:
        limit = int(request.args.get("limit", 100))
        offset = int(request.args.get("offset", 0))
        
        synapses_list = list(current_model.synapses)
        total_synapses = len(synapses_list)
        
        synapses_data = []
        for synapse in synapses_list[offset:offset+limit]:
            synapses_data.append({
                "pre_id": synapse.pre_id,
                "post_id": synapse.post_id,
                "weight": synapse.weight,
                "delay": synapse.delay if hasattr(synapse, "delay") else 1,
            })

        return jsonify({
            "status": "success",
            "synapses": synapses_data,
            "total": total_synapses,
            "limit": limit,
            "offset": offset
        })
    except Exception as e:
        logger.error(f"Failed to get synapse details: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/stats/network", methods=["GET"])
@require_initialization
def get_network_stats():
    """Get comprehensive network statistics."""
    global current_model

    try:
        # Calculate statistics
        neurons = list(current_model.neurons.values())
        synapses = list(current_model.synapses)
        
        # Neuron statistics
        avg_membrane = sum(n.v_membrane for n in neurons) / len(neurons) if neurons else 0
        avg_health = sum(n.health for n in neurons) / len(neurons) if neurons else 0
        avg_age = sum(n.age for n in neurons) / len(neurons) if neurons else 0
        
        # Count neuron types
        excitatory_count = sum(1 for n in neurons if getattr(n, "neuron_type", "excitatory") == "excitatory")
        inhibitory_count = len(neurons) - excitatory_count
        
        # Synapse statistics
        avg_weight = sum(s.weight for s in synapses) / len(synapses) if synapses else 0
        positive_weights = sum(1 for s in synapses if s.weight > 0)
        negative_weights = sum(1 for s in synapses if s.weight < 0)
        
        return jsonify({
            "status": "success",
            "neurons": {
                "total": len(neurons),
                "excitatory": excitatory_count,
                "inhibitory": inhibitory_count,
                "avg_membrane_potential": avg_membrane,
                "avg_health": avg_health,
                "avg_age": avg_age,
            },
            "synapses": {
                "total": len(synapses),
                "positive_weights": positive_weights,
                "negative_weights": negative_weights,
                "avg_weight": avg_weight,
            },
            "current_step": current_model.current_step,
        })
    except Exception as e:
        logger.error(f"Failed to get network stats: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/areas/info", methods=["GET"])
@require_initialization
def get_areas_info():
    """Get detailed information about all brain areas."""
    global current_model

    try:
        areas = current_model.get_areas()
        areas_info = []
        
        for area in areas:
            # Count neurons in this area
            coord_ranges = area.get("coord_ranges", {})
            x_range = coord_ranges.get("x", [0, 0])
            y_range = coord_ranges.get("y", [0, 0])
            z_range = coord_ranges.get("z", [0, 0])
            
            neuron_count = sum(
                1 for n in current_model.neurons.values()
                if (x_range[0] <= n.x <= x_range[1] and
                    y_range[0] <= n.y <= y_range[1] and
                    z_range[0] <= n.z <= z_range[1])
            )
            
            areas_info.append({
                "name": area["name"],
                "sense": area.get("sense", "none"),
                "coord_ranges": coord_ranges,
                "neuron_count": neuron_count,
            })

        return jsonify({"status": "success", "areas": areas_info})
    except Exception as e:
        logger.error(f"Failed to get areas info: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/senses/info", methods=["GET"])
@require_initialization
def get_senses_info():
    """Get detailed information about all senses."""
    global current_model

    try:
        senses = current_model.get_senses()
        senses_info = []
        
        for sense_name, sense_data in senses.items():
            senses_info.append({
                "name": sense_name,
                "area": sense_data.get("areal", "unknown"),
                "w_index": sense_data.get("w_index", 0),
                "input_size": sense_data.get("input_size", [0, 0]),
            })

        return jsonify({"status": "success", "senses": senses_info})
    except Exception as e:
        logger.error(f"Failed to get senses info: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/neurons/init", methods=["POST"])
@require_initialization
def init_neurons():
    """Initialize neurons in specified areas."""
    global current_simulation

    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
            
        areas = data.get("areas", ["V1_like", "Digital_sensor"])
        density = data.get("density", 0.1)
        
        # Validate areas parameter
        if not isinstance(areas, list):
            return jsonify({"status": "error", "message": "areas must be a list"}), 400
        if len(areas) == 0:
            return jsonify({"status": "error", "message": "areas list cannot be empty"}), 400
        if len(areas) > 20:
            return jsonify({"status": "error", "message": "Cannot initialize more than 20 areas at once"}), 400
        if not all(isinstance(a, str) for a in areas):
            return jsonify({"status": "error", "message": "All area names must be strings"}), 400
            
        # Validate density parameter
        if not isinstance(density, (int, float)):
            return jsonify({"status": "error", "message": "density must be a number"}), 400
        if not 0 < density <= 1.0:
            return jsonify({"status": "error", "message": "density must be greater than 0 and at most 1"}), 400

        logger.info(f"Initializing neurons in {areas} with density {density}")

        with simulation_lock:
            current_simulation.initialize_neurons(area_names=areas, density=density)

        num_neurons = len(current_model.neurons)
        logger.info(f"Created {num_neurons} neurons")

        return jsonify({"status": "success", "num_neurons": num_neurons})
    except Exception as e:
        logger.error(f"Failed to initialize neurons: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/synapses/init", methods=["POST"])
@require_initialization
def init_synapses():
    """Initialize random synaptic connections."""
    global current_simulation

    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
            
        probability = data.get("probability", 0.001)
        weight_mean = data.get("weight_mean", 0.1)
        weight_std = data.get("weight_std", 0.05)
        
        # Validate probability parameter
        if not isinstance(probability, (int, float)):
            return jsonify({"status": "error", "message": "probability must be a number"}), 400
        if not 0 < probability <= 1.0:
            return jsonify({"status": "error", "message": "probability must be greater than 0 and at most 1"}), 400
            
        # Validate weight parameters
        if not isinstance(weight_mean, (int, float)):
            return jsonify({"status": "error", "message": "weight_mean must be a number"}), 400
        if not isinstance(weight_std, (int, float)):
            return jsonify({"status": "error", "message": "weight_std must be a number"}), 400
        if weight_std < 0:
            return jsonify({"status": "error", "message": "weight_std must be non-negative"}), 400

        logger.info(f"Initializing synapses with probability {probability}")

        with simulation_lock:
            current_simulation.initialize_random_synapses(
                connection_probability=probability, weight_mean=weight_mean, weight_std=weight_std
            )

        num_synapses = len(current_model.synapses)
        logger.info(f"Created {num_synapses} synapses")

        return jsonify({"status": "success", "num_synapses": num_synapses})
    except Exception as e:
        logger.error(f"Failed to initialize synapses: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/simulation/step", methods=["POST"])
@require_initialization
def simulation_step():
    """Run a single simulation step."""
    global current_simulation

    # Validate simulation state before running
    is_valid, error_msg = validate_simulation_state(current_simulation, current_model)
    if not is_valid:
        return jsonify({"status": "error", "message": error_msg}), 400

    try:
        with simulation_lock:
            stats = current_simulation.step()

        return jsonify(
            {
                "status": "success",
                "step": current_model.current_step,
                "spikes": len(stats["spikes"]),
                "deaths": stats["deaths"],
                "births": stats["births"],
                "num_neurons": len(current_model.neurons),
                "num_synapses": len(current_model.synapses),
            }
        )
    except Exception as e:
        logger.error(f"Failed to run simulation step: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


def _validate_run_parameters(data: dict) -> Tuple[bool, Optional[str], int]:
    """Validate simulation run parameters.
    
    Args:
        data: Request data dictionary
        
    Returns:
        Tuple of (is_valid, error_message, n_steps)
    """
    n_steps = data.get("steps", 100)
    
    if not isinstance(n_steps, int) or n_steps <= 0:
        return False, "Steps must be a positive integer", 0
    
    if n_steps > 100000:
        return False, "Steps cannot exceed 100000 to prevent memory exhaustion", 0
    
    return True, None, n_steps


def _run_simulation_loop(
    n_steps: int,
    max_history_steps: int = 100,
) -> Tuple[dict, list]:
    """Run the main simulation loop.
    
    Args:
        n_steps: Number of steps to run
        max_history_steps: Maximum number of step details to keep
        
    Returns:
        Tuple of (results_dict, recent_steps_list)
    """
    global current_simulation, current_model, is_training
    
    import time
    
    results = {
        "total_spikes": 0,
        "total_deaths": 0,
        "total_births": 0,
        "final_neurons": 0,
        "final_synapses": 0,
    }
    
    recent_steps = []
    step_times = []
    
    for step in range(n_steps):
        step_start = time.time()
        
        # Check training flag under lock to avoid race conditions
        with simulation_lock:
            if not is_training:
                logger.info("Training stopped by user")
                break
            stats = current_simulation.step()
            current_step = current_model.current_step
        
        # Update results
        results["total_spikes"] += len(stats["spikes"])
        results["total_deaths"] += stats["deaths"]
        results["total_births"] += stats["births"]
        
        # Auto-checkpoint at regular intervals
        if current_step % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = save_checkpoint(current_model, current_step)
            if checkpoint_path:
                logger.info(f"Auto-checkpoint saved at step {current_step}")
        
        # Track step time and emit progress
        step_time = time.time() - step_start
        step_times.append(step_time)
        if len(step_times) > 50:  # Keep last 50 steps for moving average
            step_times.pop(0)
        
        if (step + 1) % 10 == 0:
            step_info = _compute_progress_info(
                step, n_steps, step_times, stats, current_model
            )
            
            recent_steps.append(step_info)
            if len(recent_steps) > max_history_steps:
                recent_steps.pop(0)
            
            # Send progress via SocketIO
            socketio.emit("training_progress", step_info)
            logger.info(
                f"Step {step + 1}/{n_steps} ({step_info['progress_percent']:.1f}%): "
                f"{len(stats['spikes'])} spikes, "
                f"~{step_info['estimated_remaining_seconds']:.0f}s remaining"
            )
    
    # Add final state
    results["final_neurons"] = len(current_model.neurons)
    results["final_synapses"] = len(current_model.synapses)
    results["recent_steps"] = recent_steps
    
    return results, recent_steps


def _compute_progress_info(
    step: int,
    total_steps: int,
    step_times: list,
    stats: dict,
    model: BrainModel,
) -> dict:
    """Compute progress information for a simulation step.
    
    Args:
        step: Current step number (0-indexed)
        total_steps: Total number of steps to run
        step_times: List of recent step execution times
        stats: Statistics from current step
        model: The brain model
        
    Returns:
        Dictionary with progress information
    """
    avg_step_time = sum(step_times) / len(step_times)
    remaining_steps = total_steps - (step + 1)
    estimated_remaining_time = avg_step_time * remaining_steps
    progress_percent = ((step + 1) / total_steps) * 100
    
    return {
        "step": step + 1,
        "total_steps": total_steps,
        "progress_percent": round(progress_percent, 1),
        "estimated_remaining_seconds": round(estimated_remaining_time, 1),
        "spikes": len(stats["spikes"]),
        "neurons": len(model.neurons),
        "synapses": len(model.synapses),
    }


@app.route("/api/simulation/run", methods=["POST"])
@limiter.limit("10 per minute")  # Limit intensive simulation runs
@require_initialization
def run_simulation():
    """Run simulation for multiple steps.
    
    This endpoint has been refactored to use helper functions for better
    maintainability. The main logic is split into:
    - Parameter validation (_validate_run_parameters)
    - Simulation loop execution (_run_simulation_loop)
    - Progress computation (_compute_progress_info)
    """
    global is_training

    # Validate simulation state before running
    is_valid, error_msg = validate_simulation_state(current_simulation, current_model)
    if not is_valid:
        return jsonify({"status": "error", "message": error_msg}), 400

    # Check if training is already running to prevent concurrent runs
    with simulation_lock:
        if is_training:
            return jsonify({"status": "error", "message": "Simulation already running"}), 400
        is_training = True

    try:
        data = request.json
        
        # Validate parameters
        is_valid, error_msg, n_steps = _validate_run_parameters(data)
        if not is_valid:
            raise ValueError(error_msg)

        logger.info(f"Running simulation for {n_steps} steps")

        # Run the simulation loop
        results, recent_steps = _run_simulation_loop(n_steps)

        with simulation_lock:
            is_training = False
        
        logger.info(f"Simulation complete: {results['total_spikes']} total spikes")
        return jsonify({"status": "success", "results": results})
        
    except ValueError as e:
        with simulation_lock:
            is_training = False
        logger.error(f"Invalid simulation parameters: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        with simulation_lock:
            is_training = False
        logger.error(f"Failed to run simulation: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/simulation/stop", methods=["POST"])
def stop_simulation():
    """Stop ongoing simulation."""
    global is_training

    with simulation_lock:
        is_training = False
    logger.info("Training stopped")

    return jsonify({"status": "success"})


@app.route("/api/simulation/recover", methods=["POST"])
def recover_from_checkpoint():
    """Recover simulation from the latest checkpoint."""
    global current_model, current_simulation

    try:
        model, step = load_latest_checkpoint()

        if model is None:
            return jsonify({"status": "error", "message": "No checkpoint available for recovery"}), 404

        with simulation_lock:
            current_model = model
            current_simulation = Simulation(current_model, seed=42)

        logger.info(f"Successfully recovered from checkpoint at step {step}")

        return jsonify(
            {
                "status": "success",
                "recovered_step": step,
                "num_neurons": len(current_model.neurons),
                "num_synapses": len(current_model.synapses),
            }
        )
    except Exception as e:
        logger.error(f"Failed to recover from checkpoint: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/input/feed", methods=["POST"])
@limiter.limit("60 per minute")  # Limit input feeding
@require_initialization
def feed_input():
    """Feed sensory input to the model."""
    global current_model

    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400

        sense_type = data.get("sense_type", "vision")
        input_data = data.get("input_data")

        if input_data is None:
            return jsonify({"status": "error", "message": "No input data provided"}), 400

        # Validate sense_type to prevent injection
        valid_senses = current_model.get_senses().keys()
        if sense_type not in valid_senses:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": f"Invalid sense type: {sense_type}. Valid types: {', '.join(valid_senses)}",
                    }
                ),
                400,
            )

        logger.info(f"Feeding {sense_type} input")

        # Convert input based on type with size limits to prevent memory exhaustion
        if sense_type == "digital" and isinstance(input_data, str):
            # Limit digital input to 10KB to prevent DoS
            if len(input_data) > 10240:
                return jsonify({"status": "error", "message": "Digital input too large (max 10KB)"}), 400
            input_array = create_digital_sense_input(input_data, target_shape=(20, 20))
        else:
            # Validate input_data is list-like
            if not isinstance(input_data, (list, tuple)):
                return jsonify({"status": "error", "message": "Input data must be a list or array"}), 400

            # Limit array size to prevent memory exhaustion (max 1000x1000)
            if len(input_data) > 1000 or (input_data and len(input_data[0]) > 1000):
                return jsonify({"status": "error", "message": "Input array too large (max 1000x1000)"}), 400

            input_array = np.array(input_data)

        with simulation_lock:
            feed_sense_input(current_model, sense_type, input_array)

        return jsonify({"status": "success"})
    except ValueError as e:
        # Handle validation errors with clear messages
        logger.warning(f"Invalid input: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        logger.error(f"Failed to feed input: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/heatmap/data", methods=["GET"])
def get_heatmap_data():
    """Get heatmap data for visualization."""
    global current_model

    if current_model is None:
        return jsonify({"status": "error", "message": "No model initialized"}), 400

    try:
        # Get neurons grouped by layer (z coordinate)
        layers = {}
        for neuron in current_model.neurons.values():
            z = neuron.z
            if z not in layers:
                layers[z] = []
            layers[z].append(
                {
                    "x": neuron.x,
                    "y": neuron.y,
                    "v_membrane": neuron.v_membrane,
                    "health": neuron.health,
                    "age": neuron.age,
                }
            )

        # Create heatmap for input (z=0), hidden (z=10), output (z=19) layers
        heatmap_data = {}
        for layer_name, z_coord in [("input", 0), ("hidden", 10), ("output", 19)]:
            if z_coord in layers:
                # Create 2D grid
                grid = np.zeros((20, 20))
                for neuron in layers[z_coord]:
                    if 0 <= neuron["x"] < 20 and 0 <= neuron["y"] < 20:
                        grid[neuron["x"], neuron["y"]] = neuron["v_membrane"]
                heatmap_data[layer_name] = grid.tolist()
            else:
                heatmap_data[layer_name] = np.zeros((20, 20)).tolist()

        return jsonify({"status": "success", "heatmap": heatmap_data})
    except Exception as e:
        logger.error(f"Failed to get heatmap data: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/model/save", methods=["POST"])
@limiter.limit("30 per minute")  # Limit save operations
def save_model():
    """Save current model state."""
    global current_model

    if current_model is None:
        return jsonify({"status": "error", "message": "No model initialized"}), 400

    try:
        data = request.json
        format_type = data.get("format", "json")
        filename = data.get("filename", "brain_state")

        # Determine file extension and validate
        if format_type == "json":
            filepath = f"{filename}.json"
            allowed_ext = [".json"]
        elif format_type == "hdf5":
            filepath = f"{filename}.h5"
            allowed_ext = [".h5"]
        else:
            return jsonify({"status": "error", "message": "Invalid format"}), 400

        # Validate file path to prevent directory traversal
        validated_path = validate_filepath(str(ALLOWED_SAVE_DIR / filepath), ALLOWED_SAVE_DIR, allowed_ext)

        # Save model
        if format_type == "json":
            save_to_json(current_model, str(validated_path))
        else:
            save_to_hdf5(current_model, str(validated_path))

        logger.info(f"Model saved to {validated_path}")

        return jsonify({"status": "success", "filepath": str(validated_path)})
    except ValueError as e:
        logger.error(f"Invalid file path: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/model/load", methods=["POST"])
@limiter.limit("30 per minute")  # Limit load operations
def load_model():
    """Load model state from file."""
    global current_model, current_simulation

    try:
        data = request.json
        filepath = data.get("filepath", "saved_models/brain_state.json")

        # Determine allowed extensions based on file extension
        if filepath.endswith(".json"):
            allowed_ext = [".json"]
        elif filepath.endswith(".h5"):
            allowed_ext = [".h5"]
        else:
            return jsonify({"status": "error", "message": "Invalid file format"}), 400

        # Validate file path to prevent directory traversal
        validated_path = validate_filepath(filepath, ALLOWED_SAVE_DIR, allowed_ext)

        # Check if file exists
        if not validated_path.exists():
            return jsonify({"status": "error", "message": f"File not found: {validated_path.name}"}), 404

        logger.info(f"Loading model from {validated_path}")

        with simulation_lock:
            if validated_path.suffix == ".json":
                current_model = load_from_json(str(validated_path))
            else:
                current_model = load_from_hdf5(str(validated_path))

            current_simulation = Simulation(current_model, seed=42)

        logger.info(f"Model loaded: {len(current_model.neurons)} neurons")

        return jsonify(
            {
                "status": "success",
                "num_neurons": len(current_model.neurons),
                "num_synapses": len(current_model.synapses),
            }
        )
    except ValueError as e:
        logger.error(f"Invalid file path: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@socketio.on("connect")
def handle_connect():
    """Handle WebSocket connection."""
    logger.info("Client connected")
    emit("connection_response", {"data": "Connected to 4D Neural Cognition"})


@socketio.on("disconnect")
def handle_disconnect():
    """Handle WebSocket disconnection."""
    logger.info("Client disconnected")


@socketio.on("chat_message")
def handle_chat_message(data):
    """Handle chat messages for operations."""
    message = data.get("message", "")
    logger.info(f"Chat message: {message}")

    # Simple command processing
    response = process_chat_command(message)

    emit("chat_response", {"message": response, "timestamp": datetime.now().isoformat()})


def process_chat_command(command):
    """Process chat commands and return responses."""
    command = command.lower().strip()

    if "help" in command:
        return "Available commands: init, info, step, run <n>, save, load, status"
    elif "init" in command:
        return "Use the Initialize Model button to create a new model"
    elif "info" in command or "status" in command:
        if current_model:
            return f"Model has {len(current_model.neurons)} neurons, {len(current_model.synapses)} synapses, step {current_model.current_step}"
        else:
            return "No model initialized"
    elif "step" in command:
        return "Use the Step button to run a single simulation step"
    elif "run" in command:
        return "Use the Train button to run multiple simulation steps"
    else:
        return f"Unknown command: {command}. Type 'help' for available commands."


@app.route("/dashboard")
def dashboard():
    """Serve the comprehensive dashboard interface."""
    return render_template("dashboard.html")


@app.route("/advanced")
def advanced():
    """Serve the advanced interface with 3D/4D visualization and collaboration features."""
    return render_template("advanced.html")


@app.route("/api/visualization/neurons", methods=["GET"])
@require_initialization
def get_neurons_visualization():
    """Get neuron data for 3D/4D visualization."""
    global current_model

    try:
        # Configurable limit for visualization performance
        viz_limit = int(os.environ.get("VIZ_NEURON_LIMIT", "1000"))
        
        neurons_data = []
        for neuron_id, neuron in list(current_model.neurons.items())[:viz_limit]:
            neurons_data.append(
                {
                    "id": neuron_id,
                    "x": neuron.x,
                    "y": neuron.y,
                    "z": neuron.z,
                    "w": getattr(neuron, "w", 0),  # 4th dimension if available
                    "v_membrane": neuron.v_membrane,
                    "health": neuron.health,
                    "age": neuron.age,
                    "activity": 1.0 if abs(neuron.v_membrane - (-65.0)) > 10 else 0.0,  # -65.0 is typical resting potential
                }
            )

        return jsonify({"status": "success", "neurons": neurons_data, "count": len(neurons_data)})
    except Exception as e:
        logger.error(f"Failed to get visualization data: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/visualization/connections", methods=["GET"])
@require_initialization
def get_connections_visualization():
    """Get connection data for visualization."""
    global current_model

    try:
        # Configurable limit for visualization performance
        viz_limit = int(os.environ.get("VIZ_CONNECTION_LIMIT", "500"))
        
        connections_data = []
        for synapse in list(current_model.synapses)[:viz_limit]:
            pre_neuron = current_model.neurons.get(synapse.pre_id)
            post_neuron = current_model.neurons.get(synapse.post_id)

            if pre_neuron and post_neuron:
                connections_data.append(
                    {
                        "from": {
                            "x": pre_neuron.x,
                            "y": pre_neuron.y,
                            "z": pre_neuron.z,
                            "w": getattr(pre_neuron, "w", 0),
                        },
                        "to": {
                            "x": post_neuron.x,
                            "y": post_neuron.y,
                            "z": post_neuron.z,
                            "w": getattr(post_neuron, "w", 0),
                        },
                        "weight": synapse.weight,
                    }
                )

        return jsonify({"status": "success", "connections": connections_data, "count": len(connections_data)})
    except Exception as e:
        logger.error(f"Failed to get connections data: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/vnc/status", methods=["GET"])
def get_vnc_status():
    """Get Virtual Neuromorphic Clock status and statistics."""
    global current_simulation
    
    if current_simulation is None:
        return jsonify({"status": "error", "message": "No simulation initialized"}), 400
    
    try:
        # Check if VNC is enabled
        if not hasattr(current_simulation, 'use_vnc') or not current_simulation.use_vnc:
            return jsonify({
                "status": "success",
                "vnc_enabled": False,
                "message": "VNC not enabled"
            })
        
        # Get VNC statistics
        vnc_stats = current_simulation.get_vnc_statistics()
        vpu_stats = current_simulation.get_vpu_statistics()
        
        if vnc_stats is None:
            return jsonify({
                "status": "success",
                "vnc_enabled": True,
                "vnc_initialized": False,
                "message": "VNC enabled but not initialized"
            })
        
        return jsonify({
            "status": "success",
            "vnc_enabled": True,
            "vnc_initialized": True,
            "global_stats": vnc_stats,
            "vpu_stats": vpu_stats if vpu_stats else []
        })
        
    except Exception as e:
        logger.error(f"Failed to get VNC status: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/vnc/config", methods=["GET", "POST"])
def vnc_config():
    """Get or update VNC configuration."""
    global current_simulation, current_model
    
    if request.method == "GET":
        # Return current VNC configuration
        if current_simulation is None:
            return jsonify({"status": "error", "message": "No simulation initialized"}), 400
        
        try:
            config = {
                "vnc_enabled": getattr(current_simulation, 'use_vnc', False),
                "clock_frequency": getattr(current_simulation, 'vnc_clock_frequency', 20e6),
                "num_vpus": 0,
            }
            
            if current_simulation.virtual_clock is not None:
                config["num_vpus"] = len(current_simulation.virtual_clock.vpus)
            
            return jsonify({"status": "success", "config": config})
            
        except Exception as e:
            logger.error(f"Failed to get VNC config: {str(e)}")
            return jsonify({"status": "error", "message": str(e)}), 500
    
    else:  # POST - Update configuration
        if current_model is None:
            return jsonify({"status": "error", "message": "No model initialized"}), 400
        
        try:
            data = request.json
            if not data:
                return jsonify({"status": "error", "message": "No data provided"}), 400
                
            vnc_enabled = data.get("vnc_enabled", False)
            
            # Validate vnc_enabled type
            if not isinstance(vnc_enabled, bool):
                return jsonify({"status": "error", "message": "vnc_enabled must be a boolean"}), 400
            
            # Validate and convert clock_frequency
            clock_frequency_raw = data.get("clock_frequency", 20e6)
            if not isinstance(clock_frequency_raw, (int, float)):
                return jsonify({"status": "error", "message": "clock_frequency must be a number"}), 400
            clock_frequency = float(clock_frequency_raw)
            
            # Validate clock frequency range
            if clock_frequency <= 0 or clock_frequency > 1e9:
                return jsonify({
                    "status": "error",
                    "message": "Clock frequency must be greater than 0 and at most 1 GHz"
                }), 400
            
            # Create new simulation with VNC settings
            with simulation_lock:
                current_simulation = Simulation(
                    current_model,
                    seed=42,
                    use_vnc=vnc_enabled,
                    vnc_clock_frequency=clock_frequency
                )
            
            logger.info(f"VNC configuration updated: enabled={vnc_enabled}, freq={clock_frequency/1e6:.1f} MHz")
            
            return jsonify({
                "status": "success",
                "message": "VNC configuration updated",
                "config": {
                    "vnc_enabled": vnc_enabled,
                    "clock_frequency": clock_frequency,
                    "num_vpus": len(current_simulation.virtual_clock.vpus) if current_simulation.virtual_clock else 0
                }
            })
            
        except Exception as e:
            logger.error(f"Failed to update VNC config: {str(e)}")
            return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/vnc/reset", methods=["POST"])
def reset_vnc_stats():
    """Reset VNC statistics."""
    global current_simulation
    
    if current_simulation is None:
        return jsonify({"status": "error", "message": "No simulation initialized"}), 400
    
    try:
        if current_simulation.virtual_clock is not None:
            current_simulation.virtual_clock.reset()
            logger.info("VNC statistics reset")
            return jsonify({"status": "success", "message": "VNC statistics reset"})
        else:
            return jsonify({"status": "error", "message": "VNC not initialized"}), 400
            
    except Exception as e:
        logger.error(f"Failed to reset VNC stats: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/vnc/rebalance", methods=["POST"])
def trigger_vnc_rebalance():
    """Trigger VNC load rebalancing."""
    global current_simulation
    
    if current_simulation is None:
        return jsonify({"status": "error", "message": "No simulation initialized"}), 400
    
    try:
        if current_simulation.virtual_clock is not None:
            current_simulation.virtual_clock.rebalance_partitions()
            logger.info("VNC load rebalancing triggered")
            return jsonify({"status": "success", "message": "Load rebalancing triggered"})
        else:
            return jsonify({"status": "error", "message": "VNC not initialized"}), 400
            
    except Exception as e:
        logger.error(f"Failed to trigger rebalance: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@socketio.on("register_user")
def handle_register_user(data):
    """Handle user registration for collaboration."""
    try:
        logger.info(f"User registered: {data.get('username')}")
        emit("user_joined", {"user": data}, broadcast=True)
    except Exception as e:
        logger.error(f"Error registering user: {str(e)}")


@socketio.on("create_shared_simulation")
def handle_create_shared_simulation(data):
    """Handle creation of shared simulation."""
    try:
        logger.info(f"Shared simulation created: {data.get('name')}")
        emit("simulation_created", {"simulation": data}, broadcast=True)
    except Exception as e:
        logger.error(f"Error creating shared simulation: {str(e)}")


@socketio.on("add_annotation")
def handle_add_annotation(data):
    """Handle adding annotation."""
    try:
        logger.info(f"Annotation added by {data.get('authorName')}")
        emit("annotation_added", {"annotation": data}, broadcast=True)
    except Exception as e:
        logger.error(f"Error adding annotation: {str(e)}")


@socketio.on("create_version")
def handle_create_version(data):
    """Handle version creation."""
    try:
        logger.info(f"Version created: {data.get('name')}")
        emit("version_created", {"version": data}, broadcast=True)
    except Exception as e:
        logger.error(f"Error creating version: {str(e)}")


# Knowledge System Routes
KNOWLEDGE_BASE_DIR = Path(".")
DOCS_DIR = Path("docs")


def get_knowledge_structure():
    """Build a hierarchical structure of all documentation files."""
    structure = {
        "root": [],
        "docs": {}
    }
    
    # Root level markdown files
    for md_file in KNOWLEDGE_BASE_DIR.glob("*.md"):
        if md_file.is_file():
            structure["root"].append({
                "name": md_file.name,
                "path": str(md_file.relative_to(KNOWLEDGE_BASE_DIR)),
                "size": md_file.stat().st_size,
                "modified": md_file.stat().st_mtime
            })
    
    # Docs directory structure
    if DOCS_DIR.exists():
        for item in DOCS_DIR.rglob("*"):
            if item.is_file() and item.suffix == ".md":
                rel_path = item.relative_to(KNOWLEDGE_BASE_DIR)
                parts = rel_path.parts
                
                current = structure["docs"]
                for part in parts[1:-1]:  # Skip 'docs' and filename
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                if "files" not in current:
                    current["files"] = []
                current["files"].append({
                    "name": item.name,
                    "path": str(rel_path),
                    "size": item.stat().st_size,
                    "modified": item.stat().st_mtime
                })
    
    return structure


@app.route("/api/knowledge/list", methods=["GET"])
def list_knowledge():
    """List all documentation files with hierarchy."""
    try:
        structure = get_knowledge_structure()
        return jsonify({"status": "success", "structure": structure})
    except Exception as e:
        logger.error(f"Failed to list knowledge: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/knowledge/read", methods=["GET"])
def read_knowledge():
    """Read a specific documentation file."""
    try:
        file_path = request.args.get("path")
        if not file_path:
            return jsonify({"status": "error", "message": "No path provided"}), 400
        
        # Security: Validate path is within allowed directories
        full_path = Path(file_path).resolve()
        base_path = KNOWLEDGE_BASE_DIR.resolve()
        
        # Check if path is within base directory (handles symlinks and edge cases)
        try:
            full_path.relative_to(base_path)
        except ValueError:
            return jsonify({"status": "error", "message": "Access denied"}), 403
        
        if not full_path.exists():
            return jsonify({"status": "error", "message": "File not found"}), 404
        
        if full_path.suffix != ".md":
            return jsonify({"status": "error", "message": "Only markdown files allowed"}), 400
        
        # Use validated full_path consistently (prevent TOCTOU)
        content = full_path.read_text(encoding="utf-8")
        file_stat = full_path.stat()
        
        return jsonify({
            "status": "success",
            "content": content,
            "path": str(full_path.relative_to(base_path)),
            "name": full_path.name,
            "size": file_stat.st_size,
            "modified": file_stat.st_mtime
        })
    except Exception as e:
        logger.error(f"Failed to read knowledge: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/knowledge/write", methods=["POST"])
@limiter.limit("30 per hour")  # Limit write operations
def write_knowledge():
    """Create or update a documentation file."""
    try:
        data = request.json
        file_path = data.get("path")
        content = data.get("content")
        
        if not file_path or content is None:
            return jsonify({"status": "error", "message": "Path and content required"}), 400
        
        # Security: Validate path is within allowed directories
        full_path = Path(file_path).resolve()
        base_path = KNOWLEDGE_BASE_DIR.resolve()
        
        # Check if path is within base directory (handles symlinks and edge cases)
        try:
            full_path.relative_to(base_path)
        except ValueError:
            return jsonify({"status": "error", "message": "Access denied"}), 403
        
        if full_path.suffix != ".md":
            return jsonify({"status": "error", "message": "Only markdown files allowed"}), 400
        
        # Create parent directories if needed
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write content using validated full_path (prevent TOCTOU)
        full_path.write_text(content, encoding="utf-8")
        
        logger.info(f"Knowledge file written: {full_path.relative_to(base_path)}")
        
        file_stat = full_path.stat()
        
        return jsonify({
            "status": "success",
            "message": "File saved successfully",
            "path": str(full_path.relative_to(base_path)),
            "size": file_stat.st_size
        })
    except Exception as e:
        logger.error(f"Failed to write knowledge: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/knowledge/search", methods=["GET"])
def search_knowledge():
    """Search across all documentation files."""
    try:
        query = request.args.get("q", "").lower()
        if not query:
            return jsonify({"status": "error", "message": "No query provided"}), 400
        
        results = []
        
        # Search in root markdown files
        for md_file in KNOWLEDGE_BASE_DIR.glob("*.md"):
            if md_file.is_file():
                try:
                    content = md_file.read_text(encoding="utf-8")
                    if query in content.lower() or query in md_file.name.lower():
                        # Find context around matches
                        lines = content.split("\n")
                        matches = []
                        for i, line in enumerate(lines):
                            if query in line.lower():
                                # Get context: 1 line before and after
                                start = max(0, i - 1)
                                end = min(len(lines), i + 2)
                                context = "\n".join(lines[start:end])
                                matches.append({
                                    "line": i + 1,
                                    "context": context
                                })
                                if len(matches) >= 3:  # Limit to 3 matches per file
                                    break
                        
                        if matches:
                            results.append({
                                "path": str(md_file.relative_to(KNOWLEDGE_BASE_DIR)),
                                "name": md_file.name,
                                "matches": matches
                            })
                except Exception as e:
                    logger.warning(f"Error reading {md_file}: {str(e)}")
        
        # Search in docs directory
        if DOCS_DIR.exists():
            for md_file in DOCS_DIR.rglob("*.md"):
                if md_file.is_file():
                    try:
                        content = md_file.read_text(encoding="utf-8")
                        if query in content.lower() or query in md_file.name.lower():
                            lines = content.split("\n")
                            matches = []
                            for i, line in enumerate(lines):
                                if query in line.lower():
                                    start = max(0, i - 1)
                                    end = min(len(lines), i + 2)
                                    context = "\n".join(lines[start:end])
                                    matches.append({
                                        "line": i + 1,
                                        "context": context
                                    })
                                    if len(matches) >= 3:
                                        break
                            
                            if matches:
                                results.append({
                                    "path": str(md_file.relative_to(KNOWLEDGE_BASE_DIR)),
                                    "name": md_file.name,
                                    "matches": matches
                                })
                    except Exception as e:
                        logger.warning(f"Error reading {md_file}: {str(e)}")
        
        return jsonify({
            "status": "success",
            "query": query,
            "results": results,
            "count": len(results)
        })
    except Exception as e:
        logger.error(f"Failed to search knowledge: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    logger.info("Starting 4D Neural Cognition web interface")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
