#!/usr/bin/env python3
"""Flask web application for 4D Neural Cognition frontend interface."""

import sys
import os
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from threading import Lock
from pathlib import Path
import numpy as np

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# Ensure logs directory exists BEFORE configuring logging
os.makedirs('logs', exist_ok=True)

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from brain_model import BrainModel
from simulation import Simulation
from senses import feed_sense_input, create_digital_sense_input
from storage import save_to_json, load_from_json, save_to_hdf5, load_from_hdf5

# Configure logging with rotation to prevent unbounded log file growth
# Each log file is limited to 10MB, with 5 backup files kept
file_handler = RotatingFileHandler(
    'logs/app.log',
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, stream_handler]
)
logger = logging.getLogger(__name__)

# Flask app setup
# Use environment variable for secret key for better security
# Fallback to a default only for development/testing
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get(
    'FLASK_SECRET_KEY',
    'neural-cognition-secret-key-4d-CHANGE-IN-PRODUCTION'
)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
current_model = None
current_simulation = None
simulation_lock = Lock()
is_training = False
training_thread = None

# Define allowed directories for file operations
ALLOWED_SAVE_DIR = Path('saved_models')
ALLOWED_CONFIG_DIR = Path('.')
ALLOWED_SAVE_DIR.mkdir(exist_ok=True)


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
                f"Invalid file extension: {file_path.suffix}. "
                f"Allowed: {', '.join(allowed_extensions)}"
            )
        
        return file_path
    except Exception as e:
        raise ValueError(f"Invalid file path: {str(e)}")


class WebLogger(logging.Handler):
    """Custom logging handler to send logs to web interface via SocketIO."""
    
    def emit(self, record):
        try:
            log_entry = self.format(record)
            socketio.emit('log_message', {
                'timestamp': datetime.now().isoformat(),
                'level': record.levelname,
                'message': log_entry
            })
        except Exception:
            pass


# Add web logger
web_handler = WebLogger()
web_handler.setLevel(logging.INFO)
logger.addHandler(web_handler)


@app.route('/')
def index():
    """Serve the main frontend interface."""
    return render_template('index.html')


@app.route('/api/model/init', methods=['POST'])
def init_model():
    """Initialize a new brain model."""
    global current_model, current_simulation
    
    try:
        config_path = request.json.get('config_path', 'brain_base_model.json')
        
        # Validate config path
        validated_path = validate_filepath(
            config_path,
            ALLOWED_CONFIG_DIR,
            ['.json']
        )
        
        logger.info(f"Initializing model from: {validated_path}")
        
        with simulation_lock:
            current_model = BrainModel(config_path=str(validated_path))
            current_simulation = Simulation(current_model, seed=42)
            
        logger.info(f"Model initialized: {current_model.lattice_shape}")
        
        return jsonify({
            'status': 'success',
            'lattice_shape': list(current_model.lattice_shape),
            'senses': list(current_model.get_senses().keys()),
            'areas': [a['name'] for a in current_model.get_areas()]
        })
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/model/info', methods=['GET'])
def get_model_info():
    """Get current model information."""
    global current_model
    
    if current_model is None:
        return jsonify({'status': 'error', 'message': 'No model initialized'}), 400
    
    try:
        return jsonify({
            'status': 'success',
            'lattice_shape': list(current_model.lattice_shape),
            'num_neurons': len(current_model.neurons),
            'num_synapses': len(current_model.synapses),
            'current_step': current_model.current_step,
            'senses': list(current_model.get_senses().keys()),
            'areas': [a['name'] for a in current_model.get_areas()]
        })
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/neurons/init', methods=['POST'])
def init_neurons():
    """Initialize neurons in specified areas."""
    global current_simulation
    
    if current_simulation is None:
        return jsonify({'status': 'error', 'message': 'No simulation initialized'}), 400
    
    try:
        data = request.json
        areas = data.get('areas', ['V1_like', 'Digital_sensor'])
        density = data.get('density', 0.1)
        
        logger.info(f"Initializing neurons in {areas} with density {density}")
        
        with simulation_lock:
            current_simulation.initialize_neurons(area_names=areas, density=density)
            
        num_neurons = len(current_model.neurons)
        logger.info(f"Created {num_neurons} neurons")
        
        return jsonify({
            'status': 'success',
            'num_neurons': num_neurons
        })
    except Exception as e:
        logger.error(f"Failed to initialize neurons: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/synapses/init', methods=['POST'])
def init_synapses():
    """Initialize random synaptic connections."""
    global current_simulation
    
    if current_simulation is None:
        return jsonify({'status': 'error', 'message': 'No simulation initialized'}), 400
    
    try:
        data = request.json
        probability = data.get('probability', 0.001)
        weight_mean = data.get('weight_mean', 0.1)
        weight_std = data.get('weight_std', 0.05)
        
        logger.info(f"Initializing synapses with probability {probability}")
        
        with simulation_lock:
            current_simulation.initialize_random_synapses(
                connection_probability=probability,
                weight_mean=weight_mean,
                weight_std=weight_std
            )
            
        num_synapses = len(current_model.synapses)
        logger.info(f"Created {num_synapses} synapses")
        
        return jsonify({
            'status': 'success',
            'num_synapses': num_synapses
        })
    except Exception as e:
        logger.error(f"Failed to initialize synapses: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/simulation/step', methods=['POST'])
def simulation_step():
    """Run a single simulation step."""
    global current_simulation
    
    if current_simulation is None:
        return jsonify({'status': 'error', 'message': 'No simulation initialized'}), 400
    
    try:
        with simulation_lock:
            stats = current_simulation.step()
            
        return jsonify({
            'status': 'success',
            'step': current_model.current_step,
            'spikes': len(stats['spikes']),
            'deaths': stats['deaths'],
            'births': stats['births'],
            'num_neurons': len(current_model.neurons),
            'num_synapses': len(current_model.synapses)
        })
    except Exception as e:
        logger.error(f"Failed to run simulation step: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/simulation/run', methods=['POST'])
def run_simulation():
    """Run simulation for multiple steps."""
    global current_simulation, is_training
    
    if current_simulation is None:
        return jsonify({'status': 'error', 'message': 'No simulation initialized'}), 400
    
    try:
        data = request.json
        n_steps = data.get('steps', 100)
        
        logger.info(f"Running simulation for {n_steps} steps")
        is_training = True
        
        results = {
            'total_spikes': 0,
            'total_deaths': 0,
            'total_births': 0,
            'steps': []
        }
        
        for step in range(n_steps):
            if not is_training:
                logger.info("Training stopped by user")
                break
                
            with simulation_lock:
                stats = current_simulation.step()
                
            results['total_spikes'] += len(stats['spikes'])
            results['total_deaths'] += stats['deaths']
            results['total_births'] += stats['births']
            
            if (step + 1) % 10 == 0:
                step_info = {
                    'step': step + 1,
                    'spikes': len(stats['spikes']),
                    'neurons': len(current_model.neurons),
                    'synapses': len(current_model.synapses)
                }
                results['steps'].append(step_info)
                
                # Send progress via SocketIO
                socketio.emit('training_progress', step_info)
                logger.info(f"Step {step + 1}/{n_steps}: {len(stats['spikes'])} spikes")
        
        is_training = False
        logger.info(f"Simulation complete: {results['total_spikes']} total spikes")
        
        return jsonify({
            'status': 'success',
            'results': results
        })
    except Exception as e:
        is_training = False
        logger.error(f"Failed to run simulation: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/simulation/stop', methods=['POST'])
def stop_simulation():
    """Stop ongoing simulation."""
    global is_training
    
    is_training = False
    logger.info("Training stopped")
    
    return jsonify({'status': 'success'})


@app.route('/api/input/feed', methods=['POST'])
def feed_input():
    """Feed sensory input to the model."""
    global current_model
    
    if current_model is None:
        return jsonify({'status': 'error', 'message': 'No model initialized'}), 400
    
    try:
        data = request.json
        if not data:
            return jsonify({'status': 'error', 'message': 'No data provided'}), 400
        
        sense_type = data.get('sense_type', 'vision')
        input_data = data.get('input_data')
        
        if input_data is None:
            return jsonify({'status': 'error', 'message': 'No input data provided'}), 400
        
        # Validate sense_type to prevent injection
        valid_senses = current_model.get_senses().keys()
        if sense_type not in valid_senses:
            return jsonify({
                'status': 'error',
                'message': f"Invalid sense type: {sense_type}. Valid types: {', '.join(valid_senses)}"
            }), 400
        
        logger.info(f"Feeding {sense_type} input")
        
        # Convert input based on type with size limits to prevent memory exhaustion
        if sense_type == 'digital' and isinstance(input_data, str):
            # Limit digital input to 10KB to prevent DoS
            if len(input_data) > 10240:
                return jsonify({
                    'status': 'error',
                    'message': 'Digital input too large (max 10KB)'
                }), 400
            input_array = create_digital_sense_input(input_data, target_shape=(20, 20))
        else:
            # Validate input_data is list-like
            if not isinstance(input_data, (list, tuple)):
                return jsonify({
                    'status': 'error',
                    'message': 'Input data must be a list or array'
                }), 400
            
            # Limit array size to prevent memory exhaustion (max 1000x1000)
            if len(input_data) > 1000 or (input_data and len(input_data[0]) > 1000):
                return jsonify({
                    'status': 'error',
                    'message': 'Input array too large (max 1000x1000)'
                }), 400
            
            input_array = np.array(input_data)
        
        with simulation_lock:
            feed_sense_input(current_model, sense_type, input_array)
        
        return jsonify({'status': 'success'})
    except ValueError as e:
        # Handle validation errors with clear messages
        logger.warning(f"Invalid input: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400
    except Exception as e:
        logger.error(f"Failed to feed input: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/heatmap/data', methods=['GET'])
def get_heatmap_data():
    """Get heatmap data for visualization."""
    global current_model
    
    if current_model is None:
        return jsonify({'status': 'error', 'message': 'No model initialized'}), 400
    
    try:
        # Get neurons grouped by layer (z coordinate)
        layers = {}
        for neuron in current_model.neurons.values():
            z = neuron.z
            if z not in layers:
                layers[z] = []
            layers[z].append({
                'x': neuron.x,
                'y': neuron.y,
                'v_membrane': neuron.v_membrane,
                'health': neuron.health,
                'age': neuron.age
            })
        
        # Create heatmap for input (z=0), hidden (z=10), output (z=19) layers
        heatmap_data = {}
        for layer_name, z_coord in [('input', 0), ('hidden', 10), ('output', 19)]:
            if z_coord in layers:
                # Create 2D grid
                grid = np.zeros((20, 20))
                for neuron in layers[z_coord]:
                    if 0 <= neuron['x'] < 20 and 0 <= neuron['y'] < 20:
                        grid[neuron['x'], neuron['y']] = neuron['v_membrane']
                heatmap_data[layer_name] = grid.tolist()
            else:
                heatmap_data[layer_name] = np.zeros((20, 20)).tolist()
        
        return jsonify({
            'status': 'success',
            'heatmap': heatmap_data
        })
    except Exception as e:
        logger.error(f"Failed to get heatmap data: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/model/save', methods=['POST'])
def save_model():
    """Save current model state."""
    global current_model
    
    if current_model is None:
        return jsonify({'status': 'error', 'message': 'No model initialized'}), 400
    
    try:
        data = request.json
        format_type = data.get('format', 'json')
        filename = data.get('filename', 'brain_state')
        
        # Determine file extension and validate
        if format_type == 'json':
            filepath = f"{filename}.json"
            allowed_ext = ['.json']
        elif format_type == 'hdf5':
            filepath = f"{filename}.h5"
            allowed_ext = ['.h5']
        else:
            return jsonify({'status': 'error', 'message': 'Invalid format'}), 400
        
        # Validate file path to prevent directory traversal
        validated_path = validate_filepath(
            str(ALLOWED_SAVE_DIR / filepath),
            ALLOWED_SAVE_DIR,
            allowed_ext
        )
        
        # Save model
        if format_type == 'json':
            save_to_json(current_model, str(validated_path))
        else:
            save_to_hdf5(current_model, str(validated_path))
        
        logger.info(f"Model saved to {validated_path}")
        
        return jsonify({
            'status': 'success',
            'filepath': str(validated_path)
        })
    except ValueError as e:
        logger.error(f"Invalid file path: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/model/load', methods=['POST'])
def load_model():
    """Load model state from file."""
    global current_model, current_simulation
    
    try:
        data = request.json
        filepath = data.get('filepath', 'saved_models/brain_state.json')
        
        # Determine allowed extensions based on file extension
        if filepath.endswith('.json'):
            allowed_ext = ['.json']
        elif filepath.endswith('.h5'):
            allowed_ext = ['.h5']
        else:
            return jsonify({'status': 'error', 'message': 'Invalid file format'}), 400
        
        # Validate file path to prevent directory traversal
        validated_path = validate_filepath(
            filepath,
            ALLOWED_SAVE_DIR,
            allowed_ext
        )
        
        # Check if file exists
        if not validated_path.exists():
            return jsonify({
                'status': 'error',
                'message': f'File not found: {validated_path.name}'
            }), 404
        
        logger.info(f"Loading model from {validated_path}")
        
        with simulation_lock:
            if validated_path.suffix == '.json':
                current_model = load_from_json(str(validated_path))
            else:
                current_model = load_from_hdf5(str(validated_path))
            
            current_simulation = Simulation(current_model, seed=42)
        
        logger.info(f"Model loaded: {len(current_model.neurons)} neurons")
        
        return jsonify({
            'status': 'success',
            'num_neurons': len(current_model.neurons),
            'num_synapses': len(current_model.synapses)
        })
    except ValueError as e:
        logger.error(f"Invalid file path: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection."""
    logger.info('Client connected')
    emit('connection_response', {'data': 'Connected to 4D Neural Cognition'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection."""
    logger.info('Client disconnected')


@socketio.on('chat_message')
def handle_chat_message(data):
    """Handle chat messages for operations."""
    message = data.get('message', '')
    logger.info(f"Chat message: {message}")
    
    # Simple command processing
    response = process_chat_command(message)
    
    emit('chat_response', {
        'message': response,
        'timestamp': datetime.now().isoformat()
    })


def process_chat_command(command):
    """Process chat commands and return responses."""
    command = command.lower().strip()
    
    if 'help' in command:
        return "Available commands: init, info, step, run <n>, save, load, status"
    elif 'init' in command:
        return "Use the Initialize Model button to create a new model"
    elif 'info' in command or 'status' in command:
        if current_model:
            return f"Model has {len(current_model.neurons)} neurons, {len(current_model.synapses)} synapses, step {current_model.current_step}"
        else:
            return "No model initialized"
    elif 'step' in command:
        return "Use the Step button to run a single simulation step"
    elif 'run' in command:
        return "Use the Train button to run multiple simulation steps"
    else:
        return f"Unknown command: {command}. Type 'help' for available commands."


if __name__ == '__main__':
    logger.info("Starting 4D Neural Cognition web interface")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
