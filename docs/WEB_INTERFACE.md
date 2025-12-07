# Web Interface Documentation

## Overview

The 4D Neural Cognition project provides two web interfaces for interacting with neural network simulations:

1. **Basic Interface** (`/`) - Simple controls for model management and visualization
2. **Advanced Interface** (`/advanced`) - Comprehensive features including 3D/4D visualization, analytics, experiment management, and collaboration

## Getting Started

### Starting the Server

```bash
python app.py
```

The server will start on `http://localhost:5000` by default.

### Accessing the Interfaces

- **Basic Interface**: Navigate to `http://localhost:5000`
- **Advanced Interface**: Navigate to `http://localhost:5000/advanced`

## Basic Interface

### Features

#### Model Control
- Initialize brain models from configuration files
- View model information (lattice shape, neurons, synapses)
- Initialize neurons in different brain areas
- Create synaptic connections

#### Simulation
- Run single simulation steps
- Execute multi-step training runs
- Stop ongoing simulations
- Automatic checkpointing

#### Visualization
- 2D heatmap views of different layers (input, hidden, output)
- Real-time updates during simulation
- Activity visualization

#### Data Management
- Save models (JSON or HDF5 format)
- Load models from files
- Recover from automatic checkpoints

#### Terminal & Logs
- Sensory input interface
- Real-time system logs
- Chat-based command interface

## Advanced Interface

### 3D/4D Visualization

#### Interactive 3D Viewer

The 3D neuron viewer uses Three.js to provide an interactive visualization of the neural network in 3D space.

**Features**:
- Orbit controls for rotating, zooming, and panning
- Real-time rendering of neurons and connections
- Activity-based coloring and animations
- Grid and axis helpers for spatial reference

**Controls**:
- **Left Click + Drag**: Rotate view
- **Right Click + Drag**: Pan view
- **Scroll**: Zoom in/out
- **Reset Camera**: Return to default view position

#### 4D Projection

The system supports 4D neural coordinates (x, y, z, w) and projects them into 3D space using stereographic projection.

**W-Dimension Slider**:
- Adjust the "slice" of the 4th dimension to visualize
- Range: 0 to 20
- Real-time updates as you adjust the slider

#### Color Mapping Modes

- **Membrane Potential**: Blue (resting) to red (active)
- **Health**: Opacity based on neuron health (0-1)
- **Age**: Color varies with neuron age
- **Activity**: Highlights active neurons

#### Activity Animation

Animate neural activity over time:
- **Time Step Slider**: Select specific time points
- **Animation Speed**: Control playback speed
- **Play/Pause/Stop**: Control animation playback

### Real-time Analytics

#### Spike Rate Histogram

Bar chart showing the number of spikes per time step.

**Features**:
- Rolling window of last 50 steps
- Real-time updates during simulation
- X-axis: Time step
- Y-axis: Spike count

#### Network Statistics

Line chart tracking network composition over time.

**Metrics**:
- Active neurons (green)
- Active synapses (purple)
- Rolling window of last 100 steps

#### Learning Curves

Dual y-axis chart for training metrics.

**Metrics**:
- Training loss (left y-axis, red)
- Accuracy (right y-axis, green)
- Epoch-based tracking

#### Performance Metrics Dashboard

Radar chart displaying multiple performance metrics.

**Metrics**:
- Accuracy
- Precision
- Recall
- F1-Score
- Stability

**Data Export**:
- Export all analytics data as JSON
- Clear accumulated data

### Experiment Management

#### Creating Experiments

1. Click "New Experiment"
2. Enter experiment name
3. Default parameters are applied
4. Experiment appears in the list

#### Experiment Details

Click on any experiment to view:
- Name and description
- Status (created, running, completed, failed)
- Creation date and version
- Parameters (JSON format)
- Results (if completed)

#### Batch Parameter Modification

Modify parameters for multiple experiments at once:

1. Select experiments from the list (click to select)
2. Adjust parameters:
   - Learning Rate
   - Weight Decay
   - Neuron Density
3. Click "Apply to Selected"

#### Parameter Sweeps

Automatically generate multiple experiments with different parameter values:

1. Select parameter to sweep
2. Enter comma-separated values (e.g., `0.001, 0.01, 0.1`)
3. Click "Create Sweep Experiments"
4. System generates one experiment per value

**Example**:
```
Parameter: Learning Rate
Values: 0.001, 0.01, 0.1
Result: 3 experiments with different learning rates
```

#### A/B Testing

Compare two different configurations:

1. Enter test name
2. Click "Setup A/B Test"
3. Two experiments are created (A and B)
4. Run both experiments
5. Click "Compare Results" to see winner

#### Experiment Comparison

Compare multiple experiments:

1. Select 2 or more experiments
2. Click "Compare Selected Experiments"
3. View comparison table with:
   - Ranking by score
   - Parameters
   - Results

#### Import/Export

- **Export**: Click experiment → Downloads JSON file
- **Import**: Click "Import" → Select JSON file

### Collaborative Features

#### Multi-User Support

##### Joining a Session

1. Enter your username
2. Click "Join Session"
3. You'll appear in the Active Users list
4. Other users are notified of your presence

##### Active Users

View all connected users with:
- Avatar (first letter of username)
- Username
- Status (Active)
- Unique color per user

#### Shared Simulations

##### Creating Shared Simulations

1. Join session first
2. Click "New Shared Simulation"
3. Enter simulation name
4. Simulation is created and shared with all users

##### Joining Shared Simulations

Click on any shared simulation to join and participate.

**Features**:
- Multiple users can work on the same simulation
- Real-time synchronization of changes
- Participant list shows who's involved

#### Comments & Annotations

##### Adding Annotations

1. Join session first
2. Enter annotation text
3. Select target type:
   - Simulation
   - Experiment
   - Neuron
4. Click "Add Annotation"

##### Viewing Annotations

Annotations display:
- Author name
- Timestamp
- Annotation text
- Replies (if any)

##### Replying to Annotations

Click on an annotation to reply (feature in UI thread handling).

#### Version Control

##### Creating Versions

1. Enter version name
2. Enter description (optional)
3. Click "Create Version"
4. Version is saved with timestamp

##### Version History

View all versions with:
- Version name
- Creation date
- Description
- Author

##### Comparing Versions

1. Select two versions
2. Click "Compare Versions"
3. View differences in parameters and results

## API Endpoints

### Basic Endpoints

#### Model Management

- `POST /api/model/init` - Initialize model
- `GET /api/model/info` - Get model information
- `POST /api/model/save` - Save model
- `POST /api/model/load` - Load model

#### Simulation

- `POST /api/simulation/step` - Run single step
- `POST /api/simulation/run` - Run multiple steps
- `POST /api/simulation/stop` - Stop simulation
- `POST /api/simulation/recover` - Recover from checkpoint

#### Neurons & Synapses

- `POST /api/neurons/init` - Initialize neurons
- `POST /api/synapses/init` - Initialize synapses

#### Input/Output

- `POST /api/input/feed` - Feed sensory input
- `GET /api/heatmap/data` - Get heatmap visualization data

### Advanced Endpoints

#### Visualization

- `GET /api/visualization/neurons` - Get neuron data for 3D visualization
- `GET /api/visualization/connections` - Get connection data

### Socket.IO Events

#### Client → Server

- `register_user` - Register user for collaboration
- `create_shared_simulation` - Create shared simulation
- `add_annotation` - Add annotation
- `create_version` - Create version
- `chat_message` - Send chat message

#### Server → Client

- `user_joined` - User joined notification
- `user_left` - User left notification
- `simulation_update` - Simulation state update
- `annotation_added` - Annotation added notification
- `version_created` - Version created notification
- `training_progress` - Training progress update
- `log_message` - System log message
- `chat_response` - Chat command response

## Technical Architecture

### Frontend Components

#### JavaScript Modules

1. **visualization.js** - 3D/4D neuron viewer
   - Three.js integration
   - Stereographic 4D projection
   - Animation controls

2. **analytics.js** - Real-time analytics
   - Chart.js integration
   - Multiple chart types
   - Data export

3. **experiments.js** - Experiment management
   - CRUD operations
   - Parameter sweeps
   - A/B testing
   - Comparison tools

4. **collaboration.js** - Collaboration features
   - Socket.IO integration
   - User management
   - Annotations
   - Version control

5. **advanced.js** - Main controller
   - Module integration
   - Event handling
   - UI updates

#### CSS Styling

- **style.css** - Basic interface styles
- **advanced.css** - Advanced interface styles
  - Responsive grid layouts
  - Dark theme
  - Animations

### Backend Components

#### Flask Application (app.py)

- REST API endpoints
- Socket.IO event handlers
- Model and simulation management
- File handling with security validation

#### Core Modules

- **brain_model.py** - Neural network model
- **simulation.py** - Simulation engine
- **senses.py** - Sensory input processing
- **storage.py** - Data persistence

## Best Practices

### Performance

1. **Limit Visualization**: The 3D viewer limits neurons (1000) and connections (500) for performance
2. **Rolling Windows**: Analytics use rolling windows to prevent memory growth
3. **Throttled Updates**: Real-time updates are throttled during long simulations

### Security

1. **Path Validation**: All file paths are validated to prevent traversal attacks
2. **Input Sanitization**: User inputs are validated and limited
3. **Resource Limits**: Simulation steps and data sizes are capped

### Collaboration

1. **Join Session First**: Always join a session before using collaborative features
2. **Unique Usernames**: Use unique usernames to avoid confusion
3. **Save Regularly**: Create versions regularly to track progress

### Experiment Management

1. **Descriptive Names**: Use clear, descriptive names for experiments
2. **Document Parameters**: Add descriptions explaining parameter choices
3. **Export Important Results**: Export successful experiments for backup

## Troubleshooting

### 3D Viewer Not Loading

**Problem**: Black screen in 3D viewer

**Solutions**:
1. Check browser console for errors
2. Ensure Three.js CDN is accessible
3. Try refreshing the page
4. Check WebGL support in browser

### Charts Not Updating

**Problem**: Analytics charts show no data

**Solutions**:
1. Initialize a model first
2. Run some simulation steps
3. Check browser console for errors
4. Verify Chart.js is loaded

### Socket.IO Connection Issues

**Problem**: Collaboration features not working

**Solutions**:
1. Check server is running
2. Verify Socket.IO library is loaded
3. Check browser console for connection errors
4. Ensure firewall allows WebSocket connections

### Slow Performance

**Problem**: Interface is slow or laggy

**Solutions**:
1. Reduce neuron count for visualization
2. Clear analytics data periodically
3. Limit number of active experiments
4. Use modern browser with good performance
5. Close other browser tabs

## Future Enhancements

### Planned Features

1. **Enhanced Visualization**:
   - VR/AR support
   - Custom color schemes
   - Export 3D scenes as images/videos

2. **Advanced Analytics**:
   - Statistical analysis tools
   - Correlation matrices
   - Custom metrics

3. **Experiment Management**:
   - Automated hyperparameter tuning
   - Multi-objective optimization
   - Experiment scheduling

4. **Collaboration**:
   - Video chat integration
   - Shared annotations on 3D viewer
   - Real-time collaborative editing

5. **Integration**:
   - Python notebook integration
   - External data sources
   - Cloud storage support

## Support

For questions or issues:

1. Check [SUPPORT.md](../SUPPORT.md)
2. Review [GitHub Issues](https://github.com/Thomas-Heisig/4D-Neural-Cognition/issues)
3. Consult [Documentation](../docs/)

## License

This web interface is part of the 4D Neural Cognition project and is licensed under the MIT License. See [LICENSE](../LICENSE) for details.
