# 4D Neural Cognition - Comprehensive Dashboard Guide

## Overview

The comprehensive dashboard provides a modern, feature-rich interface for complete control and monitoring of the 4D Neural Cognition system. This guide explains all features and capabilities.

## Accessing the Dashboard

The dashboard can be accessed at: `http://localhost:5000/dashboard`

Navigation links are available on all pages:
- **Main Interface**: Basic controls and visualization
- **Dashboard**: Comprehensive settings and monitoring (this page)
- **Advanced Interface**: 3D/4D visualization and collaboration

## Dashboard Sections

### 1. 📈 Systemstatus (System Status)

**Overview Section** - The main view showing current system state:

- **Model Status Card**: Shows if model is initialized and basic info
- **Neurons Card**: Total neuron count and average membrane potential
- **Synapses Card**: Total synapse count and average weight
- **Simulation Card**: Current simulation step and status

**Real-time Charts**:
- Network Activity Chart: Visual representation of network activity
- Synapse Weights Chart: Distribution of synaptic weights

**Features**:
- Auto-refresh capability
- Real-time updates via WebSocket
- Color-coded status indicators

### 2. ⚙️ Einstellungen (Settings)

**Complete Configuration Control** - Modify all system parameters:

#### Modell Konfiguration (Model Configuration)
- **Gitter Form (Lattice Shape)**: Define 4D lattice dimensions (x, y, z, w)
- **Dimensions**: Set number of dimensions (3 or 4)

#### Neuron Modell (Neuron Model)
- **Modell Typ**: Choose between LIF, Izhikevich, or Hodgkin-Huxley
- **Tau m**: Membrane time constant (ms)
- **V Resting**: Resting membrane potential (mV)
- **V Reset**: Reset potential after spike (mV)
- **V Threshold**: Spike threshold potential (mV)
- **Refractory Period**: Post-spike refractory period (ms)

#### Plastizität (Plasticity)
- **Lernregel**: Learning rule (Hebb-like, STDP, BCM)
- **Lernrate**: Learning rate for synaptic updates
- **Gewicht Min/Max**: Minimum and maximum synaptic weights
- **Gewicht Decay**: Weight decay rate
- **Homöostatisch**: Enable homeostatic plasticity

#### Zell-Lebenszyklus (Cell Lifecycle)
- **Tod aktivieren**: Enable neuron death
- **Reproduktion aktivieren**: Enable neuron reproduction
- **Max. Alter**: Maximum neuron age
- **Gesundheits-Verfall**: Health decay per simulation step
- **Mutations-Standardabweichung**: Mutation rates for parameters and weights

#### Neuromodulation
- **Dopamin**: Baseline level and decay rate
- **Serotonin**: Baseline level and decay rate
- **Norepinephrin**: Baseline level

**Actions**:
- **Laden**: Load configuration from model
- **Speichern**: Save current configuration
- **Zurücksetzen**: Reset to default values
- **Einstellungen anwenden**: Apply changes to model (requires restart)

### 3. 🕸️ Netzwerk Details (Network Details)

**Three Tabs for Detailed Network Information**:

#### Neuronen Tab
- Paginated table showing all neurons
- Columns: ID, Position (x,y,z,w), Membrane Potential, Health, Age, Type
- Search and filter functionality
- Navigate through pages with Prev/Next buttons

#### Synapsen Tab
- Paginated table showing all synapses
- Columns: Pre-Neuron ID, Post-Neuron ID, Weight, Delay
- Filter by positive/negative weights
- Search functionality

#### Bereiche Tab
- Grid view of all brain areas
- Shows: Area name, associated sense, neuron count, coordinate ranges
- Visual cards for each area

### 4. 📡 Echtzeit-Überwachung (Real-time Monitoring)

**Live Monitoring Displays**:
- **Spikes pro Sekunde**: Track spike rate over time
- **Durchschn. Membranpotential**: Average membrane potential across network
- **Netzwerkgesundheit**: Overall network health metrics
- **Synaptische Aktivität**: Synaptic activity levels

**Controls**:
- Start/Pause monitoring
- Auto-updates every 2 seconds when active

### 5. ▶️ Simulations-Steuerung (Simulation Control)

**Quick Control Panel**:
- **Initialisieren**: Initialize new model
- **Einzelschritt**: Run single simulation step
- **Starten**: Start simulation run
- **Pause**: Pause ongoing simulation
- **Stoppen**: Stop simulation
- **Wiederherstellen**: Recover from last checkpoint

**Parameters**:
- **Anzahl Schritte**: Number of steps to run (1-100,000)
- **Neuronendichte**: Neuron density for initialization (0.01-1.0)
- **Verbindungswahrscheinlichkeit**: Connection probability (0.0001-0.1)

**Progress Tracking**:
- Visual progress bar
- Current step / Total steps
- Percentage complete
- Estimated remaining time

### 6. ⚡ Neuronen (Neurons)

Dedicated control panel for neuron operations:
- Initialize neurons in specific areas
- Set density parameters
- Monitor neuron creation/deletion

### 7. 🔗 Synapsen (Synapses)

Dedicated control panel for synapse operations:
- Initialize random connections
- Set connection probabilities
- Configure weight distributions
- Monitor synapse statistics

### 8. 👁️ Sinne (Senses)

**Sensory Input Management**:

**Available Senses**:
- 👁️ Vision
- 👂 Audition
- ✋ Somatosensorisch (Somatosensory)
- 👅 Geschmack (Taste)
- 👃 Geruch (Smell)
- ⚖️ Vestibulär (Vestibular)
- 💻 Digital

**Features**:
- Select sense type
- View sense configuration (area, w-index, input size)
- Send input data (JSON or text format)
- Monitor sense activity

### 9. 📊 Statistische Analyse (Statistical Analysis)

**Comprehensive Network Statistics**:

**Neuron Statistics**:
- Total count
- Excitatory vs Inhibitory breakdown
- Average membrane potential
- Average health
- Average age

**Synapse Statistics**:
- Total count
- Positive vs Negative weights
- Average weight
- Distribution charts

**Visualizations**:
- Membrane potential distribution histogram
- Weight distribution histogram
- Age distribution histogram

### 10. 🎨 Visualisierung (Visualization)

**Heatmap Displays**:
- **Input Layer (z=0)**: Visual representation of input layer activity
- **Hidden Layer (z=10)**: Middle layer activity patterns
- **Output Layer (z=19)**: Output layer activation

**Features**:
- Color-coded membrane potentials
- Real-time updates
- Link to advanced 3D/4D visualization

### 11. 💾 Speichern & Laden (Storage)

**Model Persistence**:

**Save Model**:
- Enter filename
- Choose format (JSON or HDF5)
- Save to `saved_models/` directory

**Load Model**:
- Enter filepath
- Load from saved state
- Restores all neurons, synapses, and configuration

**Checkpoint Management**:
- View available checkpoints
- Load last checkpoint
- Automatic checkpoints every 1000 steps
- Keeps last 3 checkpoints

### 12. 📋 System-Protokolle (System Logs)

**Logging Interface**:
- Real-time log display
- Filter by level (Info, Warning, Error)
- Color-coded entries
- Clear logs function
- Export logs to file

**Log Types**:
- **Info**: General information and status updates
- **Warning**: Non-critical issues
- **Error**: Critical errors and failures

### 13. 📤 Export & Import (Export/Import)

**Configuration Management**:

**Export Configuration**:
- Download current configuration as JSON
- Includes all settings and parameters
- Can be shared or version controlled

**Import Configuration**:
- Upload configuration JSON file
- Apply settings from file
- Validates configuration before applying

**Data Export**:
- Export neurons data (CSV/JSON)
- Export synapses data (CSV/JSON)
- Export statistics and metrics
- Supports large datasets (up to 10,000 entries)

## API Endpoints

The dashboard uses the following API endpoints:

### Configuration
- `GET /api/config/full` - Get complete configuration
- `POST /api/config/update` - Update configuration parameters

### Model Information
- `GET /api/model/info` - Basic model information
- `GET /api/stats/network` - Comprehensive network statistics

### Network Details
- `GET /api/neurons/details?limit=X&offset=Y` - Paginated neuron data
- `GET /api/synapses/details?limit=X&offset=Y` - Paginated synapse data
- `GET /api/areas/info` - Brain areas information
- `GET /api/senses/info` - Senses configuration

### Existing Endpoints
- `POST /api/model/init` - Initialize model
- `POST /api/simulation/step` - Run single step
- `POST /api/simulation/run` - Run multiple steps
- `POST /api/simulation/stop` - Stop simulation
- `POST /api/simulation/recover` - Recover from checkpoint
- `POST /api/input/feed` - Send sensory input
- `POST /api/model/save` - Save model
- `POST /api/model/load` - Load model
- `GET /api/heatmap/data` - Get heatmap visualization data

## Design Features

### Modern Dark Theme
- Dark background (#1a1d29) for reduced eye strain
- High contrast text for readability
- Color-coded elements for quick identification
- Smooth animations and transitions

### Responsive Layout
- Sidebar navigation for quick access
- Grid-based layouts adapt to screen size
- Mobile-friendly design
- Collapsible sections

### Visual Feedback
- Loading indicators
- Success/error messages
- Real-time status updates
- Progress bars for long operations

### Color Coding
- **Primary Blue** (#4a90e2): Main actions and highlights
- **Success Green** (#28a745): Positive actions and confirmations
- **Warning Yellow** (#ffc107): Caution and warnings
- **Danger Red** (#dc3545): Stop actions and errors
- **Secondary Gray** (#6c757d): Secondary actions

## WebSocket Features

The dashboard uses WebSocket connections for:
- Real-time log streaming
- Simulation progress updates
- Network status notifications
- Automatic UI updates

## Best Practices

### Configuration Changes
1. Load current settings first
2. Modify desired parameters
3. Apply settings
4. Restart simulation to see effects

### Monitoring
1. Start with overview to check system health
2. Enable real-time monitoring for detailed insights
3. Check logs for any warnings or errors
4. Use analytics for deep dive into network behavior

### Data Management
1. Save models regularly during long simulations
2. Use checkpoints for automatic recovery
3. Export configurations for reproducibility
4. Import configurations for experiment comparison

### Performance
1. Use pagination for large datasets
2. Limit visualization updates to reduce load
3. Stop monitoring when not needed
4. Clear logs periodically

## Troubleshooting

### Model Not Initialized
**Solution**: Click "Initialisieren" in Simulation Control section

### Connection Lost
**Solution**: Check if server is running, refresh page to reconnect

### Settings Not Applied
**Solution**: Click "Einstellungen anwenden" and restart simulation

### Empty Tables
**Solution**: Initialize neurons/synapses first using Simulation Control

### Slow Performance
**Solution**: Reduce monitoring frequency, limit table page size

## Keyboard Shortcuts

Currently, the dashboard doesn't implement keyboard shortcuts, but this is planned for future versions.

## Future Enhancements

Planned features for future updates:
- Custom dashboard layouts
- Saved workspace configurations
- Advanced filtering and search
- Data export to multiple formats
- Integration with external analysis tools
- Collaborative features
- Custom visualization plugins

## Support

For issues or questions:
1. Check logs section for error messages
2. Refer to main project documentation
3. Open an issue on GitHub
4. Contact project maintainers

## Version Information

- **Dashboard Version**: 1.0
- **Compatible with**: 4D Neural Cognition v1.0+
- **Last Updated**: 2025-12-13

---

**Note**: This dashboard provides comprehensive control over the neural network simulation. Always ensure you understand the implications of configuration changes before applying them to active simulations.
