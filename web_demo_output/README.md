# Web Interface Demo Output

This directory contains sample data generated for the web interface demonstration.

## Files

- `analytics_data.json`: Time-series data for spike counts, neuron counts, and synapse counts
- `visualization_data.json`: 3D/4D neuron positions and connections
- `experiments.json`: Sample experiment configurations
- `performance_metrics.json`: Performance evaluation metrics

## Usage

1. Start the web server: `python app.py`
2. Navigate to http://localhost:5000/advanced
3. Use the various tabs to explore the features:
   - **Visualization**: Load the visualization data
   - **Analytics**: Import analytics data to see charts
   - **Experiments**: Import experiment configurations
   - **Collaboration**: Create annotations and versions

## Data Format

### Visualization Data
- Neurons: Array of objects with x, y, z, w coordinates and properties
- Connections: Array of objects with from/to coordinates and weights

### Analytics Data
- Time-series arrays for various metrics
- Compatible with Chart.js visualizations

### Experiments
- Experiment configurations with parameters
- Ready for import into experiment manager

### Performance Metrics
- Accuracy, precision, recall, F1-score, stability
- Format compatible with radar chart visualization
