// API Client Wrapper
// Provides safe API calls with initialization guards and error handling

class APIClient {
    constructor() {
        this.baseUrl = '';
        this.defaultHeaders = {
            'Content-Type': 'application/json'
        };
    }

    /**
     * Make an API call with initialization guard
     * @param {string} endpoint - API endpoint path
     * @param {string} method - HTTP method (GET, POST, etc.)
     * @param {object} data - Request data for POST/PUT
     * @param {boolean} requireInit - Whether to check initialization (default: true)
     * @returns {Promise} Response data
     */
    async call(endpoint, method = 'GET', data = null, requireInit = true) {
        // Check if system is initialized for protected endpoints
        if (requireInit && endpoint !== '/api/system/status' && endpoint !== '/api/model/init') {
            if (!window.systemStatus || !window.systemStatus.initialized) {
                throw new Error('System not initialized. Please initialize the model first.');
            }
        }

        const options = {
            method,
            headers: { ...this.defaultHeaders }
        };

        if (data) {
            options.body = JSON.stringify(data);
        }

        try {
            const response = await fetch(this.baseUrl + endpoint, options);
            const result = await response.json();

            if (!response.ok) {
                // Handle specific HTTP errors
                if (response.status === 400) {
                    throw new Error(result.message || 'Bad Request - System may not be initialized');
                } else if (response.status === 404) {
                    throw new Error(result.message || 'Endpoint not found');
                } else if (response.status === 500) {
                    throw new Error(result.message || 'Internal server error');
                } else {
                    throw new Error(result.message || `HTTP ${response.status}`);
                }
            }

            return result;
        } catch (error) {
            // Add context to the error
            error.endpoint = endpoint;
            error.method = method;
            throw error;
        }
    }

    /**
     * GET request
     */
    async get(endpoint, requireInit = true) {
        return this.call(endpoint, 'GET', null, requireInit);
    }

    /**
     * POST request
     */
    async post(endpoint, data, requireInit = true) {
        return this.call(endpoint, 'POST', data, requireInit);
    }

    /**
     * Check system status (never requires init)
     */
    async getSystemStatus() {
        return this.call('/api/system/status', 'GET', null, false);
    }

    /**
     * Initialize model
     */
    async initModel(configPath = 'brain_base_model.json') {
        return this.call('/api/model/init', 'POST', { config_path: configPath }, false);
    }

    /**
     * Get model info
     */
    async getModelInfo() {
        return this.get('/api/model/info');
    }

    /**
     * Get full configuration
     */
    async getFullConfig() {
        return this.get('/api/config/full');
    }

    /**
     * Update configuration
     */
    async updateConfig(configData) {
        return this.post('/api/config/update', configData);
    }

    /**
     * Get neuron details
     */
    async getNeuronDetails(limit = 50, offset = 0) {
        return this.get(`/api/neurons/details?limit=${limit}&offset=${offset}`);
    }

    /**
     * Get synapse details
     */
    async getSynapseDetails(limit = 50, offset = 0) {
        return this.get(`/api/synapses/details?limit=${limit}&offset=${offset}`);
    }

    /**
     * Get network statistics
     */
    async getNetworkStats() {
        return this.get('/api/stats/network');
    }

    /**
     * Get areas info
     */
    async getAreasInfo() {
        return this.get('/api/areas/info');
    }

    /**
     * Get senses info
     */
    async getSensesInfo() {
        return this.get('/api/senses/info');
    }

    /**
     * Initialize neurons
     */
    async initNeurons(areas = ['V1_like', 'Digital_sensor'], density = 0.1) {
        return this.post('/api/neurons/init', { areas, density });
    }

    /**
     * Initialize synapses
     */
    async initSynapses(probability = 0.001, weightMean = 0.1, weightStd = 0.05) {
        return this.post('/api/synapses/init', {
            probability,
            weight_mean: weightMean,
            weight_std: weightStd
        });
    }

    /**
     * Run simulation step
     */
    async simulationStep() {
        return this.post('/api/simulation/step', {});
    }

    /**
     * Run simulation
     */
    async runSimulation(nSteps = 100, feedbackInterval = 10) {
        return this.post('/api/simulation/run', {
            n_steps: nSteps,
            feedback_interval: feedbackInterval
        });
    }

    /**
     * Stop simulation
     */
    async stopSimulation() {
        return this.post('/api/simulation/stop', {});
    }

    /**
     * Recover from checkpoint
     */
    async recoverCheckpoint() {
        return this.post('/api/simulation/recover', {});
    }

    /**
     * Feed sensory input
     */
    async feedInput(senseType, inputData) {
        return this.post('/api/input/feed', {
            sense_type: senseType,
            input_data: inputData
        });
    }

    /**
     * Get heatmap data
     */
    async getHeatmapData(z = 0) {
        return this.get(`/api/heatmap/data?z=${z}`);
    }

    /**
     * Save model
     */
    async saveModel(filename, format = 'json') {
        return this.post('/api/model/save', { filename, format });
    }

    /**
     * Load model
     */
    async loadModel(filepath) {
        return this.post('/api/model/load', { filepath });
    }

    /**
     * Get visualization neurons
     */
    async getVisualizationNeurons() {
        return this.get('/api/visualization/neurons');
    }

    /**
     * Get visualization connections
     */
    async getVisualizationConnections() {
        return this.get('/api/visualization/connections');
    }

    /**
     * Get VNC status
     */
    async getVNCStatus() {
        return this.get('/api/vnc/status');
    }

    /**
     * Get VNC config
     */
    async getVNCConfig() {
        return this.get('/api/vnc/config');
    }

    /**
     * Update VNC config
     */
    async updateVNCConfig(config) {
        return this.post('/api/vnc/config', config);
    }

    /**
     * Reset VNC
     */
    async resetVNC() {
        return this.post('/api/vnc/reset', {});
    }

    /**
     * Rebalance VNC
     */
    async rebalanceVNC() {
        return this.post('/api/vnc/rebalance', {});
    }
}

// Create global instance
window.apiClient = new APIClient();
