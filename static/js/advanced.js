// Advanced Interface - Main Controller
// Integrates all modules: Visualization, Analytics, Experiments, and Collaboration

// Global instances
let neuralViz = null;
let analytics = null;
let experimentManager = null;
let collaborationManager = null;
let socket = null;

// Initialize Socket.IO
socket = io();

// Initialize all modules when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    initializeModules();
    setupTabNavigation();
    setupEventListeners();
    loadInitialData();
});

// Initialize all modules
function initializeModules() {
    // Initialize visualization
    if (document.getElementById('neuronViewer3D')) {
        neuralViz = new NeuralVisualization('neuronViewer3D');
    }
    
    // Initialize analytics
    analytics = new NetworkAnalytics();
    analytics.initSpikeRateHistogram('spikeRateChart');
    analytics.initNetworkStats('networkStatsChart');
    analytics.initLearningCurves('learningCurveChart');
    analytics.initPerformanceMetrics('performanceMetricsChart');
    
    // Initialize experiment manager
    experimentManager = new ExperimentManager();
    experimentManager.loadFromLocalStorage();
    
    // Initialize collaboration manager
    collaborationManager = new CollaborationManager(socket);
    
    console.log('All modules initialized');
}

// Setup tab navigation
function setupTabNavigation() {
    const tabs = document.querySelectorAll('.nav-tab');
    const contents = document.querySelectorAll('.tab-content');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', function() {
            const targetTab = this.getAttribute('data-tab');
            
            // Remove active class from all tabs and contents
            tabs.forEach(t => t.classList.remove('active'));
            contents.forEach(c => c.classList.remove('active'));
            
            // Add active class to selected tab and content
            this.classList.add('active');
            document.getElementById(`${targetTab}-tab`).classList.add('active');
        });
    });
}

// Setup event listeners for all controls
function setupEventListeners() {
    // Visualization controls
    setupVisualizationListeners();
    
    // Analytics controls
    setupAnalyticsListeners();
    
    // Experiment controls
    setupExperimentListeners();
    
    // Collaboration controls
    setupCollaborationListeners();
    
    // Socket event listeners
    setupSocketListeners();
}

// Visualization event listeners
function setupVisualizationListeners() {
    // W-Dimension slider
    const wDimSlider = document.getElementById('wDimension');
    if (wDimSlider) {
        wDimSlider.addEventListener('input', function() {
            const value = parseFloat(this.value);
            document.getElementById('wDimensionValue').textContent = value.toFixed(1);
            if (neuralViz) {
                neuralViz.set4DProjection(value);
            }
        });
    }
    
    // Load neurons button
    const loadNeuronsBtn = document.getElementById('loadNeuronViz');
    if (loadNeuronsBtn) {
        loadNeuronsBtn.addEventListener('click', async function() {
            try {
                const response = await fetch('/api/visualization/neurons');
                const data = await response.json();
                if (data.status === 'success' && neuralViz) {
                    await neuralViz.loadNeurons(data.neurons);
                    showNotification('Neurons loaded successfully', 'success');
                }
            } catch (error) {
                showNotification('Failed to load neurons: ' + error.message, 'error');
            }
        });
    }
    
    // Load connections button
    const loadConnectionsBtn = document.getElementById('loadConnections');
    if (loadConnectionsBtn) {
        loadConnectionsBtn.addEventListener('click', async function() {
            try {
                const response = await fetch('/api/visualization/connections');
                const data = await response.json();
                if (data.status === 'success' && neuralViz) {
                    await neuralViz.loadConnections(data.connections);
                    showNotification('Connections loaded successfully', 'success');
                }
            } catch (error) {
                showNotification('Failed to load connections: ' + error.message, 'error');
            }
        });
    }
    
    // Animation controls
    const timeStepSlider = document.getElementById('timeStep');
    if (timeStepSlider) {
        timeStepSlider.addEventListener('input', function() {
            const value = parseInt(this.value);
            document.getElementById('timeStepValue').textContent = value;
            if (neuralViz) {
                neuralViz.updateTimeStep(value);
            }
        });
    }
    
    const animSpeedSlider = document.getElementById('animSpeed');
    if (animSpeedSlider) {
        animSpeedSlider.addEventListener('input', function() {
            document.getElementById('animSpeedValue').textContent = this.value;
        });
    }
    
    // Reset camera
    const resetCameraBtn = document.getElementById('resetCamera');
    if (resetCameraBtn) {
        resetCameraBtn.addEventListener('click', function() {
            if (neuralViz && neuralViz.camera) {
                neuralViz.camera.position.set(30, 30, 30);
                neuralViz.camera.lookAt(0, 0, 0);
            }
        });
    }
}

// Analytics event listeners
function setupAnalyticsListeners() {
    // Export analytics data
    const exportBtn = document.getElementById('exportAnalytics');
    if (exportBtn) {
        exportBtn.addEventListener('click', function() {
            if (analytics) {
                const data = analytics.exportData();
                downloadFile('analytics_data.json', data, 'application/json');
                showNotification('Analytics data exported', 'success');
            }
        });
    }
    
    // Clear analytics
    const clearBtn = document.getElementById('clearAnalytics');
    if (clearBtn) {
        clearBtn.addEventListener('click', function() {
            if (confirm('Are you sure you want to clear all analytics data?')) {
                if (analytics) {
                    analytics.clearAllData();
                    showNotification('Analytics data cleared', 'info');
                }
            }
        });
    }
}

// Experiment event listeners
function setupExperimentListeners() {
    // Create experiment
    const createExpBtn = document.getElementById('createExperiment');
    if (createExpBtn) {
        createExpBtn.addEventListener('click', function() {
            const name = prompt('Experiment name:');
            if (name) {
                const params = {
                    learning_rate: 0.01,
                    density: 0.1,
                    weight_mean: 0.1
                };
                const exp = experimentManager.createExperiment(name, 'New experiment', params);
                updateExperimentList();
                showNotification('Experiment created: ' + name, 'success');
            }
        });
    }
    
    // Import experiment
    const importBtn = document.getElementById('importExperiment');
    if (importBtn) {
        importBtn.addEventListener('click', function() {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.json';
            input.onchange = function(e) {
                const file = e.target.files[0];
                const reader = new FileReader();
                reader.onload = function(event) {
                    try {
                        const exp = experimentManager.importExperiment(event.target.result);
                        updateExperimentList();
                        showNotification('Experiment imported: ' + exp.name, 'success');
                    } catch (error) {
                        showNotification('Failed to import: ' + error.message, 'error');
                    }
                };
                reader.readAsText(file);
            };
            input.click();
        });
    }
    
    // Apply batch parameters
    const applyBatchBtn = document.getElementById('applyBatchParams');
    if (applyBatchBtn) {
        applyBatchBtn.addEventListener('click', function() {
            const selectedExps = getSelectedExperiments();
            if (selectedExps.length === 0) {
                showNotification('No experiments selected', 'warning');
                return;
            }
            
            const params = {
                learning_rate: parseFloat(document.getElementById('batchLearningRate').value),
                weight_decay: parseFloat(document.getElementById('batchWeightDecay').value),
                density: parseFloat(document.getElementById('batchDensity').value)
            };
            
            experimentManager.batchModifyParameters(selectedExps, params);
            updateExperimentList();
            showNotification('Parameters updated for ' + selectedExps.length + ' experiments', 'success');
        });
    }
    
    // Create parameter sweep
    const createSweepBtn = document.getElementById('createSweep');
    if (createSweepBtn) {
        createSweepBtn.addEventListener('click', function() {
            const param = document.getElementById('sweepParam').value;
            const valuesStr = document.getElementById('sweepValues').value;
            
            if (!valuesStr) {
                showNotification('Please enter sweep values', 'warning');
                return;
            }
            
            const values = valuesStr.split(',').map(v => parseFloat(v.trim()));
            const sweepConfig = { [param]: values };
            
            const baseParams = {
                learning_rate: 0.01,
                density: 0.1,
                weight_mean: 0.1
            };
            
            const experiments = experimentManager.createSweepExperiments(
                'ParameterSweep',
                baseParams,
                sweepConfig
            );
            
            updateExperimentList();
            showNotification('Created ' + experiments.length + ' sweep experiments', 'success');
        });
    }
    
    // Setup A/B test
    const setupABBtn = document.getElementById('setupABTest');
    if (setupABBtn) {
        setupABBtn.addEventListener('click', function() {
            const testName = document.getElementById('abTestName').value;
            if (!testName) {
                showNotification('Please enter test name', 'warning');
                return;
            }
            
            const configA = { learning_rate: 0.01, density: 0.1 };
            const configB = { learning_rate: 0.001, density: 0.2 };
            
            const test = experimentManager.setupABTest(testName, configA, configB);
            updateExperimentList();
            showNotification('A/B test created: ' + testName, 'success');
        });
    }
    
    // Compare selected experiments
    const compareBtn = document.getElementById('compareSelected');
    if (compareBtn) {
        compareBtn.addEventListener('click', function() {
            const selected = getSelectedExperiments();
            if (selected.length < 2) {
                showNotification('Please select at least 2 experiments', 'warning');
                return;
            }
            
            const comparison = experimentManager.compareExperiments(selected);
            displayComparison(comparison);
        });
    }
}

// Collaboration event listeners
function setupCollaborationListeners() {
    // Join session
    const joinBtn = document.getElementById('joinSession');
    if (joinBtn) {
        joinBtn.addEventListener('click', function() {
            const username = document.getElementById('username').value;
            if (username) {
                const user = collaborationManager.registerUser(username);
                updateActiveUsers();
                showNotification('Joined as ' + username, 'success');
            } else {
                showNotification('Please enter a username', 'warning');
            }
        });
    }
    
    // Create shared simulation
    const createSharedBtn = document.getElementById('createSharedSim');
    if (createSharedBtn) {
        createSharedBtn.addEventListener('click', function() {
            if (!collaborationManager.currentUser) {
                showNotification('Please join session first', 'warning');
                return;
            }
            
            const name = prompt('Simulation name:');
            if (name) {
                const sim = collaborationManager.createSharedSimulation(
                    name,
                    'Shared simulation',
                    {}
                );
                updateSharedSimulations();
                showNotification('Shared simulation created', 'success');
            }
        });
    }
    
    // Add annotation
    const addAnnotationBtn = document.getElementById('addAnnotation');
    if (addAnnotationBtn) {
        addAnnotationBtn.addEventListener('click', function() {
            if (!collaborationManager.currentUser) {
                showNotification('Please join session first', 'warning');
                return;
            }
            
            const text = document.getElementById('annotationText').value;
            const target = document.getElementById('annotationTarget').value;
            
            if (text) {
                const annotation = collaborationManager.addAnnotation(target, 'current', text);
                document.getElementById('annotationText').value = '';
                updateAnnotations();
                showNotification('Annotation added', 'success');
            } else {
                showNotification('Please enter annotation text', 'warning');
            }
        });
    }
    
    // Create version
    const createVersionBtn = document.getElementById('createVersion');
    if (createVersionBtn) {
        createVersionBtn.addEventListener('click', function() {
            if (!collaborationManager.currentUser) {
                showNotification('Please join session first', 'warning');
                return;
            }
            
            const name = document.getElementById('versionName').value;
            const desc = document.getElementById('versionDescription').value;
            
            if (name) {
                const version = collaborationManager.createVersion(
                    'current',
                    name,
                    desc,
                    { timestamp: Date.now() }
                );
                document.getElementById('versionName').value = '';
                document.getElementById('versionDescription').value = '';
                updateVersionHistory();
                showNotification('Version created: ' + name, 'success');
            } else {
                showNotification('Please enter version name', 'warning');
            }
        });
    }
}

// Socket event listeners
function setupSocketListeners() {
    socket.on('connect', () => {
        console.log('Connected to server');
    });
    
    socket.on('training_progress', (data) => {
        if (analytics) {
            analytics.updateSpikeRates(data.step, data.spikes);
            analytics.updateNetworkStats(data.step, data.neurons, data.synapses);
        }
    });
    
    // Collaboration events
    document.addEventListener('collab:userJoined', (e) => {
        updateActiveUsers();
        showNotification(e.detail.username + ' joined', 'info');
    });
    
    document.addEventListener('collab:userLeft', (e) => {
        updateActiveUsers();
    });
    
    document.addEventListener('collab:annotationAdded', (e) => {
        updateAnnotations();
    });
    
    document.addEventListener('collab:versionCreated', (e) => {
        updateVersionHistory();
    });
}

// UI Update Functions
function updateExperimentList() {
    const list = document.getElementById('experimentList');
    if (!list) return;
    
    const experiments = experimentManager.getAllExperiments();
    
    list.innerHTML = experiments.map(exp => `
        <div class="experiment-item" data-id="${exp.id}">
            <div class="experiment-name">${exp.name}</div>
            <div class="experiment-meta">
                <span class="status-badge status-${exp.status}">${exp.status}</span>
                Created: ${new Date(exp.createdAt).toLocaleDateString()}
            </div>
        </div>
    `).join('');
    
    // Add click handlers
    list.querySelectorAll('.experiment-item').forEach(item => {
        item.addEventListener('click', function() {
            list.querySelectorAll('.experiment-item').forEach(i => i.classList.remove('selected'));
            this.classList.add('selected');
            
            const expId = this.getAttribute('data-id');
            displayExperimentDetails(expId);
        });
    });
}

function displayExperimentDetails(experimentId) {
    const details = document.getElementById('experimentDetails');
    if (!details) return;
    
    const experiments = experimentManager.getAllExperiments();
    const exp = experiments.find(e => e.id === experimentId);
    
    if (!exp) return;
    
    details.innerHTML = `
        <h3>${exp.name}</h3>
        <p>${exp.description}</p>
        <div class="experiment-meta">
            <p><strong>Status:</strong> <span class="status-badge status-${exp.status}">${exp.status}</span></p>
            <p><strong>Created:</strong> ${new Date(exp.createdAt).toLocaleString()}</p>
            <p><strong>Version:</strong> ${exp.version}</p>
        </div>
        <h4>Parameters:</h4>
        <pre>${JSON.stringify(exp.parameters, null, 2)}</pre>
        ${exp.results ? `<h4>Results:</h4><pre>${JSON.stringify(exp.results, null, 2)}</pre>` : ''}
    `;
}

function displayComparison(comparison) {
    const view = document.getElementById('comparisonView');
    if (!view || !comparison) return;
    
    const table = `
        <table class="comparison-table">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Name</th>
                    <th>Score</th>
                    <th>Parameters</th>
                </tr>
            </thead>
            <tbody>
                ${comparison.experiments.map((exp, idx) => `
                    <tr>
                        <td>${idx + 1}</td>
                        <td>${exp.name}</td>
                        <td>${exp.score ? exp.score.toFixed(3) : 'N/A'}</td>
                        <td><pre>${JSON.stringify(exp.parameters, null, 2)}</pre></td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
    
    view.innerHTML = table;
}

function updateActiveUsers() {
    const list = document.getElementById('activeUsers');
    if (!list) return;
    
    const users = collaborationManager.getActiveUsers();
    
    list.innerHTML = users.map(user => `
        <div class="user-item">
            <div class="user-avatar" style="background: ${user.color}">
                ${user.username.charAt(0).toUpperCase()}
            </div>
            <div class="user-info">
                <div class="user-name">${user.username}</div>
                <div class="user-status">Active</div>
            </div>
        </div>
    `).join('');
}

function updateSharedSimulations() {
    const list = document.getElementById('sharedSimulations');
    if (!list) return;
    
    const sims = collaborationManager.sharedSimulations;
    
    list.innerHTML = sims.map(sim => `
        <div class="shared-sim-item" data-id="${sim.id}">
            <div class="sim-name">${sim.name}</div>
            <div class="sim-participants">${sim.participants.length} participants</div>
        </div>
    `).join('');
}

function updateAnnotations() {
    const list = document.getElementById('annotationsList');
    if (!list) return;
    
    const annotations = collaborationManager.annotations;
    
    list.innerHTML = annotations.map(ann => `
        <div class="annotation-item">
            <div class="annotation-header">
                <span class="annotation-author">${ann.authorName}</span>
                <span class="annotation-time">${new Date(ann.createdAt).toLocaleString()}</span>
            </div>
            <div class="annotation-text">${ann.text}</div>
            ${ann.replies.length > 0 ? `
                <div class="annotation-replies">
                    ${ann.replies.map(reply => `
                        <div class="annotation-reply">
                            <strong>${reply.authorName}:</strong> ${reply.text}
                        </div>
                    `).join('')}
                </div>
            ` : ''}
        </div>
    `).join('');
}

function updateVersionHistory() {
    const list = document.getElementById('versionHistory');
    if (!list) return;
    
    const versions = collaborationManager.getVersionHistory('current');
    
    list.innerHTML = versions.map(ver => `
        <div class="version-item" data-id="${ver.id}">
            <div class="version-header">
                <span class="version-name">${ver.name}</span>
                <span class="version-date">${new Date(ver.createdAt).toLocaleString()}</span>
            </div>
            <div class="version-description">${ver.description}</div>
            <div class="version-author">By ${ver.createdByName}</div>
        </div>
    `).join('');
}

// Helper functions
function getSelectedExperiments() {
    const selected = [];
    document.querySelectorAll('.experiment-item.selected').forEach(item => {
        selected.push(item.getAttribute('data-id'));
    });
    return selected;
}

function showNotification(message, type = 'info') {
    console.log(`[${type.toUpperCase()}] ${message}`);
    
    // Could integrate with a toast notification library here
    const colors = {
        success: '#5cb85c',
        error: '#d9534f',
        warning: '#f0ad4e',
        info: '#4a90e2'
    };
    
    // Simple notification (could be replaced with a better UI component)
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${colors[type] || colors.info};
        color: white;
        padding: 15px 20px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        z-index: 10000;
        animation: slideIn 0.3s ease-out;
    `;
    // Use textContent to prevent XSS attacks
    notification.textContent = String(message);
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

function downloadFile(filename, content, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Load initial data
function loadInitialData() {
    updateExperimentList();
    updateActiveUsers();
    updateSharedSimulations();
    updateAnnotations();
    updateVersionHistory();
}

// CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style);
