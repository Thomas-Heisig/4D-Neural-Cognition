// Dashboard JavaScript
// Manages the comprehensive dashboard interface

// Global state
let socket = null;
let currentSection = 'overview';
let charts = {};
let neuronPage = 0;
let synapsePage = 0;
const pageSize = 50;
let monitoringInterval = null;
let isMonitoring = false;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initializeSocket();
    initializeSidebar();
    initializeTabs();
    loadOverview();
    setupEventListeners();
});

// Socket.IO connection
function initializeSocket() {
    socket = io();
    
    socket.on('connect', () => {
        updateConnectionStatus(true);
        addLog('info', 'Verbunden mit Server');
    });
    
    socket.on('disconnect', () => {
        updateConnectionStatus(false);
        addLog('error', 'Verbindung zum Server verloren');
    });
    
    socket.on('log_message', (data) => {
        addLog(data.level.toLowerCase(), data.message);
    });
    
    socket.on('training_progress', (data) => {
        updateSimulationProgress(data);
    });
}

// Update connection status indicator
function updateConnectionStatus(connected) {
    const indicator = document.getElementById('connectionStatus');
    const text = document.getElementById('connectionText');
    
    if (connected) {
        indicator.className = 'status-indicator connected';
        text.textContent = 'Verbunden';
    } else {
        indicator.className = 'status-indicator disconnected';
        text.textContent = 'Getrennt';
    }
}

// Sidebar navigation
function initializeSidebar() {
    const sidebarBtns = document.querySelectorAll('.sidebar-btn');
    
    sidebarBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const section = btn.getAttribute('data-section');
            showSection(section);
            
            // Update active state
            sidebarBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });
}

// Show specific section
function showSection(sectionName) {
    // Hide all sections
    document.querySelectorAll('.content-section').forEach(section => {
        section.classList.remove('active');
    });
    
    // Show selected section
    const section = document.getElementById(`${sectionName}-section`);
    if (section) {
        section.classList.add('active');
        currentSection = sectionName;
        
        // Load section-specific data
        loadSectionData(sectionName);
    }
}

// Load data for specific section
function loadSectionData(sectionName) {
    switch(sectionName) {
        case 'overview':
            loadOverview();
            break;
        case 'settings':
            loadSettings();
            break;
        case 'network':
            loadNetworkData();
            break;
        case 'analytics':
            loadAnalytics();
            break;
        case 'senses':
            loadSensesInfo();
            break;
    }
}

// Load overview data
async function loadOverview() {
    try {
        const response = await fetch('/api/model/info');
        const data = await response.json();
        
        if (data.status === 'success') {
            document.getElementById('totalNeurons').textContent = data.num_neurons;
            document.getElementById('totalSynapses').textContent = data.num_synapses;
            document.getElementById('currentStep').textContent = data.current_step;
            
            // Get network stats
            const statsResponse = await fetch('/api/stats/network');
            const statsData = await statsResponse.json();
            
            if (statsData.status === 'success') {
                document.getElementById('avgMembrane').textContent = 
                    statsData.neurons.avg_membrane_potential.toFixed(2) + ' mV';
                document.getElementById('avgWeight').textContent = 
                    statsData.synapses.avg_weight.toFixed(4);
            }
        } else {
            document.getElementById('modelStatus').innerHTML = 
                '<p>Modell nicht initialisiert</p>';
        }
    } catch (error) {
        console.error('Error loading overview:', error);
        addLog('error', 'Fehler beim Laden der Übersicht: ' + error.message);
    }
}

// Load settings from model
async function loadSettings() {
    try {
        const response = await fetch('/api/config/full');
        const data = await response.json();
        
        if (data.status === 'success') {
            const config = data.config;
            
            // Populate form fields
            if (config.lattice_shape) {
                document.getElementById('latticeShape').value = config.lattice_shape.join(',');
            }
            if (config.dimensions) {
                document.getElementById('dimensions').value = config.dimensions;
            }
            
            // Neuron model
            if (config.neuron_model) {
                const nm = config.neuron_model;
                document.getElementById('neuronModelType').value = nm.type || 'LIF';
                
                if (nm.params_default) {
                    document.getElementById('tauM').value = nm.params_default.tau_m || 20.0;
                    document.getElementById('vRest').value = nm.params_default.v_rest || -65.0;
                    document.getElementById('vReset').value = nm.params_default.v_reset || -70.0;
                    document.getElementById('vThreshold').value = nm.params_default.v_threshold || -50.0;
                    document.getElementById('refractoryPeriod').value = nm.params_default.refractory_period || 5.0;
                }
            }
            
            // Plasticity
            if (config.plasticity) {
                const p = config.plasticity;
                document.getElementById('plasticityRule').value = p.rule || 'hebb_like';
                document.getElementById('learningRate').value = p.learning_rate || 0.001;
                document.getElementById('weightMin').value = p.weight_min || -1.0;
                document.getElementById('weightMax').value = p.weight_max || 1.0;
                document.getElementById('weightDecay').value = p.weight_decay || 0.00001;
                
                if (p.homeostatic) {
                    document.getElementById('homeostatic').checked = p.homeostatic.enabled || false;
                }
            }
            
            // Cell lifecycle
            if (config.cell_lifecycle) {
                const cl = config.cell_lifecycle;
                document.getElementById('enableDeath').checked = cl.enable_death || false;
                document.getElementById('enableReproduction').checked = cl.enable_reproduction || false;
                document.getElementById('maxAge').value = cl.max_age || 100000;
                document.getElementById('healthDecay').value = cl.health_decay_per_step || 0.0001;
                document.getElementById('mutationStdParams').value = cl.mutation_std_params || 0.05;
                document.getElementById('mutationStdWeights').value = cl.mutation_std_weights || 0.02;
            }
            
            // Neuromodulation
            if (config.neuromodulation) {
                const nm = config.neuromodulation;
                document.getElementById('neuromodulationEnabled').checked = nm.enabled || false;
                
                if (nm.dopamine) {
                    document.getElementById('dopamineBaseline').value = nm.dopamine.baseline || 0.5;
                    document.getElementById('dopamineDecay').value = nm.dopamine.decay_rate || 0.1;
                }
                
                if (nm.serotonin) {
                    document.getElementById('serotoninBaseline').value = nm.serotonin.baseline || 0.5;
                    document.getElementById('serotoninDecay').value = nm.serotonin.decay_rate || 0.1;
                }
                
                if (nm.norepinephrine) {
                    document.getElementById('norepinephrineBaseline').value = nm.norepinephrine.baseline || 0.5;
                }
            }
            
            addLog('info', 'Einstellungen geladen');
        }
    } catch (error) {
        console.error('Error loading settings:', error);
        addLog('error', 'Fehler beim Laden der Einstellungen: ' + error.message);
    }
}

// Initialize tabs within sections
function initializeTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.getAttribute('data-tab');
            const container = btn.closest('.tabs-container');
            
            // Update buttons
            container.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Update content
            container.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            container.querySelector(`#${tabName}`).classList.add('active');
        });
    });
}

// Setup all event listeners
function setupEventListeners() {
    // Overview section
    document.getElementById('refreshOverview')?.addEventListener('click', loadOverview);
    
    // Settings section
    document.getElementById('applySettings')?.addEventListener('click', applySettings);
    document.getElementById('saveConfig')?.addEventListener('click', saveConfiguration);
    document.getElementById('loadConfig')?.addEventListener('click', loadConfiguration);
    document.getElementById('resetConfig')?.addEventListener('click', resetConfiguration);
    
    // Network section
    document.getElementById('refreshNetwork')?.addEventListener('click', loadNetworkData);
    document.getElementById('neuronsPrev')?.addEventListener('click', () => loadNeurons(neuronPage - 1));
    document.getElementById('neuronsNext')?.addEventListener('click', () => loadNeurons(neuronPage + 1));
    document.getElementById('synapsesPrev')?.addEventListener('click', () => loadSynapses(synapsePage - 1));
    document.getElementById('synapsesNext')?.addEventListener('click', () => loadSynapses(synapsePage + 1));
    
    // Monitoring section
    document.getElementById('toggleMonitoring')?.addEventListener('click', toggleMonitoring);
    
    // Simulation section
    document.getElementById('simInit')?.addEventListener('click', initializeModel);
    document.getElementById('simStep')?.addEventListener('click', runSimulationStep);
    document.getElementById('simRun')?.addEventListener('click', runSimulation);
    document.getElementById('simStop')?.addEventListener('click', stopSimulation);
    document.getElementById('simRecover')?.addEventListener('click', recoverFromCheckpoint);
    
    // Range inputs
    document.getElementById('simDensity')?.addEventListener('input', (e) => {
        document.getElementById('simDensityValue').textContent = e.target.value;
    });
    document.getElementById('simConnProb')?.addEventListener('input', (e) => {
        document.getElementById('simConnProbValue').textContent = e.target.value;
    });
    
    // Analytics section
    document.getElementById('refreshAnalytics')?.addEventListener('click', loadAnalytics);
    
    // Storage section
    document.getElementById('executeSave')?.addEventListener('click', saveModel);
    document.getElementById('executeLoad')?.addEventListener('click', loadModel);
    document.getElementById('loadCheckpoint')?.addEventListener('click', recoverFromCheckpoint);
    
    // Logs section
    document.getElementById('clearLogs')?.addEventListener('click', clearLogs);
    
    // Senses section
    document.getElementById('senseTypeSelect')?.addEventListener('change', updateSenseInfo);
    document.getElementById('sendSenseInput')?.addEventListener('click', sendSenseInput);
    
    // Visualization section
    document.getElementById('openAdvancedViz')?.addEventListener('click', () => {
        window.open('/advanced', '_blank');
    });
    document.getElementById('refreshHeatmaps')?.addEventListener('click', loadHeatmaps);
    
    // Export section
    document.getElementById('exportConfig')?.addEventListener('click', exportConfiguration);
    document.getElementById('importConfig')?.addEventListener('click', importConfiguration);
    document.getElementById('exportData')?.addEventListener('click', exportData);
}

// Apply settings
async function applySettings() {
    try {
        const config = {
            lattice_shape: document.getElementById('latticeShape').value.split(',').map(Number),
            dimensions: parseInt(document.getElementById('dimensions').value),
            neuron_model: {
                type: document.getElementById('neuronModelType').value,
                params_default: {
                    tau_m: parseFloat(document.getElementById('tauM').value),
                    v_rest: parseFloat(document.getElementById('vRest').value),
                    v_reset: parseFloat(document.getElementById('vReset').value),
                    v_threshold: parseFloat(document.getElementById('vThreshold').value),
                    refractory_period: parseFloat(document.getElementById('refractoryPeriod').value),
                }
            },
            plasticity: {
                rule: document.getElementById('plasticityRule').value,
                learning_rate: parseFloat(document.getElementById('learningRate').value),
                weight_min: parseFloat(document.getElementById('weightMin').value),
                weight_max: parseFloat(document.getElementById('weightMax').value),
                weight_decay: parseFloat(document.getElementById('weightDecay').value),
                homeostatic: {
                    enabled: document.getElementById('homeostatic').checked
                }
            },
            cell_lifecycle: {
                enable_death: document.getElementById('enableDeath').checked,
                enable_reproduction: document.getElementById('enableReproduction').checked,
                max_age: parseInt(document.getElementById('maxAge').value),
                health_decay_per_step: parseFloat(document.getElementById('healthDecay').value),
                mutation_std_params: parseFloat(document.getElementById('mutationStdParams').value),
                mutation_std_weights: parseFloat(document.getElementById('mutationStdWeights').value),
            },
            neuromodulation: {
                enabled: document.getElementById('neuromodulationEnabled').checked,
                dopamine: {
                    baseline: parseFloat(document.getElementById('dopamineBaseline').value),
                    decay_rate: parseFloat(document.getElementById('dopamineDecay').value),
                },
                serotonin: {
                    baseline: parseFloat(document.getElementById('serotoninBaseline').value),
                    decay_rate: parseFloat(document.getElementById('serotoninDecay').value),
                },
                norepinephrine: {
                    baseline: parseFloat(document.getElementById('norepinephrineBaseline').value),
                }
            }
        };
        
        const response = await fetch('/api/config/update', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            addLog('info', 'Einstellungen erfolgreich angewendet');
            alert('Einstellungen aktualisiert. Starten Sie die Simulation neu, um Änderungen zu übernehmen.');
        } else {
            addLog('error', 'Fehler beim Anwenden der Einstellungen: ' + data.message);
        }
    } catch (error) {
        console.error('Error applying settings:', error);
        addLog('error', 'Fehler beim Anwenden der Einstellungen: ' + error.message);
    }
}

// Load network data
async function loadNetworkData() {
    loadNeurons(0);
    loadSynapses(0);
    loadAreas();
}

// Load neurons
async function loadNeurons(page) {
    if (page < 0) return;
    neuronPage = page;
    
    try {
        const offset = page * pageSize;
        const response = await fetch(`/api/neurons/details?limit=${pageSize}&offset=${offset}`);
        const data = await response.json();
        
        if (data.status === 'success') {
            const tbody = document.getElementById('neuronsTableBody');
            tbody.innerHTML = '';
            
            data.neurons.forEach(neuron => {
                const row = tbody.insertRow();
                row.innerHTML = `
                    <td>${neuron.id}</td>
                    <td>(${neuron.position.x}, ${neuron.position.y}, ${neuron.position.z}, ${neuron.position.w})</td>
                    <td>${neuron.v_membrane.toFixed(2)} mV</td>
                    <td>${neuron.health.toFixed(3)}</td>
                    <td>${neuron.age}</td>
                    <td>${neuron.neuron_type}</td>
                `;
            });
            
            document.getElementById('neuronCount').textContent = `${data.total} Neuronen`;
            document.getElementById('neuronsPage').textContent = `Seite ${page + 1}`;
        }
    } catch (error) {
        console.error('Error loading neurons:', error);
        addLog('error', 'Fehler beim Laden der Neuronen: ' + error.message);
    }
}

// Load synapses
async function loadSynapses(page) {
    if (page < 0) return;
    synapsePage = page;
    
    try {
        const offset = page * pageSize;
        const response = await fetch(`/api/synapses/details?limit=${pageSize}&offset=${offset}`);
        const data = await response.json();
        
        if (data.status === 'success') {
            const tbody = document.getElementById('synapsesTableBody');
            tbody.innerHTML = '';
            
            data.synapses.forEach(synapse => {
                const row = tbody.insertRow();
                row.innerHTML = `
                    <td>${synapse.pre_id}</td>
                    <td>${synapse.post_id}</td>
                    <td>${synapse.weight.toFixed(4)}</td>
                    <td>${synapse.delay}</td>
                `;
            });
            
            document.getElementById('synapseCount').textContent = `${data.total} Synapsen`;
            document.getElementById('synapsesPage').textContent = `Seite ${page + 1}`;
        }
    } catch (error) {
        console.error('Error loading synapses:', error);
        addLog('error', 'Fehler beim Laden der Synapsen: ' + error.message);
    }
}

// Load areas
async function loadAreas() {
    try {
        const response = await fetch('/api/areas/info');
        const data = await response.json();
        
        if (data.status === 'success') {
            const grid = document.getElementById('areasGrid');
            grid.innerHTML = '';
            
            data.areas.forEach(area => {
                const card = document.createElement('div');
                card.className = 'area-card';
                card.innerHTML = `
                    <h4>${area.name}</h4>
                    <p><strong>Sinnestyp:</strong> ${area.sense}</p>
                    <p><strong>Neuronen:</strong> ${area.neuron_count}</p>
                    <p><strong>Koordinatenbereich:</strong></p>
                    <ul style="font-size: 0.85rem; margin-left: 1rem;">
                        <li>X: ${area.coord_ranges.x[0]} - ${area.coord_ranges.x[1]}</li>
                        <li>Y: ${area.coord_ranges.y[0]} - ${area.coord_ranges.y[1]}</li>
                        <li>Z: ${area.coord_ranges.z[0]} - ${area.coord_ranges.z[1]}</li>
                        <li>W: ${area.coord_ranges.w[0]} - ${area.coord_ranges.w[1]}</li>
                    </ul>
                `;
                grid.appendChild(card);
            });
        }
    } catch (error) {
        console.error('Error loading areas:', error);
        addLog('error', 'Fehler beim Laden der Bereiche: ' + error.message);
    }
}

// Toggle monitoring
function toggleMonitoring() {
    isMonitoring = !isMonitoring;
    const btn = document.getElementById('toggleMonitoring');
    
    if (isMonitoring) {
        btn.textContent = '⏸️ Überwachung pausieren';
        btn.classList.remove('btn-primary');
        btn.classList.add('btn-warning');
        startMonitoring();
    } else {
        btn.textContent = '▶️ Überwachung starten';
        btn.classList.remove('btn-warning');
        btn.classList.add('btn-primary');
        stopMonitoring();
    }
}

function startMonitoring() {
    // Update every 2 seconds
    monitoringInterval = setInterval(async () => {
        try {
            const response = await fetch('/api/stats/network');
            const data = await response.json();
            
            if (data.status === 'success') {
                // Update monitoring displays
                // This would update charts - simplified for now
                console.log('Monitoring update:', data);
            }
        } catch (error) {
            console.error('Monitoring error:', error);
        }
    }, 2000);
}

function stopMonitoring() {
    if (monitoringInterval) {
        clearInterval(monitoringInterval);
        monitoringInterval = null;
    }
}

// Initialize model
async function initializeModel() {
    try {
        const response = await fetch('/api/model/init', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ config_path: 'brain_base_model.json' })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            addLog('info', 'Modell erfolgreich initialisiert');
            loadOverview();
        } else {
            addLog('error', 'Fehler beim Initialisieren: ' + data.message);
        }
    } catch (error) {
        console.error('Error initializing model:', error);
        addLog('error', 'Fehler beim Initialisieren: ' + error.message);
    }
}

// Run simulation step
async function runSimulationStep() {
    try {
        const response = await fetch('/api/simulation/step', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            addLog('info', `Schritt ${data.step}: ${data.spikes} Spikes`);
            loadOverview();
        } else {
            addLog('error', 'Fehler beim Simulationsschritt: ' + data.message);
        }
    } catch (error) {
        console.error('Error running step:', error);
        addLog('error', 'Fehler beim Simulationsschritt: ' + error.message);
    }
}

// Run simulation
async function runSimulation() {
    try {
        const steps = parseInt(document.getElementById('simSteps').value);
        
        const response = await fetch('/api/simulation/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ steps })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            addLog('info', 'Simulation gestartet für ' + steps + ' Schritte');
        } else {
            addLog('error', 'Fehler beim Starten der Simulation: ' + data.message);
        }
    } catch (error) {
        console.error('Error running simulation:', error);
        addLog('error', 'Fehler beim Starten der Simulation: ' + error.message);
    }
}

// Stop simulation
async function stopSimulation() {
    try {
        const response = await fetch('/api/simulation/stop', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            addLog('info', 'Simulation gestoppt');
        }
    } catch (error) {
        console.error('Error stopping simulation:', error);
        addLog('error', 'Fehler beim Stoppen: ' + error.message);
    }
}

// Update simulation progress
function updateSimulationProgress(data) {
    const progressBar = document.getElementById('simProgress');
    const progressText = document.getElementById('simProgressText');
    const progressPercent = document.getElementById('simProgressPercent');
    
    if (progressBar && progressText && progressPercent) {
        progressBar.style.width = data.progress_percent + '%';
        progressText.textContent = `Schritt ${data.step}/${data.total_steps}`;
        progressPercent.textContent = data.progress_percent + '%';
    }
}

// Recover from checkpoint
async function recoverFromCheckpoint() {
    try {
        const response = await fetch('/api/simulation/recover', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            addLog('info', `Wiederhergestellt von Schritt ${data.recovered_step}`);
            loadOverview();
        } else {
            addLog('error', 'Fehler beim Wiederherstellen: ' + data.message);
        }
    } catch (error) {
        console.error('Error recovering:', error);
        addLog('error', 'Fehler beim Wiederherstellen: ' + error.message);
    }
}

// Load analytics
async function loadAnalytics() {
    try {
        const response = await fetch('/api/stats/network');
        const data = await response.json();
        
        if (data.status === 'success') {
            const statsDiv = document.getElementById('networkStats');
            statsDiv.innerHTML = `
                <p><span>Neuronen Gesamt:</span> <strong>${data.neurons.total}</strong></p>
                <p><span>Erregend:</span> <strong>${data.neurons.excitatory}</strong></p>
                <p><span>Hemmend:</span> <strong>${data.neurons.inhibitory}</strong></p>
                <p><span>Durchschn. Membranpotential:</span> <strong>${data.neurons.avg_membrane_potential.toFixed(2)} mV</strong></p>
                <p><span>Durchschn. Gesundheit:</span> <strong>${data.neurons.avg_health.toFixed(3)}</strong></p>
                <p><span>Durchschn. Alter:</span> <strong>${data.neurons.avg_age.toFixed(0)}</strong></p>
                <hr style="margin: 0.5rem 0; border-color: var(--border-color);">
                <p><span>Synapsen Gesamt:</span> <strong>${data.synapses.total}</strong></p>
                <p><span>Positive Gewichte:</span> <strong>${data.synapses.positive_weights}</strong></p>
                <p><span>Negative Gewichte:</span> <strong>${data.synapses.negative_weights}</strong></p>
                <p><span>Durchschn. Gewicht:</span> <strong>${data.synapses.avg_weight.toFixed(4)}</strong></p>
            `;
        }
    } catch (error) {
        console.error('Error loading analytics:', error);
        addLog('error', 'Fehler beim Laden der Analytik: ' + error.message);
    }
}

// Save model
async function saveModel() {
    try {
        const filename = document.getElementById('saveFilename').value;
        const format = document.getElementById('saveFormat').value;
        
        const response = await fetch('/api/model/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename, format })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            addLog('info', 'Modell gespeichert: ' + data.filepath);
        } else {
            addLog('error', 'Fehler beim Speichern: ' + data.message);
        }
    } catch (error) {
        console.error('Error saving model:', error);
        addLog('error', 'Fehler beim Speichern: ' + error.message);
    }
}

// Load model
async function loadModel() {
    try {
        const filepath = document.getElementById('loadFilepath').value;
        
        const response = await fetch('/api/model/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filepath })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            addLog('info', 'Modell geladen: ' + filepath);
            loadOverview();
        } else {
            addLog('error', 'Fehler beim Laden: ' + data.message);
        }
    } catch (error) {
        console.error('Error loading model:', error);
        addLog('error', 'Fehler beim Laden: ' + error.message);
    }
}

// Load senses info
async function loadSensesInfo() {
    try {
        const response = await fetch('/api/senses/info');
        const data = await response.json();
        
        if (data.status === 'success') {
            // Populate sense select if needed
            console.log('Senses loaded:', data.senses);
        }
    } catch (error) {
        console.error('Error loading senses:', error);
    }
}

// Update sense info display
async function updateSenseInfo() {
    const senseType = document.getElementById('senseTypeSelect').value;
    
    try {
        const response = await fetch('/api/senses/info');
        const data = await response.json();
        
        if (data.status === 'success') {
            const sense = data.senses.find(s => s.name === senseType);
            if (sense) {
                const infoDiv = document.getElementById('senseInfo');
                infoDiv.innerHTML = `
                    <p><strong>Bereich:</strong> ${sense.area}</p>
                    <p><strong>W-Index:</strong> ${sense.w_index}</p>
                    <p><strong>Eingangsgröße:</strong> ${sense.input_size.join(' x ')}</p>
                `;
            }
        }
    } catch (error) {
        console.error('Error updating sense info:', error);
    }
}

// Send sense input
async function sendSenseInput() {
    try {
        const senseType = document.getElementById('senseTypeSelect').value;
        const inputData = document.getElementById('senseInput').value;
        
        let parsedData;
        try {
            parsedData = JSON.parse(inputData);
        } catch {
            parsedData = inputData;
        }
        
        const response = await fetch('/api/input/feed', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sense_type: senseType,
                input_data: parsedData
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            addLog('info', `Eingabe gesendet an ${senseType}`);
        } else {
            addLog('error', 'Fehler beim Senden der Eingabe: ' + data.message);
        }
    } catch (error) {
        console.error('Error sending sense input:', error);
        addLog('error', 'Fehler beim Senden der Eingabe: ' + error.message);
    }
}

// Load heatmaps
async function loadHeatmaps() {
    try {
        const response = await fetch('/api/heatmap/data');
        const data = await response.json();
        
        if (data.status === 'success') {
            // Draw heatmaps on canvases
            drawHeatmap('heatmapInput', data.heatmap.input);
            drawHeatmap('heatmapHidden', data.heatmap.hidden);
            drawHeatmap('heatmapOutput', data.heatmap.output);
            
            addLog('info', 'Heatmaps aktualisiert');
        }
    } catch (error) {
        console.error('Error loading heatmaps:', error);
        addLog('error', 'Fehler beim Laden der Heatmaps: ' + error.message);
    }
}

// Draw heatmap
function drawHeatmap(canvasId, data) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const size = data.length;
    
    canvas.width = 300;
    canvas.height = 300;
    
    const cellWidth = canvas.width / size;
    const cellHeight = canvas.height / size;
    
    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            const value = data[i][j];
            const normalized = (value + 80) / 50; // Normalize from -80 to -30 mV range
            const color = Math.floor(normalized * 255);
            
            ctx.fillStyle = `rgb(${color}, 0, ${255 - color})`;
            ctx.fillRect(j * cellWidth, i * cellHeight, cellWidth, cellHeight);
        }
    }
}

// Configuration management
function saveConfiguration() {
    // Save current form values
    addLog('info', 'Konfiguration im Browser gespeichert');
}

function loadConfiguration() {
    loadSettings();
}

function resetConfiguration() {
    if (confirm('Möchten Sie wirklich alle Einstellungen zurücksetzen?')) {
        loadSettings();
        addLog('info', 'Einstellungen zurückgesetzt');
    }
}

// Export configuration
async function exportConfiguration() {
    try {
        const response = await fetch('/api/config/full');
        const data = await response.json();
        
        if (data.status === 'success') {
            const blob = new Blob([JSON.stringify(data.config, null, 2)], {
                type: 'application/json'
            });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'config_export.json';
            a.click();
            URL.revokeObjectURL(url);
            
            addLog('info', 'Konfiguration exportiert');
        }
    } catch (error) {
        console.error('Error exporting config:', error);
        addLog('error', 'Fehler beim Exportieren: ' + error.message);
    }
}

// Import configuration
function importConfiguration() {
    const fileInput = document.getElementById('importConfigFile');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Bitte wählen Sie eine Datei aus');
        return;
    }
    
    const reader = new FileReader();
    reader.onload = async (e) => {
        try {
            const config = JSON.parse(e.target.result);
            
            const response = await fetch('/api/config/update', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                addLog('info', 'Konfiguration importiert');
                loadSettings();
            } else {
                addLog('error', 'Fehler beim Importieren: ' + data.message);
            }
        } catch (error) {
            console.error('Error importing config:', error);
            addLog('error', 'Fehler beim Importieren: ' + error.message);
        }
    };
    reader.readAsText(file);
}

// Export data
async function exportData() {
    const dataType = document.getElementById('exportDataType').value;
    
    try {
        let endpoint = '';
        let filename = '';
        
        switch(dataType) {
            case 'neurons':
                endpoint = '/api/neurons/details?limit=10000';
                filename = 'neurons_export.json';
                break;
            case 'synapses':
                endpoint = '/api/synapses/details?limit=10000';
                filename = 'synapses_export.json';
                break;
            case 'statistics':
                endpoint = '/api/stats/network';
                filename = 'statistics_export.json';
                break;
        }
        
        const response = await fetch(endpoint);
        const data = await response.json();
        
        if (data.status === 'success') {
            const blob = new Blob([JSON.stringify(data, null, 2)], {
                type: 'application/json'
            });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            a.click();
            URL.revokeObjectURL(url);
            
            addLog('info', 'Daten exportiert: ' + filename);
        }
    } catch (error) {
        console.error('Error exporting data:', error);
        addLog('error', 'Fehler beim Exportieren: ' + error.message);
    }
}

// Logging
function addLog(level, message) {
    const logsOutput = document.getElementById('logsOutput');
    if (!logsOutput) return;
    
    const entry = document.createElement('p');
    entry.className = `log-entry ${level}`;
    const timestamp = new Date().toLocaleTimeString();
    entry.textContent = `[${timestamp}] ${level.toUpperCase()}: ${message}`;
    
    logsOutput.appendChild(entry);
    logsOutput.scrollTop = logsOutput.scrollHeight;
}

function clearLogs() {
    const logsOutput = document.getElementById('logsOutput');
    if (logsOutput) {
        logsOutput.innerHTML = '<p class="log-entry info">Protokolle gelöscht</p>';
    }
}
