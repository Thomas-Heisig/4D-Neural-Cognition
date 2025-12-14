// 4D Neural Cognition Frontend JavaScript

// Initialize Socket.IO
const socket = io();

// Wait for DOM and system status to be ready
document.addEventListener('DOMContentLoaded', async () => {
    initializeApp();
});

// DOM Elements
const elements = {
    // Buttons
    initModel: document.getElementById('initModel'),
    getInfo: document.getElementById('getInfo'),
    initNeurons: document.getElementById('initNeurons'),
    initSynapses: document.getElementById('initSynapses'),
    runStep: document.getElementById('runStep'),
    runSimulation: document.getElementById('runSimulation'),
    stopSimulation: document.getElementById('stopSimulation'),
    saveJSON: document.getElementById('saveJSON'),
    saveHDF5: document.getElementById('saveHDF5'),
    loadModel: document.getElementById('loadModel'),
    recoverCheckpoint: document.getElementById('recoverCheckpoint'),
    refreshHeatmap: document.getElementById('refreshHeatmap'),
    feedInput: document.getElementById('feedInput'),
    sendChat: document.getElementById('sendChat'),
    clearLogs: document.getElementById('clearLogs'),
    
    // Inputs
    density: document.getElementById('density'),
    densityValue: document.getElementById('densityValue'),
    nSteps: document.getElementById('nSteps'),
    loadPath: document.getElementById('loadPath'),
    senseType: document.getElementById('senseType'),
    inputData: document.getElementById('inputData'),
    chatInput: document.getElementById('chatInput'),
    
    // Outputs
    modelInfo: document.getElementById('modelInfo'),
    terminalOutput: document.getElementById('terminalOutput'),
    chatMessages: document.getElementById('chatMessages'),
    logOutput: document.getElementById('logOutput'),
    
    // Canvases
    heatmapInput: document.getElementById('heatmapInput'),
    heatmapHidden: document.getElementById('heatmapHidden'),
    heatmapOutput: document.getElementById('heatmapOutput')
};

// State
let isTraining = false;

// Utility Functions
function addTerminalLine(text, className = '') {
    const line = document.createElement('div');
    line.className = className;
    line.textContent = `> ${text}`;
    elements.terminalOutput.appendChild(line);
    elements.terminalOutput.scrollTop = elements.terminalOutput.scrollHeight;
}

function addChatMessage(text, isUser = false) {
    const message = document.createElement('div');
    message.className = `chat-message ${isUser ? 'user' : 'system'}`;
    message.textContent = text;
    elements.chatMessages.appendChild(message);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

function addLogEntry(level, message, timestamp = null) {
    const entry = document.createElement('div');
    entry.className = `log-entry ${level}`;
    
    const time = timestamp || new Date().toLocaleTimeString();
    entry.innerHTML = `<span class="timestamp">[${time}]</span> <strong>${level}:</strong> ${message}`;
    
    elements.logOutput.appendChild(entry);
    elements.logOutput.scrollTop = elements.logOutput.scrollHeight;
}

function showError(message) {
    addTerminalLine(`ERROR: ${message}`, 'error');
    addLogEntry('ERROR', message);
}

function showSuccess(message) {
    addTerminalLine(message, 'success');
    addLogEntry('SUCCESS', message);
}

function showInfo(message) {
    addTerminalLine(message, 'info');
    addLogEntry('INFO', message);
}

// API Functions - Using global apiClient from api-client.js

// Model Functions
async function initializeModel() {
    try {
        showInfo('Initialisiere Modell...');
        const result = await window.apiClient.initModel('brain_base_model.json');
        
        if (result.status === 'success') {
            showSuccess('Modell erfolgreich initialisiert!');
            updateModelInfo(result);
            // Mark system as initialized in status manager
            if (window.systemStatus) {
                window.systemStatus.markInitialized();
            }
        }
    } catch (error) {
        showError(`Fehler bei Initialisierung: ${error.message}`);
    }
}

async function getModelInfo() {
    try {
        const result = await window.apiClient.getModelInfo();
        
        if (result.status === 'success') {
            updateModelInfo(result);
            showSuccess('Modell-Information aktualisiert');
        }
    } catch (error) {
        showError(`Fehler beim Abrufen der Info: ${error.message}`);
    }
}

function updateModelInfo(info) {
    const html = `
        <strong>Gittergröße:</strong> ${info.lattice_shape?.join('×') || 'N/A'}<br>
        <strong>Neuronen:</strong> ${info.num_neurons || 0}<br>
        <strong>Synapsen:</strong> ${info.num_synapses || 0}<br>
        <strong>Schritt:</strong> ${info.current_step || 0}<br>
        <strong>Sinne:</strong> ${info.senses?.join(', ') || 'N/A'}
    `;
    elements.modelInfo.innerHTML = html;
}

async function initializeNeurons() {
    try {
        const density = parseFloat(elements.density.value);
        showInfo(`Initialisiere Neuronen mit Dichte ${density}...`);
        
        const result = await window.apiClient.initNeurons(['V1_like', 'Digital_sensor'], density);
        
        if (result.status === 'success') {
            showSuccess(`${result.num_neurons} Neuronen erstellt!`);
            await getModelInfo();
        }
    } catch (error) {
        showError(`Fehler bei Neuron-Initialisierung: ${error.message}`);
    }
}

async function initializeSynapses() {
    try {
        showInfo('Initialisiere Synapsen...');
        
        const result = await window.apiClient.initSynapses(0.001, 0.1, 0.05);
        
        if (result.status === 'success') {
            showSuccess(`${result.num_synapses} Synapsen erstellt!`);
            await getModelInfo();
        }
    } catch (error) {
        showError(`Fehler bei Synapse-Initialisierung: ${error.message}`);
    }
}

// Simulation Functions
async function runSimulationStep() {
    try {
        const result = await window.apiClient.simulationStep();
        
        if (result.status === 'success') {
            showInfo(`Schritt ${result.step}: ${result.spikes} Spikes, ${result.num_neurons} Neuronen`);
        }
    } catch (error) {
        showError(`Fehler bei Simulationsschritt: ${error.message}`);
    }
}

async function runSimulation() {
    try {
        const steps = parseInt(elements.nSteps.value);
        isTraining = true;
        
        showInfo(`Starte Training für ${steps} Schritte...`);
        elements.runSimulation.disabled = true;
        
        const result = await window.apiClient.runSimulation(steps, 10);
        
        if (result.status === 'success') {
            const res = result.results;
            showSuccess(`Training abgeschlossen! Gesamt Spikes: ${res.total_spikes}`);
            await getModelInfo();
            await refreshHeatmap();
        }
    } catch (error) {
        showError(`Fehler bei Training: ${error.message}`);
    } finally {
        isTraining = false;
        elements.runSimulation.disabled = false;
    }
}

async function stopSimulation() {
    try {
        await window.apiClient.stopSimulation();
        isTraining = false;
        showInfo('Training gestoppt');
    } catch (error) {
        showError(`Fehler beim Stoppen: ${error.message}`);
    }
}

// Input/Output Functions
async function feedInput() {
    try {
        const senseType = elements.senseType.value;
        const inputText = elements.inputData.value;
        
        if (!inputText) {
            showError('Bitte Eingabedaten angeben');
            return;
        }
        
        showInfo(`Sende ${senseType} Eingabe...`);
        
        let inputData;
        if (senseType === 'digital') {
            inputData = inputText;
        } else {
            // For other senses, try to parse as numbers
            try {
                inputData = JSON.parse(inputText);
            } catch {
                // Create random input if parse fails
                inputData = Array(20).fill().map(() => Array(20).fill().map(() => Math.random() * 10));
            }
        }
        
        const result = await window.apiClient.feedInput(senseType, inputData);
        
        if (result.status === 'success') {
            showSuccess(`${senseType} Eingabe erfolgreich gesendet`);
        }
    } catch (error) {
        showError(`Fehler bei Eingabe: ${error.message}`);
    }
}

// Heatmap Functions
async function refreshHeatmap() {
    try {
        const result = await window.apiClient.getHeatmapData(0);
        
        if (result.status === 'success' && result.heatmap) {
            drawHeatmap(elements.heatmapInput, result.heatmap.input);
            drawHeatmap(elements.heatmapHidden, result.heatmap.hidden);
            drawHeatmap(elements.heatmapOutput, result.heatmap.output);
            showSuccess('Heatmap aktualisiert');
        }
    } catch (error) {
        showError(`Fehler bei Heatmap: ${error.message}`);
    }
}

function drawHeatmap(canvas, data) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.fillStyle = '#16161e';
    ctx.fillRect(0, 0, width, height);
    
    if (!data || data.length === 0) return;
    
    const rows = data.length;
    const cols = data[0].length;
    
    // Limit processing for very large heatmaps to prevent browser freeze
    const MAX_HEATMAP_CELLS = 10000; // Maximum cells to render (e.g., 100x100)
    const totalCells = rows * cols;
    
    if (totalCells > MAX_HEATMAP_CELLS) {
        console.warn(`Heatmap zu groß (${rows}x${cols}=${totalCells} Zellen). Neuronendichte reduzieren.`);
        addLogEntry('WARNING', `Heatmap zu groß (${totalCells} Zellen). Visualisierung übersprungen.`);
        ctx.fillStyle = '#ffffff';
        ctx.font = '16px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`Zu viele Neuronen für`, width/2, height/2 - 20);
        ctx.fillText(`Visualisierung (${totalCells})`, width/2, height/2 + 20);
        return;
    }
    
    const cellWidth = width / cols;
    const cellHeight = height / rows;
    
    // Find min and max for normalization
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            if (data[i][j] < min) min = data[i][j];
            if (data[i][j] > max) max = data[i][j];
        }
    }
    
    // Draw heatmap
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            const value = data[i][j];
            const normalized = max > min ? (value - min) / (max - min) : 0;
            
            // Color mapping: blue (low) -> green -> yellow -> red (high)
            const hue = (1 - normalized) * 240; // 240 = blue, 0 = red
            ctx.fillStyle = `hsl(${hue}, 80%, 50%)`;
            ctx.fillRect(j * cellWidth, i * cellHeight, cellWidth, cellHeight);
        }
    }
}

// Storage Functions
async function saveModel(format) {
    try {
        showInfo(`Speichere Modell als ${format.toUpperCase()}...`);
        
        const filename = `brain_state_${Date.now()}`;
        const result = await window.apiClient.saveModel(filename, format);
        
        if (result.status === 'success') {
            showSuccess(`Modell gespeichert: ${result.filepath}`);
        }
    } catch (error) {
        showError(`Fehler beim Speichern: ${error.message}`);
    }
}

async function loadModel() {
    try {
        const filepath = elements.loadPath.value;
        showInfo(`Lade Modell von ${filepath}...`);
        
        const result = await window.apiClient.loadModel(filepath);
        
        if (result.status === 'success') {
            showSuccess(`Modell geladen: ${result.num_neurons} Neuronen, ${result.num_synapses} Synapsen`);
            await getModelInfo();
            await refreshHeatmap();
        }
    } catch (error) {
        showError(`Fehler beim Laden: ${error.message}`);
    }
}

async function recoverFromCheckpoint() {
    try {
        showInfo('Stelle vom letzten Checkpoint wieder her...');
        
        const result = await window.apiClient.recoverCheckpoint();
        
        if (result.status === 'success') {
            showSuccess(`Checkpoint wiederhergestellt: Schritt ${result.recovered_step}, ${result.num_neurons} Neuronen, ${result.num_synapses} Synapsen`);
            await getModelInfo();
            await refreshHeatmap();
        }
    } catch (error) {
        showError(`Fehler bei Wiederherstellung: ${error.message}`);
    }
}

// Chat Functions
function sendChatMessage() {
    const message = elements.chatInput.value.trim();
    if (!message) return;
    
    addChatMessage(message, true);
    socket.emit('chat_message', { message: message });
    elements.chatInput.value = '';
}

// Socket.IO Event Handlers
socket.on('connect', () => {
    addLogEntry('INFO', 'Verbunden mit Server');
});

socket.on('disconnect', () => {
    addLogEntry('WARNING', 'Verbindung zum Server getrennt');
});

socket.on('log_message', (data) => {
    addLogEntry(data.level, data.message, new Date(data.timestamp).toLocaleTimeString());
});

socket.on('chat_response', (data) => {
    addChatMessage(data.message, false);
});

socket.on('training_progress', (data) => {
    // Format remaining time
    let timeStr = '';
    if (data.estimated_remaining_seconds !== undefined) {
        const seconds = data.estimated_remaining_seconds;
        if (seconds < 60) {
            timeStr = `${Math.round(seconds)}s`;
        } else if (seconds < 3600) {
            const mins = Math.floor(seconds / 60);
            const secs = Math.round(seconds % 60);
            timeStr = `${mins}m ${secs}s`;
        } else {
            const hours = Math.floor(seconds / 3600);
            const mins = Math.floor((seconds % 3600) / 60);
            timeStr = `${hours}h ${mins}m`;
        }
    }
    
    // Build progress message
    let message = `Schritt ${data.step}`;
    if (data.total_steps) {
        message += `/${data.total_steps}`;
    }
    if (data.progress_percent !== undefined) {
        message += ` (${data.progress_percent}%)`;
    }
    if (timeStr) {
        message += ` - ${timeStr} verbleibend`;
    }
    message += `: ${data.spikes} Spikes, ${data.neurons} Neuronen`;
    
    showInfo(message);
});

// Initialize App - Called after DOM is ready
function initializeApp() {
    // Setup event listeners
    elements.initModel.addEventListener('click', initializeModel);
    elements.getInfo.addEventListener('click', getModelInfo);
    elements.initNeurons.addEventListener('click', initializeNeurons);
    elements.initSynapses.addEventListener('click', initializeSynapses);
    elements.runStep.addEventListener('click', runSimulationStep);
    elements.runSimulation.addEventListener('click', runSimulation);
    elements.stopSimulation.addEventListener('click', stopSimulation);
    elements.saveJSON.addEventListener('click', () => saveModel('json'));
    elements.saveHDF5.addEventListener('click', () => saveModel('hdf5'));
    elements.loadModel.addEventListener('click', loadModel);
    elements.recoverCheckpoint.addEventListener('click', recoverFromCheckpoint);
    elements.refreshHeatmap.addEventListener('click', refreshHeatmap);
    elements.feedInput.addEventListener('click', feedInput);
    elements.sendChat.addEventListener('click', sendChatMessage);
    elements.clearLogs.addEventListener('click', () => {
        elements.logOutput.innerHTML = '';
    });

    // Density slider update
    elements.density.addEventListener('input', (e) => {
        elements.densityValue.textContent = e.target.value;
    });

    // Chat input enter key
    elements.chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendChatMessage();
        }
    });

    // Initialize canvases
    elements.heatmapInput.width = 300;
    elements.heatmapInput.height = 300;
    elements.heatmapHidden.width = 300;
    elements.heatmapHidden.height = 300;
    elements.heatmapOutput.width = 300;
    elements.heatmapOutput.height = 300;

    // Initial messages
    addLogEntry('INFO', 'Frontend Interface gestartet');
    addTerminalLine('4D Neural Cognition Terminal bereit');
    addChatMessage('Willkommen! Geben Sie "help" ein für verfügbare Befehle.', false);
    
    // Note: We don't automatically check for system initialization here
    // The user needs to initialize the model explicitly
    showInfo('Bitte initialisieren Sie das Modell, um zu beginnen.');
}
