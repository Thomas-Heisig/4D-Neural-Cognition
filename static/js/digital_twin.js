/**
 * Digital Twin Visualization and Control
 * Handles embodiment, self-awareness, and motor learning visualizations
 */

class DigitalTwinManager {
    constructor() {
        this.bodyCanvas = document.getElementById('bodyCanvas');
        this.bodyCtx = this.bodyCanvas?.getContext('2d');
        this.bodyState = {
            position: [0, 0, 0],
            joints: {},
            muscles: {},
            velocity: [0, 0, 0],
        };
        this.viewAngle = 'front';
        this.isActive = false;
        
        this.initializeEventListeners();
        this.startUpdateLoop();
    }
    
    initializeEventListeners() {
        // Toggle digital twin
        const toggleBtn = document.getElementById('toggleDigitalTwin');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => this.toggleDigitalTwin());
        }
        
        // Body controls
        const resetBtn = document.getElementById('resetBodyPose');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => this.resetBodyPose());
        }
        
        const viewSelect = document.getElementById('bodyViewAngle');
        if (viewSelect) {
            viewSelect.addEventListener('change', (e) => {
                this.viewAngle = e.target.value;
                this.drawBody();
            });
        }
        
        // Motor command sliders
        const sliders = [
            'rightArmSlider', 'leftArmSlider', 
            'rightLegSlider', 'leftLegSlider'
        ];
        sliders.forEach(sliderId => {
            const slider = document.getElementById(sliderId);
            if (slider) {
                slider.addEventListener('input', (e) => {
                    const valueSpan = document.getElementById(sliderId.replace('Slider', 'Value'));
                    if (valueSpan) {
                        valueSpan.textContent = parseFloat(e.target.value).toFixed(2);
                    }
                });
            }
        });
        
        // Apply motor commands
        const applyBtn = document.getElementById('applyMotorCommands');
        if (applyBtn) {
            applyBtn.addEventListener('click', () => this.applyMotorCommands());
        }
    }
    
    toggleDigitalTwin() {
        this.isActive = !this.isActive;
        const btn = document.getElementById('toggleDigitalTwin');
        if (btn) {
            btn.textContent = this.isActive ? '⏸️ Twin pausieren' : '▶️ Twin aktivieren';
            btn.className = this.isActive ? 'btn-warning' : 'btn-primary';
        }
        
        if (this.isActive) {
            this.startUpdates();
        }
    }
    
    resetBodyPose() {
        // Reset all sliders to 0
        const sliders = [
            'rightArmSlider', 'leftArmSlider', 
            'rightLegSlider', 'leftLegSlider'
        ];
        sliders.forEach(sliderId => {
            const slider = document.getElementById(sliderId);
            if (slider) {
                slider.value = 0;
                const valueSpan = document.getElementById(sliderId.replace('Slider', 'Value'));
                if (valueSpan) {
                    valueSpan.textContent = '0.00';
                }
            }
        });
        
        // Reset body state
        this.bodyState.position = [0, 0, 0];
        this.bodyState.velocity = [0, 0, 0];
        this.drawBody();
    }
    
    applyMotorCommands() {
        // Get slider values
        const rightArm = parseFloat(document.getElementById('rightArmSlider')?.value || 0);
        const leftArm = parseFloat(document.getElementById('leftArmSlider')?.value || 0);
        const rightLeg = parseFloat(document.getElementById('rightLegSlider')?.value || 0);
        const leftLeg = parseFloat(document.getElementById('leftLegSlider')?.value || 0);
        
        // Update body state
        this.bodyState.joints = {
            right_shoulder: rightArm,
            left_shoulder: leftArm,
            right_hip: rightLeg,
            left_hip: leftLeg,
        };
        
        // Redraw body
        this.drawBody();
        
        // TODO: Send to backend if simulation is running
        console.log('Motor commands applied:', this.bodyState.joints);
    }
    
    drawBody() {
        if (!this.bodyCtx) return;
        
        const ctx = this.bodyCtx;
        const canvas = this.bodyCanvas;
        
        // Clear canvas
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw grid
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        for (let x = 0; x < canvas.width; x += 40) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, canvas.height);
            ctx.stroke();
        }
        for (let y = 0; y < canvas.height; y += 40) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(canvas.width, y);
            ctx.stroke();
        }
        
        // Draw simple stick figure based on view angle
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        
        ctx.strokeStyle = '#4a90e2';
        ctx.lineWidth = 4;
        ctx.lineCap = 'round';
        
        if (this.viewAngle === 'front') {
            this.drawFrontView(ctx, centerX, centerY);
        } else if (this.viewAngle === 'side') {
            this.drawSideView(ctx, centerX, centerY);
        } else if (this.viewAngle === 'top') {
            this.drawTopView(ctx, centerX, centerY);
        }
    }
    
    drawFrontView(ctx, cx, cy) {
        const joints = this.bodyState.joints;
        
        // Head
        ctx.beginPath();
        ctx.arc(cx, cy - 150, 30, 0, Math.PI * 2);
        ctx.stroke();
        
        // Body
        ctx.beginPath();
        ctx.moveTo(cx, cy - 120);
        ctx.lineTo(cx, cy + 50);
        ctx.stroke();
        
        // Arms
        const rightArmAngle = joints.right_shoulder || 0;
        const leftArmAngle = joints.left_shoulder || 0;
        
        // Right arm
        ctx.beginPath();
        ctx.moveTo(cx, cy - 100);
        const rightArmX = cx + 80 * Math.cos(rightArmAngle + Math.PI / 2);
        const rightArmY = cy - 100 + 80 * Math.sin(rightArmAngle + Math.PI / 2);
        ctx.lineTo(rightArmX, rightArmY);
        ctx.stroke();
        
        // Left arm
        ctx.beginPath();
        ctx.moveTo(cx, cy - 100);
        const leftArmX = cx - 80 * Math.cos(leftArmAngle + Math.PI / 2);
        const leftArmY = cy - 100 + 80 * Math.sin(leftArmAngle + Math.PI / 2);
        ctx.lineTo(leftArmX, leftArmY);
        ctx.stroke();
        
        // Legs
        const rightLegAngle = joints.right_hip || 0;
        const leftLegAngle = joints.left_hip || 0;
        
        // Right leg
        ctx.beginPath();
        ctx.moveTo(cx, cy + 50);
        const rightLegX = cx + 60 * Math.sin(rightLegAngle);
        const rightLegY = cy + 50 + 100 * Math.cos(rightLegAngle);
        ctx.lineTo(rightLegX, rightLegY);
        ctx.stroke();
        
        // Left leg
        ctx.beginPath();
        ctx.moveTo(cx, cy + 50);
        const leftLegX = cx - 60 * Math.sin(leftLegAngle);
        const leftLegY = cy + 50 + 100 * Math.cos(leftLegAngle);
        ctx.lineTo(leftLegX, leftLegY);
        ctx.stroke();
        
        // Joints (circles)
        ctx.fillStyle = '#e74c3c';
        const jointPositions = [
            [cx, cy - 100], // shoulders
            [rightArmX, rightArmY], // right hand
            [leftArmX, leftArmY], // left hand
            [cx, cy + 50], // hips
            [rightLegX, rightLegY], // right foot
            [leftLegX, leftLegY], // left foot
        ];
        jointPositions.forEach(([x, y]) => {
            ctx.beginPath();
            ctx.arc(x, y, 6, 0, Math.PI * 2);
            ctx.fill();
        });
    }
    
    drawSideView(ctx, cx, cy) {
        // Simplified side view
        ctx.beginPath();
        ctx.arc(cx, cy - 150, 30, 0, Math.PI * 2);
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(cx, cy - 120);
        ctx.lineTo(cx, cy + 50);
        ctx.stroke();
        
        // Single arm visible
        ctx.beginPath();
        ctx.moveTo(cx, cy - 100);
        ctx.lineTo(cx + 70, cy - 80);
        ctx.stroke();
        
        // Single leg visible
        ctx.beginPath();
        ctx.moveTo(cx, cy + 50);
        ctx.lineTo(cx + 40, cy + 150);
        ctx.stroke();
    }
    
    drawTopView(ctx, cx, cy) {
        // Simplified top view (head and shoulders)
        ctx.beginPath();
        ctx.arc(cx, cy, 30, 0, Math.PI * 2);
        ctx.stroke();
        
        // Shoulders
        ctx.beginPath();
        ctx.moveTo(cx - 60, cy + 40);
        ctx.lineTo(cx + 60, cy + 40);
        ctx.stroke();
        
        // Arms
        ctx.beginPath();
        ctx.moveTo(cx - 60, cy + 40);
        ctx.lineTo(cx - 60, cy + 100);
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(cx + 60, cy + 40);
        ctx.lineTo(cx + 60, cy + 100);
        ctx.stroke();
    }
    
    updateProprioception(data) {
        // Update proprioception display
        if (data.position) {
            const posText = `X: ${data.position[0].toFixed(2)}, Y: ${data.position[1].toFixed(2)}, Z: ${data.position[2].toFixed(2)}`;
            const posElem = document.getElementById('bodyPosition');
            if (posElem) posElem.textContent = posText;
        }
        
        if (data.velocity) {
            const velMag = Math.sqrt(data.velocity.reduce((sum, v) => sum + v*v, 0));
            const velElem = document.getElementById('bodyVelocity');
            if (velElem) velElem.textContent = `${velMag.toFixed(3)} m/s`;
        }
        
        // Update joint angles
        if (data.joint_angles) {
            const container = document.getElementById('jointAnglesDisplay');
            if (container) {
                container.innerHTML = '';
                for (const [joint, angle] of Object.entries(data.joint_angles)) {
                    const div = document.createElement('div');
                    div.className = 'proprio-item';
                    div.innerHTML = `<strong>${joint}:</strong><span>${angle.toFixed(3)} rad</span>`;
                    container.appendChild(div);
                }
            }
        }
    }
    
    updateMultimodalConfidence(data) {
        // Update confidence bars
        const modalities = ['visual', 'audio', 'proprio', 'overallSelf'];
        modalities.forEach(mod => {
            const confidence = data[mod + '_confidence'] || 0;
            const fillElem = document.getElementById(mod + 'Confidence');
            const textElem = document.getElementById(mod + 'ConfidenceText');
            
            if (fillElem) {
                fillElem.style.width = `${confidence * 100}%`;
            }
            if (textElem) {
                textElem.textContent = `${(confidence * 100).toFixed(1)}%`;
            }
        });
    }
    
    startUpdateLoop() {
        setInterval(() => {
            if (this.isActive) {
                this.drawBody();
                this.fetchBodyState();
            }
        }, 100); // 10 Hz update rate
    }
    
    fetchBodyState() {
        // TODO: Fetch from backend API
        // For now, simulate with random data
        if (Math.random() > 0.95) {
            this.updateProprioception({
                position: [
                    Math.random() * 2 - 1,
                    Math.random() * 2 - 1,
                    Math.random() * 0.5
                ],
                velocity: [
                    Math.random() * 0.5 - 0.25,
                    Math.random() * 0.5 - 0.25,
                    Math.random() * 0.2 - 0.1
                ],
            });
            
            this.updateMultimodalConfidence({
                visual_confidence: Math.random() * 0.3 + 0.5,
                audio_confidence: Math.random() * 0.4 + 0.3,
                proprio_confidence: Math.random() * 0.2 + 0.7,
                overallSelf_confidence: Math.random() * 0.2 + 0.6,
            });
        }
    }
    
    startUpdates() {
        // Start polling for updates
        console.log('Digital twin updates started');
    }
}

// Self-Awareness Manager
class SelfAwarenessManager {
    constructor() {
        this.anomalies = [];
        this.initializeCharts();
        this.startMonitoring();
    }
    
    initializeCharts() {
        // Self-consistency chart
        const ctx = document.getElementById('selfConsistencyChart')?.getContext('2d');
        if (ctx && typeof Chart !== 'undefined') {
            this.consistencyChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Temporal Consistency',
                        data: [],
                        borderColor: '#4a90e2',
                        tension: 0.4
                    }, {
                        label: 'Cross-Modal Integration',
                        data: [],
                        borderColor: '#28a745',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { beginAtZero: true, max: 1.0 }
                    }
                }
            });
        }
        
        // Perception stream history
        const ctx2 = document.getElementById('perceptionStreamChart')?.getContext('2d');
        if (ctx2 && typeof Chart !== 'undefined') {
            this.perceptionChart = new Chart(ctx2, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Prediction Error',
                        data: [],
                        borderColor: '#dc3545',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false
                }
            });
        }
    }
    
    addAnomaly(anomaly) {
        this.anomalies.push(anomaly);
        this.updateAnomalyDisplay();
    }
    
    updateAnomalyDisplay() {
        const container = document.getElementById('anomalyList');
        if (!container) return;
        
        if (this.anomalies.length === 0) {
            container.innerHTML = '<div class="no-anomalies">✅ Keine Anomalien erkannt</div>';
            return;
        }
        
        container.innerHTML = '';
        this.anomalies.slice(-10).forEach(anomaly => {
            const div = document.createElement('div');
            div.className = `anomaly-item severity-${anomaly.severity}`;
            div.innerHTML = `
                <div><strong>${anomaly.type}</strong></div>
                <div>${anomaly.implication}</div>
                <div style="font-size: 0.85em; color: #999;">Wert: ${anomaly.value.toFixed(3)}</div>
            `;
            container.appendChild(div);
        });
    }
    
    updateMetrics(metrics) {
        // Update consistency metrics
        const fields = ['temporalConsistency', 'crossModalIntegration', 'agencyScoreMetric'];
        fields.forEach(field => {
            const elem = document.getElementById(field);
            if (elem && metrics[field] !== undefined) {
                elem.textContent = metrics[field].toFixed(2);
            }
        });
        
        // Update chart
        if (this.consistencyChart) {
            this.consistencyChart.data.labels.push(new Date().toLocaleTimeString());
            this.consistencyChart.data.datasets[0].data.push(metrics.temporalConsistency || 0);
            this.consistencyChart.data.datasets[1].data.push(metrics.crossModalIntegration || 0);
            
            if (this.consistencyChart.data.labels.length > 50) {
                this.consistencyChart.data.labels.shift();
                this.consistencyChart.data.datasets.forEach(ds => ds.data.shift());
            }
            
            this.consistencyChart.update('none');
        }
    }
    
    startMonitoring() {
        setInterval(() => {
            // Simulate anomaly detection
            if (Math.random() > 0.97) {
                this.addAnomaly({
                    type: 'motor_prediction_error',
                    severity: 'medium',
                    value: Math.random() * 0.5 + 0.2,
                    implication: 'Body model inaccurate or external force'
                });
            }
            
            // Update metrics
            this.updateMetrics({
                temporalConsistency: Math.random() * 0.3 + 0.6,
                crossModalIntegration: Math.random() * 0.2 + 0.7,
                agencyScoreMetric: Math.random() * 0.3 + 0.5,
            });
        }, 2000);
    }
}

// Motor Learning Manager
class MotorLearningManager {
    constructor() {
        this.episodes = 0;
        this.learningActive = false;
        this.initializeCharts();
        this.initializeEventListeners();
    }
    
    initializeCharts() {
        // Learning progress chart
        const ctx = document.getElementById('learningProgressChart')?.getContext('2d');
        if (ctx && typeof Chart !== 'undefined') {
            this.learningChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Reward',
                        data: [],
                        borderColor: '#28a745',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false
                }
            });
        }
        
        // STDP visualization
        this.initializeSTDPVisualization();
        
        // VNC priority heatmap
        this.initializeVNCHeatmap();
    }
    
    initializeSTDPVisualization() {
        const canvas = document.getElementById('stdpVisualization');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#13151f';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw STDP curve
        ctx.strokeStyle = '#4a90e2';
        ctx.lineWidth = 3;
        ctx.beginPath();
        
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        
        for (let x = 0; x < canvas.width; x++) {
            const t = (x - centerX) / 30; // Time difference
            let y;
            if (t > 0) {
                y = Math.exp(-t / 20) * 50; // LTP
            } else {
                y = -Math.exp(t / 20) * 50; // LTD
            }
            
            const plotY = centerY - y;
            
            if (x === 0) {
                ctx.moveTo(x, plotY);
            } else {
                ctx.lineTo(x, plotY);
            }
        }
        ctx.stroke();
        
        // Draw axes
        ctx.strokeStyle = '#666';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, centerY);
        ctx.lineTo(canvas.width, centerY);
        ctx.moveTo(centerX, 0);
        ctx.lineTo(centerX, canvas.height);
        ctx.stroke();
    }
    
    initializeVNCHeatmap() {
        const canvas = document.getElementById('vncPriorityHeatmap');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        this.drawVNCHeatmap(ctx);
    }
    
    drawVNCHeatmap(ctx) {
        const canvas = ctx.canvas;
        
        // Clear
        ctx.fillStyle = '#13151f';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw brain slices
        const slices = [
            { w: 6, name: 'Sensory', color: '#3498db', vpus: 1 },
            { w: 10, name: 'Motor', color: '#e74c3c', vpus: 2 },
            { w: 12, name: 'Self-Perception', color: '#9b59b6', vpus: 2 },
            { w: 14, name: 'Executive', color: '#f39c12', vpus: 1 },
        ];
        
        const sliceWidth = canvas.width / slices.length;
        
        slices.forEach((slice, i) => {
            const x = i * sliceWidth;
            const height = (slice.vpus / 2) * (canvas.height * 0.8);
            const y = canvas.height - height - 40;
            
            // Draw bar
            ctx.fillStyle = slice.color;
            ctx.fillRect(x + 20, y, sliceWidth - 40, height);
            
            // Draw label
            ctx.fillStyle = '#fff';
            ctx.font = '12px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(slice.name, x + sliceWidth / 2, canvas.height - 20);
            ctx.fillText(`${slice.vpus} VPUs`, x + sliceWidth / 2, canvas.height - 5);
        });
    }
    
    initializeEventListeners() {
        const startBtn = document.getElementById('startMotorLearning');
        if (startBtn) {
            startBtn.addEventListener('click', () => this.toggleLearning());
        }
    }
    
    toggleLearning() {
        this.learningActive = !this.learningActive;
        const btn = document.getElementById('startMotorLearning');
        if (btn) {
            btn.textContent = this.learningActive ? '⏸️ Lernen pausieren' : '▶️ Lernprozess starten';
            btn.className = this.learningActive ? 'btn-warning' : 'btn-success';
        }
        
        if (this.learningActive) {
            this.startLearningLoop();
        }
    }
    
    startLearningLoop() {
        const interval = setInterval(() => {
            if (!this.learningActive) {
                clearInterval(interval);
                return;
            }
            
            // Simulate learning step
            this.episodes++;
            const reward = Math.random() + this.episodes * 0.01;
            
            // Update display
            document.getElementById('learningEpisodes').textContent = this.episodes;
            document.getElementById('avgReward').textContent = reward.toFixed(2);
            
            // Update chart
            if (this.learningChart) {
                this.learningChart.data.labels.push(this.episodes);
                this.learningChart.data.datasets[0].data.push(reward);
                
                if (this.learningChart.data.labels.length > 50) {
                    this.learningChart.data.labels.shift();
                    this.learningChart.data.datasets[0].data.shift();
                }
                
                this.learningChart.update('none');
            }
            
            // Update rewards
            const externalReward = Math.random() * 0.5;
            const intrinsicReward = Math.random() * 0.3;
            const totalReward = externalReward + intrinsicReward;
            
            document.getElementById('externalRewardValue').textContent = externalReward.toFixed(2);
            document.getElementById('intrinsicRewardValue').textContent = intrinsicReward.toFixed(2);
            document.getElementById('totalRewardValue').textContent = totalReward.toFixed(2);
            
            document.getElementById('externalRewardGauge').style.width = `${externalReward * 100}%`;
            document.getElementById('intrinsicRewardGauge').style.width = `${intrinsicReward * 100}%`;
            document.getElementById('totalRewardGauge').style.width = `${totalReward * 50}%`; // Scale for display
        }, 1000);
    }
}

// Initialize all managers when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.digitalTwinManager = new DigitalTwinManager();
    window.selfAwarenessManager = new SelfAwarenessManager();
    window.motorLearningManager = new MotorLearningManager();
});
