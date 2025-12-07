// 3D/4D Visualization Module for Neural Network
// Uses Three.js for 3D rendering and visualization

class NeuralVisualization {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.neurons = [];
        this.connections = [];
        this.animationId = null;
        this.timeStep = 0;
        this.fourthDimension = 0; // w coordinate for 4D projection
        
        this.init();
    }
    
    init() {
        // Create scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x16161e);
        
        // Create camera
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
        this.camera.position.set(30, 30, 30);
        this.camera.lookAt(0, 0, 0);
        
        // Create renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.container.appendChild(this.renderer.domElement);
        
        // Add orbit controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        
        // Add lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 10, 10);
        this.scene.add(directionalLight);
        
        // Add grid helper
        const gridHelper = new THREE.GridHelper(40, 20, 0x4a90e2, 0x3a3a4e);
        this.scene.add(gridHelper);
        
        // Add axes helper
        const axesHelper = new THREE.AxesHelper(25);
        this.scene.add(axesHelper);
        
        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize(), false);
        
        // Start animation loop
        this.animate();
    }
    
    onWindowResize() {
        if (!this.container) return;
        
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.aspect = aspect;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }
    
    // Project 4D coordinates to 3D using stereographic projection
    project4Dto3D(x, y, z, w) {
        const wOffset = w - this.fourthDimension;
        const scale = 1 / (1 + wOffset * 0.1); // Stereographic projection
        
        return {
            x: x * scale,
            y: y * scale,
            z: z * scale
        };
    }
    
    // Load and visualize neurons
    async loadNeurons(neuronData) {
        // Clear existing neurons
        this.clearNeurons();
        
        // Create neuron spheres
        neuronData.forEach(neuron => {
            const pos = this.project4Dto3D(neuron.x, neuron.y, neuron.z, neuron.w || 0);
            
            // Color based on neuron properties
            let color = new THREE.Color();
            if (neuron.v_membrane > 0) {
                color.setHSL(0.6, 1.0, 0.5); // Blue for resting
            } else {
                color.setHSL(0.0, 1.0, 0.5); // Red for active
            }
            
            // Size based on health
            const size = 0.3 + (neuron.health * 0.2);
            
            const geometry = new THREE.SphereGeometry(size, 16, 16);
            const material = new THREE.MeshPhongMaterial({
                color: color,
                emissive: color,
                emissiveIntensity: 0.3,
                transparent: true,
                opacity: 0.8
            });
            
            const sphere = new THREE.Mesh(geometry, material);
            sphere.position.set(pos.x, pos.y, pos.z);
            sphere.userData = neuron;
            
            this.scene.add(sphere);
            this.neurons.push(sphere);
        });
    }
    
    // Load and visualize connections
    async loadConnections(connectionData) {
        // Clear existing connections
        this.clearConnections();
        
        connectionData.forEach(conn => {
            const startPos = this.project4Dto3D(conn.from.x, conn.from.y, conn.from.z, conn.from.w || 0);
            const endPos = this.project4Dto3D(conn.to.x, conn.to.y, conn.to.z, conn.to.w || 0);
            
            const points = [
                new THREE.Vector3(startPos.x, startPos.y, startPos.z),
                new THREE.Vector3(endPos.x, endPos.y, endPos.z)
            ];
            
            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            
            // Color based on weight
            const color = conn.weight > 0 ? 0x5cb85c : 0xd9534f;
            const material = new THREE.LineBasicMaterial({
                color: color,
                transparent: true,
                opacity: Math.min(Math.abs(conn.weight), 0.5)
            });
            
            const line = new THREE.Line(geometry, material);
            this.scene.add(line);
            this.connections.push(line);
        });
    }
    
    // Clear all neurons from scene
    clearNeurons() {
        this.neurons.forEach(neuron => {
            this.scene.remove(neuron);
            if (neuron.geometry) neuron.geometry.dispose();
            if (neuron.material) neuron.material.dispose();
        });
        this.neurons = [];
    }
    
    // Clear all connections from scene
    clearConnections() {
        this.connections.forEach(conn => {
            this.scene.remove(conn);
            if (conn.geometry) conn.geometry.dispose();
            if (conn.material) conn.material.dispose();
        });
        this.connections = [];
    }
    
    // Update visualization for time step
    updateTimeStep(step) {
        this.timeStep = step;
        
        // Update neuron colors based on activity
        this.neurons.forEach(neuron => {
            const activity = neuron.userData.activity || 0;
            const health = neuron.userData.health || 1.0;
            
            // Pulse effect for active neurons
            const intensity = 0.3 + (activity * 0.7);
            neuron.material.emissiveIntensity = intensity;
            
            // Update opacity based on health
            neuron.material.opacity = 0.5 + (health * 0.5);
        });
    }
    
    // Set 4D projection parameter (w dimension)
    set4DProjection(wValue) {
        this.fourthDimension = wValue;
        
        // Reproject all neurons
        this.neurons.forEach(neuron => {
            const userData = neuron.userData;
            const pos = this.project4Dto3D(userData.x, userData.y, userData.z, userData.w || 0);
            neuron.position.set(pos.x, pos.y, pos.z);
        });
        
        // Reproject all connections
        this.connections.forEach(conn => {
            // Would need to store connection data to reproject
        });
    }
    
    // Animation loop
    animate() {
        this.animationId = requestAnimationFrame(() => this.animate());
        
        // Update controls
        if (this.controls) {
            this.controls.update();
        }
        
        // Render scene
        if (this.renderer && this.scene && this.camera) {
            this.renderer.render(this.scene, this.camera);
        }
    }
    
    // Stop animation
    dispose() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        
        this.clearNeurons();
        this.clearConnections();
        
        if (this.renderer) {
            this.renderer.dispose();
            if (this.container && this.renderer.domElement) {
                this.container.removeChild(this.renderer.domElement);
            }
        }
    }
}

// Export for use in main app
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NeuralVisualization;
}
