// Collaboration Module
// Handles multi-user support, shared simulations, comments/annotations, and version control

class CollaborationManager {
    constructor(socketConnection) {
        this.socket = socketConnection;
        this.users = [];
        this.currentUser = null;
        this.sharedSimulations = [];
        this.annotations = [];
        this.versions = [];
        this.setupSocketHandlers();
    }
    
    // Setup socket event handlers for real-time collaboration
    setupSocketHandlers() {
        if (!this.socket) return;
        
        // User joined
        this.socket.on('user_joined', (data) => {
            this.users.push(data.user);
            this.notifyUserJoined(data.user);
        });
        
        // User left
        this.socket.on('user_left', (data) => {
            this.users = this.users.filter(u => u.id !== data.userId);
            this.notifyUserLeft(data.userId);
        });
        
        // Simulation state update
        this.socket.on('simulation_update', (data) => {
            this.handleSimulationUpdate(data);
        });
        
        // Annotation added
        this.socket.on('annotation_added', (data) => {
            this.annotations.push(data.annotation);
            this.notifyAnnotationAdded(data.annotation);
        });
        
        // Version created
        this.socket.on('version_created', (data) => {
            this.versions.push(data.version);
            this.notifyVersionCreated(data.version);
        });
    }
    
    // Register current user
    registerUser(username, userId = null) {
        this.currentUser = {
            id: userId || this.generateUserId(),
            username: username,
            joinedAt: new Date().toISOString(),
            color: this.generateUserColor()
        };
        
        if (this.socket) {
            this.socket.emit('register_user', this.currentUser);
        }
        
        return this.currentUser;
    }
    
    // Get list of active users
    getActiveUsers() {
        return this.users;
    }
    
    // Create shared simulation
    createSharedSimulation(name, description, config) {
        const simulation = {
            id: this.generateId(),
            name: name,
            description: description,
            config: config,
            createdBy: this.currentUser.id,
            createdAt: new Date().toISOString(),
            participants: [this.currentUser.id],
            state: 'active',
            currentStep: 0
        };
        
        this.sharedSimulations.push(simulation);
        
        if (this.socket) {
            this.socket.emit('create_shared_simulation', simulation);
        }
        
        return simulation;
    }
    
    // Join shared simulation
    joinSharedSimulation(simulationId) {
        const simulation = this.sharedSimulations.find(s => s.id === simulationId);
        
        if (!simulation) {
            throw new Error('Shared simulation not found');
        }
        
        if (!simulation.participants.includes(this.currentUser.id)) {
            simulation.participants.push(this.currentUser.id);
            
            if (this.socket) {
                this.socket.emit('join_simulation', {
                    simulationId: simulationId,
                    userId: this.currentUser.id
                });
            }
        }
        
        return simulation;
    }
    
    // Leave shared simulation
    leaveSharedSimulation(simulationId) {
        const simulation = this.sharedSimulations.find(s => s.id === simulationId);
        
        if (simulation) {
            simulation.participants = simulation.participants.filter(
                id => id !== this.currentUser.id
            );
            
            if (this.socket) {
                this.socket.emit('leave_simulation', {
                    simulationId: simulationId,
                    userId: this.currentUser.id
                });
            }
        }
    }
    
    // Update shared simulation state
    updateSharedSimulation(simulationId, updates) {
        const simulation = this.sharedSimulations.find(s => s.id === simulationId);
        
        if (!simulation) {
            throw new Error('Shared simulation not found');
        }
        
        Object.assign(simulation, updates);
        simulation.lastUpdatedBy = this.currentUser.id;
        simulation.lastUpdatedAt = new Date().toISOString();
        
        if (this.socket) {
            this.socket.emit('update_simulation', {
                simulationId: simulationId,
                updates: updates,
                userId: this.currentUser.id
            });
        }
        
        return simulation;
    }
    
    // Handle simulation update from other users
    handleSimulationUpdate(data) {
        const simulation = this.sharedSimulations.find(s => s.id === data.simulationId);
        
        if (simulation) {
            Object.assign(simulation, data.updates);
            this.notifySimulationUpdated(simulation);
        }
    }
    
    // Add annotation/comment
    addAnnotation(targetType, targetId, text, position = null) {
        const annotation = {
            id: this.generateId(),
            targetType: targetType, // 'neuron', 'synapse', 'simulation', 'experiment'
            targetId: targetId,
            text: text,
            position: position, // {x, y, z} for spatial annotations
            author: this.currentUser.id,
            authorName: this.currentUser.username,
            createdAt: new Date().toISOString(),
            replies: []
        };
        
        this.annotations.push(annotation);
        
        if (this.socket) {
            this.socket.emit('add_annotation', annotation);
        }
        
        return annotation;
    }
    
    // Reply to annotation
    replyToAnnotation(annotationId, text) {
        const annotation = this.annotations.find(a => a.id === annotationId);
        
        if (!annotation) {
            throw new Error('Annotation not found');
        }
        
        const reply = {
            id: this.generateId(),
            text: text,
            author: this.currentUser.id,
            authorName: this.currentUser.username,
            createdAt: new Date().toISOString()
        };
        
        annotation.replies.push(reply);
        
        if (this.socket) {
            this.socket.emit('reply_annotation', {
                annotationId: annotationId,
                reply: reply
            });
        }
        
        return reply;
    }
    
    // Get annotations for target
    getAnnotations(targetType, targetId) {
        return this.annotations.filter(
            a => a.targetType === targetType && a.targetId === targetId
        );
    }
    
    // Delete annotation
    deleteAnnotation(annotationId) {
        const annotation = this.annotations.find(a => a.id === annotationId);
        
        if (!annotation) {
            throw new Error('Annotation not found');
        }
        
        // Only author can delete
        if (annotation.author !== this.currentUser.id) {
            throw new Error('Not authorized to delete this annotation');
        }
        
        this.annotations = this.annotations.filter(a => a.id !== annotationId);
        
        if (this.socket) {
            this.socket.emit('delete_annotation', { annotationId: annotationId });
        }
        
        return true;
    }
    
    // Create version/snapshot
    createVersion(experimentId, name, description, data) {
        const version = {
            id: this.generateId(),
            experimentId: experimentId,
            name: name,
            description: description,
            data: data,
            createdBy: this.currentUser.id,
            createdByName: this.currentUser.username,
            createdAt: new Date().toISOString(),
            tags: []
        };
        
        this.versions.push(version);
        
        if (this.socket) {
            this.socket.emit('create_version', version);
        }
        
        return version;
    }
    
    // Get version history
    getVersionHistory(experimentId) {
        return this.versions
            .filter(v => v.experimentId === experimentId)
            .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
    }
    
    // Restore version
    restoreVersion(versionId) {
        const version = this.versions.find(v => v.id === versionId);
        
        if (!version) {
            throw new Error('Version not found');
        }
        
        return version.data;
    }
    
    // Tag version
    tagVersion(versionId, tags) {
        const version = this.versions.find(v => v.id === versionId);
        
        if (!version) {
            throw new Error('Version not found');
        }
        
        version.tags = Array.isArray(tags) ? tags : [tags];
        
        return version;
    }
    
    // Compare versions
    compareVersions(versionId1, versionId2) {
        const v1 = this.versions.find(v => v.id === versionId1);
        const v2 = this.versions.find(v => v.id === versionId2);
        
        if (!v1 || !v2) {
            throw new Error('One or both versions not found');
        }
        
        return {
            version1: v1,
            version2: v2,
            differences: this.calculateDifferences(v1.data, v2.data)
        };
    }
    
    // Calculate differences between two objects
    calculateDifferences(obj1, obj2) {
        const differences = [];
        
        const compareObjects = (o1, o2, path = '') => {
            const keys = new Set([...Object.keys(o1), ...Object.keys(o2)]);
            
            keys.forEach(key => {
                const fullPath = path ? `${path}.${key}` : key;
                
                if (!(key in o1)) {
                    differences.push({ path: fullPath, type: 'added', value: o2[key] });
                } else if (!(key in o2)) {
                    differences.push({ path: fullPath, type: 'removed', value: o1[key] });
                } else if (typeof o1[key] !== typeof o2[key]) {
                    differences.push({ 
                        path: fullPath, 
                        type: 'changed', 
                        oldValue: o1[key], 
                        newValue: o2[key] 
                    });
                } else if (typeof o1[key] === 'object' && o1[key] !== null) {
                    compareObjects(o1[key], o2[key], fullPath);
                } else if (o1[key] !== o2[key]) {
                    differences.push({ 
                        path: fullPath, 
                        type: 'changed', 
                        oldValue: o1[key], 
                        newValue: o2[key] 
                    });
                }
            });
        };
        
        compareObjects(obj1, obj2);
        
        return differences;
    }
    
    // Notification handlers
    notifyUserJoined(user) {
        console.log(`User joined: ${user.username}`);
        this.dispatchEvent('userJoined', user);
    }
    
    notifyUserLeft(userId) {
        console.log(`User left: ${userId}`);
        this.dispatchEvent('userLeft', { userId });
    }
    
    notifySimulationUpdated(simulation) {
        console.log(`Simulation updated: ${simulation.name}`);
        this.dispatchEvent('simulationUpdated', simulation);
    }
    
    notifyAnnotationAdded(annotation) {
        console.log(`Annotation added by ${annotation.authorName}`);
        this.dispatchEvent('annotationAdded', annotation);
    }
    
    notifyVersionCreated(version) {
        console.log(`Version created: ${version.name}`);
        this.dispatchEvent('versionCreated', version);
    }
    
    // Event dispatcher
    dispatchEvent(eventName, data) {
        const event = new CustomEvent(`collab:${eventName}`, { detail: data });
        document.dispatchEvent(event);
    }
    
    // Helper functions
    generateId() {
        return 'id_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    generateUserId() {
        return 'user_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    generateUserColor() {
        const colors = [
            '#4a90e2', '#7b68ee', '#5cb85c', '#f0ad4e', 
            '#d9534f', '#5bc0de', '#f39c12', '#e74c3c'
        ];
        return colors[Math.floor(Math.random() * colors.length)];
    }
    
    // Save collaboration data
    saveCollaborationData() {
        return {
            annotations: this.annotations,
            versions: this.versions,
            sharedSimulations: this.sharedSimulations
        };
    }
    
    // Load collaboration data
    loadCollaborationData(data) {
        if (data.annotations) {
            this.annotations = data.annotations;
        }
        if (data.versions) {
            this.versions = data.versions;
        }
        if (data.sharedSimulations) {
            this.sharedSimulations = data.sharedSimulations;
        }
    }
}

// Export for use in main app
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CollaborationManager;
}
