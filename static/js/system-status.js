// System Status Manager
// Handles checking backend initialization status and preventing premature API calls

class SystemStatusManager {
    constructor() {
        this.initialized = false;
        this.checking = false;
        this.checkInterval = 2000; // Check every 2 seconds
        this.maxRetries = 30; // Max 1 minute of retries
        this.retries = 0;
        this.callbacks = [];
        this.statusCheckTimer = null;
    }

    /**
     * Start checking system status until initialized
     * @returns {Promise} Resolves when system is initialized
     */
    async waitForInitialization() {
        if (this.initialized) {
            return Promise.resolve();
        }

        // Return existing promise if already checking
        if (this.checking) {
            return new Promise((resolve, reject) => {
                this.callbacks.push({ resolve, reject });
            });
        }

        this.checking = true;
        this.showLoadingOverlay();

        return new Promise((resolve, reject) => {
            this.callbacks.push({ resolve, reject });
            this.startStatusCheck();
        });
    }

    /**
     * Start periodic status checks
     */
    startStatusCheck() {
        this.checkStatus();
    }

    /**
     * Check system status via API
     */
    async checkStatus() {
        try {
            const response = await fetch('/api/system/status');
            const data = await response.json();

            if (data.initialized) {
                this.handleInitialized(data);
            } else {
                this.handleNotInitialized(data);
            }
        } catch (error) {
            console.error('Status check failed:', error);
            this.retries++;

            if (this.retries >= this.maxRetries) {
                this.handleTimeout();
            } else {
                // Retry after interval
                this.statusCheckTimer = setTimeout(() => this.checkStatus(), this.checkInterval);
            }
        }
    }

    /**
     * Handle system initialized state
     */
    handleInitialized(statusData) {
        this.initialized = true;
        this.checking = false;
        
        // Clear any pending timer
        if (this.statusCheckTimer) {
            clearTimeout(this.statusCheckTimer);
            this.statusCheckTimer = null;
        }

        this.hideLoadingOverlay();
        
        // Resolve all waiting callbacks
        this.callbacks.forEach(cb => cb.resolve(statusData));
        this.callbacks = [];

        // Emit custom event for other scripts
        window.dispatchEvent(new CustomEvent('systemInitialized', { detail: statusData }));
    }

    /**
     * Handle system not initialized state
     */
    handleNotInitialized(statusData) {
        this.retries++;
        
        // Update overlay message
        this.updateLoadingMessage(
            'System starting up...',
            `Waiting for initialization (${this.retries}/${this.maxRetries})`
        );

        if (this.retries >= this.maxRetries) {
            this.handleTimeout();
        } else {
            // Schedule next check
            this.statusCheckTimer = setTimeout(() => this.checkStatus(), this.checkInterval);
        }
    }

    /**
     * Handle timeout when system doesn't initialize
     */
    handleTimeout() {
        this.checking = false;
        
        if (this.statusCheckTimer) {
            clearTimeout(this.statusCheckTimer);
            this.statusCheckTimer = null;
        }

        this.updateLoadingMessage(
            'Initialization Required',
            'Please initialize the model using the controls below.',
            'warning'
        );

        // Reject all waiting callbacks
        const error = new Error('System initialization timeout');
        this.callbacks.forEach(cb => cb.reject(error));
        this.callbacks = [];

        // Hide overlay after a delay so user can see the message
        setTimeout(() => this.hideLoadingOverlay(), 3000);
    }

    /**
     * Show loading overlay
     */
    showLoadingOverlay() {
        let overlay = document.getElementById('systemStatusOverlay');
        
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.id = 'systemStatusOverlay';
            overlay.className = 'system-status-overlay';
            overlay.innerHTML = `
                <div class="status-content">
                    <div class="status-spinner"></div>
                    <h2 class="status-title">System Starting...</h2>
                    <p class="status-message">Checking initialization status</p>
                </div>
            `;
            document.body.appendChild(overlay);
        }
        
        overlay.style.display = 'flex';
    }

    /**
     * Hide loading overlay
     */
    hideLoadingOverlay() {
        const overlay = document.getElementById('systemStatusOverlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
    }

    /**
     * Update loading overlay message
     */
    updateLoadingMessage(title, message, type = 'info') {
        const overlay = document.getElementById('systemStatusOverlay');
        if (!overlay) return;

        const titleEl = overlay.querySelector('.status-title');
        const messageEl = overlay.querySelector('.status-message');
        const content = overlay.querySelector('.status-content');

        if (titleEl) titleEl.textContent = title;
        if (messageEl) messageEl.textContent = message;
        if (content) {
            content.className = 'status-content ' + type;
        }
    }

    /**
     * Force check system status (manual trigger)
     */
    async forceCheck() {
        try {
            const response = await fetch('/api/system/status');
            const data = await response.json();
            
            if (data.initialized) {
                this.initialized = true;
            }
            
            return data;
        } catch (error) {
            console.error('Force check failed:', error);
            return { initialized: false, error: error.message };
        }
    }

    /**
     * Reset status manager (for testing or manual reset)
     */
    reset() {
        this.initialized = false;
        this.checking = false;
        this.retries = 0;
        
        if (this.statusCheckTimer) {
            clearTimeout(this.statusCheckTimer);
            this.statusCheckTimer = null;
        }
        
        this.callbacks = [];
        this.hideLoadingOverlay();
    }

    /**
     * Mark system as initialized (after manual initialization)
     */
    markInitialized() {
        if (!this.initialized) {
            this.handleInitialized({ initialized: true });
        }
    }
}

// Create global instance
window.systemStatus = new SystemStatusManager();

// Auto-start status check on page load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        // Don't auto-check on pages where initialization is not expected initially
        // Let individual pages decide when to start checking
    });
} else {
    // DOM already loaded
}
