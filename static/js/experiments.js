// Experiment Management Module
// Handles batch parameter modification, parameter sweeps, A/B testing, and experiment versioning

class ExperimentManager {
    constructor() {
        this.experiments = [];
        this.currentExperiment = null;
        this.parameterSets = [];
        this.abTests = [];
    }
    
    // Create a new experiment
    createExperiment(name, description, parameters) {
        const experiment = {
            id: this.generateId(),
            name: name,
            description: description,
            parameters: parameters,
            createdAt: new Date().toISOString(),
            status: 'created',
            results: null,
            version: 1,
            parentId: null
        };
        
        this.experiments.push(experiment);
        this.currentExperiment = experiment;
        
        return experiment;
    }
    
    // Create a variant of an existing experiment
    createVariant(parentId, name, modifiedParameters) {
        const parent = this.experiments.find(e => e.id === parentId);
        if (!parent) {
            throw new Error('Parent experiment not found');
        }
        
        const experiment = {
            id: this.generateId(),
            name: name,
            description: `Variant of ${parent.name}`,
            parameters: { ...parent.parameters, ...modifiedParameters },
            createdAt: new Date().toISOString(),
            status: 'created',
            results: null,
            version: parent.version + 1,
            parentId: parentId
        };
        
        this.experiments.push(experiment);
        
        return experiment;
    }
    
    // Batch parameter modification
    batchModifyParameters(experimentIds, parameterChanges) {
        const modified = [];
        
        experimentIds.forEach(id => {
            const experiment = this.experiments.find(e => e.id === id);
            if (experiment && experiment.status === 'created') {
                Object.assign(experiment.parameters, parameterChanges);
                modified.push(experiment);
            }
        });
        
        return modified;
    }
    
    // Generate parameter sweep configurations
    generateParameterSweep(baseParameters, sweepConfig) {
        const parameterSets = [];
        
        // sweepConfig format: { parameterName: [value1, value2, ...] }
        const paramNames = Object.keys(sweepConfig);
        
        if (paramNames.length === 0) {
            return [baseParameters];
        }
        
        // Generate all combinations
        const generateCombinations = (params, index) => {
            if (index >= paramNames.length) {
                parameterSets.push({ ...params });
                return;
            }
            
            const paramName = paramNames[index];
            const values = sweepConfig[paramName];
            
            values.forEach(value => {
                const newParams = { ...params };
                newParams[paramName] = value;
                generateCombinations(newParams, index + 1);
            });
        };
        
        generateCombinations(baseParameters, 0);
        this.parameterSets = parameterSets;
        
        return parameterSets;
    }
    
    // Create experiments for parameter sweep
    createSweepExperiments(baseName, baseParameters, sweepConfig) {
        const parameterSets = this.generateParameterSweep(baseParameters, sweepConfig);
        const experiments = [];
        
        parameterSets.forEach((params, index) => {
            const exp = this.createExperiment(
                `${baseName}_sweep_${index}`,
                `Parameter sweep configuration ${index}`,
                params
            );
            experiments.push(exp);
        });
        
        return experiments;
    }
    
    // Set up A/B test
    setupABTest(testName, configA, configB) {
        const abTest = {
            id: this.generateId(),
            name: testName,
            createdAt: new Date().toISOString(),
            experimentA: this.createExperiment(`${testName}_A`, 'Configuration A', configA),
            experimentB: this.createExperiment(`${testName}_B`, 'Configuration B', configB),
            results: null,
            winner: null
        };
        
        this.abTests.push(abTest);
        
        return abTest;
    }
    
    // Compare A/B test results
    compareABTest(testId, resultsA, resultsB) {
        const test = this.abTests.find(t => t.id === testId);
        if (!test) {
            throw new Error('A/B test not found');
        }
        
        test.results = {
            A: resultsA,
            B: resultsB,
            comparedAt: new Date().toISOString()
        };
        
        // Simple winner determination based on performance metric
        const scoreA = this.calculateScore(resultsA);
        const scoreB = this.calculateScore(resultsB);
        
        if (Math.abs(scoreA - scoreB) < 0.05) {
            test.winner = 'tie';
        } else {
            test.winner = scoreA > scoreB ? 'A' : 'B';
        }
        
        return test;
    }
    
    // Calculate overall score from results
    calculateScore(results) {
        // Weighted score calculation
        const weights = {
            accuracy: 0.3,
            precision: 0.2,
            recall: 0.2,
            f1Score: 0.2,
            stability: 0.1
        };
        
        let score = 0;
        Object.keys(weights).forEach(key => {
            if (results[key] !== undefined) {
                score += results[key] * weights[key];
            }
        });
        
        return score;
    }
    
    // Update experiment status
    updateExperimentStatus(experimentId, status, results = null) {
        const experiment = this.experiments.find(e => e.id === experimentId);
        if (!experiment) {
            throw new Error('Experiment not found');
        }
        
        experiment.status = status;
        experiment.updatedAt = new Date().toISOString();
        
        if (results) {
            experiment.results = results;
        }
        
        return experiment;
    }
    
    // Get experiment history (including variants)
    getExperimentHistory(experimentId) {
        const experiment = this.experiments.find(e => e.id === experimentId);
        if (!experiment) {
            return [];
        }
        
        const history = [experiment];
        
        // Find all descendants
        const findDescendants = (parentId) => {
            const children = this.experiments.filter(e => e.parentId === parentId);
            children.forEach(child => {
                history.push(child);
                findDescendants(child.id);
            });
        };
        
        findDescendants(experimentId);
        
        return history.sort((a, b) => a.version - b.version);
    }
    
    // Export experiment data
    exportExperiment(experimentId) {
        const experiment = this.experiments.find(e => e.id === experimentId);
        if (!experiment) {
            throw new Error('Experiment not found');
        }
        
        return JSON.stringify(experiment, null, 2);
    }
    
    // Import experiment data
    importExperiment(experimentData) {
        const experiment = typeof experimentData === 'string' 
            ? JSON.parse(experimentData) 
            : experimentData;
        
        // Assign new ID to avoid conflicts
        experiment.id = this.generateId();
        experiment.importedAt = new Date().toISOString();
        
        this.experiments.push(experiment);
        
        return experiment;
    }
    
    // Get all experiments
    getAllExperiments() {
        return this.experiments;
    }
    
    // Get experiments by status
    getExperimentsByStatus(status) {
        return this.experiments.filter(e => e.status === status);
    }
    
    // Delete experiment
    deleteExperiment(experimentId) {
        const index = this.experiments.findIndex(e => e.id === experimentId);
        if (index === -1) {
            throw new Error('Experiment not found');
        }
        
        this.experiments.splice(index, 1);
        
        // Also delete any variants
        this.experiments = this.experiments.filter(e => e.parentId !== experimentId);
        
        return true;
    }
    
    // Search experiments
    searchExperiments(query) {
        const lowerQuery = query.toLowerCase();
        return this.experiments.filter(e => 
            e.name.toLowerCase().includes(lowerQuery) ||
            e.description.toLowerCase().includes(lowerQuery)
        );
    }
    
    // Compare multiple experiments
    compareExperiments(experimentIds) {
        const experiments = experimentIds.map(id => 
            this.experiments.find(e => e.id === id)
        ).filter(e => e !== undefined);
        
        if (experiments.length === 0) {
            return null;
        }
        
        const comparison = {
            experiments: experiments.map(e => ({
                id: e.id,
                name: e.name,
                parameters: e.parameters,
                results: e.results,
                score: e.results ? this.calculateScore(e.results) : null
            })),
            timestamp: new Date().toISOString()
        };
        
        // Sort by score
        comparison.experiments.sort((a, b) => (b.score || 0) - (a.score || 0));
        
        return comparison;
    }
    
    // Generate unique ID
    generateId() {
        return 'exp_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    // Save state to localStorage
    saveToLocalStorage() {
        try {
            localStorage.setItem('experiments', JSON.stringify(this.experiments));
            localStorage.setItem('abTests', JSON.stringify(this.abTests));
            return true;
        } catch (e) {
            console.error('Failed to save to localStorage:', e);
            return false;
        }
    }
    
    // Load state from localStorage
    loadFromLocalStorage() {
        try {
            const experimentsData = localStorage.getItem('experiments');
            const abTestsData = localStorage.getItem('abTests');
            
            if (experimentsData) {
                this.experiments = JSON.parse(experimentsData);
            }
            
            if (abTestsData) {
                this.abTests = JSON.parse(abTestsData);
            }
            
            return true;
        } catch (e) {
            console.error('Failed to load from localStorage:', e);
            return false;
        }
    }
    
    // Clear all data
    clearAll() {
        this.experiments = [];
        this.currentExperiment = null;
        this.parameterSets = [];
        this.abTests = [];
        
        try {
            localStorage.removeItem('experiments');
            localStorage.removeItem('abTests');
        } catch (e) {
            console.error('Failed to clear localStorage:', e);
        }
    }
}

// Export for use in main app
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ExperimentManager;
}
