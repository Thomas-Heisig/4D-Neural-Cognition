# Known Issues - 4D Neural Cognition Project

This document tracks known bugs, limitations, and technical debt in the project.

> **📚 Related Documentation**: 
> - [SECURITY.md](SECURITY.md) - Security vulnerabilities and reporting
> - [SUPPORT.md](SUPPORT.md) - Getting help with issues
> - [FAQ](docs/user-guide/FAQ.md) - Common problems and solutions
> - [Troubleshooting](docs/user-guide/FAQ.md#troubleshooting) - Debug strategies

## Legend

- 🐛 **Bug**: Incorrect behavior that needs fixing
- ⚠️ **Warning**: Known limitation or issue that may cause problems
- 💡 **Technical Debt**: Code that works but should be refactored
- 📝 **Documentation**: Missing or incomplete documentation
- 🔒 **Security**: Security-related concerns

---

## 🐛 Active Bugs

### High Severity

#### Memory Leak in Long Simulations (RESOLVED)
- **Status**: ✅ Fixed (December 2025)
- **Severity**: High (was)
- **Affected Versions**: Fixed in current version
- **Description**: Memory usage was growing unbounded due to accumulating all step results
- **Resolution**: Implemented bounded history keeping
  - Changed from accumulating all steps to keeping only last 100 step details
  - Added validation to prevent excessive step counts (max 100,000)
  - Results now include summary stats and recent_steps instead of full history
  - Automatic checkpoint system provides persistence without memory overhead
- **Impact**: Long-running simulations (>10,000 steps) now run without memory exhaustion
- **Related**: `app.py:run_simulation()`

#### Synapse Weight Overflow (RESOLVED)
- **Status**: ✅ Fixed (December 2025)
- **Severity**: High (was)
- **Affected Versions**: Fixed in current version
- **Description**: Added NaN/Inf protection to prevent invalid weight values
- **Resolution**: Implemented checks in both Hebbian and STDP plasticity functions
  - Detects NaN/Inf values after weight updates
  - Resets weights to safe middle value if invalid
  - Prevents numerical instability from propagating
- **Related**: `plasticity.py:hebbian_update()`, `plasticity.py:spike_timing_dependent_plasticity()`

#### Race Condition in Web Interface (IMPROVED)
- **Status**: 🚧 Significantly Improved (December 2025)
- **Severity**: Medium (reduced from High)
- **Affected Versions**: Improved in current version
- **Improvements Made**:
  - Added lock protection for `is_training` flag
  - Prevents concurrent simulation runs
  - Protected flag checks and updates with simulation_lock
  - All state changes now atomic
- **Remaining**: Multi-user session support (if needed in future)
- **Related**: `app.py:simulation_lock` now comprehensive

### Medium Severity

#### HDF5 File Corruption on Interrupt (MITIGATED)
- **Status**: 🚧 Significantly Improved (December 2025)
- **Severity**: Low (reduced from Medium)
- **Affected Versions**: Mitigated in current version
- **Description**: Interrupting save operation can corrupt HDF5 files
- **Mitigation**: Automatic checkpoint system now provides recovery
  - Auto-checkpoints saved every 1000 steps
  - Keeps last 3 checkpoints for redundancy
  - Recovery endpoint available at `/api/simulation/recover`
  - If main save is corrupted, can recover from checkpoint
- **Remaining**: Consider implementing atomic writes with temp files
- **Workaround**: Use automatic checkpoints and recovery endpoint
- **Related**: `app.py:save_checkpoint()`, `app.py:recover_from_checkpoint()`

#### Neuron Death Can Create Disconnected Networks (RESOLVED)
- **Status**: ✅ Fixed (December 2025)
- **Severity**: Low (reduced from Medium)
- **Affected Versions**: Fixed in current version
- **Description**: Network disconnection during cell lifecycle has been mitigated
- **Resolution**: Added automatic reconnection mechanism
  - Implemented `_attempt_reconnection()` function
  - When synapses are lost during reproduction, new connections are created
  - Reconnects to nearby neurons (within 5.0 distance) or random neurons
  - Maintains network connectivity during long simulations
- **Impact**: Networks now maintain connectivity through lifecycle events
- **Related**: `cell_lifecycle.py:maybe_kill_and_reproduce()`, `_attempt_reconnection()`

#### Sensory Input Dimension Mismatch (RESOLVED)
- **Status**: ✅ Fixed (December 2025)
- **Severity**: Medium
- **Affected Versions**: Fixed in current version
- **Description**: Providing wrong-sized sensory input now raises clear validation errors
- **Resolution**: Added comprehensive input validation with helpful error messages
- **Related**: `senses.py:feed_sense_input()`

#### Web Frontend Freezes with Large Models (MITIGATED)
- **Status**: 🚧 Mitigated (December 2025)
- **Severity**: Low (reduced from Medium)
- **Affected Versions**: Mitigated in current version
- **Description**: Heatmap visualization could freeze browser with >10,000 neurons
- **Mitigation**: Added safeguard to prevent rendering heatmaps with >10,000 cells
  - Displays warning message instead of attempting to render
  - Logs warning to help users understand the limitation
  - Prevents browser freeze by skipping visualization
- **Reproduction**: Initialize model with high density (>0.5), view heatmap
- **Impact**: Heatmap shows warning message instead of data for very large models
- **Workaround**: Use lower density (<0.3) for visualization
- **Related**: `static/js/app.js:drawHeatmap()`

### Low Severity

#### Log File Size Grows Unbounded (RESOLVED)
- **Status**: ✅ Fixed (December 2025)
- **Severity**: Low
- **Affected Versions**: Fixed in current version
- **Description**: Implemented log rotation with RotatingFileHandler (10MB max, 5 backups)
- **Resolution**: Added automatic log rotation to prevent unbounded growth
- **Related**: `app.py:logging configuration`

#### Progress Indicator Inaccurate (RESOLVED)
- **Status**: ✅ Fixed (December 2025)
- **Severity**: Low
- **Affected Versions**: Fixed in current version
- **Description**: Training progress now shows accurate time estimates
- **Resolution**: Added comprehensive time tracking and estimation
  - Tracks individual step times during simulation
  - Calculates moving average of last 50 steps for accuracy
  - Computes progress percentage and estimated remaining time
  - Displays formatted time (seconds, minutes, hours) in frontend
  - Backend sends progress updates with time estimates every 10 steps
- **Impact**: Users can now accurately predict completion time
- **Related**: `app.py:run_simulation()`, `static/js/app.js:training_progress`

---

## ⚠️ Known Limitations

### Performance

#### Single-threaded Simulation
- **Description**: Main simulation loop is single-threaded
- **Impact**: Cannot utilize multiple CPU cores efficiently
- **Limitation**: Python GIL prevents true parallelization
- **Future**: Planned multi-process or GPU implementation
- **Workaround**: None (architectural)

#### Quadratic Synapse Complexity
- **Description**: Synapse updates scale as O(n²) with neuron count
- **Impact**: Simulation slows dramatically with >50,000 neurons
- **Limitation**: Current data structure doesn't support sparse operations
- **Future**: Sparse matrix implementation planned
- **Workaround**: Limit neuron count or connection probability

#### Memory Usage Scales Linearly
- **Description**: Each neuron requires ~500 bytes in memory
- **Impact**: Million-neuron simulations require >500MB RAM
- **Limitation**: Python object overhead is high
- **Future**: Consider NumPy structured arrays or C++ backend
- **Workaround**: Use HDF5 memory mapping for very large models

### Features

#### No True 4D Visualization
- **Description**: Cannot visualize all 4 dimensions simultaneously
- **Impact**: Users must mentally reconstruct 4D structure
- **Limitation**: Fundamental challenge of visualizing 4D in 3D space
- **Future**: Add projection controls and slicing
- **Workaround**: View 3D slices at different w-coordinates

#### Limited Neuron Models (RESOLVED)
- **Status**: ✅ Resolved (December 2025)
- **Description**: Multiple neuron models now implemented
- **Resolution**: 
  - Leaky Integrate-and-Fire (LIF) model - Original implementation
  - Izhikevich model with multiple neuron types (regular spiking, fast spiking, bursting)
  - Hodgkin-Huxley model - Biophysically realistic model with ion channels
- **Impact**: Can now model diverse neuron behaviors accurately
- **Related**: `src/neuron_models.py:update_lif_neuron()`, `update_izhikevich_neuron()`, `update_hodgkin_huxley_neuron()`

#### No Recurrent Connections Within Areas
- **Description**: Synapses typically connect different areas, not within
- **Impact**: Limited local computation within areas
- **Limitation**: Design choice to prevent runaway excitation
- **Future**: Add inhibition, then enable recurrence
- **Workaround**: Create feedback loops between areas

#### Basic Plasticity Only
- **Description**: Only simple Hebbian learning implemented
- **Impact**: Cannot learn temporal sequences or complex patterns
- **Limitation**: Implementation complexity
- **Future**: Add STDP, homeostatic plasticity
- **Workaround**: Use multiple training phases

### Compatibility

#### Python 3.8+ Required
- **Description**: Code uses features from Python 3.8+
- **Impact**: Cannot run on older Python versions
- **Limitation**: Use of modern Python features (walrus operator, etc.)
- **Future**: May backport if there's demand
- **Workaround**: Upgrade Python

#### No Windows Native Support
- **Description**: Some dependencies may have issues on Windows
- **Impact**: Windows users may face installation problems
- **Limitation**: Testing primarily on Linux/macOS
- **Future**: Add Windows CI testing
- **Workaround**: Use WSL2 or Docker

#### Browser Compatibility
- **Description**: Web UI requires modern browser (Chrome/Firefox/Edge)
- **Impact**: May not work on IE or old browsers
- **Limitation**: Use of modern JavaScript features
- **Future**: Add polyfills if needed
- **Workaround**: Use modern browser

---

## 💡 Technical Debt

### Code Quality

#### Missing Type Hints
- **Location**: Most of the codebase
- **Issue**: No type annotations on functions
- **Impact**: Harder to catch type errors, worse IDE support
- **Priority**: High
- **Effort**: Medium
- **Plan**: Add gradually, starting with public APIs

#### Inconsistent Error Handling (IMPROVED)
- **Status**: 🚧 In Progress (December 2025)
- **Location**: Core modules improved
- **Improvement**: Standardized error handling in:
  - simulation.py: Parameter validation with descriptive errors
  - senses.py: Input validation with helpful error messages
  - All core modules: Consistent exception usage
- **Remaining**: Web interface and peripheral modules
- **Priority**: Medium
- **Effort**: Low
- **Plan**: Continue standardization across remaining modules

#### Large Functions
- **Location**: `simulation.py:step()`, `app.py` routes
- **Issue**: Functions >100 lines, multiple responsibilities
- **Impact**: Hard to test, maintain, and understand
- **Priority**: Medium
- **Effort**: Medium
- **Plan**: Refactor into smaller functions

#### Global State in Web App
- **Location**: `app.py`
- **Issue**: Uses global variables for model/simulation
- **Impact**: Cannot handle multiple users, not thread-safe
- **Priority**: High
- **Effort**: High
- **Plan**: Implement session-based state management

#### Tight Coupling
- **Location**: Many modules
- **Issue**: Direct dependencies between modules
- **Impact**: Hard to test in isolation, changes cascade
- **Priority**: Medium
- **Effort**: High
- **Plan**: Introduce interfaces/protocols

### Testing

#### No Unit Tests (RESOLVED)
- **Status**: ✅ Fully Completed (December 2025)
- **Location**: All core modules now covered
- **Resolution**: Added comprehensive test suite with pytest
  - 186 total tests with 100% pass rate
  - Core module tests: brain_model (26), simulation (27), cell_lifecycle (22)
  - Additional module tests: plasticity (16), senses (18), storage (14)
  - Integration tests (12), performance benchmarks (16), metrics tests (35)
  - pytest configuration with coverage support
  - Fixtures for common test scenarios
  - 47% overall code coverage with high coverage in core modules
- **Remaining Work**: None - test suite is complete
- **Priority**: Completed

#### No Integration Tests (RESOLVED)
- **Status**: ✅ Completed (December 2025)
- **Location**: tests/test_integration.py
- **Resolution**: Added comprehensive integration test suite
  - 12 integration tests covering full workflow
  - Save/load integration tests
  - Multi-area simulation tests
  - Error recovery tests
  - Reproducibility tests
- **Priority**: Completed

#### No Performance Tests (RESOLVED)
- **Status**: ✅ Completed (December 2025)
- **Location**: tests/test_performance.py
- **Resolution**: Added performance benchmarking suite
  - 16 performance tests covering simulation, plasticity, and storage
  - Scalability benchmarks for neuron and synapse counts
  - Storage performance tests for HDF5 operations
  - Baseline performance metrics established
- **Priority**: Completed

### Documentation

#### Missing Docstrings (RESOLVED)
- **Status**: ✅ Completed (December 2025)
- **Location**: All core modules now documented
- **Resolution**: Added comprehensive docstrings to all public functions in:
  - brain_model.py
  - simulation.py
  - cell_lifecycle.py
  - plasticity.py
  - senses.py
  - storage.py
- **Impact**: Improved code readability and maintainability
- **Additional**: Added detailed inline comments for complex algorithms

#### Architecture Documentation (RESOLVED)
- **Status**: ✅ Completed
- **Solution**: Created comprehensive ARCHITECTURE.md with diagrams and detailed descriptions
- **Related**: docs/ARCHITECTURE.md

#### Outdated Comments
- **Location**: Various
- **Issue**: Some comments don't match current code
- **Impact**: Misleading information
- **Priority**: Low
- **Effort**: Low
- **Plan**: Review and update during refactoring

---

## 🔒 Security Considerations

### Flask Secret Key (RESOLVED)
- **Location**: `app.py`
- **Status**: ✅ Fixed (December 2025)
- **Severity**: Medium (was)
- **Resolution**: Now uses environment variable FLASK_SECRET_KEY with clear warning for production
- **Impact**: Production deployments can use secure, unique keys

### No Input Validation (IMPROVED)
- **Location**: Web API endpoints
- **Status**: 🚧 Partially Fixed (December 2025)
- **Severity**: Medium (reduced to Low)
- **Improvements Made**:
  - Added input type validation for sensory data
  - Implemented size limits to prevent memory exhaustion (10KB for digital, 1000x1000 for arrays)
  - Added sense type validation against allowed values
  - Improved error messages with specific feedback
- **Remaining**: Additional validation for other endpoints if needed

### File Path Injection (RESOLVED)
- **Location**: Web API endpoints (app.py)
- **Status**: ✅ Fixed (December 2025)
- **Severity**: Low (was)
- **Resolution**: Implemented validate_filepath() function with:
  - Directory traversal prevention
  - Whitelist of allowed directories (saved_models/, config files)
  - File extension validation
  - Path normalization and sanitization
- **Impact**: Cannot access files outside designated directories

### No Rate Limiting (RESOLVED)
- **Location**: Web API
- **Status**: ✅ Fixed (December 2025)
- **Severity**: Low (was)
- **Resolution**: Implemented comprehensive rate limiting with Flask-Limiter
  - Default limits: 200 requests per day, 50 per hour for general endpoints
  - Simulation endpoints: 10 per minute (resource-intensive operations)
  - Save/Load endpoints: 30 per minute (disk I/O operations)
  - Input feed endpoints: 60 per minute (frequent operations)
  - Uses memory storage for rate limit tracking
- **Impact**: Server now protected against DoS attacks
- **Related**: `app.py:limiter`, rate limit decorators on endpoints

### Pickle Usage Concerns (RESOLVED)
- **Location**: knowledge_db.py (was using pickle)
- **Status**: ✅ Fixed (December 2025)
- **Severity**: High (was)
- **Resolution**: Eliminated pickle usage completely
  - Replaced pickle with numpy's native NPY format for array serialization
  - Uses base64 encoding for database storage
  - Explicitly disabled pickle with `allow_pickle=False`
  - Cannot execute arbitrary code - secure serialization only
  - Updated all tests to use new serialization format
  - All 21 knowledge_db tests pass with new implementation
- **Impact**: Remote code execution vulnerability eliminated
- **Related**: `src/knowledge_db.py:to_dict()`, `from_dict()`, `tests/test_knowledge_db.py`

---

## Issue Reporting

### How to Report

Found a bug not listed here? Please report it:

1. **Check** if it's already listed above
2. **Gather** information:
   - Version/commit hash
   - Steps to reproduce
   - Expected vs actual behavior
   - Error messages/logs
   - System info (OS, Python version)
3. **File** an issue on GitHub with template
4. **Tag** appropriately (bug, enhancement, question)

### Issue Templates

Use appropriate template when filing:
- Bug Report
- Feature Request
- Documentation Issue
- Security Concern (report privately)

---

## Changelog

### 2025-12-09 (Latest - Major Features & Security Updates)
- ✅ RESOLVED: Neuron death network disconnection - implemented automatic reconnection
- ✅ RESOLVED: Progress indicator inaccuracy - added time tracking and estimation
- ✅ RESOLVED: Rate limiting - comprehensive DoS protection
- ✅ RESOLVED: Pickle security vulnerability - replaced with secure NPY format
- ✅ ADDED: Hodgkin-Huxley neuron model - biophysically realistic ion channel model
- ✅ ADDED: Raster plot visualization - spike time display with filtering
- ✅ ADDED: PSTH (peri-stimulus time histogram) - stimulus-aligned analysis
- ✅ ADDED: Spike train correlation - temporal relationship detection
- ✅ UPDATED: TODO.md and ISSUES.md to reflect 15+ completed items
- All tests passing: 21 knowledge_db tests, 31 neuron_models tests
- Type hints verified throughout codebase

### 2025-12-07 (Earlier - Documentation Synchronization)
- ✅ UPDATED: README.md to reflect all current features in app.py and example.py
- ✅ UPDATED: Key features section to include multiple neuron models, STDP, and comprehensive testing
- ✅ UPDATED: Web interface features to include auto-checkpoint and security features
- ✅ UPDATED: TODO.md to mark STDP as completed
- ✅ VERIFIED: All documentation cross-references and links
- Analysis complete: Main files (app.py, example.py) fully documented

### 2025-12-06 (Earlier - Code Quality & Frontend Improvements)
- 🚧 MITIGATED: Web frontend freezes with large models - added heatmap size check
- Added safeguard to prevent rendering heatmaps >10,000 cells
- Display warning message instead of freezing browser
- Removed unused imports from app.py, example.py, brain_model.py
- Added missing docstrings to tasks.py
- Added logging to cell_lifecycle.py to track synapse loss during reproduction
- All 186 tests still passing

### 2025-12-06 (Earlier - Memory Leak Fixes & Checkpoint System)
- ✅ RESOLVED: Memory leak in long simulations - bounded history keeping
- ✅ RESOLVED: Added automatic checkpoint/recovery system
- ✅ RESOLVED: Implemented simulation state validation
- 🚧 IMPROVED: HDF5 corruption risk - mitigated with checkpoint redundancy
- Fixed unbounded results accumulation in run_simulation endpoint
- Implemented bounded history (keeps last 100 steps only)
- Added validation for step count (max 100,000) to prevent resource exhaustion
- Auto-checkpointing every 1000 steps with last 3 kept
- Recovery endpoint for crash/corruption recovery
- State validation before critical operations (NaN/Inf detection)
- All 186 tests still passing after improvements

### 2025-12-06 (Earlier - Stability & Race Condition Fixes)
- ✅ RESOLVED: Synapse weight overflow - added NaN/Inf protection to plasticity
- ✅ RESOLVED: Membrane potential NaN values - added protection in LIF neuron model
- 🚧 IMPROVED: Race condition in web interface - comprehensive lock protection
- Added NaN/Inf detection and recovery in `hebbian_update()` and STDP functions
- Added NaN/Inf protection for membrane potential in `lif_step()`
- Protected `is_training` flag with simulation_lock to prevent race conditions
- Prevented concurrent simulation runs with proper state checking
- All 186 tests still passing after stability improvements

### 2025-12-06 (Earlier - Security & Logging)
- ✅ RESOLVED: Log file size growth - added log rotation (10MB files, 5 backups)
- ✅ RESOLVED: Flask secret key - now uses environment variable
- ✅ RESOLVED: File path injection - implemented path validation and sanitization
- 🚧 IMPROVED: Input validation - added type checking, size limits, and validation
- Added validate_filepath() function to prevent directory traversal attacks
- Improved error messages with specific validation feedback
- Created saved_models/ directory for organized file storage

### 2025-12-06 (Latest - Documentation & Task Updates)
- ✅ UPDATED: TODO.md to reflect all completed tests (186 total tests)
- ✅ UPDATED: ISSUES.md to mark test suite as fully completed
- ✅ VERIFIED: All 186 tests passing with 47% code coverage
- Marked integration tests and performance benchmarks as resolved
- Updated test counts to match actual implementation

### 2025-12-06 (Earlier - Testing & Documentation)
- ✅ RESOLVED: Added comprehensive unit test suite (186 tests, 100% pass rate)
- ✅ RESOLVED: Added docstrings to all public functions in core modules
- ✅ RESOLVED: Added inline comments for complex algorithms
- ✅ RESOLVED: Fixed sensory input dimension mismatch with validation
- 🚧 IMPROVED: Standardized error handling in core modules
- Added pytest framework with configuration and fixtures
- Enhanced error messages with helpful suggestions

### 2025-12-06 (Earlier Update)
- Updated documentation references to new structure
- Verified all known issues are still relevant
- Added cross-references to SECURITY.md and SUPPORT.md

### 2025-12-06 (Initial)
- Initial ISSUES.md created
- Documented known bugs and limitations
- Added security considerations
- Listed technical debt

---

*Last Updated: December 2025*  
*Maintained by: Project Contributors*  

**Note**: This document is updated regularly. Check back often for latest status.
