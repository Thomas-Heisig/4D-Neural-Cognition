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

#### Memory Leak in Long Simulations
- **Status**: Open
- **Severity**: High
- **Affected Versions**: All
- **Description**: Memory usage grows unbounded in simulations running for >10,000 steps
- **Reproduction**: Run simulation for 20,000+ steps and monitor memory
- **Impact**: Prevents long-running experiments, eventually crashes
- **Workaround**: Restart simulation periodically and checkpoint state
- **Related**: Issue #TBD

#### Synapse Weight Overflow
- **Status**: Open
- **Severity**: High
- **Affected Versions**: All
- **Description**: Weight clipping doesn't always work correctly, leading to NaN values
- **Reproduction**: Run with very high learning rate (>0.1) for extended periods
- **Impact**: Simulation becomes unstable, produces invalid results
- **Workaround**: Use lower learning rates (<0.01)
- **Related**: `plasticity.py:hebbian_update()`

#### Race Condition in Web Interface
- **Status**: Open
- **Severity**: High
- **Affected Versions**: All
- **Description**: Concurrent requests can lead to inconsistent model state
- **Reproduction**: Start training and quickly stop/restart multiple times
- **Impact**: Model corruption, crashes
- **Workaround**: Wait for operations to complete before starting new ones
- **Related**: `app.py:simulation_lock` not comprehensive enough

### Medium Severity

#### HDF5 File Corruption on Interrupt
- **Status**: Open
- **Severity**: Medium
- **Affected Versions**: All
- **Description**: Interrupting save operation can corrupt HDF5 files
- **Reproduction**: Save large model and interrupt (Ctrl+C) during write
- **Impact**: Loss of model data
- **Workaround**: Wait for save to complete, use atomic writes
- **Related**: `storage.py:save_to_hdf5()`

#### Neuron Death Can Create Disconnected Networks
- **Status**: Open
- **Severity**: Medium
- **Affected Versions**: All
- **Description**: When neurons die, their synapses are removed but no reconnection occurs
- **Reproduction**: Enable death, run for extended period, check connectivity
- **Impact**: Network becomes sparse and disconnected over time
- **Workaround**: Disable death or periodically regenerate connections
- **Related**: `cell_lifecycle.py:maybe_kill_and_reproduce()`

#### Sensory Input Dimension Mismatch (RESOLVED)
- **Status**: ✅ Fixed (December 2025)
- **Severity**: Medium
- **Affected Versions**: Fixed in current version
- **Description**: Providing wrong-sized sensory input now raises clear validation errors
- **Resolution**: Added comprehensive input validation with helpful error messages
- **Related**: `senses.py:feed_sense_input()`

#### Web Frontend Freezes with Large Models
- **Status**: Open
- **Severity**: Medium
- **Affected Versions**: All
- **Description**: Heatmap visualization freezes browser with >10,000 neurons
- **Reproduction**: Initialize model with high density (>0.5), view heatmap
- **Impact**: UI becomes unresponsive
- **Workaround**: Use lower density or disable visualization
- **Related**: `static/js/app.js:updateHeatmap()`

### Low Severity

#### Log File Size Grows Unbounded
- **Status**: Open
- **Severity**: Low
- **Affected Versions**: All
- **Description**: No log rotation, files can grow very large
- **Reproduction**: Run for extended period with verbose logging
- **Impact**: Disk space consumption
- **Workaround**: Manually delete or rotate logs
- **Related**: `app.py:logging configuration`

#### Progress Indicator Inaccurate
- **Status**: Open
- **Severity**: Low
- **Affected Versions**: All
- **Description**: Training progress bar doesn't reflect actual time remaining
- **Reproduction**: Start long training run, observe progress bar
- **Impact**: User confusion about completion time
- **Workaround**: Monitor step count directly
- **Related**: `static/js/app.js:training progress`

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

#### Limited Neuron Models
- **Description**: Only LIF model currently implemented
- **Impact**: Cannot model diverse neuron behaviors
- **Limitation**: Implementation time constraint
- **Future**: Add Izhikevich, Hodgkin-Huxley models
- **Workaround**: Tune LIF parameters to approximate different behaviors

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
- **Status**: ✅ Partially Completed (December 2025)
- **Location**: Core modules now covered
- **Resolution**: Added comprehensive test suite with pytest
  - 75 unit tests across brain_model, simulation, and cell_lifecycle
  - 100% test pass rate
  - pytest configuration with coverage support
  - Fixtures for common test scenarios
- **Remaining Work**: Add tests for plasticity, senses, and storage modules
- **Priority**: Medium (core modules covered)
- **Effort**: Medium
- **Plan**: Continue expanding test coverage

#### No Integration Tests
- **Location**: N/A
- **Issue**: No end-to-end testing
- **Impact**: Cannot verify full workflow
- **Priority**: High
- **Effort**: Medium
- **Plan**: Add after unit tests

#### No Performance Tests
- **Location**: N/A
- **Issue**: No benchmarks or performance regression tests
- **Impact**: Cannot track performance improvements/degradations
- **Priority**: Medium
- **Effort**: Medium
- **Plan**: Add benchmarking suite

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

### Flask Secret Key
- **Location**: `app.py`
- **Issue**: Hardcoded secret key
- **Severity**: Medium
- **Impact**: Session hijacking possible
- **Mitigation**: Change key, use environment variable
- **Status**: Needs fix

### No Input Validation
- **Location**: Web API endpoints
- **Issue**: User inputs not validated
- **Severity**: Medium
- **Impact**: Possible injection attacks, crashes
- **Mitigation**: Add validation, sanitization
- **Status**: Needs fix

### File Path Injection
- **Location**: `storage.py` load functions
- **Issue**: User-provided paths not validated
- **Severity**: Low
- **Impact**: Could read arbitrary files
- **Mitigation**: Validate and sanitize paths
- **Status**: Needs fix

### No Rate Limiting
- **Location**: Web API
- **Issue**: No protection against DoS
- **Severity**: Low
- **Impact**: Server could be overwhelmed
- **Mitigation**: Add rate limiting
- **Status**: Nice to have

### Pickle Usage Concerns
- **Location**: Potentially in storage
- **Issue**: Pickle can execute arbitrary code
- **Severity**: High (if used)
- **Impact**: Remote code execution
- **Mitigation**: Avoid pickle, use JSON/HDF5 only
- **Status**: Verify not used

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

### 2025-12-06 (December Update - Testing & Documentation)
- ✅ RESOLVED: Added comprehensive unit test suite (75 tests, 100% pass rate)
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
