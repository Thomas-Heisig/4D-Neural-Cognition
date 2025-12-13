# Docker Setup Guide

This guide explains how to run the 4D Neural Cognition project using Docker containers.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Services](#services)
- [Usage Examples](#usage-examples)
- [Volumes and Persistence](#volumes-and-persistence)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (version 20.10+)
- [Docker Compose](https://docs.docker.com/compose/install/) (version 1.29+)

## Quick Start

### 1. Build and Run the Application

```bash
# Clone the repository
git clone https://github.com/Thomas-Heisig/4D-Neural-Cognition.git
cd 4D-Neural-Cognition

# Build and start the container
docker-compose up -d app

# View logs
docker-compose logs -f app
```

The web interface will be available at `http://localhost:5000`

### 2. Stop the Application

```bash
docker-compose down
```

## Services

The docker-compose.yml defines several services:

### `app` (Production)

The main production application.

```bash
# Start
docker-compose up -d app

# Access at: http://localhost:5000
```

**Features:**
- Optimized production build
- Auto-restart on failure
- Persistent data volumes
- Health checks

### `dev` (Development)

Development environment with hot-reload.

```bash
# Start development environment
docker-compose --profile dev up -d dev

# Access at: http://localhost:5001
```

**Features:**
- Source code mounted for live editing
- Development dependencies included
- Port 5001 (to avoid conflicts with production)

### `jupyter` (Analysis)

Jupyter notebook for data analysis.

```bash
# Start Jupyter
docker-compose --profile analysis up -d jupyter

# Get the token
docker-compose logs jupyter | grep token

# Access at: http://localhost:8889
```

**Features:**
- Full access to codebase
- Experiment and data directories mounted
- Pre-installed analysis tools

### `test` (Testing)

Run tests in an isolated environment.

```bash
# Run all tests
docker-compose --profile test run --rm test

# Run specific tests
docker-compose --profile test run --rm test pytest tests/test_simulation.py -v
```

## Usage Examples

### Running Simulations

#### Using the Web Interface

```bash
docker-compose up -d app
# Open browser to http://localhost:5000
```

#### Running Example Scripts

```bash
# Run the example simulation
docker-compose run --rm app python example.py

# Run validation script
docker-compose run --rm app python scripts/validate_neuron_models.py
```

### Running Tests

```bash
# All tests
docker-compose --profile test run --rm test

# Specific test file
docker-compose --profile test run --rm test pytest tests/test_integration.py -v

# With coverage
docker-compose --profile test run --rm test pytest --cov=src --cov-report=html
```

### Development Workflow

```bash
# Start development environment
docker-compose --profile dev up -d dev

# Watch logs
docker-compose logs -f dev

# Run tests while developing
docker-compose exec dev pytest tests/test_simulation.py -v

# Access Python shell
docker-compose exec dev ipython

# Format code
docker-compose exec dev black src/ tests/
```

### Data Analysis with Jupyter

```bash
# Start Jupyter
docker-compose --profile analysis up -d jupyter

# Get access token
docker-compose logs jupyter | grep "http://127.0.0.1"

# The notebooks have access to:
# - /app/experiments - Experiment results
# - /app/data - Input data
# - /app/checkpoints - Model checkpoints
```

## Volumes and Persistence

The following directories are persisted using Docker volumes:

```yaml
volumes:
  - ./experiments:/app/experiments  # Experiment results
  - ./logs:/app/logs                # Application logs
  - ./checkpoints:/app/checkpoints  # Model checkpoints
  - ./data:/app/data                # Input data
```

### Accessing Data

Data is stored on your host machine in these directories and persists across container restarts.

```bash
# View experiment results
ls experiments/

# View logs
tail -f logs/app.log

# Backup data
tar -czf backup.tar.gz experiments/ checkpoints/ data/
```

## Development

### Building Specific Stages

```bash
# Build production image
docker build --target production -t 4d-neural:prod .

# Build development image
docker build --target development -t 4d-neural:dev .
```

### Custom Configuration

Create a `.env` file for custom environment variables:

```bash
# .env
FLASK_ENV=development
LOG_LEVEL=DEBUG
DATABASE_PATH=experiments/custom.db
```

Then use it with docker-compose:

```bash
docker-compose --env-file .env up -d app
```

### Interactive Shell

```bash
# Start a bash shell in the container
docker-compose run --rm app bash

# Or in a running container
docker-compose exec app bash
```

## Troubleshooting

### Port Already in Use

If port 5000 is already in use:

```yaml
# Edit docker-compose.yml
ports:
  - "5050:5000"  # Use port 5050 instead
```

### Permission Issues

If you encounter permission issues with volumes:

```bash
# Fix ownership (Linux/Mac)
sudo chown -R $USER:$USER experiments/ logs/ checkpoints/ data/
```

### Container Won't Start

```bash
# Check logs
docker-compose logs app

# Rebuild without cache
docker-compose build --no-cache app

# Remove all containers and volumes
docker-compose down -v
docker-compose up -d app
```

### Out of Memory

Increase Docker's memory limit in Docker Desktop settings, or use resource limits:

```yaml
# docker-compose.yml
services:
  app:
    deploy:
      resources:
        limits:
          memory: 4G
```

### Network Issues

```bash
# Recreate network
docker-compose down
docker network prune
docker-compose up -d
```

## Advanced Usage

### Running Multiple Experiments

```bash
# Start multiple instances with different configs
docker-compose run --rm app python example.py --config config1.json
docker-compose run --rm app python example.py --config config2.json
```

### GPU Support (Optional)

If you have NVIDIA GPU support:

1. Install [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)

2. Modify docker-compose.yml:

```yaml
services:
  app:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Production Deployment

For production deployment:

```bash
# Use production compose file
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Or use Docker Swarm
docker stack deploy -c docker-compose.yml neural-cognition
```

## Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Project README](../README.md)
- [Installation Guide](user-guide/INSTALLATION.md)

## Support

For issues related to Docker setup:

1. Check the [troubleshooting section](#troubleshooting)
2. Review [Docker logs](#troubleshooting)
3. Open an issue on [GitHub](https://github.com/Thomas-Heisig/4D-Neural-Cognition/issues)
