# Obstacle Avoidance Service with HyRL

The simulation files for Hysteresis-Based Reinforcement Laerning, as introduced in Hysteresis-Based RL: Robustifying Reinforcement Learning-based Control Policies via Hybrid Control. J. de Priester, R. G. Sanfelice, and N. van de Wouw. Proceedings of the American Control Conference, June, 2022.

# Running the HyRL Server

## Pre-requisites

- Nix installed if you intend to run with Nix, else an operational container runtime, e.g. Docker, Podman. See a simple installer for Nix [here](https://determinate.systems/nix-installer/)
- [Docker](https://docs.docker.com/get-docker/) installed and running

## Quick Start

### Running the Server

```bash
# Run the server directly (builds Docker image if needed)
nix run github:friend0/ObstacleAvoidanceHyRL

# Or clone and run locally
git clone https://github.com/your-username/ObstacleAvoidanceHyRL
cd ObstacleAvoidanceHyRL
nix run .
```

The server will:

1. Build a Docker image if it doesn't exist
2. Start the container on port 50051
3. Expose the gRPC API for your simulation framework

### Building the Docker Image Only

```bash
# Build the Docker image without running
nix run .#hyrl-server-image
```

### Development Environment

```bash
# Enter development shell with all tools
nix develop

# This provides:
# - Python 3.13 with uv
# - Docker
# - Development tools (neovim, black, grpcurl, etc.)
```

## Integration with Simulation Framework

In your simulation framework's `flake.nix`, add this as an input:

```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    hyrl-server.url = "github:your-username/ObstacleAvoidanceHyRL";
  };

  outputs = { self, nixpkgs, hyrl-server }: {
    devShells.default = pkgs.mkShell {
      packages = [
        # Your simulation packages
        hyrl-server.packages.${system}.hyrl-server
      ];
    };
  };
}
```

Then users can run your complete simulation environment:

```bash
nix develop          # Enter environment with both simulation and server
hyrl-server &        # Start server in background
your-simulation      # Run your simulation
```

## Server Configuration

The server runs with these defaults:

- **Port**: 50051 (gRPC)
- **Models**: Pre-trained DQN models included
- **Environment**: Containerized Python environment with all dependencies

### Environment Variables

You can customize the server by passing environment variables to Docker:

```bash
# Example with custom configuration
docker run --rm -p 50051:50051 \
  -e PYTHONPATH=/app/src \
  -e CUSTOM_CONFIG=value \
  hyrl-server:latest
```

## Troubleshooting

### Docker Issues

```bash
# Check if Docker is running
docker info

# Rebuild the image
docker rmi hyrl-server:latest
nix run .
```

### Port Conflicts

```bash
# Use a different port
docker run --rm -p 8080:50051 hyrl-server:latest
```

### Development

```bash
# Run server locally without Docker
nix develop
cd src
python -m rl_policy.server
```

## Package Structure

- `flake.nix` - Nix flake definition
- `Dockerfile` - Container definition
- `src/` - Python source code
- `models/` - Pre-trained DQN models
- `protos/` - gRPC protocol definitions
