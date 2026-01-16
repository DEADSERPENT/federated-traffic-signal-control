#!/bin/bash
# ==============================================================================
# Federated Learning Traffic Signal Control - Docker Runner (Linux/Mac)
# ==============================================================================
# Usage:
#   ./docker-run.sh demo          - Run quick demo
#   ./docker-run.sh full          - Run full experiment
#   ./docker-run.sh fl            - Run FL server + 4 clients
#   ./docker-run.sh simulation    - Run traffic simulation only
#   ./docker-run.sh cloudsim      - Run CloudSim simulation
#   ./docker-run.sh build         - Build Docker image
#   ./docker-run.sh clean         - Clean up containers
# ==============================================================================

set -e

echo ""
echo "======================================================================"
echo "  FEDERATED LEARNING TRAFFIC SIGNAL CONTROL - DOCKER"
echo "======================================================================"
echo ""

show_help() {
    echo "Usage: ./docker-run.sh [command]"
    echo ""
    echo "Commands:"
    echo "  demo        Run quick comprehensive demo (~5 min)"
    echo "  full        Run full experiment with scalability (~15 min)"
    echo "  fl          Start FL server + 4 clients"
    echo "  simulation  Run traffic simulation only"
    echo "  cloudsim    Run CloudSim edge/cloud simulation"
    echo "  build       Build Docker image"
    echo "  clean       Clean up containers and images"
    echo "  help        Show this help message"
    echo ""
}

case "$1" in
    demo)
        echo "[*] Running Quick Demo..."
        docker-compose up demo
        ;;
    full)
        echo "[*] Running Full Experiment (with scalability tests)..."
        docker-compose up full-experiment
        ;;
    fl)
        echo "[*] Starting Federated Learning (Server + 4 Clients)..."
        docker-compose up fl-server fl-client-0 fl-client-1 fl-client-2 fl-client-3
        ;;
    simulation)
        echo "[*] Running Traffic Simulation..."
        docker-compose up simulation
        ;;
    cloudsim)
        echo "[*] Running CloudSim Simulation..."
        docker-compose up cloudsim
        ;;
    build)
        echo "[*] Building Docker Image..."
        docker-compose build
        ;;
    clean)
        echo "[*] Cleaning up containers..."
        docker-compose down --remove-orphans
        docker system prune -f
        ;;
    help|--help|-h|"")
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        show_help
        exit 1
        ;;
esac

echo ""
echo "======================================================================"
