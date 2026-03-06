#!/bin/bash
# ==============================================================================
# ResilNet-FL — Docker Runner (Linux / macOS / WSL)
# ==============================================================================
# Usage (from anywhere):
#   ./docker/docker-run.sh demo          Run quick comprehensive demo (~5 min)
#   ./docker/docker-run.sh ieee          Run full IEEE experiments (~20 min)
#   ./docker/docker-run.sh byzantine     Run Byzantine robustness test
#   ./docker/docker-run.sh fl            Start FL server + 9 clients (3x3 grid)
#   ./docker/docker-run.sh simulation    Run traffic simulation only
#   ./docker/docker-run.sh cloudsim      Run CloudSim edge simulation
#   ./docker/docker-run.sh build         Build Docker image
#   ./docker/docker-run.sh clean         Remove containers and images
# ==============================================================================

set -e

# Always run from project root regardless of where this script is called from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

COMPOSE="docker compose -f docker/docker-compose.yml"

echo ""
echo "======================================================================"
echo "  ResilNet-FL — DOCKER RUNNER"
echo "  Project root: $PROJECT_ROOT"
echo "======================================================================"
echo ""

show_help() {
    echo "Usage: docker/docker-run.sh [command]"
    echo ""
    echo "Commands:"
    echo "  demo        Quick comprehensive demo (~5 min)"
    echo "  ieee        Full IEEE publication experiments (~20 min)"
    echo "  byzantine   Byzantine robustness evaluation"
    echo "  fl          FL server + 9 clients (3x3 grid)"
    echo "  simulation  Traffic simulation only"
    echo "  cloudsim    CloudSim edge/cloud simulation"
    echo "  build       Build Docker image"
    echo "  clean       Remove containers and free disk space"
    echo "  help        Show this message"
    echo ""
    echo "Note: SUMO-GUI requires a local SUMO install — not available in Docker."
    echo "      Run  python scripts/run_sumo_gui.py  directly on your machine."
    echo ""
}

case "$1" in
    demo)
        echo "[*] Running Quick Demo ..."
        $COMPOSE up demo
        ;;
    ieee)
        echo "[*] Running IEEE Experiments ..."
        $COMPOSE up ieee
        ;;
    byzantine)
        echo "[*] Running Byzantine Robustness Test ..."
        $COMPOSE up byzantine
        ;;
    fl)
        echo "[*] Starting FL Server + 9 Clients ..."
        $COMPOSE up fl-server \
            fl-client-0 fl-client-1 fl-client-2 \
            fl-client-3 fl-client-4 fl-client-5 \
            fl-client-6 fl-client-7 fl-client-8
        ;;
    simulation)
        echo "[*] Running Traffic Simulation ..."
        $COMPOSE up simulation
        ;;
    cloudsim)
        echo "[*] Running CloudSim Simulation ..."
        $COMPOSE up cloudsim
        ;;
    build)
        echo "[*] Building Docker Image ..."
        $COMPOSE build
        ;;
    clean)
        echo "[*] Cleaning up containers and images ..."
        $COMPOSE down --remove-orphans
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
