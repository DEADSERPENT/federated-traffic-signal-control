#!/bin/bash
#===============================================================================
# NS-3 Setup Script for WSL/Ubuntu
# Federated Learning Traffic Signal Control Project
#
# This script installs NS-3 and sets up the FL network simulation environment.
#
# Usage:
#   chmod +x setup_wsl_ns3.sh
#   ./setup_wsl_ns3.sh
#
# Requirements:
#   - WSL2 with Ubuntu 22.04 or Ubuntu 22.04 on separate machine
#   - At least 4GB RAM, 10GB disk space
#===============================================================================

set -e  # Exit on error

echo "============================================================"
echo "  NS-3 Setup for Federated Learning Traffic Control"
echo "============================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
NS3_VERSION="3.40"
WORKSPACE_DIR="$HOME/ns3-fl-traffic"
NS3_DIR="$WORKSPACE_DIR/ns-3.${NS3_VERSION}"

# Function to print status
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

#===============================================================================
# Step 1: System Update and Dependencies
#===============================================================================
echo ""
echo "Step 1: Installing system dependencies..."
echo "----------------------------------------"

sudo apt update && sudo apt upgrade -y

# Essential build tools
sudo apt install -y \
    build-essential \
    gcc \
    g++ \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    cmake \
    ninja-build \
    git \
    mercurial \
    wget \
    curl

# NS-3 specific dependencies
sudo apt install -y \
    qtbase5-dev \
    qtchooser \
    qt5-qmake \
    qtbase5-dev-tools \
    libqt5svg5-dev \
    gdb \
    valgrind \
    tcpdump \
    sqlite3 \
    libsqlite3-dev \
    libxml2-dev \
    libgtk-3-dev \
    libboost-all-dev \
    libgsl-dev \
    libfl-dev \
    bison \
    flex

# Python dependencies for NS-3 bindings
pip3 install --user \
    cppyy \
    numpy \
    pandas \
    matplotlib \
    pyzmq \
    scipy

print_status "System dependencies installed"

#===============================================================================
# Step 2: Create Workspace
#===============================================================================
echo ""
echo "Step 2: Creating workspace..."
echo "----------------------------------------"

mkdir -p $WORKSPACE_DIR
cd $WORKSPACE_DIR

print_status "Workspace created at $WORKSPACE_DIR"

#===============================================================================
# Step 3: Download NS-3
#===============================================================================
echo ""
echo "Step 3: Downloading NS-3 ${NS3_VERSION}..."
echo "----------------------------------------"

if [ -d "$NS3_DIR" ]; then
    print_warning "NS-3 directory already exists, skipping download"
else
    # Download from GitLab
    git clone --depth 1 --branch ns-${NS3_VERSION} \
        https://gitlab.com/nsnam/ns-3-dev.git ns-3.${NS3_VERSION}
    print_status "NS-3 downloaded"
fi

cd $NS3_DIR

#===============================================================================
# Step 4: Configure NS-3
#===============================================================================
echo ""
echo "Step 4: Configuring NS-3..."
echo "----------------------------------------"

# Configure with optimizations and Python bindings
./ns3 configure \
    --enable-examples \
    --enable-tests \
    --enable-python-bindings \
    --build-profile=optimized

print_status "NS-3 configured"

#===============================================================================
# Step 5: Build NS-3
#===============================================================================
echo ""
echo "Step 5: Building NS-3 (this may take 15-30 minutes)..."
echo "----------------------------------------"

# Build with parallel jobs
./ns3 build -j$(nproc)

print_status "NS-3 built successfully"

#===============================================================================
# Step 6: Verify Installation
#===============================================================================
echo ""
echo "Step 6: Verifying installation..."
echo "----------------------------------------"

# Run hello-simulator to verify
./ns3 run hello-simulator

print_status "NS-3 installation verified"

#===============================================================================
# Step 7: Create FL Traffic Module Directory
#===============================================================================
echo ""
echo "Step 7: Setting up FL Traffic Module..."
echo "----------------------------------------"

# Create scratch directory for our module
mkdir -p $NS3_DIR/scratch/fl-traffic

# Create symlink to Windows project (if on WSL)
if [ -d "/mnt/c/Users" ]; then
    # Find the Windows project directory
    WIN_PROJECT="/mnt/c/Users/AKSHAY/Music/TRAFFIC SIGNALS"
    if [ -d "$WIN_PROJECT" ]; then
        ln -sf "$WIN_PROJECT" $WORKSPACE_DIR/fl-traffic-windows
        print_status "Created symlink to Windows project"
    fi
fi

print_status "FL Traffic module directory created"

#===============================================================================
# Step 8: Create Python Virtual Environment
#===============================================================================
echo ""
echo "Step 8: Creating Python environment..."
echo "----------------------------------------"

cd $WORKSPACE_DIR
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install numpy pandas matplotlib pyzmq scipy torch

print_status "Python environment created"

#===============================================================================
# Step 9: Create Configuration File
#===============================================================================
echo ""
echo "Step 9: Creating configuration..."
echo "----------------------------------------"

cat > $WORKSPACE_DIR/config.env << 'EOF'
# NS-3 FL Traffic Configuration
export NS3_DIR="$HOME/ns3-fl-traffic/ns-3.3.40"
export FL_TRAFFIC_DIR="$HOME/ns3-fl-traffic"
export PYTHONPATH="$NS3_DIR/build/bindings/python:$PYTHONPATH"

# Activate environment
activate_ns3() {
    source $FL_TRAFFIC_DIR/venv/bin/activate
    cd $NS3_DIR
    echo "NS-3 environment activated"
}

# Run NS-3 simulation
run_ns3() {
    cd $NS3_DIR
    ./ns3 run "$@"
}
EOF

# Add to .bashrc
echo "" >> ~/.bashrc
echo "# NS-3 FL Traffic Configuration" >> ~/.bashrc
echo "source $WORKSPACE_DIR/config.env" >> ~/.bashrc

print_status "Configuration created"

#===============================================================================
# Complete
#===============================================================================
echo ""
echo "============================================================"
echo -e "${GREEN}  NS-3 SETUP COMPLETE!${NC}"
echo "============================================================"
echo ""
echo "Workspace: $WORKSPACE_DIR"
echo "NS-3 Directory: $NS3_DIR"
echo ""
echo "Next steps:"
echo "  1. Source the config: source ~/.bashrc"
echo "  2. Copy FL traffic module: cp fl-traffic-module.cc $NS3_DIR/scratch/fl-traffic/"
echo "  3. Run simulation: ./ns3 run scratch/fl-traffic/fl-traffic-module"
echo ""
echo "To connect with Windows:"
echo "  1. Start the NS-3 bridge server: python3 ns3_bridge_server.py"
echo "  2. Run FL training from Windows with NS-3 integration"
echo ""
