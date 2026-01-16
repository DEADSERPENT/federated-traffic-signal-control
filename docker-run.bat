@echo off
REM ==============================================================================
REM Federated Learning Traffic Signal Control - Docker Runner (Windows)
REM ==============================================================================
REM Usage:
REM   docker-run.bat demo          - Run quick demo
REM   docker-run.bat full          - Run full experiment
REM   docker-run.bat fl            - Run FL server + 4 clients
REM   docker-run.bat simulation    - Run traffic simulation only
REM   docker-run.bat cloudsim      - Run CloudSim simulation
REM   docker-run.bat build         - Build Docker image
REM   docker-run.bat clean         - Clean up containers
REM ==============================================================================

echo.
echo ======================================================================
echo   FEDERATED LEARNING TRAFFIC SIGNAL CONTROL - DOCKER
echo ======================================================================
echo.

if "%1"=="" goto help
if "%1"=="help" goto help
if "%1"=="demo" goto demo
if "%1"=="full" goto full
if "%1"=="fl" goto fl
if "%1"=="simulation" goto simulation
if "%1"=="cloudsim" goto cloudsim
if "%1"=="build" goto build
if "%1"=="clean" goto clean
goto help

:demo
echo [*] Running Quick Demo...
docker-compose up demo
goto end

:full
echo [*] Running Full Experiment (with scalability tests)...
docker-compose up full-experiment
goto end

:fl
echo [*] Starting Federated Learning (Server + 4 Clients)...
docker-compose up fl-server fl-client-0 fl-client-1 fl-client-2 fl-client-3
goto end

:simulation
echo [*] Running Traffic Simulation...
docker-compose up simulation
goto end

:cloudsim
echo [*] Running CloudSim Simulation...
docker-compose up cloudsim
goto end

:build
echo [*] Building Docker Image...
docker-compose build
goto end

:clean
echo [*] Cleaning up containers...
docker-compose down --remove-orphans
docker system prune -f
goto end

:help
echo.
echo Usage: docker-run.bat [command]
echo.
echo Commands:
echo   demo        Run quick comprehensive demo (~5 min)
echo   full        Run full experiment with scalability (~15 min)
echo   fl          Start FL server + 4 clients
echo   simulation  Run traffic simulation only
echo   cloudsim    Run CloudSim edge/cloud simulation
echo   build       Build Docker image
echo   clean       Clean up containers and images
echo   help        Show this help message
echo.
goto end

:end
echo.
echo ======================================================================
