@echo off
REM ==============================================================================
REM ResilNet-FL — Docker Runner (Windows)
REM ==============================================================================
REM Usage (from anywhere):
REM   docker\docker-run.bat demo          Run quick comprehensive demo (~5 min)
REM   docker\docker-run.bat ieee          Run full IEEE experiments (~20 min)
REM   docker\docker-run.bat byzantine     Run Byzantine robustness test
REM   docker\docker-run.bat fl            Start FL server + 9 clients (3x3 grid)
REM   docker\docker-run.bat simulation    Run traffic simulation only
REM   docker\docker-run.bat cloudsim      Run CloudSim edge simulation
REM   docker\docker-run.bat build         Build Docker image
REM   docker\docker-run.bat clean         Remove containers and images
REM ==============================================================================

REM Always run from project root (parent of docker\)
cd /d "%~dp0.."

echo.
echo ======================================================================
echo   ResilNet-FL -- DOCKER RUNNER
echo   Project root: %CD%
echo ======================================================================
echo.

if "%1"==""         goto help
if "%1"=="help"     goto help
if "%1"=="demo"     goto demo
if "%1"=="ieee"     goto ieee
if "%1"=="byzantine" goto byzantine
if "%1"=="fl"       goto fl
if "%1"=="simulation" goto simulation
if "%1"=="cloudsim" goto cloudsim
if "%1"=="build"    goto build
if "%1"=="clean"    goto clean
goto help

:demo
echo [*] Running Quick Demo ...
docker compose -f docker/docker-compose.yml up demo
goto end

:ieee
echo [*] Running IEEE Experiments ...
docker compose -f docker/docker-compose.yml up ieee
goto end

:byzantine
echo [*] Running Byzantine Robustness Test ...
docker compose -f docker/docker-compose.yml up byzantine
goto end

:fl
echo [*] Starting FL Server + 9 Clients ...
docker compose -f docker/docker-compose.yml up fl-server ^
    fl-client-0 fl-client-1 fl-client-2 ^
    fl-client-3 fl-client-4 fl-client-5 ^
    fl-client-6 fl-client-7 fl-client-8
goto end

:simulation
echo [*] Running Traffic Simulation ...
docker compose -f docker/docker-compose.yml up simulation
goto end

:cloudsim
echo [*] Running CloudSim Simulation ...
docker compose -f docker/docker-compose.yml up cloudsim
goto end

:build
echo [*] Building Docker Image ...
docker compose -f docker/docker-compose.yml build
goto end

:clean
echo [*] Cleaning up containers and images ...
docker compose -f docker/docker-compose.yml down --remove-orphans
docker system prune -f
goto end

:help
echo.
echo Usage: docker\docker-run.bat [command]
echo.
echo Commands:
echo   demo        Quick comprehensive demo (~5 min)
echo   ieee        Full IEEE publication experiments (~20 min)
echo   byzantine   Byzantine robustness evaluation
echo   fl          FL server + 9 clients (3x3 grid)
echo   simulation  Traffic simulation only
echo   cloudsim    CloudSim edge/cloud simulation
echo   build       Build Docker image
echo   clean       Remove containers and free disk space
echo   help        Show this message
echo.
echo Note: SUMO-GUI requires a local SUMO install -- not available in Docker.
echo       Run  python scripts/run_sumo_gui.py  directly on your machine.
echo.
goto end

:end
echo.
echo ======================================================================
