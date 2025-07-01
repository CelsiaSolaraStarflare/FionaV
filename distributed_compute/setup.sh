#!/bin/bash

# Universal Distributed Computing System - Linux/macOS Setup Script

echo "Universal Distributed Computing System - Setup"
echo "============================================="

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo -e "\n${YELLOW}Checking prerequisites...${NC}"

missing_tools=()

if ! command_exists go; then
    missing_tools+=("Go")
    echo -e "  ${RED}[X] Go not found${NC}"
else
    echo -e "  ${GREEN}[✓] Go found: $(go version)${NC}"
fi

if ! command_exists docker; then
    echo -e "  ${YELLOW}[!] Docker not found (optional)${NC}"
else
    echo -e "  ${GREEN}[✓] Docker found: $(docker --version)${NC}"
fi

if ! command_exists nvcc; then
    echo -e "  ${YELLOW}[!] CUDA not found (optional for GPU support)${NC}"
else
    echo -e "  ${GREEN}[✓] CUDA found: $(nvcc --version | grep release)${NC}"
fi

# Check for C++ compiler
if command_exists g++; then
    echo -e "  ${GREEN}[✓] g++ found${NC}"
    CXX=g++
elif command_exists clang++; then
    echo -e "  ${GREEN}[✓] clang++ found${NC}"
    CXX=clang++
else
    missing_tools+=("C++ compiler")
    echo -e "  ${RED}[X] C++ compiler not found${NC}"
fi

if [ ${#missing_tools[@]} -gt 0 ]; then
    echo -e "\n${RED}Missing required tools: ${missing_tools[*]}${NC}"
    echo "Please install the missing tools and run this script again."
    
    if [[ " ${missing_tools[@]} " =~ " Go " ]]; then
        echo -e "\n${YELLOW}To install Go:${NC}"
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            echo -e "  ${CYAN}sudo apt-get install golang-go${NC}"
            echo -e "  ${CYAN}or download from: https://golang.org/dl/${NC}"
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            echo -e "  ${CYAN}brew install go${NC}"
            echo -e "  ${CYAN}or download from: https://golang.org/dl/${NC}"
        fi
    fi
    
    if [[ " ${missing_tools[@]} " =~ " C++ compiler " ]]; then
        echo -e "\n${YELLOW}To install C++ compiler:${NC}"
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            echo -e "  ${CYAN}sudo apt-get install build-essential${NC}"
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            echo -e "  ${CYAN}xcode-select --install${NC}"
        fi
    fi
    
    exit 1
fi

# Create bin directory
echo -e "\n${YELLOW}Creating directories...${NC}"
mkdir -p bin
echo -e "  ${GREEN}[✓] Created bin directory${NC}"

# Download Go dependencies
echo -e "\n${YELLOW}Downloading Go dependencies...${NC}"
if go mod download; then
    echo -e "  ${GREEN}[✓] Go dependencies downloaded${NC}"
else
    echo -e "  ${RED}[X] Failed to download Go dependencies${NC}"
    exit 1
fi

# Build components
echo -e "\n${YELLOW}Building components...${NC}"

# Build coordinator
echo -e "  ${CYAN}Building coordinator...${NC}"
if go build -o bin/coordinator src/coordinator/main.go; then
    echo -e "  ${GREEN}[✓] Coordinator built successfully${NC}"
else
    echo -e "  ${RED}[X] Failed to build coordinator${NC}"
fi

# Build worker
echo -e "  ${CYAN}Building worker...${NC}"
if go build -o bin/worker src/worker/main.go; then
    echo -e "  ${GREEN}[✓] Worker built successfully${NC}"
else
    echo -e "  ${RED}[X] Failed to build worker${NC}"
fi

# Build hardware detector
echo -e "  ${CYAN}Building hardware detector...${NC}"
CXXFLAGS="-std=c++17 -O3"
LDFLAGS=""

# Add platform-specific flags
if [[ "$OSTYPE" == "darwin"* ]]; then
    LDFLAGS="$LDFLAGS -framework IOKit -framework Foundation"
    if command_exists nvcc; then
        CXXFLAGS="$CXXFLAGS -DCUDA_ENABLED"
        LDFLAGS="$LDFLAGS -lcuda -lnvml"
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if command_exists nvcc; then
        CXXFLAGS="$CXXFLAGS -DCUDA_ENABLED"
        LDFLAGS="$LDFLAGS -lcuda -lnvml"
    fi
fi

if $CXX $CXXFLAGS src/detector/hardware_detector.cpp -o bin/hardware_detector $LDFLAGS 2>/dev/null; then
    echo -e "  ${GREEN}[✓] Hardware detector built successfully${NC}"
else
    echo -e "  ${YELLOW}[!] Hardware detector built without GPU support${NC}"
    $CXX $CXXFLAGS src/detector/hardware_detector.cpp -o bin/hardware_detector
fi

# Build CUDA kernels if CUDA is available
if command_exists nvcc; then
    echo -e "  ${CYAN}Building CUDA kernels...${NC}"
    if nvcc -ptx src/kernels/universal_kernels.cu -o bin/universal_kernels.ptx 2>/dev/null; then
        echo -e "  ${GREEN}[✓] CUDA kernels built successfully${NC}"
    else
        echo -e "  ${YELLOW}[!] Failed to build CUDA kernels${NC}"
    fi
fi

# Copy web files
echo -e "\n${YELLOW}Copying web files...${NC}"
cp -r src/web bin/
echo -e "  ${GREEN}[✓] Web files copied${NC}"

# Create start scripts
echo -e "\n${YELLOW}Creating start scripts...${NC}"

# Coordinator start script
cat > bin/start-coordinator.sh << 'EOF'
#!/bin/bash
echo "Starting UDCS Coordinator..."
cd "$(dirname "$0")"
./coordinator --port 8080 --web-port 8081
EOF
chmod +x bin/start-coordinator.sh

# Worker start script
cat > bin/start-worker.sh << 'EOF'
#!/bin/bash
echo "Starting UDCS Worker..."
read -p "Enter coordinator address (e.g., localhost:8080): " COORDINATOR
cd "$(dirname "$0")"
./worker --coordinator "$COORDINATOR"
EOF
chmod +x bin/start-worker.sh

echo -e "  ${GREEN}[✓] Start scripts created${NC}"

# Make binaries executable
chmod +x bin/coordinator bin/worker bin/hardware_detector 2>/dev/null

echo -e "\n============================================="
echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "\n${YELLOW}To start the system:${NC}"
echo -e "  1. Start coordinator: ${CYAN}./bin/start-coordinator.sh${NC}"
echo -e "  2. Start workers:     ${CYAN}./bin/start-worker.sh${NC}"
echo -e "  3. Open web UI:       ${CYAN}http://localhost:8081${NC}"
echo -e "\n${YELLOW}For Python client:${NC}"
echo -e "  ${CYAN}pip install requests numpy websockets${NC}"
echo -e "  ${CYAN}python client.py${NC}"

# Firewall reminder
echo -e "\n${YELLOW}Note:${NC} You may need to configure your firewall to allow ports 8080 and 8081"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo -e "  ${CYAN}sudo ufw allow 8080/tcp${NC}"
    echo -e "  ${CYAN}sudo ufw allow 8081/tcp${NC}"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "  macOS will prompt when the applications try to accept connections"
fi 