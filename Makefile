# Makefile for Universal Distributed Computing System

GO_BUILD = go build
CXX = g++
NVCC = nvcc
NPM = npm
BIN_DIR = bin

.PHONY: build-all build-coordinator build-detector build-kernels build-worker build-web clean

build-all: build-coordinator build-detector build-kernels build-worker build-web

build-coordinator:
	$(GO_BUILD) -o $(BIN_DIR)/coordinator src/coordinator/main.go

build-detector:
	$(CXX) -std=c++17 -O3 src/detector/hardware_detector.cpp -shared -fPIC -o $(BIN_DIR)/libhardware_detector.so \
	    -lcuda -lnvml -lrocm_smi -lOpenCL \
	    $(shell uname | grep -q Darwin && echo "-framework Metal")

build-kernels:
	$(NVCC) -ptx src/kernels/universal_kernels.cu -o $(BIN_DIR)/universal_kernels.ptx

build-worker:
	$(GO_BUILD) -o $(BIN_DIR)/worker src/worker/main.go

build-web:
	cd src/web && $(NPM) install && $(NPM) run build && cp -r build ../../bin/web

clean:
	rm -rf $(BIN_DIR)/*

# Runtime commands
run-coordinator: build-coordinator
	./$(BIN_DIR)/coordinator --port 8080 --web-port 8081

run-worker:
	@echo "Starting worker..."
	@$(WORKER_BIN) --coordinator $(COORDINATOR_HOST):$(COORDINATOR_PORT)

run-demo:
	@echo "Running Python demo script..."
	@python examples/demo.py

run-gui:
	@echo "Starting UDCS GUI Starter..."
	@python gui_starter.py

test-build:
	python test_build.py

start-system: build-all
	@echo "Starting Universal Distributed Computing System..."
	@echo "Coordinator will start on http://localhost:8080"
	@echo "Web interface will be available at http://localhost:8081"
	@echo ""
	@echo "In separate terminals, run:"
	@echo "  make run-worker    # Start worker nodes"
	@echo "  make run-demo      # Run demo tasks"
	@echo ""
	./$(BIN_DIR)/coordinator --port 8080 --web-port 8081

deploy-dashboard:
	@echo "🌐 Deploying UDCS Dashboard for online hosting..."
	@echo ""
	@echo "Static Dashboard Available at:"
	@echo "  deploy/static-dashboard.html"
	@echo ""
	@echo "Quick Deploy Options:"
	@echo "  📱 GitHub Pages: Copy static-dashboard.html to repo root as index.html"
	@echo "  🚀 Netlify: Drag and drop static-dashboard.html to netlify.com"
	@echo "  ⚡ Vercel: Import project to vercel.com"
	@echo ""
	@echo "Access examples:"
	@echo "  https://username.github.io/repo-name"
	@echo "  https://app-name.netlify.app"
	@echo "  https://project-name.vercel.app"

deploy-docker:
	@echo "🐳 Starting Docker deployment..."
	cd deploy && docker-compose up -d
	@echo ""
	@echo "Services started:"
	@echo "  🎛️  Dashboard: http://localhost:3000"
	@echo "  🖥️  Coordinator: http://localhost:8080"
	@echo "  🌐 Web Interface: http://localhost:8081"
	@echo ""
	@echo "To stop: cd deploy && docker-compose down"

help:
	@echo "Universal Distributed Computing System - Build Commands"
	@echo "======================================================="
	@echo ""
	@echo "Build Commands:"
	@echo "  make build-all         - Build all components"
	@echo "  make build-coordinator - Build coordinator server"
	@echo "  make build-worker      - Build worker node"
	@echo "  make build-detector    - Build hardware detector"
	@echo "  make build-kernels     - Build GPU kernels"
	@echo "  make build-web         - Prepare web interface"
	@echo ""
	@echo "Runtime Commands:"
	@echo "  make start-system      - Start coordinator (interactive)"
	@echo "  make run-coordinator   - Start coordinator"
	@echo "  make run-worker        - Start worker node"
	@echo "  make run-demo          - Run demo script"
	@echo ""
	@echo "Deployment Commands:"
	@echo "  make deploy-dashboard  - Deploy dashboard online"
	@echo "  make deploy-docker     - Deploy with Docker"
	@echo ""
	@echo "Utility Commands:"
	@echo "  make test-build        - Test build system"
	@echo "  make clean             - Clean build artifacts"
	@echo "  make help              - Show this help"

.PHONY: build-all build-coordinator build-detector build-kernels build-worker build-web clean run-coordinator run-worker run-demo test-build start-system help deploy-dashboard deploy-docker 