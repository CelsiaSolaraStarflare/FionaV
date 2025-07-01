# Makefile for Universal Distributed Computing System

GO_BUILD = go build
CXX = g++
NVCC = nvcc
NPM = npm
echo done
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