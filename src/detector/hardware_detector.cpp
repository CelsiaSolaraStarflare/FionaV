#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <thread>
#include <chrono>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#include <sysinfoapi.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <IOKit/IOKitLib.h>
#elif defined(__linux__)
#include <sys/utsname.h>
#include <unistd.h>
#include <fstream>
#include <sstream>
#endif

// CUDA detection
#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#include <nvml.h>
#endif

// ROCm detection  
#ifdef ROCM_ENABLED
#include <hip/hip_runtime.h>
#include <rocm_smi/rocm_smi.h>
#endif

// OpenCL detection
#ifdef OPENCL_ENABLED
#include <CL/cl.h>
#endif

// Metal detection (macOS only)
#ifdef __APPLE__
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>
#endif

struct GPUInfo {
    std::string name;
    std::string type; // "cuda", "rocm", "metal", "opencl"
    size_t memory_bytes;
    std::string compute_capability;
    double utilization;
    int device_id;
};

struct SystemInfo {
    std::string hostname;
    std::string architecture;
    std::string os;
    int cpu_cores;
    size_t ram_bytes;
    std::vector<GPUInfo> gpus;
    std::vector<std::string> frameworks;
    std::map<std::string, std::string> capabilities;
};

class HardwareDetector {
public:
    SystemInfo detectSystem() {
        SystemInfo info;
        
        info.hostname = getHostname();
        info.architecture = getArchitecture();
        info.os = getOperatingSystem();
        info.cpu_cores = getCPUCores();
        info.ram_bytes = getRAMSize();
        info.gpus = detectGPUs();
        info.frameworks = detectFrameworks();
        info.capabilities = detectCapabilities();
        
        return info;
    }

private:
    std::string getHostname() {
#ifdef _WIN32
        char hostname[256];
        DWORD size = sizeof(hostname);
        if (GetComputerNameA(hostname, &size)) {
            return std::string(hostname);
        }
#else
        char hostname[256];
        if (gethostname(hostname, sizeof(hostname)) == 0) {
            return std::string(hostname);
        }
#endif
        return "unknown";
    }

    std::string getArchitecture() {
#ifdef _WIN32
        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);
        switch (sysInfo.wProcessorArchitecture) {
            case PROCESSOR_ARCHITECTURE_AMD64:
                return "x86_64";
            case PROCESSOR_ARCHITECTURE_ARM64:
                return "arm64";
            case PROCESSOR_ARCHITECTURE_INTEL:
                return "x86";
            default:
                return "unknown";
        }
#elif defined(__APPLE__)
        size_t size = 0;
        sysctlbyname("hw.targettype", nullptr, &size, nullptr, 0);
        if (size > 0) {
            std::vector<char> buffer(size);
            sysctlbyname("hw.targettype", buffer.data(), &size, nullptr, 0);
            std::string target(buffer.data());
            if (target.find("arm64") != std::string::npos) {
                return "arm64";
            }
        }
        return "x86_64"; // Default for Intel Macs
#elif defined(__linux__)
        struct utsname unameData;
        if (uname(&unameData) == 0) {
            return std::string(unameData.machine);
        }
#endif
        return "unknown";
    }

    std::string getOperatingSystem() {
#ifdef _WIN32
        return "windows";
#elif defined(__APPLE__)
        return "darwin";
#elif defined(__linux__)
        return "linux";
#elif defined(__FreeBSD__)
        return "freebsd";
#else
        return "unknown";
#endif
    }

    int getCPUCores() {
        return std::thread::hardware_concurrency();
    }

    size_t getRAMSize() {
#ifdef _WIN32
        MEMORYSTATUSEX statex;
        statex.dwLength = sizeof(statex);
        if (GlobalMemoryStatusEx(&statex)) {
            return statex.ullTotalPhys;
        }
#elif defined(__APPLE__)
        int64_t ram_size;
        size_t size = sizeof(ram_size);
        if (sysctlbyname("hw.memsize", &ram_size, &size, nullptr, 0) == 0) {
            return static_cast<size_t>(ram_size);
        }
#elif defined(__linux__)
        std::ifstream meminfo("/proc/meminfo");
        std::string line;
        while (std::getline(meminfo, line)) {
            if (line.find("MemTotal:") == 0) {
                std::istringstream iss(line);
                std::string label, unit;
                size_t value;
                iss >> label >> value >> unit;
                return value * 1024; // Convert KB to bytes
            }
        }
#endif
        return 0;
    }

    std::vector<GPUInfo> detectGPUs() {
        std::vector<GPUInfo> gpus;
        
        // Detect CUDA devices
#ifdef CUDA_ENABLED
        auto cudaGPUs = detectCUDAGPUs();
        gpus.insert(gpus.end(), cudaGPUs.begin(), cudaGPUs.end());
#endif

        // Detect ROCm devices
#ifdef ROCM_ENABLED
        auto rocmGPUs = detectROCmGPUs();
        gpus.insert(gpus.end(), rocmGPUs.begin(), rocmGPUs.end());
#endif

        // Detect Metal devices (macOS)
#ifdef __APPLE__
        auto metalGPUs = detectMetalGPUs();
        gpus.insert(gpus.end(), metalGPUs.begin(), metalGPUs.end());
#endif

        // Detect OpenCL devices
#ifdef OPENCL_ENABLED
        auto openclGPUs = detectOpenCLGPUs();
        gpus.insert(gpus.end(), openclGPUs.begin(), openclGPUs.end());
#endif

        return gpus;
    }

#ifdef CUDA_ENABLED
    std::vector<GPUInfo> detectCUDAGPUs() {
        std::vector<GPUInfo> gpus;
        
        int deviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        
        if (error != cudaSuccess) {
            return gpus;
        }

        // Initialize NVML for utilization monitoring
        nvmlReturn_t nvmlResult = nvmlInit();
        bool nvmlAvailable = (nvmlResult == NVML_SUCCESS);

        for (int i = 0; i < deviceCount; ++i) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            
            GPUInfo gpu;
            gpu.name = prop.name;
            gpu.type = "cuda";
            gpu.memory_bytes = prop.totalGlobalMem;
            gpu.compute_capability = std::to_string(prop.major) + "." + std::to_string(prop.minor);
            gpu.device_id = i;
            
            // Get utilization if NVML is available
            if (nvmlAvailable) {
                nvmlDevice_t device;
                if (nvmlDeviceGetHandleByIndex(i, &device) == NVML_SUCCESS) {
                    nvmlUtilization_t utilization;
                    if (nvmlDeviceGetUtilizationRates(device, &utilization) == NVML_SUCCESS) {
                        gpu.utilization = utilization.gpu;
                    }
                }
            }
            
            gpus.push_back(gpu);
        }
        
        if (nvmlAvailable) {
            nvmlShutdown();
        }
        
        return gpus;
    }
#endif

#ifdef ROCM_ENABLED
    std::vector<GPUInfo> detectROCmGPUs() {
        std::vector<GPUInfo> gpus;
        
        int deviceCount = 0;
        hipError_t error = hipGetDeviceCount(&deviceCount);
        
        if (error != hipSuccess) {
            return gpus;
        }

        // Initialize ROCm SMI
        rsmi_status_t ret = rsmi_init(0);
        bool rsmiAvailable = (ret == RSMI_STATUS_SUCCESS);

        for (int i = 0; i < deviceCount; ++i) {
            hipDeviceProp_t prop;
            hipGetDeviceProperties(&prop, i);
            
            GPUInfo gpu;
            gpu.name = prop.name;
            gpu.type = "rocm";
            gpu.memory_bytes = prop.totalGlobalMem;
            gpu.device_id = i;
            
            // Get utilization if ROCm SMI is available
            if (rsmiAvailable) {
                uint32_t busy_percent;
                if (rsmi_dev_busy_percent_get(i, &busy_percent) == RSMI_STATUS_SUCCESS) {
                    gpu.utilization = busy_percent;
                }
            }
            
            gpus.push_back(gpu);
        }
        
        if (rsmiAvailable) {
            rsmi_shut_down();
        }
        
        return gpus;
    }
#endif

#ifdef __APPLE__
    std::vector<GPUInfo> detectMetalGPUs() {
        std::vector<GPUInfo> gpus;
        
        @autoreleasepool {
            NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
            
            for (id<MTLDevice> device in devices) {
                GPUInfo gpu;
                gpu.name = [device.name UTF8String];
                gpu.type = "metal";
                gpu.memory_bytes = device.recommendedMaxWorkingSetSize;
                gpu.device_id = static_cast<int>(gpus.size());
                gpu.utilization = 0.0; // Metal doesn't provide easy utilization access
                
                gpus.push_back(gpu);
            }
        }
        
        return gpus;
    }
#endif

#ifdef OPENCL_ENABLED
    std::vector<GPUInfo> detectOpenCLGPUs() {
        std::vector<GPUInfo> gpus;
        
        cl_uint platformCount;
        clGetPlatformIDs(0, nullptr, &platformCount);
        
        if (platformCount == 0) {
            return gpus;
        }
        
        std::vector<cl_platform_id> platforms(platformCount);
        clGetPlatformIDs(platformCount, platforms.data(), nullptr);
        
        for (auto platform : platforms) {
            cl_uint deviceCount;
            clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceCount);
            
            if (deviceCount == 0) continue;
            
            std::vector<cl_device_id> devices(deviceCount);
            clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, deviceCount, devices.data(), nullptr);
            
            for (auto device : devices) {
                GPUInfo gpu;
                gpu.type = "opencl";
                
                char name[256];
                clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
                gpu.name = name;
                
                cl_ulong memSize;
                clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(memSize), &memSize, nullptr);
                gpu.memory_bytes = memSize;
                
                gpu.device_id = static_cast<int>(gpus.size());
                gpu.utilization = 0.0;
                
                gpus.push_back(gpu);
            }
        }
        
        return gpus;
    }
#endif

    std::vector<std::string> detectFrameworks() {
        std::vector<std::string> frameworks;
        
        // Check for CUDA
#ifdef CUDA_ENABLED
        int cudaDevices;
        if (cudaGetDeviceCount(&cudaDevices) == cudaSuccess && cudaDevices > 0) {
            frameworks.push_back("cuda");
        }
#endif

        // Check for ROCm
#ifdef ROCM_ENABLED
        int rocmDevices;
        if (hipGetDeviceCount(&rocmDevices) == hipSuccess && rocmDevices > 0) {
            frameworks.push_back("rocm");
        }
#endif

        // Check for OpenCL
#ifdef OPENCL_ENABLED
        cl_uint platformCount;
        if (clGetPlatformIDs(0, nullptr, &platformCount) == CL_SUCCESS && platformCount > 0) {
            frameworks.push_back("opencl");
        }
#endif

        // Check for Metal (macOS)
#ifdef __APPLE__
        frameworks.push_back("metal");
#endif

        // Check for Docker
        if (checkCommandAvailable("docker")) {
            frameworks.push_back("docker");
        }
        
        // Check for container runtimes
        if (checkCommandAvailable("podman")) {
            frameworks.push_back("podman");
        }

        return frameworks;
    }

    std::map<std::string, std::string> detectCapabilities() {
        std::map<std::string, std::string> capabilities;
        
        // Check for WASM runtime
        if (checkCommandAvailable("wasmtime")) {
            capabilities["wasm"] = "wasmtime";
        } else if (checkCommandAvailable("wasmer")) {
            capabilities["wasm"] = "wasmer";
        }
        
        // Check for compilation tools
        if (checkCommandAvailable("gcc")) {
            capabilities["compiler"] = "gcc";
        } else if (checkCommandAvailable("clang")) {
            capabilities["compiler"] = "clang";
        } else if (checkCommandAvailable("cl")) {
            capabilities["compiler"] = "msvc";
        }
        
        return capabilities;
    }

    bool checkCommandAvailable(const std::string& command) {
#ifdef _WIN32
        std::string cmd = "where " + command + " >nul 2>&1";
        return system(cmd.c_str()) == 0;
#else
        std::string cmd = "which " + command + " >/dev/null 2>&1";
        return system(cmd.c_str()) == 0;
#endif
    }
};

// C interface for Go binding
extern "C" {
    typedef struct {
        char* json_data;
    } CSystemInfo;

    CSystemInfo* detect_hardware() {
        HardwareDetector detector;
        SystemInfo info = detector.detectSystem();
        
        // Convert to JSON
        std::stringstream json;
        json << "{";
        json << "\"hostname\":\"" << info.hostname << "\",";
        json << "\"architecture\":\"" << info.architecture << "\",";
        json << "\"os\":\"" << info.os << "\",";
        json << "\"cpu_cores\":" << info.cpu_cores << ",";
        json << "\"ram_bytes\":" << info.ram_bytes << ",";
        
        // GPUs array
        json << "\"gpus\":[";
        for (size_t i = 0; i < info.gpus.size(); ++i) {
            if (i > 0) json << ",";
            json << "{";
            json << "\"name\":\"" << info.gpus[i].name << "\",";
            json << "\"type\":\"" << info.gpus[i].type << "\",";
            json << "\"memory_bytes\":" << info.gpus[i].memory_bytes << ",";
            json << "\"compute_capability\":\"" << info.gpus[i].compute_capability << "\",";
            json << "\"utilization\":" << info.gpus[i].utilization;
            json << "}";
        }
        json << "],";
        
        // Frameworks array
        json << "\"frameworks\":[";
        for (size_t i = 0; i < info.frameworks.size(); ++i) {
            if (i > 0) json << ",";
            json << "\"" << info.frameworks[i] << "\"";
        }
        json << "]";
        
        json << "}";
        
        CSystemInfo* cInfo = new CSystemInfo();
        std::string jsonStr = json.str();
        cInfo->json_data = new char[jsonStr.length() + 1];
        strcpy(cInfo->json_data, jsonStr.c_str());
        
        return cInfo;
    }

    void free_system_info(CSystemInfo* info) {
        if (info) {
            delete[] info->json_data;
            delete info;
        }
    }
}

// Main function for testing
int main() {
    HardwareDetector detector;
    SystemInfo info = detector.detectSystem();
    
    std::cout << "System Information:" << std::endl;
    std::cout << "Hostname: " << info.hostname << std::endl;
    std::cout << "Architecture: " << info.architecture << std::endl;
    std::cout << "OS: " << info.os << std::endl;
    std::cout << "CPU Cores: " << info.cpu_cores << std::endl;
    std::cout << "RAM: " << (info.ram_bytes / (1024*1024*1024)) << " GB" << std::endl;
    
    std::cout << "\nGPUs:" << std::endl;
    for (const auto& gpu : info.gpus) {
        std::cout << "  " << gpu.name << " (" << gpu.type << ")" << std::endl;
        std::cout << "    Memory: " << (gpu.memory_bytes / (1024*1024*1024)) << " GB" << std::endl;
        if (!gpu.compute_capability.empty()) {
            std::cout << "    Compute: " << gpu.compute_capability << std::endl;
        }
        std::cout << "    Utilization: " << gpu.utilization << "%" << std::endl;
    }
    
    std::cout << "\nFrameworks:" << std::endl;
    for (const auto& framework : info.frameworks) {
        std::cout << "  " << framework << std::endl;
    }
    
    return 0;
} 