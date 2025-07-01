"""
Universal Distributed Computing System - Python Client Library

This client provides an easy-to-use interface for submitting various types of
compute tasks to the distributed system.
"""

import base64
import json
import requests
import numpy as np
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import asyncio
import websockets.client as websockets


class TaskType(Enum):
    CUDA_KERNEL = "CUDA_KERNEL"
    ROCM_KERNEL = "ROCM_KERNEL"
    METAL_KERNEL = "METAL_KERNEL"
    OPENCL_KERNEL = "OPENCL_KERNEL"
    NATIVE_EXEC = "NATIVE_EXEC"
    WASM = "WASM"
    DOCKER = "DOCKER"
    GAME_RENDER = "GAME_RENDER"
    ML_TRAINING = "ML_TRAINING"
    ML_INFERENCE = "ML_INFERENCE"
    VIDEO_ENCODE = "VIDEO_ENCODE"
    COMPILE = "COMPILE"
    ASSEMBLY = "ASSEMBLY"


class TaskClient:
    """Client for submitting tasks to the distributed compute system"""
    
    def __init__(self, coordinator_url: str):
        """
        Initialize the client
        
        Args:
            coordinator_url: URL of the coordinator (e.g., "http://192.168.1.100:8080")
        """
        self.coordinator_url = coordinator_url.rstrip('/')
        self.api_base = f"{self.coordinator_url}/api"
    
    def submit_task(self, task_type: Union[TaskType, str], 
                    payload: Any, 
                    requirements: Optional[Dict[str, Any]] = None,
                    priority: int = 5,
                    callback_url: Optional[str] = None) -> str:
        """
        Submit a task to the system
        
        Args:
            task_type: Type of task to execute
            payload: Task data (will be base64 encoded)
            requirements: Hardware/software requirements
            priority: Task priority (1-10, higher is more important)
            callback_url: Optional webhook for completion notification
            
        Returns:
            Task ID
        """
        if isinstance(task_type, TaskType):
            task_type = task_type.value
        
        # Encode payload
        if isinstance(payload, (dict, list)):
            payload_str = json.dumps(payload)
        elif isinstance(payload, np.ndarray):
            payload_str = json.dumps(payload.tolist())
        elif isinstance(payload, bytes):
            payload_str = payload.decode('utf-8')
        else:
            payload_str = str(payload)
        
        encoded_payload = base64.b64encode(payload_str.encode()).decode()
        
        task_data = {
            "type": task_type,
            "payload": encoded_payload,
            "priority": priority,
            "requirements": requirements or {}
        }
        
        if callback_url:
            task_data["callback_url"] = callback_url
        
        response = requests.post(f"{self.api_base}/submit", json=task_data)
        response.raise_for_status()
        
        return response.json()["task_id"]
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of a submitted task"""
        response = requests.get(f"{self.api_base}/task/{task_id}")
        response.raise_for_status()
        return response.json()
    
    def wait_for_completion(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """
        Wait for a task to complete and return the result
        
        Args:
            task_id: ID of the task
            timeout: Maximum time to wait in seconds
            
        Returns:
            Task result
        """
        import time
        start_time = time.time()
        
        while True:
            status = self.get_task_status(task_id)
            
            if status["status"] == "completed":
                return status.get("result")
            elif status["status"] == "failed":
                raise Exception(f"Task failed: {status.get('error_msg', 'Unknown error')}")
            
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
            
            time.sleep(1)
    
    def list_devices(self) -> List[Dict[str, Any]]:
        """Get list of available compute devices"""
        response = requests.get(f"{self.api_base}/devices")
        response.raise_for_status()
        return response.json()["devices"]
    
    # Convenience methods for specific task types
    
    def submit_cuda_kernel(self, task_data_or_kernel_code: Union[Dict[str, Any], str], 
                          data: Optional[Dict[str, Any]] = None, 
                          grid_size: Optional[List[int]] = None, 
                          block_size: Optional[List[int]] = None,
                          compute_capability: Optional[str] = None) -> str:
        """Submit a custom CUDA kernel for execution"""
        requirements = {}
        if compute_capability:
            requirements["compute_capability"] = compute_capability
        
        if isinstance(task_data_or_kernel_code, dict):
            # Accept full task data object
            payload = task_data_or_kernel_code
        else:
            # Accept individual parameters
            payload = {
                "kernel_code": task_data_or_kernel_code,
                "data": data,
                "grid_size": grid_size,
                "block_size": block_size
            }
        
        return self.submit_task(TaskType.CUDA_KERNEL, payload, requirements)
    
    def submit_game_render(self, game_state: bytes, resolution: List[int],
                          quality: str = "high", split_method: str = "tiles") -> str:
        """Submit a game rendering task"""
        payload = {
            "game_state": base64.b64encode(game_state).decode(),
            "render_settings": {
                "resolution": resolution,
                "quality": quality,
                "split_method": split_method
            }
        }
        
        requirements = {
            "gpu_required": True,
            "min_memory_gb": 4
        }
        
        return self.submit_task(TaskType.GAME_RENDER, payload, requirements)
    
    def submit_ml_training(self, model_code: str, training_data: np.ndarray,
                          epochs: int, batch_size: int, 
                          distributed: bool = True) -> str:
        """Submit a machine learning training task"""
        payload = {
            "model_code": model_code,
            "training_data": training_data.tolist(),
            "epochs": epochs,
            "batch_size": batch_size,
            "distributed": distributed
        }
        
        requirements = {
            "gpu_required": True,
            "frameworks": ["tensorflow", "pytorch"]
        }
        
        return self.submit_task(TaskType.ML_TRAINING, payload, requirements, priority=7)
    
    def submit_video_encode(self, video_data: bytes, codec: str = "h264",
                           quality: int = 75, resolution: Optional[List[int]] = None) -> str:
        """Submit a video encoding task"""
        payload = {
            "video_data": base64.b64encode(video_data).decode(),
            "codec": codec,
            "quality": quality
        }
        
        if resolution:
            payload["resolution"] = resolution
        
        return self.submit_task(TaskType.VIDEO_ENCODE, payload)
    
    def submit_native_binary(self, binary_path: str, args: List[str],
                           target_os: str, target_arch: str,
                           env: Optional[Dict[str, str]] = None) -> str:
        """Submit a native binary for execution"""
        with open(binary_path, 'rb') as f:
            binary_data = f.read()
        
        payload = {
            "binary": base64.b64encode(binary_data).decode(),
            "args": args,
            "environment": env or {}
        }
        
        requirements = {
            "os": target_os,
            "architecture": target_arch
        }
        
        return self.submit_task(TaskType.NATIVE_EXEC, payload, requirements)
    
    def submit_docker_task(self, task_data_or_image: Union[Dict[str, Any], str], 
                          command: Optional[List[str]] = None,
                          volumes: Optional[Dict[str, str]] = None,
                          environment: Optional[Dict[str, str]] = None) -> str:
        """Submit a Docker container task"""
        if isinstance(task_data_or_image, dict):
            # Accept full task data object
            payload = task_data_or_image
        else:
            # Accept individual parameters
            payload = {
                "image": task_data_or_image,
                "command": command or [],
                "volumes": volumes or {},
                "environment": environment or {}
            }
        
        requirements = {
            "docker": True
        }
        
        return self.submit_task(TaskType.DOCKER, payload, requirements)

    def submit_compile_task(self, source_code: str, language: str = "c++",
                           compiler_flags: Optional[List[str]] = None) -> str:
        """Submit a compilation task"""
        payload = {
            "source_code": source_code,
            "language": language,
            "compiler_flags": compiler_flags or []
        }
        
        return self.submit_task(TaskType.COMPILE, payload)

    def submit_ml_task(self, task_data: Dict[str, Any]) -> str:
        """Submit a machine learning inference task"""
        return self.submit_task(TaskType.ML_INFERENCE, task_data)

    def submit_video_task(self, task_data: Dict[str, Any]) -> str:
        """Submit a video encoding task"""
        return self.submit_task(TaskType.VIDEO_ENCODE, task_data)

    def submit_game_render_task(self, task_data: Dict[str, Any]) -> str:
        """Submit a game rendering task"""
        return self.submit_task(TaskType.GAME_RENDER, task_data)

    def get_devices(self) -> List[Dict[str, Any]]:
        """Alias for list_devices for compatibility"""
        return self.list_devices()


class AsyncTaskClient:
    """Async client for real-time task monitoring via WebSocket"""
    
    def __init__(self, coordinator_ws_url: str):
        """
        Initialize async client
        
        Args:
            coordinator_ws_url: WebSocket URL (e.g., "ws://192.168.1.100:8080/ws")
        """
        self.ws_url = coordinator_ws_url
        self.device_id = "python_client_" + str(np.random.randint(1000000))
        
    async def monitor_tasks(self, callback):
        """
        Connect and monitor task updates in real-time
        
        Args:
            callback: Async function to call with task updates
        """
        uri = f"{self.ws_url}?device_id={self.device_id}"
        
        async with websockets.connect(uri) as websocket:
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                await callback(data)


# Example usage
if __name__ == "__main__":
    # Create client
    client = TaskClient("http://localhost:8080")
    
    # Example 1: Submit a CUDA kernel
    cuda_kernel = """
    extern "C" __global__ void my_kernel(float* data, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            data[idx] = data[idx] * 2.0f + 1.0f;
        }
    }
    """
    
    data = {"data": [1.0, 2.0, 3.0, 4.0, 5.0]}
    task_id = client.submit_cuda_kernel(
        cuda_kernel, 
        data, 
        grid_size=[1, 1, 1], 
        block_size=[256, 1, 1]
    )
    
    print(f"Submitted CUDA kernel task: {task_id}")
    
    # Wait for result
    result = client.wait_for_completion(task_id)
    print(f"Result: {result}")
    
    # Example 2: Submit a Docker task
    docker_task_id = client.submit_docker_task(
        task_data_or_image="python:3.9-slim",
        command=["python", "-c", "print('Hello from distributed Docker!')"],
        environment={"MY_VAR": "test"}
    )
    
    print(f"Submitted Docker task: {docker_task_id}")
    
    # Example 3: List available devices
    devices = client.list_devices()
    print(f"Available devices: {len(devices)}")
    for device in devices:
        print(f"  - {device['hostname']} ({device['architecture']}, {device['os']})")
        if device.get('gpus'):
            for gpu in device['gpus']:
                print(f"    GPU: {gpu['name']} ({gpu['memory_bytes']/(1024**3):.1f} GB)") 