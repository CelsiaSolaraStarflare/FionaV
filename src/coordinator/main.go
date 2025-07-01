package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
)

type TaskType string

const (
	CudaKernel   TaskType = "CUDA_KERNEL"
	RocmKernel   TaskType = "ROCM_KERNEL"
	MetalKernel  TaskType = "METAL_KERNEL"
	OpenCLKernel TaskType = "OPENCL_KERNEL"
	NativeExec   TaskType = "NATIVE_EXEC"
	WasmExec     TaskType = "WASM"
	DockerExec   TaskType = "DOCKER"
	GameRender   TaskType = "GAME_RENDER"
	MLTraining   TaskType = "ML_TRAINING"
	MLInference  TaskType = "ML_INFERENCE"
	VideoEncode  TaskType = "VIDEO_ENCODE"
	Compile      TaskType = "COMPILE"
	Assembly     TaskType = "ASSEMBLY"
)

type DeviceCapability struct {
	ID           string    `json:"id"`
	Hostname     string    `json:"hostname"`
	Architecture string    `json:"architecture"`
	OS           string    `json:"os"`
	CPUCores     int       `json:"cpu_cores"`
	RAMBytes     uint64    `json:"ram_bytes"`
	GPUs         []GPU     `json:"gpus"`
	Frameworks   []string  `json:"frameworks"`
	LastSeen     time.Time `json:"last_seen"`
	Status       string    `json:"status"`
}

type GPU struct {
	Name               string  `json:"name"`
	MemoryBytes        uint64  `json:"memory_bytes"`
	ComputeCapability  string  `json:"compute_capability"`
	Utilization        float64 `json:"utilization"`
	Type               string  `json:"type"` // "cuda", "rocm", "metal", "opencl"
}

type Task struct {
	ID           string                 `json:"id"`
	Type         TaskType               `json:"type"`
	Priority     int                    `json:"priority"`
	Requirements map[string]interface{} `json:"requirements"`
	Payload      string                 `json:"payload"` // Base64 encoded
	CallbackURL  string                 `json:"callback_url,omitempty"`
	SubmittedAt  time.Time              `json:"submitted_at"`
	StartedAt    *time.Time             `json:"started_at,omitempty"`
	CompletedAt  *time.Time             `json:"completed_at,omitempty"`
	AssignedTo   string                 `json:"assigned_to,omitempty"`
	Status       string                 `json:"status"`
	Result       interface{}            `json:"result,omitempty"`
	ErrorMsg     string                 `json:"error_msg,omitempty"`
}

type Coordinator struct {
	devices    map[string]*DeviceCapability
	taskQueue  []*Task
	activeTasks map[string]*Task
	completedTasks map[string]*Task
	wsConnections map[string]*websocket.Conn
	
	deviceMutex sync.RWMutex
	taskMutex   sync.RWMutex
	wsMutex     sync.RWMutex
	
	upgrader websocket.Upgrader
}

func NewCoordinator() *Coordinator {
	return &Coordinator{
		devices:        make(map[string]*DeviceCapability),
		taskQueue:      make([]*Task, 0),
		activeTasks:    make(map[string]*Task),
		completedTasks: make(map[string]*Task),
		wsConnections:  make(map[string]*websocket.Conn),
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool { return true },
		},
	}
}

func (c *Coordinator) HandleDeviceRegistration(w http.ResponseWriter, r *http.Request) {
	var device DeviceCapability
	if err := json.NewDecoder(r.Body).Decode(&device); err != nil {
		http.Error(w, "Invalid device data", http.StatusBadRequest)
		return
	}

	device.LastSeen = time.Now()
	device.Status = "available"

	c.deviceMutex.Lock()
	c.devices[device.ID] = &device
	c.deviceMutex.Unlock()

	log.Printf("Device registered: %s (%s %s, %d cores, %d GPUs)", 
		device.ID, device.Architecture, device.OS, device.CPUCores, len(device.GPUs))
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status": "registered",
		"device_id": device.ID,
	})

	// Try to assign queued tasks
	go c.scheduleNextTask()
}

func (c *Coordinator) HandleTaskSubmission(w http.ResponseWriter, r *http.Request) {
	var task Task
	if err := json.NewDecoder(r.Body).Decode(&task); err != nil {
		http.Error(w, "Invalid task data", http.StatusBadRequest)
		return
	}

	task.ID = generateTaskID()
	task.SubmittedAt = time.Now()
	task.Status = "queued"

	c.taskMutex.Lock()
	c.taskQueue = append(c.taskQueue, &task)
	c.taskMutex.Unlock()

	log.Printf("Task submitted: %s (type: %s, priority: %d)", task.ID, task.Type, task.Priority)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status": "submitted",
		"task_id": task.ID,
	})

	go c.scheduleNextTask()
}

func (c *Coordinator) HandleTaskStatus(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	taskID := vars["task_id"]

	c.taskMutex.RLock()
	defer c.taskMutex.RUnlock()

	// Check completed tasks first
	if task, exists := c.completedTasks[taskID]; exists {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(task)
		return
	}

	// Check active tasks
	if task, exists := c.activeTasks[taskID]; exists {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(task)
		return
	}

	// Check queued tasks
	for _, task := range c.taskQueue {
		if task.ID == taskID {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(task)
			return
		}
	}

	http.Error(w, "Task not found", http.StatusNotFound)
}

func (c *Coordinator) HandleDeviceList(w http.ResponseWriter, r *http.Request) {
	c.deviceMutex.RLock()
	defer c.deviceMutex.RUnlock()

	devices := make([]*DeviceCapability, 0, len(c.devices))
	for _, device := range c.devices {
		devices = append(devices, device)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"devices": devices,
		"count": len(devices),
	})
}

func (c *Coordinator) HandleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := c.upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("WebSocket upgrade error: %v", err)
		return
	}
	defer conn.Close()

	deviceID := r.URL.Query().Get("device_id")
	if deviceID == "" {
		log.Printf("WebSocket connection without device_id")
		return
	}

	c.wsMutex.Lock()
	c.wsConnections[deviceID] = conn
	c.wsMutex.Unlock()

	log.Printf("WebSocket connected: %s", deviceID)

	// Update device status
	c.deviceMutex.Lock()
	if device, exists := c.devices[deviceID]; exists {
		device.Status = "connected"
		device.LastSeen = time.Now()
	}
	c.deviceMutex.Unlock()

	for {
		var message map[string]interface{}
		err := conn.ReadJSON(&message)
		if err != nil {
			log.Printf("WebSocket read error from %s: %v", deviceID, err)
			break
		}

		c.handleWebSocketMessage(deviceID, message, conn)
	}

	// Clean up on disconnect
	c.wsMutex.Lock()
	delete(c.wsConnections, deviceID)
	c.wsMutex.Unlock()

	c.deviceMutex.Lock()
	if device, exists := c.devices[deviceID]; exists {
		device.Status = "disconnected"
	}
	c.deviceMutex.Unlock()
}

func (c *Coordinator) handleWebSocketMessage(deviceID string, message map[string]interface{}, conn *websocket.Conn) {
	msgType, ok := message["type"].(string)
	if !ok {
		return
	}

	switch msgType {
	case "ready":
		c.assignTaskToDevice(deviceID, conn)
	case "heartbeat":
		c.handleHeartbeat(deviceID)
	case "task_progress":
		c.handleTaskProgress(message)
	case "task_complete":
		c.handleTaskComplete(deviceID, message)
	case "task_error":
		c.handleTaskError(deviceID, message)
	}
}

func (c *Coordinator) handleHeartbeat(deviceID string) {
	c.deviceMutex.Lock()
	if device, exists := c.devices[deviceID]; exists {
		device.LastSeen = time.Now()
	}
	c.deviceMutex.Unlock()
}

func (c *Coordinator) handleTaskProgress(message map[string]interface{}) {
	taskID, _ := message["task_id"].(string)
	progress, _ := message["progress"].(float64)
	
	c.taskMutex.Lock()
	if task, exists := c.activeTasks[taskID]; exists {
		// Store progress in task metadata
		if task.Requirements == nil {
			task.Requirements = make(map[string]interface{})
		}
		task.Requirements["progress"] = progress
	}
	c.taskMutex.Unlock()
	
	log.Printf("Task %s progress: %.2f%%", taskID, progress)
}

func (c *Coordinator) handleTaskComplete(deviceID string, message map[string]interface{}) {
	taskID, _ := message["task_id"].(string)
	result := message["result"]
	
	c.taskMutex.Lock()
	task, exists := c.activeTasks[taskID]
	if exists {
		now := time.Now()
		task.CompletedAt = &now
		task.Status = "completed"
		task.Result = result
		
		// Move to completed
		c.completedTasks[taskID] = task
		delete(c.activeTasks, taskID)
	}
	c.taskMutex.Unlock()
	
	// Mark device as available
	c.deviceMutex.Lock()
	if device, exists := c.devices[deviceID]; exists {
		device.Status = "available"
	}
	c.deviceMutex.Unlock()
	
	log.Printf("Task %s completed by device %s", taskID, deviceID)
	
	// Try to assign next task
	go c.assignTaskToDevice(deviceID, nil)
}

func (c *Coordinator) handleTaskError(deviceID string, message map[string]interface{}) {
	taskID, _ := message["task_id"].(string)
	errorMsg, _ := message["error"].(string)
	
	c.taskMutex.Lock()
	task, exists := c.activeTasks[taskID]
	if exists {
		task.Status = "failed"
		task.ErrorMsg = errorMsg
		
		// Requeue the task
		c.taskQueue = append([]*Task{task}, c.taskQueue...)
		delete(c.activeTasks, taskID)
	}
	c.taskMutex.Unlock()
	
	// Mark device as available
	c.deviceMutex.Lock()
	if device, exists := c.devices[deviceID]; exists {
		device.Status = "available"
	}
	c.deviceMutex.Unlock()
	
	log.Printf("Task %s failed on device %s: %s", taskID, deviceID, errorMsg)
}

func (c *Coordinator) scheduleNextTask() {
	c.taskMutex.Lock()
	defer c.taskMutex.Unlock()
	c.deviceMutex.RLock()
	defer c.deviceMutex.RUnlock()

	// Sort tasks by priority (highest first)
	for i := 0; i < len(c.taskQueue); i++ {
		for j := i + 1; j < len(c.taskQueue); j++ {
			if c.taskQueue[j].Priority > c.taskQueue[i].Priority {
				c.taskQueue[i], c.taskQueue[j] = c.taskQueue[j], c.taskQueue[i]
			}
		}
	}

	// Find available devices
	availableDevices := make([]string, 0)
	for id, device := range c.devices {
		if device.Status == "available" || device.Status == "connected" {
			availableDevices = append(availableDevices, id)
		}
	}

	// Try to assign tasks
	for i := len(c.taskQueue) - 1; i >= 0 && len(availableDevices) > 0; i-- {
		task := c.taskQueue[i]
		
		for j, deviceID := range availableDevices {
			device := c.devices[deviceID]
			if c.canDeviceHandleTask(device, task) {
				// Remove from queue
				c.taskQueue = append(c.taskQueue[:i], c.taskQueue[i+1:]...)
				
				// Add to active tasks
				now := time.Now()
				task.StartedAt = &now
				task.AssignedTo = device.ID
				task.Status = "running"
				c.activeTasks[task.ID] = task
				
				// Mark device as busy
				device.Status = "busy"
				
				// Remove from available list
				availableDevices = append(availableDevices[:j], availableDevices[j+1:]...)
				
				log.Printf("Task %s assigned to device %s", task.ID, device.ID)
				
				// Send task via websocket if connected
				c.wsMutex.RLock()
				if conn, connected := c.wsConnections[deviceID]; connected {
					go c.sendTaskToDevice(conn, task)
				}
				c.wsMutex.RUnlock()
				
				break
			}
		}
	}
}

func (c *Coordinator) assignTaskToDevice(deviceID string, conn *websocket.Conn) {
	c.scheduleNextTask()
}

func (c *Coordinator) sendTaskToDevice(conn *websocket.Conn, task *Task) {
	message := map[string]interface{}{
		"type":         "execute_task",
		"task_id":      task.ID,
		"task_type":    string(task.Type),
		"requirements": task.Requirements,
		"payload":      task.Payload,
	}
	
	if err := conn.WriteJSON(message); err != nil {
		log.Printf("Failed to send task %s: %v", task.ID, err)
		// Requeue task
		c.taskMutex.Lock()
		task.Status = "queued"
		task.AssignedTo = ""
		c.taskQueue = append([]*Task{task}, c.taskQueue...)
		delete(c.activeTasks, task.ID)
		c.taskMutex.Unlock()
	}
}

func (c *Coordinator) canDeviceHandleTask(device *DeviceCapability, task *Task) bool {
	// Check OS and architecture requirements
	if reqOS, ok := task.Requirements["os"].(string); ok {
		if device.OS != reqOS {
			return false
		}
	}

	if reqArch, ok := task.Requirements["architecture"].(string); ok {
		if device.Architecture != reqArch {
			return false
		}
	}

	// Check GPU requirements
	switch task.Type {
	case CudaKernel:
		hasCuda := false
		for _, gpu := range device.GPUs {
			if gpu.Type == "cuda" {
				hasCuda = true
				// Check compute capability if specified
				if reqCC, ok := task.Requirements["compute_capability"].(string); ok {
					if gpu.ComputeCapability < reqCC {
						continue
					}
				}
				// Check memory requirement
				if reqMem, ok := task.Requirements["gpu_memory_gb"].(float64); ok {
					if float64(gpu.MemoryBytes)/(1024*1024*1024) < reqMem {
						continue
					}
				}
				return true
			}
		}
		return hasCuda
		
	case RocmKernel:
		for _, gpu := range device.GPUs {
			if gpu.Type == "rocm" {
				return true
			}
		}
		return false
		
	case MetalKernel:
		for _, gpu := range device.GPUs {
			if gpu.Type == "metal" {
				return true
			}
		}
		return false
		
	case DockerExec:
		// Check if docker is available
		for _, framework := range device.Frameworks {
			if framework == "docker" {
				return true
			}
		}
		return false
	}

	// Check memory requirements
	if reqMem, ok := task.Requirements["min_memory_gb"].(float64); ok {
		if float64(device.RAMBytes)/(1024*1024*1024) < reqMem {
			return false
		}
	}

	// Check CPU core requirements
	if reqCores, ok := task.Requirements["min_cpu_cores"].(float64); ok {
		if float64(device.CPUCores) < reqCores {
			return false
		}
	}

	return true
}

func (c *Coordinator) startHealthChecker() {
	ticker := time.NewTicker(30 * time.Second)
	go func() {
		for range ticker.C {
			c.checkDeviceHealth()
		}
	}()
}

func (c *Coordinator) checkDeviceHealth() {
	c.deviceMutex.Lock()
	c.taskMutex.Lock()
	defer c.deviceMutex.Unlock()
	defer c.taskMutex.Unlock()

	timeout := 60 * time.Second
	now := time.Now()

	for deviceID, device := range c.devices {
		if now.Sub(device.LastSeen) > timeout && device.Status != "disconnected" {
			log.Printf("Device %s timed out", deviceID)
			device.Status = "disconnected"
			
			// Requeue any tasks assigned to this device
			for taskID, task := range c.activeTasks {
				if task.AssignedTo == deviceID {
					log.Printf("Requeueing task %s from timed-out device %s", taskID, deviceID)
					task.Status = "queued"
					task.AssignedTo = ""
					c.taskQueue = append([]*Task{task}, c.taskQueue...)
					delete(c.activeTasks, taskID)
				}
			}
		}
	}
}

func generateTaskID() string {
	return fmt.Sprintf("task_%d", time.Now().UnixNano())
}

func main() {
	coordinator := NewCoordinator()
	
	// Start health checker
	coordinator.startHealthChecker()
	
	r := mux.NewRouter()
	
	// API routes
	api := r.PathPrefix("/api").Subrouter()
	api.HandleFunc("/devices", coordinator.HandleDeviceRegistration).Methods("POST")
	api.HandleFunc("/devices", coordinator.HandleDeviceList).Methods("GET")
	api.HandleFunc("/submit", coordinator.HandleTaskSubmission).Methods("POST")
	api.HandleFunc("/task/{task_id}", coordinator.HandleTaskStatus).Methods("GET")
	
	// WebSocket endpoint
	r.HandleFunc("/ws", coordinator.HandleWebSocket)
	
	// Serve static files
	r.PathPrefix("/").Handler(http.FileServer(http.Dir("./web/")))
	
	log.Println("Coordinator starting on :8080")
	log.Fatal(http.ListenAndServe(":8080", r))
} 