package main

// Worker for Universal Distributed Computing System

import (
    "C"
    "bytes"
    "encoding/json"
    "flag"
    "fmt"
    "io/ioutil"
    "log"
    "net/http"
    "net/url"
    "os"
    "time"

    "github.com/gorilla/websocket"
    "github.com/google/uuid"
)

// CSystemInfo is a simplified representation of hardware info from C detector
// via CGO bridging. In this stub, we mock detection.

// Device represents the registration payload
type Device struct {
    ID           string   `json:"id"`
    Hostname     string   `json:"hostname"`
    Architecture string   `json:"architecture"`
    OS           string   `json:"os"`
    CPUCores     int      `json:"cpu_cores"`
    RAMBytes     int64    `json:"ram_bytes"`
    GPUs         []string `json:"gpus"`
}

func detectHardware() Device {
    // TODO: replace with actual CGO calls to C.detect_hardware
    hostname, _ := os.Hostname()
    return Device{
        ID:           uuid.NewString(),
        Hostname:     hostname,
        Architecture: "$(HOSTARCH)", // placeholder
        OS:           "$(OS)",       // placeholder
        CPUCores:     4,
        RAMBytes:     8 * 1024 * 1024 * 1024,
        GPUs:         []string{"cuda:0"},
    }
}

func main() {
    coordAddr := flag.String("coordinator", "localhost:8080", "Coordinator address host:port")
    flag.Parse()

    // 1. Detect hardware
    device := detectHardware()

    // 2. Register device
    regURL := fmt.Sprintf("http://%s/api/devices", *coordAddr)
    body, _ := json.Marshal(device)
    resp, err := http.Post(regURL, "application/json", bytes.NewBuffer(body))
    if err != nil {
        log.Fatalf("failed to register device: %v", err)
    }
    defer resp.Body.Close()
    data, _ := ioutil.ReadAll(resp.Body)
    var regResp map[string]string
    json.Unmarshal(data, &regResp)
    deviceID, ok := regResp["device_id"]
    if !ok {
        log.Fatalf("invalid register response: %s", string(data))
    }
    log.Printf("Registered with device_id: %s", deviceID)

    // 3. Connect WebSocket
    wsURL := url.URL{Scheme: "ws", Host: *coordAddr, Path: "/ws", RawQuery: "device_id=" + deviceID}
    conn, _, err := websocket.DefaultDialer.Dial(wsURL.String(), nil)
    if err != nil {
        log.Fatalf("failed to connect WebSocket: %v", err)
    }
    defer conn.Close()
    log.Println("WebSocket connection established")

    // Notify ready
    conn.WriteJSON(map[string]interface{}{"type": "ready", "device_id": deviceID})

    // 4. Listen for tasks
    for {
        var msg map[string]interface{}
        if err := conn.ReadJSON(&msg); err != nil {
            log.Printf("read error: %v", err)
            break
        }
        if msgType, ok := msg["type"].(string); ok && msgType == "execute_task" {
            go handleTask(conn, deviceID, msg)
        }
    }
}

func handleTask(conn *websocket.Conn, deviceID string, msg map[string]interface{}) {
    taskID, _ := msg["task_id"].(string)
    log.Printf("Received task: %s", taskID)

    // Simulate execution
    time.Sleep(2 * time.Second)

    // Report completion
    result := map[string]interface{}{
        "type":      "task_complete",
        "task_id":   taskID,
        "device_id": deviceID,
        "result":    "success",
    }
    conn.WriteJSON(result)
    log.Printf("Completed task: %s", taskID)
} 