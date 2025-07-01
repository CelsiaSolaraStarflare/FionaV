#!/usr/bin/env python3
"""
Universal Distributed Computing System Starter

This script makes it easy to start the entire UDCS system.
It handles building, starting components, and monitoring.
"""

import subprocess
import sys
import time
import os
import signal
import json
import argparse
import threading
from pathlib import Path

class UDCSStarter:
    def __init__(self):
        self.processes = []
        self.running = True
        
    def run_command_background(self, cmd, name):
        """Run a command in the background and track the process"""
        print(f"🚀 Starting {name}...")
        
        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            self.processes.append((process, name))
            
            # Monitor output in a separate thread
            def monitor_output():
                while self.running:
                    if process.stdout:
                        line = process.stdout.readline()
                        if line:
                            print(f"[{name}] {line.strip()}")
                    time.sleep(0.1)
            
            thread = threading.Thread(target=monitor_output, daemon=True)
            thread.start()
            
            return process
            
        except Exception as e:
            print(f"❌ Failed to start {name}: {e}")
            return None
    
    def build_system(self):
        """Build the entire system"""
        print("🔨 Building Universal Distributed Computing System...")
        
        result = subprocess.run(
            "make build-all",
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"❌ Build failed: {result.stderr}")
            return False
        
        print("✅ Build completed successfully")
        return True
    
    def start_coordinator(self):
        """Start the coordinator"""
        cmd = "./bin/coordinator.exe --port 8080 --web-port 8081" if os.name == 'nt' else "./bin/coordinator --port 8080 --web-port 8081"
        return self.run_command_background(cmd, "Coordinator")
    
    def start_worker(self, worker_id=1):
        """Start a worker"""
        cmd = f"./bin/worker.exe --coordinator localhost:8080" if os.name == 'nt' else f"./bin/worker --coordinator localhost:8080"
        return self.run_command_background(cmd, f"Worker-{worker_id}")
    
    def wait_for_coordinator(self, timeout=30):
        """Wait for coordinator to be ready"""
        import requests
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get("http://localhost:8080/api/devices", timeout=2)
                if response.status_code == 200:
                    print("✅ Coordinator is ready")
                    return True
            except:
                pass
            time.sleep(1)
        
        print("❌ Coordinator failed to start within timeout")
        return False
    
    def run_demo(self):
        """Run the demo script"""
        print("🎯 Running demo tasks...")
        
        result = subprocess.run(
            "python examples/demo.py",
            shell=True,
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        if result.stderr:
            print(f"Demo stderr: {result.stderr}")
        
        return result.returncode == 0
    
    def show_status(self):
        """Show system status"""
        print("\n📊 System Status:")
        print("-" * 40)
        
        alive_processes = []
        for process, name in self.processes:
            if process.poll() is None:
                alive_processes.append(name)
                print(f"✅ {name}: Running (PID: {process.pid})")
            else:
                print(f"❌ {name}: Stopped")
        
        if alive_processes:
            print(f"\n🌐 Web Interface: http://localhost:8081")
            print(f"🔌 API Endpoint: http://localhost:8080/api")
        
        return len(alive_processes)
    
    def cleanup(self):
        """Clean up all processes"""
        print("\n🧹 Shutting down system...")
        self.running = False
        
        for process, name in self.processes:
            if process.poll() is None:
                print(f"Stopping {name}...")
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                except:
                    pass
        
        print("✅ System shutdown complete")
    
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print("\n⚠️  Interrupt received, shutting down...")
        self.cleanup()
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Universal Distributed Computing System Starter")
    parser.add_argument("--no-build", action="store_true", help="Skip building the system")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker nodes to start")
    parser.add_argument("--demo", action="store_true", help="Run demo after starting")
    parser.add_argument("--daemon", action="store_true", help="Run in daemon mode (no interactive monitoring)")
    
    args = parser.parse_args()
    
    # Change to the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    starter = UDCSStarter()
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, starter.signal_handler)
    if os.name != 'nt':  # SIGTERM not available on Windows
        signal.signal(signal.SIGTERM, starter.signal_handler)
    
    try:
        print("🌟 Universal Distributed Computing System")
        print("=" * 50)
        
        # Build system if requested
        if not args.no_build:
            if not starter.build_system():
                return 1
        
        # Create necessary directories
        Path("cache").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        
        # Start coordinator
        if not starter.start_coordinator():
            return 1
        
        # Wait for coordinator to be ready
        if not starter.wait_for_coordinator():
            starter.cleanup()
            return 1
        
        # Start workers
        for i in range(args.workers):
            time.sleep(2)  # Stagger worker starts
            starter.start_worker(i + 1)
        
        time.sleep(3)  # Wait for workers to connect
        
        # Show initial status
        starter.show_status()
        
        # Run demo if requested
        if args.demo:
            time.sleep(2)
            starter.run_demo()
        
        if args.daemon:
            print("\n🤖 Running in daemon mode...")
            while starter.running:
                time.sleep(60)
        else:
            print("\n🎮 Interactive Mode")
            print("Commands:")
            print("  status - Show system status")
            print("  demo   - Run demo tasks")
            print("  quit   - Shutdown system")
            print()
            
            while starter.running:
                try:
                    command = input("UDCS> ").strip().lower()
                    
                    if command == "status":
                        starter.show_status()
                    elif command == "demo":
                        starter.run_demo()
                    elif command in ["quit", "exit", "q"]:
                        break
                    elif command == "help":
                        print("Available commands: status, demo, quit")
                    elif command:
                        print(f"Unknown command: {command}")
                        
                except EOFError:
                    break
        
        starter.cleanup()
        return 0
        
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        starter.cleanup()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 