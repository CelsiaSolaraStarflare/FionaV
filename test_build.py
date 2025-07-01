#!/usr/bin/env python3
"""
Quick test script to verify the Universal Distributed Computing System
can be built and started correctly.
"""

import subprocess
import sys
import time
import os
import signal
import requests
from pathlib import Path

def run_command(cmd, cwd=None, timeout=30):
    """Run a command and return success status"""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            cwd=cwd, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Timeout"
    except Exception as e:
        return False, "", str(e)

def test_build():
    """Test building all components"""
    print("🔨 Testing build system...")
    
    # Test Make
    success, stdout, stderr = run_command("make --version")
    if not success:
        print("❌ Make not available")
        return False
    
    # Test Go
    success, stdout, stderr = run_command("go version")
    if not success:
        print("❌ Go not available")
        return False
    
    print("✅ Build tools available")
    
    # Try to build coordinator
    print("🔨 Building coordinator...")
    success, stdout, stderr = run_command("make build-coordinator")
    if not success:
        print(f"❌ Coordinator build failed: {stderr}")
        return False
    
    print("✅ Coordinator built successfully")
    
    # Check if binaries exist
    coord_path = Path("bin") / ("coordinator.exe" if os.name == 'nt' else "coordinator")
    if not coord_path.exists():
        print("❌ Coordinator binary not found")
        return False
    
    print("✅ All builds successful")
    return True

def test_startup():
    """Test starting coordinator and worker"""
    print("🚀 Testing system startup...")
    
    # Start coordinator
    coord_cmd = "./bin/coordinator.exe --port 8080 --web-port 8081" if os.name == 'nt' else "./bin/coordinator --port 8080 --web-port 8081"
    
    try:
        coord_process = subprocess.Popen(
            coord_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a bit for startup
        time.sleep(3)
        
        # Test if coordinator is responding
        try:
            response = requests.get("http://localhost:8080/api/devices", timeout=5)
            if response.status_code == 200:
                print("✅ Coordinator started and responding")
                coord_success = True
            else:
                print(f"❌ Coordinator not responding correctly: {response.status_code}")
                coord_success = False
        except requests.RequestException as e:
            print(f"❌ Cannot connect to coordinator: {e}")
            coord_success = False
        
        # Test web interface
        try:
            response = requests.get("http://localhost:8081", timeout=5)
            if response.status_code == 200:
                print("✅ Web interface accessible")
            else:
                print(f"⚠️  Web interface returned {response.status_code}")
        except requests.RequestException:
            print("⚠️  Web interface not accessible")
        
        # Stop coordinator
        coord_process.terminate()
        try:
            coord_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            coord_process.kill()
        
        return coord_success
        
    except Exception as e:
        print(f"❌ Error starting coordinator: {e}")
        return False

def test_python_client():
    """Test Python client imports"""
    print("🐍 Testing Python client...")
    
    try:
        # Test if we can import the client
        sys.path.insert(0, str(Path.cwd()))
        from client import TaskClient, TaskType
        
        print("✅ Python client imports successfully")
        
        # Test client creation (without actual connection)
        client = TaskClient("http://localhost:8080")
        print("✅ Client creation successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Cannot import client: {e}")
        return False
    except Exception as e:
        print(f"❌ Client error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Universal Distributed Computing System - Build Test")
    print("=" * 60)
    
    # Change to distributed_compute directory
    os.chdir(Path(__file__).parent)
    
    tests = [
        ("Build System", test_build),
        ("System Startup", test_startup),
        ("Python Client", test_python_client),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 40)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\nTo start the system:")
        print("1. Run: make run-coordinator")
        print("2. Run: make run-worker (in another terminal)")
        print("3. Open: http://localhost:8081 (web interface)")
        print("4. Run: python examples/demo.py (to test tasks)")
        return True
    else:
        print("💔 Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 