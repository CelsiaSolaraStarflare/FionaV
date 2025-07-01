import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import threading
import os
import time
from pathlib import Path

class UDCS_GUI:
    def __init__(self, master):
        self.master = master
        self.master.title("UDCS - Universal Distributed Computing System Starter")
        self.master.geometry("800x600")
        
        # Style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Processes
        self.coordinator_process = None
        self.worker_process = None
        
        # Main container
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 1. System Control
        control_frame = ttk.LabelFrame(main_frame, text="System Control", padding="10")
        control_frame.pack(fill=tk.X, pady=5)
        
        self.start_system_button = ttk.Button(control_frame, text="Start System", command=self.start_system)
        self.start_system_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_system_button = ttk.Button(control_frame, text="Stop System", command=self.stop_system, state=tk.DISABLED)
        self.stop_system_button.pack(side=tk.LEFT, padx=5)

        # 2. Auto-Task Discovery
        auto_task_frame = ttk.LabelFrame(main_frame, text="Auto-Task Discovery", padding="10")
        auto_task_frame.pack(fill=tk.X, pady=5)
        
        self.auto_task_var = tk.BooleanVar()
        self.auto_task_check = ttk.Checkbutton(auto_task_frame, text="Enable Auto-Discovery", variable=self.auto_task_var, command=self.toggle_auto_task_ui)
        self.auto_task_check.pack(anchor=tk.W)
        
        # --- UI for auto-task options (initially disabled) ---
        self.auto_task_options_frame = ttk.Frame(auto_task_frame, padding="5")
        self.auto_task_options_frame.pack(fill=tk.X, expand=True)

        # Target directory
        ttk.Label(self.auto_task_options_frame, text="Directory to Scan:").pack(anchor=tk.W, pady=2)
        self.directory_var = tk.StringVar()
        self.dir_entry = ttk.Entry(self.auto_task_options_frame, textvariable=self.directory_var, state='disabled')
        self.dir_entry.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(0, 5))
        self.browse_button = ttk.Button(self.auto_task_options_frame, text="Browse...", command=self.browse_directory, state='disabled')
        self.browse_button.pack(side=tk.LEFT)
        
        # Task types
        ttk.Label(self.auto_task_options_frame, text="Task Types to Find:").pack(anchor=tk.W, pady=(10, 2))
        self.task_types_frame = ttk.Frame(self.auto_task_options_frame)
        self.task_types_frame.pack(fill=tk.X)
        
        self.task_vars = {
            "Image Processing": tk.BooleanVar(),
            "Video Encoding": tk.BooleanVar(),
            "Code Compilation": tk.BooleanVar()
        }
        
        self.task_checkboxes = []
        for i, (text, var) in enumerate(self.task_vars.items()):
            cb = ttk.Checkbutton(self.task_types_frame, text=text, variable=var, state='disabled')
            cb.pack(side=tk.LEFT, padx=10)
            self.task_checkboxes.append(cb)

        # 3. Log Output
        log_frame = ttk.LabelFrame(main_frame, text="System Logs", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = tk.Text(log_frame, wrap=tk.WORD, height=15, state=tk.DISABLED, bg="#f0f0f0")
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self.toggle_auto_task_ui() # Set initial state

    def browse_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.directory_var.set(directory)

    def toggle_auto_task_ui(self):
        state = ['!disabled'] if self.auto_task_var.get() else ['disabled']
        
        self.dir_entry.state(state)
        self.browse_button.state(state)
        for cb in self.task_checkboxes:
            cb.state(state)

    def log(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.config(state=tk.DISABLED)
        self.log_text.see(tk.END)

    def stream_output(self, process, name):
        for line in iter(process.stdout.readline, ''):
            self.log(f"[{name}] {line.strip()}")

    def start_system(self):
        self.log("Starting UDCS...")
        
        # --- Start Coordinator ---
        try:
            exe = "./bin/coordinator.exe" if os.name == 'nt' else "./bin/coordinator"
            self.coordinator_process = subprocess.Popen(
                [exe],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            threading.Thread(target=self.stream_output, args=(self.coordinator_process, "Coordinator"), daemon=True).start()
            self.log("Coordinator process started.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start coordinator: {e}")
            return

        # --- Start Worker ---
        exe = "./bin/worker.exe" if os.name == 'nt' else "./bin/worker"
        worker_cmd = [exe, "--coordinator", "localhost:8080"]
        
        if self.auto_task_var.get():
            scan_dir = self.directory_var.get()
            if not scan_dir:
                messagebox.showwarning("Warning", "Please select a directory to scan for auto-discovery.")
                self.stop_system()
                return
            
            task_types = [t for t, v in self.task_vars.items() if v.get()]
            if not task_types:
                messagebox.showwarning("Warning", "Please select at least one task type for auto-discovery.")
                self.stop_system()
                return

            worker_cmd.extend(["--auto-discover", "--scan-dir", scan_dir, "--task-types", ",".join(task_types).replace(" ", "_").lower()])
            
        try:
            self.worker_process = subprocess.Popen(
                worker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            threading.Thread(target=self.stream_output, args=(self.worker_process, "Worker"), daemon=True).start()
            self.log("Worker process started.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start worker: {e}")
            self.stop_system()
            return
            
        self.start_system_button.config(state=tk.DISABLED)
        self.stop_system_button.config(state=tk.NORMAL)

    def stop_system(self):
        self.log("Stopping UDCS...")
        
        if self.worker_process:
            self.worker_process.terminate()
            self.log("Terminated worker.")
        if self.coordinator_process:
            self.coordinator_process.terminate()
            self.log("Terminated coordinator.")
            
        self.coordinator_process = None
        self.worker_process = None
        
        self.start_system_button.config(state=tk.NORMAL)
        self.stop_system_button.config(state=tk.DISABLED)

if __name__ == "__main__":
    # Ensure we are in the script's directory
    os.chdir(Path(__file__).parent)
    
    # Check if binaries exist
    coordinator_path = Path("./bin/coordinator.exe" if os.name == 'nt' else "./bin/coordinator")
    worker_path = Path("./bin/worker.exe" if os.name == 'nt' else "./bin/worker")
    if not coordinator_path.exists() or not worker_path.exists():
        messagebox.showerror("Error", "Binaries not found. Please run 'make build-all' first.")
        
    root = tk.Tk()
    app = UDCS_GUI(root)
    
    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to stop the UDCS system and quit?"):
            app.stop_system()
            root.destroy()
            
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop() 