import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import os
import subprocess
import sys

# Load Config
import json
def load_config(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)
config = load_config("config.json")
MODEL_PATH = config.get("model_path")
OUTPUT_PATH = config.get("output_path")

#Start Apps
class ROITrackingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Region Interest Tracking App")
        self.root.geometry("1200x800")
        # Variables
        self.video_path = None
        self.video_cap = None
        self.current_frame = None
        self.points = []
        self.tracking_active = False
        self.output_video_path = OUTPUT_PATH
        self.is_done = False
        # Create GUI
        self.create_gui()
        
    def create_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Load video button
        self.load_btn = ttk.Button(control_frame, text="Load Video", command=self.load_video)
        self.load_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Start tracking button
        self.track_btn = ttk.Button(control_frame, text="Start Tracking", 
                                   command=self.start_tracking, state=tk.DISABLED)
        self.track_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Reset points button
        self.reset_btn = ttk.Button(control_frame, text="Reset Points", 
                                   command=self.reset_points, state=tk.DISABLED)
        self.reset_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Ready - Please load a video")
        self.status_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # Progress bar
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Video display frame
        video_frame = ttk.Frame(main_frame, relief=tk.SUNKEN, borderwidth=2)
        video_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Canvas for video display
        self.canvas = tk.Canvas(video_frame, bg='black', width=800, height=600)
        self.canvas.pack(expand=True, fill=tk.BOTH)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # Points info frame
        info_frame = ttk.Frame(main_frame)
        info_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Points display
        self.points_label = ttk.Label(info_frame, text="Points selected: 0/4")
        self.points_label.pack(side=tk.LEFT)
        
        # Coordinates display
        self.coords_text = tk.Text(info_frame, height=3, width=60)
        self.coords_text.pack(side=tk.RIGHT, padx=(10, 0))
        
    def load_video(self):
        """Load video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.video_path = file_path
            self.load_first_frame()
            self.reset_points()
            self.status_label.config(text=f"Video loaded: {os.path.basename(file_path)}")
            
    def load_first_frame(self):
        """Load and display first frame of video"""
        try:
            self.video_cap = cv2.VideoCapture(self.video_path)
            ret, frame = self.video_cap.read()
            
            if ret:
                self.current_frame = frame
                self.display_frame(frame)
                self.reset_btn.config(state=tk.NORMAL)
            else:
                messagebox.showerror("Error", "Could not read video file")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error loading video: {str(e)}")
            
    def display_frame(self, frame):
        """Display frame on canvas"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            self.root.after(100, lambda: self.display_frame(frame))
            return
            
        # Calculate scaling to fit canvas while maintaining aspect ratio
        frame_height, frame_width = frame.shape[:2]
        scale_x = canvas_width / frame_width
        scale_y = canvas_height / frame_height
        scale = min(scale_x, scale_y)
        
        # Resize frame
        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)
        
        # Store scale for coordinate conversion
        self.scale = scale
        self.display_width = new_width
        self.display_height = new_height
        self.offset_x = (canvas_width - new_width) // 2
        self.offset_y = (canvas_height - new_height) // 2
        
        frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
        
        # Draw points on frame
        if self.points:
            for i, point in enumerate(self.points):
                # Convert original coordinates to display coordinates
                display_x = int(point[0] * scale) + self.offset_x
                display_y = int(point[1] * scale) + self.offset_y
                
                # Draw point
                cv2.circle(frame_resized, (display_x - self.offset_x, display_y - self.offset_y), 5, (255, 0, 0), -1)
                # Draw point number
                cv2.putText(frame_resized, str(i+1), (display_x - self.offset_x + 10, display_y - self.offset_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Draw lines between points if we have more than 1 point
        if len(self.points) > 1:
            for i in range(len(self.points)):
                start_point = self.points[i]
                end_point = self.points[(i + 1) % len(self.points)]
                
                start_display = (int(start_point[0] * scale), int(start_point[1] * scale))
                end_display = (int(end_point[0] * scale), int(end_point[1] * scale))
                # cv2.line(frame_resized, start_display, end_display, (0, 255, 0), 2)
        
        # Convert to PIL Image and display
        image_pil = Image.fromarray(frame_resized)
        self.photo = ImageTk.PhotoImage(image_pil)
        
        # Clear canvas and display image
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.photo)
        
    def on_canvas_click(self, event):
        """Handle canvas click to select points"""
        if self.current_frame is None or len(self.points) >= 4:
            return
            
        # Convert display coordinates to original frame coordinates
        click_x = event.x - self.offset_x
        click_y = event.y - self.offset_y
        
        # Check if click is within the displayed frame
        if (0 <= click_x <= self.display_width and 0 <= click_y <= self.display_height):
            original_x = int(click_x / self.scale)
            original_y = int(click_y / self.scale)
            
            self.points.append((original_x, original_y))
            self.update_points_display()
            self.display_frame(self.current_frame)
            
            # Enable tracking button if 4 points selected
            if len(self.points) == 4:
                self.track_btn.config(state=tk.NORMAL)
                self.status_label.config(text="4 points selected - Ready to track")
        print(self.points)
                
    def update_points_display(self):
        """Update points display"""
        self.points_label.config(text=f"Points selected: {len(self.points)}/4")
        
        # Update coordinates display
        self.coords_text.delete(1.0, tk.END)
        for i, point in enumerate(self.points):
            self.coords_text.insert(tk.END, f"Point {i+1}: ({point[0]}, {point[1]})\n")
            
    def reset_points(self):
        """Reset selected points"""
        self.points = []
        self.update_points_display()
        self.track_btn.config(state=tk.DISABLED)
        if self.current_frame is not None:
            self.display_frame(self.current_frame)
        self.status_label.config(text="Points reset - Select 4 points to track")
        
    def start_tracking(self):
        """Start tracking process"""
        if len(self.points) != 4:
            messagebox.showwarning("Warning", "Please select exactly 4 points")
            return
            
        # Disable buttons during tracking
        self.load_btn.config(state=tk.DISABLED)
        self.track_btn.config(state=tk.DISABLED)
        self.reset_btn.config(state=tk.DISABLED)
        
        # Start progress bar
        self.progress.start()
        self.status_label.config(text="Tracking in progress...")
        
        # Start tracking in separate thread
        tracking_thread = threading.Thread(target=self.run_tracking)
        tracking_thread.daemon = True
        tracking_thread.start()
        
    def run_tracking(self):
        """Run tracking process in separate thread"""
            # Import track module
        from tracker.track import PersonTracker
        tracker = PersonTracker(MODEL_PATH, self.video_path, self.output_video_path)
        # Bắt đầu tracking
        print("Bắt đầu tracking...")
        tracker.track_video(self.points)
        success = True
        if success:
            self.status_label.config(text="Tracking completed successfully!")
            if os.path.exists(self.output_video_path):
                self.video_path = self.output_video_path
                self.play_output_video()
            else:
                messagebox.showerror("Error", "Output video not found")
                self.status_label.config(text="Tracking failed: Output video not found")
                self.reset_points()
            
    def tracking_complete(self, success, error_msg=None):
        """Called when tracking is complete"""
        self.progress.stop()
        
        self.load_btn.config(state=tk.NORMAL)
        self.reset_btn.config(state=tk.NORMAL)
        
        if success:
            self.status_label.config(text="Tracking completed successfully!")
            if os.path.exists(self.output_video_path):
                self.video_path = self.output_video_path
                self.play_output_video()
            else:
                messagebox.showerror("Error", "Output video not found")
                self.status_label.config(text="Tracking failed: Output video not found")
                self.reset_points()
        else:
            self.status_label.config(text="Tracking failed")
            if error_msg:
                messagebox.showerror("Tracking Error", f"Tracking failed: {error_msg}")
            else:
                messagebox.showerror("Tracking Error", "Tracking failed")
            self.reset_points()
            
    def play_output_video(self):
        """Play output video on canvas"""
        self.reset_points()
        if self.video_path:
            if self.video_cap:
                self.video_cap.release()
                
            self.video_cap = cv2.VideoCapture(self.output_video_path)
            
            self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = 30
            self.frame_delay = max(1, int(1000 / self.fps))
            
            if not self.video_cap.isOpened():
                messagebox.showerror("Error", "Could not open output video")
                self.reset_points()
                return
                
            self.is_playing = True
            self.current_frame_index = 0
            self.play_next_frame()
            
    def play_next_frame(self):
        """Play next frame"""
        if not self.is_playing or not self.video_cap:
            return
            
        ret, frame = self.video_cap.read()
        if ret:
            self.current_frame = frame
            self.display_frame(frame)
            self.current_frame_index += 1
            
            if self.is_playing:
                self.root.after(self.frame_delay, self.play_next_frame)
        else:
            total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.current_frame_index >= total_frames:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame_index = 0
                if self.is_playing:
                    self.root.after(self.frame_delay, self.play_next_frame)
            else:
                print(f"Error reading frame at index {self.current_frame_index}")
                self.is_playing = False
                self.status_label.config(text="Video playback error")
                self.reset_points()
def main():
    root = tk.Tk()
    app = ROITrackingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()