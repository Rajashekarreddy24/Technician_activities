import sqlite3
import datetime
import win32gui
import win32process
import psutil
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import keyboard
import threading
import json
from datetime import datetime, timedelta
import requests
import pandas as pd
from typing import Dict, List
import configparser
import os
import logging
from tkinter.scrolledtext import ScrolledText
import plotly.express as px
import plotly.io as pio
from PIL import Image, ImageTk
import time
import csv
import cv2
import numpy as np
from PIL import ImageGrab
import os
from datetime import datetime
import shutil
from threading import Thread, Event
import mss
import mss.tools

class DatabaseManager:
    def __init__(self, db_path='technician_activities.db'):
        self.db_path = db_path
        self.setup_database()

    def setup_database(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
           
            # Tickets table with enhanced fields
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS tickets (
                ticket_id TEXT PRIMARY KEY,
                status TEXT,
                description TEXT,
                category TEXT,
                tags TEXT,
                priority TEXT,
                start_time DATETIME,
                last_updated DATETIME,
                resolution_time DATETIME,
                resolution_notes TEXT,
                automated_prompts INTEGER DEFAULT 0,
                integration_status TEXT
            )
            ''')
           
            # Activities table with enhanced tracking
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS activities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT,
                timestamp DATETIME,
                application TEXT,
                action TEXT,
                notes TEXT,
                duration INTEGER,
                category TEXT,
                automated_flag BOOLEAN,
                FOREIGN KEY (ticket_id) REFERENCES tickets (ticket_id)
            )
            ''')
           
            # Prompts and responses tracking
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT,
                timestamp DATETIME,
                prompt_type TEXT,
                response TEXT,
                response_time DATETIME,
                FOREIGN KEY (ticket_id) REFERENCES tickets (ticket_id)
            )
            ''')

class TicketSystemIntegration:
    def __init__(self, config):
        self.config = config
        self.api_url = config['DEFAULT']['TicketSystemAPI']
        self.api_key = config['DEFAULT']['APIKey']
        self.headers = {'Authorization': f'Bearer {self.api_key}'}

    def sync_ticket_status(self, ticket_id: str, status: str) -> bool:
        try:
            response = requests.post(
                f"{self.api_url}/tickets/{ticket_id}/status",
                json={'status': status},
                headers=self.headers
            )
            return response.status_code == 200
        except Exception as e:
            logging.error(f"Failed to sync ticket status: {e}")
            return False

    def get_ticket_details(self, ticket_id: str) -> Dict:
        try:
            response = requests.get(
                f"{self.api_url}/tickets/{ticket_id}",
                headers=self.headers
            )
            return response.json() if response.status_code == 200 else {}
        except Exception as e:
            logging.error(f"Failed to get ticket details: {e}")
            return {}

class ActivityMonitor:
    def __init__(self, db_manager: DatabaseManager, config: configparser.ConfigParser):
        self.db_manager = db_manager
        self.config = config
        self.last_activity_time = datetime.now()
        self.current_window = None
        self.recording = False

    def start_monitoring(self, ticket_id: str):
        self.ticket_id = ticket_id
        self.recording = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        self.recording = False

    def _monitor_loop(self):
        while self.recording:
            try:
                window = win32gui.GetForegroundWindow()
                if window != self.current_window:
                    self._log_window_change(window)
                    self.current_window = window
               
                if self._check_inactivity():
                    self._prompt_for_update()
               
                time.sleep(1)
            except Exception as e:
                logging.error(f"Monitoring error: {e}")

    def _log_window_change(self, window):
        try:
            window_title = win32gui.GetWindowText(window)
            pid = win32process.GetWindowThreadProcessId(window)[1]
            process = psutil.Process(pid)
            app_name = process.name()
           
            with sqlite3.connect(self.db_manager.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT INTO activities (ticket_id, timestamp, application, action, category)
                VALUES (?, datetime('now'), ?, ?, 'WINDOW_CHANGE')
                ''', (self.ticket_id, app_name, f"Switched to: {window_title}"))
                conn.commit()
           
            self.last_activity_time = datetime.now()
        except Exception as e:
            logging.error(f"Failed to log window change: {e}")

    def _check_inactivity(self) -> bool:
        inactivity_threshold = int(self.config['DEFAULT']['InactivityThreshold'])
        return (datetime.now() - self.last_activity_time).seconds > inactivity_threshold

class ReportGenerator:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def generate_activity_report(self, ticket_id: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        query = '''
        SELECT timestamp, application, action, notes
        FROM activities
        WHERE ticket_id = ? AND timestamp BETWEEN ? AND ?
        ORDER BY timestamp DESC
        '''
       
        with sqlite3.connect(self.db_manager.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=[ticket_id, start_date, end_date])
        return df

    def generate_time_analysis(self, ticket_id: str) -> dict:
        with sqlite3.connect(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
            SELECT
                SUM(duration) as total_time,
                AVG(duration) as avg_duration,
                COUNT(*) as activity_count
            FROM activities
            WHERE ticket_id = ?
            ''', (ticket_id,))
            return dict(zip(['total_time', 'avg_duration', 'activity_count'], cursor.fetchone()))

    def export_to_csv(self, data: pd.DataFrame, filename: str):
        data.to_csv(filename, index=False)

class TechnicianAssistant:
    def __init__(self):
        self.load_config()
        self.setup_logging()
        self.db_manager = DatabaseManager()
        self.ticket_system = TicketSystemIntegration(self.config)
        self.activity_monitor = ActivityMonitor(self.db_manager, self.config)
        self.report_generator = ReportGenerator(self.db_manager)
        self.setup_gui()

    def load_config(self):
        self.config = configparser.ConfigParser()
        if not os.path.exists('config.ini'):
            self.create_default_config()
        self.config.read('config.ini')

    def create_default_config(self):
        self.config['DEFAULT'] = {
            'TicketSystemAPI': 'http://your-ticket-system-api.com',
            'APIKey': 'your-api-key',
            'InactivityThreshold': '300',
            'AutoPromptInterval': '1800',
            'DataExportPath': 'exports/'
        }
        os.makedirs('exports', exist_ok=True)
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)

    def setup_logging(self):
        logging.basicConfig(
            filename='technician_assistant.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Technician Assistant Pro")
        self.root.geometry("800x600")
       
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill='both', padx=5, pady=5)
       
        # Setup tabs
        self.setup_main_tab()
        self.setup_reports_tab()
        self.setup_settings_tab()

    def setup_main_tab(self):
        # Main tab setup code (as shown in previous version)
        pass

    def setup_reports_tab(self):
        # Reports tab setup code (as shown in previous version)
        pass

    def setup_settings_tab(self):
        # Settings tab setup code (as shown in previous version)
        pass

    def start_recording(self):
        ticket_id = self.ticket_entry.get()
        if not ticket_id:
            messagebox.showerror("Error", "Please enter a ticket ID")
            return

        # Start activity monitoring
        self.activity_monitor.start_monitoring(ticket_id)
       
        # Update UI
        self.status_label.config(text=f"Status: Recording Ticket {ticket_id}")
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
       
        # Sync with ticket system
        if self.ticket_system.sync_ticket_status(ticket_id, "IN_PROGRESS"):
            self.log_message("Successfully synced ticket status")
        else:
            self.log_message("Failed to sync ticket status")

    def stop_recording(self):
        self.activity_monitor.stop_monitoring()
       
        # Update UI
        self.status_label.config(text="Status: Not Recording")
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')

    def generate_report(self):
        if not self.current_ticket:
            messagebox.showerror("Error", "No active ticket selected")
            return

        report_type = self.report_type_var.get()
        start_date = datetime.now() - timedelta(days=7)  # Default to last 7 days
       
        if report_type == "Activity Summary":
            df = self.report_generator.generate_activity_report(
                self.current_ticket,
                start_date,
                datetime.now()
            )
            self.display_report(df)
        elif report_type == "Time Analysis":
            analysis = self.report_generator.generate_time_analysis(self.current_ticket)
            self.display_time_analysis(analysis)

    def export_data(self):
        if not self.current_ticket:
            messagebox.showerror("Error", "No active ticket selected")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialdir=self.config['DEFAULT']['DataExportPath']
        )
       
        if filename:
            df = self.report_generator.generate_activity_report(
                self.current_ticket,
                datetime.now() - timedelta(days=30),
                datetime.now()
            )
            self.report_generator.export_to_csv(df, filename)
            messagebox.showinfo("Success", "Data exported successfully")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    assistant = TechnicianAssistant()
    assistant.run()

# Add these imports to the existing imports section

class ScreenRecorder:
    def __init__(self, output_dir="recordings"):
        self.output_dir = output_dir
        self.recording = False
        self.current_video = None
        self.current_ticket = None
        os.makedirs(output_dir, exist_ok=True)
       
        # Screen recording settings
        self.fps = 10  # Lower FPS to reduce file size while maintaining usability
        self.codec = cv2.VideoWriter_fourcc(*'mp4v')
        self.sct = mss.mss()
       
        # Recording metadata
        self.recording_start = None
        self.recording_segments = []

    def start_recording(self, ticket_id):
        if self.recording:
            return
           
        self.current_ticket = ticket_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ticket_{ticket_id}_{timestamp}.mp4"
        filepath = os.path.join(self.output_dir, filename)
       
        # Get screen dimensions
        monitor = self.sct.monitors[1]  # Primary monitor
        self.width = monitor["width"]
        self.height = monitor["height"]
       
        # Initialize video writer
        self.current_video = cv2.VideoWriter(
            filepath,
            self.codec,
            self.fps,
            (self.width, self.height)
        )
       
        self.recording = True
        self.recording_start = datetime.now()
        self.recording_thread = Thread(target=self._record_screen)
        self.recording_thread.daemon = True
        self.recording_thread.start()
       
        # Log recording start
        logging.info(f"Started screen recording for ticket {ticket_id}: {filepath}")
        return filepath

    def stop_recording(self):
        if not self.recording:
            return
           
        self.recording = False
        if self.current_video:
            self.current_video.release()
           
        # Log recording segment
        duration = (datetime.now() - self.recording_start).total_seconds()
        self.recording_segments.append({
            'ticket_id': self.current_ticket,
            'start_time': self.recording_start,
            'duration': duration,
            'filepath': self.current_video.filename
        })
       
        logging.info(f"Stopped screen recording for ticket {self.current_ticket}")
        return self.recording_segments[-1]

    def _record_screen(self):
        while self.recording:
            try:
                # Capture screen
                screenshot = self.sct.grab(self.sct.monitors[1])
                frame = np.array(screenshot)
               
                # Convert from BGRA to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
               
                # Write frame
                self.current_video.write(frame)
               
                # Control frame rate
                cv2.waitKey(int(1000/self.fps))
               
            except Exception as e:
                logging.error(f"Error during screen recording: {e}")
                break

class VideoManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
       
        # Create videos table if it doesn't exist
        with sqlite3.connect(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS video_recordings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT,
                start_time DATETIME,
                duration FLOAT,
                filepath TEXT,
                filesize BIGINT,
                resolution TEXT,
                fps INTEGER,
                FOREIGN KEY (ticket_id) REFERENCES tickets (ticket_id)
            )
            ''')

    def save_recording_metadata(self, recording_data):
        try:
            filesize = os.path.getsize(recording_data['filepath'])
           
            with sqlite3.connect(self.db_manager.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT INTO video_recordings (
                    ticket_id, start_time, duration, filepath,
                    filesize, resolution, fps
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    recording_data['ticket_id'],
                    recording_data['start_time'],
                    recording_data['duration'],
                    recording_data['filepath'],
                    filesize,
                    f"{recording_data.get('width', 1920)}x{recording_data.get('height', 1080)}",
                    10  # Fixed FPS as defined in ScreenRecorder
                ))
               
        except Exception as e:
            logging.error(f"Error saving recording metadata: {e}")

    def get_recordings_for_ticket(self, ticket_id):
        with sqlite3.connect(self.db_manager.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
            SELECT * FROM video_recordings
            WHERE ticket_id = ?
            ORDER BY start_time DESC
            ''', (ticket_id,))
            return cursor.fetchall()

# Add these methods to the TechnicianAssistant class

def setup_recording_tab(self):
    recording_tab = ttk.Frame(self.notebook)
    self.notebook.add(recording_tab, text='Recordings')
   
    # Recording controls
    controls_frame = ttk.LabelFrame(recording_tab, text="Recording Controls")
    controls_frame.pack(fill='x', padx=5, pady=5)
   
    self.recording_status_label = ttk.Label(controls_frame, text="Status: Not Recording")
    self.recording_status_label.pack(pady=5)
   
    button_frame = ttk.Frame(controls_frame)
    button_frame.pack(pady=5)
   
    self.record_button = ttk.Button(button_frame, text="Start Recording",
                                  command=self.toggle_recording)
    self.record_button.pack(side='left', padx=5)
   
    # Recordings list
    list_frame = ttk.LabelFrame(recording_tab, text="Recent Recordings")
    list_frame.pack(fill='both', expand=True, padx=5, pady=5)
   
    self.recordings_tree = ttk.Treeview(list_frame, columns=('Date', 'Duration', 'Size'),
                                      show='headings')
    self.recordings_tree.heading('Date', text='Date')
    self.recordings_tree.heading('Duration', text='Duration')
    self.recordings_tree.heading('Size', text='Size')
    self.recordings_tree.pack(fill='both', expand=True, padx=5, pady=5)
   
    # Playback controls
    playback_frame = ttk.LabelFrame(recording_tab, text="Playback Controls")
    playback_frame.pack(fill='x', padx=5, pady=5)
   
    ttk.Button(playback_frame, text="Play Selected",
              command=self.play_selected_recording).pack(side='left', padx=5)
    ttk.Button(playback_frame, text="Export Selected",
              command=self.export_selected_recording).pack(side='left', padx=5)

def toggle_recording(self):
    if not hasattr(self, 'screen_recorder'):
        self.screen_recorder = ScreenRecorder()
        self.video_manager = VideoManager(self.db_manager)
   
    if not self.screen_recorder.recording:
        if not self.current_ticket:
            messagebox.showerror("Error", "Please start tracking a ticket first")
            return
           
        # Start recording
        filepath = self.screen_recorder.start_recording(self.current_ticket)
        self.record_button.config(text="Stop Recording")
        self.recording_status_label.config(text=f"Recording: {os.path.basename(filepath)}")
    else:
        # Stop recording
        recording_data = self.screen_recorder.stop_recording()
        self.video_manager.save_recording_metadata(recording_data)
        self.record_button.config(text="Start Recording")
        self.recording_status_label.config(text="Status: Not Recording")
        self.update_recordings_list()

def update_recordings_list(self):
    # Clear existing items
    for item in self.recordings_tree.get_children():
        self.recordings_tree.delete(item)
   
    # Add recordings for current ticket
    if self.current_ticket:
        recordings = self.video_manager.get_recordings_for_ticket(self.current_ticket)
        for recording in recordings:
            self.recordings_tree.insert('', 'end', values=(
                recording[2],  # start_time
                f"{recording[3]:.1f}s",  # duration
                f"{recording[5] / (1024*1024):.1f} MB"  # filesize in MB
            ))

def play_selected_recording(self):
    selection = self.recordings_tree.selection()
    if not selection:
        messagebox.showwarning("Warning", "Please select a recording to play")
        return
       
    item = self.recordings_tree.item(selection[0])
    recording_date = item['values'][0]
   
    # Find recording file
    recordings = self.video_manager.get_recordings_for_ticket(self.current_ticket)
    for recording in recordings:
        if recording[2] == recording_date:
            filepath = recording[4]
            if os.path.exists(filepath):
                # Open with default video player
                os.startfile(filepath)
            else:
                messagebox.showerror("Error", "Recording file not found")
            break

def export_selected_recording(self):
    selection = self.recordings_tree.selection()
    if not selection:
        messagebox.showwarning("Warning", "Please select a recording to export")
        return
       
    item = self.recordings_tree.item(selection[0])
    recording_date = item['values'][0]
   
    # Find recording file
    recordings = self.video_manager.get_recordings_for_ticket(self.current_ticket)
    for recording in recordings:
        if recording[2] == recording_date:
            source_path = recording[4]
            if not os.path.exists(source_path):
                messagebox.showerror("Error", "Recording file not found")
                return
               
            # Get export location
            target_path = filedialog.asksaveasfilename(
                defaultextension=".mp4",
                filetypes=[("MP4 files", "*.mp4")],
                initialdir=self.config['DEFAULT']['DataExportPath']
            )
           
            if target_path:
                shutil.copy2(source_path, target_path)
                messagebox.showinfo("Success", "Recording exported successfully")
            break

# Modify the __init__ method of TechnicianAssistant to include the recording tab
def __init__(self):
    # ... (existing initialization code) ...
    self.setup_recording_tab()  # Add this line after setting up other tabs