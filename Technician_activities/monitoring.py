import threading
import logging
import time
from datetime import datetime
from .models import Activity
from django.conf import settings
import win32gui
import win32process
import psutil

class ActivityMonitor:
    def __init__(self, ticket_id: str):
        self.ticket_id = ticket_id
        self.last_activity_time = datetime.now()
        self.current_window = None
        self.recording = False

    def start_monitoring(self):
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
           
            # Use Django ORM to log activity in the database
            Activity.objects.create(
                ticket_id=self.ticket_id,
                timestamp=datetime.now(),
                application=app_name,
                action=f"Switched to: {window_title}",
                category='WINDOW_CHANGE'
            )
           
            self.last_activity_time = datetime.now()
        except Exception as e:
            logging.error(f"Failed to log window change: {e}")

    def _check_inactivity(self) -> bool:
        inactivity_threshold = getattr(settings, 'INACTIVITY_THRESHOLD', 300)  # default threshold 5 mins
        return (datetime.now() - self.last_activity_time).seconds > inactivity_threshold

    def _prompt_for_update(self):
        logging.info("User inactive, prompt for update.")
        # Implement prompt logic, perhaps sending a notification or updating a status in the database.
