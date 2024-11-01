from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse, HttpResponse
from .models import Ticket, Activity
from .services import TicketSystemIntegration
from .utils import generate_activity_dataframe, export_to_csv
from .monitoring import ActivityMonitor
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import datetime
import mss
import cv2
import numpy as np
import threading
import os
import csv
import pygetwindow as gw
import messages
from django.shortcuts import render
from rest_framework import viewsets
from .models import TicketInfo, Action
from .serializers import TicketInfoSerializer, ActionSerializer
from django.shortcuts import render, redirect, get_object_or_404
import pytesseract
import re
import json
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass
import threading
import time
import pyautogui
from queue import Queue
import threading
import time
import mouse
import keyboard
import difflib
from datetime import datetime
from typing import List
import re
import threading
from typing import Dict
import re
import yaml
import re
import json
from PIL import Image
from datetime import datetime
import difflib
import hashlib
from typing import Dict, List, Optional, Tuple, Set
import logging
from dataclasses import dataclass, asdict
import time
from queue import Queue
from datetime import datetime
import logging
import re
import logging
from datetime import datetime
from typing import Optional
import yaml
import re
from django.shortcuts import render, redirect
from rest_framework import viewsets
from .models import Pattern
from .serializers import PatternSerializer
from django.shortcuts import render
import yaml
import re
from .pattern_matcher import PatternMatcher
from django.views import View
from .actioncontext import ActionContext
import cache



is_recording = False
ticket_system = TicketSystemIntegration()
active_monitors = {}

def dashboard(request):
    tickets = Ticket.objects.all()
    return render(request, 'technician_activities/dashboard.html', {'tickets': tickets})

def sync_ticket(request, ticket_id):
    if request.method == 'POST':
        status = request.POST.get('status')
        success = ticket_system.sync_ticket_status(ticket_id, status)
        if success:
            return JsonResponse({'status': 'success', 'message': 'Ticket status synced successfully.'})
        return JsonResponse({'status': 'error', 'message': 'Failed to sync ticket status.'}, status=500)

def view_ticket(request, ticket_id):
    ticket = get_object_or_404(Ticket, ticket_id=ticket_id)
    # activities = Activity.objects.filter(ticket=ticket)
    # external_ticket_details = ticket_system.get_ticket_details(ticket_id)
    # return render(request, 'technician_activities/report.html', {
    #     'ticket': ticket,
    #     'activities': activities,
    #     'external_ticket_details': external_ticket_details,
    # })
    return render(request, 'technician_activities/recording_template.html', {'ticket': ticket})


def record_screen(ticket_id):
    global is_recording
    is_recording = True
    output_dir = "recordings"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{ticket_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi")
    
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        width, height = monitor['width'], monitor['height']
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
        
        while is_recording:
            img = sct.grab(monitor)
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)
            out.write(frame)
        out.release()

from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def start_recording(request, ticket_id):
    global is_recording
    if is_recording:
        return JsonResponse({'status': 'Recording is already in progress', 'ticket_id': ticket_id})
    threading.Thread(target=record_screen, args=(ticket_id,)).start()
    is_recording = True  
    return render(request, 'technician_activities/recording_in_progress.html', {'ticket_id': ticket_id})

@csrf_exempt
def stop_recording(request, ticket_id):
    global is_recording
    if request.method == 'POST':
        if not is_recording:
            return JsonResponse({'status': 'No recording is in progress', 'ticket_id': ticket_id})
        is_recording = False
        return JsonResponse({'status': 'Recording stopped successfully', 'ticket_id': ticket_id})
    return JsonResponse({'status': 'Invalid request method'}, status=405)
    # return JsonResponse({'status': 'Invalid request method'}, status=405)

def download_report(request, ticket_id):
    ticket = get_object_or_404(Ticket, ticket_id=ticket_id)
    activities = Activity.objects.filter(ticket=ticket)
    
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{ticket_id}_report.csv"'
    writer = csv.writer(response)
    writer.writerow(['Activity ID', 'Ticket ID', 'Timestamp', 'Application', 'Action', 'Notes', 'Duration', 'Category', 'Automated Flag'])
    for activity in activities:
        writer.writerow([
            activity.id,
            activity.ticket.ticket_id,
            activity.timestamp,
            activity.application,
            activity.action,
            activity.notes,
            activity.duration,
            activity.category,
            activity.automated_flag
        ])
    return response


def download_report_pdf(request, ticket_id):
    ticket = get_object_or_404(Ticket, ticket_id=ticket_id)
    activities = Activity.objects.filter(ticket=ticket)
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{ticket_id}_report.pdf"'
    
    p = canvas.Canvas(response, pagesize=letter)
    width, height = letter
    p.drawString(100, height - 50, f"Report for Ticket ID: {ticket_id}")
    y_position = height - 90
    for activity in activities:
        p.drawString(100, y_position, f"Activity ID: {activity.id}, Action: {activity.action}, Timestamp: {activity.timestamp}")
        y_position -= 20
    p.showPage()
    p.save()
    return response

def start_monitoring(request, ticket_id):
    if ticket_id not in active_monitors:
        monitor = ActivityMonitor(ticket_id)
        monitor.start_monitoring()
        active_monitors[ticket_id] = monitor
        return JsonResponse({'status': f'Monitoring started for ticket {ticket_id}'})
    return JsonResponse({'status': f'Monitoring already active for ticket {ticket_id}'})

def stop_monitoring(request, ticket_id):
    monitor = active_monitors.get(ticket_id)
    if monitor:
        monitor.stop_monitoring()
        del active_monitors[ticket_id]
        return JsonResponse({'status': f'Monitoring stopped for ticket {ticket_id}'})
    return JsonResponse({'status': f'No active monitor found for ticket {ticket_id}'})

def generate_activity_report(request, ticket_id):
    ticket = get_object_or_404(Ticket, ticket_id=ticket_id)
    start_date = request.GET.get('start_date', '2023-01-01')
    end_date = request.GET.get('end_date', datetime.datetime.now().strftime('%Y-%m-%d'))
    activities = Activity.get_activity_report(ticket_id, start_date, end_date)
    df = generate_activity_dataframe(activities)
    
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="activity_report_{ticket_id}.csv"'
    df.to_csv(response, index=False)
    return response


def generate_time_analysis(request, ticket_id):
    analysis = Activity.get_time_analysis(ticket_id)
    return JsonResponse(analysis)

class TicketInfoViewSet(viewsets.ModelViewSet):
    queryset = TicketInfo.objects.all()
    serializer_class = TicketInfoSerializer

class ActionViewSet(viewsets.ModelViewSet):
    queryset = Action.objects.all()
    serializer_class = ActionSerializer
    
class PatternViewSet(viewsets.ModelViewSet):
    queryset = Pattern.objects.all()
    serializer_class = PatternSerializer



def ticket_list(request):
    """View to display the list of tickets."""
    tickets = TicketInfo.objects.all()  
    return render(request, 'tickets/ticket_list.html', {'tickets': tickets})

def ticket_detail(request, ticket_id ):
    """View to display details of a specific ticket."""
    ticket = get_object_or_404(TicketInfo, ticket_id=ticket_id)  # Get ticket by primary key
    return render(request, 'tickets/ticket_detail.html', {'ticket': ticket})


class EnhancedTicketAnalyzer(View):

    """Enhanced ticket analyzer with better pattern recognition."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pattern_matcher = PatternMatcher()
        self.ocr_queue = Queue()
        self.latest_ocr_result = ""
        self.start_ocr_worker()

    def start_ocr_worker(self):
        """Start background OCR processing."""
        def worker():
            while True:
                frame = self.ocr_queue.get()
                if frame is None:
                    break
                self._process_frame_ocr(frame)

        self.ocr_thread = threading.Thread(target=worker, daemon=True)
        self.ocr_thread.start()

    def _process_frame_ocr(self, frame):
        """Process OCR in background."""
        try:
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(thresh, config=custom_config)
            # Store result in class variable
            self.latest_ocr_result = text
        except Exception as e:
            logging.error(f"OCR processing error: {str(e)}")

    def post(self, request, *args, **kwargs):
        """Handle POST requests for analyzing frames."""
        frame_file = request.FILES['frame']
        
        # Read the frame using OpenCV
        file_bytes = np.asarray(bytearray(frame_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Process the frame with OCR
        self.ocr_queue.put(frame)

        # Get the latest OCR result after processing (this may need synchronization)
        text_result = self.latest_ocr_result

        # Extract environment info from the OCR result (if needed)
        env_info = self.pattern_matcher.extract_environment_info(text_result)

        return render(request, 'tickets/analyze.html', {'text_result': text_result, 'env_info': env_info})

    def get(self, request, *args, **kwargs):
        """Render the analyze page."""
        return render(request, 'tickets/analyze.html')
enhanced_analyzer = EnhancedTicketAnalyzer()

def analyze_frame(request):

    """Analyze a video frame for ticket information."""
    if request.method == 'POST':
        # Assume frame is uploaded as a file (you can modify this as needed)
        frame_file = request.FILES['frame']
        
        # Read the frame using OpenCV
        file_bytes = np.asarray(bytearray(frame_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Process the frame with OCR
        enhanced_analyzer.ocr_queue.put(frame)

        # Get the latest OCR result after processing (this may need synchronization)
        text_result = enhanced_analyzer.latest_ocr_result

        
        env_info = enhanced_analyzer.extract_environment_info(text_result)

        return render(request, 'tickets/analyze.html', {'text_result': text_result, 'env_info': env_info})

    return render(request, 'tickets/analyze.html')




# class EnhancedActionRecorder:
#     """Enhanced action recorder with more detailed action capture"""

#     def __init__(self):
#         self.actions = []  # List to store recorded actions
#         self.recording = False
#         self.last_actions = []
#         self.action_patterns = []
#         self.logger = logging.getLogger(__name__)

#     def start_recording(self):
#         """Start recording user actions"""
#         self.recording = True
#         self.record_thread = threading.Thread(target=self._record_actions)
#         self.record_thread.start()

#     def stop_recording(self):
#         """Stop recording user actions"""
#         self.recording = False
#         self.record_thread.join()

#     def _record_actions(self):
#         """Record user actions in real-time"""
#         while self.recording:
#             if mouse.is_pressed():
#                 x, y = mouse.get_position()
#                 self._handle_click(x, y)
#             if keyboard.is_pressed('ctrl+c'):
#                 self._handle_keyboard_shortcut('copy')
#             elif keyboard.is_pressed('ctrl+v'):
#                 self._handle_keyboard_shortcut('paste')
#             time.sleep(0.1)

#     def _handle_click(self, x: int, y: int):
#         """Handle mouse click events"""
#         try:
#             # Capture screen region around click
#             region = (max(0, x-50), max(0, y-50), min(x+50, 1920), min(y+50, 1080))
#             screenshot = pyautogui.screenshot(region=region)
#             # Get active window information
#             window_info = self._get_window_info()  
            
#             # Create action context (you should define ActionContext)
#             context = ActionContext(
#                 window_title=window_info.get('title', ''),
#                 active_element=self._get_element_at_position(x, y),
#                 parent_element=self._get_parent_element(x, y),
#                 screen_region=region
#             )
            
#             # Create and store action
#             action = Action(
#                 action_type='click',
#                 target=f'pos_{x}_{y}',
#                 parameters={'x': x, 'y': y},
#                 timestamp=datetime.now(),
#                 screen_location=(x, y),
#                 context=context,
#                 verification={'element_visible': 'true'},
#                 wait_time=0.5
#             )
#             self.actions.append(action)
#             self._analyze_pattern(action)
#         except Exception as e:
#             self.logger.error(f"Error handling click: {str(e)}")

#     def _analyze_pattern(self, action: Action):
#         """Analyze actions for patterns"""
#         self.last_actions.append(action)
#         if len(self.last_actions) > 5:
#             self.last_actions.pop(0)
        
#         # Look for repeated sequences
#         self._find_repeated_sequences()

#     def _find_repeated_sequences(self):
#         """Find repeated sequences of actions"""
#         sequence = self._serialize_actions(self.last_actions)  # Implement this method to serialize actions
#         for pattern in self.action_patterns:
#             similarity = difflib.SequenceMatcher(None, sequence, pattern).ratio()
#             if similarity > 0.8:
#                 self.logger.info(f"Found similar pattern with {similarity:.2f} confidence")
#                 return
        
#         # Add new pattern if sequence is long enough
#         if len(self.last_actions) >= 3:
#             self.action_patterns.append(sequence)

#     def _get_window_info(self):
#         """Get information about the currently active window."""
#         try:
#             active_window = gw.getActiveWindow()
#             if active_window:
#                 return {
#                     'title': active_window.title,
#                     'size': (active_window.width, active_window.height),
#                     'position': (active_window.left, active_window.top)
#                 }
#             else:
#                 return {'title': 'Unknown', 'size': (0, 0), 'position': (0, 0)}
#         except Exception as e:
#             self.logger.error(f"Error getting window info: {str(e)}")
#             return {'title': 'Unknown', 'size': (0, 0), 'position': (0, 0)}
 
import threading
from django.core.cache import cache

class EnhancedActionRecorder:
    """Class to handle the recording of user actions."""
    
    def __init__(self):
        self.record_thread = None
        self.is_recording = False
    
    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.record_thread = threading.Thread(target=self.record_actions)
            self.record_thread.start()
            print("Recording started.")
            cache.set('is_recording', True) 
    
    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            cache.set('is_recording', False)  
            print("Recording stopped.")

    def record_actions(self):
        while self.is_recording:
            print("Recording user actions...")
            time.sleep(1)

from django.contrib import messages
from django.shortcuts import redirect

class RecordActionView(View):
    """View to handle recording actions."""

    def get(self, request, ticket_id):
        """Render the recording UI page."""
        return render(request, 'tickets/record.html', {'ticket_id': ticket_id})

    def post(self, request, ticket_id):
        action_recorder = EnhancedActionRecorder()
        if request.POST.get('action') == 'start':
            action_recorder.start_recording()
            messages.success(request, 'Recording in progress')
            return redirect('recording_status', ticket_id=ticket_id)  # Redirect to recording status page
        elif request.POST.get('action') == 'stop':
            action_recorder.stop_recording()
            messages.success(request, 'Recording has been stopped.')  # Show success message
            return redirect('dashboard')  # Redirect to the dashboard

        return render(request, 'tickets/record.html', {'ticket_id': ticket_id})


class RecordingStatusView(View):
    """View to display recording status"""
    
    def get(self, request, ticket_id):
        is_recording = cache.get('is_recording', False)
        status_message = "Recording is in progress..." if is_recording else "Recording has been stopped."
        return render(request, 'tickets/recording_status.html', {
            'ticket_id': ticket_id,
            'status_message': status_message
        })
