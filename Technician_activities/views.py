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

# Global variable to manage recording state
is_recording = False
ticket_system = TicketSystemIntegration()
active_monitors = {}

# Dashboard view
def dashboard(request):
    tickets = Ticket.objects.all()
    return render(request, 'technician_activities/dashboard.html', {'tickets': tickets})

# Ticket sync function
def sync_ticket(request, ticket_id):
    if request.method == 'POST':
        status = request.POST.get('status')
        success = ticket_system.sync_ticket_status(ticket_id, status)
        if success:
            return JsonResponse({'status': 'success', 'message': 'Ticket status synced successfully.'})
        return JsonResponse({'status': 'error', 'message': 'Failed to sync ticket status.'}, status=500)

# Ticket view with external details
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

# Recording functions
def record_screen(ticket_id):
    global is_recording
    is_recording = True
    output_dir = "recordings"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{ticket_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.avi")
    
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
        # If recording is already in progress, return a JSON response with a message
        return JsonResponse({'status': 'Recording is already in progress', 'ticket_id': ticket_id})
    
    # Start a new recording thread
    threading.Thread(target=record_screen, args=(ticket_id,)).start()
    is_recording = True  # Set the global variable to indicate recording has started
    
    # Render the template showing recording in progress
    return render(request, 'technician_activities/recording_in_progress.html', {'ticket_id': ticket_id})
def stop_recording(request, ticket_id):
    global is_recording
    if not is_recording:
        message = "No recording is in progress for this ticket."
    else:
        is_recording = False
        message = "Recording has been stopped successfully."

    return render(request, 'technician_activities/stop_recording.html', {'message': message})
    # return JsonResponse({'status': 'Invalid request method'}, status=405)
# Download report as CSV
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

# Download report as PDF
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

# Monitoring functions
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

# Generate activity report CSV
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

# Generate time analysis
def generate_time_analysis(request, ticket_id):
    analysis = Activity.get_time_analysis(ticket_id)
    return JsonResponse(analysis)
