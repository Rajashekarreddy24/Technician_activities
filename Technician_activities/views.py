
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from .models import Ticket, Activity, VideoRecording, Prompt
import datetime
import mss
import cv2
import numpy as np
import threading
import os
from datetime import datetime
import csv
from django.http import HttpResponse
from django.shortcuts import get_object_or_404
from .models import Ticket, Activity
from django.http import HttpResponse
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from django.shortcuts import render, get_object_or_404
from .models import Ticket, Activity
from .services import TicketSystemIntegration
from django.http import JsonResponse



def dashboard(request):
    tickets = Ticket.objects.all()
    return render(request, 'technician_activities/dashboard.html', {'tickets': tickets})


ticket_system = TicketSystemIntegration()

def sync_ticket(request, ticket_id):
    if request.method == 'POST':
        status = request.POST.get('status')
        success = ticket_system.sync_ticket_status(ticket_id, status)
        if success:
            return JsonResponse({'status': 'success', 'message': 'Ticket status synced successfully.'})
        else:
            return JsonResponse({'status': 'error', 'message': 'Failed to sync ticket status.'}, status=500)

def view_ticket(request, ticket_id):
    ticket = get_object_or_404(Ticket, ticket_id=ticket_id)
    activities = Activity.objects.filter(ticket=ticket)
    
    # Fetch ticket details from the external system
    external_ticket_details = ticket_system.get_ticket_details(ticket_id)

    return render(request, 'technician_activities/report.html', {
        'ticket': ticket,
        'activities': activities,
        'external_ticket_details': external_ticket_details,
    })

is_recording = False

def record_screen(ticket_id):
    global is_recording
    is_recording = True
    
    # Define the output directory and filename
    output_dir = "recordings"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{ticket_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi")
    
    # Set up screen capture
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Using the first monitor
        width = monitor['width']
        height = monitor['height']
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
        
        while is_recording:
            img = sct.grab(monitor)
            frame = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Convert BGRA to BGR
            out.write(frame)

        # Release the video writer
        out.release()

def start_recording(request, ticket_id):
    global is_recording
    
    if is_recording:
        return JsonResponse({'status': 'Recording is already in progress'})

    recording_thread = threading.Thread(target=record_screen, args=(ticket_id,))
    recording_thread.start()
    
    return JsonResponse({'status': 'Recording started', 'ticket_id': ticket_id})

def stop_recording(request, ticket_id):
    global is_recording

    if not is_recording:
        return JsonResponse({'status': 'No recording is in progress', 'ticket_id': ticket_id})

    is_recording = False

    return JsonResponse({'status': 'Recording stopped', 'ticket_id': ticket_id})


def download_report(request, ticket_id):
    # Fetch the ticket and related activities
    ticket = get_object_or_404(Ticket, ticket_id=ticket_id)
    activities = Activity.objects.filter(ticket=ticket)

    # Create the HTTP response with CSV content type
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{ticket_id}_report.csv"'

    # Create a CSV writer object
    writer = csv.writer(response)
    
    # Write the CSV header
    writer.writerow(['Activity ID', 'Ticket ID', 'Timestamp', 'Application', 'Action', 'Notes', 'Duration', 'Category', 'Automated Flag'])

    # Write data rows for each activity
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

from django.http import HttpResponse
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def download_report_pdf(request, ticket_id):
    ticket = get_object_or_404(Ticket, ticket_id=ticket_id)
    activities = Activity.objects.filter(ticket=ticket)

    # Create a PDF response
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{ticket_id}_report.pdf"'

    # Create a canvas
    p = canvas.Canvas(response, pagesize=letter)
    width, height = letter

    # Write content to the PDF
    p.drawString(100, height - 50, f"Report for Ticket ID: {ticket_id}")
    p.drawString(100, height - 70, "Activities:")

    # Adding activity details
    y_position = height - 90
    for activity in activities:
        p.drawString(100, y_position, f"Activity ID: {activity.id}, Action: {activity.action}, Timestamp: {activity.timestamp}")
        y_position -= 20  # Move down for the next line

    # Finalize and return the PDF
    p.showPage()
    p.save()
    return response







