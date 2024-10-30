from django.urls import path
from . import views


urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('ticket/<str:ticket_id>/', views.view_ticket, name='view_ticket'),
    # path('ticket/<str:ticket_id>/start-recording/', views.start_recording, name='start_recording'),
    # path('ticket/<str:ticket_id>/stop-recording/', views.stop_recording, name='stop_recording'),
    path('start_recording/<str:ticket_id>/', views.start_recording, name='start_recording'),
    path('stop_recording/<str:ticket_id>/', views.stop_recording, name='stop_recording'),  # URL for stopping recording
    path('ticket/<str:ticket_id>/download-report/', views.download_report, name='download_report'),
    path('time-analysis/<str:ticket_id>/', views.generate_time_analysis, name='generate_time_analysis'),
]
