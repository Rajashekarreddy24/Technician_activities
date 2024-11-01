from django.urls import path
from . import views
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import TicketInfoViewSet, ActionViewSet, ticket_list, ticket_detail, analyze_frame, EnhancedTicketAnalyzer,EnhancedActionRecorder,RecordActionView, RecordingStatusView

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('ticket/<str:ticket_id>/', views.view_ticket, name='view_ticket'),
    path('start_recording/<str:ticket_id>/', views.start_recording, name='start_recording'),
    path('stop_recording/<str:ticket_id>/', views.stop_recording, name='stop_recording'),  
    path('ticket/<str:ticket_id>/download-report/', views.download_report, name='download_report'),
    path('time-analysis/<str:ticket_id>/', views.generate_time_analysis, name='generate_time_analysis'),
    path('ticket_list/', ticket_list, name='ticket_list'),
    path('tickets/<str:ticket_id>/', ticket_detail, name= 'ticket_detail'),
    path('Actions/', ActionViewSet.as_view({'get': 'list', 'post': 'create'}), name='ticket_action_list'), 
    path('analyze/<int:pk>/',EnhancedTicketAnalyzer.as_view(), name='analyze_ticket'),  
    path('analyze/', analyze_frame, name='analyze_frame'),  
    path('record/<str:ticket_id>/', RecordActionView.as_view(), name='record_action'),
    path('recording_status/<str:ticket_id>/', RecordingStatusView.as_view(), name='recording_status'),
   
]

