from django.db import models

class Ticket(models.Model):
    ticket_id = models.CharField(max_length=255, primary_key=True)
    status = models.CharField(max_length=100)
    description = models.TextField()
    category = models.CharField(max_length=100)
    tags = models.CharField(max_length=255)
    priority = models.CharField(max_length=50)
    start_time = models.DateTimeField()
    last_updated = models.DateTimeField(auto_now=True)
    resolution_time = models.DateTimeField(null=True, blank=True)
    resolution_notes = models.TextField(blank=True, null=True)
    automated_prompts = models.IntegerField(default=0)
    integration_status = models.CharField(max_length=100, blank=True, null=True)

class Activity(models.Model):
    ticket = models.ForeignKey(Ticket, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    application = models.CharField(max_length=100)
    action = models.CharField(max_length=100)
    notes = models.TextField(blank=True, null=True)
    duration = models.IntegerField(default=0)
    category = models.CharField(max_length=100, blank=True, null=True)
    automated_flag = models.BooleanField(default=False)

class Prompt(models.Model):
    ticket = models.ForeignKey(Ticket, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    prompt_type = models.CharField(max_length=100)
    response = models.TextField()
    response_time = models.DateTimeField(blank=True, null=True)

class VideoRecording(models.Model):
    ticket = models.ForeignKey(Ticket, on_delete=models.CASCADE)
    start_time = models.DateTimeField(auto_now_add=True)
    duration = models.FloatField()
    filepath = models.CharField(max_length=255)
    filesize = models.BigIntegerField()
    resolution = models.CharField(max_length=50)
    fps = models.IntegerField()
