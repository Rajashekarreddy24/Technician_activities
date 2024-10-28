
import requests
import logging
from django.conf import settings
from typing import Dict

class TicketSystemIntegration:
    def __init__(self):
        self.api_url = settings.TICKET_SYSTEM_API_URL
        self.api_key = settings.TICKET_SYSTEM_API_KEY
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

