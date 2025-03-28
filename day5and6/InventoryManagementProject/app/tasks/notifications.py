from fastapi import BackgroundTasks
from typing import Optional
from datetime import datetime
import smtplib
from email.mime.text import MIMEText

def send_email_notification(email: str, message: str):
    # This is a simplified example - in production use a proper email service
    msg = MIMEText(message)
    msg["Subject"] = "Inventory Manager Notification"
    msg["From"] = "noreply@inventory.com"
    msg["To"] = email
    
    try:
        with smtplib.SMTP("localhost") as server:
            server.send_message(msg)
    except Exception as e:
        print(f"Failed to send email: {e}")
