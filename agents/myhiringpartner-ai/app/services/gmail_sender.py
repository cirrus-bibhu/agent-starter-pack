import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

class GmailSender:
    def __init__(self):
        load_dotenv()
        self.username = os.getenv("GMAIL_USERNAME")
        self.app_password = os.getenv("GMAIL_APP_PASSWORD")

        if not self.username or not self.app_password:
            raise ValueError("GMAIL_USERNAME or GMAIL_APP_PASSWORD is not set in the .env file.")

    def send_email(self, sender: str, recipient: str, subject: str, body: str):
        try:
            message = MIMEMultipart()
            message['From'] = sender
            message['To'] = recipient
            message['Subject'] = subject
            message.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(self.username, self.app_password)
                server.sendmail(sender, recipient, message.as_string())

            print(f"Email sent successfully to {recipient}")
        except Exception as e:
            print(f"Failed to send email: {e}")