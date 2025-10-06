import smtplib
import os
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import pandas as pd

logger = logging.getLogger(__name__)

class EmailService:
    def __init__(self):
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.sender_email = os.getenv("GMAIL_EMAIL")
        self.sender_password = os.getenv("GMAIL_APP_PASSWORD")

    def send_csv_email(self, recipient_email: str, csv_path: str, domain: str) -> bool:
        """
        Send CSV file as email attachment

        Args:
            recipient_email: Email address to send to
            csv_path: Path to the CSV file
            domain: Domain name for email template

        Returns:
            bool: True if email sent successfully, False otherwise
        """
        if not self.sender_email or not self.sender_password:
            logger.error("Gmail credentials not configured in environment variables")
            return False

        try:
            # Read CSV to count keywords
            df = pd.read_csv(csv_path)
            keywords_count = len(df)

            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = recipient_email
            msg['Subject'] = f"{domain} {keywords_count} keywords"

            # Email body
            body = f"yo, here you go with the CSV of keywords data for {domain}"
            msg.attach(MIMEText(body, 'plain'))

            # Attach CSV file
            with open(csv_path, 'rb') as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f"attachment; filename={domain}-keywords-expanded.csv")
                msg.attach(part)

            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.sender_email, self.sender_password)

            text = msg.as_string()
            server.sendmail(self.sender_email, recipient_email, text)
            server.quit()

            logger.info(f"Email sent successfully to {recipient_email}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email to {recipient_email}: {str(e)}")
            return False

# Global email service instance
email_service = EmailService()
