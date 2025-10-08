import os
import logging
import pandas as pd
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import (
    Mail, Attachment, FileContent, FileName, FileType, Disposition
)

logger = logging.getLogger(__name__)

class EmailService:
    def __init__(self):
        self.sendgrid_api_key = os.getenv("SENDGRID_API_KEY")
        self.from_email = os.getenv("FROM_EMAIL", "noreply@yourdomain.com")

    def send_csv_email(self, recipient_email: str, csv_path: str, domain: str) -> bool:
        """
        Send CSV file as email attachment using SendGrid

        Args:
            recipient_email: Email address to send to
            csv_path: Path to the CSV file
            domain: Domain name for email template

        Returns:
            bool: True if email sent successfully, False otherwise
        """
        if not self.sendgrid_api_key:
            logger.error("SendGrid API key not configured in environment variables")
            return False

        try:
            # Read CSV to count keywords
            df = pd.read_csv(csv_path)
            keywords_count = len(df)

            # Create email
            message = Mail(
                from_email=self.from_email,
                to_emails=recipient_email,
                subject=f"{domain} {keywords_count} keywords",
                html_content=f"""
                <h2>Keyword Research Results</h2>
                <p>Here are your keyword research results for <strong>{domain}</strong></p>
                <p><strong>Total Keywords:</strong> {keywords_count:,}</p>
                <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <br>
                <p>Best regards,<br>Search Tool Team</p>
                """
            )

            # Attach CSV file
            with open(csv_path, 'rb') as f:
                file_data = f.read()

            encoded_file = FileContent(file_data)
            attachment = Attachment(
                file_content=encoded_file,
                file_name=f"{domain}-keywords-expanded.csv",
                file_type="text/csv",
                disposition=Disposition("attachment")
            )

            message.attachment = attachment

            # Send email
            sg = SendGridAPIClient(self.sendgrid_api_key)
            response = sg.send(message)

            if response.status_code in [200, 201, 202]:
                logger.info(f"Email sent successfully to {recipient_email}")
                return True
            else:
                logger.error(f"SendGrid API error: {response.status_code} - {response.body}")
                return False

        except Exception as e:
            logger.error(f"Failed to send email to {recipient_email}: {str(e)}")
            return False

# Global email service instance
email_service = EmailService()
