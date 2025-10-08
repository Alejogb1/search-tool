import os
import logging
import base64
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

        # Debug logging for troubleshooting
        logger.info(f"DEBUG: SENDGRID_API_KEY set: {'YES' if self.sendgrid_api_key else 'NO'}")
        if self.sendgrid_api_key:
            logger.info(f"DEBUG: SENDGRID_API_KEY prefix: {self.sendgrid_api_key[:10]}...")
        logger.info(f"DEBUG: FROM_EMAIL: {self.from_email}")

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
                subject=f"{domain} {keywords_count:,} keyword dataset ready",
                html_content=f"""
                <div itemscope itemtype="https://schema.org/EmailMessage">
                    <meta itemprop="description" content="Automated keyword data delivery  transactional report generated from your recent request." />
                    <div style="font-family: Arial, sans-serif; color: #202124; max-width: 600px; margin: auto;">
                                                
                        <p>Hey {recipient_email.split('@')[0]},</p>

                        <p>I’m Alejo  an ML/AI engineering student from Argentina, researching <strong>machine learning interpretability and search intelligence systems</strong>. 
                        You (or your team) requested a keyword data extraction for <strong>{domain}</strong> on 
                        <strong>{(pd.Timestamp.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')}</strong>. 
                        This email contains the automatically generated dataset for your request.</p>

                        <hr style="border: none; border-top: 1px solid #ddd; margin: 20px 0;">

                        <ul>
                            <li><strong>Domain analyzed:</strong> {domain}</li>
                            <li><strong>Total keywords processed:</strong> {keywords_count:,}</li>
                            <li><strong>Data source:</strong> Google Ads verified metrics + search partner datasets</li>
                            <li><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</li>
                        </ul>

                        <p>The attached file contains raw keyword level data with search volume and intent signals. 
                        This message is <strong>purely transactional</strong> and automatically generated upon your request on the deployed URL, as part of an open search intelligence project  
                        no marketing or tracking links are included.</p>

                        <p>If you’d like more context on how the dataset was built or the pipeline used, I’d be glad to share technical documentation 
                        (sampling logic, validation steps, and ranking metrics). It’s part of my ongoing academic research in interpretable search models.</p>

                        <p>Thanks for checking it out,<br>
                        <strong>Alejo García</strong><br>
                        ML / AI Engineering Student – Argentina<br>
                        <a href="https://github.com/Alejogb1/search-tool">github.com/Alejogb1/search-tool</a><br>
                        <a href="https://x.com/alejogb1">x.com/alejogb1</a> | 
                        <a href="https://www.linkedin.com/in/alejo-garcia-6b232129b/">LinkedIn</a></p>

                        <p style="font-size: 0.85em; color: #5f6368; margin-top: 30px;">
                            This email was sent automatically as a data delivery confirmation for a requested report.
                            If you didn’t request this, you can safely ignore it  no personal data is stored.
                        </p>
                    </div>
                </div>
                """
            )

            # Attach CSV file
            with open(csv_path, 'rb') as f:
                file_data = f.read()

            # Base64 encode for SendGrid
            encoded_file = FileContent(base64.b64encode(file_data).decode('ascii'))
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
                logger.error(f"DEBUG: SendGrid request headers: {response.headers}")
                logger.error(f"DEBUG: SendGrid API key used: {self.sendgrid_api_key[:15]}..." if self.sendgrid_api_key else "DEBUG: No API key set")
                return False

        except Exception as e:
            logger.error(f"Failed to send email to {recipient_email}: {str(e)}")
            return False

# Global email service instance
email_service = EmailService()
