"""Simple email sending for transcription job completion."""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import structlog

from ..config import settings

logger = structlog.get_logger()


def send_transcription_ready_email(
    to_email: str,
    job_id: int,
    agent_id: int,
    source_filename: str,
) -> bool:
    """
    Send a plain-text email notifying the user that their transcription job is ready.
    Link format: {FRONTEND_BASE_URL}/agents/{agent_id}?job={job_id}

    Returns True if sent (or skipped because email disabled), False on send failure.
    """
    if not settings.EMAIL_ENABLED or not settings.SMTP_HOST:
        logger.info(
            "Email not configured or disabled, skipping transcription ready email",
            to_email=to_email,
            job_id=job_id,
            link=f"{settings.FRONTEND_BASE_URL.rstrip('/')}/agents/{agent_id}?job={job_id}",
        )
        return True

    link = f"{settings.FRONTEND_BASE_URL.rstrip('/')}/agents/{agent_id}?job={job_id}"
    subject = "Your transcription job is ready"
    body = (
        f"Your video transcription job is complete.\n\n"
        f"File: {source_filename}\n\n"
        f"View and download your results here:\n{link}\n"
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = settings.SMTP_FROM or settings.SMTP_USER or "noreply@localhost"
    msg["To"] = to_email
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
            if settings.SMTP_USER and settings.SMTP_PASSWORD:
                server.starttls()
                server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            server.sendmail(msg["From"], [to_email], msg.as_string())
        logger.info("Transcription ready email sent", to_email=to_email, job_id=job_id)
        return True
    except Exception as e:
        logger.error(
            "Failed to send transcription ready email",
            to_email=to_email,
            job_id=job_id,
            error=str(e),
            exc_info=True,
        )
        return False
