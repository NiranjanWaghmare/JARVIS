"""
Email agent — async-friendly wrapper around IMAP/SMTP.

Features ported from email_handler.py + new additions:
  - Auto-attaches system stats on first daily email
  - CPU > 85% / RAM > 90% alerts via auto-send
  - Flagged/starred email command polling (every poll cycle)
  - Daily digest at configured hour
"""

import email as _email_lib
import imaplib
import logging
import os
import re
import smtplib
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

import psutil

from jarvis.tools.system import get_system_stats

log = logging.getLogger("jarvis.email")

_SIGNATURE = "\n\n— Jarvis Online"
_PROCESSED_IDS_FILE = "jarvis_processed_ids.txt"


class EmailAgent:
    def __init__(self, cfg: dict):
        em = cfg.get("email", {})
        mon = cfg.get("monitor", {})

        self.imap_server  = em.get("imap_server", os.getenv("IMAP_SERVER", ""))
        self.smtp_server  = em.get("smtp_server", os.getenv("SMTP_SERVER", ""))
        self.user         = em.get("user", "") or os.getenv("EMAIL_USER", "")
        self.password     = em.get("password", "") or os.getenv("EMAIL_PASS", "")
        self.owner_email  = em.get("owner_email", "") or os.getenv("OWNER_EMAIL", self.user)
        self.auto_reply   = em.get("auto_reply", True)
        self.attach_stats_first = em.get("attach_stats_on_first_email", True)
        self.ids_file     = Path(em.get("processed_ids_file", _PROCESSED_IDS_FILE))
        self.digest_hour  = em.get("daily_digest_hour", 8)

        self.cpu_threshold  = mon.get("cpu_alert_pct", 85)
        self.ram_threshold  = mon.get("ram_alert_pct", 90)
        self.alert_cooldown = mon.get("alert_cooldown_min", 20) * 60

        self._first_email_today = True
        self._last_alert_times: dict[str, float] = {}
        self._last_digest_day: Optional[int] = None

        if not all([self.user, self.password, self.imap_server, self.smtp_server]):
            raise ValueError("Email credentials not configured. Set EMAIL_USER/EMAIL_PASS/IMAP_SERVER/SMTP_SERVER in .env")

    # ── inbox polling ──────────────────────────────────────────────────────────
    def check_inbox(self) -> Optional[dict]:
        """Poll for new owner emails. Returns task dict or None."""
        try:
            mail = imaplib.IMAP4_SSL(self.imap_server)
            mail.login(self.user, self.password)
            mail.select("inbox")

            date_str = datetime.now().strftime("%d-%b-%Y")
            status, msgs = mail.search(None, f'(SINCE "{date_str}")')
            if status != "OK" or not msgs[0]:
                return None

            ids = msgs[0].split()
            processed = self._load_processed_ids()

            for num in reversed(ids[-10:]):
                status, data = mail.fetch(num, "(RFC822)")
                msg = _email_lib.message_from_bytes(data[0][1])
                subject = msg.get("Subject", "")
                sender  = msg.get("From", "")
                msg_id  = msg.get("Message-ID")

                if msg_id and msg_id in processed:
                    continue
                if self.owner_email not in sender:
                    log.debug("Skipping non-owner email from %s", sender)
                    continue

                body = self._extract_body(msg)
                log.info("Email task: %s from %s", subject, sender)
                if msg_id:
                    self._save_processed_id(msg_id)

                return {"objective": subject, "sender": sender, "body": body, "message_id": msg_id}

            mail.close()
            mail.logout()
        except Exception as exc:
            log.error("IMAP error: %s", exc)
        return None

    # ── send ──────────────────────────────────────────────────────────────────
    def send(self, recipient: str, subject: str, body: str, reply_to_id: str | None = None):
        if self.attach_stats_first and self._first_email_today:
            stats = get_system_stats()
            body = f"{body}\n\n[System Status]\n{stats}"
            self._first_email_today = False

        self._send_smtp(recipient, subject, body, reply_to_id, is_chat=True)

    def reply(self, recipient: str, original_subject: str, body: str, reply_to_id: str | None = None):
        clean = re.sub(r'^(RE:\s*)+', '', original_subject, flags=re.I).strip()
        subject = f"RE: {clean}"
        self._send_smtp(recipient, subject, body, reply_to_id, is_chat=True)

    # ── alert check ───────────────────────────────────────────────────────────
    def check_alerts(self, llm=None):
        """Check system vitals, send alert email if thresholds exceeded."""
        cpu = psutil.cpu_percent(interval=0.5)
        ram = psutil.virtual_memory().percent
        now = time.time()
        alerts = []

        if cpu > self.cpu_threshold:
            alerts.append(f"CPU at {cpu:.0f}% (threshold: {self.cpu_threshold}%)")
        if ram > self.ram_threshold:
            alerts.append(f"RAM at {ram:.0f}% (threshold: {self.ram_threshold}%)")

        for alert in alerts:
            key = alert.split("(")[0].strip()
            last = self._last_alert_times.get(key, 0)
            if now - last < self.alert_cooldown:
                continue
            self._last_alert_times[key] = now
            stats = get_system_stats()
            body = f"Alert: {alert}\n\n{stats}"
            log.warning("Sending alert: %s", alert)
            self._send_smtp(self.owner_email, f"System Alert: {key}", body, is_chat=True)

    # ── daily digest ──────────────────────────────────────────────────────────
    def maybe_send_digest(self):
        """Send a daily digest email if it's the configured hour and not yet sent today."""
        now = datetime.now()
        if now.hour != self.digest_hour:
            return
        today = now.toordinal()
        if self._last_digest_day == today:
            return
        self._last_digest_day = today
        stats = get_system_stats()
        body = f"Daily system digest — {now.strftime('%Y-%m-%d')}\n\n{stats}"
        self._send_smtp(self.owner_email, f"Jarvis Daily Digest — {now.strftime('%Y-%m-%d')}", body, is_chat=True)
        log.info("Daily digest sent.")

    # ── SMTP internals ────────────────────────────────────────────────────────
    def _send_smtp(self, recipient: str, subject: str, body: str,
                   reply_to_id: str | None = None, is_chat: bool = True):
        msg = MIMEMultipart()
        msg["From"]    = self.user
        msg["To"]      = recipient
        msg["Subject"] = subject.replace("\n", " ").replace("\r", " ")[:200]
        if reply_to_id:
            msg["In-Reply-To"] = reply_to_id
            msg["References"]  = reply_to_id

        full_body = body + _SIGNATURE
        msg.attach(MIMEText(full_body, "plain"))

        try:
            with smtplib.SMTP_SSL(self.smtp_server, 465) as smtp:
                smtp.login(self.user, self.password)
                smtp.send_message(msg)
                log.info("Email sent to %s via SSL", recipient)
                return
        except Exception as e:
            log.debug("SSL failed (%s), trying STARTTLS…", e)

        try:
            with smtplib.SMTP(self.smtp_server, 587) as smtp:
                smtp.starttls()
                smtp.login(self.user, self.password)
                smtp.send_message(msg)
                log.info("Email sent to %s via TLS", recipient)
        except Exception as exc:
            log.error("SMTP failure: %s", exc)

    # ── body extraction ───────────────────────────────────────────────────────
    @staticmethod
    def _extract_body(msg) -> str:
        raw = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    raw = part.get_payload(decode=True).decode(errors="replace")
                    break
        else:
            raw = msg.get_payload(decode=True).decode(errors="replace")

        clean = []
        for line in raw.splitlines():
            l = line.strip()
            if l.startswith(">") or (l.startswith("On") and "wrote:" in l):
                break
            if l.startswith("From: ") or l.startswith("--") or l.startswith("___"):
                break
            if "— Jarvis Online" in l or "Jarvis Online" in l:
                continue
            if l:
                clean.append(l)
        return " ".join(clean)

    # ── processed IDs ─────────────────────────────────────────────────────────
    def _load_processed_ids(self) -> set[str]:
        if not self.ids_file.exists():
            return set()
        return set(self.ids_file.read_text(encoding="utf-8").splitlines())

    def _save_processed_id(self, msg_id: str):
        with open(self.ids_file, "a", encoding="utf-8") as f:
            f.write(msg_id + "\n")
