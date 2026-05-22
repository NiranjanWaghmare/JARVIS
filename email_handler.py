import imaplib
import smtplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
import os
from datetime import datetime

def log_jarvis(msg):
    # Print to console AND append to file
    print(msg, flush=True)
    try:
        with open("jarvis_errors.log", "a", encoding="utf-8") as f:
            f.write(f"[{time.ctime()}] {msg}\n")
    except:
        pass

def check_mail(user, password, server, owner_email=None):
    """Poll for new emails with objectives in the subject."""
    try:
        log_jarvis(f"[DEBUG] Connecting to IMAP: {server}...")
        mail = imaplib.IMAP4_SSL(server)
        log_jarvis(f"[DEBUG] Logging in as: {user}...")
        mail.login(user, password)
        mail.select("inbox")
        
        # Use Message-ID tracking instead of UNSEEN
        date_str = datetime.now().strftime("%d-%b-%Y")
        log_jarvis(f"[DEBUG] Searching for emails since {date_str}...")
        status, messages = mail.search(None, f'(SINCE "{date_str}")')
        
        if status != "OK":
            log_jarvis(f"[DEBUG] Search failed with status: {status}")
            return None
        
        ids = messages[0].split()
        if not ids:
            return None
            
        # Load processed Message-IDs
        processed_ids = set()
        if os.path.exists("jarvis_processed_ids.txt"):
            with open("jarvis_processed_ids.txt", "r", encoding="utf-8") as f:
                processed_ids = set(line.strip() for line in f)
        
        # Optimize: Only look at the latest 10 emails
        for num in reversed(ids[-10:]):
            status, data = mail.fetch(num, '(RFC822)')
            msg = email.message_from_bytes(data[0][1])
            subject = msg["Subject"]
            sender = msg["From"]
            message_id = msg.get("Message-ID")
            
            if message_id and message_id in processed_ids:
                continue
            
            # Security Authorization: Only respond to emails sent by the owner
            if owner_email:
                if owner_email not in sender:
                    log_jarvis(f"[DEBUG] Skipping unauthorized email from {sender}.")
                    continue
            else:
                if user not in sender:
                    log_jarvis(f"[DEBUG] Skipping unauthorized email from {sender}.")
                    continue
            
            # EXTRACT AND CLEAN BODY
            raw_body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        raw_body = part.get_payload(decode=True).decode()
                        break
            else:
                raw_body = msg.get_payload(decode=True).decode()

            clean_lines = []
            for line in raw_body.splitlines():
                l = line.strip()
                if l.startswith(">") or (l.startswith("On") and "wrote:" in l): break
                if l.startswith("From: ") or l.startswith("--") or l.startswith("___"): break
                if "I'm always ready" in l or "Objective Accomplished" in l: continue
                if l.startswith('"Hey Sir') or l.startswith("Hey Sir"): continue
                if "-- J.A.R.V.I.S. Online" in l: continue
                if l: clean_lines.append(l)
            body = " ".join(clean_lines)

            log_jarvis(f"[EMAIL] Incoming: {subject} from {sender}")
            if message_id:
                with open("jarvis_processed_ids.txt", "a", encoding="utf-8") as f:
                    f.write(f"{message_id}\n")
            
            return {"objective": subject, "sender": sender, "body": body.strip(), "message_id": message_id}
            
        mail.close()
        mail.logout()
    except Exception as e:
        log_jarvis(f"[EMAIL ERROR] IMAP Failure: {e}")
    return None

def send_reply(user, password, server, recipient, objective, report, message_id=None, is_chat=False):
    """Send the final report back as an email."""
    try:
        import re
        log_jarvis(f"[DEBUG] Sending reply to {recipient} via {server}...")
        msg = MIMEMultipart()
        msg['From'] = user
        msg['To'] = recipient
        
        # Clean subject to prevent RE: RE: RE: compounding and ensure RFC compliance
        clean_subj = re.sub(r'^(RE:\s*)+', '', objective, flags=re.IGNORECASE)
        clean_subj = clean_subj.replace('\n', ' ').replace('\r', ' ').strip()
        
        final_subj = f"RE: {clean_subj}"
        if not is_chat:
            final_subj += " - Jarvis Report"
        
        msg['Subject'] = final_subj
            
        if message_id:
            msg['In-Reply-To'] = message_id
            msg['References'] = message_id
        
        signature = "\n\n-- J.A.R.V.I.S. Online"
        body = (report + signature) if is_chat else f"Objective Accomplished, Sir.\n\n{report}{signature}"
        msg.attach(MIMEText(body, 'plain'))
        
        # Try SSL on 465
        try:
            with smtplib.SMTP_SSL(server, 465) as smtp:
                smtp.login(user, password)
                smtp.send_message(msg)
                log_jarvis(f"[SUCCESS] Email delivered to {recipient}")
        except Exception as e:
            # Fallback to TLS on 587
            log_jarvis(f"[DEBUG] SSL failed ({e}), trying STARTTLS on 587...")
            with smtplib.SMTP(server, 587) as smtp:
                smtp.set_debuglevel(1)
                smtp.starttls()
                smtp.login(user, password)
                smtp.send_message(msg)
                log_jarvis(f"[SUCCESS] Email delivered via TLS to {recipient}")
                
    except Exception as e:
        log_jarvis(f"[EMAIL ERROR] SMTP Failure: {e}")
