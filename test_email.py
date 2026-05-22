from email_handler import check_mail, send_reply
import os
from dotenv import load_dotenv

load_dotenv()

user = os.getenv("EMAIL_USER")
pw = os.getenv("EMAIL_PASS")
imap_s = os.getenv("IMAP_SERVER")
smtp_s = os.getenv("SMTP_SERVER")

print("--- JARVIS EMAIL DIAGNOSTIC ---")
print(f"User: {user}")
print(f"IMAP: {imap_s}")
print(f"SMTP: {smtp_s}")

print("\n[STEP 1] Testing SMTP (Startup Greeting)...")
send_reply(user, pw, smtp_s, user, "Diagnostic", "Jarvis Diagnostic: SMTP is working.")

print("\n[STEP 2] Testing IMAP (Polling)...")
task = check_mail(user, pw, imap_s)
if task:
    print(f"\n[SUCCESS] Found task: {task['objective']}")
else:
    print("\n[INFO] No unseen tasks found (or polling worked but inbox was checked).")

print("\n--- DIAGNOSTIC COMPLETE ---")
print("Check jarvis_errors.log for details.")
