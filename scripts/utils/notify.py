#!/usr/bin/env python3
"""
notify.py — Send email/SMS notifications for pipeline job events.

For SMS: sends to ALL major carrier gateways simultaneously. Only the
correct carrier delivers; the rest silently fail. No carrier selection needed.

Usage (from SLURM scripts):
    python scripts/utils/notify.py \
        --config config/my_config.yaml \
        --method proseg \
        --sample-id XETG00143__0032645 \
        --status success \
        --job-id $SLURM_JOB_ID \
        --elapsed "45m12s"
"""

import os
import sys
import yaml
import argparse
import subprocess
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from datetime import datetime

# All major US carrier email-to-SMS gateways
SMS_GATEWAYS = [
    "txt.att.net",               # AT&T
    "tmomail.net",               # T-Mobile
    "vtext.com",                 # Verizon
    "messaging.sprintpcs.com",   # Sprint / T-Mobile
    "msg.fi.google.com",         # Google Fi
    "email.uscc.net",            # US Cellular
    "sms.myboostmobile.com",     # Boost Mobile
    "vmobl.com",                 # Virgin Mobile
    "mmst5.tracfone.com",        # Tracfone
    "mymetropcs.com",            # Metro by T-Mobile
]


def load_notification_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("notifications", {})


def build_mime(to_addr: str, subject: str, body: str, attachment_path: str = None):
    """Build a MIME message, optionally with a PDF attachment."""
    from_addr = f"segmentation-pipeline@{os.uname().nodename}"
    if attachment_path and os.path.exists(attachment_path):
        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["To"] = to_addr
        msg["From"] = from_addr
        msg.attach(MIMEText(body))
        fname = os.path.basename(attachment_path)
        with open(attachment_path, "rb") as f:
            part = MIMEApplication(f.read(), _subtype="pdf")
            part.add_header("Content-Disposition", "attachment", filename=fname)
            msg.attach(part)
    else:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["To"] = to_addr
        msg["From"] = from_addr
    return msg


def send_via_sendmail(to_addr: str, subject: str, body: str, attachment_path: str = None) -> bool:
    msg = build_mime(to_addr, subject, body, attachment_path)
    try:
        proc = subprocess.run(
            ["sendmail", "-t"],
            input=msg.as_string(),
            capture_output=True, text=True, timeout=30,
        )
        return proc.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def send_via_mail(to_addr: str, subject: str, body: str, attachment_path: str = None) -> bool:
    cmd = ["mail", "-s", subject]
    if attachment_path and os.path.exists(attachment_path):
        # Try mailx -a flag (works on most Linux systems)
        cmd += ["-a", attachment_path]
    cmd.append(to_addr)
    try:
        proc = subprocess.run(
            cmd,
            input=body,
            capture_output=True, text=True, timeout=30,
        )
        return proc.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def send_one(to_addr: str, subject: str, body: str, attachment_path: str = None) -> bool:
    return send_via_sendmail(to_addr, subject, body, attachment_path) or \
           send_via_mail(to_addr, subject, body, attachment_path)


def send_sms(phone: str, subject: str, body: str):
    """Send to all carrier gateways (plain text only — no attachment for SMS)."""
    for gateway in SMS_GATEWAYS:
        addr = f"{phone}@{gateway}"
        send_one(addr, subject, body)


def build_message(method, sample_id, status, job_id, elapsed, node):
    tag = "COMPLETED" if status == "success" else "FAILED"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    subject = f"[seg] {method} {tag} — {sample_id}"

    body = (
        f"Segmentation Pipeline\n"
        f"{'=' * 35}\n"
        f"Method:    {method}\n"
        f"Sample:    {sample_id}\n"
        f"Status:    {tag}\n"
        f"Job ID:    {job_id}\n"
        f"Node:      {node}\n"
        f"Elapsed:   {elapsed}\n"
        f"Time:      {ts}\n"
    )
    if status != "success":
        body += f"\nCheck logs: logs/seg_{method}_{sample_id}_{job_id}.{{out,err}}\n"

    sms_body = f"{method} {tag}\n{sample_id}\nJob {job_id} | {elapsed}"

    return subject, body, sms_body


def main():
    parser = argparse.ArgumentParser(description="Send pipeline notifications")
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--status", required=True, choices=["success", "failed"])
    parser.add_argument("--job-id", default="unknown")
    parser.add_argument("--elapsed", default="unknown")
    parser.add_argument("--attachment", default=None, help="Path to file to attach (e.g. QC PDF report)")
    args = parser.parse_args()

    notif = load_notification_config(args.config)
    if not notif:
        return

    email = notif.get("email", "")
    phone = notif.get("phone", "")

    if not email and not phone:
        return

    node = os.uname().nodename
    subject, body, sms_body = build_message(
        args.method, args.sample_id, args.status,
        args.job_id, args.elapsed, node,
    )

    attachment = args.attachment if args.attachment and os.path.exists(args.attachment) else None
    if args.attachment and not attachment:
        print(f"[NOTIFY] attachment not found, sending without: {args.attachment}")

    if email:
        if send_one(email, subject, body, attachment):
            suffix = f" (+{os.path.basename(attachment)})" if attachment else ""
            print(f"[NOTIFY] email → {email}{suffix}")

    if phone:
        send_sms(phone, subject, sms_body)
        print(f"[NOTIFY] text → {phone}")


if __name__ == "__main__":
    main()
