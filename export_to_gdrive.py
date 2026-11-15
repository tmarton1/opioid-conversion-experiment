#!/usr/bin/env python3
"""
Export figures, markdowns, and PDFs to Google Drive.

Uploads everything into a new "Opioid Conversion Figures" subfolder inside
the parent folder: https://drive.google.com/drive/folders/1lEi8O_0VnCfbmFJUax-x4rG6Acp5pFeI

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIRST-TIME SETUP (one-time only)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Go to https://console.cloud.google.com/
2. Create a new project (or select an existing one).
3. In the sidebar: APIs & Services → Enable APIs → search for
   "Google Drive API" → Enable it.
4. APIs & Services → Credentials → + Create Credentials → OAuth client ID
   → Application type: Desktop app → Download JSON → save as
   "credentials.json" in this directory.
5. Install dependencies:
       pip install google-api-python-client google-auth-oauthlib

The first time you run this script, a browser window will open asking
you to authorise access.  The token is cached in "token.json" so you
won't be asked again.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Usage:
    python export_to_gdrive.py              # upload everything
    python export_to_gdrive.py --dry-run    # print what would be uploaded
    python export_to_gdrive.py --replace    # delete existing subfolder first
"""

from __future__ import annotations

from common.config import FIGURES_DIR
import argparse
import mimetypes
import os
import sys
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.resolve()
PARENT_FOLDER_ID = "1lEi8O_0VnCfbmFJUax-x4rG6Acp5pFeI"
EXPORT_FOLDER_NAME = "Opioid Conversion Figures"

CREDENTIALS_FILE = REPO_ROOT / "credentials.json"
TOKEN_FILE = REPO_ROOT / "token.json"

SCOPES = ["https://www.googleapis.com/auth/drive"]

# ── Files to export ──────────────────────────────────────────────────────────
# Each entry is (local_path, destination_subfolder_inside_export_folder)
# Use "" to place the file at the top level of the export folder.

def collect_files() -> list[tuple[Path, str]]:
    """Return (local_path, gdrive_subfolder) pairs for every file to upload."""
    files: list[tuple[Path, str]] = []

    # Root-level markdown docs
    for md in ["README.md", "FIGURES.md", "STATISTICAL_METHODS.md",
               "FUTURE_IMPROVEMENTS.md", "CHANGING_ANNOTATIONS.md"]:
        p = REPO_ROOT / md
        if p.exists():
            files.append((p, ""))

    # Combined annotated PDF + summary tables
    for name in ["all_comprehensive_figures.pdf",
                 "accuracy_summary_tables.pdf",
                 "accuracy_summary_tables.html",
                 "figure_stats.json"]:
        p = REPO_ROOT / FIGURES_DIR / name
        if p.exists():
            files.append((p, ""))

    # Individual publication figures (PNG + PDF + HTML)
    pub = REPO_ROOT / FIGURES_DIR / "pub_figures"
    if pub.exists():
        for f in sorted(pub.iterdir()):
            if f.suffix in {".png", ".pdf", ".html"}:
                files.append((f, "figures"))

    return files


# ── Google Drive helpers ──────────────────────────────────────────────────────

def get_drive_service():
    """Authenticate and return an authorised Drive API service object."""
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        sys.exit(
            "Missing dependencies. Run:\n"
            "    pip install google-api-python-client google-auth-oauthlib"
        )

    if not CREDENTIALS_FILE.exists():
        sys.exit(
            f"credentials.json not found at {CREDENTIALS_FILE}\n"
            "See the FIRST-TIME SETUP instructions at the top of this script."
        )

    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_FILE), SCOPES
            )
            creds = flow.run_local_server(port=0)
        TOKEN_FILE.write_text(creds.to_json())

    return build("drive", "v3", credentials=creds)


def find_folder(service, name: str, parent_id: str) -> str | None:
    """Return the ID of an existing folder with the given name, or None."""
    query = (
        f"name = '{name}' and '{parent_id}' in parents "
        f"and mimeType = 'application/vnd.google-apps.folder' "
        f"and trashed = false"
    )
    result = service.files().list(q=query, fields="files(id, name)").execute()
    files = result.get("files", [])
    return files[0]["id"] if files else None


def create_folder(service, name: str, parent_id: str) -> str:
    """Create a Drive folder and return its ID."""
    meta = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    f = service.files().create(body=meta, fields="id").execute()
    return f["id"]


def get_or_create_folder(service, name: str, parent_id: str,
                         replace: bool = False) -> str:
    """Return (creating if necessary) the ID of a subfolder."""
    existing_id = find_folder(service, name, parent_id)
    if existing_id:
        if replace:
            print(f"  Deleting existing '{name}' folder…")
            service.files().delete(fileId=existing_id).execute()
        else:
            print(f"  Using existing '{name}' folder (id={existing_id})")
            return existing_id
    fid = create_folder(service, name, parent_id)
    print(f"  Created '{name}' folder (id={fid})")
    return fid


def upload_file(service, local_path: Path, parent_id: str) -> str:
    """Upload a single file and return its Drive ID."""
    from googleapiclient.http import MediaFileUpload

    mime = mimetypes.guess_type(str(local_path))[0] or "application/octet-stream"
    meta = {"name": local_path.name, "parents": [parent_id]}
    media = MediaFileUpload(str(local_path), mimetype=mime, resumable=True)
    f = service.files().create(body=meta, media_body=media, fields="id").execute()
    return f["id"]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be uploaded without doing it.")
    parser.add_argument("--replace", action="store_true",
                        help="Delete the existing export folder before uploading.")
    args = parser.parse_args()

    files = collect_files()
    if not files:
        sys.exit("No files found to upload.")

    # Compute subfolder set for summary
    subfolders = sorted({sf for _, sf in files if sf})
    print(f"\nFiles to upload: {len(files)}")
    print(f"  Top-level: {sum(1 for _, sf in files if not sf)}")
    for sf in subfolders:
        print(f"  {sf}/: {sum(1 for _, s in files if s == sf)}")

    if args.dry_run:
        print("\n[dry-run] Would upload:")
        for p, sf in files:
            dest = f"{EXPORT_FOLDER_NAME}/{sf}/{p.name}" if sf else f"{EXPORT_FOLDER_NAME}/{p.name}"
            print(f"  {dest}  ({p.stat().st_size / 1024:.0f} KB)")
        return

    print("\nAuthenticating with Google Drive…")
    service = get_drive_service()

    print(f"\nSetting up export folder '{EXPORT_FOLDER_NAME}'…")
    export_id = get_or_create_folder(
        service, EXPORT_FOLDER_NAME, PARENT_FOLDER_ID, replace=args.replace
    )

    # Create subfolders as needed
    subfolder_ids: dict[str, str] = {"": export_id}
    for sf in subfolders:
        subfolder_ids[sf] = get_or_create_folder(service, sf, export_id)

    print(f"\nUploading {len(files)} files…")
    errors = []
    for i, (local_path, sf) in enumerate(files, 1):
        parent_id = subfolder_ids[sf]
        dest = f"{EXPORT_FOLDER_NAME}/{sf}/{local_path.name}" if sf else f"{EXPORT_FOLDER_NAME}/{local_path.name}"
        size_kb = local_path.stat().st_size / 1024
        print(f"  [{i:2d}/{len(files)}] {dest}  ({size_kb:.0f} KB)", end="", flush=True)
        try:
            upload_file(service, local_path, parent_id)
            print(" ✓")
        except Exception as e:
            print(f" ✗  {e}")
            errors.append((dest, str(e)))

    print()
    if errors:
        print(f"Completed with {len(errors)} error(s):")
        for dest, msg in errors:
            print(f"  ✗ {dest}: {msg}")
        sys.exit(1)
    else:
        print(f"✓ All {len(files)} files uploaded to '{EXPORT_FOLDER_NAME}'")
        print(f"  https://drive.google.com/drive/folders/{PARENT_FOLDER_ID}")


if __name__ == "__main__":
    main()
