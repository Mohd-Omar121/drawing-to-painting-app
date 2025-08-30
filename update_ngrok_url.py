#!/usr/bin/env python3
"""
Automatic ngrok URL updater for the drawing-to-painting app
This script updates the frontend's ngrok_config.py with the latest backend URL

Usage:
    python update_ngrok_url.py [ngrok-url]
    python update_ngrok_url.py --help
    python update_ngrok_url.py --auto

Examples:
    python update_ngrok_url.py https://abc123.ngrok-free.app
    python update_ngrok_url.py --auto
"""

import os
import sys
import re
import requests
import time
from pathlib import Path

def show_help():
    """Show help information"""
    print("🚀 Automatic ngrok URL Updater")
    print("=" * 50)
    print("Updates the frontend's ngrok_config.py with the latest backend URL")
    print()
    print("Usage:")
    print("  python update_ngrok_url.py [ngrok-url]")
    print("  python update_ngrok_url.py --help")
    print("  python update_ngrok_url.py --auto")
    print()
    print("Arguments:")
    print("  ngrok-url    The new ngrok URL to use")
    print("  --help       Show this help message")
    print("  --auto       Automatically detect and update from backend")
    print()
    print("Examples:")
    print("  python update_ngrok_url.py https://abc123.ngrok-free.app")
    print("  python update_ngrok_url.py --auto")
    print()
    print("If no arguments are provided, the script will:")
    print("1. Try to auto-detect the URL from your current backend")
    print("2. Prompt you to enter a URL manually")
    print("3. Update the frontend configuration")


def _update_single_config_file(config_file: Path, new_url: str) -> bool:
    """Update a single ngrok_config.py with the provided URL"""
    if not config_file.exists():
        print(f"⚠️ Config file not found (skipping): {config_file}")
        return False

    try:
        with open(config_file, "r") as f:
            content = f.read()

        new_content = re.sub(
            r'CURRENT_NGROK_URL = ".*"',
            f'CURRENT_NGROK_URL = "{new_url}"',
            content
        )

        with open(config_file, "w") as f:
            f.write(new_content)

        print(f"✅ Updated: {config_file}")
        return True
    except Exception as e:
        print(f"❌ Error updating {config_file}: {e}")
        return False


def update_ngrok_config(new_url):
    """Update the ngrok URL in BOTH frontend/ngrok_config.py and root ngrok_config.py"""
    targets = [
        Path("frontend/ngrok_config.py"),
        Path("ngrok_config.py"),
    ]

    any_updated = False
    for cfg in targets:
        updated = _update_single_config_file(cfg, new_url)
        any_updated = any_updated or updated

    if any_updated:
        print(f"🌐 New URL applied: {new_url}")
    else:
        print("❌ No configuration files were updated")
    return any_updated

def get_ngrok_url_from_backend(backend_url):
    """Try to get ngrok URL from backend's /ngrok-url endpoint"""
    try:
        # Remove /generate if present to get base URL
        base_url = backend_url.replace("/generate", "")
        ngrok_endpoint = f"{base_url}/ngrok-url"
        
        print(f"🔍 Fetching ngrok URL from: {ngrok_endpoint}")
        
        response = requests.get(ngrok_endpoint, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("ngrok_url")
        else:
            print(f"⚠️ Backend returned status {response.status_code}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Could not fetch from backend: {e}")
        return None

def validate_url(url):
    """Basic URL validation"""
    if not url:
        return False
    
    # Check if it's a valid ngrok URL
    if "ngrok" in url and ("http://" in url or "https://" in url):
        return True
    
    # Check if it's a localhost URL
    if "localhost" in url or "127.0.0.1" in url:
        return True
    
    return False

def main():
    # Check for help flag
    if "--help" in sys.argv or "-h" in sys.argv:
        show_help()
        return
    
    print("🚀 Automatic ngrok URL Updater")
    print("=" * 50)
    
    # Check if URL was provided as argument
    if len(sys.argv) > 1:
        new_url = sys.argv[1].strip()
        
        # Skip help and auto flags
        if new_url.startswith("--"):
            if new_url == "--auto":
                new_url = None  # Will trigger auto-detection
            else:
                print(f"❌ Unknown flag: {new_url}")
                show_help()
                return
        else:
            print(f"📝 Using provided URL: {new_url}")
    else:
        new_url = None
    
    # Auto-detection mode
    if new_url is None:
        print("🔍 Attempting to auto-detect ngrok URL from backend...")
        
        # Check if there's a current config to try
        try:
            # Try importing from frontend first
            try:
                from frontend.ngrok_config import get_base_url as get_base_url_frontend
                current_backend = get_base_url_frontend() + "/generate"
            except Exception:
                from ngrok_config import get_base_url as get_base_url_root
                current_backend = get_base_url_root() + "/generate"
            print(f"🔗 Current backend: {current_backend}")
            
            ngrok_url = get_ngrok_url_from_backend(current_backend)
            if ngrok_url and validate_url(ngrok_url):
                new_url = ngrok_url
                print(f"✅ Retrieved URL from backend: {new_url}")
            else:
                print("❌ Could not retrieve URL from backend")
                new_url = input("Enter new backend ngrok URL: ").strip()
        except ImportError:
            print("⚠️ Could not import current config")
            new_url = input("Enter new backend ngrok URL: ").strip()
    
    # Validate the URL
    if not validate_url(new_url):
        print("❌ Invalid URL format. Please provide a valid ngrok or localhost URL.")
        print("💡 Example: https://abc123.ngrok-free.app")
        return
    
    # Update the config files
    if update_ngrok_config(new_url):
        print("\n🎉 Frontend ngrok URL updated successfully!")
        print(f"🌐 New backend URL: {new_url}")
        print(f"🔗 Test the connection: {new_url}/test")
        
        # Test the connection
        print("\n🧪 Testing backend connection...")
        try:
            response = requests.get(f"{new_url}/test", timeout=10)
            if response.status_code == 200:
                print("✅ Backend is responding!")
                data = response.json()
                if "message" in data:
                    print(f"📝 Message: {data['message']}")
            else:
                print(f"⚠️ Backend returned status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Could not connect to backend: {e}")
    else:
        print("❌ Failed to update ngrok URL")

if __name__ == "__main__":
    main() 