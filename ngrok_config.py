#!/usr/bin/env python3
"""
Global ngrok URL configuration
Update this URL when your Kaggle backend generates a new ngrok tunnel
"""

# Global ngrok URL configuration
# Update this URL when your Kaggle backend generates a new ngrok tunnel

CURRENT_NGROK_URL = "https://9ca1342d4666.ngrok-free.app"

def get_backend_url():
    """Get the full backend URL for API calls"""
    return f"{CURRENT_NGROK_URL}/generate"

def get_base_url():
    """Get the base ngrok URL"""
    return CURRENT_NGROK_URL

def print_current_config():
    """Print current configuration for debugging"""
    print(f"🌐 Current ngrok URL: {CURRENT_NGROK_URL}")
    print(f"🔗 Backend endpoint: {get_backend_url()}")
    print(f"🧪 Test endpoint: {CURRENT_NGROK_URL}/test")
    print(f"📊 Parameters endpoint: {CURRENT_NGROK_URL}/parameters")

def update_ngrok_url(new_url):
    """Update the ngrok URL"""
    global CURRENT_NGROK_URL
    CURRENT_NGROK_URL = new_url
    print(f"✅ Updated ngrok URL to: {CURRENT_NGROK_URL}")
    print(f"🔗 New backend endpoint: {get_backend_url()}")
    print(f"🧪 New test endpoint: {CURRENT_NGROK_URL}/test")

# Print current configuration when imported
if __name__ == "__main__":
    import sys
    import re
    # CLI: allow passing URL as argument or prompt
    if len(sys.argv) > 1:
        new_url = sys.argv[1]
    else:
        new_url = input("Enter new backend ngrok URL: ").strip()
    # Update the file itself
    with open(__file__, "r") as f:
        content = f.read()
    new_content = re.sub(
        r'CURRENT_NGROK_URL = "https://9ca1342d4666.ngrok-free.app"',
        f'CURRENT_NGROK_URL = "https://9ca1342d4666.ngrok-free.app"',
        content
    )
    with open(__file__, "w") as f:
        f.write(new_content)
    print(f"Ngrok URL updated to: {new_url}")
else:
    print(f"🌐 Using ngrok URL: {CURRENT_NGROK_URL}") 