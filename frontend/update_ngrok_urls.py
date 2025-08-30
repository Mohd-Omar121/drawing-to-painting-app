"""
Helper script to update SDXL and FLUX ngrok URLs for the Streamlit frontend.
Run this script after you restart your Kaggle notebooks and get new ngrok URLs.
"""
import sys
import re
from ngrok_config import save_ngrok_urls, print_current_config

def main():
    ngrok_url = input("Enter new backend ngrok URL: ").strip()
    with open("ngrok_config.py", "r") as f:
        content = f.read()
    new_content = content
    if 'CURRENT_NGROK_URL' in content:
        new_content = re.sub(r'CURRENT_NGROK_URL = ".*"', f'CURRENT_NGROK_URL = "{ngrok_url}"', content)
    with open("ngrok_config.py", "w") as f:
        f.write(new_content)
    print(f"Ngrok URL updated to: {ngrok_url}")

if __name__ == "__main__":
    main()
