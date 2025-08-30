# Ngrok URL Setup Guide

## 🎯 Problem Solved
Instead of updating multiple hardcoded URLs throughout your codebase, you now have a **single global configuration** that manages all ngrok URLs.

## 📁 Files Created
- `ngrok_config.py` - Global configuration file
- `update_ngrok_url.py` - Script to update the URL
- `test_config.py` - Script to verify configuration

## 🚀 Quick Setup

### 1. Get Your Ngrok URL from Kaggle
When you start your Kaggle notebook, look for output like:
```
✅ Ngrok tunnel created: https://abc123.ngrok-free.app
```

### 2. Update the URL (One Time Per Run)
```bash
python update_ngrok_url.py https://abc123.ngrok-free.app
```

### 3. Verify Configuration
```bash
python test_config.py
```

### 4. Start Your Frontend
```bash
cd frontend && streamlit run app.py
```

## 🔧 How It Works

### Before (Multiple URLs to Update)
```python
# Had to update these in multiple places:
backend_url = "https://old-url.ngrok-free.app/generate"
kaggle_url = "https://old-url.ngrok-free.app/generate"
union_url = "https://old-url.ngrok-free.app/generate"
linoyts_url = "https://old-url.ngrok-free.app/generate"
```

### After (Single Global Configuration)
```python
# All functions now use:
from ngrok_config import get_backend_url
backend_url = get_backend_url()  # Gets URL from global config
```

## 📋 Benefits

✅ **One Update**: Change URL in one place  
✅ **No More 404s**: All functions use the same URL  
✅ **Easy Testing**: Built-in verification scripts  
✅ **Error Prevention**: Validates URL format  
✅ **Clear Messages**: Helpful error messages when URL not set  

## 🛠️ Troubleshooting

### URL Not Working?
1. Check if your Kaggle backend is running
2. Verify the ngrok tunnel is active
3. Test the URL manually in browser
4. Update with new URL: `python update_ngrok_url.py <new-url>`

### Getting 500 Errors?
1. Check Kaggle notebook for backend errors
2. Verify all dependencies are installed
3. Check if the model is loading correctly

### Getting 404 Errors?
1. Make sure the `/generate` endpoint exists
2. Check if the backend is running on port 8000
3. Verify the ngrok tunnel is pointing to the right port

## 📝 Example Workflow

```bash
# 1. Start your Kaggle notebook
# 2. Copy the ngrok URL from output
# 3. Update the configuration
python update_ngrok_url.py https://abc123.ngrok-free.app

# 4. Verify it's working
python test_config.py

# 5. Start your frontend
cd frontend && streamlit run app.py

# 6. Test in browser
# Draw something and generate!
```

## 🔄 For Each New Kaggle Run

1. **Start Kaggle notebook**
2. **Copy new ngrok URL**
3. **Run**: `python update_ngrok_url.py <new-url>`
4. **Start frontend**: `cd frontend && streamlit run app.py`

That's it! No more hunting for hardcoded URLs throughout your codebase. 