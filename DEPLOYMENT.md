# Deployment Guide

## Deploying to Streamlit Cloud

### Prerequisites
- A GitHub account
- A Streamlit Cloud account (free tier available)

### Steps

1. **Push your code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
   git push -u origin main
   ```

2. **Deploy to Streamlit Cloud**
   - Go to [Streamlit Cloud](https://share.streamlit.io/)
   - Click "New app"
   - Select your repository and branch
   - Set the main file path to `app.py`
   - Click "Deploy!"

### Environment Variables
No environment variables are required for basic deployment.

### Resources
- Streamlit Cloud provides a free tier with limited resources
- For production use, consider upgrading to a paid plan for better performance

### Troubleshooting
- If the app crashes, check the logs in the Streamlit Cloud dashboard
- Make sure all dependencies are listed in `requirements.txt`
- Ensure your app works locally before deploying
