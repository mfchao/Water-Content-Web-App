# Railway Deployment Guide

## Prerequisites
1. **GitHub Account** - Your code should be in a GitHub repository
2. **Railway Account** - Sign up at [railway.app](https://railway.app)

## Step-by-Step Deployment

### 1. Connect Railway to GitHub
1. Go to [railway.app](https://railway.app) and sign in
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your Water-Content-Web-App repository
5. Click "Deploy Now"

### 2. Configure Environment Variables
Railway will automatically detect this is a Python app. No additional configuration needed.

### 3. Deploy Backend First
1. Railway will automatically detect the Python backend
2. It will install dependencies from `api/requirements.txt`
3. The Flask app will start on the assigned port

### 4. Get Your Railway Domain
1. After deployment, go to your project dashboard
2. Click on your service
3. Copy the generated domain (e.g., `https://your-app-name.railway.app`)

### 5. Update Frontend API URL
1. Replace `https://your-app-name.railway.app` in `src/components/Preview.jsx` with your actual Railway domain
2. Commit and push the changes

### 6. Deploy Frontend
1. Railway will automatically build and deploy your React frontend
2. The frontend will be accessible at your Railway domain

## Configuration Files Explained

- **`railway.json`** - Railway-specific configuration
- **`Procfile`** - Tells Railway how to run the app
- **`runtime.txt`** - Specifies Python version
- **`api/requirements.txt`** - Python dependencies

## Benefits of Railway

✅ **No function size limits** - Can handle PyTorch models  
✅ **Better Python support** - Native Python environment  
✅ **Automatic scaling** - Handles traffic spikes  
✅ **Persistent storage** - Model files stay available  
✅ **Custom domains** - Can use your own domain  

## Troubleshooting

### Common Issues:
1. **Build fails** - Check `requirements.txt` for version conflicts
2. **Model not found** - Ensure `model.pth` is in the `api/` directory
3. **Port issues** - Railway automatically assigns ports

### Support:
- Railway Discord: [discord.gg/railway](https://discord.gg/railway)
- Documentation: [docs.railway.app](https://docs.railway.app)

## Cost
- **Free tier**: $5 credit monthly
- **Pay-as-you-use**: Only pay for actual usage
- **No hidden fees** - Transparent pricing 