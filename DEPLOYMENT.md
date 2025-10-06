# 🚀 Search Tool Cloud Deployment Guide

## **Super Simple Railway Deployment**

This guide shows you how to deploy your search tool to Railway with background workers in just a few clicks.

## **What You Get**
✅ **Main API Service** - Your FastAPI application
✅ **Background Worker** - Processes keyword expansion jobs
✅ **Redis Database** - Job queue and caching
✅ **PostgreSQL Database** - Your keyword data
✅ **Auto-scaling** - Workers scale automatically
✅ **HTTPS/SSL** - Automatic certificate management

## **Step 1: Push to GitHub**
```bash
git add .
git commit -m "Add Railway deployment configuration"
git push origin main
```

## **Step 2: Deploy to Railway**
1. Go to [railway.app](https://railway.app) and sign up/login
2. Click **"New Project"**
3. Choose **"Deploy from GitHub"**
4. Connect your repository
5. Railway will automatically detect the configuration and deploy

## **Step 3: Set Environment Variables**
In your Railway dashboard, go to **"Variables"** and add:

```
GOOGLE_API_KEYS=your_api_keys_separated_by_commas
GOOGLE_ADS_CUSTOMER_ID=your_customer_id
DATABASE_URL=postgresql://... (Railway provides this)
REDIS_URL=redis://... (Railway provides this)
```

## **Step 4: Test Your Deployment**

### **Check API Status**
```bash
curl https://your-app.railway.app/
# Should return: {"message": "Welcome to the Market Intelligence API"}
```

### **Test Background Jobs**
```bash
curl -X POST "https://your-app.railway.app/v1/keywords/expanded-async" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "domain=https://example.com"
```

## **How It Works**

### **Main App (search-tool-api)**
- Runs your FastAPI application on port 8000
- Handles API requests
- Queues background jobs to Redis

### **Worker (search-tool-worker)**
- Runs continuously listening for jobs
- Processes keyword expansion tasks
- Scales automatically based on queue size

### **Job Flow**
1. User submits job via `/v1/keywords/expanded-async`
2. API queues job in Redis
3. Worker picks up job and processes it
4. Results are saved and email sent (if requested)

## **Monitoring Your Deployment**

### **Check Job Status**
```bash
curl https://your-app.railway.app/v1/jobs/YOUR_JOB_ID
```

### **View Logs**
- Railway dashboard → **"Logs"** tab
- Check both API and worker logs

### **Database**
- Railway provides PostgreSQL dashboard
- Access via **"Data"** tab in Railway

## **Scaling**

Railway automatically scales your worker service based on:
- Queue size
- CPU usage
- Memory usage

No configuration needed!

## **Cost Estimate**
- **Railway**: ~$5-10/month for small usage
- **Railway PostgreSQL**: ~$5/month
- **Railway Redis**: ~$3/month

## **Troubleshooting**

### **Worker Not Processing Jobs**
1. Check Railway dashboard logs for errors
2. Verify Redis connection in environment variables
3. Check if worker service is running

### **API Connection Issues**
1. Verify DATABASE_URL format
2. Check PostgreSQL credentials
3. Ensure all required environment variables are set

### **Background Jobs Failing**
1. Check worker logs for specific errors
2. Verify Google Ads API credentials
3. Check database connectivity

## **Need Help?**

1. Check Railway [documentation](https://docs.railway.app)
2. Review the logs in your Railway dashboard
3. Test locally first: `python api/main.py`

## **Local Development**
```bash
# Install dependencies
pip install -r requirements.txt

# Run API locally
python api/main.py

# Run worker locally (in another terminal)
python worker.py
```

That's it! Your search tool is now deployed with background workers. 🎉
