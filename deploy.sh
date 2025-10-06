#!/bin/bash

# 🚀 Simple Deployment Script for Search Tool
# This script helps you deploy to Railway with background workers

set -e  # Exit on any error

echo "🚀 Search Tool Deployment Script"
echo "================================="

# Check if we're in git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Error: Not in a git repository"
    exit 1
fi

# Check git status
if [[ -n $(git status --porcelain) ]]; then
    echo "📦 Found uncommitted changes. Adding and committing..."
    git add .
    git commit -m "Prepare for deployment: Add cloud configuration"
else
    echo "✅ Git working directory is clean"
fi

# Push to GitHub
echo "⬆️  Pushing to GitHub..."
if git push origin main; then
    echo "✅ Successfully pushed to GitHub"
else
    echo "❌ Failed to push to GitHub"
    exit 1
fi

echo ""
echo "🎉 Deployment preparation complete!"
echo ""
echo "Next steps:"
echo "1. Go to https://railway.app"
echo "2. Create a new project"
echo "3. Choose 'Deploy from GitHub'"
echo "4. Connect your repository"
echo "5. Railway will automatically deploy with your configuration"
echo ""
echo "📖 See DEPLOYMENT.md for detailed instructions"
echo ""
echo "🔧 After deployment, set these environment variables in Railway:"
echo "   - GOOGLE_API_KEYS"
echo "   - GOOGLE_ADS_CUSTOMER_ID"
echo "   - DATABASE_URL (provided by Railway)"
echo "   - REDIS_URL (provided by Railway)"
