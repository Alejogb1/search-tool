# 📧 Email Service Setup Guide

## **🚀 SendGrid Setup for Production Email**

This guide shows you how to set up SendGrid for reliable email delivery in your deployed search tool.

## **✅ Step 1: Create SendGrid Account**

1. **Go to:** [sendgrid.com](https://sendgrid.com)
2. **Sign up** for a free account
3. **Verify your email** address

## **✅ Step 2: Get SendGrid API Key**

1. **Login to SendGrid Dashboard**
2. **Navigate to:** Settings → API Keys
3. **Click:** "Create API Key"
4. **Name:** `search-tool-production`
5. **Permissions:** `Mail Send` (Full Access)
6. **Copy the API key** (you won't see it again!)

## **✅ Step 3: Configure Domain (Optional but Recommended)**

1. **Go to:** Settings → Sender Authentication
2. **Authenticate your domain** for better deliverability
3. **Or use:** SendGrid's default domain for testing

## **✅ Step 4: Update Render Environment Variables**

In your Render dashboard → **"Environment"** tab:

```bash
SENDGRID_API_KEY=SG.your_sendgrid_api_key_here
FROM_EMAIL=your-email@yourdomain.com
```

**⚠️ Important:**
- Replace `your_sendgrid_api_key_here` with your actual API key
- Use a verified sender email address
- The `FROM_EMAIL` should match your SendGrid verified sender

## **✅ Step 5: Redeploy Your Application**

1. **Go to:** Your Render service → **"Manual Deploy"** → **"Deploy latest commit"**
2. **Wait for deployment** to complete
3. **Test email functionality**

## **🧪 Test Email Functionality**

```bash
# Submit a background job with email
curl -X POST "https://search-tool-vwmv.onrender.com/v1/keywords/expanded-async" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "domain=https://example.com" \
  -d "email=your-email@example.com"
```

## **📊 SendGrid Free Tier Limits**

- **✅ 100 emails/day** (perfect for testing)
- **✅ No time limit** (free forever)
- **✅ Full API access** (all features included)

## **💰 Cost for Production**

- **$0/month** (first 100 emails free)
- **$19.95/month** (40,000 emails)
- **$89.95/month** (100,000 emails)

## **🔧 Troubleshooting**

### **Email Not Sending:**
1. **Check Render logs** for SendGrid API errors
2. **Verify API key** is correct and active
3. **Check FROM_EMAIL** is verified in SendGrid
4. **Ensure recipient email** is valid

### **Email in Spam:**
1. **Verify your domain** in SendGrid
2. **Use professional FROM_EMAIL** address
3. **Check email content** for spam triggers

## **🎯 What This Fixes:**

✅ **Network connectivity issues** - SendGrid API works in all deployment environments
✅ **Reliable delivery** - Professional email service with high deliverability
✅ **Better formatting** - HTML emails with proper styling
✅ **Analytics** - Track email opens, clicks, and bounces
✅ **Scalability** - Handle high volume email sending

## **🚀 Next Steps:**

1. **Set up SendGrid account** (5 minutes)
2. **Add environment variables** to Render
3. **Redeploy application**
4. **Test email functionality**

**Your email service will now work reliably in production!** 🎉

---

## **🔄 Alternative Email Services**

If SendGrid doesn't work for your needs:

### **Mailgun**
- **Free tier:** 5,000 emails/month
- **Setup:** Similar to SendGrid
- **API:** Well documented

### **Amazon SES**
- **Free tier:** 62,000 emails/month for first year
- **Cost:** $0.10 per 1,000 emails after
- **Reliability:** Excellent deliverability

### **Postmark**
- **Free tier:** 100 emails/month
- **Focus:** Transactional emails
- **Deliverability:** Excellent for business emails
