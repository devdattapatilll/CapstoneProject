# Vercel Deployment Guide

## Quick Deploy Steps

### Step 1: Install Vercel CLI
```bash
npm install -g vercel
```

### Step 2: Login to Vercel
```bash
vercel login
```

### Step 3: Deploy from Project Directory
```bash
cd "c:\Projects Uploaded on github\CapstoneProject"
vercel --prod
```

## Manual Deploy via Dashboard

1. Go to [vercel.com/new](https://vercel.com/new)
2. Import Git Repository: `devdattapatilll/CapstoneProject`
3. Framework Preset: **Other** (Static)
4. Root Directory: `./`
5. Build Command: *(leave empty)*
6. Output Directory: `./`
7. Click **Deploy**

## Post-Deployment Configuration

### Set Environment Variables (if needed)
In Vercel Dashboard → Project Settings → Environment Variables:
- `ML_SERVICE_URL` = `https://civictrack-ml.onrender.com`

### Configure Custom Domain (Optional)
1. Vercel Dashboard → Project → Settings → Domains
2. Add your custom domain
3. Follow DNS configuration instructions

## Troubleshooting

| Issue | Solution |
|-------|----------|
| 404 errors on refresh | SPA routing configured in vercel.json |
| Styles not loading | Check static file paths in index.html |
| API calls failing | Update ML_SERVICE_URL in index.html |
| Build fails | Ensure index.html is at project root |

## Project Structure for Vercel
```
CapstoneProject/
├── index.html          ← Entry point (required)
├── styles.css          ← Static asset
├── vercel.json         ← Vercel config
├── .vercelignore       ← Exclude files from deploy
└── ...
```
