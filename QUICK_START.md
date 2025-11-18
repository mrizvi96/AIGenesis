# 🚀 Quick Start Guide

**Get AI-Powered Insurance Claims Processing running in 5 minutes!**

## ⚡ One-Click Setup (Recommended)

### GitHub Codespaces (Easiest)
Click here to open in your browser with everything pre-installed:

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo=true&ref=master&repo=mrizvi96/AIGenesis)

### Quick Install Script
```bash
# Clone and setup automatically
curl -sSL https://raw.githubusercontent.com/mrizvi96/AIGenesis/master/install.sh | bash
```

## 🛠️ Manual Setup (5 minutes)

### 1. Clone Repository
```bash
git clone https://github.com/mrizvi96/AIGenesis.git
cd AIGenesis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Qdrant Cloud (Free)
```bash
# 1. Visit https://cloud.qdrant.io/ (2 minutes)
# 2. Create free account and cluster
# 3. Copy your URL and API key

# Configure environment
cp .env.example .env
nano .env  # Add your Qdrant credentials
```

### 4. Run Demo
```bash
python demo.py --auto
```

## 🎮 Try It Now

### Interactive Demo
```bash
python demo.py
```

### Test Everything
```bash
python demo.py --test
```

### Check Installation
```bash
python demo.py --install
```

## 🌐 Access Points

Once running, access at:
- **Demo Interface**: `http://localhost:8501`
- **API Documentation**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`

## ✅ What You Get

- **🏥 Medical Claim Processing**: Analyze ER reports and medical documents
- **🚗 Auto Claims**: Process vehicle damage and accident reports
- **💰 Property Claims**: Handle home and property damage claims
- **🔍 Fraud Detection**: AI-powered risk assessment
- **📊 Analytics**: Real-time processing and insights

## 🆘 Need Help?

- **GitHub Issues**: [Create an issue](https://github.com/mrizvi96/AIGenesis/issues)
- **Email**: mohammad.rizvi@csuglobal.edu
- **Documentation**: See [README.md](README.md)

**Ready in 5 minutes or less! ⏱️**