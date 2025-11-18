# Google Cloud Setup Instructions for Enhanced AI Insurance Claims System

## 🚀 Quick Setup (5 minutes)

### Step 1: Enable Required APIs
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Select your project (or create a new one)
3. Navigate to **APIs & Services → Library**
4. Enable these APIs:
   - ✅ **Cloud Vision API**
   - ✅ **Cloud Speech-to-Text API**
   - ✅ **Cloud Natural Language API**
   - ✅ **Document AI API**
   - ✅ **Cloud Storage API**

### Step 2: Create Service Account
1. Navigate to **IAM & Admin → Service Accounts**
2. Click **"Create Service Account"**
3. Name: `ai-claims-service`
4. Role: `Cloud AI Services User`
5. Click **"Create and Continue"**
6. Click **"Done"**

### Step 3: Generate Service Account Key
1. Find your service account in the list
2. Click **"⋮" → Manage keys**
3. Click **"Add Key" → "Create new key"**
4. Select **JSON** format
5. Download the key file
6. **Save as**: `google-credentials.json` in your project root

### Step 4: Set Environment Variables
Add these lines to your `.env` file:

```bash
# Google Cloud Configuration
GOOGLE_APPLICATION_CREDENTIALS=./google-credentials.json
GOOGLE_CLOUD_PROJECT=your-project-id-here
GOOGLE_VISION_API_KEY=your-api-key-here

# Enhanced features
USE_ENHANCED_FEATURES=true
ENABLE_GOOGLE_CLOUD_APIS=true
```

## 🔧 Verification Commands

### Test Google Cloud Connection
```bash
# Test Vision API
curl -X POST "https://vision.googleapis.com/v1/images:annotate?key=YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [{
      "image": {"source": {"imageUri": "https://example.com/image.jpg"}},
      "features": [{"type": "LABEL_DETECTION"}]
    }]
  }'

# Test Speech-to-Text
curl -X POST "https://speech.googleapis.com/v1/speech:recognize?key=YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "config": {"languageCode": "en-US"},
    "audio": {"uri": "gs://bucket/audio.wav"}
  }'
```

### Test Python Integration
```bash
# Install Google Cloud libraries
pip install google-cloud-vision google-cloud-speech google-cloud-language google-cloud-documentai

# Test with Python
python -c "
from google.cloud import vision
import os
print('✅ Google Cloud Vision available' if os.getenv('GOOGLE_APPLICATION_CREDENTIALS') else '❌ Credentials not set')
"
```

## 📊 Free Tier Limits

| Service | Free Tier | Enhanced System Usage |
|---------|-----------|---------------------|
| Vision API | 1,000 units/month | ✅ Sufficient for demo |
| Speech-to-Text | 60 minutes/month | ✅ Sufficient for demo |
| Natural Language | 5,000 units/month | ✅ Sufficient for demo |
| Document AI | 100 pages/month | ✅ Sufficient for demo |
| Cloud Storage | 5 GB | ✅ Sufficient for demo |

## 🎯 Enhanced Features Enabled

### ✅ Medical OCR with Vision API
- Extract text from medical documents
- Detect medical forms and prescriptions
- Identify medical equipment in images

### ✅ Speech-to-Text with Medical Vocabulary
- Transcribe call center recordings
- Medical terminology recognition
- Multi-speaker diarization

### ✅ Natural Language Processing
- Medical entity extraction
- Sentiment analysis of calls
- Document classification

### ✅ Document AI for Forms
- Parse medical claim forms
- Extract structured data
- Automate data entry

## 🔄 Fallback Mode

If Google Cloud APIs are not available, the system automatically falls back to:
- ✅ Basic text processing with sentence-transformers
- ✅ Local image processing with PIL
- ✅ Audio processing with basic features
- ✅ All core functionality remains available

## 🚨 Troubleshooting

### Common Issues:

#### 1. "Credentials not found"
```bash
# Check if credentials file exists
ls -la google-credentials.json

# Check environment variable
echo $GOOGLE_APPLICATION_CREDENTIALS
```

#### 2. "API not enabled"
- Go to Google Cloud Console
- Navigate to APIs & Services → Library
- Search and enable the required API

#### 3. "Quota exceeded"
- Check your usage in Google Cloud Console
- Upgrade to paid tier if needed
- System will continue working with basic features

#### 4. "Permission denied"
- Ensure service account has correct roles
- Regenerate service account key
- Check IAM permissions

### Debug Mode:
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test Google Cloud availability
from enhanced_embeddings import EnhancedMultimodalEmbedder
embedder = EnhancedMultimodalEmbedder()
info = embedder.get_embedding_info()
print(json.dumps(info, indent=2))
```

## 🎉 Success Indicators

When properly configured, you'll see:
```
[OK] Google Vision API client initialized
[OK] Google Speech-to-Text API client initialized  
[OK] Google Natural Language API client initialized
[OK] Google Document AI client initialized
[INFO] Enhanced systems initialized successfully
```

## 📞 Support

- **Google Cloud Documentation**: https://cloud.google.com/docs
- **API Reference**: https://cloud.google.com/docs/reference
- **Billing & Quotas**: https://console.cloud.google.com/billing

---

## 🏁 Ready for Demo!

Once configured, your system will have:
- 🧠 **Medical entity extraction** from claims
- 🏥 **ICD-10/CPT medical coding**
- 👥 **Provider network verification**
- 🔍 **Enhanced fraud detection**
- 📊 **Advanced settlement estimation**
- 🖼️ **Medical document OCR**
- 🎵 **Call transcription & analysis**
- 📹 **Multimodal processing**

All with **zero cost** using free tiers! 🚀
