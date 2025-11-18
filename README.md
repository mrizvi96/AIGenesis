# AI-Powered Insurance Claims Processing Assistant

## Disk-Efficient Setup

This project is designed to minimize disk usage while maintaining functionality:

### Approach
- **In-memory Qdrant**: No Docker required, uses local memory
- **Lightweight models**: Smaller pre-trained models
- **On-demand loading**: Models loaded only when needed
- **CPU inference**: No GPU requirements

### Quick Start
```bash
pip install -r requirements.txt
python backend/main.py
```

### Space Usage
- **Dependencies**: ~500MB (vs 5GB+ for full ML stack)
- **Models**: ~200MB downloaded on-demand
- **Data**: Minimal sample data included

## Features
- Text-based claim processing (primary focus)
- Basic image analysis
- In-memory vector search
- AI recommendations
- Web interface

## Alternative: Cloud Qdrant
For better performance, you can use Qdrant Cloud (free tier available):
```bash
export QDRANT_URL="your-cloud-url"
export QDRANT_API_KEY="your-api-key"
```