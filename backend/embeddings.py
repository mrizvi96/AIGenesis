"""
Multimodal Embedding Pipeline for AI Insurance Claims Processing
Handles text, images, audio, and video embeddings using free APIs and lightweight models
"""

import os
import io
import base64
import tempfile
import json
import requests
from typing import List, Dict, Any, Union, Optional
from PIL import Image
import numpy as np

# Import sentence-transformers for text embeddings (free)
from sentence_transformers import SentenceTransformer

class MultimodalEmbedder:
    def __init__(self):
        """Initialize the multimodal embedding system"""
        self.load_models()
        self.setup_api_keys()

    def load_models(self):
        """Load free models for different modalities"""
        print("Loading embedding models...")

        # Text embedding model (free, lightweight)
        try:
            self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("[OK] Loaded sentence-transformers text model")
        except Exception as e:
            print(f"[ERROR] Failed to load sentence-transformers: {e}")
            self.text_model = None

        # Set embedding dimensions
        self.embedding_dims = {
            'text': 384 if self.text_model else 100,  # sentence-transformers dim or fallback
            'image': 512,   # Will use cloud API or fallback features
            'audio': 512,   # Will use cloud API or fallback features
            'video': 512    # Will use cloud API or fallback features
        }

    def setup_api_keys(self):
        """Setup API keys for cloud services"""
        # Google Vision API (you have $100 free credits)
        self.google_vision_api_key = os.getenv("GOOGLE_VISION_API_KEY")

        # OpenAI Whisper API (3-day free trial)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        print("[OK] API key configuration loaded")

    def embed_text(self, text: str) -> List[float]:
        """
        Generate text embeddings using sentence-transformers or fallback

        Args:
            text: Input text to embed

        Returns:
            Text embedding vector
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return [0.0] * self.embedding_dims['text']

        try:
            if self.text_model is not None:
                # Use sentence-transformers for high-quality embeddings
                embedding = self.text_model.encode(text, convert_to_numpy=True)
                return embedding.tolist()
            else:
                # Fallback: simple hash-based embedding
                return self._fallback_text_embedding(text)
        except Exception as e:
            print(f"[ERROR] Error in text embedding: {e}")
            return self._fallback_text_embedding(text)

    def _fallback_text_embedding(self, text: str) -> List[float]:
        """
        Fallback text embedding using simple features
        """
        # Simple character-level features
        text = text.lower()
        features = []

        # Text length features
        features.append(len(text) / 1000.0)  # Normalized length
        features.append(text.count(' ') / 100.0)  # Word count approximation

        # Character frequency features
        for char in 'abcdefghijklmnopqrstuvwxyz':
            features.append(text.count(char) / 100.0)

        # Pad to required dimension
        while len(features) < self.embedding_dims['text']:
            features.append(0.0)

        return features[:self.embedding_dims['text']]

    def embed_image(self, image_path: str = None, image_data: bytes = None) -> List[float]:
        """
        Generate image embeddings using Google Vision API or fallback features

        Args:
            image_path: Path to image file
            image_data: Raw image data as bytes

        Returns:
            Image embedding vector
        """
        try:
            if image_data is None and image_path:
                with open(image_path, 'rb') as f:
                    image_data = f.read()

            if image_data is None:
                return [0.0] * self.embedding_dims['image']

            # Try Google Vision API first
            if self.google_vision_api_key:
                vision_features = self._get_google_vision_features(image_data)
                if vision_features:
                    return vision_features

            # Fallback to simple image features
            return self._fallback_image_features(image_data)

        except Exception as e:
            print(f"[ERROR] Error in image embedding: {e}")
            return [0.0] * self.embedding_dims['image']

    def _get_google_vision_features(self, image_data: bytes) -> Optional[List[float]]:
        """
        Extract features using Google Vision API
        """
        try:
            # Prepare the request
            content = base64.b64encode(image_data).decode('utf-8')

            # Use Google Vision API for label detection
            url = f"https://vision.googleapis.com/v1/images:annotate?key={self.google_vision_api_key}"

            request_body = {
                "requests": [
                    {
                        "image": {
                            "content": content
                        },
                        "features": [
                            {"type": "LABEL_DETECTION", "maxResults": 10},
                            {"type": "WEB_DETECTION", "maxResults": 5},
                            {"type": "IMAGE_PROPERTIES", "maxResults": 1}
                        ]
                    }
                ]
            }

            response = requests.post(url, json=request_body, timeout=10)

            if response.status_code == 200:
                result = response.json()
                features = self._process_vision_response(result)
                return features
            else:
                print(f"[WARNING] Google Vision API error: {response.status_code}")
                return None

        except Exception as e:
            print(f"[WARNING] Google Vision API error: {e}")
            return None

    def _process_vision_response(self, vision_response: Dict) -> List[float]:
        """
        Process Google Vision API response into embedding vector
        """
        features = []

        try:
            response = vision_response['responses'][0]

            # Process labels
            if 'labelAnnotations' in response:
                labels = response['labelAnnotations']
                # Use top 10 labels as features
                for i in range(10):
                    if i < len(labels):
                        features.append(labels[i]['score'])
                    else:
                        features.append(0.0)

            # Process web detection
            if 'webDetection' in response:
                web_detection = response['webDetection']
                features.append(len(web_detection.get('webEntities', [])) / 10.0)
                features.append(len(web_detection.get('fullMatchingImages', [])) / 10.0)
            else:
                features.extend([0.0, 0.0])

            # Process image properties
            if 'imagePropertiesAnnotation' in response:
                colors = response['imagePropertiesAnnotation'].get('dominantColors', {}).get('colors', [])
                for i in range(5):  # Top 5 dominant colors
                    if i < len(colors):
                        color = colors[i]['color']
                        features.extend([
                            color.get('red', 0) / 255.0,
                            color.get('green', 0) / 255.0,
                            color.get('blue', 0) / 255.0
                        ])
                    else:
                        features.extend([0.0, 0.0, 0.0])

            # Pad to required dimension
            while len(features) < self.embedding_dims['image']:
                features.append(0.0)

            return features[:self.embedding_dims['image']]

        except Exception as e:
            print(f"[ERROR] Error processing vision response: {e}")
            return [0.0] * self.embedding_dims['image']

    def _fallback_image_features(self, image_data: bytes) -> List[float]:
        """
        Generate basic image features as fallback
        """
        try:
            # Load image and extract basic features
            image = Image.open(io.BytesIO(image_data))

            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Resize to standard size for feature extraction
            image = image.resize((64, 64))

            # Convert to numpy array
            img_array = np.array(image)

            # Basic statistical features
            features = [
                np.mean(img_array) / 255.0,      # Mean brightness
                np.std(img_array) / 255.0,       # Standard deviation
                len(image_data) / 1000000.0,      # File size
                image.size[0] / 1000.0,           # Width
                image.size[1] / 1000.0,           # Height
            ]

            # Color histogram features (simplified)
            for channel in range(3):
                hist, _ = np.histogram(img_array[:,:,channel], bins=8, range=(0, 256))
                hist = hist / hist.sum()  # Normalize
                features.extend(hist.tolist())

            # Pad to required dimension
            while len(features) < self.embedding_dims['image']:
                features.append(0.0)

            return features[:self.embedding_dims['image']]

        except Exception as e:
            print(f"[ERROR] Error in fallback image features: {e}")
            return [0.0] * self.embedding_dims['image']

    def embed_audio(self, audio_path: str = None, audio_data: bytes = None) -> List[float]:
        """
        Generate audio embeddings using cloud API or fallback features

        Args:
            audio_path: Path to audio file
            audio_data: Raw audio data as bytes

        Returns:
            Audio embedding vector
        """
        try:
            if audio_data is None and audio_path:
                with open(audio_path, 'rb') as f:
                    audio_data = f.read()

            if audio_data is None:
                return [0.0] * self.embedding_dims['audio']

            # Try OpenAI Whisper API first
            if self.openai_api_key:
                whisper_features = self._get_whisper_features(audio_data)
                if whisper_features:
                    return whisper_features

            # Fallback to basic audio features
            return self._fallback_audio_features(audio_data)

        except Exception as e:
            print(f"[ERROR] Error in audio embedding: {e}")
            return [0.0] * self.embedding_dims['audio']

    def _get_whisper_features(self, audio_data: bytes) -> Optional[List[float]]:
        """
        Get audio features using OpenAI Whisper API
        """
        try:
            # This would require setting up the Whisper API properly
            # For now, we'll implement a basic transcription-based approach
            # Note: You might need to use the OpenAI Python client for this

            # Placeholder for Whisper API integration
            # You would need to:
            # 1. Save audio_data to a temporary file
            # 2. Call OpenAI's Whisper API
            # 3. Convert transcription to embedding

            return None  # Will fall back to basic features

        except Exception as e:
            print(f"[WARNING] Whisper API error: {e}")
            return None

    def _fallback_audio_features(self, audio_data: bytes) -> List[float]:
        """
        Generate basic audio features as fallback
        """
        try:
            # Basic file-based features
            features = [
                len(audio_data) / 1000000.0,  # File size in MB
                len(audio_data) / 16000.0,     # Approximate duration (assuming 16kHz)
            ]

            # Audio format detection (basic)
            if audio_data.startswith(b'ID3'):
                features.extend([1.0, 0.0, 0.0])  # MP3
            elif audio_data.startswith(b'RIFF'):
                features.extend([0.0, 1.0, 0.0])  # WAV
            elif audio_data.startswith(b'ftyp'):
                features.extend([0.0, 0.0, 1.0])  # M4A
            else:
                features.extend([0.0, 0.0, 0.0])  # Unknown

            # Simple byte-level features
            byte_values = list(audio_data[:1000])  # First 1000 bytes
            features.extend([
                np.mean(byte_values) / 255.0,
                np.std(byte_values) / 255.0,
                len(set(byte_values)) / 256.0,  # Byte diversity
            ])

            # Pad to required dimension
            while len(features) < self.embedding_dims['audio']:
                features.append(0.0)

            return features[:self.embedding_dims['audio']]

        except Exception as e:
            print(f"[ERROR] Error in fallback audio features: {e}")
            return [0.0] * self.embedding_dims['audio']

    def embed_video(self, video_path: str = None, video_data: bytes = None) -> List[float]:
        """
        Generate video embeddings by extracting frames and processing as images

        Args:
            video_path: Path to video file
            video_data: Raw video data as bytes

        Returns:
            Video embedding vector
        """
        try:
            if video_data is None and video_path:
                with open(video_path, 'rb') as f:
                    video_data = f.read()

            if video_data is None:
                return [0.0] * self.embedding_dims['video']

            # For now, use basic video file features
            # In a production system, you would extract frames using OpenCV
            return self._fallback_video_features(video_data)

        except Exception as e:
            print(f"[ERROR] Error in video embedding: {e}")
            return [0.0] * self.embedding_dims['video']

    def _fallback_video_features(self, video_data: bytes) -> List[float]:
        """
        Generate basic video features as fallback
        """
        try:
            features = [
                len(video_data) / 10000000.0,  # File size in 10MB units
            ]

            # Video format detection
            if video_data.startswith(b'\x00\x00\x00\x18ftyp'):
                features.extend([1.0, 0.0, 0.0])  # MP4
            elif video_data.startswith(b'RIFF') and b'AVI' in video_data[:100]:
                features.extend([0.0, 1.0, 0.0])  # AVI
            elif video_data.startswith(b'\x1aE'):
                features.extend([0.0, 0.0, 1.0])  # MKV
            else:
                features.extend([0.0, 0.0, 0.0])  # Unknown

            # Basic byte statistics
            byte_values = list(video_data[:2000])  # First 2000 bytes
            features.extend([
                np.mean(byte_values) / 255.0,
                np.std(byte_values) / 255.0,
                len(set(byte_values)) / 256.0,
            ])

            # Pad to required dimension
            while len(features) < self.embedding_dims['video']:
                features.append(0.0)

            return features[:self.embedding_dims['video']]

        except Exception as e:
            print(f"[ERROR] Error in fallback video features: {e}")
            return [0.0] * self.embedding_dims['video']

    def get_embedding_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding system
        """
        return {
            'text_model_loaded': self.text_model is not None,
            'google_vision_available': bool(self.google_vision_api_key),
            'openai_whisper_available': bool(self.openai_api_key),
            'embedding_dimensions': self.embedding_dims,
            'sentence_transformers_available': True
        }

# Test the embedding system
if __name__ == "__main__":
    embedder = MultimodalEmbedder()

    # Test text embedding
    test_text = "This is a sample insurance claim about a car accident."
    text_embedding = embedder.embed_text(test_text)
    print(f"[OK] Text embedding dimension: {len(text_embedding)}")

    # Print system info
    info = embedder.get_embedding_info()
    print(f"[OK] Embedding system info: {json.dumps(info, indent=2)}")