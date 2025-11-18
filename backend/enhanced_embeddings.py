"""
Enhanced Multimodal Embedding Pipeline for AI Insurance Claims Processing
Integrates Google Cloud APIs for advanced health insurance processing
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

# Google Cloud imports
try:
    from google.cloud import vision
    from google.cloud import speech
    from google.cloud import language_v1
    from google.cloud import documentai
    from google.oauth2 import service_account
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False
    print("[WARNING] Google Cloud libraries not installed. Run: pip install google-cloud-vision google-cloud-speech google-cloud-language google-cloud-documentai")

class EnhancedMultimodalEmbedder:
    def __init__(self):
        """Initialize enhanced multimodal embedding system"""
        self.load_models()
        self.setup_apis()

    def load_models(self):
        """Load enhanced models for different modalities"""
        print("Loading enhanced embedding models...")

        # Text embedding model (384-dim to match Qdrant)
        try:
            self.text_model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim, matches Qdrant
            print("[OK] Loaded sentence-transformers text model (384-dim)")
        except Exception as e:
            print(f"[WARNING] Failed to load model: {e}")
            try:
                self.text_model = SentenceTransformer('average_word_embeddings_glove.6B.100d')
                print("[OK] Loaded fallback embedding model")
            except Exception as e2:
                print(f"[ERROR] Failed to load any sentence-transformers model: {e2}")
                self.text_model = None

        # Set embedding dimensions to match Qdrant collections
        self.embedding_dims = {
            'text': 384,    # Match Qdrant text_claims collection
            'image': 512,   # Match Qdrant image_claims collection
            'audio': 512,   # Match Qdrant audio_claims collection
            'video': 512    # Match Qdrant video_claims collection
        }

    def setup_apis(self):
        """Setup Google Cloud API clients"""
        print("Setting up Google Cloud APIs...")

        # Check for credentials
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not creds_path or not os.path.exists(creds_path):
            print("[WARNING] Google Cloud credentials not found. Some features will be limited.")

        # Initialize Google Cloud clients
        self.google_clients = {}
        
        try:
            if GOOGLE_CLOUD_AVAILABLE and creds_path and os.path.exists(creds_path):
                # Vision API
                self.google_clients['vision'] = vision.ImageAnnotatorClient()
                print("[OK] Google Vision API client initialized")

                # Speech-to-Text API
                self.google_clients['speech'] = speech.SpeechClient()
                print("[OK] Google Speech-to-Text API client initialized")

                # Natural Language API
                self.google_clients['language'] = language_v1.LanguageServiceClient()
                print("[OK] Google Natural Language API client initialized")

                # Document AI
                self.google_clients['documentai'] = documentai.DocumentProcessorServiceClient()
                print("[OK] Google Document AI client initialized")

            else:
                print("[INFO] Google Cloud clients not initialized - will use fallback methods")

        except Exception as e:
            print(f"[ERROR] Failed to initialize Google Cloud clients: {e}")

        # API keys for direct API calls (backup)
        self.google_vision_api_key = os.getenv("GOOGLE_VISION_API_KEY")
        self.google_cloud_project = os.getenv("GOOGLE_CLOUD_PROJECT")

    def embed_text(self, text: str, extract_medical_entities: bool = True) -> List[float]:
        """
        Generate enhanced text embeddings with medical entity extraction

        Args:
            text: Input text to embed
            extract_medical_entities: Whether to extract medical entities

        Returns:
            Enhanced text embedding vector
        """
        if not text or not text.strip():
            return [0.0] * self.embedding_dims['text']

        try:
            # Extract medical entities if requested and available
            medical_features = []
            if extract_medical_entities and 'language' in self.google_clients:
                medical_features = self._extract_medical_entities(text)

            # Generate base embedding
            if self.text_model is not None:
                base_embedding = self.text_model.encode(text, convert_to_numpy=True)
                base_embedding = base_embedding.tolist()
            else:
                base_embedding = self._fallback_text_embedding(text)

            # Combine with medical features
            if medical_features:
                # Pad or truncate medical features to match base embedding
                medical_features = medical_features[:len(base_embedding)]
                if len(medical_features) < len(base_embedding):
                    medical_features.extend([0.0] * (len(base_embedding) - len(medical_features)))
                
                # Weighted combination (70% text, 30% medical)
                enhanced_embedding = [
                    0.7 * base + 0.3 * med 
                    for base, med in zip(base_embedding, medical_features)
                ]
            else:
                enhanced_embedding = base_embedding

            return enhanced_embedding[:self.embedding_dims['text']]

        except Exception as e:
            print(f"[ERROR] Error in enhanced text embedding: {e}")
            return self._fallback_text_embedding(text)

    def _extract_medical_entities(self, text: str) -> List[float]:
        """
        Extract medical entities using Google Natural Language API
        """
        try:
            client = self.google_clients['language']
            
            # Analyze entities
            document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
            response = client.analyze_entities(request={'document': document})

            # Extract medical-related entities
            medical_features = []
            medical_keywords = ['diagnosis', 'treatment', 'medication', 'symptom', 'procedure', 'condition', 'therapy', 'surgery']
            
            entity_count = 0
            medical_entity_count = 0
            sentiment_scores = []

            for entity in response.entities:
                entity_count += 1
                entity_type = language_v1.Entity.Type(entity.type_).name
                
                # Check if it's medical-related
                if any(keyword in entity.name.lower() for keyword in medical_keywords):
                    medical_entity_count += 1
                
                # Get sentiment if available
                if entity.sentiment:
                    sentiment_scores.append(entity.sentiment.score)

            # Create feature vector
            medical_features = [
                entity_count / 100.0,  # Normalize
                medical_entity_count / 100.0,
                np.mean(sentiment_scores) if sentiment_scores else 0.0,
                len(text.split()) / 1000.0,  # Text complexity
                text.count(' ') / 100.0,  # Word count approximation
            ]

            # Pad with zeros if needed
            while len(medical_features) < 20:  # Target 20 medical features
                medical_features.append(0.0)

            return medical_features

        except Exception as e:
            print(f"[WARNING] Medical entity extraction failed: {e}")
            return []

    def embed_image(self, image_path: str = None, image_data: bytes = None, extract_text: bool = True) -> List[float]:
        """
        Generate enhanced image embeddings with OCR and medical text extraction

        Args:
            image_path: Path to image file
            image_data: Raw image data as bytes
            extract_text: Whether to extract text from image

        Returns:
            Enhanced image embedding vector
        """
        try:
            if image_data is None and image_path:
                with open(image_path, 'rb') as f:
                    image_data = f.read()

            if image_data is None:
                return [0.0] * self.embedding_dims['image']

            # Try enhanced Google Vision API first
            if 'vision' in self.google_clients:
                vision_features = self._get_enhanced_vision_features(image_data, extract_text)
                if vision_features:
                    return vision_features

            # Fallback to basic features
            return self._fallback_image_features(image_data)

        except Exception as e:
            print(f"[ERROR] Error in enhanced image embedding: {e}")
            return [0.0] * self.embedding_dims['image']

    def _get_enhanced_vision_features(self, image_data: bytes, extract_text: bool) -> Optional[List[float]]:
        """
        Extract enhanced features using Google Vision API
        """
        try:
            client = self.google_clients['vision']
            
            # Create image object
            image = vision.Image(content=image_data)

            # Enhanced feature detection
            features = [
                vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION, max_results=15),
                vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION, max_results=10),
                vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION, max_results=1),
                vision.Feature(type_=vision.Feature.Type.WEB_DETECTION, max_results=10),
                vision.Feature(type_=vision.Feature.Type.IMAGE_PROPERTIES, max_results=1),
                vision.Feature(type_=vision.Feature.Type.OBJECT_LOCALIZATION, max_results=20)
            ]

            # Perform annotation
            response = client.annotate_image({'image': image, 'features': features})

            return self._process_enhanced_vision_response(response, extract_text)

        except Exception as e:
            print(f"[WARNING] Enhanced Vision API error: {e}")
            return None

    def _process_enhanced_vision_response(self, response, extract_text: bool) -> List[float]:
        """
        Process enhanced Vision API response into embedding vector
        """
        features = []

        try:
            # Process labels with confidence scores
            if response.label_annotations:
                for i in range(15):  # Top 15 labels
                    if i < len(response.label_annotations):
                        label = response.label_annotations[i]
                        features.append(label.score)
                        features.append(hash(label.description) % 1000 / 1000.0)  # Description hash
                    else:
                        features.extend([0.0, 0.0])

            # Process text extraction
            text_features = []
            if extract_text:
                if response.text_annotations:
                    extracted_text = response.text_annotations[0].description
                    text_features = self._extract_text_features(extracted_text)
                elif response.full_text_annotation:
                    extracted_text = response.full_text_annotation.text
                    text_features = self._extract_text_features(extracted_text)

            # Pad text features
            while len(text_features) < 50:  # Target 50 text features
                text_features.append(0.0)
            features.extend(text_features[:50])

            # Process web detection
            if response.web_detection:
                web_detection = response.web_detection
                features.append(len(web_detection.web_entities) / 20.0)
                features.append(len(web_detection.full_matching_images) / 10.0)
                features.append(len(web_detection.visually_similar_images) / 10.0)
            else:
                features.extend([0.0, 0.0, 0.0])

            # Process image properties (dominant colors)
            if response.image_properties_annotation:
                colors = response.image_properties_annotation.dominant_colors.colors
                for i in range(10):  # Top 10 dominant colors
                    if i < len(colors):
                        color = colors[i].color
                        features.extend([
                            color.red / 255.0,
                            color.green / 255.0,
                            color.blue / 255.0,
                            colors[i].score
                        ])
                    else:
                        features.extend([0.0, 0.0, 0.0, 0.0])
            else:
                features.extend([0.0] * 40)  # 10 colors * 4 features each

            # Process object localization
            if response.localized_object_annotations:
                for i in range(20):  # Top 20 objects
                    if i < len(response.localized_object_annotations):
                        obj = response.localized_object_annotations[i]
                        features.append(obj.score)
                        # Add bounding box features
                        if obj.bounding_poly.normalized_vertices:
                            vertices = obj.bounding_poly.normalized_vertices
                            features.extend([
                                vertices[0].x if vertices else 0.0,
                                vertices[0].y if vertices else 0.0,
                                vertices[2].x if len(vertices) > 2 else 0.0,
                                vertices[2].y if len(vertices) > 2 else 0.0
                            ])
                        else:
                            features.extend([0.0] * 4)
                    else:
                        features.extend([0.0] * 5)  # score + 4 bbox coordinates
            else:
                features.extend([0.0] * 100)  # 20 objects * 5 features each

            # Pad to required dimension
            while len(features) < self.embedding_dims['image']:
                features.append(0.0)

            return features[:self.embedding_dims['image']]

        except Exception as e:
            print(f"[ERROR] Error processing enhanced vision response: {e}")
            return [0.0] * self.embedding_dims['image']

    def _extract_text_features(self, text: str) -> List[float]:
        """
        Extract features from extracted text
        """
        features = []
        
        # Basic text features
        features.append(len(text) / 1000.0)
        features.append(len(text.split()) / 100.0)
        features.append(text.count('\n') / 50.0)  # Line count
        
        # Medical keyword detection
        medical_keywords = ['patient', 'doctor', 'diagnosis', 'treatment', 'prescription', 'hospital', 'clinic', 'medical']
        keyword_count = sum(1 for keyword in medical_keywords if keyword in text.lower())
        features.append(keyword_count / len(medical_keywords))
        
        # Number detection (for dates, amounts, etc.)
        import re
        numbers = re.findall(r'\d+\.?\d*', text)
        features.append(len(numbers) / 20.0)
        
        # Date pattern detection
        date_patterns = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)
        features.append(len(date_patterns) / 10.0)
        
        # Amount pattern detection
        amount_patterns = re.findall(r'\$?\d+\.?\d*', text)
        features.append(len(amount_patterns) / 10.0)
        
        # Character diversity
        features.append(len(set(text)) / 100.0)
        
        # Uppercase ratio
        uppercase = sum(1 for c in text if c.isupper())
        features.append(uppercase / len(text) if text else 0.0)
        
        return features

    def embed_audio(self, audio_path: str = None, audio_data: bytes = None, transcribe: bool = True) -> List[float]:
        """
        Generate enhanced audio embeddings with transcription

        Args:
            audio_path: Path to audio file
            audio_data: Raw audio data as bytes
            transcribe: Whether to transcribe audio to text

        Returns:
            Enhanced audio embedding vector
        """
        try:
            if audio_data is None and audio_path:
                with open(audio_path, 'rb') as f:
                    audio_data = f.read()

            if audio_data is None:
                return [0.0] * self.embedding_dims['audio']

            # Try Google Speech-to-Text API first
            if transcribe and 'speech' in self.google_clients:
                speech_features = self._get_speech_features(audio_data)
                if speech_features:
                    return speech_features

            # Fallback to basic audio features
            return self._fallback_audio_features(audio_data)

        except Exception as e:
            print(f"[ERROR] Error in enhanced audio embedding: {e}")
            return [0.0] * self.embedding_dims['audio']

    def _get_speech_features(self, audio_data: bytes) -> Optional[List[float]]:
        """
        Get audio features using Google Speech-to-Text API
        """
        try:
            client = self.google_clients['speech']

            # Configure speech recognition
            audio = speech.RecognitionAudio(content=audio_data)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED,
                sample_rate_hertz=16000,
                language_code="en-US",
                enable_automatic_punctuation=True,
                enable_word_confidence=True,
                enable_word_time_offsets=True,
                model="medical_dictation"  # Medical transcription model
            )

            # Perform transcription
            response = client.recognize(config=config, audio=audio)

            return self._process_speech_response(response)

        except Exception as e:
            print(f"[WARNING] Speech-to-Text API error: {e}")
            return None

    def _process_speech_response(self, response) -> List[float]:
        """
        Process speech recognition response into features
        """
        features = []

        try:
            # Extract transcription
            transcription = ""
            word_confidences = []
            word_durations = []
            speaker_changes = 0

            for result in response.results:
                if result.alternatives:
                    transcription += result.alternatives[0].transcript + " "
                    
                    # Word-level features
                    for word_info in result.alternatives[0].words:
                        word_confidences.append(word_info.confidence)
                        if hasattr(word_info, 'start_time') and hasattr(word_info, 'end_time'):
                            duration = (word_info.end_time.seconds + word_info.end_time.nanos/1e9) - \
                                      (word_info.start_time.seconds + word_info.start_time.nanos/1e9)
                            word_durations.append(duration)

            # Basic transcription features
            features.append(len(transcription) / 1000.0)  # Length
            features.append(len(transcription.split()) / 100.0)  # Word count
            features.append(transcription.count('?') / 10.0)  # Questions
            features.append(transcription.count('!') / 10.0)  # Exclamations

            # Confidence features
            if word_confidences:
                features.append(np.mean(word_confidences))
                features.append(np.std(word_confidences))
                features.append(min(word_confidences))
                features.append(max(word_confidences))
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])

            # Duration features
            if word_durations:
                features.append(np.mean(word_durations))
                features.append(np.std(word_durations))
            else:
                features.extend([0.0, 0.0])

            # Medical keyword features
            medical_keywords = ['pain', 'symptom', 'doctor', 'medicine', 'treatment', 'diagnosis', 'prescription']
            medical_count = sum(1 for keyword in medical_keywords if keyword in transcription.lower())
            features.append(medical_count / len(medical_keywords))

            # Get text embedding of transcription
            if transcription.strip():
                text_embedding = self.embed_text(transcription, extract_medical_entities=True)
                # Add first 50 features from text embedding
                features.extend(text_embedding[:50])
            else:
                features.extend([0.0] * 50)

            # Pad to required dimension
            while len(features) < self.embedding_dims['audio']:
                features.append(0.0)

            return features[:self.embedding_dims['audio']]

        except Exception as e:
            print(f"[ERROR] Error processing speech response: {e}")
            return [0.0] * self.embedding_dims['audio']

    def embed_video(self, video_path: str = None, video_data: bytes = None) -> List[float]:
        """
        Generate enhanced video embeddings with frame analysis

        Args:
            video_path: Path to video file
            video_data: Raw video data as bytes

        Returns:
            Enhanced video embedding vector
        """
        try:
            if video_data is None and video_path:
                with open(video_path, 'rb') as f:
                    video_data = f.read()

            if video_data is None:
                return [0.0] * self.embedding_dims['video']

            # For now, enhance basic video features
            return self._enhanced_video_features(video_data)

        except Exception as e:
            print(f"[ERROR] Error in enhanced video embedding: {e}")
            return [0.0] * self.embedding_dims['video']

    def _enhanced_video_features(self, video_data: bytes) -> List[float]:
        """
        Generate enhanced video features
        """
        try:
            features = []

            # Basic file features
            features.append(len(video_data) / 10000000.0)  # File size in 10MB units

            # Video format detection with more formats
            format_signatures = {
                b'\x00\x00\x00\x18ftyp': ('mp4', [1.0, 0.0, 0.0, 0.0]),
                b'RIFF': ('avi', [0.0, 1.0, 0.0, 0.0]),
                b'\x1aE': ('mkv', [0.0, 0.0, 1.0, 0.0]),
                b'FLV': ('flv', [0.0, 0.0, 0.0, 1.0]),
                b'webm': ('webm', [0.25, 0.25, 0.25, 0.25])
            }

            format_features = [0.0, 0.0, 0.0, 0.0]  # mp4, avi, mkv, flv/webm
            for signature, (format_name, fmt_features) in format_signatures.items():
                if video_data.startswith(signature):
                    format_features = fmt_features
                    break
            features.extend(format_features)

            # Enhanced byte statistics
            byte_values = list(video_data[:5000])  # First 5000 bytes
            features.extend([
                np.mean(byte_values) / 255.0,
                np.std(byte_values) / 255.0,
                len(set(byte_values)) / 256.0,
                np.median(byte_values) / 255.0,
                np.percentile(byte_values, 90) / 255.0
            ])

            # Pattern detection
            patterns = {
                b'0x000001B3': 'mpeg_header',
                b'H264': 'h264_codec',
                b'AVC1': 'avc1_codec',
                b'vp9': 'vp9_codec'
            }

            pattern_features = []
            for pattern, codec in patterns.items():
                if pattern in video_data[:1000]:  # Check first 1KB
                    pattern_features.append(1.0)
                else:
                    pattern_features.append(0.0)
            features.extend(pattern_features)

            # Audio stream detection (simplified)
            audio_indicators = [b'WAVE', b'audio', b'mp3', b'aac']
            audio_score = 0.0
            for indicator in audio_indicators:
                if indicator in video_data:
                    audio_score += 0.25
            features.append(min(audio_score, 1.0))

            # Frame estimation (very rough approximation)
            estimated_frames = len(video_data) / 10000  # Rough estimate
            features.append(min(estimated_frames / 1000.0, 1.0))  # Normalize to thousands

            # Duration estimation (very rough)
            estimated_duration = len(video_data) / 1000000  # Rough MB/second estimate
            features.append(min(estimated_duration / 60.0, 1.0))  # Normalize to minutes

            # Quality indicators (based on file size vs format)
            quality_score = min(len(video_data) / 50000000.0, 1.0)  # 50MB as reference for high quality
            features.append(quality_score)

            # Pad to required dimension
            while len(features) < self.embedding_dims['video']:
                features.append(0.0)

            return features[:self.embedding_dims['video']]

        except Exception as e:
            print(f"[ERROR] Error in enhanced video features: {e}")
            return [0.0] * self.embedding_dims['video']

    def _fallback_text_embedding(self, text: str) -> List[float]:
        """
        Fallback text embedding using simple features
        """
        text = text.lower()
        features = []

        # Enhanced character-level features
        features.append(len(text) / 1000.0)
        features.append(text.count(' ') / 100.0)
        features.append(text.count('\n') / 50.0)

        # Character frequency with medical focus
        for char in 'abcdefghijklmnopqrstuvwxyz':
            features.append(text.count(char) / 100.0)

        # Medical keyword features
        medical_keywords = ['pain', 'symptom', 'doctor', 'medicine', 'treatment', 'diagnosis', 'prescription', 'hospital']
        for keyword in medical_keywords:
            features.append(1.0 if keyword in text else 0.0)

        # Number and date detection
        import re
        features.append(len(re.findall(r'\d+', text)) / 20.0)
        features.append(len(re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)) / 10.0)
        features.append(len(re.findall(r'\$?\d+\.?\d*', text)) / 10.0)

        # Pad to required dimension
        while len(features) < self.embedding_dims['text']:
            features.append(0.0)

        return features[:self.embedding_dims['text']]

    def _fallback_image_features(self, image_data: bytes) -> List[float]:
        """
        Generate basic image features as fallback
        """
        try:
            image = Image.open(io.BytesIO(image_data))

            if image.mode != 'RGB':
                image = image.convert('RGB')

            image = image.resize((128, 128))  # Larger size for better features
            img_array = np.array(image)

            # Enhanced statistical features
            features = [
                np.mean(img_array) / 255.0,
                np.std(img_array) / 255.0,
                len(image_data) / 1000000.0,
                image.size[0] / 1000.0,
                image.size[1] / 1000.0,
                np.median(img_array) / 255.0,
                np.percentile(img_array, 25) / 255.0,
                np.percentile(img_array, 75) / 255.0
            ]

            # Enhanced color histogram (16 bins per channel)
            for channel in range(3):
                hist, _ = np.histogram(img_array[:,:,channel], bins=16, range=(0, 256))
                hist = hist / hist.sum() if hist.sum() > 0 else hist
                features.extend(hist.tolist())

            # Edge detection approximation
            gray = np.mean(img_array, axis=2)
            edges = np.abs(gray[1:, :] - gray[:-1, :]) + np.abs(gray[:, 1:] - gray[:, :-1])
            features.append(np.mean(edges) / 255.0)
            features.append(np.std(edges) / 255.0)

            # Texture features (simplified)
            try:
                from scipy import ndimage
                # Local binary pattern approximation
                texture = ndimage.generic_filter(gray, lambda x: len(set(x)) - 1, size=3)
                features.append(np.mean(texture) / 8.0)
                features.append(np.std(texture) / 8.0)
            except:
                features.extend([0.0, 0.0])

            # Pad to required dimension
            while len(features) < self.embedding_dims['image']:
                features.append(0.0)

            return features[:self.embedding_dims['image']]

        except Exception as e:
            print(f"[ERROR] Error in fallback image features: {e}")
            return [0.0] * self.embedding_dims['image']

    def _fallback_audio_features(self, audio_data: bytes) -> List[float]:
        """
        Generate enhanced basic audio features as fallback
        """
        try:
            features = [
                len(audio_data) / 1000000.0,  # File size in MB
                len(audio_data) / 16000.0,     # Approximate duration
            ]

            # Enhanced format detection
            format_signatures = {
                b'ID3': [1.0, 0.0, 0.0],  # MP3
                b'RIFF': [0.0, 1.0, 0.0],  # WAV
                b'ftyp': [0.0, 0.0, 1.0],  # M4A
                b'OggS': [0.25, 0.25, 0.5]  # OGG
            }

            format_features = [0.0, 0.0, 0.0]  # mp3, wav, m4a
            for signature, fmt_features in format_signatures.items():
                if audio_data.startswith(signature):
                    format_features = fmt_features
                    break
            features.extend(format_features)

            # Enhanced byte-level features
            byte_values = list(audio_data[:2000])
            features.extend([
                np.mean(byte_values) / 255.0,
                np.std(byte_values) / 255.0,
                len(set(byte_values)) / 256.0,
                np.median(byte_values) / 255.0,
                np.percentile(byte_values, 90) / 255.0
            ])

            # Audio quality indicators
            sample_rate_indicators = [b'44100', b'48000', b'22050', b'16000']
            sample_rates = [0.0, 0.0, 0.0, 0.0]
            for i, indicator in enumerate(sample_rate_indicators):
                if indicator in audio_data[:1000]:
                    sample_rates[i] = 1.0
            features.extend(sample_rates)

            # Stereo detection (simplified)
            channel_indicator = audio_data.count(b'stereo') / 10.0
            features.append(min(channel_indicator, 1.0))

            # Pad to required dimension
            while len(features) < self.embedding_dims['audio']:
                features.append(0.0)

            return features[:self.embedding_dims['audio']]

        except Exception as e:
            print(f"[ERROR] Error in fallback audio features: {e}")
            return [0.0] * self.embedding_dims['audio']

    def get_embedding_info(self) -> Dict[str, Any]:
        """
        Get information about enhanced embedding system
        """
        return {
            'text_model_loaded': self.text_model is not None,
            'google_cloud_available': GOOGLE_CLOUD_AVAILABLE,
            'google_vision_available': 'vision' in self.google_clients,
            'google_speech_available': 'speech' in self.google_clients,
            'google_language_available': 'language' in self.google_clients,
            'google_documentai_available': 'documentai' in self.google_clients,
            'embedding_dimensions': self.embedding_dims,
            'sentence_transformers_available': True,
            'enhanced_features': True
        }

# Test the enhanced embedding system
if __name__ == "__main__":
    embedder = EnhancedMultimodalEmbedder()

    # Test text embedding with medical entities
    test_text = "Patient presents with severe chest pain and shortness of breath. ECG shows abnormal rhythm. Doctor recommends immediate cardiac evaluation."
    text_embedding = embedder.embed_text(test_text, extract_medical_entities=True)
    print(f"[OK] Enhanced text embedding dimension: {len(text_embedding)}")

    # Print system info
    info = embedder.get_embedding_info()
    print(f"[OK] Enhanced embedding system info: {json.dumps(info, indent=2)}")
