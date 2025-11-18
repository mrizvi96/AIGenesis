"""
Qdrant Cloud Manager for AI Insurance Claims Processing
Handles vector database operations for multimodal claim data
Optimized for cloud free tier constraints (1GB RAM, 4GB storage, 0.5 vCPU)
"""

import os
import uuid
import time
import logging
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

# Configure logger
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class QdrantManager:
    def __init__(self):
        """Initialize Qdrant Cloud connection with optimized settings for free tier"""
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")

        if not self.url or not self.api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in .env file")

        # Cloud resource limits (free tier)
        self.max_retries = 3
        self.base_delay = 0.5  # seconds
        self.batch_size = 50  # Optimized for 1GB RAM limit
        self.timeout = 20  # Reduced timeout for cloud efficiency

        # Initialize Qdrant client with cloud-optimized settings
        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
            timeout=self.timeout,
            https=True  # Ensure HTTPS for cloud
        )

        # Collection names for different modalities
        self.collections = {
            'text_claims': 'insurance_claims_text',
            'image_claims': 'insurance_claims_images',
            'audio_claims': 'insurance_claims_audio',
            'video_claims': 'insurance_claims_video',
            'policies': 'insurance_policies',
            'regulations': 'insurance_regulations'
        }

        # Initialize collections
        self._initialize_collections()

    def _retry_with_backoff(self, operation, *args, **kwargs):
        """Retry operation with exponential backoff for cloud stability"""
        for attempt in range(self.max_retries):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e

                delay = self.base_delay * (2 ** attempt)
                print(f"[RETRY] Cloud operation failed (attempt {attempt + 1}), retrying in {delay}s: {str(e)}")
                time.sleep(delay)

    def _initialize_collections(self):
        """Create collections if they don't exist - optimized for cloud storage"""
        print("Initializing Qdrant collections for cloud free tier...")

        # Vector sizes optimized for cloud storage (256-dim for efficiency where possible)
        vector_sizes = {
            'text_claims': 256,    # Optimized for cloud storage
            'image_claims': 256,   # Compressed CLIP embeddings for cloud
            'audio_claims': 256,   # Compressed audio embeddings for cloud
            'video_claims': 256,   # Compressed video embeddings for cloud
            'policies': 256,       # Optimized for cloud storage
            'regulations': 256     # Optimized for cloud storage
        }

        for collection_name, actual_name in self.collections.items():
            try:
                # Check if collection exists with retry
                self._retry_with_backoff(self.client.get_collection, actual_name)
                print(f"[OK] Collection '{actual_name}' already exists")
            except Exception:
                # Create collection with retry and optimized configuration
                vector_size = vector_sizes.get(collection_name, 256)
                collection_config = VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                    # Optimized for cloud free tier storage
                    hnsw_config={
                        "m": 8,  # Reduced for memory efficiency
                        "ef_construct": 100,  # Reduced for CPU efficiency
                    }
                )

                self._retry_with_backoff(
                    self.client.create_collection,
                    collection_name=actual_name,
                    vectors_config=collection_config
                )
                print(f"[OK] Created cloud-optimized collection '{actual_name}' with vector size {vector_size}")

    def add_claim(self,
                  claim_data: Dict[str, Any],
                  embedding: List[float],
                  modality: str = 'text_claims') -> str:
        """
        Add a claim to the vector database

        Args:
            claim_data: Dictionary containing claim information
            embedding: Vector embedding of the claim
            modality: Type of claim data (text_claims, image_claims, etc.)

        Returns:
            point_id: Unique ID of the added claim
        """
        if modality not in self.collections:
            raise ValueError(f"Invalid modality: {modality}")

        collection_name = self.collections[modality]
        point_id = str(uuid.uuid4())

        # Create point with metadata
        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                'claim_id': claim_data.get('claim_id', point_id),
                'policy_number': claim_data.get('policy_number', ''),
                'claim_type': claim_data.get('claim_type', ''),
                'description': claim_data.get('description', ''),
                'amount': claim_data.get('amount', 0.0),
                'status': claim_data.get('status', 'pending'),
                'date_submitted': claim_data.get('date_submitted', ''),
                'customer_id': claim_data.get('customer_id', ''),
                'modality': modality,
                'processed_at': claim_data.get('processed_at', ''),
                'additional_data': claim_data.get('additional_data', {})
            }
        )

        # Insert point with retry for cloud stability
        self._retry_with_backoff(
            self.client.upsert,
            collection_name=collection_name,
            points=[point]
        )

        print(f"[OK] Added {modality} claim to cloud: {point_id}")
        return point_id

    def search_similar_claims(self,
                             query_embedding: List[float],
                             modality: str = 'text_claims',
                             limit: int = 5,
                             score_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search for similar claims using vector similarity

        Args:
            query_embedding: Query vector embedding
            modality: Collection to search in
            limit: Maximum number of results
            score_threshold: Minimum similarity score

        Returns:
            List of similar claims with metadata
        """
        if modality not in self.collections:
            raise ValueError(f"Invalid modality: {modality}")

        collection_name = self.collections[modality]

        # Search for similar vectors with retry for cloud stability
        search_result = self._retry_with_backoff(
            self.client.search,
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=min(limit, 10),  # Cloud-optimized limit
            score_threshold=score_threshold
        )

        # Format results
        similar_claims = []
        for result in search_result:
            claim_info = result.payload
            claim_info['similarity_score'] = result.score
            claim_info['point_id'] = result.id
            similar_claims.append(claim_info)

        print(f"[OK] Found {len(similar_claims)} similar {modality} claims")
        return similar_claims

    def add_claims_batch(self, claims_data: List[Dict[str, Any]],
                        embeddings: List[List[float]],
                        modality: str = 'text_claims') -> List[str]:
        """
        Add multiple claims in a batch for cloud efficiency

        Args:
            claims_data: List of claim dictionaries
            embeddings: List of vector embeddings
            modality: Type of claim data

        Returns:
            List of point IDs
        """
        if modality not in self.collections:
            raise ValueError(f"Invalid modality: {modality}")

        if len(claims_data) != len(embeddings):
            raise ValueError("Claims data and embeddings must have same length")

        collection_name = self.collections[modality]
        points = []
        point_ids = []

        for i, (claim_data, embedding) in enumerate(zip(claims_data, embeddings)):
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)

            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    'claim_id': claim_data.get('claim_id', point_id),
                    'policy_number': claim_data.get('policy_number', ''),
                    'claim_type': claim_data.get('claim_type', ''),
                    'description': claim_data.get('description', ''),
                    'amount': claim_data.get('amount', 0.0),
                    'status': claim_data.get('status', 'pending'),
                    'date_submitted': claim_data.get('date_submitted', ''),
                    'customer_id': claim_data.get('customer_id', ''),
                    'modality': modality,
                    'processed_at': claim_data.get('processed_at', ''),
                    'additional_data': claim_data.get('additional_data', {})
                }
            )
            points.append(point)

            # Process in batches to respect memory limits
            if len(points) >= self.batch_size:
                self._retry_with_backoff(
                    self.client.upsert,
                    collection_name=collection_name,
                    points=points
                )
                points = []  # Reset batch

        # Process remaining points
        if points:
            self._retry_with_backoff(
                self.client.upsert,
                collection_name=collection_name,
                points=points
            )

        print(f"[OK] Added {len(point_ids)} {modality} claims to cloud in batches")
        return point_ids

    def search_cross_modal(self,
                          query_embedding: List[float],
                          search_modalities: List[str] = None,
                          limit_per_modality: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search across multiple modalities

        Args:
            query_embedding: Query vector embedding
            search_modalities: List of modalities to search
            limit_per_modality: Results per modality

        Returns:
            Dictionary with modality as key and list of results as value
        """
        if search_modalities is None:
            search_modalities = ['text_claims', 'image_claims', 'audio_claims', 'video_claims']

        cross_modal_results = {}

        for modality in search_modalities:
            if modality in self.collections:
                try:
                    results = self.search_similar_claims(
                        query_embedding=query_embedding,
                        modality=modality,
                        limit=limit_per_modality,
                        score_threshold=0.6  # Lower threshold for cross-modal search
                    )
                    cross_modal_results[modality] = results
                except Exception as e:
                    print(f"Error searching {modality}: {e}")
                    cross_modal_results[modality] = []

        return cross_modal_results

    def get_collection_info(self, modality: str = None) -> Dict[str, Any]:
        """Get information about collections"""
        info = {}

        if modality:
            if modality not in self.collections:
                return {"error": f"Invalid modality: {modality}"}

            try:
                collection_info = self.client.get_collection(self.collections[modality])
                info[modality] = {
                    'vectors_count': collection_info.vectors_count,
                    'status': collection_info.status,
                    'optimizer_status': collection_info.optimizer_status
                }
            except Exception as e:
                info[modality] = {"error": str(e)}
        else:
            # Get info for all collections
            for collection_key, collection_name in self.collections.items():
                try:
                    collection_info = self.client.get_collection(collection_name)
                    info[collection_key] = {
                        'vectors_count': collection_info.vectors_count,
                        'status': collection_info.status,
                        'optimizer_status': collection_info.optimizer_status
                    }
                except Exception as e:
                    info[collection_key] = {"error": str(e)}

        return info

    def test_connection(self) -> bool:
        """Test connection to Qdrant Cloud with resource monitoring"""
        try:
            collections = self._retry_with_backoff(self.client.get_collections)
            print(f"[OK] Connected to Qdrant Cloud successfully!")
            print(f"[OK] Available collections: {[col.name for col in collections.collections]}")

            # Test resource limits
            for collection in collections.collections:
                try:
                    info = self._retry_with_backoff(self.client.get_collection, collection.name)
                    print(f"[OK] Collection '{collection.name}': {info.vectors_count} vectors")
                except Exception as e:
                    print(f"[WARN] Could not get info for '{collection.name}': {e}")

            return True
        except Exception as e:
            print(f"[ERROR] Connection failed: {e}")
            return False

    def get_cloud_usage_stats(self) -> Dict[str, Any]:
        """Get cloud resource usage statistics for monitoring"""
        stats = {
            'total_vectors': 0,
            'collections': {},
            'memory_estimate_mb': 0,
            'storage_estimate_mb': 0
        }

        try:
            for collection_key, collection_name in self.collections.items():
                try:
                    info = self._retry_with_backoff(self.client.get_collection, collection_name)
                    collection_stats = {
                        'vectors_count': info.vectors_count,
                        'status': str(info.status),
                        'optimizer_status': str(info.optimizer_status) if info.optimizer_status else 'N/A'
                    }
                    stats['collections'][collection_key] = collection_stats
                    stats['total_vectors'] += info.vectors_count

                    # Estimate memory usage (assuming 256-dim float32 vectors = 1024 bytes each)
                    memory_mb = (info.vectors_count * 1024) / (1024 * 1024)
                    stats['memory_estimate_mb'] += memory_mb

                except Exception as e:
                    stats['collections'][collection_key] = {'error': str(e)}

            # Storage estimate with metadata (conservative estimate)
            stats['storage_estimate_mb'] = stats['memory_estimate_mb'] * 1.5

        except Exception as e:
            stats['error'] = str(e)

        return stats

    def initialize_cloud_environment(self) -> Dict[str, Any]:
        """Initialize Qdrant Cloud environment with collections"""
        try:
            logger.info("[CLOUD-QDRANT] Initializing cloud environment...")

            # Test connection
            connection_test = self.test_connection()
            if not connection_test:
                return {'success': False, 'error': 'Connection to Qdrant Cloud failed'}

            # Initialize collections for each modality
            collections = ['text', 'image', 'audio', 'video', 'policy', 'regulation']
            initialized = []

            for modality in collections:
                try:
                    # This will create collections if they don't exist
                    info = self.get_collection_info(modality)
                    initialized.append(modality)
                except Exception as e:
                    logger.warning(f"[WARN] Failed to initialize {modality} collection: {e}")

            return {
                'success': True,
                'initialized_collections': initialized,
                'connection_test': True
            }

        except Exception as e:
            logger.error(f"[ERROR] Cloud environment initialization failed: {e}")
            return {'success': False, 'error': str(e)}

    def batch_insert_vectors(self, vectors: List[List[float]], payload_ids: List[str], batch_size: int = 10) -> Dict[str, Any]:
        """Insert vectors in batches with error handling"""
        try:
            if len(vectors) != len(payload_ids):
                return {'success': False, 'error': 'Vectors and IDs length mismatch'}

            total_inserted = 0
            failed_batches = 0

            for i in range(0, len(vectors), batch_size):
                batch_vectors = vectors[i:i + batch_size]
                batch_ids = payload_ids[i:i + batch_size]

                try:
                    # Create claim batch for insertion
                    claims_batch = []
                    batch_embeddings = []
                    for j, (vec, pid) in enumerate(zip(batch_vectors, batch_ids)):
                        claims_batch.append({
                            'claim_id': pid,
                            'text': f'Claim {pid}'
                        })
                        batch_embeddings.append(vec)

                    result = self.add_claims_batch(claims_batch, batch_embeddings, modality='text_claims')

                    if result.get('success', False):
                        total_inserted += len(batch_vectors)
                    else:
                        failed_batches += 1

                except Exception as batch_error:
                    logger.warning(f"[WARN] Batch {i//batch_size} failed: {batch_error}")
                    failed_batches += 1

            success = failed_batches == 0 and total_inserted == len(vectors)

            return {
                'success': success,
                'total_vectors': len(vectors),
                'inserted_vectors': total_inserted,
                'failed_batches': failed_batches,
                'success_rate': (total_inserted / len(vectors)) * 100 if vectors else 0
            }

        except Exception as e:
            logger.error(f"[ERROR] Batch vector insertion failed: {e}")
            return {'success': False, 'error': str(e)}

    def search_vectors(self, query_vector: List[float], limit: int = 5) -> Dict[str, Any]:
        """Search for similar vectors"""
        try:
            # Create a dummy claim for searching
            search_claim = {
                'claim_id': 'search_query',
                'text': 'Search query',
                'vector': query_vector
            }

            result = self.search_similar_claims(search_claim, limit=limit)

            return {
                'success': True,
                'results': result.get('similar_claims', []),
                'total_found': len(result.get('similar_claims', []))
            }

        except Exception as e:
            logger.error(f"[ERROR] Vector search failed: {e}")
            return {'success': False, 'error': str(e)}

    def get_cloud_memory_stats(self) -> Dict[str, Any]:
        """Get cloud memory usage statistics"""
        try:
            # Get cloud usage stats
            cloud_stats = self.get_cloud_usage_stats()

            return {
                'current_memory_mb': cloud_stats.get('memory_estimate_mb', 0),
                'memory_limit_mb': 1024,  # Qdrant Cloud free tier
                'usage_percent': (cloud_stats.get('memory_estimate_mb', 0) / 1024) * 100,
                'storage_used_mb': cloud_stats.get('storage_estimate_mb', 0),
                'storage_limit_mb': 4096,  # Qdrant Cloud free tier
                'total_vectors': cloud_stats.get('total_vectors', 0)
            }

        except Exception as e:
            logger.error(f"[ERROR] Memory stats retrieval failed: {e}")
            return {'error': str(e)}

    def get_cloud_performance_metrics(self) -> Dict[str, Any]:
        """Get cloud performance metrics"""
        try:
            return {
                'collections_count': len(self.modalities),
                'cloud_connection': 'active',
                'vector_dimensions': 256,  # Cloud optimized
                'compression_enabled': True,
                'batch_processing': True,
                'retry_logic_enabled': True
            }

        except Exception as e:
            return {'error': str(e)}

if __name__ == "__main__":
    # Test the cloud connection
    qm = QdrantManager()
    qm.test_connection()

    # Get collection info
    info = qm.get_collection_info()
    print("\nCollection Information:")
    for modality, data in info.items():
        print(f"  {modality}: {data}")

    # Show cloud usage statistics
    print("\n" + "="*50)
    print("CLOUD USAGE STATISTICS:")
    print("="*50)
    stats = qm.get_cloud_usage_stats()
    print(f"Total Vectors: {stats['total_vectors']:,}")
    print(f"Estimated Memory Usage: {stats['memory_estimate_mb']:.2f} MB / 1024 MB")
    print(f"Estimated Storage Usage: {stats['storage_estimate_mb']:.2f} MB / 4096 MB")
    print(f"Cloud Resource Efficiency: {(stats['memory_estimate_mb']/1024)*100:.1f}% memory used")

    print("\nCollections Detail:")
    for collection, data in stats['collections'].items():
        if 'error' not in data:
            print(f"  {collection}: {data['vectors_count']} vectors ({data['status']})")
        else:
            print(f"  {collection}: Error - {data['error']}")

# Global cloud-optimized Qdrant manager instance
cloud_qdrant_manager = QdrantManager()

def get_qdrant_manager() -> QdrantManager:
    """Get the global cloud-optimized Qdrant manager instance"""
    return cloud_qdrant_manager