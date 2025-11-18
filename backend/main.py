"""
FastAPI Backend for AI Insurance Claims Processing
Provides REST API endpoints for multimodal claim processing
"""

import os
import uuid
import json
import io
import base64
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

import json

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

# Import our modules
from qdrant_manager import QdrantManager
from embeddings import MultimodalEmbedder
from enhanced_embeddings import EnhancedMultimodalEmbedder
from recommender import ClaimsRecommender
from enhanced_recommender import EnhancedClaimsRecommender
from enhanced_recommender_advanced import EnhancedClaimsRecommenderAdvanced

# Import enhanced AIML components
from multitext_classifier import get_multitext_classifier
from enhanced_bert_classifier import get_enhanced_bert_classifier
from safe_features import get_safe_features
from enhanced_safe_features import get_enhanced_safe_features
from performance_validator import get_performance_validator
from memory_manager import get_memory_manager

# Initialize FastAPI app
app = FastAPI(
    title="AI Insurance Claims Processing API",
    description="Multimodal AI-powered insurance claims processing using Qdrant vector search",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
qdrant_manager = None
embedder = None
recommender = None

# Enhanced AIML components
enhanced_text_classifier = None
enhanced_safe_features = None
performance_validator = None
memory_manager = None

# Pydantic models for request/response
class ClaimData(BaseModel):
    claim_id: Optional[str] = None
    customer_id: str
    policy_number: str
    claim_type: str
    description: str
    amount: float
    date_submitted: Optional[str] = None
    location: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = {}

class ClaimSubmissionRequest(BaseModel):
    claim_data: ClaimData
    text_data: Optional[str] = None

class RecommendationRequest(BaseModel):
    claim_data: ClaimData
    text_embedding: Optional[List[float]] = None

class SearchRequest(BaseModel):
    query: str
    modality: str = "text_claims"
    limit: int = 5

class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global qdrant_manager, embedder, recommender, enhanced_embedder, enhanced_recommender, advanced_recommender
    global enhanced_text_classifier, enhanced_safe_features, performance_validator, memory_manager
    try:
        print("[INFO] Starting AI Insurance Claims Processing API...")

        qdrant_manager = QdrantManager()

        # Initialize memory manager first
        try:
            memory_manager = get_memory_manager()
            print("[OK] Memory manager initialized")
        except Exception as e:
            print(f"[WARNING] Memory manager failed to initialize: {e}")

        # Initialize enhanced AIML components
        try:
            enhanced_text_classifier = get_enhanced_bert_classifier()
            enhanced_safe_features = get_enhanced_safe_features()
            performance_validator = get_performance_validator()
            print("[OK] Enhanced AIML components initialized")
            print("[OK] BERT-based text classifier with CRF layer loaded")
            print("[OK] Enhanced SAFE feature engineering with 200+ features ready")
            print("[OK] Performance validation framework activated")
        except Exception as e:
            print(f"[WARNING] Enhanced AIML components failed to initialize: {e}")
            # Fall back to basic components
            enhanced_text_classifier = get_multitext_classifier()
            enhanced_safe_features = get_safe_features()

        # Try to initialize enhanced embedding systems
        try:
            enhanced_embedder = EnhancedMultimodalEmbedder()
            enhanced_recommender = EnhancedClaimsRecommender(qdrant_manager, enhanced_embedder)

            # Initialize advanced recommender with fraud detection capabilities
            advanced_recommender = EnhancedClaimsRecommenderAdvanced(qdrant_manager)

            embedder = enhanced_embedder  # Use enhanced as default
            recommender = enhanced_recommender  # Use enhanced as default
            print("[OK] Enhanced embedding systems initialized successfully")
            print("[OK] Advanced fraud detection capabilities loaded")
        except Exception as e:
            print(f"[WARNING] Enhanced embedding systems failed to initialize, falling back to basic: {e}")
            embedder = MultimodalEmbedder()
            recommender = ClaimsRecommender(qdrant_manager, embedder)
            enhanced_embedder = None
            enhanced_recommender = None
            advanced_recommender = None

        print("[OK] All services initialized successfully")
        print(f"[INFO] System ready with enhanced capabilities:")
        print(f"  - Enhanced BERT text classifier: {'Yes' if enhanced_text_classifier else 'No'}")
        print(f"  - Enhanced SAFE features: {'Yes' if enhanced_safe_features else 'No'}")
        print(f"  - Performance validation: {'Yes' if performance_validator else 'No'}")
        print(f"  - Memory management: {'Yes' if memory_manager else 'No'}")
    except Exception as e:
        print(f"[ERROR] Failed to initialize services: {e}")
        raise e

# Health check endpoint
@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Check API health status"""
    try:
        # Test Qdrant connection
        qdrant_status = qdrant_manager.test_connection() if qdrant_manager else False

        # Get embedding system info
        embedder_info = embedder.get_embedding_info() if embedder else {}

        # Check enhanced components status
        enhanced_status = {
            "enhanced_text_classifier": enhanced_text_classifier is not None,
            "enhanced_safe_features": enhanced_safe_features is not None,
            "performance_validator": performance_validator is not None,
            "memory_manager": memory_manager is not None
        }

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "qdrant_connected": qdrant_status,
                "embedder_loaded": embedder_info.get("text_model_loaded", False),
                "recommender_ready": recommender is not None,
                "enhanced_components": enhanced_status
            },
            "version": "2.0.0-enhanced",
            "capabilities": {
                "aiml_compliant_text_processing": enhanced_status["enhanced_text_classifier"],
                "enhanced_safe_feature_engineering": enhanced_status["enhanced_safe_features"],
                "performance_validation": enhanced_status["performance_validator"],
                "memory_optimization": enhanced_status["memory_manager"]
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Root endpoint
@app.get("/")
async def root():
    """API information"""
    return {
        "name": "Enhanced AI Insurance Claims Processing API",
        "version": "2.0.0-enhanced",
        "description": "Multimodal AI-powered insurance claims processing with AIML-compliant enhancements",
        "enhancements": [
            "Enhanced BERT-based text classification with CRF layer",
            "Multi-task learning for 6 AIML-specified tasks",
            "Enhanced SAFE feature engineering (200+ features)",
            "Smart feature interactions and mathematical transformations",
            "Comprehensive performance validation framework",
            "Memory optimization for Qdrant free tier",
            "Real-time A/B testing and benchmarking"
        ],
        "endpoints": [
            "/health - Health check with enhanced component status",
            "/submit_claim - Submit a new claim",
            "/get_recommendation - Get AI recommendation",
            "/enhanced_text_classification - Enhanced BERT+CRF classification",
            "/enhanced_feature_generation - Enhanced SAFE feature engineering",
            "/advanced_fraud_analysis - Comprehensive fraud analysis",
            "/performance_validation - Performance metrics and validation",
            "/run_ab_test - A/B testing for components",
            "/run_benchmark - System benchmarking",
            "/optimize_memory - Memory optimization",
            "/search_claims - Search similar claims",
            "/process_file - Process uploaded file",
            "/collections - Get collection info",
            "/system_info - Detailed system information"
        ]
    }

# Claim submission endpoint
@app.post("/submit_claim", response_model=APIResponse)
async def submit_claim(request: ClaimSubmissionRequest):
    """
    Submit a new insurance claim for processing

    Args:
        request: Claim submission request with claim data and text

    Returns:
        Processing results with recommendations
    """
    try:
        claim_data = request.claim_data.dict()

        # Generate claim ID if not provided
        if not claim_data.get('claim_id'):
            claim_data['claim_id'] = f"CLAIM_{uuid.uuid4().hex[:8].upper()}"

        # Add timestamp
        claim_data['date_submitted'] = claim_data.get('date_submitted') or datetime.now().isoformat()
        claim_data['processed_at'] = datetime.now().isoformat()

        # Generate text embedding
        text_content = request.text_data or claim_data.get('description', '')
        text_embedding = embedder.embed_text(text_content)

        # Store claim in Qdrant
        point_id = qdrant_manager.add_claim(
            claim_data=claim_data,
            embedding=text_embedding,
            modality='text_claims'
        )

        # Generate recommendation
        recommendation = recommender.recommend_outcome(
            claim_data=claim_data,
            text_embedding=text_embedding,
            modality='text_claims'
        )

        # Add claim ID to recommendation
        recommendation['claim_id'] = claim_data['claim_id']

        return APIResponse(
            success=True,
            message="Claim processed successfully",
            data={
                "claim_id": claim_data['claim_id'],
                "point_id": point_id,
                "recommendation": recommendation
            }
        )

    except Exception as e:
        return APIResponse(
            success=False,
            message="Failed to process claim",
            error=str(e)
        )

# Recommendation endpoint
@app.post("/get_recommendation", response_model=APIResponse)
async def get_recommendation(request: RecommendationRequest):
    """
    Get AI recommendation for an existing claim

    Args:
        request: Recommendation request with claim data

    Returns:
        AI recommendation with confidence scores
    """
    try:
        claim_data = request.claim_data.dict()

        # Generate recommendation
        recommendation = recommender.recommend_outcome(
            claim_data=claim_data,
            text_embedding=request.text_embedding,
            modality='text_claims'
        )

        return APIResponse(
            success=True,
            message="Recommendation generated successfully",
            data=recommendation
        )

    except Exception as e:
        return APIResponse(
            success=False,
            message="Failed to generate recommendation",
            error=str(e)
        )

# Search claims endpoint
@app.post("/search_claims", response_model=APIResponse)
async def search_claims(request: SearchRequest):
    """
    Search for similar claims using vector similarity

    Args:
        request: Search request with query and parameters

    Returns:
        List of similar claims with similarity scores
    """
    try:
        # Generate query embedding
        query_embedding = embedder.embed_text(request.query)

        # Search similar claims
        similar_claims = qdrant_manager.search_similar_claims(
            query_embedding=query_embedding,
            modality=request.modality,
            limit=request.limit
        )

        return APIResponse(
            success=True,
            message=f"Found {len(similar_claims)} similar claims",
            data={
                "query": request.query,
                "modality": request.modality,
                "similar_claims": similar_claims,
                "count": len(similar_claims)
            }
        )

    except Exception as e:
        return APIResponse(
            success=False,
            message="Failed to search claims",
            error=str(e)
        )

# File processing endpoint
@app.post("/process_file", response_model=APIResponse)
async def process_file(
    file: UploadFile = File(...),
    claim_id: str = Form(...),
    customer_id: str = Form(...),
    claim_type: str = Form(...),
    description: str = Form(...),
    amount: float = Form(...),
    modality: str = Form("text")
):
    """
    Process uploaded file (image, audio, video) and generate claim

    Args:
        file: Uploaded file
        claim_id: Claim ID
        customer_id: Customer ID
        claim_type: Type of claim
        description: Claim description
        amount: Claim amount
        modality: Type of file (image, audio, video)

    Returns:
        Processing results with recommendations
    """
    try:
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)

        print(f"[INFO] Processing {modality} file: {file.filename} ({file_size} bytes)")

        # Prepare claim data
        claim_data = {
            "claim_id": claim_id,
            "customer_id": customer_id,
            "policy_number": f"POL_{customer_id[-4:]}",
            "claim_type": claim_type,
            "description": description,
            "amount": amount,
            "date_submitted": datetime.now().isoformat(),
            "processed_at": datetime.now().isoformat(),
            "file_info": {
                "filename": file.filename,
                "size": file_size,
                "type": file.content_type,
                "modality": modality
            }
        }

        # Generate embedding based on modality
        if modality == "image":
            embedding = embedder.embed_image(image_data=file_content)
            qdrant_modality = "image_claims"
        elif modality == "audio":
            embedding = embedder.embed_audio(audio_data=file_content)
            qdrant_modality = "audio_claims"
        elif modality == "video":
            embedding = embedder.embed_video(video_data=file_content)
            qdrant_modality = "video_claims"
        else:
            # Default to text processing
            embedding = embedder.embed_text(description)
            qdrant_modality = "text_claims"

        # Store in Qdrant
        point_id = qdrant_manager.add_claim(
            claim_data=claim_data,
            embedding=embedding,
            modality=qdrant_modality
        )

        # Generate recommendation using text embedding for analysis
        text_embedding = embedder.embed_text(description)
        recommendation = recommender.recommend_outcome(
            claim_data=claim_data,
            text_embedding=text_embedding,
            modality=qdrant_modality
        )

        return APIResponse(
            success=True,
            message=f"{modality.capitalize()} file processed successfully",
            data={
                "claim_id": claim_id,
                "point_id": point_id,
                "file_info": claim_data["file_info"],
                "recommendation": recommendation
            }
        )

    except Exception as e:
        return APIResponse(
            success=False,
            message=f"Failed to process {modality} file",
            error=str(e)
        )

# Cross-modal search endpoint
@app.post("/search_cross_modal", response_model=APIResponse)
async def search_cross_modal(
    query: str = Form(...),
    modalities: str = Form("text_claims,image_claims")
):
    """
    Search across multiple modalities simultaneously

    Args:
        query: Search query
        modalities: Comma-separated list of modalities to search

    Returns:
        Cross-modal search results
    """
    try:
        # Parse modalities
        modality_list = [m.strip() for m in modalities.split(",")]

        # Generate query embedding
        query_embedding = embedder.embed_text(query)

        # Search across modalities
        cross_modal_results = qdrant_manager.search_cross_modal(
            query_embedding=query_embedding,
            search_modalities=modality_list,
            limit_per_modality=3
        )

        total_results = sum(len(results) for results in cross_modal_results.values())

        return APIResponse(
            success=True,
            message=f"Found {total_results} results across {len(modality_list)} modalities",
            data={
                "query": query,
                "modalities_searched": modality_list,
                "results": cross_modal_results,
                "total_results": total_results
            }
        )

    except Exception as e:
        return APIResponse(
            success=False,
            message="Failed to perform cross-modal search",
            error=str(e)
        )

# Collections info endpoint
@app.get("/collections", response_model=APIResponse)
async def get_collections_info():
    """Get information about Qdrant collections"""
    try:
        collection_info = qdrant_manager.get_collection_info()

        return APIResponse(
            success=True,
            message="Collection information retrieved successfully",
            data=collection_info
        )

    except Exception as e:
        return APIResponse(
            success=False,
            message="Failed to retrieve collection information",
            error=str(e)
        )

# System info endpoint
@app.get("/system_info", response_model=APIResponse)
async def get_system_info():
    """Get detailed system information"""
    try:
        # Get embedding system info
        embedder_info = embedder.get_embedding_info()

        # Get collection info
        collection_info = qdrant_manager.get_collection_info()

        # System status
        system_status = {
            "api_status": "running",
            "qdrant_connected": qdrant_manager.test_connection(),
            "timestamp": datetime.now().isoformat(),
            "python_version": os.sys.version,
            "working_directory": os.getcwd()
        }

        return APIResponse(
            success=True,
            message="System information retrieved successfully",
            data={
                "system_status": system_status,
                "embedder_info": embedder_info,
                "collection_info": collection_info
            }
        )

    except Exception as e:
        return APIResponse(
            success=False,
            message="Failed to retrieve system information",
            error=str(e)
        )

# Advanced fraud analysis endpoint
@app.post("/advanced_fraud_analysis", response_model=APIResponse)
async def advanced_fraud_analysis(request: ClaimSubmissionRequest):
    """
    Perform comprehensive fraud analysis using advanced ML techniques

    Args:
        request: Claim submission request with claim data and text

    Returns:
        Comprehensive fraud analysis with risk factors and recommendations
    """
    try:
        claim_data = request.claim_data.dict()
        text_content = request.text_data or claim_data.get('description', '')

        # Check if advanced recommender is available
        if not advanced_recommender:
            return APIResponse(
                success=False,
                message="Advanced fraud analysis not available",
                error="Advanced recommender not initialized"
            )

        # Perform comprehensive analysis
        analysis_result = advanced_recommender.recommend_outcome(
            claim_data=claim_data,
            include_detailed_analysis=True
        )

        # Create response with custom JSON encoder for numpy types
        response_data = {
            "success": True,
            "message": "Advanced fraud analysis completed successfully",
            "data": analysis_result
        }

        return JSONResponse(
            content=json.dumps(response_data, cls=NumpyEncoder),
            media_type="application/json"
        )

    except Exception as e:
        return APIResponse(
            success=False,
            message="Failed to perform advanced fraud analysis",
            error=str(e)
        )

# Enhanced text classification endpoint
@app.post("/enhanced_text_classification", response_model=APIResponse)
async def enhanced_text_classification(request: ClaimSubmissionRequest):
    """
    Perform enhanced text classification using BERT + CRF with multi-task learning

    Args:
        request: Claim submission request with claim data and text

    Returns:
        Enhanced classification results with task-specific predictions
    """
    try:
        claim_data = request.claim_data.dict()
        text_content = request.text_data or claim_data.get('description', '')

        if not enhanced_text_classifier:
            return APIResponse(
                success=False,
                message="Enhanced text classifier not available",
                error="Enhanced text classifier not initialized"
            )

        # Perform enhanced classification
        start_time = time.time()
        classification_result = enhanced_text_classifier.classify_claim(text_content, claim_data)
        processing_time = time.time() - start_time

        # Record in performance validator if available
        if performance_validator:
            performance_validator.metrics_collector.record_prediction(
                {
                    'confidence': classification_result.get('fraud_indicators', {}).get('analysis_confidence', 0.0),
                    'fraud_score': classification_result.get('fraud_indicators', {}).get('total_risk_score', 0.0)
                },
                processing_time,
                success=True
            )

        return APIResponse(
            success=True,
            message="Enhanced text classification completed successfully",
            data={
                "classification_result": classification_result,
                "processing_time_ms": processing_time * 1000,
                "model_type": classification_result.get('model_type', 'enhanced_bert'),
                "domain_adaptation": classification_result.get('domain_adaptation', False)
            }
        )

    except Exception as e:
        return APIResponse(
            success=False,
            message="Failed to perform enhanced text classification",
            error=str(e)
        )

# Enhanced feature generation endpoint
@app.post("/enhanced_feature_generation", response_model=APIResponse)
async def enhanced_feature_generation(request: ClaimSubmissionRequest):
    """
    Generate enhanced SAFE features with smart interactions and transformations

    Args:
        request: Claim submission request with claim data

    Returns:
        Enhanced feature set with AIML-style comprehensive features
    """
    try:
        claim_data = request.claim_data.dict()

        if not enhanced_safe_features:
            return APIResponse(
                success=False,
                message="Enhanced SAFE features not available",
                error="Enhanced SAFE features not initialized"
            )

        # Generate enhanced features
        start_time = time.time()
        enhanced_features = enhanced_safe_features.generate_comprehensive_features(claim_data)
        processing_time = time.time() - start_time

        # Analyze feature statistics
        feature_stats = {
            'total_features': len(enhanced_features),
            'feature_categories': {},
            'memory_usage_mb': enhanced_safe_features.get_memory_usage().get('system_memory', {}).get('current_usage_mb', 0)
        }

        # Categorize features
        for feature_name in enhanced_features.keys():
            if 'temporal' in feature_name:
                feature_stats['feature_categories']['temporal'] = feature_stats['feature_categories'].get('temporal', 0) + 1
            elif 'amount' in feature_name:
                feature_stats['feature_categories']['amount'] = feature_stats['feature_categories'].get('amount', 0) + 1
            elif 'frequency' in feature_name:
                feature_stats['feature_categories']['frequency'] = feature_stats['feature_categories'].get('frequency', 0) + 1
            elif 'geo' in feature_name or 'location' in feature_name:
                feature_stats['feature_categories']['geographic'] = feature_stats['feature_categories'].get('geographic', 0) + 1
            elif 'behavioral' in feature_name:
                feature_stats['feature_categories']['behavioral'] = feature_stats['feature_categories'].get('behavioral', 0) + 1
            elif '_x_' in feature_name:
                feature_stats['feature_categories']['interactions'] = feature_stats['feature_categories'].get('interactions', 0) + 1
            elif any(prefix in feature_name for prefix in ['log_', 'sqrt_', 'squared_']):
                feature_stats['feature_categories']['transformations'] = feature_stats['feature_categories'].get('transformations', 0) + 1

        # Record in performance validator if available
        if performance_validator:
            performance_validator.metrics_collector.record_prediction(
                {'feature_count': len(enhanced_features)},
                processing_time,
                success=True
            )

        return APIResponse(
            success=True,
            message="Enhanced feature generation completed successfully",
            data={
                "enhanced_features": enhanced_features,
                "feature_statistics": feature_stats,
                "processing_time_ms": processing_time * 1000,
                "target_feature_count": enhanced_safe_features.max_features,
                "meets_target": len(enhanced_features) >= 200  # AIML target
            }
        )

    except Exception as e:
        return APIResponse(
            success=False,
            message="Failed to generate enhanced features",
            error=str(e)
        )

# Performance validation endpoint
@app.get("/performance_validation", response_model=APIResponse)
async def get_performance_validation():
    """
    Get comprehensive performance validation results and real-time metrics

    Returns:
        Performance report with A/B test results, benchmarks, and system metrics
    """
    try:
        if not performance_validator:
            return APIResponse(
                success=False,
                message="Performance validator not available",
                error="Performance validator not initialized"
            )

        # Generate comprehensive performance report
        performance_report = performance_validator.generate_performance_report()

        # Get real-time dashboard data
        dashboard_data = performance_validator.get_real_time_dashboard_data()

        # Get memory status
        memory_status = {}
        if memory_manager:
            memory_status = memory_manager.check_memory_usage()

        return APIResponse(
            success=True,
            message="Performance validation report generated successfully",
            data={
                "performance_report": performance_report,
                "dashboard_data": dashboard_data,
                "memory_status": memory_status,
                "system_targets": performance_validator.aiml_targets
            }
        )

    except Exception as e:
        return APIResponse(
            success=False,
            message="Failed to generate performance validation report",
            error=str(e)
        )

# A/B testing endpoint
@app.post("/run_ab_test", response_model=APIResponse)
async def run_ab_test(
    test_name: str = Form(...),
    component_type: str = Form("text_classifier"),
    test_iterations: int = Form(50)
):
    """
    Run A/B test comparing enhanced vs baseline components

    Args:
        test_name: Name of the A/B test
        component_type: Type of component to test (text_classifier, safe_features)
        test_iterations: Number of test iterations

    Returns:
        A/B test results with statistical significance analysis
    """
    try:
        if not performance_validator:
            return APIResponse(
                success=False,
                message="Performance validator not available",
                error="Performance validator not initialized"
            )

        if component_type == "text_classifier":
            # Get baseline and enhanced classifiers
            baseline_classifier = get_multitext_classifier()
            enhanced_classifier = enhanced_text_classifier

            if not enhanced_classifier:
                return APIResponse(
                    success=False,
                    message="Enhanced text classifier not available",
                    error="Enhanced text classifier not initialized"
                )

            # Run A/B test
            ab_test_result = performance_validator.run_ab_test(
                control_component=baseline_classifier,
                treatment_component=enhanced_classifier,
                test_name=test_name,
                component_type='text_classifier'
            )

        elif component_type == "safe_features":
            # Get baseline and enhanced SAFE features
            baseline_safe = get_safe_features()
            enhanced_safe = enhanced_safe_features

            if not enhanced_safe:
                return APIResponse(
                    success=False,
                    message="Enhanced SAFE features not available",
                    error="Enhanced SAFE features not initialized"
                )

            # Run A/B test
            ab_test_result = performance_validator.run_ab_test(
                control_component=baseline_safe,
                treatment_component=enhanced_safe,
                test_name=test_name,
                component_type='safe_features'
            )

        else:
            return APIResponse(
                success=False,
                message="Unknown component type",
                error=f"Component type '{component_type}' not supported"
            )

        return APIResponse(
            success=True,
            message=f"A/B test '{test_name}' completed successfully",
            data={
                "test_name": test_name,
                "component_type": component_type,
                "ab_test_result": {
                    "control_accuracy": ab_test_result.control_metrics.accuracy,
                    "treatment_accuracy": ab_test_result.treatment_metrics.accuracy,
                    "improvement_percentage": ab_test_result.improvement_percentage,
                    "is_significant": ab_test_result.is_significant,
                    "p_value": ab_test_result.p_value,
                    "recommendation": ab_test_result.recommendation,
                    "control_processing_time_ms": ab_test_result.control_metrics.processing_time * 1000,
                    "treatment_processing_time_ms": ab_test_result.treatment_metrics.processing_time * 1000
                }
            }
        )

    except Exception as e:
        return APIResponse(
            success=False,
            message="Failed to run A/B test",
            error=str(e)
        )

# Memory management endpoint
@app.post("/optimize_memory", response_model=APIResponse)
async def optimize_memory():
    """
    Optimize system memory usage

    Returns:
        Memory optimization results with freed memory statistics
    """
    try:
        if not memory_manager:
            return APIResponse(
                success=False,
                message="Memory manager not available",
                error="Memory manager not initialized"
            )

        # Get memory before optimization
        memory_before = memory_manager.check_memory_usage()

        # Perform memory optimization
        optimization_result = memory_manager.optimize_for_memory()

        # Get memory after optimization
        memory_after = memory_manager.check_memory_usage()

        return APIResponse(
            success=True,
            message="Memory optimization completed successfully",
            data={
                "memory_before_mb": memory_before.get('current_usage_mb', 0),
                "memory_after_mb": memory_after.get('current_usage_mb', 0),
                "memory_freed_mb": optimization_result.get('total_memory_saved_mb', 0),
                "optimization_actions": optimization_result.get('actions_taken', []),
                "current_efficiency_score": memory_manager.get_memory_efficiency_score(),
                "recommendations": memory_manager.get_recommendations()
            }
        )

    except Exception as e:
        return APIResponse(
            success=False,
            message="Failed to optimize memory",
            error=str(e)
        )

# System benchmark endpoint
@app.post("/run_benchmark", response_model=APIResponse)
async def run_benchmark(
    component_name: str = Form("enhanced_system"),
    test_iterations: int = Form(100)
):
    """
    Run performance benchmark for system components

    Args:
        component_name: Name of component to benchmark
        test_iterations: Number of benchmark iterations

    Returns:
        Benchmark results with performance metrics
    """
    try:
        if not performance_validator:
            return APIResponse(
                success=False,
                message="Performance validator not available",
                error="Performance validator not initialized"
            )

        # Select component to benchmark
        if component_name == "enhanced_text_classifier" and enhanced_text_classifier:
            component = enhanced_text_classifier
        elif component_name == "enhanced_safe_features" and enhanced_safe_features:
            component = enhanced_safe_features
        elif component_name == "enhanced_system":
            # Benchmark the entire system with a sample claim
            component = type('SystemBenchmark', (), {
                'classify_claim': lambda text, data: enhanced_text_classifier.classify_claim(text, data) if enhanced_text_classifier else {},
                'generate_comprehensive_features': lambda data: enhanced_safe_features.generate_comprehensive_features(data) if enhanced_safe_features else {}
            })()
        else:
            return APIResponse(
                success=False,
                message="Component not available",
                error=f"Component '{component_name}' not found or not initialized"
            )

        # Run benchmark
        benchmark_result = performance_validator.run_benchmark(
            component=component,
            component_name=component_name,
            test_iterations=test_iterations
        )

        return APIResponse(
            success=True,
            message=f"Benchmark for '{component_name}' completed successfully",
            data={
                "benchmark_result": {
                    "component_name": benchmark_result.component_name,
                    "test_iterations": benchmark_result.test_data_size,
                    "avg_processing_time_ms": benchmark_result.avg_processing_time * 1000,
                    "max_processing_time_ms": benchmark_result.max_processing_time * 1000,
                    "min_processing_time_ms": benchmark_result.min_processing_time * 1000,
                    "throughput_per_second": benchmark_result.throughput_per_second,
                    "success_rate": benchmark_result.success_rate,
                    "error_rate": benchmark_result.error_rate,
                    "memory_peak_mb": benchmark_result.memory_peak,
                    "memory_average_mb": benchmark_result.memory_average
                }
            }
        )

    except Exception as e:
        return APIResponse(
            success=False,
            message="Failed to run benchmark",
            error=str(e)
        )

# Run server
if __name__ == "__main__":
    print("[INFO] Starting AI Insurance Claims Processing Server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
