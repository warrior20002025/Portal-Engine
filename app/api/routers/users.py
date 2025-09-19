"""
Users API router
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import Response
from app.core.logging import get_logger, log_exception, log_api_call, log_api_response
from app.models.schemas import UserProfile, LocationData, InteractionData
from app.models.responses import APIResponse
from app.models.requests import (
    RecommendationRequest, 
    UserProfileRequest, 
    ProcessingRequest, 
    RefreshRequest, 
    ResultsFilterRequest,
    TaskStatusRequest
)
from app.api.dependencies import (
    get_user_profile_service,
    get_lie_service,
    get_cis_service,
    get_llm_service,
    get_results_service,
    get_celery_app,
    get_optional_user_profile_service,
    get_optional_lie_service,
    get_optional_cis_service
)
from app.services.user_profile import UserProfileService
from app.services.lie_service import LIEService
from app.services.cis_service import CISService
from app.services.llm_service import LLMService
from app.services.results_service import ResultsService
from app.workers.tasks import process_user_comprehensive
from app.utils.prompt_builder import PromptBuilder
from celery import Celery
from app.utils.serialization import safe_model_dump

router = APIRouter(prefix="/users", tags=["users"])
logger = get_logger("users_router")


# OPTIONS handlers for CORS
@router.options("/{user_id}/profile")
async def options_user_profile(user_id: str):
    """Handle OPTIONS request for user profile endpoint"""
    return Response(status_code=200)


@router.options("/{user_id}/location")
async def options_user_location(user_id: str):
    """Handle OPTIONS request for user location endpoint"""
    return Response(status_code=200)


@router.options("/{user_id}/interactions")
async def options_user_interactions(user_id: str):
    """Handle OPTIONS request for user interactions endpoint"""
    return Response(status_code=200)


@router.options("/{user_id}/recommendations")
async def options_user_recommendations(user_id: str):
    """Handle OPTIONS request for user recommendations endpoint"""
    return Response(status_code=200)


@router.options("/{user_id}/results")
async def options_user_results(user_id: str):
    """Handle OPTIONS request for user results endpoint"""
    return Response(status_code=200)


@router.get("/{user_id}/profile")
async def get_user_profile(
    user_id: str,
    user_service: UserProfileService = Depends(get_optional_user_profile_service)
):
    """
    Get comprehensive user profile with mock data
    
    - **user_id**: User identifier
    """
    try:
        log_api_call("user_profile_service", f"/{user_id}/profile", "GET", user_id=user_id)
        logger.info("Getting user profile", user_id=user_id, endpoint="get_user_profile")
        
        if not user_service:
            log_api_response("user_profile_service", f"/{user_id}/profile", False, 
                           user_id=user_id, error="service_unavailable")
            return APIResponse.service_unavailable_response(
                message="User profile service temporarily unavailable",
                service="user_profile"
            )
        
        user_profile = await user_service.get_user_profile(user_id)
        
        if not user_profile:
            log_api_response("user_profile_service", f"/{user_id}/profile", False, 
                           user_id=user_id, error="user_not_found")
            return APIResponse.error_response(
                message=f"User profile not found for user_id: {user_id}",
                status_code=404
            )
        
        log_api_response("user_profile_service", f"/{user_id}/profile", True, 
                        user_id=user_id, 
                        profile_name=user_profile.name,
                        profile_email=user_profile.email)
        logger.info("Retrieved user profile", 
                   user_id=user_id, 
                   profile_name=user_profile.name,
                   profile_email=user_profile.email,
                   endpoint="get_user_profile")
        return APIResponse.success_response(
            data=user_profile,
            message="User profile retrieved successfully"
        )
        
    except Exception as e:
        log_api_response("user_profile_service", f"/{user_id}/profile", False, 
                        user_id=user_id, error=str(e))
        logger.error("Error getting user profile", 
                   user_id=user_id, 
                   error=str(e),
                   endpoint="get_user_profile")
        log_exception("users_router", e, {"user_id": user_id, "endpoint": "get_user_profile"})
        return APIResponse.error_response(
            message="Failed to retrieve user profile",
            status_code=500,
            error={"details": str(e)}
        )





@router.get("/{user_id}/location")
async def get_user_location_data(
    user_id: str,
    lie_service: LIEService = Depends(get_optional_lie_service)
):
    """
    Get comprehensive location data for a user
    
    - **user_id**: User identifier
    """
    try:
        log_api_call("lie_service", f"/{user_id}/location", "GET", user_id=user_id)
        logger.info("Getting user location data", user_id=user_id, endpoint="get_user_location")
        
        if not lie_service:
            log_api_response("lie_service", f"/{user_id}/location", False, 
                           user_id=user_id, error="service_unavailable")
            return APIResponse.service_unavailable_response(
                message="Location service temporarily unavailable",
                service="lie"
            )
        
        location_data = await lie_service.get_location_data(user_id)
        
        if not location_data:
            log_api_response("lie_service", f"/{user_id}/location", False, 
                           user_id=user_id, error="location_data_not_found")
            return APIResponse.error_response(
                message=f"Location data not found for user_id: {user_id}",
                status_code=404
            )
        
        log_api_response("lie_service", f"/{user_id}/location", True, 
                        user_id=user_id, 
                        current_location=location_data.current_location,
                        home_location=location_data.home_location)
        logger.info("Retrieved user location data", 
                   user_id=user_id, 
                   current_location=location_data.current_location,
                   home_location=location_data.home_location,
                   endpoint="get_user_location")
        return APIResponse.success_response(
            data=location_data,
            message="Location data retrieved successfully"
        )
        
    except Exception as e:
        log_api_response("lie_service", f"/{user_id}/location", False, 
                        user_id=user_id, error=str(e))
        logger.error("Error getting user location data", 
                   user_id=user_id, 
                   error=str(e),
                   endpoint="get_user_location")
        log_exception("users_router", e, {"user_id": user_id, "endpoint": "get_user_location"})
        return APIResponse.error_response(
            message="Failed to retrieve user location data",
            status_code=500,
            error={"details": str(e)}
        )





@router.get("/{user_id}/interactions")
async def get_user_interaction_data(user_id: str):
    """
    Get comprehensive interaction data for a user
    
    - **user_id**: User identifier
    """
    try:
        log_api_call("cis_service", f"/{user_id}/interactions", "GET", user_id=user_id)
        logger.info("Getting user interaction data", user_id=user_id, endpoint="get_user_interactions")
        
        cis_service = CISService(timeout=30)
        interaction_data = await cis_service.get_interaction_data(user_id)
        
        if not interaction_data:
            log_api_response("cis_service", f"/{user_id}/interactions", False, 
                           user_id=user_id, error="interaction_data_not_found")
            raise HTTPException(
                status_code=404,
                detail=f"Interaction data not found for user_id: {user_id}"
            )
        
        log_api_response("cis_service", f"/{user_id}/interactions", True, 
                        user_id=user_id, 
                        engagement_score=interaction_data.engagement_score,
                        recent_interactions_count=len(interaction_data.recent_interactions))
        logger.info("Retrieved user interaction data", 
                   user_id=user_id, 
                   engagement_score=interaction_data.engagement_score,
                   recent_interactions_count=len(interaction_data.recent_interactions),
                   endpoint="get_user_interactions")
        return interaction_data
        
    except HTTPException:
        raise
    except Exception as e:
        log_api_response("cis_service", f"/{user_id}/interactions", False, 
                        user_id=user_id, error=str(e))
        logger.error("Error getting user interaction data", 
                   user_id=user_id, 
                   error=str(e),
                   endpoint="get_user_interactions")
        log_exception("users_router", e, {"user_id": user_id, "endpoint": "get_user_interactions"})
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve user interaction data"
        )





@router.post("/{user_id}/process-comprehensive")
async def process_user_comprehensive_endpoint(user_id: str, priority: int = 5):
    """
    Enqueue a user for comprehensive processing via RabbitMQ
    
    - **user_id**: User identifier to process
    - **priority**: Processing priority (1-10, default: 5)
    """
    try:
        log_api_call("celery_queue", f"/{user_id}/process-comprehensive", "POST", 
                    user_id=user_id, priority=priority)
        logger.info("Enqueueing user for comprehensive processing", 
                   user_id=user_id, 
                   priority=priority,
                   endpoint="process_user_comprehensive")
        
        # Enqueue the user for comprehensive processing directly
        result = process_user_comprehensive.apply_async(
            args=[user_id],
            queue="user_processing",
            routing_key=f"user_processing_{hash(user_id) % 10}",  # Distribute across workers
            priority=priority,
            expires=None,
            retry=True,
            retry_policy={
                'max_retries': 3,
                'interval_start': 0,
                'interval_step': 0.2,
                'interval_max': 0.2,
            }
        )
        
        log_api_response("celery_queue", f"/{user_id}/process-comprehensive", True, 
                        user_id=user_id, 
                        task_id=result.id,
                        priority=priority)
        logger.info("User enqueued for comprehensive processing", 
                   user_id=user_id, 
                   task_id=result.id,
                   priority=priority,
                   endpoint="process_user_comprehensive")
        
        return {
            "success": True,
            "user_id": user_id,
            "priority": priority,
            "task_id": result.id,
            "message": f"User {user_id} enqueued for comprehensive processing",
            "status": "queued",
            "queue": "user_processing"
        }
        
    except Exception as e:
        log_api_response("celery_queue", f"/{user_id}/process-comprehensive", False, 
                        user_id=user_id, error=str(e))
        logger.error("Error enqueueing user for comprehensive processing", 
                   user_id=user_id, 
                   error=str(e),
                   endpoint="process_user_comprehensive")
        log_exception("users_router", e, {"user_id": user_id, "endpoint": "process_user_comprehensive"})
        raise HTTPException(
            status_code=500,
            detail=f"Failed to enqueue user {user_id} for comprehensive processing"
        )


@router.post("/{user_id}/process-comprehensive-direct")
async def process_user_comprehensive_direct_endpoint(user_id: str):
    """
    Process a user comprehensively (direct execution, not via queue)
    
    - **user_id**: User identifier to process
    """
    try:
        logger.info("Processing user comprehensively (direct)", user_id=user_id)
        
        # Process the user directly (synchronous)
        result = process_user_comprehensive.delay(user_id)
        
        # Wait for the result
        comprehensive_data = result.get(timeout=30)  # 30 second timeout
        
        if not comprehensive_data.get("success"):
            raise HTTPException(
                status_code=500,
                detail=f"Comprehensive processing failed: {comprehensive_data.get('error', 'Unknown error')}"
            )
        
        logger.info("User processed comprehensively", user_id=user_id, task_id=result.id)
        
        return {
            "success": True,
            "user_id": user_id,
            "task_id": result.id,
            "comprehensive_data": comprehensive_data.get("comprehensive_data"),
            "message": f"User {user_id} processed comprehensively",
            "status": "completed"
        }
        
    except Exception as e:
        logger.error("Error processing user comprehensively", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process user {user_id} comprehensively"
        )




@router.get("/{user_id}/processing-status/{task_id}")
async def get_processing_status(
    user_id: str,
    task_id: str,
    celery_app: Celery = Depends(get_celery_app)
):
    """Get processing status for a task."""
    try:
        # Add validation to prevent recursion
        task_result = celery_app.AsyncResult(task_id)
        
        # Convert to simple dict to avoid recursion
        result_data = {
            "status": task_result.status,
            "successful": task_result.successful(),
            "failed": task_result.failed(),
            "date_done": task_result.date_done.isoformat() if task_result.date_done else None
        }
        
        # Handle result safely
        if hasattr(task_result, 'result') and task_result.result:
            if isinstance(task_result.result, dict):
                result_data["result"] = task_result.result
            else:
                # Convert to dict if it's a model
                result_data["result"] = safe_model_dump(task_result.result)
        
        return {
            "success": True,
            "data": result_data,
            "user_id": user_id,
            "task_id": task_id
        }
        
    except Exception as e:
        logger.error(f"Error getting processing status: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "user_id": user_id,
            "task_id": task_id
        }


@router.post("/{user_id}/generate-recommendations")
async def generate_recommendations_endpoint(
    user_id: str, 
    request: RecommendationRequest,
    llm_service: LLMService = Depends(get_llm_service)
):
    """Generate recommendations using LLM service"""
    try:
        logger.info(f"Generating recommendations for user {user_id}")
        
        prompt = request.prompt
        builder = PromptBuilder()
        if prompt is not None:
            # Wrap custom prompt to maintain strict JSON structure
            prompt = builder.build_custom_prompt(prompt, current_city="Barcelona", max_results=5)
        else:
            # Build the original prompt using live data; if any data missing, create minimal stand-ins
            user_service = get_optional_user_profile_service()
            lie_service = get_optional_lie_service()
            cis_service = get_optional_cis_service()
            user_profile = await user_service.get_user_profile(user_id) if user_service else None
            location_data = await lie_service.get_location_data(user_id) if lie_service else None
            interaction_data = await cis_service.get_interaction_data(user_id) if cis_service else None
            # Minimal stand-ins when any component is missing
            from app.models.schemas import UserProfile, LocationData, InteractionData
            if user_profile is None:
                user_profile = UserProfile(
                    user_id=user_id,
                    name=f"User-{user_id}",
                    email=f"user{user_id}@example.com",
                    preferences={},
                    interests=[],
                    age=30,
                    location="Barcelona"
                )
            if location_data is None:
                location_data = LocationData(
                    user_id=user_id,
                    current_location="Barcelona",
                    home_location="Barcelona",
                    work_location="Barcelona",
                    travel_history=[],
                    location_preferences={}
                )
            if interaction_data is None:
                interaction_data = InteractionData(
                    user_id=user_id,
                    recent_interactions=[],
                    interaction_history=[],
                    preferences={},
                    engagement_score=0.5
                )
            from app.core.constants import RecommendationType
            if not any([user_profile, location_data, interaction_data]):
                prompt = builder.build_fallback_prompt(
                    user_profile=user_profile,
                    location_data=location_data,
                    interaction_data=interaction_data,
                    recommendation_type=RecommendationType.PLACE,
                    max_results=5,
                )
            else:
                prompt = builder.build_recommendation_prompt(
                    user_profile=user_profile,
                    location_data=location_data,
                    interaction_data=interaction_data,
                    recommendation_type=RecommendationType.PLACE,
                    max_results=5,
                )
        logger.info("Generated prompt for user", user_id=user_id, prompt_length=len(prompt) if prompt else 0)
        response = await llm_service.generate_recommendations(prompt, user_id)
        
        if response.get("success", False):
            return APIResponse.success_response(
                data=response,
                message="Recommendations generated successfully"
            )
        else:
            return APIResponse.error_response(
                message="Failed to generate recommendations",
                data=response,
                status_code=500
            )
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return APIResponse.error_response(
            message="Error generating recommendations",
            status_code=500,
            error={"details": str(e)}
        )


@router.delete("/{user_id}/recommendations")
async def clear_user_recommendations(
    user_id: str,
    llm_service: LLMService = Depends(get_llm_service)
):
    """Clear stored recommendations for a user"""
    try:
        logger.info(f"Clearing recommendations for user {user_id}")
        
        llm_service.clear_recommendations(user_id)
        
        return APIResponse.success_response(
            data=None,
            message="Recommendations cleared successfully"
        )
        
    except Exception as e:
        logger.error(f"Error clearing recommendations: {str(e)}")
        return APIResponse.error_response(
            message="Error clearing recommendations",
            status_code=500,
            error={"details": str(e)}
        )


@router.post("/generate-recommendations")
async def generate_recommendations_direct(
    request: RecommendationRequest,
    llm_service: LLMService = Depends(get_llm_service)
):
    """Generate recommendations without storing (for testing)"""
    try:
        logger.info("Generating recommendations without storing")
        
        prompt = request.prompt
        builder = PromptBuilder()
        if prompt is not None:
            prompt = builder.build_custom_prompt(prompt, current_city="Barcelona", max_results=5)
        else:
            # Build the original prompt with minimal anonymous context
            from app.models.schemas import UserProfile, LocationData, InteractionData
            anon_profile = UserProfile(
                user_id="anon",
                name="Friend",
                email="friend@example.com",
                preferences={},
                interests=[],
                age=30,
                location="Barcelona"
            )
            anon_location = LocationData(
                user_id="anon",
                current_location="Barcelona",
                home_location="Barcelona",
                work_location="Barcelona",
                travel_history=[],
                location_preferences={}
            )
            anon_interactions = InteractionData(
                user_id="anon",
                recent_interactions=[],
                interaction_history=[],
                preferences={},
                engagement_score=0.5
            )
                
            from app.core.constants import RecommendationType
            if not any([anon_profile, anon_location, anon_interactions]):
                prompt = builder.build_fallback_prompt(
                    user_profile=anon_profile,
                    location_data=anon_location,
                    interaction_data=anon_interactions,
                    recommendation_type=RecommendationType.PLACE,
                    max_results=5,
                )
            else:
                prompt = builder.build_recommendation_prompt(
                    user_profile=anon_profile,
                    location_data=anon_location,
                    interaction_data=anon_interactions,
                    recommendation_type=RecommendationType.PLACE,
                    max_results=5,
                )
        response = await llm_service.generate_recommendations(prompt)
        
        if response.get("success", False):
            return APIResponse.success_response(
                data=response,
                message="Recommendations generated successfully"
            )
        else:
            return APIResponse.error_response(
                message="Failed to generate recommendations",
                data=response,
                status_code=500
            )
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return APIResponse.error_response(
            message="Error generating recommendations",
            status_code=500,
            error={"details": str(e)}
        )


@router.get("/{user_id}/results")
async def get_ranked_results(
    user_id: str,
    category: Optional[str] = Query(None, description="Filter by category (movies, music, places, events)"),
    limit: Optional[int] = Query(5, description="Limit results per category"),
    min_score: Optional[float] = Query(0.0, description="Minimum ranking score"),
    results_service: ResultsService = Depends(get_results_service)
):
    """
    Get ranked and filtered final results for a user
    
    - **user_id**: User identifier
    - **category**: Optional category filter
    - **limit**: Maximum results per category (default: 5)
    - **min_score**: Minimum ranking score (default: 0.0)
    """
    try:
        logger.info(f"Getting ranked results for user {user_id}")
        
        # Prepare filters
        filters = {
            "category": category,
            "limit": limit,
            "min_score": min_score
        }
        
        # Get ranked results
        results = results_service.get_ranked_results(user_id, filters)
        
        if results.get("success"):
            return APIResponse.success_response(
                data=results,
                message="Ranked results retrieved successfully"
            )
        else:
            return APIResponse.error_response(
                message=results.get("error", "No results found"),
                status_code=404
            )
        
    except Exception as e:
        logger.error(f"Error getting ranked results: {str(e)}")
        return APIResponse.error_response(
            message="Error getting ranked results",
            status_code=500,
            error={"details": str(e)}
        )
