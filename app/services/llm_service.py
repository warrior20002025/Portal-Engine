import json
import time
import random
import httpx
from typing import Dict, Any, List, Optional
from app.core.logging import get_logger, log_api_call, log_api_response, log_exception
from app.core.config import settings
import redis
from datetime import datetime, timezone
import math

logger = get_logger("llm_service")


class LLMService:
    """Service to generate recommendations from prompts and store in Redis"""
    
    def __init__(self, timeout: int = 120):
        self.timeout = timeout
        logger.info("Initializing LLM service",
                   timeout=timeout,
                   redis_host=settings.redis_host,
                   redis_port=6379,
                   redis_db=1)
        
        try:
            self.redis_client = redis.Redis(
                host=settings.redis_host,
                port=6379,
                db=1,  # Use different DB for recommendations
                decode_responses=True
            )
            # Test Redis connection
            self.redis_client.ping()
            logger.info("Redis connection established successfully",
                       service="llm_service",
                       redis_host=settings.redis_host)
        except Exception as e:
            logger.error("Failed to connect to Redis",
                        service="llm_service",
                        redis_host=settings.redis_host,
                        error=str(e))
            raise
        # Action weights for ranking score calculation
        self.ACTION_WEIGHTS = {
            "liked": 2.0,
            "saved": 1.5,
            "shared": 1.2,
            "clicked": 0.8,
            "view": 0.4,
            "ignored": -1.0,
            "disliked": -1.5,
        }
        self.BASE_SCORE = 0.5
        self.SCALE = 0.2  # scale for converting weighted sum into [0,1] range
        self.HALF_LIFE_DAYS = 30  # recency half-life for interactions
    
    def _normalize_key(self, value: str) -> str:
        """Normalize string keys for comparison"""
        if not isinstance(value, str):
            return ""
        return ''.join(ch.lower() for ch in value if ch.isalnum() or ch.isspace()).strip()

    def _tokenize(self, value: str) -> List[str]:
        """Tokenize normalized value into words for partial matching."""
        norm = self._normalize_key(value)
        return [t for t in norm.split() if t]

    def _recency_weight(self, iso_timestamp: str) -> float:
        """Compute exponential recency weight in (0,1], newer -> closer to 1."""
        try:
            if iso_timestamp and isinstance(iso_timestamp, str):
                ts = iso_timestamp.replace("Z", "+00:00")
                dt = datetime.fromisoformat(ts)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                now = datetime.now(timezone.utc)
                age_days = max(0.0, (now - dt).total_seconds() / 86400.0)
                lam = math.log(2) / max(1.0, float(self.HALF_LIFE_DAYS))
                weight = math.exp(-lam * age_days)
                return max(0.1, min(1.0, weight))
        except Exception:
            pass
        return 0.5

    def _category_prior(self, item: Dict[str, Any], category: str) -> float:
        """Small prior boost based on intrinsic item quality/popularity."""
        try:
            if category in ["movies", "music"]:
                # rating often string; monthly_listeners may exist for music
                rating_raw = item.get("rating")
                rating = float(rating_raw) if isinstance(rating_raw, (int, float, str)) and str(rating_raw).replace('.', '', 1).isdigit() else None
                listeners_raw = item.get("monthly_listeners")
                listeners = float(str(listeners_raw).replace('M', '').replace(',', '')) if listeners_raw and isinstance(listeners_raw, (int, float, str)) and str(listeners_raw) else None
                prior = 0.0
                if rating is not None:
                    prior += (max(0.0, min(10.0, rating)) - 5.0) / 50.0  # up to ±0.1
                if listeners is not None:
                    prior += min(0.1, listeners / 1e7)  # cap at 0.1
                return prior
            if category in ["places", "events"]:
                rating_raw = item.get("rating")
                rating = float(rating_raw) if isinstance(rating_raw, (int, float, str)) and str(rating_raw).replace('.', '', 1).isdigit() else None
                total_raw = item.get("user_ratings_total")
                total = float(total_raw) if isinstance(total_raw, (int, float)) else None
                prior = 0.0
                if rating is not None:
                    prior += (max(0.0, min(5.0, rating)) - 3.0) / 20.0  # up to ±0.1
                if total is not None:
                    prior += min(0.1, total / 5000.0)
                return prior
        except Exception:
            return 0.0
        return 0.0

    def _setup_demo_data(self) -> None:
        """Setup demo data (placeholder for test compatibility)"""
        logger.info("Setting up demo data")
        return None

    def _generate_demo_recommendations(self, prompt: str) -> Dict[str, List[Dict]]:
        """Generate demo recommendations for testing purposes"""
        logger.info(f"Generating demo recommendations for prompt: {prompt}")
        return {
            "movies": [
                {"title": "Demo Movie 1", "genre": "Action", "year": "2023"},
                {"title": "Demo Movie 2", "genre": "Drama", "year": "2022"},
            ],
            "music": [
                {"title": "Demo Song 1", "artist": "Artist 1", "genre": "Pop"},
                {"title": "Demo Song 2", "artist": "Artist 2", "genre": "Rock"},
            ],
            "places": [
                {"name": "Demo Place 1", "type": "Park", "rating": 4.5},
                {"name": "Demo Place 2", "type": "Museum", "rating": 4.0},
            ],
            "events": [
                {"name": "Demo Event 1", "date": "2025-01-01", "venue": "Venue 1"},
                {"name": "Demo Event 2", "date": "2025-02-01", "venue": "Venue 2"},
            ]
        }

    def _get_user_interaction_history(self, user_id: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Mock retrieval of user's historical interactions from a table.
        Returns interaction data with proper field matching for each category.
        """
        seed = sum(ord(c) for c in user_id) if user_id else 0
        rnd = random.Random(seed)
        
        interaction_types = ["view", "liked", "saved", "shared", "clicked", "ignored", "disliked"]
        
        history = {
            "movies": [
                {"title": "Inception", "genre": "Science Fiction/Action", "action": "liked", "timestamp": "2024-08-15T10:30:00Z"},
                {"title": "The Dark Knight", "genre": "Action/Crime", "action": "view", "timestamp": "2024-08-14T15:20:00Z"},
                {"title": "Vicky Cristina Barcelona", "genre": "Drama/Romance", "action": "ignored", "timestamp": "2024-08-13T09:45:00Z"},
                {"title": "The Shawshank Redemption", "genre": "Drama", "action": rnd.choice(["liked", "view", "saved"]), "timestamp": "2024-08-12T20:10:00Z"},
                {"title": "All About My Mother", "genre": "Drama", "action": rnd.choice(["view", "ignored"]), "timestamp": "2024-08-11T14:30:00Z"},
            ],
            "music": [
                {"title": "Blinding Lights", "genre": "Pop/Electronic", "action": "liked", "timestamp": "2024-08-16T12:00:00Z"},
                {"title": "Barcelona", "genre": "Pop", "action": "saved", "timestamp": "2024-08-15T18:45:00Z"},
                {"title": "Shape of You", "genre": "Pop", "action": rnd.choice(["view", "liked", "shared"]), "timestamp": "2024-08-14T11:20:00Z"},
                {"title": "Mediterráneo", "genre": "Folk", "action": rnd.choice(["ignored", "view"]), "timestamp": "2024-08-13T16:30:00Z"},
                {"title": "Dance Monkey", "genre": "Pop", "action": rnd.choice(["view", "disliked"]), "timestamp": "2024-08-12T13:15:00Z"},
            ],
            "places": [
                {"name": "Sagrada Família", "type": "attraction", "action": "liked", "timestamp": "2024-08-17T09:00:00Z"},
                {"name": "Eiffel Tower", "type": "attraction", "action": "view", "timestamp": "2024-08-16T14:30:00Z"},
                {"name": "Park Güell", "type": "park", "action": rnd.choice(["liked", "saved"]), "timestamp": "2024-08-15T11:45:00Z"},
                {"name": "Big Ben", "type": "attraction", "action": rnd.choice(["view", "ignored"]), "timestamp": "2024-08-14T17:20:00Z"},
                {"name": "Colosseum", "type": "attraction", "action": rnd.choice(["view", "clicked"]), "timestamp": "2024-08-13T10:10:00Z"},
            ],
            "events": [
                {"name": "Primavera Sound", "category": "music", "action": "liked", "timestamp": "2024-08-18T08:30:00Z"},
                {"name": "Coachella Valley Music and Arts Festival", "category": "music", "action": "ignored", "timestamp": "2024-08-17T19:15:00Z"},
                {"name": "La Mercè Festival", "category": "festival", "action": rnd.choice(["liked", "saved"]), "timestamp": "2024-08-16T15:45:00Z"},
                {"name": "Oktoberfest", "category": "festival", "action": rnd.choice(["view", "clicked"]), "timestamp": "2024-08-15T12:30:00Z"},
                {"name": "Mardi Gras", "category": "festival", "action": rnd.choice(["view", "ignored"]), "timestamp": "2024-08-14T16:00:00Z"},
            ],
        }
        
        genres_list = {
            "movies": ["Action", "Drama", "Comedy", "Science Fiction", "Adventure"],
            "music": ["Pop", "Electronic", "R&B", "Singer-Songwriter", "Folk"],
            "places": ["attraction", "restaurant", "park", "museum", "trail"],
            "events": ["music", "sports", "art", "festival"]
        }
        
        for category in history:
            if rnd.random() > 0.3:
                extra_interactions = rnd.randint(1, 3)
                for _ in range(extra_interactions):
                    action = rnd.choice(interaction_types)
                    timestamp = f"2024-08-{rnd.randint(10, 18):02d}T{rnd.randint(8, 20):02d}:{rnd.randint(0, 59):02d}:00Z"
                    genre_field = "genre" if category in ["movies", "music"] else "type" if category == "places" else "category"
                    item_name = "title" if category in ["movies", "music"] else "name"
                    item_value = f"Random {category.capitalize()} {rnd.randint(1, 100)}"
                    genre_value = rnd.choice(genres_list[category])
                    history[category].append({item_name: item_value, genre_field: genre_value, "action": action, "timestamp": timestamp})
        
        return history

    def _compute_ranking_score(self, item: Dict[str, Any], category: str, history: Dict[str, List[Dict[str, str]]], 
                               user_profile: Dict[str, Any] = None, location_data: Dict[str, Any] = None, 
                               interaction_data: Dict[str, Any] = None) -> float:
        """
        Compute raw ranking score using multiple factors with expanded ranges for more variation:
        - Base: 0.2
        - Quality: 0.0-0.3 (ratings, popularity, box office/chart/capacity)
        - Profile: 0.0-0.25 (age, interest/keyword matches)
        - Location: 0.0-0.2 (overlap, distance)
        - Interaction: -0.2-0.3 (actions with recency and amplification)
        - Recency: -0.2-0.2 (item age/date, penalty for past events)
        """
        base_score = 0.2

        # Quality boost with more factors
        quality_boost = 0.0
        rating = item.get('rating')
        if rating:
            try:
                rating_val = float(str(rating).replace('/10', '').replace('/5', ''))
                if category in ["places", "events"]:
                    if rating_val >= 4.5:
                        quality_boost += 0.25
                    elif rating_val >= 4.0:
                        quality_boost += 0.18
                    elif rating_val >= 3.5:
                        quality_boost += 0.12
                    elif rating_val >= 3.0:
                        quality_boost += 0.06
                else:
                    if rating_val >= 8.5:
                        quality_boost += 0.25
                    elif rating_val >= 7.5:
                        quality_boost += 0.18
                    elif rating_val >= 6.5:
                        quality_boost += 0.12
                    elif rating_val >= 5.0:
                        quality_boost += 0.06
            except:
                pass
        
        if category == "movies":
            box = item.get('box_office', '')
            if box:
                try:
                    num = float(box.strip('$').rstrip('M').strip())
                    if num > 300:
                        quality_boost += 0.15
                    elif num > 200:
                        quality_boost += 0.1
                    elif num > 100:
                        quality_boost += 0.05
                except:
                    pass
        elif category == "music":
            listeners = item.get('monthly_listeners', '')
            try:
                num_listeners = float(str(listeners).rstrip('M').strip()) if 'M' in str(listeners) else 0
                quality_boost += min(0.15, num_listeners / 500)
            except:
                pass
            chart = item.get('chart_position', '')
            if chart:
                if '#1' in chart:
                    quality_boost += 0.2
                else:
                    try:
                        pos = int(chart.strip('#').split()[0])
                        if pos <= 5:
                            quality_boost += 0.15
                        elif pos <= 10:
                            quality_boost += 0.1
                        elif pos <= 20:
                            quality_boost += 0.05
                    except:
                        pass
        elif category in ["places", "events"]:
            total_ratings = item.get('user_ratings_total', 0)
            if total_ratings > 5000:
                quality_boost += 0.15
            elif total_ratings > 1000:
                quality_boost += 0.1
            elif total_ratings > 100:
                quality_boost += 0.05
            if category == "events":
                cap = item.get('capacity', 0)
                if cap > 20000:
                    quality_boost += 0.15
                elif cap > 5000:
                    quality_boost += 0.1
                elif cap > 1000:
                    quality_boost += 0.05
                price_min = item.get('price_min', float('inf'))
                if price_min == 0:
                    quality_boost += 0.1
                elif price_min < 20:
                    quality_boost += 0.07
                elif price_min < 50:
                    quality_boost += 0.03
        quality_boost = min(0.3, quality_boost)

        # Profile boost with keyword matching
        profile_boost = 0.0
        if user_profile:
            user_age = user_profile.get('age')
            if user_age and category in ["movies", "events"]:
                if category == "movies":
                    age_rating = item.get('age_rating', '')
                    if ('R' in age_rating and user_age >= 17) or ('PG-13' in age_rating and user_age >= 13) or 'PG' in age_rating:
                        profile_boost += 0.08
                elif category == "events":
                    age_restriction = item.get('age_restriction', '')
                    if 'All ages' in age_restriction or ('18+' in age_restriction and user_age >= 18):
                        profile_boost += 0.08
            interests = user_profile.get('interests', [])
            if interests:
                item_text = f"{item.get('title', '')} {item.get('name', '')} {item.get('genre', '')} {item.get('description', '')} { ' '.join(item.get('keywords', [])) }".lower()
                match_count = sum(1 for interest in interests if interest.lower() in item_text)
                keyword_match = sum(1 for kw in item.get('keywords', []) if any(i.lower() in kw.lower() for i in interests))
                profile_boost += min(0.15, (match_count + keyword_match) * 0.05)
            interests_lower = [i.lower() for i in interests]
            if 'sociable' in interests_lower:
                if category == "music":
                    mood_lower = item.get('mood', '').lower()
                    if 'upbeat' in mood_lower:
                        profile_boost += 0.1
                    elif 'melancholic' in mood_lower:
                        profile_boost -= 0.1
                if category == "movies":
                    genre_lower = item.get('genre', '').lower()
                    if 'comedy' in genre_lower or 'adventure' in genre_lower:
                        profile_boost += 0.05
                if category == "places":
                    if item.get('outdoor_seating', False) or item.get('wifi_available', False):
                        profile_boost += 0.05
                if category == "events":
                    cat_lower = item.get('category', '').lower()
                    if 'music' in cat_lower or 'festival' in cat_lower:
                        profile_boost += 0.1
                    elif 'art' in cat_lower:
                        profile_boost += 0.08
                    elif 'sports' in cat_lower:
                        profile_boost += 0.05
            if 'science-enthusiast' in interests_lower:
                if category == "movies" or category == "music":
                    genre_lower = item.get('genre', '').lower()
                    if 'science fiction' in genre_lower or 'electronic' in genre_lower:
                        profile_boost += 0.1
        profile_boost = max(-0.2, min(0.25, profile_boost))

        # Location boost with overlap score and distance
        location_boost = 0.0
        if location_data and category in ["places", "events"]:
            current_location = location_data.get('current_location', '').lower()
            if current_location:
                item_location = ""
                if category == "places":
                    item_location = f"{item.get('vicinity', '')} {item.get('query', '')}".lower()
                elif category == "events":
                    item_location = f"{item.get('address', '')} {item.get('venue', '')}".lower()
                if current_location in item_location:
                    location_boost = 0.2
                else:
                    current_words = set(current_location.split())
                    item_words = set(item_location.split())
                    overlap = len(current_words & item_words) / len(current_words) if current_words else 0
                    location_boost = overlap * 0.15
                distance = item.get('distance_from_user', float('inf'))
                if distance < 10:
                    location_boost += 0.1
                elif distance < 20:
                    location_boost += 0.05
                elif distance < 50:
                    location_boost += 0.02
            location_boost = min(0.2, location_boost)

        # Interaction boost with recency and amplification
        interaction_boost = 0.0
        similarity_boost = 0.0
        field_mapping = {"movies": "title", "music": "title", "places": "name", "events": "name"}
        genre_field = {"movies": "genre", "music": "genre", "places": "type", "events": "category"}[category]
        field_name = field_mapping[category]
        item_identifier = self._normalize_key(item.get(field_name, ""))
        item_genres = item.get(genre_field, '').lower().split('/')
        if item_identifier:
            total_weight = 0.0
            count = 0
            for inter in history.get(category, []):
                hist_id = self._normalize_key(inter.get(field_name, ""))
                if hist_id == item_identifier:
                    action = inter.get("action", "view").lower()
                    weight = {
                        'liked': 0.2, 'saved': 0.15, 'shared': 0.12, 'clicked': 0.08,
                        'view': 0.04, 'ignored': -0.1, 'disliked': -0.2
                    }.get(action, 0.0)
                    recency = self._recency_weight(inter.get('timestamp', ''))
                    total_weight += weight * recency
                    count += 1
                # Similarity
                hist_genres = inter.get(genre_field, '').lower().split('/')
                overlap = len(set(item_genres) & set(hist_genres)) / max(1, len(item_genres)) if item_genres else 0
                recency = self._recency_weight(inter.get('timestamp', ''))
                if inter.get("action", "").lower() in ['liked', 'saved', 'shared', 'clicked']:
                    similarity_boost += 0.1 * overlap * recency
                elif inter.get("action", "").lower() in ['ignored', 'disliked']:
                    similarity_boost -= 0.05 * overlap * recency
            if count > 0:
                interaction_boost = (total_weight / count) * (1 + 0.5 * (count - 1))
            similarity_boost = max(-0.1, min(0.15, similarity_boost))
            interaction_boost += similarity_boost
        interaction_boost = max(-0.2, min(0.3, interaction_boost))

        # Recency boost with penalty for old/past
        recency_boost = 0.0
        penalty = 0.0
        now = datetime.now(timezone.utc)
        if category == "movies":
            year = item.get('year')
            if year:
                try:
                    y = int(year)
                    age = now.year - y
                    if age <= 1:
                        recency_boost = 0.2
                    elif age <= 3:
                        recency_boost = 0.15
                    elif age <= 5:
                        recency_boost = 0.1
                    if age > 10:
                        penalty = -0.05
                    if age > 20:
                        penalty = -0.1
                except:
                    pass
        elif category == "music":
            year = item.get('release_year')
            if year:
                try:
                    y = int(year)
                    age = now.year - y
                    if age <= 1:
                        recency_boost = 0.2
                    elif age <= 2:
                        recency_boost = 0.15
                    elif age <= 4:
                        recency_boost = 0.1
                    if age > 6:
                        penalty = -0.05
                except:
                    pass
        elif category == "events":
            date_str = item.get('date')
            if date_str:
                try:
                    event_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    if event_date.tzinfo is None:
                        event_date = event_date.replace(tzinfo=timezone.utc)
                    days_diff = (event_date - now).days
                    if days_diff > 90:
                        recency_boost = 0.05
                    elif days_diff > 30:
                        recency_boost = 0.1
                    elif days_diff > 7:
                        recency_boost = 0.15
                    elif days_diff >= 0:
                        recency_boost = 0.2
                    else:
                        penalty = min(-0.2, -0.05 * abs(days_diff) / 30)
                except:
                    pass
        recency_boost += penalty
        recency_boost = max(-0.2, min(0.2, recency_boost))

        raw_score = base_score + quality_boost + profile_boost + location_boost + interaction_boost + recency_boost
        return max(0.0, min(1.5, raw_score))  # Allow up to 1.5 for normalization buffer

    async def generate_recommendations(self, prompt: str, user_id: str = None, current_city: str = "Barcelona") -> Dict[str, Any]:
        """
        Generate recommendations based on prompt and store in Redis
        """
        try:
            logger.info(f"Generating recommendations for prompt: {prompt[:100]}...")
            
            start_time = time.time()
            
            recommendations = await self._call_llm_api(prompt, user_id, current_city)
            
            processing_time = time.time() - start_time
            
            response = {
                "success": True,
                "prompt": prompt,
                "user_id": user_id,
                "current_city": current_city,
                "generated_at": time.time(),
                "processing_time": processing_time,
                "recommendations": recommendations,
                "metadata": {
                    "total_recommendations": sum(len(cat) for cat in recommendations.values()) if recommendations else 0,
                    "categories": list(recommendations.keys()) if recommendations else [],
                    "model": "llm-api-v1.0",
                    "ranking_enabled": user_id is not None
                }
            }
            
            if user_id:
                self._store_in_redis(user_id, response)
            
            logger.info(f"Generated {response['metadata']['total_recommendations']} recommendations for user {user_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "prompt": prompt,
                "user_id": user_id
            }
    
    async def _call_llm_api(self, prompt: str, user_id: str = None, current_city: str = "Barcelona") -> Dict[str, List[Dict]]:
        """
        Call the actual LLM API to generate recommendations
        """
        start_time = time.time()
        
        # Log API call initiation
        log_api_call(
            service_name="llm_api",
            endpoint="/process-text",
            method="POST",
            user_id=user_id,
            prompt_length=len(prompt),
            provider=settings.recommendation_api_provider
        )
        
        try:
            payload = {
                "text": prompt,
                "provider": settings.recommendation_api_provider
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{settings.recommendation_api_url}/process-text",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                
                result = response.json()
                raw_result = result.get("result")
                
                response_time = time.time() - start_time
                
                if raw_result is None:
                    logger.warning("Empty result field from LLM API",
                                 user_id=user_id,
                                 response_time_ms=response_time * 1000)
                    log_api_response("llm_api", "/process-text", False, 
                                   status_code=response.status_code, 
                                   response_time=response_time,
                                   user_id=user_id,
                                   error="empty_result")
                    return self._get_fallback_recommendations()
                
                if isinstance(raw_result, dict):
                    log_api_response("llm_api", "/process-text", True,
                                   status_code=response.status_code,
                                   response_time=response_time,
                                   user_id=user_id)
                    return self._process_llm_recommendations(raw_result, user_id, current_city)
                
                if isinstance(raw_result, str):
                    parsed = self._robust_parse_json(raw_result)
                    if parsed is not None:
                        log_api_response("llm_api", "/process-text", True,
                                       status_code=response.status_code,
                                       response_time=response_time,
                                       user_id=user_id)
                        return self._process_llm_recommendations(parsed, user_id, current_city)
                    
                    logger.info("LLM response is not valid JSON, attempting text parsing",
                               user_id=user_id,
                               response_time_ms=response_time * 1000)
                    recommendations = self._parse_text_response(raw_result)
                    log_api_response("llm_api", "/process-text", True,
                                   status_code=response.status_code,
                                   response_time=response_time,
                                   user_id=user_id,
                                   parsing_method="text")
                    return self._process_llm_recommendations(recommendations, user_id, current_city)
                
                logger.error("Unexpected type for LLM result",
                           user_id=user_id,
                           result_type=type(raw_result).__name__,
                           response_time_ms=response_time * 1000)
                log_api_response("llm_api", "/process-text", False,
                               status_code=response.status_code,
                               response_time=response_time,
                               user_id=user_id,
                               error="unexpected_result_type")
                return self._get_fallback_recommendations()
                    
        except httpx.TimeoutException as e:
            response_time = time.time() - start_time
            logger.error("Timeout calling LLM API",
                        user_id=user_id,
                        timeout_seconds=self.timeout,
                        response_time_ms=response_time * 1000)
            log_api_response("llm_api", "/process-text", False,
                           response_time=response_time,
                           user_id=user_id,
                           error="timeout")
            return self._get_fallback_recommendations()
        except httpx.HTTPStatusError as e:
            response_time = time.time() - start_time
            logger.error("HTTP error calling LLM API",
                        user_id=user_id,
                        status_code=e.response.status_code,
                        response_text=e.response.text,
                        response_time_ms=response_time * 1000)
            log_api_response("llm_api", "/process-text", False,
                           status_code=e.response.status_code,
                           response_time=response_time,
                           user_id=user_id,
                           error="http_error")
            return self._get_fallback_recommendations()
        except Exception as e:
            response_time = time.time() - start_time
            logger.error("Unexpected error calling LLM API",
                        user_id=user_id,
                        error=str(e),
                        response_time_ms=response_time * 1000)
            log_exception("llm_service", e, {"user_id": user_id, "response_time": response_time})
            log_api_response("llm_api", "/process-text", False,
                           response_time=response_time,
                           user_id=user_id,
                           error="unexpected_error")
            return self._get_fallback_recommendations()
    
    def _parse_text_response(self, response_text: str) -> Dict[str, List[Dict]]:
        """Parse text response from LLM and convert to structured format"""
        try:
            recommendations = {
                "movies": [],
                "music": [],
                "places": [],
                "events": []
            }
            return recommendations
        except Exception as e:
            logger.error(f"Error parsing text response: {str(e)}")
            return self._get_fallback_recommendations()
    
    def _extract_items_from_text(self, text: str, category: str) -> List[Dict]:
        """
        Extract individual items from text section
        """
        items = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
                
            item_text = line.split('.', 1)[1].strip() if '.' in line else line
            title = self._extract_title(item_text)
            
            if title:
                item = {
                    "title" if category in ["movies", "music"] else "name": title,
                    "description": item_text,
                    "category": category
                }
                
                if category == "movies":
                    item.update({
                        "year": "Unknown",
                        "genre": "Unknown",
                        "rating": "Unknown"
                    })
                elif category == "music":
                    item.update({
                        "artist": "Unknown",
                        "genre": "Unknown",
                        "release_year": "Unknown"
                    })
                elif category == "places":
                    item.update({
                        "type": "Unknown",
                        "rating": 4.0,
                        "location": {"lat": 0, "lng": 0}
                    })
                elif category == "events":
                    item.update({
                        "date": "Unknown",
                        "venue": "Unknown",
                        "price": "Unknown"
                    })
                
                items.append(item)
        
        return items
    
    def _extract_title(self, text: str) -> str:
        """
        Extract title/name from item text
        """
        import re
        bold_match = re.search(r'\*\*(.*?)\*\*', text)
        if bold_match:
            return bold_match.group(1).strip()
        
        quote_match = re.search(r'"([^"]*)"', text)
        if quote_match:
            return quote_match.group(1).strip()
        
        title = text.split('(')[0].split(' - ')[0].strip()
        
        if len(title) > 100:
            title = title[:100] + "..."
            
        return title
    
    def _robust_parse_json(self, raw: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to parse JSON that may be wrapped in code fences or include extra prose.
        """
        try:
            text = raw.strip()
            if text.startswith("```"):
                lines = text.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                text = "\n".join(lines).strip()
            
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                candidate = text[start_idx:end_idx + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
            return None
        except Exception:
            return None
    
    def _process_llm_recommendations(self, recommendations: Dict[str, Any], user_id: str = None, current_city: str = "Barcelona") -> Dict[str, List[Dict]]:
        """
        Process and enhance recommendations from the LLM API with normalized scores
        """
        try:
            processed = {
                "movies": recommendations.get("movies", []),
                "music": recommendations.get("music", []),
                "places": recommendations.get("places", []),
                "events": recommendations.get("events", [])
            }
            
            history = self._get_user_interaction_history(user_id) if user_id else {}
            
            user_profile = None
            location_data = None
            interaction_data = None
            
            if user_id:
                try:
                    from app.services.user_profile import UserProfileService
                    from app.services.lie_service import LIEService
                    from app.services.cis_service import CISService
                    import asyncio
                    
                    user_service = UserProfileService(timeout=10)
                    lie_service = LIEService(timeout=10)
                    cis_service = CISService(timeout=10)
                    
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    try:
                        user_profile = loop.run_until_complete(user_service.get_user_profile(user_id))
                        location_data = loop.run_until_complete(lie_service.get_location_data(user_id))
                        interaction_data = loop.run_until_complete(cis_service.get_interaction_data(user_id))
                        
                        if user_profile:
                            user_profile = user_profile.safe_dump() if hasattr(user_profile, 'safe_dump') else user_profile.model_dump()
                        if location_data:
                            location_data = location_data.safe_dump() if hasattr(location_data, 'safe_dump') else location_data.model_dump()
                        if interaction_data:
                            interaction_data = interaction_data.safe_dump() if hasattr(interaction_data, 'safe_dump') else interaction_data.model_dump()
                            
                    finally:
                        loop.close()
                        
                except Exception as e:
                    logger.warning(f"Could not fetch user data for enhanced scoring: {str(e)}")
            
            for category, items in processed.items():
                if not isinstance(items, list) or not items:
                    continue
                    
                # Compute raw scores
                raw_scores = []
                for item in items:
                    if isinstance(item, dict):
                        raw = self._compute_ranking_score(
                            item, category, history, user_profile, location_data, interaction_data
                        )
                        item['_raw_score'] = raw
                        raw_scores.append(raw)
                
                # Normalize to 0.1-1.0 range per category
                if raw_scores:
                    min_s = min(raw_scores)
                    max_s = max(raw_scores)
                    if max_s == min_s:
                        norm_score = 0.5
                        for item in items:
                            item["ranking_score"] = round(norm_score, 2)
                    else:
                        for item in items:
                            norm = 0.1 + 0.9 * (item['_raw_score'] - min_s) / (max_s - min_s)
                            item["ranking_score"] = round(norm, 2)
                    # Clean up
                    for item in items:
                        del item['_raw_score']
                
                # Generate reasons if missing
                for item in items:
                    if not item.get("why_would_you_like_this"):
                        item["why_would_you_like_this"] = self._generate_personalized_reason(
                            item, category, "", user_id, current_city
                        )
                
                # Sort by ranking_score descending
                items.sort(key=lambda x: x.get("ranking_score", 0), reverse=True)
            
            return processed
        except Exception as e:
            logger.error(f"Error processing LLM recommendations: {str(e)}")
            return self._get_fallback_recommendations()

    def _get_fallback_recommendations(self) -> Dict[str, List[Dict]]:
        """
        Get fallback recommendations when LLM API fails
        """
        logger.info("Using fallback recommendations due to API failure")
        return {
            "movies": [],
            "music": [],
            "places": [],
            "events": []
        }
    
    def _generate_personalized_reason(self, item: Dict[str, Any], category: str, prompt: str, user_id: str = None, current_city: str = "Barcelona") -> str:
        """Generate personalized reason why user would like this recommendation"""
        prompt_text = prompt.lower() if prompt else "your interests"
        base_reasons = {
            "movies": [
                f"Based on your interest in {prompt_text}, this {item.get('genre', 'film') if item else 'film'} offers compelling storytelling in {current_city}",
                f"The cast featuring {', '.join(item.get('cast', ['talented actors'])[:2]) if item else 'talented actors'} aligns with quality performances you appreciate in {current_city}",
                f"This {item.get('year', 'recent') if item else 'recent'} film's themes resonate with your viewing preferences in {current_city}"
            ],
            "music": [
                f"This {item.get('genre', 'track') if item else 'track'} matches your musical taste with its {item.get('mood', 'engaging') if item else 'engaging'} energy in {current_city}",
                f"The artist {item.get('artist', 'musician') if item else 'musician'} creates the perfect soundtrack for your {current_city} lifestyle",
                f"With {item.get('monthly_listeners', 'many') if item else 'many'} monthly listeners, this song captures the zeitgeist you're looking for in {current_city}"
            ],
            "places": [
                f"Located in {current_city}, this {item.get('type', 'location') if item else 'location'} offers the perfect {item.get('preferred_time', 'anytime') if item else 'anytime'} experience",
                f"With a {item.get('rating', 4.5) if item else 4.5} rating and {item.get('user_ratings_total', 'many') if item else 'many'} reviews, it's a local favorite in {current_city}",
                f"The {item.get('category', 'venue') if item else 'venue'} provides exactly what you're seeking in {current_city}"
            ],
            "events": [
                f"This {item.get('category', 'event') if item else 'event'} happening in {current_city} perfectly matches your cultural interests",
                f"Organized by {item.get('organizer', 'top promoters') if item else 'top promoters'}, it promises a {item.get('duration', 'memorable') if item else 'memorable'} experience in {current_city}",
                f"The {item.get('event_type', 'gathering') if item else 'gathering'} offers {item.get('languages', ['multilingual'])[0] if item else 'multilingual'} accessibility in {current_city}"
            ]
        }
        
        if user_id:
            personalized_additions = [
                "your previous positive interactions suggest you'll love this",
                "based on your engagement history, this aligns perfectly with your preferences",
                "your activity pattern indicates this will be a great match",
                "considering your past choices, this recommendation scores highly for you"
            ]
            base_reason = random.choice(base_reasons.get(category, [f"This recommendation suits your taste in {current_city}"]))
            personal_touch = random.choice(personalized_additions)
            return f"{base_reason}, and {personal_touch}."
        else:
            return random.choice(base_reasons.get(category, [f"This recommendation suits your taste in {current_city}"])) + "."
    
    def _store_in_redis(self, user_id: str, data: Dict[str, Any]):
        """Store recommendations in Redis"""
        try:
            key = f"recommendations:{user_id}"
            data_size = len(json.dumps(data, default=str))
            
            logger.info("Storing recommendations in Redis",
                       user_id=user_id,
                       key=key,
                       data_size_bytes=data_size,
                       ttl_seconds=86400)
            
            self.redis_client.setex(
                key,
                86400,
                json.dumps(data, default=str)
            )
            
            logger.info("Recommendations stored successfully in Redis",
                       user_id=user_id,
                       key=key,
                       data_size_bytes=data_size)
            
            # Publish notification
            try:
                notify_payload = {
                    "type": "notification",
                    "user_id": str(user_id),
                    "message": {"content": "Your new recommendations are ready!"}
                }
                pub_client = redis.Redis(host=settings.redis_host, port=6379, db=0, decode_responses=True)
                pub_client.publish("notifications:user", json.dumps(notify_payload))
                
                logger.info("Notification published successfully",
                           user_id=user_id,
                           channel="notifications:user",
                           notification_type="recommendations_ready")
            except Exception as pub_err:
                logger.error("Failed to publish notification",
                            user_id=user_id,
                            error=str(pub_err),
                            channel="notifications:user")
        except Exception as e:
            logger.error("Error storing recommendations in Redis",
                        user_id=user_id,
                        key=key,
                        error=str(e))
            log_exception("llm_service", e, {"user_id": user_id, "operation": "store_redis"})
    
    def get_recommendations_from_redis(self, user_id: str) -> Dict[str, Any]:
        """Retrieve recommendations from Redis"""
        try:
            key = f"recommendations:{user_id}"
            logger.info("Retrieving recommendations from Redis",
                       user_id=user_id,
                       key=key)
            
            data = self.redis_client.get(key)
            if data:
                data_size = len(data)
                logger.info("Recommendations retrieved successfully from Redis",
                           user_id=user_id,
                           key=key,
                           data_size_bytes=data_size)
                return json.loads(data)
            else:
                logger.info("No recommendations found in Redis",
                           user_id=user_id,
                           key=key)
                return None
        except Exception as e:
            logger.error("Error retrieving recommendations from Redis",
                        user_id=user_id,
                        key=key,
                        error=str(e))
            log_exception("llm_service", e, {"user_id": user_id, "operation": "get_redis"})
            return None
    
    def clear_recommendations(self, user_id: str = None):
        """Clear recommendations from Redis"""
        try:
            if user_id:
                key = f"recommendations:{user_id}"
                logger.info("Clearing recommendations for specific user",
                           user_id=user_id,
                           key=key)
                
                deleted_count = self.redis_client.delete(key)
                logger.info("Recommendations cleared successfully",
                           user_id=user_id,
                           key=key,
                           deleted_count=deleted_count)
            else:
                logger.info("Clearing all recommendations from Redis")
                keys = self.redis_client.keys("recommendations:*")
                if keys:
                    deleted_count = self.redis_client.delete(*keys)
                    logger.info("All recommendations cleared successfully",
                               total_keys=len(keys),
                               deleted_count=deleted_count)
                else:
                    logger.info("No recommendation keys found to clear")
        except Exception as e:
            logger.error("Error clearing recommendations from Redis",
                        user_id=user_id,
                        error=str(e))
            log_exception("llm_service", e, {"user_id": user_id, "operation": "clear_redis"})


llm_service = LLMService(timeout=120)