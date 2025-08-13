"""
AI-Powered YouTube Channel Verification System
Uses image recognition to verify user subscription to YouTube channel
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple
import base64
import io
from datetime import datetime

from PIL import Image
import openai
from anthropic import Anthropic

from user_management import AuthorizationResult, user_manager

logger = logging.getLogger(__name__)


class YouTubeVerificationAI:
    """AI system for verifying YouTube channel subscription through screenshot analysis"""
    
    def __init__(self):
        """Initialize YouTube verification AI"""
        self.anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Get channel configuration from user manager
        self.channel_config = user_manager.get_youtube_channel_info()
        
        # AI prompts for verification
        self.verification_prompt = self._create_verification_prompt()
        
        logger.info("YouTube Verification AI initialized")
    
    def _create_verification_prompt(self) -> str:
        """Create AI prompt for YouTube verification"""
        channel_name = self.channel_config["channel_name"]
        
        return f"""
You are an expert image analysis AI specialized in verifying YouTube channel subscriptions.

Your task is to analyze a screenshot of a YouTube channel page and determine if the user is subscribed.

CHANNEL TO VERIFY:
- Channel Name: "{channel_name}"
- Expected URL pattern: youtube.com/@{channel_name.lower().replace(' ', '')}

VERIFICATION CRITERIA:
1. **Channel Identity**: Confirm this is the correct channel by checking:
   - Channel name matches exactly: "{channel_name}"
   - Channel handle: "@BilalSirbuland" is visible
   - Channel URL contains: youtube.com/@BilalSirbuland
   - Content description mentions: "TikTok Shop Mastery"

2. **Subscription Status**: Look for subscription indicators:
   - "SUBSCRIBED" button (gray/white with checkmark) - REQUIRED
   - Bell notification icon next to subscribe button - PREFERRED
   - User must be clearly subscribed, not just visiting the channel
   - Subscriber count should be visible (around 104K+ subscribers)

3. **Dynamic Subscriber Count**: Accept any subscriber count that is:
   - 104,000 or higher (channel is growing)
   - Displayed in format: "104K", "105K", "110K", etc.
   - Do NOT reject if count is higher than expected - this is normal growth

3. **Screenshot Authenticity**: Verify the screenshot shows:
   - Genuine YouTube interface (correct fonts, colors, layout)
   - No obvious editing or manipulation
   - Clear, readable content
   - Proper YouTube URL in address bar

RESPONSE FORMAT:
Respond with a JSON object containing:
{{
    "verification_result": "APPROVED" | "REJECTED",
    "channel_found": true | false,
    "channel_name_match": true | false,
    "subscription_confirmed": true | false,
    "confidence_score": 0.0-1.0,
    "details": {{
        "channel_name_detected": "detected name",
        "subscription_indicators": ["list of indicators found"],
        "issues_found": ["list of any issues"],
        "screenshot_quality": "excellent" | "good" | "poor"
    }},
    "reasoning": "Detailed explanation of the verification decision"
}}

IMPORTANT RULES:
- Only approve if ALL criteria are met with high confidence
- Be strict about channel name matching (case-sensitive)
- Reject if screenshot quality is too poor to verify
- Reject if any signs of manipulation are detected
- Require clear subscription indicators, not just channel visit

Analyze the provided screenshot now.
"""
    
    async def verify_subscription_screenshot(
        self, 
        image_data: bytes, 
        uid: str
    ) -> Tuple[AuthorizationResult, Dict[str, Any]]:
        """
        Verify YouTube subscription using screenshot analysis
        
        Args:
            image_data: Screenshot image bytes
            uid: User ID for logging
            
        Returns:
            Tuple of (result, verification_details)
        """
        try:
            # Validate image
            if not self._validate_image(image_data):
                return AuthorizationResult.INVALID_IMAGE, {
                    "error": "Invalid image format or size",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Convert image to base64 for AI analysis
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # Analyze with AI (try Anthropic first, fallback to OpenAI)
            verification_result = await self._analyze_with_anthropic(image_base64)
            
            if not verification_result:
                verification_result = await self._analyze_with_openai(image_base64)
            
            if not verification_result:
                return AuthorizationResult.REJECTED, {
                    "error": "AI analysis failed",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Process AI response
            result, details = self._process_ai_response(verification_result)
            
            # Log verification attempt
            logger.info(f"YouTube verification for {uid}: {result.value}")
            logger.info(f"Verification details: {details}")
            
            return result, details
            
        except Exception as e:
            logger.error(f"YouTube verification failed for {uid}: {e}")
            return AuthorizationResult.REJECTED, {
                "error": "Verification system error",
                "timestamp": datetime.now().isoformat()
            }
    
    def _validate_image(self, image_data: bytes) -> bool:
        """Validate image format and size"""
        try:
            image = Image.open(io.BytesIO(image_data))
            
            # Check format
            if image.format not in ['JPEG', 'PNG', 'WEBP']:
                return False
            
            # Check size (should be reasonable screenshot size)
            width, height = image.size
            if width < 800 or height < 600:  # Too small
                return False
            if width > 4000 or height > 3000:  # Too large
                return False
            
            # Check file size (max 10MB)
            if len(image_data) > 10 * 1024 * 1024:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Image validation failed: {e}")
            return False
    
    async def _analyze_with_anthropic(self, image_base64: str) -> Optional[Dict[str, Any]]:
        """Analyze screenshot using Anthropic Claude Vision"""
        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.verification_prompt
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_base64
                                }
                            }
                        ]
                    }
                ]
            )
            
            # Parse JSON response
            import json
            response_text = response.content[0].text
            
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
            
            return None
            
        except Exception as e:
            logger.error(f"Anthropic analysis failed: {e}")
            return None
    
    async def _analyze_with_openai(self, image_base64: str) -> Optional[Dict[str, Any]]:
        """Analyze screenshot using OpenAI GPT-4 Vision (fallback)"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.verification_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            # Parse JSON response
            import json
            response_text = response.choices[0].message.content
            
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
            
            return None
            
        except Exception as e:
            logger.error(f"OpenAI analysis failed: {e}")
            return None
    
    def _process_ai_response(self, ai_response: Dict[str, Any]) -> Tuple[AuthorizationResult, Dict[str, Any]]:
        """Process AI analysis response and determine final result"""
        try:
            verification_result = ai_response.get("verification_result", "REJECTED")
            channel_found = ai_response.get("channel_found", False)
            channel_name_match = ai_response.get("channel_name_match", False)
            subscription_confirmed = ai_response.get("subscription_confirmed", False)
            confidence_score = ai_response.get("confidence_score", 0.0)
            
            details = {
                "ai_response": ai_response,
                "timestamp": datetime.now().isoformat(),
                "channel_config": self.channel_config,
                "verification_summary": {
                    "channel_found": channel_found,
                    "channel_name_match": channel_name_match,
                    "subscription_confirmed": subscription_confirmed,
                    "confidence_score": confidence_score
                }
            }
            
            # Determine final result based on AI analysis
            if verification_result == "APPROVED" and confidence_score >= 0.8:
                if channel_found and channel_name_match and subscription_confirmed:
                    return AuthorizationResult.APPROVED, details
            
            # Determine specific rejection reason
            if not channel_found:
                return AuthorizationResult.CHANNEL_NOT_FOUND, details
            elif not subscription_confirmed:
                return AuthorizationResult.NOT_SUBSCRIBED, details
            else:
                return AuthorizationResult.REJECTED, details
                
        except Exception as e:
            logger.error(f"Failed to process AI response: {e}")
            return AuthorizationResult.REJECTED, {
                "error": "Failed to process AI response",
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_verification_instructions(self) -> Dict[str, Any]:
        """Get instructions for users on how to take verification screenshot"""
        channel_config = self.channel_config
        
        return {
            "channel_info": {
                "name": channel_config["channel_name"],
                "url": channel_config["channel_url"],
                "profile_image": channel_config["profile_image_url"]
            },
            "instructions": [
                f"1. Visit our YouTube channel: {channel_config['channel_url']}",
                "2. Click the SUBSCRIBE button if you haven't already",
                "3. Make sure the button shows 'SUBSCRIBED' (gray/white with checkmark)",
                "4. Take a full screenshot of the channel page showing:",
                "   - Channel name and profile picture",
                "   - SUBSCRIBED button clearly visible",
                "   - YouTube URL in the address bar",
                "5. Upload the screenshot below for verification"
            ],
            "screenshot_requirements": [
                "Full browser window screenshot (not cropped)",
                "Clear and readable text",
                "No editing or manipulation",
                "JPEG, PNG, or WebP format",
                "Maximum 10MB file size",
                "Minimum 800x600 resolution"
            ],
            "common_issues": [
                "Screenshot too blurry or small",
                "Wrong channel (check the channel name carefully)",
                "Not subscribed (button shows 'SUBSCRIBE' instead of 'SUBSCRIBED')",
                "Cropped screenshot missing important details"
            ]
        }


# Global verification AI instance
youtube_verification_ai = YouTubeVerificationAI()
