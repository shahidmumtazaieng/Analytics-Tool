"""
Enhanced User Management System
Handles authentication, authorization, and API configuration
"""

import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import base64
import json

from firebase_admin import auth, firestore, storage
from PIL import Image
import io

logger = logging.getLogger(__name__)


class UserStatus(str, Enum):
    """User authorization status"""
    PENDING = "pending"           # Just signed up, needs authorization
    AUTHORIZED = "authorized"     # YouTube verification passed
    INACTIVE = "inactive"         # Failed authorization or suspended
    API_PENDING = "api_pending"   # Authorized but needs API configuration


class AuthorizationResult(str, Enum):
    """Authorization check results"""
    APPROVED = "approved"
    REJECTED = "rejected"
    INVALID_IMAGE = "invalid_image"
    CHANNEL_NOT_FOUND = "channel_not_found"
    NOT_SUBSCRIBED = "not_subscribed"


@dataclass
class UserProfile:
    """Enhanced user profile with authorization data"""
    uid: str
    email: str
    display_name: str
    status: UserStatus
    created_at: datetime
    last_login: datetime
    authorization_attempts: int
    api_configured: bool
    youtube_verification_data: Optional[Dict[str, Any]] = None
    api_keys: Optional[Dict[str, str]] = None


@dataclass
class APIConfiguration:
    """User's API configuration"""
    anthropic_api_key: Optional[str] = None
    bright_data_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    configured_at: Optional[datetime] = None
    last_validated: Optional[datetime] = None


class UserManager:
    """Enhanced user management with authorization and API configuration"""
    
    def __init__(self):
        """Initialize user manager"""
        self.db = firestore.client()
        self.bucket = storage.bucket()
        
        # YouTube channel configuration - Bilal Sirbuland
        self.youtube_channel_config = {
            "channel_name": os.getenv("YOUTUBE_CHANNEL_NAME", "Bilal Sirbuland"),
            "channel_handle": os.getenv("YOUTUBE_CHANNEL_HANDLE", "@BilalSirbuland"),
            "channel_id": os.getenv("YOUTUBE_CHANNEL_ID", "UCYourChannelIdHere"),  # Will be extracted from API
            "channel_url": os.getenv("YOUTUBE_CHANNEL_URL", "https://www.youtube.com/@BilalSirbuland"),
            "subscriber_count": os.getenv("YOUTUBE_SUBSCRIBER_COUNT", "104000"),
            "video_count": os.getenv("YOUTUBE_VIDEO_COUNT", "132"),
            "description": os.getenv("YOUTUBE_DESCRIPTION", "TikTok Shop Mastery (UK & USA)"),
            "profile_image_url": os.getenv("YOUTUBE_CHANNEL_PROFILE_IMAGE", "https://yt3.ggpht.com/your-channel-image.jpg"),
            "subscriber_threshold": 1,  # Minimum subscribers needed
            "verification_instructions": "Please subscribe to Bilal Sirbuland's YouTube channel and enable bell notifications, then upload a screenshot showing your subscription status.",
            "expected_elements": [
                "Channel name: Bilal Sirbuland",
                "Handle: @BilalSirbuland",
                "Subscribed button (gray/white with checkmark)",
                "Bell notification icon (enabled)",
                "Subscriber count 104K or higher (channel is growing)",
                "TikTok Shop related content description"
            ],
            "subscriber_count_validation": {
                "minimum_count": 104000,  # Baseline from your image
                "accept_higher": True,    # Accept growth - this is GOOD!
                "accept_formats": ["104K", "105K", "110K", "120K", "104,000", "105,000"],
                "growth_expected": True   # Channel is actively growing
            }
        }
        
        logger.info("User Manager initialized")
    
    async def create_user_profile(self, user_data: Dict[str, Any]) -> UserProfile:
        """Create new user profile with pending authorization"""
        try:
            uid = user_data["uid"]
            email = user_data["email"]
            display_name = user_data.get("display_name", email.split("@")[0])
            
            # Check if user already exists
            existing_user = await self.get_user_profile(uid)
            if existing_user:
                # Update last login
                await self.update_last_login(uid)
                return existing_user
            
            # Create new user profile
            profile = UserProfile(
                uid=uid,
                email=email,
                display_name=display_name,
                status=UserStatus.PENDING,
                created_at=datetime.now(),
                last_login=datetime.now(),
                authorization_attempts=0,
                api_configured=False
            )
            
            # Save to Firestore
            user_doc = {
                "uid": profile.uid,
                "email": profile.email,
                "display_name": profile.display_name,
                "status": profile.status.value,
                "created_at": profile.created_at,
                "last_login": profile.last_login,
                "authorization_attempts": profile.authorization_attempts,
                "api_configured": profile.api_configured,
                "youtube_verification_data": None,
                "api_keys": None
            }
            
            self.db.collection("users").document(uid).set(user_doc)
            
            logger.info(f"Created new user profile: {email}")
            return profile
            
        except Exception as e:
            logger.error(f"Failed to create user profile: {e}")
            raise
    
    async def get_user_profile(self, uid: str) -> Optional[UserProfile]:
        """Get user profile by UID"""
        try:
            doc = self.db.collection("users").document(uid).get()
            
            if not doc.exists:
                return None
            
            data = doc.to_dict()
            
            return UserProfile(
                uid=data["uid"],
                email=data["email"],
                display_name=data["display_name"],
                status=UserStatus(data["status"]),
                created_at=data["created_at"],
                last_login=data["last_login"],
                authorization_attempts=data["authorization_attempts"],
                api_configured=data["api_configured"],
                youtube_verification_data=data.get("youtube_verification_data"),
                api_keys=data.get("api_keys")
            )
            
        except Exception as e:
            logger.error(f"Failed to get user profile: {e}")
            return None
    
    async def update_last_login(self, uid: str) -> bool:
        """Update user's last login timestamp"""
        try:
            self.db.collection("users").document(uid).update({
                "last_login": datetime.now()
            })
            return True
        except Exception as e:
            logger.error(f"Failed to update last login: {e}")
            return False
    
    async def check_user_access(self, uid: str) -> Dict[str, Any]:
        """Check if user has access to the system"""
        try:
            profile = await self.get_user_profile(uid)
            
            if not profile:
                return {
                    "has_access": False,
                    "reason": "user_not_found",
                    "requires_action": "signup"
                }
            
            # Update last login
            await self.update_last_login(uid)
            
            if profile.status == UserStatus.PENDING:
                return {
                    "has_access": False,
                    "reason": "authorization_pending",
                    "requires_action": "youtube_verification",
                    "is_first_time": True
                }
            
            elif profile.status == UserStatus.INACTIVE:
                return {
                    "has_access": False,
                    "reason": "account_inactive",
                    "requires_action": "contact_support",
                    "authorization_attempts": profile.authorization_attempts
                }
            
            elif profile.status == UserStatus.API_PENDING:
                return {
                    "has_access": True,
                    "limited_access": True,
                    "reason": "api_configuration_needed",
                    "requires_action": "configure_apis",
                    "show_api_popup": True
                }
            
            elif profile.status == UserStatus.AUTHORIZED:
                # Check if APIs are configured
                if not profile.api_configured:
                    return {
                        "has_access": True,
                        "limited_access": True,
                        "reason": "api_configuration_needed",
                        "requires_action": "configure_apis",
                        "show_api_popup": False  # Not first time
                    }
                
                return {
                    "has_access": True,
                    "full_access": True,
                    "status": "active",
                    "api_configured": True
                }
            
            return {
                "has_access": False,
                "reason": "unknown_status",
                "requires_action": "contact_support"
            }
            
        except Exception as e:
            logger.error(f"Failed to check user access: {e}")
            return {
                "has_access": False,
                "reason": "system_error",
                "requires_action": "try_again"
            }
    
    async def save_api_configuration(self, uid: str, api_config: APIConfiguration) -> bool:
        """Save user's API configuration securely"""
        try:
            # Encrypt API keys (in production, use proper encryption)
            encrypted_keys = {
                "anthropic_api_key": self._encrypt_api_key(api_config.anthropic_api_key) if api_config.anthropic_api_key else None,
                "bright_data_api_key": self._encrypt_api_key(api_config.bright_data_api_key) if api_config.bright_data_api_key else None,
                "openai_api_key": self._encrypt_api_key(api_config.openai_api_key) if api_config.openai_api_key else None,
                "gemini_api_key": self._encrypt_api_key(api_config.gemini_api_key) if api_config.gemini_api_key else None,
            }
            
            # Update user document
            self.db.collection("users").document(uid).update({
                "api_keys": encrypted_keys,
                "api_configured": True,
                "api_configured_at": datetime.now(),
                "status": UserStatus.AUTHORIZED.value
            })
            
            logger.info(f"Saved API configuration for user: {uid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save API configuration: {e}")
            return False
    
    async def get_user_api_keys(self, uid: str) -> Optional[Dict[str, str]]:
        """Get user's decrypted API keys"""
        try:
            profile = await self.get_user_profile(uid)
            
            if not profile or not profile.api_keys:
                return None
            
            # Decrypt API keys
            decrypted_keys = {}
            for key, encrypted_value in profile.api_keys.items():
                if encrypted_value:
                    decrypted_keys[key] = self._decrypt_api_key(encrypted_value)
            
            return decrypted_keys
            
        except Exception as e:
            logger.error(f"Failed to get user API keys: {e}")
            return None
    
    def _encrypt_api_key(self, api_key: str) -> str:
        """Encrypt API key (simplified - use proper encryption in production)"""
        if not api_key:
            return ""
        
        # In production, use proper encryption like Fernet
        # For now, simple base64 encoding (NOT SECURE)
        return base64.b64encode(api_key.encode()).decode()
    
    def _decrypt_api_key(self, encrypted_key: str) -> str:
        """Decrypt API key (simplified - use proper decryption in production)"""
        if not encrypted_key:
            return ""
        
        # In production, use proper decryption
        # For now, simple base64 decoding
        try:
            return base64.b64decode(encrypted_key.encode()).decode()
        except Exception:
            return ""
    
    async def update_authorization_status(
        self, 
        uid: str, 
        status: UserStatus, 
        verification_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update user authorization status"""
        try:
            update_data = {
                "status": status.value,
                "last_updated": datetime.now()
            }
            
            if verification_data:
                update_data["youtube_verification_data"] = verification_data
            
            if status == UserStatus.INACTIVE:
                # Increment authorization attempts
                profile = await self.get_user_profile(uid)
                if profile:
                    update_data["authorization_attempts"] = profile.authorization_attempts + 1
            
            self.db.collection("users").document(uid).update(update_data)
            
            logger.info(f"Updated authorization status for {uid}: {status.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update authorization status: {e}")
            return False
    
    def get_youtube_channel_info(self) -> Dict[str, Any]:
        """Get YouTube channel information for verification"""
        return self.youtube_channel_config
    
    async def validate_user_apis(self, uid: str) -> Dict[str, bool]:
        """Validate user's API keys"""
        try:
            api_keys = await self.get_user_api_keys(uid)
            
            if not api_keys:
                return {"valid": False, "reason": "no_apis_configured"}
            
            validation_results = {}
            
            # Validate Anthropic API
            if api_keys.get("anthropic_api_key"):
                validation_results["anthropic"] = await self._validate_anthropic_api(api_keys["anthropic_api_key"])
            
            # Validate Bright Data API
            if api_keys.get("bright_data_api_key"):
                validation_results["bright_data"] = await self._validate_bright_data_api(api_keys["bright_data_api_key"])
            
            # Validate OpenAI API
            if api_keys.get("openai_api_key"):
                validation_results["openai"] = await self._validate_openai_api(api_keys["openai_api_key"])
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Failed to validate user APIs: {e}")
            return {"valid": False, "reason": "validation_error"}
    
    async def _validate_anthropic_api(self, api_key: str) -> bool:
        """Validate Anthropic API key"""
        try:
            # In production, make a test API call
            # For now, just check if key looks valid
            return len(api_key) > 20 and api_key.startswith("sk-")
        except Exception:
            return False
    
    async def _validate_bright_data_api(self, api_key: str) -> bool:
        """Validate Bright Data API key"""
        try:
            # In production, make a test API call
            # For now, just check if key looks valid
            return len(api_key) > 10
        except Exception:
            return False
    
    async def _validate_openai_api(self, api_key: str) -> bool:
        """Validate OpenAI API key"""
        try:
            # In production, make a test API call
            # For now, just check if key looks valid
            return len(api_key) > 20 and api_key.startswith("sk-")
        except Exception:
            return False


# Global user manager instance
user_manager = UserManager()
