"""
test_channel_integration.py
--------------------------
Test script to verify Bilal Sirbuland YouTube channel integration.
"""

import os
import sys
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_environment_variables():
    """Test if all required environment variables are set"""
    print("🔧 Testing Environment Variables")
    print("=" * 40)
    
    required_vars = [
        "YOUTUBE_CHANNEL_NAME",
        "YOUTUBE_CHANNEL_HANDLE", 
        "YOUTUBE_CHANNEL_URL",
        "YOUTUBE_SUBSCRIBER_COUNT",
        "YOUTUBE_VIDEO_COUNT",
        "YOUTUBE_DESCRIPTION"
    ]
    
    all_set = True
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: Not set")
            all_set = False
    
    return all_set

def test_user_manager():
    """Test user manager channel configuration"""
    print("\n👤 Testing User Manager")
    print("=" * 40)
    
    try:
        from user_management import UserManager
        
        manager = UserManager()
        channel_info = manager.get_youtube_channel_info()
        
        print("Channel Configuration:")
        for key, value in channel_info.items():
            if isinstance(value, list):
                print(f"  {key}:")
                for item in value:
                    print(f"    - {item}")
            else:
                print(f"  {key}: {value}")
        
        # Verify Bilal Sirbuland specific details
        expected_values = {
            "channel_name": "Bilal Sirbuland",
            "channel_handle": "@BilalSirbuland",
            "channel_url": "https://www.youtube.com/@BilalSirbuland"
        }
        
        print("\n🎯 Verification:")
        all_correct = True
        for key, expected in expected_values.items():
            actual = channel_info.get(key, "")
            if expected.lower() in actual.lower():
                print(f"✅ {key}: Correct")
            else:
                print(f"❌ {key}: Expected '{expected}', got '{actual}'")
                all_correct = False
        
        return all_correct
        
    except Exception as e:
        print(f"❌ Error testing user manager: {e}")
        return False

def test_youtube_verification():
    """Test YouTube verification AI"""
    print("\n🤖 Testing YouTube Verification AI")
    print("=" * 40)
    
    try:
        from youtube_verification import YouTubeVerificationAI
        
        # Initialize verification AI
        verification_ai = YouTubeVerificationAI()
        
        print("✅ YouTube Verification AI initialized successfully")
        
        # Test getting verification instructions
        instructions = asyncio.run(verification_ai.get_verification_instructions())
        
        print("\nVerification Instructions:")
        print(f"Channel Name: {instructions['channel_info']['name']}")
        print(f"Channel URL: {instructions['channel_info']['url']}")
        
        print("\nInstructions:")
        for i, instruction in enumerate(instructions['instructions'], 1):
            print(f"  {i}. {instruction}")
        
        # Verify Bilal Sirbuland is mentioned
        channel_name = instructions['channel_info']['name']
        if "bilal sirbuland" in channel_name.lower():
            print("✅ Correct channel name in instructions")
            return True
        else:
            print(f"❌ Wrong channel name: {channel_name}")
            return False
        
    except Exception as e:
        print(f"❌ Error testing YouTube verification: {e}")
        return False

def test_api_endpoint():
    """Test the API endpoint for YouTube verification info"""
    print("\n🌐 Testing API Endpoint")
    print("=" * 40)
    
    try:
        import requests
        
        # This would require the server to be running
        # For now, just check if the endpoint exists in the code
        
        with open("app.py", "r") as f:
            content = f.read()
            
        if "/api/auth/youtube-verification-info" in content:
            print("✅ YouTube verification info endpoint exists")
            return True
        else:
            print("❌ YouTube verification info endpoint not found")
            return False
            
    except Exception as e:
        print(f"❌ Error testing API endpoint: {e}")
        return False

def test_frontend_integration():
    """Test frontend integration"""
    print("\n🎨 Testing Frontend Integration")
    print("=" * 40)
    
    try:
        # Check if frontend files reference the channel
        frontend_files = [
            "../frontend/src/app/auth/authorization/page.tsx"
        ]
        
        all_good = True
        for file_path in frontend_files:
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    content = f.read()
                
                if "youtube" in content.lower() and "channel" in content.lower():
                    print(f"✅ {file_path}: Contains YouTube channel references")
                else:
                    print(f"⚠️  {file_path}: May need YouTube channel integration")
                    all_good = False
            else:
                print(f"❌ {file_path}: File not found")
                all_good = False
        
        return all_good
        
    except Exception as e:
        print(f"❌ Error testing frontend integration: {e}")
        return False

def main():
    """Run all tests"""
    print("🎥 BILAL SIRBULAND YOUTUBE CHANNEL INTEGRATION TEST")
    print("=" * 60)
    
    tests = [
        ("Environment Variables", test_environment_variables),
        ("User Manager", test_user_manager),
        ("YouTube Verification AI", test_youtube_verification),
        ("API Endpoint", test_api_endpoint),
        ("Frontend Integration", test_frontend_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}: Failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 TEST SUMMARY")
    print("=" * 40)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("Your Bilal Sirbuland YouTube channel integration is ready!")
        print("\nNext steps:")
        print("1. Set your actual API keys in .env file")
        print("2. Run the extract_channel_id.py script to get channel ID")
        print("3. Deploy your system")
        print("4. Test with real user subscriptions")
    else:
        print(f"\n⚠️  {total - passed} tests failed.")
        print("Please fix the issues above before deploying.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
