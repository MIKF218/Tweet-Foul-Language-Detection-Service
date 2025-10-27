"""
Tweet Classifier API - Test Suite
==================================



Run tests:
    pytest test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the FastAPI app and the startup function
from app import app, load_artifacts

# Load models before creating test client
load_artifacts()

# Create test client with raise_server_exceptions=False to see actual errors
client = TestClient(app, raise_server_exceptions=False)


# ============================================================================
# DEBUG TEST - Run this first to see any errors
# ============================================================================

def test_debug_api_startup():
    """
    DEBUG: Test API startup and show any errors
    """
    response = client.get("/health")

    print(f"\n{'='*70}")
    print(f"Health Check Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {response.json()}")
    else:
        print(f"Error: {response.text}")
    print(f"{'='*70}\n")

    assert response.status_code == 200, f"API not healthy: {response.text}"


# ============================================================================
# TEST 1: HAPPY PATH - Proper Tweet
# ============================================================================

def test_happy_path_proper_tweet():
    """
    TEST 1: Happy Path

    A clearly innocent/proper tweet should be classified as 'proper' (label 0)
    """
    # Arrange
    proper_tweet = {
        "text": "I love this beautiful sunny day! Great weather for a picnic."
    }

    # Act
    response = client.post("/predict", json=proper_tweet)

    # DEBUG: Show error if not 200
    if response.status_code != 200:
        print(f"\n{'='*70}")
        print(f"❌ ERROR DETAILS:")
        print(f"Status Code: {response.status_code}")
        print(f"Response Text: {response.text}")
        try:
            print(f"Response JSON: {response.json()}")
        except:
            pass
        print(f"{'='*70}\n")

    # Assert
    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Error: {response.text}"

    data = response.json()

    # Verify response structure
    assert "prediction" in data
    assert "label" in data
    assert "confidence" in data
    assert "text" in data

    # Verify classification
    assert data["prediction"] == "proper", f"Expected 'proper', got '{data['prediction']}'"
    assert data["label"] == 0, f"Expected label 0, got {data['label']}"

    # Verify confidence is valid
    assert 0 <= data["confidence"] <= 1, f"Confidence out of range: {data['confidence']}"

    print("✅ TEST 1 PASSED: Happy path - Proper tweet classified correctly")


# ============================================================================
# TEST 2: FOUL TWEET DETECTION
# ============================================================================

def test_foul_tweet_detection():
    """
    TEST 2: Foul Example

    A clearly offensive/foul tweet should be classified as 'foul' (label 1)
    """
    # Arrange - Use stronger offensive language
    foul_tweet = {
        "text": "fuck you bitch stupid ass hoe I hate you trash"
    }

    # Act
    response = client.post("/predict", json=foul_tweet)

    # Assert
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    data = response.json()

    # Verify response structure
    assert "prediction" in data
    assert "label" in data
    assert "confidence" in data

    # Verify classification
    assert data["prediction"] == "foul", f"Expected 'foul', got '{data['prediction']}' with confidence {data['confidence']}"
    assert data["label"] == 1, f"Expected label 1, got {data['label']}"

    # Verify confidence (should be reasonably high for clearly foul content)
    assert data["confidence"] > 0.5, f"Expected confidence > 0.5, got {data['confidence']}"

    print("✅ TEST 2 PASSED: Foul tweet detected correctly")


# ============================================================================
# TEST 3: INVALID INPUT
# ============================================================================

def test_invalid_input_empty_text():
    """
    TEST 3: Invalid Input

    Empty text should be rejected with 422 Validation Error
    """
    # Arrange
    invalid_tweet = {
        "text": ""
    }

    # Act
    response = client.post("/predict", json=invalid_tweet)

    # Assert
    assert response.status_code == 422, \
        f"Expected 422 for empty text, got {response.status_code}"

    print("✅ TEST 3 PASSED: Invalid input (empty text) rejected correctly")


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    """
    Run tests directly
    
    Usage:
        python test_api.py
        
    Or with pytest:
        pytest test_api.py -v
        pytest test_api.py -v -s  # Show print statements
    """
    import sys

    print("\n" + "="*70)
    print("RUNNING TWEET CLASSIFIER API TESTS")
    print("="*70)

    # Run with pytest
    exit_code = pytest.main([__file__, "-v", "-s"])

    print("\n" + "="*70)
    print("TEST SUITE SUMMARY")
    print("="*70)
    print("✅ Test 1: Happy path (proper tweet)")
    print("✅ Test 2: Foul tweet detection")
    print("✅ Test 3: Invalid input handling")
    print("\nTotal: 3 tests (minimum required)")
    print("="*70 + "\n")

    sys.exit(exit_code)