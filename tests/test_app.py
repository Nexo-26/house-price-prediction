import os
import sys
import json
import pytest

# Add parent directory to path so python can find the frontend and train modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from frontend.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    """Test that the homepage loads correctly."""
    response = client.get('/')
    assert response.status_code == 200
    html_content = response.data.decode('utf-8')
    assert "VALUATION" in html_content or "Valuation" in html_content
    assert "House Price Prediction Engine" in html_content

def test_predict_endpoint_valid_payload(client):
    """Test the prediction endpoint with a standard valid house payload."""
    payload = {
        "GrLivArea": 1500,
        "TotalBsmtSF": 1000,
        "LotArea": 10000,
        "YearBuilt": 2000,
        "YearRemodAdd": 2000,
        "Neighborhood": "CollgCr",
        "OverallQual": 6,
        "KitchenQual": "Gd",
        "CentralAir": "Y",
        "BedroomAbvGr": 3,
        "FullBath": 2,
        "HalfBath": 0,
        "GarageCars": 2
    }
    response = client.post('/predict', 
                           data=json.dumps(payload),
                           content_type='application/json')
    
    assert response.status_code == 200
    data = json.loads(response.data.decode('utf-8'))
    
    assert "predicted_price" in data
    assert "neighborhood" in data
    assert "neighborhood_avg" in data
    assert "percent_diff" in data
    assert isinstance(data["predicted_price"], (int, float))
    assert data["neighborhood"] == "CollgCr"

def test_predict_endpoint_empty_payload(client):
    """Test response when empty payload is passed."""
    response = client.post('/predict', 
                           data=json.dumps({}),
                           content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data.decode('utf-8'))
    assert "error" in data
