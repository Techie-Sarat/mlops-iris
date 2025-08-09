import pytest
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predict_api.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    """Test the health endpoint"""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'status' in data

def test_predict_endpoint(client):
    """Test the predict endpoint"""
    test_data = {
        'features': [5.1, 3.5, 1.4, 0.2]
    }
    response = client.post('/predict', 
                          data=json.dumps(test_data),
                          content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'prediction' in data
    assert 'predicted_class' in data

def test_invalid_input(client):
    """Test invalid input handling"""
    test_data = {
        'features': [5.1, 3.5]  # Too few features
    }
    response = client.post('/predict', 
                          data=json.dumps(test_data),
                          content_type='application/json')
    assert response.status_code == 400

def test_metrics_endpoint(client):
    """Test the metrics endpoint"""
    response = client.get('/metrics')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'total_predictions' in data