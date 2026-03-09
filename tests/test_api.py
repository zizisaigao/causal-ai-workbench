from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get('/api/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_method_recommendation_default_psm():
    response = client.post(
        '/api/methods/recommend',
        json={
            'has_time': False,
            'has_group': False,
            'treatment_binary': True,
            'observational': True,
            'wants_targeting': False,
        },
    )
    assert response.status_code == 200
    assert response.json()['primary_method'] == 'psm'
