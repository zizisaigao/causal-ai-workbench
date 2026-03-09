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


def test_psm_no_match_returns_structured_business_error():
    with open('data/sample/hillstrom_style_sample.csv', 'rb') as f:
        response = client.post(
            '/api/analyze/psm',
            files={'file': ('hillstrom_style_sample.csv', f, 'text/csv')},
            data={
                'treatment_col': 'treatment',
                'outcome_col': 'outcome',
                'covariates': 'age,income,prior_spend',
                'psm_caliper': '0.000001',
            },
        )
    assert response.status_code == 422
    body = response.json()
    assert body['error']['code'] == 'psm_no_matches'
    assert 'caliper is too strict' in body['error']['message']
