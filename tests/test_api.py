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


def test_uplift_analysis_success():
    with open('data/sample/hillstrom_style_sample.csv', 'rb') as f:
        response = client.post(
            '/api/analyze/uplift',
            files={'file': ('hillstrom_style_sample.csv', f, 'text/csv')},
            data={
                'treatment_col': 'treatment',
                'outcome_col': 'outcome',
                'covariates': 'age,income,prior_spend',
                'uplift_buckets': '5',
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body['result']['method'] == 'uplift'
    assert 'bucket_summary' in body['result']['diagnostics']
    assert 'top_segment_profile_mean' in body['result']['diagnostics']


def test_causal_forest_analysis_success():
    with open('data/sample/hillstrom_style_sample.csv', 'rb') as f:
        response = client.post(
            '/api/analyze/causal_forest',
            files={'file': ('hillstrom_style_sample.csv', f, 'text/csv')},
            data={
                'treatment_col': 'treatment',
                'outcome_col': 'outcome',
                'covariates': 'age,income,prior_spend',
                'uplift_buckets': '5',
                'cf_n_estimators': '50',
                'cf_min_samples_leaf': '2',
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body['result']['method'] == 'causal_forest'
    assert 'bucket_summary' in body['result']['diagnostics']
    assert 'implementation' in body['result']['diagnostics']


def test_rdd_analysis_success():
    with open('data/sample/hillstrom_style_sample.csv', 'rb') as f:
        response = client.post(
            '/api/analyze/rdd',
            files={'file': ('hillstrom_style_sample.csv', f, 'text/csv')},
            data={
                'treatment_col': 'treatment',
                'outcome_col': 'outcome',
                'covariates': 'age,income,prior_spend',
                'running_col': 'prior_spend',
                'cutoff': '90',
                'rdd_bandwidth': '60',
            },
        )
    assert response.status_code == 200
    assert response.json()['result']['method'] == 'rdd'


def test_iv_analysis_success():
    with open('data/sample/hillstrom_style_sample.csv', 'rb') as f:
        response = client.post(
            '/api/analyze/iv',
            files={'file': ('hillstrom_style_sample.csv', f, 'text/csv')},
            data={
                'treatment_col': 'treatment',
                'outcome_col': 'outcome',
                'covariates': 'age,income,prior_spend',
                'instrument_col': 'is_target_group',
            },
        )
    assert response.status_code == 200
    assert response.json()['result']['method'] == 'iv'


def test_iv_missing_instrument_returns_structured_error():
    with open('data/sample/hillstrom_style_sample.csv', 'rb') as f:
        response = client.post(
            '/api/analyze/iv',
            files={'file': ('hillstrom_style_sample.csv', f, 'text/csv')},
            data={
                'treatment_col': 'treatment',
                'outcome_col': 'outcome',
                'covariates': 'age,income,prior_spend',
            },
        )
    assert response.status_code == 422
    assert response.json()['error']['code'] == 'invalid_analysis_input'


def test_rdd_missing_cutoff_returns_structured_error():
    with open('data/sample/hillstrom_style_sample.csv', 'rb') as f:
        response = client.post(
            '/api/analyze/rdd',
            files={'file': ('hillstrom_style_sample.csv', f, 'text/csv')},
            data={
                'treatment_col': 'treatment',
                'outcome_col': 'outcome',
                'covariates': 'age,income,prior_spend',
                'running_col': 'prior_spend',
            },
        )
    assert response.status_code == 422
    assert response.json()['error']['code'] == 'invalid_analysis_input'
