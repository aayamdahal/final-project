import json
import base64
from unittest import mock

import app as app_module


def _client():
    app_module.app.config['TESTING'] = True
    return app_module.app.test_client()


def test_index_serves_html():
    resp = _client().get('/')
    assert resp.status_code == 200
    assert b'canvas' in resp.data.lower()


def test_evaluate_returns_result():
    payload = {"image": "data:image/png;base64," +
               base64.b64encode(b"fake").decode()}
    with mock.patch.object(app_module, 'evaluate_image',
                           return_value={"expression": "4+3", "result": 7}):
        resp = _client().post('/api/evaluate', json=payload)
    assert resp.status_code == 200
    body = json.loads(resp.data)
    assert body["expression"] == "4+3"
    assert body["result"] == 7


def test_evaluate_handles_value_error():
    payload = {"image": "data:image/png;base64," +
               base64.b64encode(b"fake").decode()}
    with mock.patch.object(app_module, 'evaluate_image',
                           side_effect=ValueError("nothing drawn")):
        resp = _client().post('/api/evaluate', json=payload)
    assert resp.status_code == 400
    assert "error" in json.loads(resp.data)


def test_evaluate_missing_image():
    resp = _client().post('/api/evaluate', json={})
    assert resp.status_code == 400
