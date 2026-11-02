"""A failed external integration may be shown without exposing its login."""
import json

import pytest

import supermag_loader as sm


@pytest.mark.parametrize("failure", ["http", "connection"])
def test_service_failure_redacts_query_login(monkeypatch, caplog, failure):
    url = "https://supermag.jhuapl.edu/services/indices.php?python&logon=private-login&extent=60"

    def request(*args, **kwargs):
        if failure == "connection":
            raise ConnectionError("Failed URL: " + url)
        return type("Response", (), {"status_code": 404})()

    monkeypatch.setattr(sm, "_request", request)
    _, error = sm._read_body(url, retries=1)
    assert error["error"] == "supermag_unavailable"
    assert "private-login" not in json.dumps(error) + caplog.text
    assert "[redacted]" in error["detail"]
