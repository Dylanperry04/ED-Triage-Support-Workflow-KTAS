"""
Tests for the AutoGen clinician chat API route (app/api/chat_routes.py).

These are the first API-route-level tests in this codebase; everything else
exercises agent functions directly. Scoped to just the new chat route, since
backfilling route-level tests for every existing endpoint is a separate,
larger task than the one requested.
"""
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


class TestChatRoute:
    def test_returns_503_when_azure_not_configured(self, monkeypatch):
        for key in (
            "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_API_VERSION",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(
            "app.agents.autogen_team.ENV_PATH", Path("/nonexistent/.env")
        )

        response = client.post("/chat/ask", json={"question": "Tell me about stay 1"})

        assert response.status_code == 503
        body = response.json()
        assert body["detail"]["human_review_required"] is True
        assert "not configured" in body["detail"]["reply_text"].lower()

    def test_requires_question_field(self):
        response = client.post("/chat/ask", json={})
        assert response.status_code == 422  # FastAPI/Pydantic validation error
