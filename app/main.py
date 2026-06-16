"""
AI Triage Agentic System — FastAPI Backend

NOT FOR CLINICAL USE. Research prototype only.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health_routes import router as health_router
from app.api.triage_routes import router as triage_router
from app.api.review_routes import router as review_router
from app.api.governance_routes import router as governance_router
from app.api.explanation_routes import router as explanation_router
from app.api.chat_routes import router as chat_router

app = FastAPI(
    title="AI Triage Agentic System",
    version="2.0.0-ktas",
    description=(
        "Multi-agent AI triage research workflow using public Kaggle KTAS data. "
        "NOT FOR CLINICAL USE. KTAS is not Manchester; no Manchester mapping configured."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Streamlit runs on a different port locally
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    from app.agents.autogen_team import load_azure_config

    return {
        "status": "running",
        "version": "2.0.0-ktas",
        "active_dataset": "Kaggle-KTAS",
        "clinical_use": "NOT_FOR_CLINICAL_USE",
        "automated_manchester_triage": "NOT_IMPLEMENTED",
        "ktas_to_manchester_mapping": "NOT_IMPLEMENTED",
        "chat_agent_orchestration_framework": "autogen-agentchat",
        "chat_agent_status": "configured" if load_azure_config() else "not_configured",
        "docs": "/docs",
    }

app.include_router(health_router)
app.include_router(triage_router)
app.include_router(review_router)
app.include_router(governance_router)
app.include_router(explanation_router)
app.include_router(chat_router)
