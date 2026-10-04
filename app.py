"""
FinVerify AI - Hallucination-Resistant Financial RAG System
Streamlit Application for Hackathon Demo
"""

from __future__ import annotations
import os
import io
from typing import List, Dict, Any, Tuple
import streamlit as st
from dotenv import load_dotenv

# Load local environment variables
load_dotenv()

from ingest import ingest_multiple_pdfs, DocumentChunk
from rag import (
    VectorStoreManager,
    LLMGenerator,
    FinVerifyRAG,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIM,
)
from guardrail import EXACT_FALLBACK_ANSWER

# Page Configuration
st.set_page_config(
    page_title="FinVerify AI | Financial RAG",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Hackathon Demo Polish
st.markdown(
    """
    <style>
    /* ========================================================
       FINVERIFY AI - CYBERNETIC FINANCIAL GLASS & NEON THEME
       ======================================================== */
    
    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes cyberGlow {
        0% {
            box-shadow: 0 0 15px rgba(56, 189, 248, 0.25), inset 0 0 15px rgba(56, 189, 248, 0.05);
            border-color: rgba(56, 189, 248, 0.4);
        }
        50% {
            box-shadow: 0 0 35px rgba(56, 189, 248, 0.5), inset 0 0 25px rgba(56, 189, 248, 0.12);
            border-color: rgba(56, 189, 248, 0.85);
        }
        100% {
            box-shadow: 0 0 15px rgba(56, 189, 248, 0.25), inset 0 0 15px rgba(56, 189, 248, 0.05);
            border-color: rgba(56, 189, 248, 0.4);
        }
    }

    @keyframes pulseGreen {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { box-shadow: 0 0 0 14px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    @keyframes pulseAmber {
        0% { box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.7); }
        70% { box-shadow: 0 0 0 14px rgba(245, 158, 11, 0); }
        100% { box-shadow: 0 0 0 0 rgba(245, 158, 11, 0); }
    }

    @keyframes pulseRed {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
        70% { box-shadow: 0 0 0 14px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }

    @keyframes slideUpFade {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Hero Branding */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 35%, #34d399 70%, #38bdf8 100%);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientFlow 6s ease infinite;
        margin-bottom: 2px;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1.25rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 4px;
    }
    .hero-tagline {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-bottom: 18px;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, #131d35 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: #38bdf8;
        box-shadow: 0 8px 24px rgba(56, 189, 248, 0.25);
    }
    .metric-val {
        font-size: 1.9rem;
        font-weight: 800;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }

    /* Confidence Badges with Live Radar Pulse */
    .confidence-badge-high {
        background: rgba(16, 185, 129, 0.2);
        border: 1.5px solid #10b981;
        color: #34d399;
        padding: 7px 18px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.92rem;
        animation: pulseGreen 2.2s infinite;
    }
    .confidence-badge-medium {
        background: rgba(245, 158, 11, 0.2);
        border: 1.5px solid #f59e0b;
        color: #fbbf24;
        padding: 7px 18px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.92rem;
        animation: pulseAmber 2.2s infinite;
    }
    .confidence-badge-insufficient {
        background: rgba(239, 68, 68, 0.2);
        border: 1.5px solid #ef4444;
        color: #f87171;
        padding: 7px 18px;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.92rem;
        animation: pulseRed 2.2s infinite;
    }

    /* Glowing Answer Container */
    .answer-container {
        background: linear-gradient(145deg, #0b1329 0%, #0f172a 100%);
        border: 1px solid rgba(56, 189, 248, 0.4);
        border-radius: 14px;
        padding: 24px;
        margin: 16px 0;
        animation: cyberGlow 5s ease-in-out infinite, slideUpFade 0.5s ease-out;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
    }

    /* Wide Horizontal Citation Cards */
    .citation-grid-card {
        background: linear-gradient(145deg, #0f172a 0%, #172338 100%);
        border: 1px solid #334155;
        border-left: 4px solid #38bdf8;
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 14px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
        white-space: normal !important;
        word-break: normal !important;
        overflow-wrap: break-word !important;
    }
    .citation-grid-card:hover {
        transform: translateY(-4px) scale(1.01);
        border-color: #38bdf8;
        box-shadow: 0 12px 28px rgba(56, 189, 248, 0.3);
    }
    .citation-tag {
        font-size: 0.78rem;
        background: #1e293b;
        color: #e2e8f0;
        padding: 4px 9px;
        border-radius: 6px;
        margin-right: 6px;
        display: inline-block;
        font-weight: 500;
        border: 1px solid #334155;
    }
    .citation-snippet {
        font-size: 0.94rem;
        color: #cbd5e1;
        line-height: 1.65;
        background: #090e1a;
        padding: 14px 16px;
        border-radius: 8px;
        border: 1px solid #1e293b;
        margin-top: 10px;
        white-space: normal !important;
        word-break: normal !important;
        overflow-wrap: break-word !important;
        display: block !important;
        width: 100% !important;
        text-align: left !important;
    }

    /* ========================================================
       MOVING AI BOTS ALONG EDGES & FLOATING CHARACTERS
       ======================================================== */

    /* Bot 1: Left Edge Floating Patrol Sentinel */
    @keyframes patrolLeftEdge {
        0% {
            top: 18%;
            transform: translateX(0px) rotate(-3deg);
        }
        50% {
            top: 72%;
            transform: translateX(8px) rotate(4deg);
        }
        100% {
            top: 18%;
            transform: translateX(0px) rotate(-3deg);
        }
    }

    @keyframes thrusterFlicker {
        0%, 100% { opacity: 0.6; height: 12px; }
        50% { opacity: 1; height: 18px; filter: drop-shadow(0 0 8px #38bdf8); }
    }

    @keyframes eyeBlink {
        0%, 90%, 100% { transform: scaleY(1); }
        95% { transform: scaleY(0.1); }
    }

    .bot-edge-left {
        position: fixed;
        left: 14px;
        top: 25%;
        z-index: 999999;
        animation: patrolLeftEdge 13s ease-in-out infinite;
        cursor: pointer;
        display: flex;
        flex-direction: column;
        align-items: center;
        transition: transform 0.2s ease;
    }
    .bot-edge-left:hover {
        transform: scale(1.15) !important;
    }
    .bot-edge-left:hover .bot-speech {
        opacity: 1;
        transform: translateY(0) scale(1);
    }

    .bot-speech {
        opacity: 0;
        transform: translateY(6px) scale(0.9);
        transition: all 0.25s ease;
        background: #0f172a;
        color: #38bdf8;
        border: 1px solid #38bdf8;
        padding: 6px 12px;
        border-radius: 12px;
        font-size: 0.76rem;
        font-weight: 700;
        white-space: nowrap;
        position: absolute;
        left: 64px;
        top: 6px;
        box-shadow: 0 4px 15px rgba(56, 189, 248, 0.4);
        pointer-events: none;
    }

    .bot-frame {
        width: 52px;
        height: 62px;
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 2px solid #38bdf8;
        border-radius: 18px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.35);
        position: relative;
    }
    .bot-antenna-dot {
        width: 6px;
        height: 6px;
        background: #34d399;
        border-radius: 50%;
        position: absolute;
        top: -8px;
        box-shadow: 0 0 8px #34d399;
    }
    .bot-visor {
        width: 32px;
        height: 12px;
        background: #0284c7;
        border-radius: 8px;
        box-shadow: 0 0 10px #38bdf8;
        display: flex;
        align-items: center;
        justify-content: space-around;
        padding: 0 3px;
        animation: eyeBlink 3.5s infinite;
    }
    .bot-eye-glow {
        width: 6px;
        height: 6px;
        background: #38bdf8;
        border-radius: 50%;
        box-shadow: 0 0 6px #ffffff;
    }
    .bot-core-light {
        width: 14px;
        height: 14px;
        border-radius: 50%;
        background: #10b981;
        margin-top: 6px;
        box-shadow: 0 0 10px #10b981;
    }
    .bot-thruster-flame {
        width: 16px;
        background: linear-gradient(180deg, #38bdf8, transparent);
        border-radius: 0 0 8px 8px;
        animation: thrusterFlicker 0.6s infinite;
    }

    /* Bot 2: Top Edge Hovering Scanning Drone */
    @keyframes patrolTopEdge {
        0% {
            right: 14%;
            transform: translateY(0px) rotate(0deg);
        }
        50% {
            right: 48%;
            transform: translateY(-8px) rotate(-2deg);
        }
        100% {
            right: 14%;
            transform: translateY(0px) rotate(0deg);
        }
    }
    @keyframes laserSweep {
        0%, 100% { opacity: 0.3; transform: scaleX(0.7); }
        50% { opacity: 0.9; transform: scaleX(1.3); filter: drop-shadow(0 0 12px #38bdf8); }
    }

    .drone-edge-top {
        position: fixed;
        top: 12px;
        right: 18%;
        z-index: 999999;
        animation: patrolTopEdge 15s ease-in-out infinite;
        cursor: pointer;
        display: flex;
        flex-direction: column;
        align-items: center;
        transition: transform 0.25s ease;
    }
    .drone-edge-top:hover {
        transform: scale(1.18) !important;
    }
    .drone-edge-top:hover .drone-speech {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
    .drone-speech {
        opacity: 0;
        transform: translateY(-6px) scale(0.9);
        transition: all 0.25s ease;
        background: #0f172a;
        color: #34d399;
        border: 1px solid #34d399;
        padding: 5px 12px;
        border-radius: 12px;
        font-size: 0.74rem;
        font-weight: 700;
        white-space: nowrap;
        position: absolute;
        bottom: -28px;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
        pointer-events: none;
    }
    .drone-hull {
        width: 68px;
        height: 24px;
        background: linear-gradient(135deg, #1e293b, #090e1a);
        border: 2px solid #34d399;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 18px rgba(16, 185, 129, 0.4);
        position: relative;
    }
    .drone-rotor-l, .drone-rotor-r {
        width: 22px;
        height: 3px;
        background: #38bdf8;
        border-radius: 2px;
        position: absolute;
        top: -4px;
        box-shadow: 0 0 6px #38bdf8;
    }
    .drone-rotor-l { left: -6px; }
    .drone-rotor-r { right: -6px; }
    .drone-scanner-dot {
        width: 10px;
        height: 10px;
        background: #38bdf8;
        border-radius: 50%;
        box-shadow: 0 0 10px #38bdf8;
    }
    .drone-laser-beam {
        width: 48px;
        height: 8px;
        background: linear-gradient(180deg, rgba(56, 189, 248, 0.6), transparent);
        border-radius: 0 0 12px 12px;
        animation: laserSweep 1.8s infinite;
    }

    /* Bot 3: Cute Corner Waving Assistant Mascot (Sparky) */
    @keyframes sparkyBob {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-9px); }
    }
    @keyframes armWaveMotion {
        0%, 100% { transform: rotate(0deg); }
        50% { transform: rotate(26deg); }
    }

    .sparky-mascot {
        position: fixed;
        right: 18px;
        bottom: 20px;
        z-index: 999999;
        animation: sparkyBob 3.6s ease-in-out infinite;
        cursor: pointer;
        display: flex;
        flex-direction: column;
        align-items: flex-end;
    }
    .sparky-mascot:hover .sparky-speech {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
    .sparky-speech {
        opacity: 0;
        transform: translateY(6px) scale(0.9);
        transition: all 0.25s ease;
        background: #0f172a;
        color: #f8fafc;
        border: 1px solid #818cf8;
        padding: 8px 14px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        white-space: nowrap;
        margin-bottom: 6px;
        box-shadow: 0 6px 20px rgba(129, 140, 248, 0.35);
        pointer-events: none;
    }
    .sparky-head-box {
        width: 58px;
        height: 52px;
        background: linear-gradient(145deg, #1e1b4b, #0f172a);
        border: 2px solid #818cf8;
        border-radius: 16px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 20px rgba(129, 140, 248, 0.4);
        position: relative;
    }
    .sparky-antenna {
        width: 4px;
        height: 8px;
        background: #818cf8;
        position: absolute;
        top: -8px;
        border-radius: 2px;
    }
    .sparky-antenna-tip {
        width: 8px;
        height: 8px;
        background: #38bdf8;
        border-radius: 50%;
        position: absolute;
        top: -14px;
        box-shadow: 0 0 10px #38bdf8;
    }
    .sparky-face-screen {
        width: 42px;
        height: 24px;
        background: #020617;
        border-radius: 8px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        border: 1px solid #312e81;
    }
    .sparky-eyes-row {
        display: flex;
        gap: 12px;
    }
    .sparky-eye {
        width: 7px;
        height: 7px;
        background: #38bdf8;
        border-radius: 50%;
        box-shadow: 0 0 8px #38bdf8;
        animation: eyeBlink 4s infinite;
    }
    .sparky-smile {
        width: 14px;
        height: 3px;
        border-bottom: 2px solid #34d399;
        border-radius: 0 0 4px 4px;
        margin-top: 3px;
    }
    .sparky-waving-hand {
        position: absolute;
        right: -12px;
        top: 14px;
        font-size: 1.25rem;
        transform-origin: bottom left;
        animation: armWaveMotion 1.4s ease-in-out infinite;
    }

    /* Floating Action Buttons Dock */
    @keyframes fabHoverDock {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-5px); }
    }
    .floating-fab-dock {
        position: fixed;
        left: 20px;
        bottom: 22px;
        z-index: 999999;
        display: flex;
        flex-direction: row;
        gap: 10px;
        animation: fabHoverDock 4s ease-in-out infinite;
    }
    .fab-icon-btn {
        width: 42px;
        height: 42px;
        border-radius: 50%;
        background: linear-gradient(135deg, #1e293b, #090e1a);
        border: 1.5px solid #38bdf8;
        color: #38bdf8 !important;
        text-decoration: none !important;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.15rem;
        box-shadow: 0 4px 16px rgba(56, 189, 248, 0.3);
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
        cursor: pointer;
    }
    .fab-icon-btn:hover {
        transform: translateY(-4px) scale(1.18);
        border-color: #34d399;
        box-shadow: 0 8px 24px rgba(52, 211, 153, 0.5);
        color: #ffffff !important;
        background: #0284c7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Render Global Moving AI Bots & Interactive Characters
st.markdown(
    """
    <!-- Bot 1: Left Edge Floating Perimeter Patrol Sentinel -->
    <div class="bot-edge-left" title="FinSentinel: Guarding your FAISS vector perimeter">
        <div class="bot-speech">🛡️ FinSentinel: Perimeter Secure!</div>
        <div class="bot-frame">
            <div class="bot-antenna-dot"></div>
            <div class="bot-visor">
                <div class="bot-eye-glow"></div>
                <div class="bot-eye-glow"></div>
            </div>
            <div class="bot-core-light"></div>
        </div>
        <div class="bot-thruster-flame"></div>
    </div>

    <!-- Bot 2: Top Edge Flying Scanning Drone -->
    <div class="drone-edge-top" title="CyberDrone-9: Zero-Hallucination Scanner">
        <div class="drone-speech">🛸 Zero Hallucination Scanner Active!</div>
        <div class="drone-hull">
            <div class="drone-rotor-l"></div>
            <div class="drone-scanner-dot"></div>
            <div class="drone-rotor-r"></div>
        </div>
        <div class="drone-laser-beam"></div>
    </div>

    <!-- Bot 3: Bottom-Right Waving Robot Mascot (Sparky) -->
    <div class="sparky-mascot" title="Sparky: Click or hover to chat!">
        <div class="sparky-speech">👋 <b>Sparky:</b> Press <b>Enter ↵</b> to verify any financial claim!</div>
        <div class="sparky-head-box">
            <div class="sparky-antenna"></div>
            <div class="sparky-antenna-tip"></div>
            <div class="sparky-face-screen">
                <div class="sparky-eyes-row">
                    <div class="sparky-eye"></div>
                    <div class="sparky-eye"></div>
                </div>
                <div class="sparky-smile"></div>
            </div>
            <div class="sparky-waving-hand">👋</div>
        </div>
    </div>

    <!-- Floating Action Buttons Dock -->
    <div class="floating-fab-dock">
        <div class="fab-icon-btn" title="AI Verification Shield Active" style="border-color: #10b981;">🛡️</div>
        <div class="fab-icon-btn" title="Press Enter ↵ to Search" style="border-color: #38bdf8;">⚡</div>
        <div class="fab-icon-btn" title="100% Vector Provenance" style="border-color: #818cf8;">📑</div>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_global_vector_store() -> VectorStoreManager:
    """Caches sentence-transformers model and FAISS vector index."""
    return VectorStoreManager(model_name=EMBEDDING_MODEL_NAME)


def initialize_session_state():
    """Initializes persistent session state variables."""
    if "rag_app" not in st.session_state:
        v_store = get_global_vector_store()
        st.session_state.rag_app = FinVerifyRAG(vector_store=v_store)
    if "ingested_documents" not in st.session_state:
        st.session_state.ingested_documents = []
    if "total_chunks_indexed" not in st.session_state:
        st.session_state.total_chunks_indexed = 0
    if "last_query_result" not in st.session_state:
        st.session_state.last_query_result = None
    if "sample_pdf_created" not in st.session_state:
        st.session_state.sample_pdf_created = False


initialize_session_state()
rag_app: FinVerifyRAG = st.session_state.rag_app
vector_store: VectorStoreManager = rag_app.vector_store

# ==========================================
# SIDEBAR: Ingestion & Model Configuration
# ==========================================
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    # LLM Provider Configuration
    st.subheader("🤖 LLM Provider")
    env_provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    default_provider_idx = 1 if env_provider == "groq" else 0
    provider = st.selectbox(
        "Select Provider",
        options=["Gemini (Google)", "Groq"],
        index=default_provider_idx,
        help="Select whether to use Google Gemini or Groq for grounded generation.",
    )

    is_gemini = "Gemini" in provider
    active_provider = "gemini" if is_gemini else "groq"

    # API Key Input (Hidden for Security & Presentation Safety)
    if is_gemini:
        env_gemini_key = os.getenv("GEMINI_API_KEY", "")
        if env_gemini_key:
            st.success("🔒 Gemini Key: Loaded from .env (Hidden)", icon="🛡️")
            override = st.checkbox("Override Key (Manual Input)", value=False)
            if override:
                gemini_key = st.text_input(
                    "Enter Alternative Gemini Key",
                    type="password",
                    placeholder="Paste new Gemini key...",
                )
            else:
                gemini_key = env_gemini_key
        else:
            gemini_key = st.text_input(
                "Gemini API Key",
                type="password",
                placeholder="Paste Gemini API key...",
                help="Key is masked and never exposed on screen.",
            )
            if gemini_key:
                st.success("🔒 Gemini Key Configured", icon="🛡️")
            else:
                st.info("No key set. Offline grounded extractor will be used.", icon="ℹ️")

        groq_key = ""
        model_name = st.selectbox(
            "Gemini Model",
            options=[
                "gemini-3.8-flash",
                "gemini-3.5-flash",
                "gemini-3.5-flash-lite",
                "gemini-3.7-flash",
                "gemini-flash-latest",
            ],
            index=0,
            help="High-concurrency model. Automatically fails over if a model experiences high demand.",
        )
    else:
        env_groq_key = os.getenv("GROQ_API_KEY", "")
        if env_groq_key:
            st.success("🔒 Groq Key: Loaded from .env (Hidden)", icon="🛡️")
            override = st.checkbox("Override Key (Manual Input)", value=False)
            if override:
                groq_key = st.text_input(
                    "Enter Alternative Groq Key",
                    type="password",
                    placeholder="Paste new Groq key...",
                )
            else:
                groq_key = env_groq_key
        else:
            groq_key = st.text_input(
                "Groq API Key",
                type="password",
                placeholder="gsk_...",
                help="Key is masked and never exposed on screen.",
            )
            if groq_key:
                st.success("🔒 Groq Key Configured", icon="🛡️")
            else:
                st.info("No key set. Offline grounded extractor will be used.", icon="ℹ️")

        gemini_key = ""
        model_name = st.selectbox(
            "Groq Model",
            options=["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
            index=0,
        )

    st.markdown("---")

    # Document Ingestion Section
    st.subheader("📁 Document Ingestion")
    st.caption("Upload one or multiple financial PDF reports.")

    uploaded_files = st.file_uploader(
        "Choose Financial PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload 10-K, 10-Q, quarterly earnings, balance sheets, or audit reports.",
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        process_btn = st.button("🚀 Process PDFs", use_container_width=True, type="primary")
    with col_btn2:
        clear_btn = st.button("🗑️ Reset DB", use_container_width=True)

    if clear_btn:
        vector_store.clear()
        st.session_state.ingested_documents = []
        st.session_state.total_chunks_indexed = 0
        st.session_state.last_query_result = None
        st.rerun()

    if process_btn and uploaded_files:
        with st.spinner("Extracting text and chunking (Size: 700, Overlap: 120)..."):
            files_payload = [(f.name, f.read()) for f in uploaded_files]
            chunks = ingest_multiple_pdfs(
                files_payload,
                chunk_size=700,
                chunk_overlap=120,
            )

        if not chunks:
            st.error("No readable text found in the uploaded PDFs. Please check the files.")
        else:
            with st.spinner(f"Generating local embeddings ({EMBEDDING_MODEL_NAME}) and indexing in FAISS..."):
                added_count = vector_store.add_chunks(chunks)
                st.session_state.ingested_documents.extend([f.name for f in uploaded_files])
                st.session_state.ingested_documents = list(dict.fromkeys(st.session_state.ingested_documents))
                st.session_state.total_chunks_indexed = vector_store.total_chunks
            st.success(f"Indexed {added_count} chunks from {len(uploaded_files)} PDF(s) into FAISS!")
            st.rerun()

    st.markdown("---")
    # Quick Sample Report Loader for Hackathon Demonstrations
    st.subheader("⚡ Quick Demo Data")
    load_sample_btn = st.button("📥 Load Sample 10-Q Filing", use_container_width=True)
    if load_sample_btn:
        sample_doc_name = "Apex_Financial_Holdings_Q3_2024.pdf"
        sample_text_p1 = (
            "Apex Financial Holdings Inc. - Form 10-Q Quarterly Report for Period Ended September 30, 2024.\n"
            "Item 1. Financial Statements. For the third quarter ended September 30, 2024, Apex Financial Holdings reported "
            "total net revenue of $142.8 million, representing an increase of 14.2% compared to $125.0 million in Q3 2023. "
            "Net income for the quarter reached $38.4 million, or $1.42 per diluted common share, compared to $31.1 million, "
            "or $1.18 per diluted share in the prior year period. Total operating expenses stood at $94.6 million, primarily "
            "driven by strategic investments in AI-driven compliance technology and algorithmic risk underwriting platforms."
        )
        sample_text_p2 = (
            "Apex Financial Holdings Inc. - Capital Resources & Liquidity Analysis.\n"
            "As of September 30, 2024, cash, cash equivalents, and marketable securities totaled $412.5 million, compared with "
            "$380.0 million as of December 31, 2023. Total long-term debt was $185.0 million with a weighted average interest "
            "rate of 4.85%. On October 15, 2024, the Board of Directors approved a quarterly cash dividend of $0.35 per share, "
            "payable on November 20, 2024 to shareholders of record as of November 1, 2024. Capital expenditures for the nine "
            "months ended September 30, 2024 totaled $18.2 million, focused on cybersecurity infrastructure upgrades."
        )
        sample_text_p3 = (
            "Apex Financial Holdings Inc. - Risk Factors and Regulatory Compliance.\n"
            "Our operations are subject to risks including fluctuating interest rates, foreign exchange rate volatility, and "
            "strict regulatory compliance oversight under Basel III and Dodd-Frank frameworks. We maintain a Tier 1 Common Equity "
            "Capital Ratio of 13.8%, well above regulatory minimum requirements. The company does not currently operate in "
            "cryptocurrency staking or retail consumer mortgage lending."
        )

        # Ingest directly as chunks
        from ingest import extract_chunks_from_pdf
        # Create minimal PDF bytes using pypdf writer
        from pypdf import PdfWriter
        writer = PdfWriter()
        # Add blank pages with annotation or construct DocumentChunk directly
        c_p1 = DocumentChunk(
            chunk_id="apex_q3_2024_p1_c1",
            document_name=sample_doc_name,
            page_number=1,
            content=sample_text_p1,
        )
        c_p2 = DocumentChunk(
            chunk_id="apex_q3_2024_p2_c1",
            document_name=sample_doc_name,
            page_number=2,
            content=sample_text_p2,
        )
        c_p3 = DocumentChunk(
            chunk_id="apex_q3_2024_p3_c1",
            document_name=sample_doc_name,
            page_number=3,
            content=sample_text_p3,
        )
        vector_store.add_chunks([c_p1, c_p2, c_p3])
        st.session_state.ingested_documents.append(sample_doc_name)
        st.session_state.ingested_documents = list(dict.fromkeys(st.session_state.ingested_documents))
        st.session_state.total_chunks_indexed = vector_store.total_chunks
        st.success("Loaded Apex Financial Q3 2024 10-Q sample report!")
        st.rerun()

    st.markdown("---")
    st.caption("🔒 **Security & Grounding**: Context chunks are strictly bounded. Zero out-of-document generation.")


# ==========================================
# MAIN INTERFACE
# ==========================================
# Header with Animated Gradient Branding
st.markdown(
    """
    <div style="margin-bottom: 16px;">
        <div class="hero-title">⚖️ FinVerify AI</div>
        <div class="hero-subtitle">Hallucination-Resistant Financial Document Intelligence</div>
        <div class="hero-tagline">
            Built for financial analysts, auditors, and compliance officers. 
            Every response is mathematically verified against local FAISS vector evidence with deterministic provenance.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Top KPI Status Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f'<div class="metric-card"><div class="metric-val">{len(st.session_state.ingested_documents)}</div>'
        f'<div class="metric-label">Active Documents</div></div>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f'<div class="metric-card"><div class="metric-val">{st.session_state.total_chunks_indexed}</div>'
        f'<div class="metric-label">Vector Chunks in FAISS</div></div>',
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f'<div class="metric-card"><div class="metric-val">{EMBEDDING_DIM}d</div>'
        f'<div class="metric-label">Local MiniLM Embeddings</div></div>',
        unsafe_allow_html=True,
    )
with col4:
    guardrail_status = "Active 🛡️" if st.session_state.total_chunks_indexed > 0 else "Ready"
    st.markdown(
        f'<div class="metric-card"><div class="metric-val" style="color: #10b981;">{guardrail_status}</div>'
        f'<div class="metric-label">Guardrail Engine</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Tabs
tab_query, tab_docs, tab_guardrails = st.tabs(["🔍 Ask & Verify", "📄 Document Explorer", "🛡️ Guardrail Architecture"])

# -------------------------------------------------------------
# TAB 1: Query & Verification
# -------------------------------------------------------------
with tab_query:
    st.markdown("#### Enter Financial Question")

    # Sample query chips for instant hackathon demonstration
    st.caption("💡 Quick Test Prompts (Click to test):")
    chip_col1, chip_col2, chip_col3 = st.columns(3)
    
    if "input_query_text" not in st.session_state:
        st.session_state.input_query_text = ""
    
    chip_clicked = False
    with chip_col1:
        if st.button("📊 Answerable: Total Revenue in Q3 2024?", use_container_width=True):
            st.session_state.input_query_text = "What was the total net revenue reported for Q3 2024, and how much did it increase compared to Q3 2023?"
            chip_clicked = True
    with chip_col2:
        if st.button("💰 Answerable: Quarterly Dividend?", use_container_width=True):
            st.session_state.input_query_text = "What was the quarterly cash dividend approved per share, and when is it payable?"
            chip_clicked = True
    with chip_col3:
        if st.button("❌ Unanswerable: Forecast for 2029?", use_container_width=True):
            st.session_state.input_query_text = "What are the projected revenue figures and market share targets for the year 2029?"
            chip_clicked = True

    # Enter key submission enabled via st.form
    with st.form(key="financial_qa_form", clear_on_submit=False):
        user_query = st.text_input(
            "Enter Financial Question (Press Enter ↵ to Answer):",
            value=st.session_state.input_query_text,
            placeholder="Type your question here and press Enter ↵ (or click Verify & Answer)...",
            label_visibility="collapsed",
        )
        col_q1, col_q2 = st.columns([1.2, 3.8])
        with col_q1:
            submit_btn = st.form_submit_button("🔍 Verify & Answer (↵)", type="primary", use_container_width=True)

    # Determine query to run: from form submission (Enter key / Submit button) or instant chip click
    target_query = ""
    if chip_clicked:
        target_query = st.session_state.input_query_text
    elif submit_btn and user_query.strip():
        target_query = user_query.strip()
        st.session_state.input_query_text = target_query

    if target_query:
        if vector_store.total_chunks == 0:
            st.warning("⚠️ No documents uploaded yet! Please upload PDF files in the sidebar or click 'Load Sample 10-Q Filing'.")
        else:
            with st.spinner("Retrieving top 5 FAISS chunks & evaluating evidence sufficiency..."):
                llm = LLMGenerator(
                    provider=active_provider,
                    gemini_api_key=gemini_key,
                    groq_api_key=groq_key,
                    model_name=model_name,
                )
                result = rag_app.query(target_query, llm_generator=llm, top_k=5)
                st.session_state.last_query_result = result

    # Display Query Results in a Wide, Horizontal Futuristic Dashboard
    if st.session_state.last_query_result:
        res = st.session_state.last_query_result
        conf_level = res["confidence_level"]
        is_suff = res["is_sufficient"]

        st.markdown("---")
        
        # 1. Verified Answer Section (Full Width, Glowing Cyber Container)
        st.markdown("### 📝 Verified Financial Answer")
        
        # Question & Badge Header
        badge_html = {
            "High": '<span class="confidence-badge-high">🟢 High Confidence</span>',
            "Medium": '<span class="confidence-badge-medium">🟡 Medium Confidence</span>',
            "Insufficient Evidence": '<span class="confidence-badge-insufficient">🔴 Insufficient Evidence</span>',
        }.get(conf_level, '<span class="confidence-badge-medium">Verified</span>')

        st.markdown(
            f"""
            <div style="background: linear-gradient(90deg, #131d35 0%, #1e293b 100%); border-radius: 10px; padding: 14px 18px; margin-bottom: 14px; border: 1px solid #334155;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-size: 0.82rem; color: #94a3b8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;">Query Audit Target</span>
                    {badge_html}
                </div>
                <div style="font-size: 1.15rem; color: #f8fafc; font-weight: 600;">
                    "{res['question']}"
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Answer Box with Cyber Glow
        st.markdown('<div class="answer-container">', unsafe_allow_html=True)
        if not is_suff or res["answer"] == EXACT_FALLBACK_ANSWER:
            st.markdown(
                f"""
                <div style="padding: 6px 0;">
                    <div style="color: #ef4444; font-weight: 800; font-size: 1.15rem; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;">
                        🛡️ Strict Grounding Guardrail Triggered
                    </div>
                    <div style="background: #1c1917; border-left: 5px solid #ef4444; padding: 16px 20px; border-radius: 8px; font-family: monospace; font-size: 1.1rem; color: #fecaca; margin-bottom: 12px;">
                        {EXACT_FALLBACK_ANSWER}
                    </div>
                    <div style="font-size: 0.92rem; color: #94a3b8;">
                        <b>Guardrail Diagnosis:</b> {res['guardrail_explanation']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(res["answer"])
        st.markdown('</div>', unsafe_allow_html=True)

        # Horizontal Metrics Strip (4 Micro-Cards)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Top Chunk Match", f"{res['top_similarity']:.2f}")
        with m2:
            st.metric("Keyword Overlap", f"{res['keyword_overlap_ratio']:.1%}")
        with m3:
            st.metric("Composite Confidence", f"{res['confidence_score']:.2f}")
        with m4:
            status_str = "Verified Grounded ✅" if is_suff else "Strict Fallback 🛡️"
            st.metric("Audit Status", status_str)

        if res.get("matched_keywords"):
            st.caption(f"**Verified Financial Keywords:** {', '.join([f'`{k}`' for k in res['matched_keywords']])}")

        st.markdown("<br>", unsafe_allow_html=True)

        # 2. Grounded Citations Section (Wide Horizontal Cards Grid)
        st.markdown("### 📑 Grounded Citations & Audit Trail")
        st.caption("Deterministic provenance linking every verified claim to its exact source document, page, and chunk.")

        cite_list = res.get("citations", [])
        if not cite_list:
            st.info("No citations retrieved.")
        else:
            cols_per_row = 2
            for row_start in range(0, len(cite_list), cols_per_row):
                row_cites = cite_list[row_start : row_start + cols_per_row]
                grid_cols = st.columns(cols_per_row, gap="medium")
                for c_idx, cite in enumerate(row_cites):
                    # Strip any newlines and multi-spaces to guarantee 100% continuous horizontal prose
                    raw_txt = str(cite.get('content_preview', ''))
                    clean_snippet = " ".join(raw_txt.split())
                    with grid_cols[c_idx]:
                        st.markdown(
                            f"""
                            <div class="citation-grid-card">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                    <span class="citation-tag" style="background: linear-gradient(90deg, #0284c7, #0369a1); color: #ffffff; font-weight: bold; border: none;">
                                        ⚡ Relevance: {cite['similarity_pct']}
                                    </span>
                                    <span style="font-size: 0.8rem; color: #38bdf8; font-weight: 600;">
                                        Chunk #{row_start + c_idx + 1}
                                    </span>
                                </div>
                                <div style="margin-bottom: 8px; display: flex; flex-wrap: wrap; gap: 6px;">
                                    <span class="citation-tag">📄 <b>{cite['document_name']}</b></span>
                                    <span class="citation-tag">Page <b>{cite['page_number']}</b></span>
                                    <span class="citation-tag" style="font-family: monospace; color: #94a3b8;">ID: {cite['chunk_id']}</span>
                                </div>
                                <div class="citation-snippet">
                                    "{clean_snippet}"
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        st.markdown("<br>", unsafe_allow_html=True)

        # 3. Expandable Full Vector Chunks Section
        with st.expander("🔍 Inspect Full Retrieved Context Chunks (Raw FAISS Text)", expanded=False):
            st.caption("Exact text passages retrieved from local FAISS vectors and evaluated by the guardrail:")
            for i, chunk_dict in enumerate(res["retrieved_chunks"]):
                st.markdown(f"**Chunk {i+1}** — `{chunk_dict['chunk_id']}` (Page {chunk_dict['page_number']} of `{chunk_dict['document_name']}`)")
                st.text_area(
                    f"Chunk {i+1} Raw Text",
                    value=chunk_dict["content"],
                    height=100,
                    key=f"chunk_txt_{i}",
                    disabled=True,
                )


# -------------------------------------------------------------
# TAB 2: Document Explorer
# -------------------------------------------------------------
with tab_docs:
    st.markdown("#### 📚 Indexed Documents & Chunks")
    if vector_store.total_chunks == 0:
        st.info("No documents indexed yet. Use the sidebar to upload financial PDFs.")
    else:
        st.write(f"**Total Indexed Documents:** {len(vector_store.unique_documents)}")
        st.write(f"**Total Vector Embeddings:** {vector_store.total_chunks}")

        for doc_name in vector_store.unique_documents:
            doc_chunks = [c for c in vector_store.chunks if c.document_name == doc_name]
            with st.expander(f"📄 {doc_name} ({len(doc_chunks)} chunks)", expanded=False):
                for c in doc_chunks:
                    st.markdown(f"- **Chunk ID:** `{c.chunk_id}` | **Page:** {c.page_number} | **Length:** {len(c.content)} chars")
                    st.caption(c.content[:250] + ("..." if len(c.content) > 250 else ""))


# -------------------------------------------------------------
# TAB 3: Guardrail Architecture
# -------------------------------------------------------------
with tab_guardrails:
    st.markdown("#### 🛡️ FinVerify AI Anti-Hallucination Architecture")
    st.markdown(
        """
        FinVerify AI employs a **multi-tiered verification pipeline** specifically calibrated for high-stakes financial auditing:

        1. **Deterministic PDF Ingestion & Provenance Tracking (`ingest.py`)**:
           - Extracts text page-by-page using `pypdf`.
           - Preserves immutable provenance metadata: `document_name`, `page_number`, and unique `chunk_id`.
           - Recursive chunking with 700 character boundaries and 120 character overlap to avoid semantic fragmentation.

        2. **Local Vector Search (`rag.py`)**:
           - Encodes text locally via `sentence-transformers` (`all-MiniLM-L6-v2`, 384 dimensions).
           - Executes fast Cosine Similarity retrieval using `FAISS` (`IndexFlatIP` with normalized vectors).
           - Gathers exactly the top 5 most relevant excerpts.

        3. **Dual-Layer Evidence Sufficiency Guardrail (`guardrail.py`)**:
           - **Layer 1: Semantic Cosine Thresholds**: Rejects retrieval results where top chunk similarity < 0.38.
           - **Layer 2: Lexical Entity & Financial Keyword Overlap**: Verifies that specific financial terms, figures, and entities from the query appear in the retrieved context.
           - **Layer 3: Exact Fallback Enforcement**: If evidence is insufficient, immediately emits:
             `"I don't know based on the provided documents."`

        4. **Grounded Generation Step**:
           - LLM (Gemini or Groq) receives ONLY retrieved context with strict system instructions prohibiting outside knowledge or extrapolation.
           - If the model generates hesitation or fails factual grounding, the guardrail intercepts and substitutes the exact fallback string.
        """
    )
