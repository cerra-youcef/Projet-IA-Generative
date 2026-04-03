import streamlit as st
import os, operator, json
from typing import TypedDict, List, Annotated
from datetime import datetime

# --- CONFIGURATION ---
API_KEY = "sk-proj-..."  # Remplacer par votre clé

st.set_page_config(page_title="Auditeur Clinique Expert v3", layout="wide")

# --- MÉMOIRE LANGGRAPH ---
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

memory = MemorySaver()

@st.cache_resource
def get_compiled_app():
    return build_workflow()


# --- ÉTAT GLOBAL ---
class AgentState(TypedDict):
    messages:        Annotated[List[BaseMessage], operator.add]
    protocol_data:   str
    is_safe:         bool
    revision_count:  int
    shared_memory:   dict   # mémoire partagée inter-agents
    agent_plans:     dict   # plan + actions par agent

# --- OUTIL 1 : Calcul de dosage ---
@tool
def calculate_dosage(weight_kg: float, dose_per_kg: float, frequency_per_day: int) -> dict:
    """Calcule la dose journalière totale et par prise."""
    dose_per_intake = weight_kg * dose_per_kg
    return {
        "dose_per_intake_mg": round(dose_per_intake, 2),
        "total_daily_mg":     round(dose_per_intake * frequency_per_day, 2),
        "frequency":          frequency_per_day,
    }

# --- OUTIL 2 : Interactions médicamenteuses ---
@tool
def lookup_drug_interactions(drug_a: str, drug_b: str) -> dict:
    """Vérifie les interactions médicamenteuses connues."""
    known = {
        ("warfarine","aspirine"):      {"level":"MAJEURE","effect":"Risque hémorragique accru"},
        ("metformine","alcool"):       {"level":"MODÉRÉE","effect":"Risque acidose lactique"},
        ("ibuprofène","lithium"):      {"level":"MAJEURE","effect":"Toxicité lithium augmentée"},
        ("amoxicilline","metformine"): {"level":"MINEURE","effect":"Interaction faible"},
    }
    key = (drug_a.lower(), drug_b.lower())
    result = known.get(key) or known.get((drug_b.lower(), drug_a.lower()))
    return result or {"level":"AUCUNE CONNUE","effect":f"Pas d'interaction documentée entre {drug_a} et {drug_b}"}

# --- OUTIL 2 : Interactions médicamenteuses ---
@tool
def lookup_drug_interactions(drug_a: str, drug_b: str) -> dict:
    """Vérifie les interactions médicamenteuses connues."""
    known = {
        ("warfarine","aspirine"):      {"level":"MAJEURE","effect":"Risque hémorragique accru"},
        ("metformine","alcool"):       {"level":"MODÉRÉE","effect":"Risque acidose lactique"},
        ("ibuprofène","lithium"):      {"level":"MAJEURE","effect":"Toxicité lithium augmentée"},
        ("amoxicilline","metformine"): {"level":"MINEURE","effect":"Interaction faible"},
    }
    key = (drug_a.lower(), drug_b.lower())
    result = known.get(key) or known.get((drug_b.lower(), drug_a.lower()))
    return result or {"level":"AUCUNE CONNUE","effect":f"Pas d'interaction documentée entre {drug_a} et {drug_b}"}

# --- OUTIL 2 : Interactions médicamenteuses ---
@tool
def lookup_drug_interactions(drug_a: str, drug_b: str) -> dict:
    """Vérifie les interactions médicamenteuses connues."""
    known = {
        ("warfarine","aspirine"):      {"level":"MAJEURE","effect":"Risque hémorragique accru"},
        ("metformine","alcool"):       {"level":"MODÉRÉE","effect":"Risque acidose lactique"},
        ("ibuprofène","lithium"):      {"level":"MAJEURE","effect":"Toxicité lithium augmentée"},
        ("amoxicilline","metformine"): {"level":"MINEURE","effect":"Interaction faible"},
    }
    key = (drug_a.lower(), drug_b.lower())
    result = known.get(key) or known.get((drug_b.lower(), drug_a.lower()))
    return result or {"level":"AUCUNE CONNUE","effect":f"Pas d'interaction documentée entre {drug_a} et {drug_b}"}
