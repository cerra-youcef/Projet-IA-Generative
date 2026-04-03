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

# --- OUTIL 3 : Contre-indications ---
@tool
def check_contraindications(drug: str, condition: str) -> dict:
    """Vérifie les contre-indications d'un médicament pour une condition."""
    ci = {
        ("metformine","insuffisance rénale"): "CONTRE-INDIQUÉ — Acidose lactique",
        ("ibuprofène","insuffisance rénale"): "CONTRE-INDIQUÉ — Aggravation rénale",
        ("ibuprofène","ulcère gastrique"):    "CONTRE-INDIQUÉ — Risque hémorragique",
        ("corticoïdes","diabète"):            "PRÉCAUTION — Hyperglycémie",
        ("bêtabloquants","asthme"):           "CONTRE-INDIQUÉ — Bronchospasme",
    }
    verdict = ci.get((drug.lower(), condition.lower()), "Pas de contre-indication connue.")
    return {"drug": drug, "condition": condition, "verdict": verdict}

!git add app.py
!git commit -m "feat(tools): add check_contraindications tool"
!git push origin main

# --- OUTIL 3 : Contre-indications ---
@tool
def check_contraindications(drug: str, condition: str) -> dict:
    """Vérifie les contre-indications d'un médicament pour une condition."""
    ci = {
        ("metformine","insuffisance rénale"): "CONTRE-INDIQUÉ — Acidose lactique",
        ("ibuprofène","insuffisance rénale"): "CONTRE-INDIQUÉ — Aggravation rénale",
        ("ibuprofène","ulcère gastrique"):    "CONTRE-INDIQUÉ — Risque hémorragique",
        ("corticoïdes","diabète"):            "PRÉCAUTION — Hyperglycémie",
        ("bêtabloquants","asthme"):           "CONTRE-INDIQUÉ — Bronchospasme",
    }
    verdict = ci.get((drug.lower(), condition.lower()), "Pas de contre-indication connue.")
    return {"drug": drug, "condition": condition, "verdict": verdict}


# --- OUTIL 4 : Pharmacovigilance ---
@tool
def pharmacovigilance_search(drug_name: str) -> dict:
    """Recherche dans la base de pharmacovigilance simulée."""
    alerts = {
        "metformine": ["Acidose lactique rare","Arrêt avant injection iodée"],
        "warfarine":  ["Marge étroite — INR requis","Interactions alimentaires"],
        "amiodarone": ["Toxicité pulmonaire","Dysthyroïdie","Hépatotoxicité"],
        "ibuprofène": ["Risque cardiovasculaire","Toxicité rénale"],
    }
    return {
        "drug":    drug_name,
        "alerts":  alerts.get(drug_name.lower(), ["Aucune alerte majeure."]),
        "source":  "PharmacoDB v3.1 (simulé)"
    }

# --- OUTIL 5 : Score EBM ---
@tool
def evidence_score(protocol_description: str) -> dict:
    """Attribue un niveau de preuve EBM au protocole."""
    desc = protocol_description.lower()
    if any(k in desc for k in ["randomisé","rct","méta-analyse"]):    score = 1
    elif any(k in desc for k in ["cohorte","prospective","guideline"]): score = 2
    elif any(k in desc for k in ["cas clinique","avis expert"]):        score = 4
    else:                                                                score = 3
    levels = {1:"A — Preuve forte",2:"B — Preuve modérée",3:"C — Consensus",4:"D — Données limitées"}
    return {"score": score, "label": levels[score]}

# Mapping outils → agents
AGENT_TOOLS = {
    "step_back":        [check_contraindications, evidence_score],
    "analyzer_cot":     [calculate_dosage, check_contraindications],
    "explorer_tot":     [lookup_drug_interactions, pharmacovigilance_search],
    "verifier_react":   [pharmacovigilance_search, lookup_drug_interactions, calculate_dosage],
    "critic_correction":[evidence_score, check_contraindications],
}

def run_tool_safe(tool_fn, **kwargs):
    try:    return tool_fn.invoke(kwargs)
    except Exception as e: return {"error": str(e)}
