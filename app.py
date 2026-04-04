
import streamlit as st
import os
import operator
import json
import re
from typing import TypedDict, List, Annotated
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ==============================================================================
# CONFIGURATION
# ==============================================================================
API_KEY = "sk-proj......"  # <-- Remplacez par votre clé

st.set_page_config(
    page_title="Auditeur Clinique Expert v4",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================================================================
# CSS PERSONNALISÉ
# ==============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif; }

.step-bar { display: flex; gap: 8px; margin-bottom: 2rem; }
.step-dot {
    width: 32px; height: 32px; border-radius: 50%;
    background: #e2e8f0; display: flex; align-items: center;
    justify-content: center; font-size: 13px; font-weight: 600;
    color: #94a3b8; transition: all 0.3s;
}
.step-dot.active { background: #0f4c81; color: white; box-shadow: 0 0 0 4px rgba(15,76,129,0.15); }
.step-dot.done   { background: #10b981; color: white; }
.step-line { flex: 1; height: 2px; margin-top: 15px; background: #e2e8f0; }
.step-line.done  { background: #10b981; }

.question-card {
    background: white; border: 1px solid #e2e8f0;
    border-left: 4px solid #0f4c81; border-radius: 12px;
    padding: 1.5rem 2rem; margin-bottom: 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.question-label {
    font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 1px; color: #0f4c81; margin-bottom: 6px;
}
.question-title  { font-family: 'DM Serif Display', serif; font-size: 18px; color: #1e293b; margin-bottom: 4px; }
.question-hint   { font-size: 13px; color: #94a3b8; margin-bottom: 12px; }

.recap-box {
    background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 12px;
    padding: 1.5rem; font-size: 14px; white-space: pre-wrap; color: #0c4a6e;
}

.verdict-safe   { background:#dcfce7; border:2px solid #16a34a; border-radius:12px; padding:1rem 1.5rem; color:#14532d; font-weight:600; font-size:18px; }
.verdict-unsafe { background:#fee2e2; border:2px solid #dc2626; border-radius:12px; padding:1rem 1.5rem; color:#7f1d1d; font-weight:600; font-size:18px; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MÉMOIRE LANGGRAPH
# ==============================================================================
memory = MemorySaver()

@st.cache_resource
def get_compiled_app():
    return build_workflow()

# ==============================================================================
# ÉTAT GLOBAL DU SYSTÈME
# ==============================================================================
class AgentState(TypedDict):
    messages:       Annotated[List[BaseMessage], operator.add]
    protocol_data:  str
    is_safe:        bool
    revision_count: int
    shared_memory:  dict   # mémoire partagée inter-agents
    agent_plans:    dict   # plan + actions par agent

# ==============================================================================
# OUTILS (Tools)
# ==============================================================================

@tool
def calculate_dosage(weight_kg: float, dose_per_kg: float, frequency_per_day: int) -> dict:
    """Calcule la dose journalière totale et par prise à partir du poids et de la posologie."""
    dose_per_intake = weight_kg * dose_per_kg
    total_daily     = dose_per_intake * frequency_per_day
    return {
        "dose_per_intake_mg": round(dose_per_intake, 2),
        "total_daily_mg":     round(total_daily, 2),
        "frequency":          frequency_per_day,
    }

@tool
def lookup_drug_interactions(drug_a: str, drug_b: str) -> dict:
    """Vérifie les interactions médicamenteuses connues entre deux molécules."""
    known = {
        ("warfarine",    "aspirine"):    {"level": "MAJEURE",     "effect": "Risque hémorragique accru"},
        ("warfarine",    "ibuprofène"):  {"level": "MAJEURE",     "effect": "Risque hémorragique accru + anticoagulation potentialisée"},
        ("metformine",   "alcool"):      {"level": "MODÉRÉE",     "effect": "Risque d'acidose lactique"},
        ("ibuprofène",   "lithium"):     {"level": "MAJEURE",     "effect": "Toxicité du lithium augmentée"},
        ("amoxicilline", "metformine"):  {"level": "MINEURE",     "effect": "Interaction faible, surveiller"},
        ("aspirine",     "warfarine"):   {"level": "MAJEURE",     "effect": "Double anticoagulation — risque hémorragique fatal"},
        ("amiodarone",   "warfarine"):   {"level": "MAJEURE",     "effect": "Augmentation INR — risque hémorragique sévère"},
        ("ibuprofène",   "warfarine"):   {"level": "MAJEURE",     "effect": "Potentialisation anticoagulant + risque GI"},
        ("spironolactone","insuffisance rénale"): {"level": "MAJEURE", "effect": "Hyperkaliémie sévère potentiellement fatale"},
    }
    key     = (drug_a.lower(), drug_b.lower())
    key_rev = (drug_b.lower(), drug_a.lower())
    result  = known.get(key) or known.get(key_rev)
    return result or {"level": "AUCUNE CONNUE", "effect": f"Pas d'interaction documentée entre {drug_a} et {drug_b}"}

@tool
def check_contraindications(drug: str, condition: str) -> dict:
    """Vérifie les contre-indications connues d'un médicament pour une condition médicale."""
    ci = {
        ("metformine",    "insuffisance rénale"):  "CONTRE-INDIQUÉ ABSOLU — DFG < 30 : risque d'acidose lactique mortelle",
        ("ibuprofène",    "insuffisance rénale"):  "CONTRE-INDIQUÉ — Aggravation de la fonction rénale, rétention hydrosodée",
        ("ibuprofène",    "ulcère gastrique"):     "CONTRE-INDIQUÉ — Risque de perforation et hémorragie digestive",
        ("ibuprofène",    "insuffisance cardiaque"):"CONTRE-INDIQUÉ — Rétention hydrosodée, décompensation cardiaque",
        ("corticoïdes",   "diabète"):              "PRÉCAUTION MAJEURE — Hyperglycémie, déséquilibre diabétique",
        ("bêtabloquants", "asthme"):               "CONTRE-INDIQUÉ — Bronchospasme sévère, crise asthmatique",
        ("aspirine",      "ulcère gastrique"):     "CONTRE-INDIQUÉ — Risque hémorragique digestif majeur",
        ("amiodarone",    "insuffisance cardiaque"):"PRÉCAUTION — Dépression myocardique, surveiller FEVG",
        ("metformine",    "insuffisance cardiaque"):"PRÉCAUTION — Risque d'acidose lactique en cas de bas débit",
    }
    verdict = ci.get((drug.lower(), condition.lower()), "Pas de contre-indication connue pour cette association.")
    return {"drug": drug, "condition": condition, "verdict": verdict}

@tool
def pharmacovigilance_search(drug_name: str) -> dict:
    """Recherche les alertes de pharmacovigilance pour un médicament donné."""
    alerts = {
        "metformine":    ["Acidose lactique rare mais mortelle", "Arrêt obligatoire avant injection iodée", "Contre-indiqué si DFG < 30"],
        "warfarine":     ["Marge thérapeutique étroite — INR requis régulièrement", "Très nombreuses interactions alimentaires et médicamenteuses", "Risque hémorragique majeur si surdosage"],
        "amiodarone":    ["Toxicité pulmonaire (pneumopathie)", "Dysthyroïdie (hypo et hyperthyroïdie)", "Hépatotoxicité", "Demi-vie très longue (40-55 jours)", "Nombreuses interactions dont warfarine"],
        "ibuprofène":    ["Risque cardiovasculaire à long terme", "Toxicité rénale aiguë", "Hémorragie digestive", "Contre-indiqué insuffisance rénale/cardiaque/ulcère"],
        "aspirine":      ["Risque hémorragique digestif", "Interactions anticoagulants", "Syndrome de Reye chez l'enfant"],
        "spironolactone":["Hyperkaliémie — surveillance kaliémie obligatoire", "Risque accru si IR ou IEC/ARA2 associés"],
        "bisoprolol":    ["Bradycardie", "Bronchospasme si asthme/BPCO", "Masque les signes d'hypoglycémie"],
        "furosémide":    ["Déshydratation", "Hypokaliémie", "Hyperuricémie"],
    }
    found = alerts.get(drug_name.lower(), ["Aucune alerte de pharmacovigilance majeure répertoriée."])
    return {"drug": drug_name, "alerts": found, "source": "PharmacoDB v3.1 (simulé)"}

@tool
def evidence_score(protocol_description: str) -> dict:
    """Attribue un score de niveau de preuve EBM au protocole décrit."""
    score = 3
    desc  = protocol_description.lower()
    if any(k in desc for k in ["randomisé", "essai contrôlé", "rct", "méta-analyse"]):
        score = 1
    elif any(k in desc for k in ["cohorte", "prospective", "guideline"]):
        score = 2
    elif any(k in desc for k in ["cas clinique", "avis expert", "série de cas"]):
        score = 4
    levels = {
        1: "A — Preuve forte (RCT/méta-analyse)",
        2: "B — Preuve modérée (cohorte/guideline)",
        3: "C — Consensus d'experts",
        4: "D — Données limitées (cas cliniques)"
    }
    return {"score": score, "label": levels[score]}

# Mapping outils → agents
AGENT_TOOLS = {
    "step_back":         [check_contraindications, evidence_score],
    "analyzer_cot":      [calculate_dosage, check_contraindications],
    "explorer_tot":      [lookup_drug_interactions, pharmacovigilance_search],
    "verifier_react":    [pharmacovigilance_search, lookup_drug_interactions, calculate_dosage],
    "critic_correction": [evidence_score, check_contraindications],
}

def run_tool_safe(tool_fn, **kwargs):
    try:    return tool_fn.invoke(kwargs)
    except Exception as e: return {"error": str(e)}

# ==============================================================================
# LLM
# ==============================================================================
def get_llm():
    return ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=API_KEY)

# ==============================================================================
# HELPERS MÉMOIRE PARTAGÉE + PLANS
# ==============================================================================
def update_memory(state: AgentState, key: str, value) -> dict:
    mem      = dict(state.get("shared_memory", {}))
    mem[key] = value
    return mem

def update_plan(state: AgentState, agent_name: str, plan: list, actions: list) -> dict:
    plans              = dict(state.get("agent_plans", {}))
    plans[agent_name]  = {
        "plan":      plan,
        "actions":   actions,
        "timestamp": datetime.now().isoformat()
    }
    return plans

# ==============================================================================
# ✅ HELPER VERDICT — Détection robuste SAFE / UNSAFE
# ==============================================================================
def detect_verdict(content: str) -> bool:
    """
    Retourne True (SAFE) uniquement si le texte contient explicitement VERDICT: SAFE
    ET ne contient pas VERDICT: UNSAFE ni UNSAFE seul.
    Par précaution médicale, tout doute → UNSAFE.
    """
    upper = content.upper()
    # UNSAFE a la priorité absolue
    if "VERDICT: UNSAFE" in upper:
        return False
    if "UNSAFE" in upper:
        return False
    # SAFE validé seulement si mentionné explicitement avec le mot-clé VERDICT
    if "VERDICT: SAFE" in upper:
        return True
    # Aucun verdict clair → UNSAFE par précaution médicale
    return False

# ==============================================================================
# AGENT 1 — STEP-BACK
# ==============================================================================
def step_back_agent(state: AgentState):
    llm  = get_llm()
    plan = [
        "1. Identifier les classes médicamenteuses du protocole",
        "2. Vérifier les contre-indications majeures via check_contraindications",
        "3. Évaluer le niveau de preuve EBM avec evidence_score",
        "4. Synthétiser les principes généraux de vigilance",
    ]
    actions_log = [
        {"tool": "check_contraindications",
         "result": run_tool_safe(check_contraindications, drug="ibuprofène", condition="insuffisance rénale")},
        {"tool": "check_contraindications_2",
         "result": run_tool_safe(check_contraindications, drug="metformine", condition="insuffisance rénale")},
        {"tool": "check_contraindications_3",
         "result": run_tool_safe(check_contraindications, drug="bêtabloquants", condition="asthme")},
        {"tool": "evidence_score",
         "result": run_tool_safe(evidence_score, protocol_description=state["protocol_data"])},
    ]
    system   = SystemMessage(content=(
        "Tu es un pharmacologue expert en sécurité médicamenteuse. "
        "Tu appliques le Step-Back Prompting : avant d'analyser le cas particulier, "
        "tu rappelles les principes fondamentaux et les contre-indications majeures "
        "des classes de médicaments concernées. Sois exhaustif et rigoureux."
    ))
    prompt   = HumanMessage(content=(
        f"ÉTAPE 0 — [STEP-BACK ABSTRACTION]\n\n"
        f"PLAN D'EXÉCUTION :\n" + "\n".join(plan) + "\n\n"
        f"RÉSULTATS DES OUTILS :\n{json.dumps(actions_log, ensure_ascii=False, indent=2)}\n\n"
        f"PROTOCOLE À AUDITER :\n{state['protocol_data']}\n\n"
        "Sur la base des résultats d'outils et des principes généraux de pharmacologie, "
        "synthétise TOUS les points de vigilance fondamentaux à garder en tête pour l'audit complet."
    ))
    response = llm.invoke([system, prompt])
    return {
        "messages":      [response],
        "shared_memory": update_memory(state, "principles", response.content),
        "agent_plans":   update_plan(state, "step_back", plan, actions_log),
    }

# ==============================================================================
# AGENT 2 — CHAIN OF THOUGHT (CoT)
# ==============================================================================
def analyzer_cot_agent(state: AgentState):
    llm   = get_llm()
    poids = float(st.session_state.get("intake", {}).get("poids", 70))
    plan  = [
        "1. Extraire les paramètres de dosage (poids, dose/kg, fréquence)",
        "2. Calculer la dose journalière via calculate_dosage",
        "3. Vérifier les contre-indications spécifiques au patient",
        "4. Identifier toute anomalie de dosage ou incompatibilité",
    ]
    actions_log = [
        {"tool": "calculate_dosage",
         "result": run_tool_safe(calculate_dosage, weight_kg=poids, dose_per_kg=10, frequency_per_day=3)},
        {"tool": "check_contraindications_ibuprofene_IR",
         "result": run_tool_safe(check_contraindications, drug="ibuprofène", condition="insuffisance rénale")},
        {"tool": "check_contraindications_ibuprofene_ulcere",
         "result": run_tool_safe(check_contraindications, drug="ibuprofène", condition="ulcère gastrique")},
        {"tool": "check_contraindications_ibuprofene_IC",
         "result": run_tool_safe(check_contraindications, drug="ibuprofène", condition="insuffisance cardiaque")},
        {"tool": "check_contraindications_metformine_IR",
         "result": run_tool_safe(check_contraindications, drug="metformine", condition="insuffisance rénale")},
    ]
    principles = state.get("shared_memory", {}).get("principles", "")
    system     = SystemMessage(content=(
        "Tu es un médecin interniste expert en pharmaco-clinique. "
        "Tu appliques la Chain of Thought : chaque étape de raisonnement doit être "
        "explicitement écrite avant de conclure. Tu ne sautes aucune étape."
    ))
    prompt     = HumanMessage(content=(
        f"ÉTAPE 1 — [CHAIN OF THOUGHT]\n\n"
        f"PLAN :\n" + "\n".join(plan) + "\n\n"
        f"MÉMOIRE (Principes Step-Back) :\n{principles}\n\n"
        f"RÉSULTATS OUTILS :\n{json.dumps(actions_log, ensure_ascii=False, indent=2)}\n\n"
        f"PROTOCOLE :\n{state['protocol_data']}\n\n"
        "Raisonne étape par étape. Pour chaque médicament prescrit : "
        "vérifie le dosage, identifie les contre-indications avec les antécédents du patient, "
        "et conclus sur la cohérence pharmacologique. Sois TRÈS précis sur les anomalies détectées."
    ))
    response   = llm.invoke([system, prompt])
    mem        = update_memory(state, "cot_analysis", response.content)
    return {
        "messages":      [response],
        "shared_memory": mem,
        "agent_plans":   update_plan(state, "analyzer_cot", plan, actions_log),
    }

# ==============================================================================
# AGENT 3 — TREE OF THOUGHTS (ToT)
# ==============================================================================
def explorer_tot_agent(state: AgentState):
    llm  = get_llm()
    plan = [
        "1. Vérifier toutes les interactions médicamenteuses du protocole",
        "2. Consulter la pharmacovigilance pour chaque molécule critique",
        "3. Générer 3 branches de stratégie thérapeutique",
        "4. Évaluer le score bénéfice/risque de chaque branche",
    ]
    actions_log = [
        {"tool": "interaction_warfarine_aspirine",
         "result": run_tool_safe(lookup_drug_interactions, drug_a="warfarine", drug_b="aspirine")},
        {"tool": "interaction_warfarine_amiodarone",
         "result": run_tool_safe(lookup_drug_interactions, drug_a="amiodarone", drug_b="warfarine")},
        {"tool": "interaction_warfarine_ibuprofene",
         "result": run_tool_safe(lookup_drug_interactions, drug_a="warfarine", drug_b="ibuprofène")},
        {"tool": "pharmacovigilance_amiodarone",
         "result": run_tool_safe(pharmacovigilance_search, drug_name="amiodarone")},
        {"tool": "pharmacovigilance_warfarine",
         "result": run_tool_safe(pharmacovigilance_search, drug_name="warfarine")},
        {"tool": "pharmacovigilance_ibuprofene",
         "result": run_tool_safe(pharmacovigilance_search, drug_name="ibuprofène")},
    ]
    mem    = state.get("shared_memory", {})
    system = SystemMessage(content=(
        "Tu es un expert en arbres de décisions médicales (Tree of Thoughts). "
        "Tu génères plusieurs stratégies thérapeutiques alternatives et évalues "
        "chacune objectivement avec un score bénéfice/risque basé sur les données disponibles."
    ))
    prompt = HumanMessage(content=(
        f"ÉTAPE 2 — [TREE OF THOUGHTS]\n\n"
        f"PLAN :\n" + "\n".join(plan) + "\n\n"
        f"MÉMOIRE :\n"
        f"Principes : {mem.get('principles', '')}\n"
        f"Analyse CoT : {mem.get('cot_analysis', '')}\n\n"
        f"RÉSULTATS OUTILS :\n{json.dumps(actions_log, ensure_ascii=False, indent=2)}\n\n"
        "Génère exactement 3 branches :\n"
        "  🌿 Branche A — Validation du protocole tel quel\n"
        "  🌿 Branche B — Modification mineure (ajustement doses/substitution partielle)\n"
        "  🌿 Branche C — Substitution totale (protocole alternatif complet)\n\n"
        "Pour chaque branche : arguments POUR, arguments CONTRE, score risque/bénéfice (1-10), "
        "puis identifie la branche optimale avec justification claire."
    ))
    response = llm.invoke([system, prompt])
    return {
        "messages":      [response],
        "shared_memory": update_memory(state, "tot_branches", response.content),
        "agent_plans":   update_plan(state, "explorer_tot", plan, actions_log),
    }

# ==============================================================================
# AGENT 4 — ReAct
# ==============================================================================
def verifier_react_agent(state: AgentState):
    llm   = get_llm()
    poids = float(st.session_state.get("intake", {}).get("poids", 65))
    plan  = [
        "1. Re-consulter la pharmacovigilance pour toutes les molécules critiques",
        "2. Recalculer le dosage avec le poids réel du patient",
        "3. Vérifier les interactions résiduelles de la branche choisie en ToT",
        "4. Produire un rapport de validation factuel avec verdict",
    ]
    actions_log = [
        {"tool": "pharmacovigilance_metformine",
         "result": run_tool_safe(pharmacovigilance_search, drug_name="metformine")},
        {"tool": "pharmacovigilance_spironolactone",
         "result": run_tool_safe(pharmacovigilance_search, drug_name="spironolactone")},
        {"tool": "pharmacovigilance_bisoprolol",
         "result": run_tool_safe(pharmacovigilance_search, drug_name="bisoprolol")},
        {"tool": "interaction_aspirine_warfarine",
         "result": run_tool_safe(lookup_drug_interactions, drug_a="aspirine", drug_b="warfarine")},
        {"tool": "calculate_dosage_patient",
         "result": run_tool_safe(calculate_dosage, weight_kg=poids, dose_per_kg=12, frequency_per_day=3)},
        {"tool": "ci_bétabloquants_asthme",
         "result": run_tool_safe(check_contraindications, drug="bêtabloquants", condition="asthme")},
    ]
    mem    = state.get("shared_memory", {})
    system = SystemMessage(content=(
        "Tu es un pharmacien hospitalier expert en pharmacovigilance. "
        "Tu appliques le paradigme ReAct : pour chaque décision tu justifies ton raisonnement (Reason), "
        "tu exécutes une action (Act), et tu observes le résultat (Observe) avant de continuer."
    ))
    prompt = HumanMessage(content=(
        f"ÉTAPE 3 — [ReAct]\n\n"
        f"PLAN :\n" + "\n".join(plan) + "\n\n"
        f"MÉMOIRE :\n"
        f"Principes : {mem.get('principles', '')}\n"
        f"Branches ToT : {mem.get('tot_branches', '')}\n\n"
        f"RÉSULTATS OUTILS :\n{json.dumps(actions_log, ensure_ascii=False, indent=2)}\n\n"
        "Applique le cycle Reason→Act→Observe pour valider ou invalider la stratégie optimale "
        "identifiée à l'Étape 2. Conclus par un verdict factuel parmi : "
        "VALIDÉ / MODIFIÉ / REJETÉ, avec justification basée sur les alertes de pharmacovigilance "
        "et les contre-indications détectées."
    ))
    response = llm.invoke([system, prompt])
    return {
        "messages":      [response],
        "shared_memory": update_memory(state, "react_verdict", response.content),
        "agent_plans":   update_plan(state, "verifier_react", plan, actions_log),
    }

# ==============================================================================
# AGENT 5 — SELF-REFLEXION (CRITIQUE) — ✅ BUG CORRIGÉ
# ==============================================================================
def critic_self_correction_agent(state: AgentState):
    llm      = get_llm()
    revision = state.get("revision_count", 0)
    plan     = [
        "1. Réévaluer le niveau de preuve EBM du protocole final proposé",
        "2. Vérifier la cohérence logique entre Step-Back, CoT, ToT et ReAct",
        "3. Identifier toute erreur de raisonnement ou biais cognitif",
        "4. Produire un verdict SAFE ou UNSAFE clair avec score de confiance",
    ]
    actions_log = [
        {"tool": "evidence_score_final",
         "result": run_tool_safe(evidence_score, protocol_description=state["protocol_data"])},
        {"tool": "ci_corticoides_diabete",
         "result": run_tool_safe(check_contraindications, drug="corticoïdes", condition="diabète")},
        {"tool": "ci_aspirine_ulcere",
         "result": run_tool_safe(check_contraindications, drug="aspirine", condition="ulcère gastrique")},
        {"tool": "ci_metformine_IC",
         "result": run_tool_safe(check_contraindications, drug="metformine", condition="insuffisance cardiaque")},
    ]
    mem    = state.get("shared_memory", {})
    system = SystemMessage(content=(
        "Tu es un auditeur médical externe STRICT et indépendant, spécialisé en gestion des risques cliniques. "
        "Tu es le dernier rempart avant la validation d'un protocole. "
        "Ta règle absolue : UN SEUL problème critique non résolu = VERDICT: UNSAFE. "
        "Tu DOIS terminer ta réponse par exactement l'une de ces deux lignes :\n"
        "VERDICT: SAFE\n"
        "VERDICT: UNSAFE\n"
        "Aucune autre formulation n'est acceptée."
    ))
    prompt = HumanMessage(content=(
        f"ÉTAPE 4 — [SELF-REFLEXION & MÉTACOGNITION] (Révision n°{revision + 1})\n\n"
        f"PLAN :\n" + "\n".join(plan) + "\n\n"
        f"MÉMOIRE COMPLÈTE DE L'AUDIT :\n"
        f"━━ Principes (Step-Back) :\n{mem.get('principles', '')}\n\n"
        f"━━ Analyse CoT :\n{mem.get('cot_analysis', '')}\n\n"
        f"━━ Branches ToT :\n{mem.get('tot_branches', '')}\n\n"
        f"━━ Verdict ReAct :\n{mem.get('react_verdict', '')}\n\n"
        f"RÉSULTATS OUTILS :\n{json.dumps(actions_log, ensure_ascii=False, indent=2)}\n\n"
        "CONSIGNE STRICTE :\n"
        "1. Liste TOUS les problèmes détectés dans ce protocole (contre-indications, "
        "interactions, surdosages, incohérences).\n"
        "2. Pour chaque problème : indique le niveau de gravité (CRITIQUE / MODÉRÉ / MINEUR).\n"
        "3. Identifie les erreurs éventuelles dans le raisonnement des agents précédents.\n"
        "4. Si AU MOINS UN problème CRITIQUE existe → tu DOIS écrire VERDICT: UNSAFE\n"
        "5. Si AUCUN problème critique → tu DOIS écrire VERDICT: SAFE\n"
        "6. Donne un score de confiance global (0-100%).\n"
        "7. DERNIÈRE LIGNE de ta réponse = le verdict (VERDICT: SAFE ou VERDICT: UNSAFE)."
    ))

    response = llm.invoke([system, prompt])

    # ✅ DÉTECTION ROBUSTE — UNSAFE a la priorité absolue
    is_safe = detect_verdict(response.content)

    mem2 = update_memory(state, "critic_verdict", response.content)
    return {
        "messages":       [response],
        "shared_memory":  mem2,
        "agent_plans":    update_plan(state, "critic_correction", plan, actions_log),
        "is_safe":        is_safe,
        "revision_count": revision + 1,
    }

# ==============================================================================
# AGENT FINAL — SYNTHÈSE
# ==============================================================================
def finalizer_agent(state: AgentState):
    llm    = get_llm()
    mem    = state.get("shared_memory", {})
    is_safe = state.get("is_safe", False)
    verdict_str = "✅ APPROUVÉ" if is_safe else "🚨 REJETÉ — PROTOCOLE DANGEREUX"

    system = SystemMessage(content=(
        "Tu es le rédacteur du rapport médical final d'audit pharmaceutique. "
        "Tu synthétises les conclusions de tous les agents en un document structuré, "
        "professionnel et directement actionnable par le clinicien. "
        "Sois précis, clair, et hiérarchise les informations par ordre de criticité."
    ))
    prompt = HumanMessage(content=(
        f"SYNTHÈSE FINALE — RAPPORT D'AUDIT\n\n"
        f"VERDICT GLOBAL : {verdict_str}\n\n"
        f"Protocole audité :\n{state['protocol_data']}\n\n"
        f"Mémoire consolidée :\n"
        f"• Principes (Step-Back) : {mem.get('principles', '')}\n"
        f"• Analyse CoT : {mem.get('cot_analysis', '')}\n"
        f"• Stratégies ToT : {mem.get('tot_branches', '')}\n"
        f"• Verdict ReAct : {mem.get('react_verdict', '')}\n"
        f"• Audit critique : {mem.get('critic_verdict', '')}\n\n"
        "Rédige un rapport structuré avec les sections suivantes :\n"
        "  🔴 PROBLÈMES CRITIQUES (liste exhaustive, par ordre de gravité)\n"
        "  ✅ DÉCISION FINALE (APPROUVÉ / MODIFIÉ / REJETÉ) avec justification\n"
        "  ⚠️ POINTS DE VIGILANCE RESTANTS\n"
        "  💊 RECOMMANDATIONS PRATIQUES pour le clinicien (alternatives thérapeutiques)\n"
        "  📊 NIVEAU DE PREUVE EBM global\n"
        "  📋 CONDUITE À TENIR immédiate"
    ))
    response = llm.invoke([system, prompt])
    return {"messages": [response]}

# ==============================================================================
# CONSTRUCTION DU GRAPHE LANGGRAPH
# ==============================================================================
def build_workflow():
    workflow = StateGraph(AgentState)

    workflow.add_node("step_back",         step_back_agent)
    workflow.add_node("analyzer_cot",      analyzer_cot_agent)
    workflow.add_node("explorer_tot",      explorer_tot_agent)
    workflow.add_node("verifier_react",    verifier_react_agent)
    workflow.add_node("critic_correction", critic_self_correction_agent)
    workflow.add_node("finalizer",         finalizer_agent)

    workflow.set_entry_point("step_back")
    workflow.add_edge("step_back",      "analyzer_cot")
    workflow.add_edge("analyzer_cot",   "explorer_tot")
    workflow.add_edge("explorer_tot",   "verifier_react")
    workflow.add_edge("verifier_react", "critic_correction")

    def route_final(state):
        # ✅ UNSAFE prioritaire — on relance seulement si pas encore révisé
        if state["is_safe"] or state["revision_count"] >= 2:
            return "finalizer"
        return "step_back"

    workflow.add_conditional_edges(
        "critic_correction",
        route_final,
        {"finalizer": "finalizer", "step_back": "step_back"}
    )
    workflow.add_edge("finalizer", END)
    return workflow.compile(checkpointer=memory)

# ==============================================================================
# SESSION STATE
# ==============================================================================
if "step"       not in st.session_state: st.session_state.step       = 0
if "intake"     not in st.session_state: st.session_state.intake     = {}
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

app = get_compiled_app()

# ==============================================================================
# FORMULAIRE — 7 QUESTIONS CLINIQUES
# ==============================================================================
QUESTIONS = [
    {
        "section": "Identification du patient",
        "label":   "Question 1 / 7",
        "title":   "Quel est le profil du patient ?",
        "hint":    "Âge, sexe, poids approximatif",
        "fields":  [
            {"key": "age",   "type": "number", "label": "Âge (années)",   "min": 0,   "max": 120, "default": 45},
            {"key": "sexe",  "type": "select", "label": "Sexe biologique", "options": ["Homme", "Femme", "Autre"]},
            {"key": "poids", "type": "number", "label": "Poids (kg)",      "min": 1,   "max": 300, "default": 70},
        ]
    },
    {
        "section": "Antécédents médicaux",
        "label":   "Question 2 / 7",
        "title":   "Quelles sont les pathologies connues ?",
        "hint":    "Maladies chroniques, antécédents chirurgicaux significatifs",
        "fields":  [
            {"key": "pathologies", "type": "multiselect", "label": "Pathologies",
             "options": ["Diabète type 2", "Insuffisance rénale", "Hypertension", "Insuffisance cardiaque",
                         "Asthme/BPCO", "Ulcère gastrique", "Épilepsie", "Fibrillation auriculaire",
                         "Thrombopénie", "Aucune connue", "Autre"]},
            {"key": "pathologies_autres", "type": "text", "label": "Préciser si 'Autre'", "default": ""},
        ]
    },
    {
        "section": "Allergies & Intolérances",
        "label":   "Question 3 / 7",
        "title":   "Y a-t-il des allergies ou intolérances médicamenteuses connues ?",
        "hint":    "Préciser la molécule et le type de réaction si possible",
        "fields":  [
            {"key": "allergies_flag",   "type": "select", "label": "Allergies connues ?", "options": ["Non", "Oui"]},
            {"key": "allergies_detail", "type": "text",   "label": "Si oui, préciser",    "default": ""},
        ]
    },
    {
        "section": "Traitements en cours",
        "label":   "Question 4 / 7",
        "title":   "Quels médicaments sont actuellement pris par le patient ?",
        "hint":    "Inclure automédication, compléments alimentaires, traitements chroniques",
        "fields":  [
            {"key": "traitements_actuels", "type": "textarea", "label": "Traitement(s) en cours", "default": ""},
        ]
    },
    {
        "section": "Prescription à auditer",
        "label":   "Question 5 / 7",
        "title":   "Quelle est la prescription à évaluer ?",
        "hint":    "Molécule(s), dosage, voie d'administration, durée prévue",
        "fields":  [
            {"key": "prescription",     "type": "textarea", "label": "Prescription complète",                  "default": ""},
            {"key": "voie_admin",        "type": "select",   "label": "Voie d'administration principale",
             "options": ["Orale", "Intraveineuse", "Sous-cutanée", "Intramusculaire", "Topique", "Autre"]},
            {"key": "duree_traitement", "type": "text",     "label": "Durée prévue (ex: 7 jours, chronique)", "default": ""},
        ]
    },
    {
        "section": "Contexte clinique",
        "label":   "Question 6 / 7",
        "title":   "Quel est le contexte clinique de cette prescription ?",
        "hint":    "Motif, urgence, objectif thérapeutique",
        "fields":  [
            {"key": "motif",   "type": "textarea", "label": "Motif de prescription", "default": ""},
            {"key": "urgence", "type": "select",   "label": "Niveau d'urgence",
             "options": ["Électif (programmé)", "Semi-urgent (< 48h)", "Urgent (< 6h)", "Extrême urgence"]},
        ]
    },
    {
        "section": "Paramètres biologiques",
        "label":   "Question 7 / 7",
        "title":   "Avez-vous des résultats biologiques récents ?",
        "hint":    "Fonction rénale, hépatique, NFS, INR si disponibles",
        "fields":  [
            {"key": "bio_flag",   "type": "select",   "label": "Bilan biologique disponible ?", "options": ["Non disponible", "Disponible"]},
            {"key": "creatinine", "type": "text",     "label": "Créatinine (µmol/L) ou DFG (mL/min)", "default": ""},
            {"key": "bio_autres", "type": "textarea", "label": "Autres valeurs pertinentes",           "default": ""},
        ]
    },
]

# ==============================================================================
# HELPERS UI
# ==============================================================================
def render_progress_bar(current_step):
    total = len(QUESTIONS)
    html  = '<div class="step-bar">'
    for i in range(total):
        if i < current_step:   cls, icon = "done",   "✓"
        elif i == current_step: cls, icon = "active", str(i + 1)
        else:                   cls, icon = "",        str(i + 1)
        html += f'<div class="step-dot {cls}">{icon}</div>'
        if i < total - 1:
            line_cls = "done" if i < current_step else ""
            html += f'<div class="step-line {line_cls}"></div>'
    html += '</div>'
    return html

def render_field(field, intake):
    key = field["key"]
    val = intake.get(key, field.get("default", ""))
    ft  = field["type"]
    if ft == "number":
        return st.number_input(field["label"], min_value=field.get("min", 0), max_value=field.get("max", 999),
                               value=int(val) if val else field.get("default", 0), key=f"field_{key}")
    elif ft == "select":
        opts = field["options"]
        idx  = opts.index(val) if val in opts else 0
        return st.selectbox(field["label"], opts, index=idx, key=f"field_{key}")
    elif ft == "multiselect":
        default_val = val if isinstance(val, list) else []
        return st.multiselect(field["label"], field["options"], default=default_val, key=f"field_{key}")
    elif ft == "textarea":
        return st.text_area(field["label"], value=val, key=f"field_{key}", height=100)
    else:
        return st.text_input(field["label"], value=val, key=f"field_{key}")

def build_protocol_text(intake):
    lines = ["=== DOSSIER PATIENT ==="]
    lines.append(f"Patient : {intake.get('age','?')} ans, {intake.get('sexe','?')}, {intake.get('poids','?')} kg")
    pathos = intake.get("pathologies", [])
    detail = intake.get("pathologies_autres", "")
    lines.append(f"Antécédents : {', '.join(pathos) if pathos else 'Aucun'}" + (f" ({detail})" if detail else ""))
    if intake.get("allergies_flag") == "Oui":
        lines.append(f"Allergies : {intake.get('allergies_detail', 'Non précisé')}")
    else:
        lines.append("Allergies : Aucune connue")
    ttt = intake.get("traitements_actuels", "").strip()
    lines.append(f"Traitements en cours : {ttt if ttt else 'Aucun'}")
    lines.append("")
    lines.append("=== PRESCRIPTION À AUDITER ===")
    lines.append(f"Prescription : {intake.get('prescription', 'Non renseigné')}")
    lines.append(f"Voie : {intake.get('voie_admin','?')} | Durée : {intake.get('duree_traitement','?')}")
    lines.append(f"Motif : {intake.get('motif','Non renseigné')} | Urgence : {intake.get('urgence','?')}")
    if intake.get("bio_flag") == "Disponible":
        bio_parts = []
        if intake.get("creatinine"):  bio_parts.append(f"Créatinine/DFG : {intake['creatinine']}")
        if intake.get("bio_autres"):  bio_parts.append(intake["bio_autres"])
        lines.append(f"Biologie : {' | '.join(bio_parts) if bio_parts else 'Non renseignée'}")
    else:
        lines.append("Biologie : Non disponible")
    return "\n".join(lines)

# ==============================================================================
# RENDU PRINCIPAL STREAMLIT
# ==============================================================================
st.markdown("# 🩺 Auditeur Clinique Expert v4")
st.markdown("**Pipeline :** Step-Back → CoT → ToT → ReAct → Reflexion &nbsp;|&nbsp; 🛠️ Tools &nbsp;•&nbsp; 🧠 Memory &nbsp;•&nbsp; 📋 Planning")
st.divider()

current = st.session_state.step

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 1 — QUESTIONNAIRE
# ──────────────────────────────────────────────────────────────────────────────
if current < len(QUESTIONS):
    q = QUESTIONS[current]
    st.markdown(render_progress_bar(current), unsafe_allow_html=True)
    st.markdown(f"""
    <div class="question-card">
        <div class="question-label">{q['label']} — {q['section']}</div>
        <div class="question-title">{q['title']}</div>
        <div class="question-hint">💡 {q['hint']}</div>
    </div>
    """, unsafe_allow_html=True)

    field_values = {}
    for field in q["fields"]:
        field_values[field["key"]] = render_field(field, st.session_state.intake)

    col_prev, col_next = st.columns([1, 3])
    with col_prev:
        if current > 0:
            if st.button("← Précédent", use_container_width=True):
                st.session_state.step -= 1
                st.rerun()
    with col_next:
        label_next = "Suivant →" if current < len(QUESTIONS) - 1 else "Voir le récapitulatif →"
        if st.button(label_next, type="primary", use_container_width=True):
            for field in q["fields"]:
                st.session_state.intake[field["key"]] = field_values[field["key"]]
            st.session_state.step += 1
            st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2 — RÉCAPITULATIF
# ──────────────────────────────────────────────────────────────────────────────
elif current == len(QUESTIONS):
    protocol_text = build_protocol_text(st.session_state.intake)
    st.markdown(render_progress_bar(len(QUESTIONS)), unsafe_allow_html=True)
    st.subheader("📋 Récapitulatif du dossier patient")
    st.markdown(f'<div class="recap-box">{protocol_text}</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Tout est correct ?** Le pipeline multi-agents va s'exécuter sur ce dossier.")
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("← Modifier", use_container_width=True):
            st.session_state.step = len(QUESTIONS) - 1
            st.rerun()
    with col2:
        if st.button("🔄 Recommencer", use_container_width=True):
            st.session_state.step  = 0
            st.session_state.intake = {}
            st.rerun()
    with col3:
        if st.button("🚀 Lancer l'Audit Expert", type="primary", use_container_width=True):
            st.session_state.protocol_text = protocol_text
            st.session_state.step = len(QUESTIONS) + 1
            st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 3 — EXÉCUTION & RÉSULTATS
# ──────────────────────────────────────────────────────────────────────────────
elif current == len(QUESTIONS) + 1:
    protocol_text = st.session_state.get("protocol_text", "")
    initial_state: AgentState = {
        "messages":       [],
        "protocol_data":  protocol_text,
        "is_safe":        False,
        "revision_count": 0,
        "shared_memory":  {},
        "agent_plans":    {},
    }
    config = {"configurable": {"thread_id": st.session_state.session_id}}

    with st.spinner("🔄 Pipeline multi-agents en cours d'exécution..."):
        result = app.invoke(initial_state, config=config)

    # Sidebar — dossier patient
    with st.sidebar:
        st.markdown("### 📁 Dossier Patient")
        st.code(protocol_text, language=None)
        st.markdown(f"**Session :** `{st.session_state.session_id}`")
        if st.button("🔄 Nouvel audit"):
            st.session_state.step   = 0
            st.session_state.intake = {}
            st.rerun()

    # ── Verdict principal ──
    is_safe = result.get("is_safe", False)
    if is_safe:
        st.markdown('<div class="verdict-safe">✅ VERDICT FINAL : PROTOCOLE SAFE — Approuvé pour administration</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="verdict-unsafe">🚨 VERDICT FINAL : PROTOCOLE UNSAFE — Ne pas administrer en l\'état</div>', unsafe_allow_html=True)

    st.markdown("")

    # ── Rapport final ──
    st.subheader("📋 Rapport d'Audit Complet")
    if is_safe:
        st.success(result["messages"][-1].content)
    else:
        st.error(result["messages"][-1].content)

    # ── Métriques ──
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Révisions effectuées", result.get("revision_count", 1))
    m2.metric("Verdict sécurité",     "✅ SAFE" if is_safe else "🚨 UNSAFE")
    m3.metric("Agents mobilisés",     5)
    m4.metric("Appels outils",        sum(len(p.get("actions", [])) for p in result.get("agent_plans", {}).values()))

    # ── Détail par agent ──
    st.divider()
    st.subheader("🛠️ Détail du Raisonnement Multi-Agents")
    agent_labels = [
        ("📌 Agent 1 — Step-Back (Principes généraux)",   "step_back"),
        ("🧠 Agent 2 — CoT (Analyse détaillée + Calculs)", "analyzer_cot"),
        ("🌳 Agent 3 — ToT (Arbre de stratégies)",         "explorer_tot"),
        ("🔍 Agent 4 — ReAct (Pharmacovigilance)",         "verifier_react"),
        ("🛡️ Agent 5 — Reflexion (Audit critique)",       "critic_correction"),
    ]
    for i, (label, agent_key) in enumerate(agent_labels):
        with st.expander(label):
            tabs = st.tabs(["💬 Raisonnement", "📋 Plan", "🔧 Outils", "🧠 Mémoire"])
            with tabs[0]:
                if i < len(result["messages"]) - 1:
                    st.markdown(result["messages"][i].content)
            with tabs[1]:
                plan_data = result.get("agent_plans", {}).get(agent_key, {})
                if plan_data:
                    for s in plan_data.get("plan", []):
                        st.markdown(f"- {s}")
                    st.caption(f"Timestamp : {plan_data.get('timestamp', '')}")
            with tabs[2]:
                for action in result.get("agent_plans", {}).get(agent_key, {}).get("actions", []):
                    st.json(action)
            with tabs[3]:
                mem = result.get("shared_memory", {})
                st.json({
                    k: (v[:300] + "..." if isinstance(v, str) and len(v) > 300 else v)
                    for k, v in mem.items()
                })
