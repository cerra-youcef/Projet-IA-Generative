import streamlit as st
import os
import operator
from typing import TypedDict, List, Annotated, Union
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END

# --- CONFIGURATION ---
API_KEY = "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"

st.set_page_config(page_title="Auditeur Haute-Fiabilité Ultra", layout="wide")
st.title("🩺 Auditeur Clinique - Système Expert Multi-Agents")
st.markdown("🚀 **Pipeline de Raisonnement de Pointe : Step-Back ➔ CoT ➔ ToT ➔ ReAct ➔ Reflexion**")

# --- ÉTAT DU SYSTÈME ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    protocol_data: str
    is_safe: bool
    revision_count: int

def get_llm():
    return ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=API_KEY)

# --- NOUVEL AGENT : STEP-BACK PROMPTING (Recherche Avancée) ---
def step_back_agent(state: AgentState):
    """
    Technique : Step-Back Prompting.
    L'agent s'abstrait du cas particulier pour rappeler les principes généraux.
    """
    llm = get_llm()
    prompt = f"ÉTAPE 0 : [STEP-BACK ABSTRACTION]\nOublie un instant les détails de ce cas. Quels sont les principes généraux de sécurité et les contre-indications majeures concernant les classes de médicaments mentionnées dans cette prescription ?\nCAS : {state['protocol_data']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [response]}

# --- AGENTS EXISTANTS AMÉLIORÉS ---
def analyzer_cot_agent(state: AgentState):
    llm = get_llm()
    history = state['messages'][-1].content # Utilise le step-back comme base
    prompt = f"ÉTAPE 1 : [CHAIN OF THOUGHT]\nEn t'appuyant sur les principes généraux identifiés, analyse maintenant les détails spécifiques (dosages, poids, âge) de ce cas :\n{state['protocol_data']}\n\nPrincipes de rappel : {history}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [response]}

def explorer_tot_agent(state: AgentState):
    llm = get_llm()
    prompt = f"ÉTAPE 2 : [TREE OF THOUGHTS]\nExplore 3 stratégies : 1. Validation 2. Modification mineure 3. Substitution totale. Évalue la balance bénéfice/risque de chaque branche."
    response = llm.invoke(state['messages'] + [HumanMessage(content=prompt)])
    return {"messages": [response]}

def verifier_react_agent(state: AgentState):
    llm = get_llm()
    prompt = "ÉTAPE 3 : [ReAct]\nRecherche interne : Simule une consultation des bases de données de pharmacovigilance pour confirmer la stratégie choisie."
    response = llm.invoke(state['messages'] + [HumanMessage(content=prompt)])
    return {"messages": [response]}

def critic_self_correction_agent(state: AgentState):
    llm = get_llm()
    prompt = "ÉTAPE 4 : [SELF-REFLECTION & METACOGNITION]\nAgis comme un auditeur externe. Vérifie si les agents précédents n'ont pas fait d'erreur de logique. Réponds 'SAFE' uniquement si le protocole respecte les principes de l'étape 0 et les calculs de l'étape 1."
    response = llm.invoke(state['messages'] + [HumanMessage(content=prompt)])
    is_safe = "SAFE" in response.content.upper()
    return {"messages": [response], "is_safe": is_safe, "revision_count": state.get('revision_count', 0) + 1}

def finalizer_agent(state: AgentState):
    llm = get_llm()
    prompt = "Synthèse finale : Présente la décision médicale finale de manière structurée et justifiée."
    response = llm.invoke(state['messages'] + [HumanMessage(content=prompt)])
    return {"messages": [response]}

# --- CONSTRUCTION DU GRAPHE ---
workflow = StateGraph(AgentState)

workflow.add_node("step_back", step_back_agent) # Nouveau départ
workflow.add_node("analyzer_cot", analyzer_cot_agent)
workflow.add_node("explorer_tot", explorer_tot_agent)
workflow.add_node("verifier_react", verifier_react_agent)
workflow.add_node("critic_correction", critic_self_correction_agent)
workflow.add_node("finalizer", finalizer_agent)

workflow.set_entry_point("step_back")
workflow.add_edge("step_back", "analyzer_cot")
workflow.add_edge("analyzer_cot", "explorer_tot")
workflow.add_edge("explorer_tot", "verifier_react")
workflow.add_edge("verifier_react", "critic_correction")

def route_final(state):
    if state["is_safe"] or state["revision_count"] >= 2:
        return "finalizer"
    return "step_back" # On recommence au début pour une correction totale

workflow.add_conditional_edges("critic_correction", route_final, {
    "finalizer": "finalizer",
    "step_back": "step_back"
})

workflow.add_edge("finalizer", END)
app = workflow.compile()

# --- UI STREAMLIT ---
protocol_input = st.text_area("📄 Entrez le protocole (Audit Haute-Fiabilité) :", height=150)

if st.button("🚀 Lancer l'Audit Expert"):
    if protocol_input:
        with st.spinner("Exécution du pipeline multi-méthodes..."):
            result = app.invoke({"protocol_data": protocol_input, "messages": [], "is_safe": False, "revision_count": 0})

            st.subheader("📋 Rapport d'Audit Final")
            st.success(result["messages"][-1].content)

            with st.expander("🛠️ Analyse du Raisonnement Multi-Agents"):
                labels = ["📌 Step-Back (Principes)", "🧠 CoT (Analyse)", "🌳 ToT (Arbre de choix)", "🔍 ReAct (Vérification)", "🛡️ Reflexion (Audit)"]
                for i, msg in enumerate(result["messages"][:-1]):
                    label = labels[i % len(labels)]
                    st.markdown(f"**{label}**")
                    st.info(msg.content)
