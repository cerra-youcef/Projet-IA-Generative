import streamlit as st
import os
import operator
from typing import TypedDict, List, Annotated, Union
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END

# --- CONFIGURATION SÉCURISÉE ---
API_KEY = "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"

st.set_page_config(page_title="Pipeline Audit Expert", layout="wide")
st.title("🩺 Auditeur Clinique - Pipeline Expert")
st.markdown("🚀 **Flux : CoT ➔ ToT ➔ ReAct ➔ Self-Correction**")

# --- ÉTAT DE L'AGENT ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    protocol_data: str
    is_safe: bool
    revision_count: int

# --- AGENTS DE RAISONNEMENT ---
def get_llm():
    return ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=API_KEY)

# 1. AGENT CoT (Chain of Thought)
def analyzer_cot_agent(state: AgentState):
    llm = get_llm()
    prompt = f"ÉTAPE 1 : [CHAIN OF THOUGHT]\nDécompose ce protocole médical. Identifie les composants clés et les risques de base :\n{state['protocol_data']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [response]}

# 2. AGENT ToT (Tree of Thoughts)
def explorer_tot_agent(state: AgentState):
    llm = get_llm()
    last_msg = state['messages'][-1].content
    prompt = f"ÉTAPE 2 : [TREE OF THOUGHTS]\nBasé sur l'analyse précédente, génère 3 options de traitement possibles (ex: maintenir, modifier dose, ou changer de molécule). Évalue chaque option et choisis la plus sûre (élagage).\nANALYSE PRÉCÉDENTE : {last_msg}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [response]}

# 3. AGENT ReAct (Reason + Act)
def verifier_react_agent(state: AgentState):
    llm = get_llm()
    history = state['messages']
    prompt = "ÉTAPE 3 : [ReAct]\nAction : Simule une vérification dans la base de données Vidal/HAS pour prouver la validité de l'option choisie.\nObservation : Quelles sont les preuves médicales ?\nRéponse : Formule une recommandation basée sur ces preuves."
    response = llm.invoke(history + [HumanMessage(content=prompt)])
    return {"messages": [response]}

# 4. AGENT SELF-CORRECTION (Reflection)
def critic_self_correction_agent(state: AgentState):
    llm = get_llm()
    history = state['messages']
    prompt = "ÉTAPE 4 : [SELF-CORRECTION]\nExamine tout le processus précédent (CoT, ToT, ReAct). Cherche une faille de dernière minute. Si tout est parfait, réponds 'SAFE'. Sinon, propose une correction finale."
    response = llm.invoke(history + [HumanMessage(content=prompt)])
    is_safe = "SAFE" in response.content.upper()
    return {"messages": [response], "is_safe": is_safe, "revision_count": state.get('revision_count', 0) + 1}

# 5. FINALISEUR
def finalizer_agent(state: AgentState):
    llm = get_llm()
    prompt = "Produis la synthèse médicale finale officielle et sécurisée pour le médecin."
    response = llm.invoke(state['messages'] + [HumanMessage(content=prompt)])
    return {"messages": [response]}

# --- CONSTRUCTION DU GRAPHE EXPERT ---
workflow = StateGraph(AgentState)

workflow.add_node("analyzer_cot", analyzer_cot_agent)
workflow.add_node("explorer_tot", explorer_tot_agent)
workflow.add_node("verifier_react", verifier_react_agent)
workflow.add_node("critic_correction", critic_self_correction_agent)
workflow.add_node("finalizer", finalizer_agent)

# Définition de l'ordre d'exécution
workflow.set_entry_point("analyzer_cot")
workflow.add_edge("analyzer_cot", "explorer_tot")
workflow.add_edge("explorer_tot", "verifier_react")
workflow.add_edge("verifier_react", "critic_correction")

# Boucle conditionnelle pour la Self-Correction
def route_final(state):
    if state["is_safe"] or state["revision_count"] >= 2:
        return "finalizer"
    return "analyzer_cot" # Recommencer le cycle si ce n'est pas sûr

workflow.add_conditional_edges("critic_correction", route_final, {
    "finalizer": "finalizer",
    "analyzer_cot": "analyzer_cot"
})

workflow.add_edge("finalizer", END)
app = workflow.compile()

# --- INTERFACE STREAMLIT ---
protocol_input = st.text_area("📄 Entrez le protocole à auditer (Pipeline Expert) :", height=150)

if st.button("🚀 Lancer l'Audit Expert"):
    if protocol_input:
        with st.spinner("Pipeline Multi-Agents en action..."):
            result = app.invoke({
                "protocol_data": protocol_input,
                "messages": [],
                "is_safe": False,
                "revision_count": 0
            })

            st.subheader("📋 Résultat Final du Pipeline")
            st.success(result["messages"][-1].content)

            with st.expander("🛠️ Détails du Raisonnement (CoT -> ToT -> ReAct -> Correction)"):
                # On définit des labels pour chaque étape pour que ce soit clair
                labels = ["🧠 CoT (Analyse)", "🌳 ToT (Exploration)", "🔍 ReAct (Vérification)", "🛡️ Self-Correction (Audit)"]
                for i, msg in enumerate(result["messages"][:-1]):
                    # On utilise un modulo pour boucler sur les labels si l'agent a fait plusieurs tours
                    label = labels[i % len(labels)]
                    st.markdown(f"**{label}**")
                    st.info(msg.content)
