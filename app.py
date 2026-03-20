import streamlit as st
import os
import operator
from typing import TypedDict, List, Annotated, Union
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END

# --- CONFIGURATION SÉCURISÉE ---
API_KEY = "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"

st.set_page_config(page_title="Auditeur Clinique Haute-Fiabilité", layout="wide")
st.title("🩺 Auditeur de Protocoles de Soins Critiques")
st.markdown("🔍 **Système Multi-Agents avec Raisonnement Avancé (CoT & Self-Correction)**")

# --- DÉFINITION DE L'ÉTAT (Correction : Ajout de la mémoire cumulative) ---
class AgentState(TypedDict):
    """Représente la mémoire accumulée de l'agent."""
    # Annotated + operator.add force LangGraph à AJOUTER les messages au lieu de les écraser
    messages: Annotated[List[BaseMessage], operator.add]
    protocol_data: str
    is_safe: bool
    revision_count: int

# --- AGENTS DE RAISONNEMENT ---
def get_llm():
    return ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=API_KEY)

def analyzer_agent(state: AgentState):
    """Agent de planification utilisant le Chain of Thought (CoT)."""
    llm = get_llm()
    prompt = f"ANALYSE (CoT) : Décompose ce protocole médical étape par étape :\n{state['protocol_data']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [response], "revision_count": state.get('revision_count', 0) + 1}

def critic_agent(state: AgentState):
    """Agent de réflexion et d'auto-critique (Self-Reflection)."""
    llm = get_llm()
    # On passe tout l'historique pour que la critique soit contextuelle
    history = state['messages']
    prompt = "CRITIQUE : Vérifie les étapes précédentes. Cherche des erreurs. Réponds 'SAFE' si c'est parfait."
    response = llm.invoke(history + [HumanMessage(content=prompt)])
    is_safe = "SAFE" in response.content.upper()
    return {"messages": [response], "is_safe": is_safe}

def finalizer_agent(state: AgentState):
    """Synthétise l'échange final pour l'utilisateur."""
    llm = get_llm()
    prompt = "Produis la synthèse finale sécurisée basée sur tout l'audit précédent."
    response = llm.invoke(state['messages'] + [HumanMessage(content=prompt)])
    return {"messages": [response]}

# --- GRAPH DE DÉCISION (Architecture ReAct Loop) ---
workflow = StateGraph(AgentState)
workflow.add_node("analyzer", analyzer_agent)
workflow.add_node("critic", critic_agent)
workflow.add_node("finalizer", finalizer_agent)

workflow.set_entry_point("analyzer")
workflow.add_edge("analyzer", "critic")

def route_decision(state):
    if state["is_safe"] or state["revision_count"] >= 2:
        return "finalizer"
    return "analyzer"

workflow.add_conditional_edges("critic", route_decision, {
    "finalizer": "finalizer",
    "analyzer": "analyzer"
})
workflow.add_edge("finalizer", END)
app = workflow.compile()

# --- INTERFACE UTILISATEUR ---
protocol_input = st.text_area("📄 Entrez le protocole médical ou la prescription :", height=150)

if st.button("🚀 Lancer l'Audit de Haute Fiabilité"):
    if protocol_input:
        with st.spinner("Les agents délibèrent..."):
            result = app.invoke({
                "protocol_data": protocol_input,
                "messages": [],
                "is_safe": False,
                "revision_count": 0
            })

            st.subheader("📋 Recommandation Finale")
            st.write(result["messages"][-1].content)

            # Affichage des logs (Maintenant remplis grâce à Annotated)
            with st.expander("🛠️ Voir le raisonnement interne (Logs de fiabilité)"):
                # On affiche tous les messages sauf le dernier (qui est la réponse finale)
                for i, msg in enumerate(result["messages"][:-1]):
                    tag = "🧠 ANALYSE" if i % 2 == 0 else "🛡️ CRITIQUE"
                    st.markdown(f"**{tag}**")
                    st.info(msg.content)
