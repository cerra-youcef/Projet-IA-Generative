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
