import streamlit as st
import os, operator, json
from typing import TypedDict, List, Annotated
from datetime import datetime

# --- CONFIGURATION ---
API_KEY = "sk-proj-..."  # Remplacer par votre clé

st.set_page_config(page_title="Auditeur Clinique Expert v3", layout="wide")
