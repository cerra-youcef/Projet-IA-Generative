# 🩺 Auditeur Clinique Expert — Système Multi-Agents v4

Ce projet implémente un système d'audit intelligent pour sécuriser les prescriptions médicales critiques via une architecture multi-agents avec outils, mémoire persistante et planification.

---

## 🚀 Évolution du Projet (4 Versions)

Le dépôt est structuré pour montrer la montée en puissance progressive du raisonnement IA :

| Version | Agents | Techniques | Nouveautés |
|---------|--------|------------|------------|
| **V1** | 2 | Self-Correction | Analyseur + Critique. Boucle de correction basique. |
| **V2** | 4 | CoT · ToT · ReAct | Chain of Thought, Tree of Thoughts, pharmacovigilance simulée. |
| **V3** | 5 | + Step-Back · Tools · Memory | Step-Back Prompting, 5 outils médicaux, MemorySaver LangGraph. |
| **V4** | 5 | + Planning · Fix UNSAFE · Intake | Formulaire médical 7 questions, détection SAFE/UNSAFE robuste, rapport coloré. |

---

## 🆕 Nouveautés V4

- **🏥 Formulaire d'intake médical** — 7 questions cliniques structurées (profil patient, antécédents, allergies, traitements, prescription, contexte, biologie) qui reconstituent automatiquement le protocole avant de lancer le pipeline
- **🐛 Bug SAFE/UNSAFE corrigé** — La fonction `detect_verdict()` donne la priorité absolue à `VERDICT: UNSAFE`. Tout doute → UNSAFE par précaution médicale
- **🛠️ Outils enrichis** — Nouvelles interactions critiques ajoutées (warfarine+amiodarone, ibuprofène+insuffisance cardiaque, aspirine+ulcère)
- **🎨 UI verdict colorée** — Bannière rouge si UNSAFE, verte si SAFE, avec rapport `st.error` / `st.success`
- **📋 Plan par agent** — Chaque agent expose son plan d'exécution en 4 étapes avant d'agir

---

## 🧠 Architecture du Pipeline

```
Formulaire (7 questions)
        ↓
 Récapitulatif dossier
        ↓
┌─────────────────────────────────────────────┐
│         PIPELINE MULTI-AGENTS               │
│                                             │
│  Agent 1 — Step-Back (Principes généraux)   │
│       ↓  [check_contraindications]          │
│       ↓  [evidence_score]                   │
│                                             │
│  Agent 2 — CoT (Analyse + Calculs)          │
│       ↓  [calculate_dosage]                 │
│       ↓  [check_contraindications x4]       │
│                                             │
│  Agent 3 — ToT (Arbre de stratégies)        │
│       ↓  [lookup_drug_interactions x3]      │
│       ↓  [pharmacovigilance_search x3]      │
│                                             │
│  Agent 4 — ReAct (Pharmacovigilance)        │
│       ↓  [pharmacovigilance_search x3]      │
│       ↓  [calculate_dosage]                 │
│       ↓  [lookup_drug_interactions]         │
│                                             │
│  Agent 5 — Reflexion (Audit critique)       │
│       ↓  [evidence_score]                   │
│       ↓  [check_contraindications x3]       │
│                                             │
│  ┌─── VERDICT: UNSAFE → Relance Step-Back   │
│  └─── VERDICT: SAFE  → Rapport Final        │
└─────────────────────────────────────────────┘
        ↓
  Rapport d'audit complet
```

---

## 🛠️ Les 5 Outils Médicaux

| Outil | Rôle | Agents |
|-------|------|--------|
| `calculate_dosage` | Calcule dose/prise et dose journalière | CoT, ReAct |
| `lookup_drug_interactions` | Vérifie interactions entre 2 molécules | ToT, ReAct |
| `check_contraindications` | Détecte contre-indications drogue/pathologie | Step-Back, CoT, Critique |
| `pharmacovigilance_search` | Alertes base PharmacoDB simulée | ToT, ReAct |
| `evidence_score` | Score EBM niveau de preuve A/B/C/D | Step-Back, Critique |

---

## ☁️ Utilisation sur Google Colab (Accès Tunnel)

Pour tester l'interface Streamlit depuis Colab :

1. **Mot de passe du tunnel** : Exécutez dans une cellule :
   ```bash
   !curl https://loca.lt/mytunnelpassword
   ```
2. **Lancement** : Lancez la cellule Streamlit et cliquez sur le lien `https://...loca.lt`
3. **Validation** : Entrez l'IP récupérée à l'étape 1 dans le champ *"Tunnel Password"* et validez (tapez `y` si demandé dans le terminal)
4. **Clé API** : Remplacez `sk-proj-...` par votre clé OpenAI dans `app.py` ligne `API_KEY`

---

## ▶️ Lancement rapide (Colab)

```python
# 1. Installer les dépendances
!pip install -r requirements.txt

# 2. Lancer Streamlit avec tunnel
!streamlit run app.py &
!npx localtunnel --port 8501
```

---

## 📁 Structure du projet

```
auditeur-clinique/
├── app.py              # Application principale (v4)
├── requirements.txt    # Dépendances Python
└── README.md           # Ce fichier
```

---

## 🧠 Pourquoi ce système est fiable ?

Le pipeline force l'IA à s'abstraire des données brutes pour rappeler les principes de sécurité **(Step-Back)**, calculer les dosages réels **(CoT)**, explorer plusieurs stratégies **(ToT)**, consulter la pharmacovigilance **(ReAct)**, puis s'auto-critiquer **(Reflexion)**. La règle absolue : **un seul problème critique suffit pour déclencher `VERDICT: UNSAFE`**.
