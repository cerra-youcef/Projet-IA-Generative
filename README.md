# 🩺 Audit de Protocoles Médicaux (Multi-Agents)

Ce projet implémente un système d'audit intelligent pour sécuriser les prescriptions médicales critiques via une architecture multi-agents.

## 🚀 Évolution du Projet (3 Versions)
Le dépôt est structuré pour montrer la montée en puissance du raisonnement IA :
1. **V1 : Base** ➔ 2 agents (Analyseur + Critique). Boucle de **Self-Correction**.
2. **V2 : Avancé** ➔ 4 agents. **Chain of Thought (CoT)**, **Tree of Thoughts (ToT)** et **ReAct**.
3. **V3 : Expert** ➔ 5 agents. Ajout du **Step-Back Prompting** (Abstraction) pour une sécurité maximale.

## ☁️ Utilisation sur Google Colab (Accès Tunnel)
Pour tester l'interface Streamlit depuis Colab :

1. **Mot de passe du tunnel :** Exécutez `!curl https://loca.lt/mytunnelpassword`
2. **Lancement :** Lancez la cellule Streamlit et cliquez sur le lien `https://...loca.lt`.
3. **Validation :** Entrez l'IP récupérée à l'étape 1 dans le champ "Tunnel Password" et validez (tapez 'y' si demandé dans le terminal).
4. **Clé API :** Saisissez votre clé OpenAI dans la barre latérale pour activer les agents.

## 🧠 Pourquoi ce système est fiable ?
Le pipeline force l'IA à s'extraire des données brutes pour rappeler les principes de sécurité (**Step-Back**), explorer plusieurs pistes (**ToT**) et s'auto-critiquer (**Reflexion**).
