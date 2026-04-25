import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score

# Configuration de la page
st.set_page_config(page_title="Leaderboard AI Club", page_icon="🤖")

@st.cache_data
def load_true_data(file_path):
    # On suppose que le fichier contient une colonne cible (ex: 'target')
    return pd.read_csv(file_path)

# Chargement des données de référence (Y_test)
try:
    y_true_df = load_true_data('Y_test.csv')
    # Extraction de la série si c'est un DataFrame à une colonne
    y_true = y_true_df.iloc[:, 0] 
except FileNotFoundError:
    st.error("Erreur : Le fichier 'Y_test.csv' est introuvable sur le serveur.")
    st.stop()

st.title("🏆 Hackathon Machine Learning")
st.subheader("Club IA - Faculté d'Agronomie (Avril 2026)")

st.info("Chargez votre fichier CSV contenant uniquement vos prédictions (une seule colonne).")

file_soumission = st.file_uploader("Déposez votre fichier de soumission", type="csv")

if file_soumission is not None:
    try:
        soumission = pd.read_csv(file_soumission)
        
        # Vérification rapide de la taille
        if len(soumission) != len(y_true):
            st.warning(f"Attention : Votre fichier a {len(soumission)} lignes, mais {len(y_true)} sont attendues.")
        else:
            # Extraction des prédictions
            y_pred = soumission.iloc[:, 0]

            # Calcul des métriques
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted') # 'weighted' pour gérer le multiclasse si besoin

            # Affichage des résultats
            st.divider()
            col1, col2 = st.columns(2)
            
            col1.metric(label="Accuracy Score", value=f"{acc:.3f}")
            col2.metric(label="F1-Score", value=f"{f1:.3f}")

            if acc > 0.8:
                st.balloons()
                st.success("Excellent travail ! Vos performances sont solides.")
            
    except Exception as e:
        st.error(f"Erreur lors de l'analyse du fichier : {e}")