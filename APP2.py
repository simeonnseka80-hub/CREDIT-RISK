import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go

# ==========================================
# CONFIG
# ==========================================
st.set_page_config(page_title="Credit Risk AI", page_icon="💳", layout="wide")

# ==========================================
# LOAD MODEL
# ==========================================
# Utilise tes fichiers actuels (Random Forest ou autre)
try:
    model = joblib.load("modele_credit_rf.pkl")  # Nom mis à jour ici
    scaler = joblib.load("scaler.pkl")
    features_list = joblib.load("features_list.pkl") # Chargement des colonnes
except FileNotFoundError:
    st.error("❌ Erreur : Impossible de trouver 'modele_credit_rf.pkl'. Vérifiez le dossier !")
    st.stop()
# ==========================================
# TITLE
# ==========================================
st.title("💳 Credit Risk AI - Decision System")
st.markdown("Système expert d'analyse du risque de crédit en temps réel")

# ==========================================
# SIDEBAR INPUT
# ==========================================
with st.sidebar:
    st.header("👤 Profil Client")
    
    age = st.number_input("Âge", 18, 100, 30)
    income = st.number_input("Revenu annuel ($)", 0, 1000000, 50000)
    emp_length = st.number_input("Ancienneté (années)", 0, 40, 5)
    
    home = st.selectbox("Statut logement", ["RENT", "OWN", "MORTGAGE"])
    default = st.selectbox("Historique défaut", ["N", "Y"])

    st.divider()
    st.caption("🔒 **Note de sécurité** : Traitement local immédiat. Aucune donnée n'est stockée sur le serveur.")

# ==========================================
# LOAN INFO
# ==========================================
st.subheader("💰 Détails du Financement")
col1, col2 = st.columns(2)

with col1:
    loan_amnt = st.number_input("Montant demandé ($)", 500, 100000, 10000)
    loan_intent = st.selectbox("Objectif du prêt", [
        "EDUCATION", "MEDICAL", "VENTURE",
        "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"
    ])

with col2:
    loan_grade = st.selectbox("Grade Interne (A-G)", ["A","B","C","D","E","F","G"])
    int_rate = st.slider("Taux d'intérêt (%)", 5.0, 35.0, 12.0)

# ==========================================
# PREDICTION ENGINE
# ==========================================
if st.button("🚀 Lancer l'Analyse du Risque"):

    if income <= 0:
        st.error("⚠️ Erreur : Le revenu doit être supérieur à 0 pour calculer le ratio d'endettement.")
    else:
        # ----- Feature Engineering -----
        grade_mapping = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6}
        default_mapping = {'N':0,'Y':1}

        input_data = pd.DataFrame({
            'person_age':[age],
            'person_income':[income],
            'person_emp_length':[emp_length],
            'loan_amnt':[loan_amnt],
            'loan_int_rate':[int_rate],
            'loan_percent_income':[loan_amnt / (income+1)],
            'cb_person_cred_hist_length':[emp_length],  
            'loan_grade':[grade_mapping[loan_grade]],
            'cb_person_default_on_file':[default_mapping[default]]
        })

        # ----- One-Hot Encoding manuel (pour rester léger) -----
        home_cols = ['person_home_ownership_OWN','person_home_ownership_RENT']
        intent_cols = ['loan_intent_EDUCATION','loan_intent_MEDICAL','loan_intent_PERSONAL','loan_intent_VENTURE','loan_intent_HOMEIMPROVEMENT']

        for col in home_cols + intent_cols:
            input_data[col] = 0

        if home == "OWN": input_data['person_home_ownership_OWN'] = 1
        elif home == "RENT": input_data['person_home_ownership_RENT'] = 1

        if loan_intent != "DEBTCONSOLIDATION":
            input_data[f'loan_intent_{loan_intent}'] = 1

        # ----- Alignement & Prediction -----
        input_data = input_data.reindex(columns=scaler.feature_names_in_, fill_value=0)
        input_scaled = scaler.transform(input_data)
        proba = model.predict_proba(input_scaled)[0][1]

        # ==========================================
        # VISUALISATION DES RÉSULTATS
        # ==========================================
        st.divider()
        
        # Jauge Plotly (Impact visuel fort)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba*100,
            title={'text': "Score de Risque de Défaut (%)", 'font': {'size': 20}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "#2ecc71"}, # Vert
                    {'range': [25, 50], 'color': "#f1c40f"}, # Jaune
                    {'range': [50, 100], 'color': "#e74c3c"} # Rouge
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': proba*100
                }
            }
        ))
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

        # Recommandation Finale
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.metric("Montant à risque estimé", f"${loan_amnt * proba:,.0f}")
            
        with col_res2:
            if proba > 0.5:
                st.error("🚨 DÉCISION : RISQUE ÉLEVÉ")
                st.write("Le profil présente une probabilité de défaut critique.")
            elif proba > 0.25:
                st.warning("⚠️ DÉCISION : ATTENTION")
                st.write("Analyse manuelle et garanties supplémentaires recommandées.")
            else:
                st.success("✅ DÉCISION : RISQUE FAIBLE")
                st.write("Le profil répond aux critères de solvabilité standards.")