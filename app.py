# =========================================================
# 🌤️ Mon Application Météo Simple — Beauvais (version "cours" + graphes réduits)
# =========================================================
# Version pédagogique : requests + pandas/numpy + régression linéaire maison
# Graphiques plus petits (figsize réduite) pour meilleure lisibilité sur Streamlit.
# =========================================================

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Configuration de la page
# ---------------------------------------------------------
st.set_page_config(page_title="Climat Beauvais (cours)", layout="wide", page_icon="🌦️")
st.markdown(
    "<h1 style='text-align:center;'>🌤️ Climat de Beauvais — Version Cours</h1>",
    unsafe_allow_html=True
)
st.write("Analyse et projection météo (2004, 2024, 2044) avec les outils Python vus en cours : pandas, numpy, régression linéaire simple et matplotlib.")

# ---------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------
ORDRE_MOIS = ["Janvier","Février","Mars","Avril","Mai","Juin",
              "Juillet","Août","Septembre","Octobre","Novembre","Décembre"]
NOMS_MOIS = {i+1: ORDRE_MOIS[i] for i in range(12)}

def telecharger_journalier(annee):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 49.43,
        "longitude": 2.08,
        "start_date": f"{annee}-01-01",
        "end_date": f"{annee}-12-31",
        "daily": "temperature_2m_mean,precipitation_sum,et0_fao_evapotranspiration",
        "timezone": "Europe/Paris"
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()["daily"]

    df = pd.DataFrame({
        "date": pd.to_datetime(data["time"]),
        "Température": data["temperature_2m_mean"],
        "Pluie (mm)": data["precipitation_sum"],
        "ET0 (mm)": data["et0_fao_evapotranspiration"]
    })
    df["mois"] = df["date"].dt.month
    return df

def agregation_mensuelle(df):
    df_m = (
        df.groupby("mois", as_index=False)
          .agg({"Température":"mean", "Pluie (mm)":"sum", "ET0 (mm)":"sum"})
    )
    df_m["Nom du Mois"] = df_m["mois"].map(NOMS_MOIS)
    df_m["Pluie Totale Progressive (mm)"] = df_m["Pluie (mm)"].cumsum()
    df_m["ET0 Totale Progressive (mm)"] = df_m["ET0 (mm)"].cumsum()
    return df_m.round(1)

def telecharger_et_preparer_donnees(annee):
    return agregation_mensuelle(telecharger_journalier(annee))

def preparer_donnees_pour_ml(an_debut=2004, an_fin=2024):
    all_years = []
    for an in range(an_debut, an_fin+1):
        df = telecharger_et_preparer_donnees(an)
        df["Année"] = an
        all_years.append(df[["Année","mois","Température","Pluie (mm)","ET0 (mm)"]])
    df_all = pd.concat(all_years, ignore_index=True)
    angle = 2*np.pi*(df_all["mois"]-1)/12
    df_all["sin_saison"], df_all["cos_saison"] = np.sin(angle), np.cos(angle)
    return df_all

def regression_lineaire_maison(X, y):
    Xb = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return beta

def predire_reg_lin_maison(X, beta):
    Xb = np.column_stack([np.ones(len(X)), X])
    return Xb @ beta

def faire_projection_simple(train_df, col, annee_cible):
    X = train_df[["Année","sin_saison","cos_saison"]].to_numpy()
    y = train_df[col].to_numpy()
    beta = regression_lineaire_maison(X, y)
    mois = np.arange(1,13)
    angle = 2*np.pi*(mois-1)/12
    X_pred = np.column_stack([
        np.full(12, annee_cible),
        np.sin(angle),
        np.cos(angle)
    ])
    y_pred = predire_reg_lin_maison(X_pred, beta)
    return pd.DataFrame({
        "mois": mois,
        "Nom du Mois": [ORDRE_MOIS[m-1] for m in mois],
        col: np.round(y_pred, 1)
    })

# ---------------------------------------------------------
# Données de base
# ---------------------------------------------------------
with st.spinner("⏳ Téléchargement des données 2004 et 2024..."):
    df_2004 = telecharger_et_preparer_donnees(2004)
    df_2024 = telecharger_et_preparer_donnees(2024)

# ---------------------------------------------------------
# Onglets
# ---------------------------------------------------------
onglet1, onglet2, onglet3 = st.tabs([
    "🆚 Comparaison 2004 / 2024",
    "📅 Une seule année",
    "🔮 Projection 2044"
])

# =========================================================
# 🆚 Comparaison
# =========================================================
with onglet1:
    c1, c2, c3 = st.columns(3)
    temp_diff = df_2024["Température"].mean() - df_2004["Température"].mean()
    pluie_diff = df_2024["Pluie (mm)"].sum() - df_2004["Pluie (mm)"].sum()
    et0_diff = df_2024["ET0 (mm)"].sum() - df_2004["ET0 (mm)"].sum()

    c1.metric("🌡️ Moyenne Temp. 2024", f"{df_2024['Température'].mean():.1f} °C", f"{temp_diff:+.1f}")
    c2.metric("🌧️ Total Pluie 2024", f"{df_2024['Pluie (mm)'].sum():.1f} mm", f"{pluie_diff:+.1f}")
    c3.metric("💧 Total ET0 2024", f"{df_2024['ET0 (mm)'].sum():.1f} mm", f"{et0_diff:+.1f}")

    st.markdown("#### 🌡️ Températures mensuelles")
    fig, ax = plt.subplots(figsize=(5,3))
    ax.plot(df_2004["Nom du Mois"], df_2004["Température"], marker="o", label="2004")
    ax.plot(df_2024["Nom du Mois"], df_2024["Température"], marker="o", label="2024")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xlabel("Mois"); ax.set_ylabel("Température (°C)")
    plt.xticks(rotation=45)
    st.pyplot(fig, clear_figure=True)

    st.markdown("#### 🌧️ Pluie mensuelle")
    fig, ax = plt.subplots(figsize=(5,3))
    x = np.arange(12); width = 0.4
    ax.bar(x - width/2, df_2004["Pluie (mm)"], width, label="2004")
    ax.bar(x + width/2, df_2024["Pluie (mm)"], width, label="2024")
    ax.set_xticks(x, ORDRE_MOIS, rotation=45)
    ax.set_ylabel("Pluie (mm)")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    st.pyplot(fig, clear_figure=True)

# =========================================================
# 📅 Une seule année
# =========================================================
with onglet2:
    annee = st.radio("Choisis une année :", [2004, 2024], horizontal=True)
    df = df_2004 if annee == 2004 else df_2024
    st.dataframe(df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🌡️ Température")
        fig, ax = plt.subplots(figsize=(5,3))
        ax.plot(df["Nom du Mois"], df["Température"], marker="o")
        ax.set_xlabel("Mois"); ax.set_ylabel("Température (°C)")
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig, clear_figure=True)
    with col2:
        st.markdown("#### 🌧️ Pluie")
        fig, ax = plt.subplots(figsize=(5,3))
        ax.bar(df["Nom du Mois"], df["Pluie (mm)"])
        ax.set_xlabel("Mois"); ax.set_ylabel("Pluie (mm)")
        ax.grid(True, axis="y", alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig, clear_figure=True)

    st.markdown("#### 💧 ET0 cumulée")
    fig, ax = plt.subplots(figsize=(5,3))
    ax.fill_between(df["Nom du Mois"], df["ET0 Totale Progressive (mm)"], alpha=0.6, step="mid")
    ax.set_xlabel("Mois"); ax.set_ylabel("ET0 cumulée (mm)")
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig, clear_figure=True)

# =========================================================
# 🔮 Projection 2044
# =========================================================
with onglet3:
    st.subheader("Projection 2044 (Régression linéaire maison)")

    with st.spinner("⏳ Calcul de la tendance..."):
        train = preparer_donnees_pour_ml(2004, 2024)
        temp_2044 = faire_projection_simple(train, "Température", 2044)
        pluie_2044 = faire_projection_simple(train, "Pluie (mm)", 2044)
        et0_2044 = faire_projection_simple(train, "ET0 (mm)", 2044)

    df_2044 = (
        temp_2044.merge(pluie_2044, on=["mois","Nom du Mois"])
                 .merge(et0_2044, on=["mois","Nom du Mois"])
    )
    df_2044["Pluie Totale Progressive (mm)"] = df_2044["Pluie (mm)"].cumsum()
    df_2044["ET0 Totale Progressive (mm)"] = df_2044["ET0 (mm)"].cumsum()

    st.dataframe(df_2044, use_container_width=True)

    # Graph Températures
    st.markdown("#### 🌡️ Évolution des températures")
    fig, ax = plt.subplots(figsize=(5,3))
    ax.plot(df_2004["Nom du Mois"], df_2004["Température"], marker="o", label="2004")
    ax.plot(df_2024["Nom du Mois"], df_2024["Température"], marker="o", label="2024")
    ax.plot(df_2044["Nom du Mois"], df_2044["Température"], marker="o", label="2044 (proj.)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig, clear_figure=True)

    # Graph Pluies
    st.markdown("#### 🌧️ Évolution des pluies")
    fig, ax = plt.subplots(figsize=(5,3))
    x = np.arange(12); width = 0.25
    ax.bar(x - width, df_2004["Pluie (mm)"], width, label="2004")
    ax.bar(x, df_2024["Pluie (mm)"], width, label="2024")
    ax.bar(x + width, df_2044["Pluie (mm)"], width, label="2044 (proj.)")
    ax.set_xticks(x, ORDRE_MOIS, rotation=45)
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    st.pyplot(fig, clear_figure=True)
