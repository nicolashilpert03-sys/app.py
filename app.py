# =========================================================
# 🌤️ Mon Application Météo Simple — Beauvais (version "cours" + graphes compacts)
# =========================================================
# Pile "cours" :
# - requests pour récupérer l'API Open-Meteo
# - pandas / numpy pour manipuler les données
# - régression linéaire "maison" (numpy.linalg.lstsq) avec saisonnalité sin/cos
# - matplotlib pour les graphiques (tailles réduites)
# =========================================================

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Config & constantes
# -----------------------------
st.set_page_config(page_title="Climat Beauvais (cours)", layout="wide", page_icon="🌦️")
st.markdown("<h1 style='text-align:center;'>🌤️ Climat de Beauvais — Version Cours</h1>", unsafe_allow_html=True)
st.write("Comparaison 2004/2024 et projection 2044 avec régression linéaire **maison** et saisonnalité (sin/cos).")

ORDRE_MOIS = ["Janvier","Février","Mars","Avril","Mai","Juin",
              "Juillet","Août","Septembre","Octobre","Novembre","Décembre"]
NOMS_MOIS = {i+1: ORDRE_MOIS[i] for i in range(12)}
SMALL_FIGSIZE = (3.5, 2.2)   # ✅ format compact

# -----------------------------
# Fonctions Data
# -----------------------------
def telecharger_journalier(annee):
    """Récupère les séries journalières via l'API Open-Meteo."""
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
    d = r.json()["daily"]
    df = pd.DataFrame({
        "date": pd.to_datetime(d["time"]),
        "Température": d["temperature_2m_mean"],
        "Pluie (mm)": d["precipitation_sum"],
        "ET0 (mm)": d["et0_fao_evapotranspiration"]
    })
    df["mois"] = df["date"].dt.month
    return df

def agregation_mensuelle(df):
    """Agrège par mois : Température=mean, Pluie/ET0=sum + cumuls et noms de mois."""
    df_m = (
        df.groupby("mois", as_index=False)
          .agg({"Température": "mean", "Pluie (mm)": "sum", "ET0 (mm)": "sum"})
    )
    df_m["Nom du Mois"] = df_m["mois"].map(NOMS_MOIS)
    df_m["Pluie Totale Progressive (mm)"] = df_m["Pluie (mm)"].cumsum()
    df_m["ET0 Totale Progressive (mm)"] = df_m["ET0 (mm)"].cumsum()
    return df_m.round(1)

def telecharger_et_preparer_donnees(annee):
    return agregation_mensuelle(telecharger_journalier(annee))

def preparer_donnees_pour_ml(an_debut=2004, an_fin=2024):
    """Assemble 2004..2024 + features saisonnières sin/cos."""
    all_years = []
    for an in range(an_debut, an_fin+1):
        dfm = telecharger_et_preparer_donnees(an)
        dfm["Année"] = an
        all_years.append(dfm[["Année","mois","Température","Pluie (mm)","ET0 (mm)"]])
    train = pd.concat(all_years, ignore_index=True)
    angle = 2*np.pi*(train["mois"]-1)/12
    train["sin_saison"] = np.sin(angle)
    train["cos_saison"] = np.cos(angle)
    return train

# -----------------------------
# Régression linéaire "maison"
# -----------------------------
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

# -----------------------------
# Chargement des données
# -----------------------------
with st.spinner("⏳ Téléchargement des données 2004 & 2024..."):
    df_2004 = telecharger_et_preparer_donnees(2004)
    df_2024 = telecharger_et_preparer_donnees(2024)

# -----------------------------
# Onglets
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🆚 Comparaison 2004/2024", "📅 Une année", "🔮 Projection 2044"])

# =========================================================
# 🆚 Comparaison
# =========================================================
with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("🌡️ Moyenne Temp. 2024",
              f"{df_2024['Température'].mean():.1f} °C",
              f"{(df_2024['Température'].mean()-df_2004['Température'].mean()):+.1f}")
    c2.metric("🌧️ Total Pluie 2024",
              f"{df_2024['Pluie (mm)'].sum():.1f} mm",
              f"{(df_2024['Pluie (mm)'].sum()-df_2004['Pluie (mm)'].sum()):+.1f}")
    c3.metric("💧 Total ET0 2024",
              f"{df_2024['ET0 (mm)'].sum():.1f} mm",
              f"{(df_2024['ET0 (mm)'].sum()-df_2004['ET0 (mm)'].sum()):+.1f}")

    # Températures (lignes)
    st.markdown("#### 🧪 Températures mensuelles")
    fig, ax = plt.subplots(figsize=SMALL_FIGSIZE)
    ax.plot(df_2004["Nom du Mois"], df_2004["Température"], marker="o", label="2004")
    ax.plot(df_2024["Nom du Mois"], df_2024["Température"], marker="o", label="2024")
    ax.set_xlabel("Mois"); ax.set_ylabel("Température (°C)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig, clear_figure=True, use_container_width=False)
    plt.close(fig)

    # Pluies (barres côte à côte)
    st.markdown("#### 🌧️ Pluie mensuelle")
    fig, ax = plt.subplots(figsize=SMALL_FIGSIZE)
    x = np.arange(12); width = 0.4
    ax.bar(x - width/2, df_2004["Pluie (mm)"], width, label="2004")
    ax.bar(x + width/2, df_2024["Pluie (mm)"], width, label="2024")
    ax.set_xticks(x, ORDRE_MOIS)   # ✅ parenthèse fermée (corrigé)
    ax.set_ylabel("Pluie (mm)")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    st.pyplot(fig, clear_figure=True, use_container_width=False)
    plt.close(fig)

# =========================================================
# 📅 Une année
# =========================================================
with tab2:
    annee = st.radio("Choisis une année :", [2004, 2024], horizontal=True)
    df = df_2004 if annee == 2004 else df_2024
    st.dataframe(df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🌡️ Température")
        fig, ax = plt.subplots(figsize=SMALL_FIGSIZE)
        ax.plot(df["Nom du Mois"], df["Température"], marker="o")
        ax.set_xlabel("Mois"); ax.set_ylabel("Température (°C)")
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig, clear_figure=True, use_container_width=False)
        plt.close(fig)
    with col2:
        st.markdown("#### 🌧️ Pluie")
        fig, ax = plt.subplots(figsize=SMALL_FIGSIZE)
        ax.bar(df["Nom du Mois"], df["Pluie (mm)"])
        ax.set_xlabel("Mois"); ax.set_ylabel("Pluie (mm)")
        ax.grid(True, axis="y", alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig, clear_figure=True, use_container_width=False)
        plt.close(fig)

    st.markdown("#### 💧 ET0 cumulée")
    fig, ax = plt.subplots(figsize=SMALL_FIGSIZE)
    ax.fill_between(df["Nom du Mois"], df["ET0 Totale Progressive (mm)"], alpha=0.6, step="mid")
    ax.set_xlabel("Mois"); ax.set_ylabel("ET0 cumulée (mm)")
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig, clear_figure=True, use_container_width=False)
    plt.close(fig)

# =========================================================
# 🔮 Projection 2044
# =========================================================
with tab3:
    st.subheader("Projection 2044 (régression linéaire maison)")
    with st.spinner("⏳ Calcul en cours..."):
        train = preparer_donnees_pour_ml(2004, 2024)
        t2044 = faire_projection_simple(train, "Température", 2044)
        p2044 = faire_projection_simple(train, "Pluie (mm)", 2044)
        e2044 = faire_projection_simple(train, "ET0 (mm)", 2044)

    df_2044 = (
        t2044.merge(p2044, on=["mois","Nom du Mois"])
             .merge(e2044, on=["mois","Nom du Mois"])
    )
    df_2044["Pluie Totale Progressive (mm)"] = df_2044["Pluie (mm)"].cumsum()
    df_2044["ET0 Totale Progressive (mm)"] = df_2044["ET0 (mm)"].cumsum()
    st.dataframe(df_2044, use_container_width=True)

    # Températures comparées
    st.markdown("#### 🌡️ 2004 / 2024 / 2044")
    fig, ax = plt.subplots(figsize=SMALL_FIGSIZE)
    ax.plot(df_2004["Nom du Mois"], df_2004["Température"], marker="o", label="2004")
    ax.plot(df_2024["Nom du Mois"], df_2024["Température"], marker="o", label="2024")
    ax.plot(df_2044["Nom du Mois"], df_2044["Température"], marker="o", label="2044 (proj.)")
    ax.set_xlabel("Mois"); ax.set_ylabel("Température (°C)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig, clear_figure=True, use_container_width=False)
    plt.close(fig)

    # Pluies comparées
    st.markdown("#### 🌧️ 2004 / 2024 / 2044")
    fig, ax = plt.subplots(figsize=SMALL_FIGSIZE)
    x = np.arange(12); width = 0.25
    ax.bar(x - width, df_2004["Pluie (mm)"], width, label="2004")
    ax.bar(x,          df_2024["Pluie (mm)"], width, label="2024")
    ax.bar(x + width,  df_2044["Pluie (mm)"], width, label="2044 (proj.)")
    ax.set_xticks(x, ORDRE_MOIS)  # ✅ signature correcte
    ax.set_ylabel("Pluie (mm)")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    st.pyplot(fig, clear_figure=True, use_container_width=False)
    plt.close(fig)
