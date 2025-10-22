# =========================================================
# 🌤️ Mon Application Météo — version "cours" (graphiques compacts)
# =========================================================

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- paramètres d'affichage compacts ----
FIGSIZE = (3.6, 2.3)   # largeur, hauteur en pouces
DPI = 160
FS_BASE = 9
plt.rcParams.update({
    "figure.figsize": FIGSIZE,
    "figure.dpi": DPI,
    "axes.titlesize": FS_BASE,
    "axes.labelsize": FS_BASE,
    "xtick.labelsize": FS_BASE-1,
    "ytick.labelsize": FS_BASE-1,
    "legend.fontsize": FS_BASE-1,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

# ---------------------------------------------------------
# Configuration Streamlit
# ---------------------------------------------------------
st.set_page_config(page_title="Climat Beauvais (cours)", layout="wide", page_icon="🌦️")
st.markdown("<h1 style='text-align:center;'>🌤️ Climat de Beauvais — Version Cours</h1>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Utilitaires data
# ---------------------------------------------------------
ORDRE_MOIS = ["Janvier","Février","Mars","Avril","Mai","Juin",
              "Juillet","Août","Septembre","Octobre","Novembre","Décembre"]
NOMS_MOIS = {i+1: ORDRE_MOIS[i] for i in range(12)}

def telecharger_journalier(annee):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 49.43, "longitude": 2.08,
        "start_date": f"{annee}-01-01",
        "end_date": f"{annee}-12-31",
        "daily": "temperature_2m_mean,precipitation_sum,et0_fao_evapotranspiration",
        "timezone": "Europe/Paris"
    }
    r = requests.get(url, params=params, timeout=60); r.raise_for_status()
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
    dfm = (df.groupby("mois", as_index=False)
             .agg({"Température":"mean","Pluie (mm)":"sum","ET0 (mm)":"sum"}))
    dfm["Nom du Mois"] = dfm["mois"].map(NOMS_MOIS)
    dfm["Pluie Totale Progressive (mm)"] = dfm["Pluie (mm)"].cumsum()
    dfm["ET0 Totale Progressive (mm)"] = dfm["ET0 (mm)"].cumsum()
    return dfm.round(1)

def telecharger_et_preparer_donnees(annee):
    return agregation_mensuelle(telecharger_journalier(annee))

def preparer_donnees_pour_ml(an_debut=2004, an_fin=2024):
    all_df = []
    for an in range(an_debut, an_fin+1):
        dfm = telecharger_et_preparer_donnees(an)
        dfm["Année"] = an
        all_df.append(dfm[["Année","mois","Température","Pluie (mm)","ET0 (mm)"]])
    train = pd.concat(all_df, ignore_index=True)
    angle = 2*np.pi*(train["mois"]-1)/12
    train["sin_saison"], train["cos_saison"] = np.sin(angle), np.cos(angle)
    return train

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

    mois = np.arange(1, 13)
    angle = 2*np.pi*(mois-1)/12
    X_pred = np.column_stack([np.full(12, annee_cible), np.sin(angle), np.cos(angle)])
    y_pred = predire_reg_lin_maison(X_pred, beta)

    return pd.DataFrame({
        "mois": mois,
        "Nom du Mois": [ORDRE_MOIS[m-1] for m in mois],
        col: np.round(y_pred, 1)
    })

# ---------------------------------------------------------
# Données
# ---------------------------------------------------------
with st.spinner("⏳ Téléchargement 2004 & 2024..."):
    df_2004 = telecharger_et_preparer_donnees(2004)
    df_2024 = telecharger_et_preparer_donnees(2024)

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
t1, t2, t3 = st.tabs(["🆚 2004 vs 2024", "📅 Une année", "🔮 2044"])

# =======================
# 🆚 Comparaison
# =======================
with t1:
    c1, c2, c3 = st.columns(3)
    c1.metric("🌡️ Moy. Temp 2024", f"{df_2024['Température'].mean():.1f} °C",
              f"{(df_2024['Température'].mean()-df_2004['Température'].mean()):+.1f}")
    c2.metric("🌧️ Pluie 2024", f"{df_2024['Pluie (mm)'].sum():.1f} mm",
              f"{(df_2024['Pluie (mm)'].sum()-df_2004['Pluie (mm)'].sum()):+.1f}")
    c3.metric("💧 ET0 2024", f"{df_2024['ET0 (mm)'].sum():.1f} mm",
              f"{(df_2024['ET0 (mm)'].sum()-df_2004['ET0 (mm)'].sum()):+.1f}")

    st.markdown("#### 🌡️ Températures mensuelles")
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, constrained_layout=True)
    ax.plot(df_2004["Nom du Mois"], df_2004["Température"], marker="o", ms=3.5, lw=1.4, label="2004")
    ax.plot(df_2024["Nom du Mois"], df_2024["Température"], marker="o", ms=3.5, lw=1.4, label="2024")
    ax.set_xlabel("Mois"); ax.set_ylabel("Température (°C)")
    ax.legend(frameon=True)
    plt.xticks(rotation=45)
    st.pyplot(fig, clear_figure=True, use_container_width=False)

    st.markdown("#### 🌧️ Pluie mensuelle")
    x = np.arange(12); width = 0.36
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, constrained_layout=True)
    ax.bar(x - width/2, df_2004["Pluie (mm)"], width, label="2004")
    ax.bar(x + width/2, df_2024["Pluie (mm)"], width, label="2024")
    ax.set_xticks(x, ORDRE_MOIS, rotation=45)
    ax.set_ylabel("Pluie (mm)"); ax.legend(frameon=True)
    st.pyplot(fig, clear_figure=True, use_container_width=False)

# =======================
# 📅 Une seule année
# =======================
with t2:
    annee = st.radio("Choisis une année :", [2004, 2024], horizontal=True)
    df = df_2004 if annee == 2004 else df_2024
    st.dataframe(df, use_container_width=True, height=320)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🌡️ Température")
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, constrained_layout=True)
        ax.plot(df["Nom du Mois"], df["Température"], marker="o", ms=3.5, lw=1.4)
        ax.set_xlabel("Mois"); ax.set_ylabel("Température (°C)")
        plt.xticks(rotation=45)
        st.pyplot(fig, clear_figure=True, use_container_width=False)
    with col2:
        st.markdown("#### 🌧️ Pluie")
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, constrained_layout=True)
        ax.bar(df["Nom du Mois"], df["Pluie (mm)"])
        ax.set_xlabel("Mois"); ax.set_ylabel("Pluie (mm)")
        plt.xticks(rotation=45)
        st.pyplot(fig, clear_figure=True, use_container_width=False)

    st.markdown("#### 💧 ET0 cumulée")
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, constrained_layout=True)
    ax.fill_between(df["Nom du Mois"], df["ET0 Totale Progressive (mm)"], alpha=0.55, step="mid")
    ax.set_xlabel("Mois"); ax.set_ylabel("ET0 cumulée (mm)")
    plt.xticks(rotation=45)
    st.pyplot(fig, clear_figure=True, use_container_width=False)

# =======================
# 🔮 Projection 2044
# =======================
with t3:
    with st.spinner("⏳ Calcul de la projection..."):
        train = preparer_donnees_pour_ml(2004, 2024)
        t2044 = faire_projection_simple(train, "Température", 2044)
        p2044 = faire_projection_simple(train, "Pluie (mm)", 2044)
        e2044 = faire_projection_simple(train, "ET0 (mm)", 2044)
    df_2044 = (t2044.merge(p2044, on=["mois","Nom du Mois"])
                      .merge(e2044, on=["mois","Nom du Mois"]))
    df_2044["Pluie Totale Progressive (mm)"] = df_2044["Pluie (mm)"].cumsum()
    df_2044["ET0 Totale Progressive (mm)"] = df_2044["ET0 (mm)"].cumsum()
    st.dataframe(df_2044, use_container_width=True, height=320)

    st.markdown("#### 🌡️ Températures 2004 / 2024 / 2044")
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, constrained_layout=True)
    ax.plot(df_2004["Nom du Mois"], df_2004["Température"], marker="o", ms=3.5, lw=1.4, label="2004")
    ax.plot(df_2024["Nom du Mois"], df_2024["Température"], marker="o", ms=3.5, lw=1.4, label="2024")
    ax.plot(df_2044["Nom du Mois"], df_2044["Température"], marker="o", ms=3.5, lw=1.4, label="2044 (proj.)")
    ax.set_xlabel("Mois"); ax.set_ylabel("Température (°C)")
    ax.legend(frameon=True)
    plt.xticks(rotation=45)
    st.pyplot(fig, clear_figure=True, use_container_width=False)

    st.markdown("#### 🌧️ Pluies 2004 / 2024 / 2044")
    x = np.arange(12); width = 0.24
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, constrained_layout=True)
    ax.bar(x - width, df_2004["Pluie (mm)"], width, label="2004")
    ax.bar(x,          df_2024["Pluie (mm)"], width, label="2024")
    ax.bar(x + width,  df_2044["Pluie (mm)"], width, label="2044 (proj.)")
    ax.set_xticks(x, ORDRE_MOIS
