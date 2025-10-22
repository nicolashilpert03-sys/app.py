# =========================================================
# 🌤️ Mon Application Météo Simple — Beauvais (version "cours")
# =========================================================
# Version compacte : mêmes fonctions, mais graphiques réduits
# =========================================================

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Configuration de la page
# ---------------------------------------------------------
st.set_page_config(page_title="Climat Beauvais (version cours)", layout="wide", page_icon="🌦️")
st.markdown(
    "<h1 style='text-align:center;'>🌤️ Climat de Beauvais — Version « Cours »</h1>",
    unsafe_allow_html=True
)
st.write("Données historiques via Open-Meteo (2004 & 2024), comparaisons, et projection 2044 avec une régression linéaire **faite maison** (+ saisonnalité sin/cos).")

# ---------------------------------------------------------
# Utilitaires "cours"
# ---------------------------------------------------------
ORDRE_MOIS = ["Janvier","Février","Mars","Avril","Mai","Juin",
              "Juillet","Août","Septembre","Octobre","Novembre","Décembre"]
NOMS_MOIS = {i+1: ORDRE_MOIS[i] for i in range(12)}

def telecharger_journalier(annee, lat=49.43, lon=2.08, tz="Europe/Paris"):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{annee}-01-01",
        "end_date": f"{annee}-12-31",
        "daily": "temperature_2m_mean,precipitation_sum,et0_fao_evapotranspiration",
        "timezone": tz
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
          .agg({"Température": "mean", "Pluie (mm)": "sum", "ET0 (mm)": "sum"})
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
        df_m = telecharger_et_preparer_donnees(an)
        df_m["Année"] = an
        all_years.append(df_m[["Année", "mois", "Température", "Pluie (mm)", "ET0 (mm)"]])
    train = pd.concat(all_years, ignore_index=True)
    angle = 2 * np.pi * (train["mois"] - 1) / 12.0
    train["sin_saison"] = np.sin(angle)
    train["cos_saison"] = np.cos(angle)
    return train

def regression_lineaire_maison(X, y):
    Xb = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return beta

def predire_reg_lin_maison(X, beta):
    Xb = np.column_stack([np.ones(len(X)), X])
    return Xb @ beta

def faire_projection_simple(train_df, nom_cible, an_cible=2044):
    X_train = train_df[["Année", "sin_saison", "cos_saison"]].to_numpy()
    y = train_df[nom_cible].to_numpy()
    beta = regression_lineaire_maison(X_train, y)
    mois = np.arange(1, 13)
    angle = 2 * np.pi * (mois - 1) / 12.0
    X_pred = np.column_stack([
        np.full(12, an_cible, dtype=float),
        np.sin(angle),
        np.cos(angle)
    ])
    y_hat = predire_reg_lin_maison(X_pred, beta)
    return pd.DataFrame({
        "mois": mois,
        "Nom du Mois": [ORDRE_MOIS[m-1] for m in mois],
        nom_cible: np.round(y_hat, 1)
    })

def metrique_mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

# ---------------------------------------------------------
# Télécharger les deux années de base
# ---------------------------------------------------------
with st.spinner("⏳ Téléchargement des données 2004 et 2024..."):
    df_2004 = telecharger_et_preparer_donnees(2004)
    df_2024 = telecharger_et_preparer_donnees(2024)

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
onglet_comp, onglet_annee, onglet_proj = st.tabs([
    "🆚 Comparaison 2004 vs 2024",
    "📅 Une seule année",
    "🔮 Prédictions 2044"
])

# =========================================================
# 🆚 Comparaison 2004 vs 2024
# =========================================================
with onglet_comp:
    st.subheader("Chiffres annuels : 2004 vs 2024")

    c1, c2, c3 = st.columns(3)
    temp_2024, temp_2004 = df_2024["Température"].mean(), df_2004["Température"].mean()
    pluie_2024, pluie_2004 = df_2024["Pluie (mm)"].sum(), df_2004["Pluie (mm)"].sum()
    et0_2024, et0_2004 = df_2024["ET0 (mm)"].sum(), df_2004["ET0 (mm)"].sum()

    c1.metric("Moyenne Temp. 2024", f"{temp_2024:.1f} °C", f"{(temp_2024-temp_2004):+.1f} vs 2004")
    c2.metric("Total Pluie 2024", f"{pluie_2024:.1f} mm", f"{(pluie_2024-pluie_2004):+.1f} vs 2004")
    c3.metric("Total ET0 2024", f"{et0_2024:.1f} mm", f"{(et0_2024-et0_2004):+.1f} vs 2004")

    # Températures
    st.markdown("#### 🌡️ Températures mensuelles (2004 vs 2024)")
    fig1, ax1 = plt.subplots(figsize=(3.5, 2.2))
    ax1.plot(df_2004["Nom du Mois"], df_2004["Température"], marker="o", label="2004")
    ax1.plot(df_2024["Nom du Mois"], df_2024["Température"], marker="o", label="2024")
    ax1.set_xlabel("Mois"); ax1.set_ylabel("Température (°C)")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig1, clear_figure=True, use_container_width=False)
    plt.close(fig1)

    # Pluies
    st.markdown("#### 🌧️ Pluies mensuelles (2004 vs 2024)")
    width = 0.4
    x = np.arange(12)
    fig2, ax2 = plt.subplots(figsize=(3.5, 2.2))
    ax2.bar(x - width/2, df_2004["Pluie (mm)"], width=width, label="2004")
    ax2.bar(x + width/2, df_2024["Pluie (mm)"], width=width, label="2024")
    ax2.set_xticks(x, ORDRE_MOIS, rotation=45)
    ax2.set_ylabel("Pluie (mm)")
    ax2.legend(); ax2.grid(True, axis="y", alpha=0.3)
    st.pyplot(fig2, clear_figure=True, use_container_width=False)
    plt.close(fig2)

# =========================================================
# 📅 Une seule année
# =========================================================
with onglet_annee:
    annee_choisie = st.radio("Choisis l'année :", [2004, 2024], horizontal=True)
    df_a = df_2004 if annee_choisie == 2004 else df_2024
    st.markdown(f"### 📅 Données {annee_choisie}")
    st.dataframe(df_a, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🌡️ Température par mois")
        fig3, ax3 = plt.subplots(figsize=(3.5, 2.2))
        ax3.plot(df_a["Nom du Mois"], df_a["Température"], marker="o")
        ax3.set_xlabel("Mois"); ax3.set_ylabel("Température (°C)")
        ax3.grid(True, alpha=0.3); plt.xticks(rotation=45)
        st.pyplot(fig3, clear_figure=True, use_container_width=False)
        plt.close(fig3)
    with col2:
        st.markdown("#### 🌧️ Pluie par mois")
        fig4, ax4 = plt.subplots(figsize=(3.5, 2.2))
        ax4.bar(df_a["Nom du Mois"], df_a["Pluie (mm)"])
        ax4.set_xlabel("Mois"); ax4.set_ylabel("Pluie (mm)")
        ax4.grid(True, axis="y", alpha=0.3); plt.xticks(rotation=45)
        st.pyplot(fig4, clear_figure=True, use_container_width=False)
        plt.close(fig4)

    st.markdown("#### 💧 ET0 Totale Progressive (cumul)")
    fig5, ax5 = plt.subplots(figsize=(3.5, 2.2))
    ax5.fill_between(df_a["Nom du Mois"], df_a["ET0 Totale Progressive (mm)"], alpha=0.6, step="mid")
    ax5.set_xlabel("Mois"); ax5.set_ylabel("ET0 cumulée (mm)")
    ax5.grid(True, alpha=0.3); plt.xticks(rotation=45)
    st.pyplot(fig5, clear_figure=True, use_container_width=False)
    plt.close(fig5)

# =========================================================
# 🔮 Prédictions 2044
# =========================================================
with onglet_proj:
    st.subheader("🔮 Projection 2044 (régression linéaire « maison »)")
    st.caption("Modèle simple : cible ~ Année + sin(mois) + cos(mois).")

    with st.spinner("⏳ Calcul en cours..."):
        train = preparer_donnees_pour_ml(2004, 2024)
        t2044 = faire_projection_simple(train, "Température", 2044)
        p2044 = faire_projection_simple(train, "Pluie (mm)", 2044)
        e2044 = faire_projection_simple(train, "ET0 (mm)", 2044)

    df_2044 = (
        t2044.merge(p2044, on=["mois","Nom du Mois"])
             .merge(e2044, on=["mois","Nom du Mois"])
             .sort_values("mois")
             .reset_index(drop=True)
    )
    df_2044["Pluie Totale Progressive (mm)"] = df_2044["Pluie (mm)"].cumsum().round(1)
    df_2044["ET0 Totale Progressive (mm)"] = df_2044["ET0 (mm)"].cumsum().round(1)
    st.dataframe(df_2044, use_container_width=True)

    st.markdown("#### 🌡️ Températures 2004 / 2024 / 2044")
    fig6, ax6 = plt.subplots(figsize=(3.5, 2.2))
    ax6.plot(df_2004["Nom du Mois"], df_2004["Température"], marker="o", label="2004")
    ax6.plot(df_2024["Nom du Mois"], df_2024["Température"], marker="o", label="2024")
    ax6.plot(df_2044["Nom du Mois"], df_2044["Température"], marker="o", label="2044 (proj.)")
    ax6.set_xlabel("Mois"); ax6.set_ylabel("Température (°C)")
    ax6.legend(); ax6.grid(True, alpha=0.3); plt.xticks(rotation=45)
    st.pyplot(fig6, clear_figure=True, use_container_width=False)
    plt.close(fig6)

    st.markdown("#### 🌧️ Pluies 2004 / 2024 / 2044")
    width = 0.25
    x = np.arange(12)
    fig7, ax7 = plt.subplots(figsize=(3.5, 2.2))
    ax7.bar(x - width, df_2004["Pluie (mm)"], width=width, label="2004")
    ax7.bar(x, df_2024["Pluie (mm)"], width=width, label="2024")
    ax7.bar(x + width, df_2044["Pluie (mm)"], width=width, label="2044 (proj.)")
    ax7.set_xticks(x, ORDRE_MOIS, rotation=45)
    ax7.set_ylabel("Pluie (mm)")
    ax7.legend(); ax7.grid(True, axis="y", alpha=0.3)
    st.pyplot(fig7, clear_figure=True, use_container_width=False)
    plt.close(fig7)
