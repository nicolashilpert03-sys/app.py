# =========================================================
# 🌤️ Mon Application Météo Simple — Beauvais 
# =========================================================

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Config & style global
# -----------------------------
st.set_page_config(page_title="Climat Beauvais", layout="wide", page_icon="🌦️")
st.markdown("<h1 style='text-align:center;'>🌤️ Climat de Beauvais </h1>", unsafe_allow_html=True)
st.write("Données historiques via Open-Meteo (2004 & 2024), comparaisons et projection 2044 (régression linéaire maison + saisonnalité).")

# style matplotlib (tailles, grilles, légendes)
plt.rcParams.update({
    "figure.dpi": 200,
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

ORDRE_MOIS = ["Janvier","Février","Mars","Avril","Mai","Juin",
              "Juillet","Août","Septembre","Octobre","Novembre","Décembre"]
MOIS_ABR   = ["Jan","Fév","Mar","Avr","Mai","Juin","Juil","Août","Sep","Oct","Nov","Déc"]
NOMS_MOIS  = {i+1: ORDRE_MOIS[i] for i in range(12)}
SMALL_FIG  = (4.0, 2.6)  # compact mais lisible

def _prettify_ax(ax, y_label=""):
    # bords plus clean
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # labels
    ax.set_ylabel(y_label)
    # ticks x = mois abrégés bien droits
    ax.set_xticklabels(MOIS_ABR, rotation=0, ha="center")
    # marges & agencement
    ax.margins(x=0.02)
    # espace pour légende si besoin
    plt.tight_layout()

def _legend(ax):
    # légende compacte, semi-déportée
    leg = ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=True, framealpha=0.85)
    leg.get_frame().set_linewidth(0.5)

# -----------------------------
# Données
# -----------------------------
def telecharger_journalier(annee, lat=49.43, lon=2.08, tz="Europe/Paris"):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": f"{annee}-01-01", "end_date": f"{annee}-12-31",
        "daily": "temperature_2m_mean,precipitation_sum,et0_fao_evapotranspiration",
        "timezone": tz
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    d = r.json()["daily"]
    df = pd.DataFrame({
        "date": pd.to_datetime(d["time"]),
        "Température": d["temperature_2m_mean"],
        "Pluie (mm)": d["precipitation_sum"],
        "ET0 (mm)": d["et0_fao_evapotranspiration"],
    })
    df["mois"] = df["date"].dt.month
    return df

def agregation_mensuelle(df):
    dfm = (df.groupby("mois", as_index=False)
             .agg({"Température":"mean","Pluie (mm)":"sum","ET0 (mm)":"sum"}))
    dfm["Nom du Mois"] = dfm["mois"].map(NOMS_MOIS)
    dfm["Pluie Totale Progressive (mm)"] = dfm["Pluie (mm)"].cumsum()
    dfm["ET0 Totale Progressive (mm)"]   = dfm["ET0 (mm)"].cumsum()
    return dfm.round(1)

def telecharger_et_preparer_donnees(annee):
    return agregation_mensuelle(telecharger_journalier(annee))

def preparer_donnees_pour_ml(an_debut=2004, an_fin=2024):
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

# régression 
def regression_lineaire_maison(X, y):
    Xb = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return beta

def predire_reg_lin_maison(X, beta):
    Xb = np.column_stack([np.ones(len(X)), X])
    return Xb @ beta

def faire_projection_simple(train_df, nom_cible, an_cible=2044):
    X = train_df[["Année","sin_saison","cos_saison"]].to_numpy()
    y = train_df[nom_cible].to_numpy()
    beta = regression_lineaire_maison(X, y)
    mois = np.arange(1,13)
    angle = 2*np.pi*(mois-1)/12
    Xp = np.column_stack([np.full(12, an_cible, dtype=float), np.sin(angle), np.cos(angle)])
    yhat = predire_reg_lin_maison(Xp, beta)
    return pd.DataFrame({
        "mois": mois,
        "Nom du Mois": [ORDRE_MOIS[m-1] for m in mois],
        nom_cible: np.round(yhat, 1)
    })

def metrique_mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

# -----------------------------
# Chargement
# -----------------------------
with st.spinner("⏳ Téléchargement 2004 & 2024..."):
    df_2004 = telecharger_et_preparer_donnees(2004)
    df_2024 = telecharger_et_preparer_donnees(2024)

# -----------------------------
# Onglets
# -----------------------------
onglet_comp, onglet_annee, onglet_proj = st.tabs([
    "🆚 Comparaison 2004 vs 2024",
    "📅 Une seule année",
    "🔮 Prédictions 2044"
])

# =========================================================
# 🆚 Comparaison
# =========================================================
with onglet_comp:
    c1, c2, c3 = st.columns(3)
    c1.metric("Moyenne Temp. 2024", f"{df_2024['Température'].mean():.1f} °C",
              f"{(df_2024['Température'].mean()-df_2004['Température'].mean()):+.1f} vs 2004")
    c2.metric("Total Pluie 2024", f"{df_2024['Pluie (mm)'].sum():.1f} mm",
              f"{(df_2024['Pluie (mm)'].sum()-df_2004['Pluie (mm)'].sum()):+.1f} vs 2004")
    c3.metric("Total ET0 2024", f"{df_2024['ET0 (mm)'].sum():.1f} mm",
              f"{(df_2024['ET0 (mm)'].sum()-df_2004['ET0 (mm)'].sum()):+.1f} vs 2004")

    # Températures
    st.markdown("#### 🌡️ Températures mensuelles (2004 vs 2024)")
    fig, ax = plt.subplots(figsize=SMALL_FIG)
    ax.plot(df_2004["Nom du Mois"], df_2004["Température"], marker="o", ms=3.5, lw=1.6, label="2004")
    ax.plot(df_2024["Nom du Mois"], df_2024["Température"], marker="o", ms=3.5, lw=1.6, label="2024")
    ax.set_xlabel("Mois"); _prettify_ax(ax, "Température (°C)"); _legend(ax)
    st.pyplot(fig, use_container_width=False, clear_figure=True); plt.close(fig)

    # Pluies
    st.markdown("#### 🌧️ Pluies mensuelles (2004 vs 2024)")
    x = np.arange(12); width = 0.38
    fig, ax = plt.subplots(figsize=SMALL_FIG)
    ax.bar(x - width/2, df_2004["Pluie (mm)"], width=width, label="2004")
    ax.bar(x + width/2, df_2024["Pluie (mm)"], width=width, label="2024")
    ax.set_xticks(x, MOIS_ABR)
    ax.set_xlabel("Mois"); _prettify_ax(ax, "Pluie (mm)"); _legend(ax)
    st.pyplot(fig, use_container_width=False, clear_figure=True); plt.close(fig)

# =========================================================
# 📅 Une seule année
# =========================================================
with onglet_annee:
    annee = st.radio("Choisis l'année :", [2004, 2024], horizontal=True)
    df = df_2004 if annee == 2004 else df_2024
    st.dataframe(df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🌡️ Température par mois")
        fig, ax = plt.subplots(figsize=SMALL_FIG)
        ax.plot(df["Nom du Mois"], df["Température"], marker="o", ms=3.5, lw=1.6)
        ax.set_xlabel("Mois"); _prettify_ax(ax, "Température (°C)")
        st.pyplot(fig, use_container_width=False, clear_figure=True); plt.close(fig)
    with col2:
        st.markdown("#### 🌧️ Pluie par mois")
        fig, ax = plt.subplots(figsize=SMALL_FIG)
        ax.bar(df["Nom du Mois"], df["Pluie (mm)"])
        ax.set_xlabel("Mois"); _prettify_ax(ax, "Pluie (mm)")
        st.pyplot(fig, use_container_width=False, clear_figure=True); plt.close(fig)

    st.markdown("#### 💧 ET0 Totale Progressive (cumul)")
    fig, ax = plt.subplots(figsize=SMALL_FIG)
    ax.fill_between(df["Nom du Mois"], df["ET0 Totale Progressive (mm)"], alpha=0.6, step="mid")
    ax.set_xlabel("Mois"); _prettify_ax(ax, "ET0 cumulée (mm)")
    st.pyplot(fig, use_container_width=False, clear_figure=True); plt.close(fig)

# =========================================================
# 🔮 Prédictions 2044
# =========================================================
with onglet_proj:
    st.subheader("🔮 Projection 2044 (régression linéaire )")
    with st.spinner("⏳ Calcul…"):
        train = preparer_donnees_pour_ml(2004, 2024)
        t2044 = faire_projection_simple(train, "Température", 2044)
        p2044 = faire_projection_simple(train, "Pluie (mm)", 2044)
        e2044 = faire_projection_simple(train, "ET0 (mm)", 2044)

    df_2044 = (t2044.merge(p2044, on=["mois","Nom du Mois"])
                     .merge(e2044, on=["mois","Nom du Mois"]))
    df_2044["Pluie Totale Progressive (mm)"] = df_2044["Pluie (mm)"].cumsum().round(1)
    df_2044["ET0 Totale Progressive (mm)"]   = df_2044["ET0 (mm)"].cumsum().round(1)
    st.dataframe(df_2044, use_container_width=True)

    # Températures comparées
    st.markdown("#### 🌡️ 2004 / 2024 / 2044")
    fig, ax = plt.subplots(figsize=SMALL_FIG)
    ax.plot(df_2004["Nom du Mois"], df_2004["Température"], marker="o", ms=3.5, lw=1.6, label="2004")
    ax.plot(df_2024["Nom du Mois"], df_2024["Température"], marker="o", ms=3.5, lw=1.6, label="2024")
    ax.plot(df_2044["Nom du Mois"], df_2044["Température"], marker="o", ms=3.5, lw=1.6, label="2044 (proj.)")
    ax.set_xlabel("Mois"); _prettify_ax(ax, "Température (°C)"); _legend(ax)
    st.pyplot(fig, use_container_width=False, clear_figure=True); plt.close(fig)

    # Pluies comparées
    st.markdown("#### 🌧️ 2004 / 2024 / 2044")
    x = np.arange(12); width = 0.25
    fig, ax = plt.subplots(figsize=SMALL_FIG)
    ax.bar(x - width, df_2004["Pluie (mm)"], width=width, label="2004")
    ax.bar(x,         df_2024["Pluie (mm)"], width=width, label="2024")
    ax.bar(x + width, df_2044["Pluie (mm)"], width=width, label="2044 (proj.)")
    ax.set_xticks(x, MOIS_ABR)
    ax.set_xlabel("Mois"); _prettify_ax(ax, "Pluie (mm)"); _legend(ax)
    st.pyplot(fig, use_container_width=False, clear_figure=True); plt.close(fig)
