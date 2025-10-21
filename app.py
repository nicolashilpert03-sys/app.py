# =========================================================
# 🌤️ Application Streamlit — Climat de Beauvais (2004 vs 2024)
# =========================================================
# Prérequis :
# pip install streamlit openmeteo-requests requests-cache retry-requests pandas numpy altair scikit-learn cdsapi xarray netCDF4

import streamlit as st
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import altair as alt

# ---------------------------------------------------------
# Configuration générale de la page
# ---------------------------------------------------------
st.set_page_config(page_title="Climat Beauvais", layout="wide", page_icon="🌦️")
st.markdown("<h1 style='text-align:center;'>🌤️ Climat de Beauvais — Comparaison 2004 / 2024</h1>", unsafe_allow_html=True)
st.write("Analyse basée sur les données **Open-Meteo Archive API** (quotidiennes agrégées par mois).")

# ---------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------
@st.cache_data
def charger_donnees(annee: int):
    """Charge et prépare les données météo pour une année donnée."""
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 49.4333,
        "longitude": 2.0833,
        "start_date": f"{annee}-01-01",
        "end_date": f"{annee}-12-31",
        "daily": [
            "temperature_2m_mean",
            "weather_code",
            "precipitation_sum",
            "wind_direction_10m_dominant",
            "et0_fao_evapotranspiration"
        ],
        "timezone": "Europe/Paris",
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    daily = response.Daily()

    def deg_to_cardinal(deg):
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        ix = int((deg + 22.5) // 45) % 8
        return directions[ix]

    # Extraction et préparation
    temperature = daily.Variables(0).ValuesAsNumpy()
    weather_code = daily.Variables(1).ValuesAsNumpy()
    precipitations = daily.Variables(2).ValuesAsNumpy()
    wind_direction = daily.Variables(3).ValuesAsNumpy()
    evapotranspiration = daily.Variables(4).ValuesAsNumpy()

    t0 = pd.to_datetime(daily.Time(), unit="s", utc=True).tz_convert("Europe/Paris")
    t1_excl = pd.to_datetime(daily.TimeEnd(), unit="s", utc=True).tz_convert("Europe/Paris")
    dt = pd.Timedelta(seconds=daily.Interval())
    dates = pd.date_range(start=t0, end=t1_excl - dt, freq=dt)

    df = pd.DataFrame({
        "date": dates,
        "temperature_2m_mean": temperature,
        "weather_code": weather_code,
        "precipitation_sum": precipitations,
        "wind_direction_10m_dominant": wind_direction,
        "et0_fao_evapotranspiration": evapotranspiration
    })

    # 🔹 On garde uniquement les données de janvier à décembre de l’année exacte
    df = df[df["date"].dt.year == annee]
    df["mois"] = df["date"].dt.month
    df = df[(df["mois"] >= 1) & (df["mois"] <= 12)]

    def mode_as_int(series):
        m = series.mode()
        return int(m.iloc[0]) if not m.empty else int(round(series.iloc[0]))

    df_mensuel = df.groupby("mois").agg({
        "temperature_2m_mean": "mean",
        "precipitation_sum": "sum",
        "et0_fao_evapotranspiration": "sum",
        "wind_direction_10m_dominant": mode_as_int,
        "weather_code": mode_as_int
    }).reset_index()

    # 🔹 Ajout du nom du mois en français
    mois_noms = {
        1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
        5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
        9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"
    }
    df_mensuel["Mois (nom)"] = df_mensuel["mois"].map(mois_noms)

    # Nettoyage et calculs
    df_mensuel["wind_direction_10m_dominant"] = df_mensuel["wind_direction_10m_dominant"].apply(deg_to_cardinal)
    df_mensuel["temperature_2m_mean"] = df_mensuel["temperature_2m_mean"].round(1)
    df_mensuel["precipitation_sum"] = df_mensuel["precipitation_sum"].round(1)
    df_mensuel["et0_fao_evapotranspiration"] = df_mensuel["et0_fao_evapotranspiration"].round(1)

    df_mensuel["Précipitations cumulées (mm)"] = df_mensuel["precipitation_sum"].cumsum()
    df_mensuel["Evapotranspiration cumulée (mm)"] = df_mensuel["et0_fao_evapotranspiration"].cumsum()

    df_mensuel.rename(columns={
        "mois": "Mois (numéro)",
        "temperature_2m_mean": "Température moyenne (°C)",
        "precipitation_sum": "Précipitations totales (mm)",
        "et0_fao_evapotranspiration": "Evapotranspiration (mm)",
        "wind_direction_10m_dominant": "Direction du vent dominante"
    }, inplace=True)

    # 🔹 Réorganisation + tri forcé janvier → décembre
    df_mensuel = df_mensuel.sort_values("Mois (numéro)").reset_index(drop=True)
    df_mensuel = df_mensuel[[
        "Mois (numéro)",
        "Mois (nom)",
        "Température moyenne (°C)",
        "Précipitations totales (mm)",
        "Précipitations cumulées (mm)",
        "Evapotranspiration (mm)",
        "Evapotranspiration cumulée (mm)",
        "Direction du vent dominante"
    ]]

    return df_mensuel

# ---------------------------------------------------------
# Utilitaires ML : construction du dataset 2004→2024 et projection 2044
# ---------------------------------------------------------
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

@st.cache_data(show_spinner=False)
def charger_toutes_annees(debut=2004, fin=2024):
    """Charge et concatène les données mensuelles de toutes les années [debut..fin]."""
    frames = []
    for an in range(debut, fin + 1):
        df = charger_donnees(an).copy()
        df["Année"] = an
        frames.append(df[[
            "Année", "Mois (numéro)", "Mois (nom)",
            "Température moyenne (°C)", "Précipitations totales (mm)", "Evapotranspiration (mm)"
        ]])
    big = pd.concat(frames, ignore_index=True)
    # Ajout des features saisonnières sin/cos
    angle = 2 * np.pi * (big["Mois (numéro)"] - 1) / 12.0
    big["sin_mois"], big["cos_mois"] = np.sin(angle), np.cos(angle)
    return big

@st.cache_data(show_spinner=False)
def proj_ml_2044(df_all_years: pd.DataFrame, col: str, deg: int = 2, alpha: float = 1.0) -> pd.DataFrame:
    """Projection 2044 par régression polynomiale sur l'année + sin/cos(mois) avec Ridge."""
    # Dataset d'entraînement
    X = df_all_years[["Année", "sin_mois", "cos_mois"]].copy()
    y = df_all_years[col].values
    # Construction des features polynomiales sur Année (sin/cos restent linéaires)
    X_poly = X.copy()
    for d in range(2, deg + 1):
        X_poly[f"Année^{d}"] = X_poly["Année"] ** d
    feats = X_poly.columns
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha, random_state=42))
    ])
    pipe.fit(X_poly, y)

    # Données 2044 pour les 12 mois
    mois = np.arange(1, 13)
    angle = 2 * np.pi * (mois - 1) / 12.0
    df_2044 = pd.DataFrame({
        "Année": 2044,
        "Mois (numéro)": mois,
        "Mois (nom)": [
            "Janvier","Février","Mars","Avril","Mai","Juin",
            "Juillet","Août","Septembre","Octobre","Novembre","Décembre"
        ],
        "sin_mois": np.sin(angle),
        "cos_mois": np.cos(angle)
    })
    X_new = df_2044[["Année", "sin_mois", "cos_mois"]].copy()
    for d in range(2, deg + 1):
        X_new[f"Année^{d}"] = X_new["Année"] ** d
    X_new = X_new[[c for c in feats]]

    y_hat = pipe.predict(X_new)
    out = df_2044[["Mois (numéro)", "Mois (nom)"]].copy()
    out[col] = np.round(y_hat, 1)
    return out

# ---------------------------------------------------------
# Chargement des données + préparation comparaison
# ---------------------------------------------------------
with st.spinner("Chargement des données 2004 et 2024..."):
    df_2004 = charger_donnees(2004)
    df_2024 = charger_donnees(2024)

# Ajout d'un identifiant d'année et concaténation
_df_2004 = df_2004.copy(); _df_2004["Année"] = 2004
_df_2024 = df_2024.copy(); _df_2024["Année"] = 2024

df_all = pd.concat([_df_2004, _df_2024], ignore_index=True)

# Ordre des mois en français
ordre_mois = ["Janvier","Février","Mars","Avril","Mai","Juin","Juillet","Août","Septembre","Octobre","Novembre","Décembre"]
df_all["Mois (nom)"] = pd.Categorical(df_all["Mois (nom)"], categories=ordre_mois, ordered=True)

# Table de comparaison (2004 vs 2024) par mois
cols_base = [
    "Mois (numéro)", "Mois (nom)",
    "Température moyenne (°C)",
    "Précipitations totales (mm)",
    "Evapotranspiration (mm)"
]
left = df_2004[cols_base].rename(columns={
    "Température moyenne (°C)": "Température 2004 (°C)",
    "Précipitations totales (mm)": "Précipitations 2004 (mm)",
    "Evapotranspiration (mm)": "ET0 2004 (mm)",
})
right = df_2024[cols_base].rename(columns={
    "Température moyenne (°C)": "Température 2024 (°C)",
    "Précipitations totales (mm)": "Précipitations 2024 (mm)",
    "Evapotranspiration (mm)": "ET0 2024 (mm)",
})
df_comp = left.merge(right, on=["Mois (numéro)", "Mois (nom)"]).sort_values("Mois (numéro)")

# Deltas
df_comp["Δ Temp (°C)"] = (df_comp["Température 2024 (°C)"] - df_comp["Température 2004 (°C)"]).round(1)
df_comp["Δ Précip (mm)"] = (df_comp["Précipitations 2024 (mm)"] - df_comp["Précipitations 2004 (mm)"]).round(1)
df_comp["Δ ET0 (mm)"] = (df_comp["ET0 2024 (mm)"] - df_comp["ET0 2004 (mm)"]).round(1)

# ---------------------------------------------------------
# Mise en page : onglets (Comparaison vs Vue par année)
# ---------------------------------------------------------
onglet_comp, onglet_annee, onglet_proj, onglet_climat = st.tabs(["🆚 Comparaison 2004 vs 2024", "📅 Vue par année", "🔮 Projection 2044", "🌍 Modèle climatique 2044"])

with onglet_comp:
    # KPIs annuels
    kpi = {}
    kpi["Temp moy annuelle 2004"] = round(df_2004["Température moyenne (°C)"].mean(), 1)
    kpi["Temp moy annuelle 2024"] = round(df_2024["Température moyenne (°C)"].mean(), 1)
    kpi["Δ Temp annuelle"] = round(kpi["Temp moy annuelle 2024"] - kpi["Temp moy annuelle 2004"], 1)

    kpi["Précip tot annuelles 2004"] = round(df_2004["Précipitations totales (mm)"].sum(), 1)
    kpi["Précip tot annuelles 2024"] = round(df_2024["Précipitations totales (mm)"].sum(), 1)
    kpi["Δ Précip annuelles"] = round(kpi["Précip tot annuelles 2024"] - kpi["Précip tot annuelles 2004"], 1)

    kpi["ET0 tot annuelle 2004"] = round(df_2004["Evapotranspiration (mm)"].sum(), 1)
    kpi["ET0 tot annuelle 2024"] = round(df_2024["Evapotranspiration (mm)"].sum(), 1)
    kpi["Δ ET0 annuelle"] = round(kpi["ET0 tot annuelle 2024"] - kpi["ET0 tot annuelle 2004"], 1)

    st.markdown("### 🧮 Indicateurs annuels clés")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Température moyenne (°C)", kpi["Temp moy annuelle 2024"], delta=kpi["Δ Temp annuelle"])
    with c2:
        st.metric("Précipitations totales (mm)", kpi["Précip tot annuelles 2024"], delta=kpi["Δ Précip annuelles"])
    with c3:
        st.metric("ET0 totale (mm)", kpi["ET0 tot annuelle 2024"], delta=kpi["Δ ET0 annuelle"])

    # Sélecteur de métrique
    st.markdown("### 📊 Comparaison 2004 vs 2024 par mois")
    metrique_label = st.radio(
        "Choisissez la métrique à comparer :",
        ["Température moyenne (°C)", "Précipitations totales (mm)", "Evapotranspiration (mm)"],
        horizontal=True
    )

    base = alt.Chart(df_all).encode(
        x=alt.X("Mois (nom):O", sort=ordre_mois),
        color=alt.Color("Année:N")
    )

    if metrique_label == "Température moyenne (°C)":
        chart = base.mark_line(point=True).encode(
            y=alt.Y("mean(Température moyenne (°C)):Q"),
            tooltip=["Année:N", "Mois (nom):O", alt.Tooltip("Température moyenne (°C):Q", format=".1f")]
        )
    else:
        chart = base.mark_bar().encode(
            y=alt.Y(f"sum({metrique_label}):Q"),
            column=alt.Column("Année:N", header=alt.Header(title=None)),
            tooltip=["Année:N", "Mois (nom):O", alt.Tooltip(f"{metrique_label}:Q", format=".1f")]
        )

    st.altair_chart(chart, use_container_width=True)

    # Aire cumulée ET0
    st.markdown("#### 💧 Évolution cumulée (ET0)")
    df_cum = df_all.copy()
    df_cum["ET0 cumulée (mm)"] = df_cum.groupby("Année")["Evapotranspiration (mm)"].cumsum()
    area = alt.Chart(df_cum).mark_area(opacity=0.4).encode(
        x=alt.X("Mois (nom):O", sort=ordre_mois),
        y=alt.Y("ET0 cumulée (mm):Q"),
        color="Année:N",
        tooltip=["Année", "Mois (nom)", alt.Tooltip("ET0 cumulée (mm):Q", format=".1f")]
    )
    st.altair_chart(area, use_container_width=True)

    # Tableau de comparaison mensuelle
    st.markdown("### 🧾 Tableau de comparaison mensuelle (2004 vs 2024)")
    colonnes_aff = [
        "Mois (numéro)", "Mois (nom)",
        "Température 2004 (°C)", "Température 2024 (°C)", "Δ Temp (°C)",
        "Précipitations 2004 (mm)", "Précipitations 2024 (mm)", "Δ Précip (mm)",
        "ET0 2004 (mm)", "ET0 2024 (mm)", "Δ ET0 (mm)"
    ]
    st.dataframe(df_comp[colonnes_aff], use_container_width=True)

    # Téléchargements comparaison
    csv_bi = df_all.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Télécharger les données (2004 + 2024)",
        data=csv_bi,
        file_name="climat_beauvais_2004_2024.csv",
        mime="text/csv"
    )

    csv_comp = df_comp[colonnes_aff].to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Télécharger le tableau de comparaison (mois & deltas)",
        data=csv_comp,
        file_name="comparaison_2004_2024.csv",
        mime="text/csv"
    )

with onglet_annee:
    # Sélecteur d’année (vue simple)
    annee = st.radio("Choisissez une année à afficher :", [2004, 2024], horizontal=True)
    df_sel = df_2004 if annee == 2004 else df_2024

    # Affichage principal
    st.markdown(f"### 📅 Données mensuelles - {annee}")
    st.dataframe(df_sel, use_container_width=True)

    # Graphiques
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🌡️ Température moyenne mensuelle")
        chart_temp = alt.Chart(df_sel).mark_line(point=True).encode(
            x=alt.X("Mois (nom):O", sort=ordre_mois),
            y=alt.Y("Température moyenne (°C):Q"),
            tooltip=["Mois (nom)", "Température moyenne (°C)"]
        )
        st.altair_chart(chart_temp, use_container_width=True)

    with col2:
        st.markdown("#### 🌧️ Précipitations totales mensuelles")
        chart_precip = alt.Chart(df_sel).mark_bar().encode(
            x=alt.X("Mois (nom):O", sort=ordre_mois),
            y=alt.Y("Précipitations totales (mm):Q"),
            tooltip=["Mois (nom)", "Précipitations totales (mm)"]
        )
        st.altair_chart(chart_precip, use_container_width=True)

    st.markdown("#### 💧 Évapotranspiration cumulée annuelle")
    chart_et0 = alt.Chart(df_sel).mark_area(opacity=0.6).encode(
        x=alt.X("Mois (nom):O", sort=ordre_mois),
        y=alt.Y("Evapotranspiration cumulée (mm):Q"),
        tooltip=["Mois (nom)", "Evapotranspiration cumulée (mm)"]
    )
    st.altair_chart(chart_et0, use_container_width=True)

    # Téléchargement CSV spécifique à l'année
    csv = df_sel.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Télécharger les données {annee}",
        data=csv,
        file_name=f"climat_beauvais_{annee}.csv",
        mime="text/csv"
    )

    st.success("✅ Données prêtes et graphiques affichés (Janvier → Décembre) !")

with onglet_proj:
    st.markdown("### 🔮 Projection 2044 (Machine Learning : Ridge polynomiale + saisonnalité)")

    with st.spinner("Entraînement sur 2004→2024 et génération 2044..."):
        df_all_years = charger_toutes_annees(2004, 2024)
        temp_2044 = proj_ml_2044(df_all_years, "Température moyenne (°C)", deg=2, alpha=1.0)
        prec_2044 = proj_ml_2044(df_all_years, "Précipitations totales (mm)", deg=2, alpha=1.0)
        et0_2044  = proj_ml_2044(df_all_years, "Evapotranspiration (mm)", deg=2, alpha=1.0)

    df_2044 = temp_2044.merge(prec_2044, on=["Mois (numéro)", "Mois (nom)"]).merge(et0_2044, on=["Mois (numéro)", "Mois (nom)"])
    df_2044["Précipitations cumulées (mm)"] = df_2044["Précipitations totales (mm)"].cumsum().round(1)
    df_2044["Evapotranspiration cumulée (mm)"] = df_2044["Evapotranspiration (mm)"].cumsum().round(1)

    # KPIs projetés (ML)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Température moyenne annuelle prévue (°C)", round(df_2044["Température moyenne (°C)"].mean(), 1))
    with c2:
        st.metric("Précipitations totales annuelles prévues (mm)", round(df_2044["Précipitations totales (mm)"].sum(), 1))
    with c3:
        st.metric("ET0 totale annuelle prévue (mm)", round(df_2044["Evapotranspiration (mm)"].sum(), 1))

    # Table 2044
    st.markdown("#### 📅 Données mensuelles prévues – 2044 (ML)")
    st.dataframe(df_2044, use_container_width=True)

    # Graphiques comparatifs 2004 / 2024 / 2044 (ML)
    ordre_mois = ["Janvier","Février","Mars","Avril","Mai","Juin","Juillet","Août","Septembre","Octobre","Novembre","Décembre"]

    # Température
    df_plot_temp = pd.concat([
        df_2004[["Mois (nom)", "Température moyenne (°C)"]].assign(Année=2004),
        df_2024[["Mois (nom)", "Température moyenne (°C)"]].assign(Année=2024),
        df_2044[["Mois (nom)", "Température moyenne (°C)"]].assign(Année=2044)
    ])
    df_plot_temp["Mois (nom)"] = pd.Categorical(df_plot_temp["Mois (nom)"], categories=ordre_mois, ordered=True)
    st.markdown("#### 🌡️ Température moyenne mensuelle – comparaison 2004 / 2024 / 2044 (ML)")
    chart_t = alt.Chart(df_plot_temp).mark_line(point=True).encode(
        x=alt.X("Mois (nom):O", sort=ordre_mois),
        y=alt.Y("Température moyenne (°C):Q"),
        color=alt.Color("Année:N"),
        tooltip=["Année", "Mois (nom)", alt.Tooltip("Température moyenne (°C):Q", format=".1f")]
    )
    st.altair_chart(chart_t, use_container_width=True)

    # Précipitations
    df_plot_p = pd.concat([
        df_2004[["Mois (nom)", "Précipitations totales (mm)"]].assign(Année=2004),
        df_2024[["Mois (nom)", "Précipitations totales (mm)"]].assign(Année=2024),
        df_2044[["Mois (nom)", "Précipitations totales (mm)"]].assign(Année=2044)
    ])
    df_plot_p["Mois (nom)"] = pd.Categorical(df_plot_p["Mois (nom)"], categories=ordre_mois, ordered=True)
    st.markdown("#### 🌧️ Précipitations totales mensuelles – comparaison 2004 / 2024 / 2044 (ML)")
    chart_p = alt.Chart(df_plot_p).mark_bar().encode(
        x=alt.X("Mois (nom):O", sort=ordre_mois),
        y=alt.Y("Précipitations totales (mm):Q"),
        color=alt.Color("Année:N"),
        tooltip=["Année", "Mois (nom)", alt.Tooltip("Précipitations totales (mm):Q", format=".1f")]
    )
    st.altair_chart(chart_p, use_container_width=True)

    # ET0 cumulée
    df_plot_e = pd.concat([
        df_2004[["Mois (nom)", "Evapotranspiration cumulée (mm)"]].assign(Année=2004),
        df_2024[["Mois (nom)", "Evapotranspiration cumulée (mm)"]].assign(Année=2024),
        df_2044[["Mois (nom)", "Evapotranspiration cumulée (mm)"]].assign(Année=2044)
    ])
    df_plot_e["Mois (nom)"] = pd.Categorical(df_plot_e["Mois (nom)"], categories=ordre_mois, ordered=True)
    st.markdown("#### 💧 Évapotranspiration cumulée – comparaison 2004 / 2024 / 2044 (ML)")
    chart_e = alt.Chart(df_plot_e).mark_area(opacity=0.4).encode(
        x=alt.X("Mois (nom):O", sort=ordre_mois),
        y=alt.Y("Evapotranspiration cumulée (mm):Q"),
        color=alt.Color("Année:N"),
        tooltip=["Année", "Mois (nom)", alt.Tooltip("Evapotranspiration cumulée (mm):Q", format=".1f")]
    )
    st.altair_chart(chart_e, use_container_width=True)

    # Export
    csv_2044 = df_2044.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Télécharger les données projetées 2044 (ML)",
        data=csv_2044,
        file_name="projection_2044_ML.csv",
        mime="text/csv"
    )

# ---------------------------------------------------------
# 🌍 Onglet : Modèle climatique 2044 (CMIP6 via Copernicus CDS)
# ---------------------------------------------------------
with onglet_climat:
    st.markdown("### 🌍 Projection 2044 basée sur un modèle climatique (CMIP6 via Copernicus – Copernicus Climate Data Store)")
    st.caption("Deux options : 1) Télécharger automatiquement via l'API Copernicus (nécessite un compte et ~/.cdsapirc), 2) Importer un fichier NetCDF (.nc) que tu as déjà téléchargé.")

    # Sélection scénario et modèle (liste non exhaustive)
    colA, colB = st.columns(2)
    with colA:
        scenario_label = st.selectbox("Scénario d'émission (SSP)", ["SSP2-4.5", "SSP1-2.6", "SSP5-8.5"], index=0)
    with colB:
        model_label = st.selectbox("Modèle CMIP6", [
            "ACCESS-ESM1-5", "BCC-CSM2-MR", "CNRM-ESM2-1", "MPI-ESM1-2-HR", "MRI-ESM2-0"
        ], index=0)

    ssp_map = {"SSP1-2.6": "ssp126", "SSP2-4.5": "ssp245", "SSP5-8.5": "ssp585"}
    ssp = ssp_map[scenario_label]

    # Tentative de téléchargement via cdsapi (optionnelle)
    st.markdown("#### ⬇️ Option A — Télécharger depuis Copernicus")
    do_download = st.button("Télécharger 2044 (mensuel) depuis CDS")

    ds = None
    if do_download:
        try:
            import cdsapi
            from pathlib import Path
            out_path = Path("projection_2044_beauvais.nc")
            c = cdsapi.Client()
            c.retrieve(
                "projections-cmip6",
                {
                    "format": "netcdf",
                    "temporal_resolution": "monthly",
                    "experiment": ssp,
                    "variable": ["2m_temperature", "precipitation"],
                    "model": [model_label],
                    "year": ["2044"],
                    "month": [f"{m:02d}" for m in range(1, 13)],
                    # zone autour de Beauvais (N, W, S, E)
                    "area": [49.6, 2.0, 49.3, 2.3],
                },
                str(out_path)
            )
            import xarray as xr
            ds = xr.open_dataset(out_path)
            st.success("Fichier téléchargé : projection_2044_beauvais.nc")
        except ModuleNotFoundError:
            st.error("Le module 'cdsapi' n'est pas installé. Ajoute 'cdsapi' dans requirements.txt et configure ~/.cdsapirc.")
        except Exception as e:
            st.warning(f"Téléchargement CDS impossible : {e}")

    # Option B : Upload manuel d'un NetCDF
    st.markdown("#### 📤 Option B — Importer un fichier NetCDF (.nc)")
    uploaded = st.file_uploader("Dépose un fichier NetCDF CMIP6 (mensuel, 2044) contenant température/ précipitations.", type=["nc"]) 
    if uploaded is not None and ds is None:
        try:
            import xarray as xr
            ds = xr.open_dataset(uploaded)
            st.success("Fichier NetCDF chargé avec succès.")
        except ModuleNotFoundError:
            st.error("Le module 'xarray' n'est pas installé. Ajoute 'xarray' et 'netCDF4' dans requirements.txt.")
        except Exception as e:
            st.warning(f"Lecture du NetCDF impossible : {e}")

    # Lecture + agrégation spatiale et conversion d'unités
    if ds is not None:
        import pandas as pd
        import numpy as np
        # Tentatives de noms de variables (peuvent varier selon la source)
        var_temp_candidates = ["tas", "tas_monthly", "2m_temperature", "t2m"]
        var_prec_candidates = ["pr", "precipitation", "pr_monthly"]

        def pick_var(ds, candidates):
            for v in candidates:
                if v in ds.variables:
                    return v
            raise KeyError(f"Variables candidates non trouvées : {candidates}")

        try:
            vtemp = pick_var(ds, var_temp_candidates)
            vprec = pick_var(ds, var_prec_candidates)
        except Exception as e:
            st.error(f"Variables climatiques non trouvées dans le fichier : {e}")
            st.stop()

        # Agrégation spatiale (moyenne sur la tuile) si lat/lon présents
        dims = list(ds[vtemp].dims)
        da_temp = ds[vtemp]
        da_prec = ds[vprec]
        if "latitude" in dims and "longitude" in dims:
            da_temp = da_temp.mean(dim=["latitude", "longitude"]) 
            da_prec = da_prec.mean(dim=["latitude", "longitude"]) 
        elif "lat" in dims and "lon" in dims:
            da_temp = da_temp.mean(dim=["lat", "lon"]) 
            da_prec = da_prec.mean(dim=["lat", "lon"]) 

        # Convertir temps en index Pandas
        time = pd.to_datetime(ds["time"].values)
        mois_num = time.month

        # Conversion d'unités
        # Température : Kelvin -> °C si valeurs > 100
        temp_vals = da_temp.values
        if np.nanmean(temp_vals) > 100:
            temp_vals = temp_vals - 273.15
        # Précipitations : kg m^-2 s^-1 -> mm/mois (1 kg/m^2 = 1 mm)
        prec_vals = da_prec.values
        if np.nanmean(prec_vals) <
