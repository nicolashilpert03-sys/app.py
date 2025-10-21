# =========================================================
# 🌤️ Application Streamlit — Climat de Beauvais (2004 vs 2024)
# =========================================================
# Prérequis :
# pip install streamlit openmeteo-requests requests-cache retry-requests pandas numpy altair

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
# Chargement des données
# ---------------------------------------------------------
with st.spinner("Chargement des données 2004 et 2024..."):
    df_2004 = charger_donnees(2004)
    df_2024 = charger_donnees(2024)

# ---------------------------------------------------------
# Sélecteur d’année
# ---------------------------------------------------------
annee = st.radio("Choisissez une année à afficher :", [2004, 2024], horizontal=True)
df_sel = df_2004 if annee == 2004 else df_2024

# ---------------------------------------------------------
# Affichage principal
# ---------------------------------------------------------
st.markdown(f"### 📅 Données mensuelles - {annee}")
st.dataframe(df_sel, use_container_width=True)

# ---------------------------------------------------------
# Graphiques
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🌡️ Température moyenne mensuelle")
    chart_temp = alt.Chart(df_sel).mark_line(point=True).encode(
        x=alt.X("Mois (nom):O", sort=["Janvier","Février","Mars","Avril","Mai","Juin","Juillet","Août","Septembre","Octobre","Novembre","Décembre"]),
        y=alt.Y("Température moyenne (°C):Q"),
        tooltip=["Mois (nom)", "Température moyenne (°C)"]
    )
    st.altair_chart(chart_temp, use_container_width=True)

with col2:
    st.markdown("#### 🌧️ Précipitations totales mensuelles")
    chart_precip = alt.Chart(df_sel).mark_bar().encode(
        x=alt.X("Mois (nom):O", sort=["Janvier","Février","Mars","Avril","Mai","Juin","Juillet","Août","Septembre","Octobre","Novembre","Décembre"]),
        y=alt.Y("Précipitations totales (mm):Q"),
        tooltip=["Mois (nom)", "Précipitations totales (mm)"]
    )
    st.altair_chart(chart_precip, use_container_width=True)

st.markdown("#### 💧 Évapotranspiration cumulée annuelle")
chart_et0 = alt.Chart(df_sel).mark_area(opacity=0.6).encode(
    x=alt.X("Mois (nom):O", sort=["Janvier","Février","Mars","Avril","Mai","Juin","Juillet","Août","Septembre","Octobre","Novembre","Décembre"]),
    y=alt.Y("Evapotranspiration cumulée (mm):Q"),
    tooltip=["Mois (nom)", "Evapotranspiration cumulée (mm)"]
)
st.altair_chart(chart_et0, use_container_width=True)

# ---------------------------------------------------------
# Téléchargement CSV
# ---------------------------------------------------------
csv = df_sel.to_csv(index=False).encode("utf-8")
st.download_button(
    label=f"📥 Télécharger les données {annee}",
    data=csv,
    file_name=f"climat_beauvais_{annee}.csv",
    mime="text/csv"
)

st.success("✅ Données prêtes et graphiques affichés (Janvier → Décembre) !")
