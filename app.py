# --- Pré-requis ---
# pip install streamlit openmeteo-requests requests-cache retry-requests pandas numpy altair

import streamlit as st
import openmeteo_requests
import pandas as pd
import numpy as np
import requests_cache
from retry_requests import retry
import altair as alt

# --- Config Streamlit ---
st.set_page_config(page_title="Climat de Beauvais 2004", layout="wide")
st.title("🌦️ Climat mensuel de Beauvais - Année 2004")

# --- Fonction utilitaire ---
def deg_to_cardinal(deg):
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    ix = int((deg + 22.5) // 45) % 8
    return directions[ix]

@st.cache_data
def charger_donnees():
    """Charge et prépare les données météo de Beauvais 2004."""
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 49.4333,
        "longitude": 2.0833,
        "start_date": "2004-01-01",
        "end_date": "2004-12-31",
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

    df = pd.DataFrame({
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True).tz_convert("Europe/Paris"),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True).tz_convert("Europe/Paris") - pd.Timedelta(seconds=daily.Interval()),
            freq=pd.Timedelta(seconds=daily.Interval())
        ),
        "temperature_2m_mean": daily.Variables(0).ValuesAsNumpy(),
        "weather_code": daily.Variables(1).ValuesAsNumpy(),
        "precipitation_sum": daily.Variables(2).ValuesAsNumpy(),
        "wind_direction_10m_dominant": daily.Variables(3).ValuesAsNumpy(),
        "et0_fao_evapotranspiration": daily.Variables(4).ValuesAsNumpy()
    })

    # Agrégation mensuelle
    df["mois"] = df["date"].dt.to_period("M")

    def mode_as_int(series):
        m = series.mode()
        return int(m.iloc[0]) if not m.empty else int(round(series.iloc[0]))

    df_m = df.groupby("mois").agg({
        "temperature_2m_mean": "mean",
        "precipitation_sum": "sum",
        "et0_fao_evapotranspiration": "sum",
        "wind_direction_10m_dominant": mode_as_int,
        "weather_code": mode_as_int
    }).reset_index()

    df_m["wind_direction_10m_dominant"] = df_m["wind_direction_10m_dominant"].apply(deg_to_cardinal)
    df_m["temperature_2m_mean"] = df_m["temperature_2m_mean"].round(1)
    df_m["precipitation_sum"] = df_m["precipitation_sum"].round(1)
    df_m["et0_fao_evapotranspiration"] = df_m["et0_fao_evapotranspiration"].round(1)

    df_m["Précipitations cumulées (mm)"] = df_m["precipitation_sum"].cumsum()
    df_m["Evapotranspiration cumulée (mm)"] = df_m["et0_fao_evapotranspiration"].cumsum()

    weather_code_map = {
        0: "Ciel dégagé", 1: "Principalement dégagé", 2: "Partiellement nuageux", 3: "Couvert",
        45: "Brouillard", 48: "Brouillard givrant", 51: "Bruine légère", 53: "Bruine modérée",
        55: "Forte bruine", 61: "Pluie légère", 63: "Pluie modérée", 65: "Forte pluie",
        71: "Neige légère", 73: "Neige modérée", 75: "Forte neige", 80: "Averses", 81: "Forte averse",
        82: "Averse violente", 95: "Orage", 99: "Orage avec grêle"
    }
    df_m["Temps dominant"] = df_m["weather_code"].map(weather_code_map)

    df_m.columns = [
        "Mois", "Température moyenne (°C)", "Précipitations totales (mm)",
        "Evapotranspiration (mm)", "Direction du vent dominante",
        "Code météo", "Précipitations cumulées (mm)",
        "Evapotranspiration cumulée (mm)", "Temps dominant"
    ]

    return df_m

# --- Corps de page ---
with st.spinner("Chargement des données météo..."):
    df_mensuel = charger_donnees()

st.success("✅ Données chargées avec succès !")

# --- Affichage du tableau ---
st.subheader("Tableau récapitulatif mensuel")
st.dataframe(df_mensuel, use_container_width=True)

# --- Graphiques ---
st.subheader("📈 Températures et précipitations mensuelles")
col1, col2 = st.columns(2)

with col1:
    chart_temp = alt.Chart(df_mensuel).mark_line(point=True).encode(
        x="Mois:T", y="Température moyenne (°C):Q"
    )
    st.altair_chart(chart_temp, use_container_width=True)

with col2:
    chart_precip = alt.Chart(df_mensuel).mark_bar().encode(
        x="Mois:T", y="Précipitations totales (mm):Q"
    )
    st.altair_chart(chart_precip, use_container_width=True)

# --- Téléchargement ---
csv = df_mensuel.to_csv(index=False).encode('utf-8')
st.download_button("📥 Télécharger le CSV", data=csv, file_name="climat_beauvais_2004.csv", mime="text/csv")

st.caption("Source : [Open-Meteo Archive API](https://open-meteo.com/)")

