# --- Pré-requis ---
# pip install streamlit openmeteo-requests requests-cache retry-requests pandas numpy altair


import streamlit as st
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import altair as alt

# --- Configuration de la page ---
st.set_page_config(page_title="Climat Beauvais 2004 & 2024", layout="wide")
st.title("🌦️ Climat mensuel de Beauvais — Comparaison 2004 vs 2024")

# --- Fonction utilitaire ---
def deg_to_cardinal(deg):
    """Convertit un azimut (0-360°) en point cardinal simple."""
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    ix = int((deg + 22.5) // 45) % 8
    return directions[ix]

@st.cache_data
def charger_donnees(annee):
    """Charge et prépare les données météo de Beauvais pour une année donnée."""
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

    # Appel API
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    daily = response.Daily()

    # Extraction
    t0 = pd.to_datetime(daily.Time(), unit="s", utc=True).tz_convert("Europe/Paris")
    t1 = pd.to_datetime(daily.TimeEnd(), unit="s", utc=True).tz_convert("Europe/Paris")
    dt = pd.Timedelta(seconds=daily.Interval())
    dates = pd.date_range(start=t0, end=t1 - dt, freq=dt)

    df = pd.DataFrame({
        "date": dates,
        "temperature_2m_mean": daily.Variables(0).ValuesAsNumpy(),
        "weather_code": daily.Variables(1).ValuesAsNumpy(),
        "precipitation_sum": daily.Variables(2).ValuesAsNumpy(),
        "wind_direction_10m_dominant": daily.Variables(3).ValuesAsNumpy(),
        "et0_fao_evapotranspiration": daily.Variables(4).ValuesAsNumpy()
    })

    df["mois"] = df["date"].dt.to_period("M")

    def mode_as_int(series):
        m = series.mode()
        return int(m.iloc[0]) if not m.empty else int(round(series.iloc[0]))

    # Agrégation mensuelle
    df_m = df.groupby("mois").agg({
        "temperature_2m_mean": "mean",
        "precipitation_sum": "sum",
        "et0_fao_evapotranspiration": "sum",
        "wind_direction_10m_dominant": mode_as_int,
        "weather_code": mode_as_int
    }).reset_index()

    # Nettoyage
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
        71: "Neige légère", 73: "Neige modérée", 75: "Forte neige",
        80: "Averses", 81: "Forte averse", 82: "Averse violente",
        95: "Orage", 99: "Orage avec grêle"
    }
    df_m["Temps dominant"] = df_m["weather_code"].map(weather_code_map)

    df_m = df_m.rename(columns={
        "mois": "Mois",
        "temperature_2m_mean": "Température moyenne (°C)",
        "precipitation_sum": "Précipitations totales (mm)",
        "et0_fao_evapotranspiration": "Evapotranspiration (mm)",
        "wind_direction_10m_dominant": "Direction du vent dominante"
    })

    df_m["Année"] = annee
    return df_m

# --- Chargement des données ---
with st.spinner("Chargement des données météo..."):
    df_2004 = charger_donnees(2004)
    df_2024 = charger_donnees(2024)

st.success("✅ Données chargées avec succès !")

# --- Sélecteur d’année ---
choix = st.radio("Choisissez l’année à afficher :", [2004, 2024], horizontal=True)
df = df_2004 if choix == 2004 else df_2024

# --- Tableau ---
st.subheader(f"📅 Climat mensuel à Beauvais ({choix})")
st.dataframe(df, use_container_width=True)

# --- Graphiques comparatifs ---
st.markdown("### 📈 Évolution mensuelle")

col1, col2 = st.columns(2)
with col1:
    chart_temp = alt.Chart(df).mark_line(point=True).encode(
        x="Mois:T", y="Température moyenne (°C):Q", color=alt.value("#E4572E")
    )
    st.altair_chart(chart_temp, use_container_width=True)

with col2:
    chart_precip = alt.Chart(df).mark_bar().encode(
        x="Mois:T", y="Précipitations totales (mm):Q", color=alt.value("#3B82F6")
    )
    st.altair_chart(chart_precip, use_container_width=True)

# --- Comparaison globale ---
st.markdown("### 🔍 Comparaison annuelle 2004 vs 2024")

df_comp = pd.concat([df_2004, df_2024])
moyennes = df_comp.groupby("Année")[["Température moyenne (°C)", "Précipitations totales (mm)"]].mean().round(1)

st.table(moyennes)

# --- Téléchargement ---
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Télécharger le CSV de l'année sélectionnée", csv, f"climat_beauvais_{choix}.csv", mime="text/csv")

st.caption("Source des données : [Open-Meteo Archive API](https://open-meteo.com/)")


