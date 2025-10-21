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
onglet_comp, onglet_annee = st.tabs(["🆚 Comparaison 2004 vs 2024", "📅 Vue par année"])

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
