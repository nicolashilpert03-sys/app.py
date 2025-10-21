# =========================================================
# 🌤️ Application Streamlit — Climat de Beauvais (2004 / 2024 / 2044)
# =========================================================
# Prérequis :
# pip install streamlit openmeteo-requests requests-cache retry-requests pandas numpy altair scikit-learn cdsapi xarray netCDF4

import streamlit as st
import openmeteo_requests
import pandas as pd
import numpy as np
import requests_cache
from retry_requests import retry
import altair as alt

# ---------------------------------------------------------
# Configuration générale
# ---------------------------------------------------------
st.set_page_config(page_title="Climat Beauvais", layout="wide", page_icon="🌦️")
st.markdown(
    "<h1 style='text-align:center;'>🌤️ Climat de Beauvais — 2004 / 2024 / Projection 2044</h1>",
    unsafe_allow_html=True
)
st.write("Données historiques : **Open-Meteo Archive API**. Projections : **ML** (Ridge + saisonnalité) et **CMIP6 Copernicus**.")

# ---------------------------------------------------------
# Utilitaires : chargement et préparation Open-Meteo
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def charger_donnees(annee: int) -> pd.DataFrame:
    """Charge les données journalières Open-Meteo et agrège au mois."""
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
    if not responses:
        st.error("Aucune réponse de l'API Open-Meteo.")
        st.stop()
    response = responses[0]
    daily = response.Daily()

    # Extraction
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

    df["mois"] = df["date"].dt.month

    # Agrégation mensuelle
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

    mois_noms = {
        1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
        5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
        9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"
    }
    df_mensuel["Mois (nom)"] = df_mensuel["mois"].map(mois_noms)

    # Nettoyage / renommage
    df_mensuel.rename(columns={
        "mois": "Mois (numéro)",
        "temperature_2m_mean": "Température moyenne (°C)",
        "precipitation_sum": "Précipitations totales (mm)",
        "et0_fao_evapotranspiration": "Evapotranspiration (mm)",
        "wind_direction_10m_dominant": "Direction du vent dominante",
    }, inplace=True)

    # Cumulés
    df_mensuel["Précipitations cumulées (mm)"] = df_mensuel["Précipitations totales (mm)"].cumsum()
    df_mensuel["Evapotranspiration cumulée (mm)"] = df_mensuel["Evapotranspiration (mm)"].cumsum()

    # Arrondis
    for c in ["Température moyenne (°C)", "Précipitations totales (mm)", "Evapotranspiration (mm)",
              "Précipitations cumulées (mm)", "Evapotranspiration cumulée (mm)"]:
        df_mensuel[c] = df_mensuel[c].round(1)

    return df_mensuel

# ---------------------------------------------------------
# Chargement des deux années de référence
# ---------------------------------------------------------
with st.spinner("⏳ Chargement des données 2004 et 2024..."):
    df_2004 = charger_donnees(2004)
    df_2024 = charger_donnees(2024)

ordre_mois = ["Janvier","Février","Mars","Avril","Mai","Juin","Juillet","Août","Septembre","Octobre","Novembre","Décembre"]

# ---------------------------------------------------------
# Tabs UI
# ---------------------------------------------------------
onglet_comp, onglet_annee, onglet_proj, onglet_climat = st.tabs([
    "🆚 Comparaison 2004 vs 2024",
    "📅 Vue par année",
    "🔮 Projection 2044 (ML)",
    "🌍 Modèle climatique 2044"
])

# =========================================================
# 🆚 Onglet : Comparaison 2004 vs 2024
# =========================================================
with onglet_comp:
    st.subheader("Comparaison 2004 / 2024 — Indicateurs annuels")
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Température moyenne (°C) — 2024",
        f"{df_2024['Température moyenne (°C)'].mean():.1f}",
        f"{df_2024['Température moyenne (°C)'].mean() - df_2004['Température moyenne (°C)'].mean():+.1f}"
    )
    c2.metric(
        "Précipitations totales (mm) — 2024",
        f"{df_2024['Précipitations totales (mm)'].sum():.1f}",
        f"{df_2024['Précipitations totales (mm)'].sum() - df_2004['Précipitations totales (mm)'].sum():+.1f}"
    )
    c3.metric(
        "ET0 (mm) — 2024",
        f"{df_2024['Evapotranspiration (mm)'].sum():.1f}",
        f"{df_2024['Evapotranspiration (mm)'].sum() - df_2004['Evapotranspiration (mm)'].sum():+.1f}"
    )

    # Graphiques comparés
    left = df_2004[["Mois (nom)", "Température moyenne (°C)"]].assign(Année=2004)
    right = df_2024[["Mois (nom)", "Température moyenne (°C)"]].assign(Année=2024)
    df_temp_all = pd.concat([left, right], ignore_index=True)
    df_temp_all["Mois (nom)"] = pd.Categorical(df_temp_all["Mois (nom)"], categories=ordre_mois, ordered=True)

    st.markdown("#### 🌡️ Température moyenne mensuelle (comparée)")
    chart_t = alt.Chart(df_temp_all).mark_line(point=True).encode(
        x=alt.X("Mois (nom):O", sort=ordre_mois),
        y=alt.Y("Température moyenne (°C):Q"),
        color="Année:N",
        tooltip=["Année", "Mois (nom)", alt.Tooltip("Température moyenne (°C):Q", format=".1f")]
    )
    st.altair_chart(chart_t, use_container_width=True)

    st.markdown("#### 🌧️ Précipitations totales mensuelles (comparées)")
    p_left = df_2004[["Mois (nom)", "Précipitations totales (mm)"]].assign(Année=2004)
    p_right = df_2024[["Mois (nom)", "Précipitations totales (mm)"]].assign(Année=2024)
    df_p_all = pd.concat([p_left, p_right], ignore_index=True)
    df_p_all["Mois (nom)"] = pd.Categorical(df_p_all["Mois (nom)"], categories=ordre_mois, ordered=True)
    chart_p = alt.Chart(df_p_all).mark_bar().encode(
        x=alt.X("Mois (nom):O", sort=ordre_mois),
        y=alt.Y("Précipitations totales (mm):Q"),
        color="Année:N",
        tooltip=["Année", "Mois (nom)", alt.Tooltip("Précipitations totales (mm):Q", format=".1f")]
    )
    st.altair_chart(chart_p, use_container_width=True)

# =========================================================
# 📅 Onglet : Vue par année
# =========================================================
with onglet_annee:
    annee = st.radio("Choisissez une année :", [2004, 2024], horizontal=True)
    df_sel = df_2004 if annee == 2004 else df_2024
    st.markdown(f"### 📅 Données mensuelles — {annee}")
    st.dataframe(df_sel, use_container_width=True)

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

    # Téléchargement
    csv = df_sel.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Télécharger les données {annee}",
        data=csv,
        file_name=f"climat_beauvais_{annee}.csv",
        mime="text/csv"
    )

# =========================================================
# 🔮 Onglet : Projection 2044 (Machine Learning)
# =========================================================
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

@st.cache_data(show_spinner=False)
def charger_toutes_annees(debut=2004, fin=2024) -> pd.DataFrame:
    frames = []
    for an in range(debut, fin + 1):
        df = charger_donnees(an).copy()
        df["Année"] = an
        frames.append(df[[
            "Année", "Mois (numéro)", "Mois (nom)",
            "Température moyenne (°C)", "Précipitations totales (mm)", "Evapotranspiration (mm)"
        ]])
    big = pd.concat(frames, ignore_index=True)
    # Saisonnalité
    angle = 2 * np.pi * (big["Mois (numéro)"] - 1) / 12.0
    big["sin_mois"], big["cos_mois"] = np.sin(angle), np.cos(angle)
    return big

@st.cache_data(show_spinner=False)
def proj_ml_2044(df_all_years: pd.DataFrame, col: str, deg: int = 2, alpha: float = 1.0) -> pd.DataFrame:
    """Projection 2044 : Ridge sur [Année, Année^2..] + sin/cos(mois)."""
    X = df_all_years[["Année", "sin_mois", "cos_mois"]].copy()
    # Termes polynomiaux sur Année
    for d in range(2, deg + 1):
        X[f"Année^{d}"] = X["Année"] ** d
    y = df_all_years[col].values

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha, random_state=42))
    ])
    pipe.fit(X, y)

    # 2044 pour 12 mois
    mois = np.arange(1, 13)
    angle = 2 * np.pi * (mois - 1) / 12.0
    X_new = pd.DataFrame({
        "Année": 2044,
        "sin_mois": np.sin(angle),
        "cos_mois": np.cos(angle),
    })
    for d in range(2, deg + 1):
        X_new[f"Année^{d}"] = 2044 ** d

    y_hat = pipe.predict(X_new)
    out = pd.DataFrame({
        "Mois (numéro)": mois,
        "Mois (nom)": [ordre_mois[m-1] for m in mois],
        col: np.round(y_hat, 1)
    })
    return out

with onglet_proj:
    st.subheader("Projection 2044 (ML : Ridge polynomiale + saisonnalité)")
    # Hyperparamètres optionnels
    colA, colB = st.columns(2)
    with colA:
        deg = st.slider("Degré polynôme (Année)", 1, 3, 2, help="1=linéaire, 2=quadratique, 3=cubique")
    with colB:
        alpha = st.slider("Régularisation Ridge (alpha)", 0.1, 10.0, 1.0)

    with st.spinner("Entraînement sur 2004→2024..."):
        df_all_years = charger_toutes_annees(2004, 2024)
        temp_2044 = proj_ml_2044(df_all_years, "Température moyenne (°C)", deg=deg, alpha=alpha)
        prec_2044 = proj_ml_2044(df_all_years, "Précipitations totales (mm)", deg=deg, alpha=alpha)
        et0_2044  = proj_ml_2044(df_all_years, "Evapotranspiration (mm)", deg=deg, alpha=alpha)

    df_2044_ml = temp_2044.merge(prec_2044, on=["Mois (numéro)", "Mois (nom)"]).merge(et0_2044, on=["Mois (numéro)", "Mois (nom)"])
    df_2044_ml["Précipitations cumulées (mm)"] = df_2044_ml["Précipitations totales (mm)"].cumsum().round(1)
    df_2044_ml["Evapotranspiration cumulée (mm)"] = df_2044_ml["Evapotranspiration (mm)"].cumsum().round(1)

    st.markdown("#### 📅 Données mensuelles prévues — 2044 (ML)")
    st.dataframe(df_2044_ml, use_container_width=True)

    # Graphiques comparatifs
    # Température
    df_plot_t = pd.concat([
        df_2004[["Mois (nom)", "Température moyenne (°C)"]].assign(Année=2004),
        df_2024[["Mois (nom)", "Température moyenne (°C)"]].assign(Année=2024),
        df_2044_ml[["Mois (nom)", "Température moyenne (°C)"]].assign(Année=2044)
    ], ignore_index=True)
    df_plot_t["Mois (nom)"] = pd.Categorical(df_plot_t["Mois (nom)"], categories=ordre_mois, ordered=True)

    st.markdown("#### 🌡️ Température moyenne mensuelle — 2004 / 2024 / 2044 (ML)")
    chart_t_ml = alt.Chart(df_plot_t).mark_line(point=True).encode(
        x=alt.X("Mois (nom):O", sort=ordre_mois),
        y=alt.Y("Température moyenne (°C):Q"),
        color="Année:N",
        tooltip=["Année", "Mois (nom)", alt.Tooltip("Température moyenne (°C):Q", format=".1f")]
    )
    st.altair_chart(chart_t_ml, use_container_width=True)

    # Précipitations
    df_plot_p = pd.concat([
        df_2004[["Mois (nom)", "Précipitations totales (mm)"]].assign(Année=2004),
        df_2024[["Mois (nom)", "Précipitations totales (mm)"]].assign(Année=2024),
        df_2044_ml[["Mois (nom)", "Précipitations totales (mm)"]].assign(Année=2044)
    ], ignore_index=True)
    df_plot_p["Mois (nom)"] = pd.Categorical(df_plot_p["Mois (nom)"], categories=ordre_mois, ordered=True)

    st.markdown("#### 🌧️ Précipitations totales mensuelles — 2004 / 2024 / 2044 (ML)")
    chart_p_ml = alt.Chart(df_plot_p).mark_bar().encode(
        x=alt.X("Mois (nom):O", sort=ordre_mois),
        y=alt.Y("Précipitations totales (mm):Q"),
        color="Année:N",
        tooltip=["Année", "Mois (nom)", alt.Tooltip("Précipitations totales (mm):Q", format=".1f")]
    )
    st.altair_chart(chart_p_ml, use_container_width=True)

    # Export
    csv_ml = df_2044_ml.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Télécharger la projection 2044 (ML)",
        data=csv_ml,
        file_name="projection_2044_ML.csv",
        mime="text/csv"
    )

# =========================================================
# 🌍 Onglet : Modèle climatique 2044 (CMIP6 via Copernicus CDS)
# =========================================================
with onglet_climat:
    st.subheader("Projection 2044 (CMIP6 via Copernicus Climate Data Store)")
    st.caption("Option A : téléchargement via l'API CDS (requiert ~/.cdsapirc). Option B : importer un NetCDF (.nc) que tu as déjà.")

    # Sélection scénario / modèle
    colA, colB = st.columns(2)
    with colA:
        scenario_label = st.selectbox("Scénario d'émissions (SSP)", ["SSP2-4.5", "SSP1-2.6", "SSP5-8.5"], index=0)
    with colB:
        model_label = st.selectbox("Modèle CMIP6", [
            "ACCESS-ESM1-5", "BCC-CSM2-MR", "CNRM-ESM2-1", "MPI-ESM1-2-HR", "MRI-ESM2-0"
        ], index=0)

    ssp_map = {"SSP1-2.6": "ssp126", "SSP2-4.5": "ssp245", "SSP5-8.5": "ssp585"}
    ssp = ssp_map[scenario_label]

    st.markdown("#### ⬇️ Option A — Télécharger depuis Copernicus (CMIP6)")
    do_download = st.button("Télécharger 2044 (mensuel)")

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
            st.success("✅ Fichier téléchargé : projection_2044_beauvais.nc")
        except ModuleNotFoundError:
            st.error("Module 'cdsapi' manquant. Ajoute 'cdsapi' à requirements.txt et configure ~/.cdsapirc.")
        except Exception as e:
            st.warning(f"Téléchargement CDS impossible : {e}")

    st.markdown("#### 📤 Option B — Importer un fichier NetCDF (.nc)")
    uploaded = st.file_uploader("Dépose un fichier NetCDF 2044 (mensuel) avec température/précipitations.", type=["nc"])
    if uploaded is not None and ds is None:
        try:
            import xarray as xr
            ds = xr.open_dataset(uploaded)
            st.success("✅ Fichier NetCDF chargé.")
        except ModuleNotFoundError:
            st.error("Modules manquants : 'xarray' et 'netCDF4'. Ajoute-les au requirements.txt.")
        except Exception as e:
            st.warning(f"Lecture du NetCDF impossible : {e}")

    if ds is not None:
        import xarray as xr  # safe import si Option A
        # Recherche des variables
        var_temp_candidates = ["tas", "tas_monthly", "2m_temperature", "t2m"]
        var_prec_candidates = ["pr", "precipitation", "pr_monthly"]

        def pick_var(ds_obj, candidates):
            for v in candidates:
                if v in ds_obj.variables:
                    return v
            raise KeyError(f"Variables candidates non trouvées : {candidates}")

        try:
            vtemp = pick_var(ds, var_temp_candidates)
            vprec = pick_var(ds, var_prec_candidates)
        except Exception as e:
            st.error(f"Variables climatiques non trouvées dans le fichier : {e}")
            st.stop()

        # Agrégation spatiale si nécessaire
        dims_t = list(ds[vtemp].dims)
        dims_p = list(ds[vprec].dims)
        da_temp = ds[vtemp]
        da_prec = ds[vprec]
        if ("latitude" in dims_t and "longitude" in dims_t) or ("lat" in dims_t and "lon" in dims_t):
            latdims = ["latitude", "lat"]
            londims = ["longitude", "lon"]
            latdim = [d for d in latdims if d in dims_t][0]
            londim = [d for d in londims if d in dims_t][0]
            da_temp = da_temp.mean(dim=[latdim, londim])
        if ("latitude" in dims_p and "longitude" in dims_p) or ("lat" in dims_p and "lon" in dims_p):
            latdims = ["latitude", "lat"]
            londims = ["longitude", "lon"]
            latdim = [d for d in latdims if d in dims_p][0]
            londim = [d for d in londims if d in dims_p][0]
            da_prec = da_prec.mean(dim=[latdim, londim])

        # Temps et unités
        time = pd.to_datetime(ds["time"].values)
        mois_num = time.month

        # Température : Kelvin -> °C si nécessaire
        temp_vals = da_temp.values
        if np.nanmean(temp_vals) > 100:
            temp_vals = temp_vals - 273.15

        # ✅ Précipitations : kg m^-2 s^-1 -> mm/mois
        prec_vals = da_prec.values
        if np.nanmean(prec_vals) < 5:  # valeur très petite => probablement kg m^-2 s^-1
            months = pd.PeriodIndex(time, freq="M")
            days_in_month = months.days_in_month.values
            prec_vals = prec_vals * 86400 * days_in_month  # 1 kg/m^2 = 1 mm

        # DataFrame final
        mois_noms = ordre_mois
        df_2044_clim = pd.DataFrame({
            "Mois (numéro)": mois_num,
            "Mois (nom)": [mois_noms[m-1] for m in mois_num],
            "Température moyenne (°C)": np.round(temp_vals, 1),
            "Précipitations totales (mm)": np.round(prec_vals, 1),
        }).sort_values("Mois (numéro)")
        df_2044_clim["Précipitations cumulées (mm)"] = df_2044_clim["Précipitations totales (mm)"].cumsum().round(1)

        # Affichage
        st.markdown(f"#### 📅 Résultats 2044 — {scenario_label} — {model_label}")
        st.dataframe(df_2044_clim, use_container_width=True)

        df_2044_clim["Mois (nom)"] = pd.Categorical(df_2044_clim["Mois (nom)"], categories=ordre_mois, ordered=True)

        st.markdown("#### 🌡️ Température mensuelle (modèle climatique)")
        chart_temp_clim = alt.Chart(df_2044_clim).mark_line(point=True).encode(
            x=alt.X("Mois (nom):O", sort=ordre_mois),
            y=alt.Y("Température moyenne (°C):Q"),
            tooltip=["Mois (nom)", alt.Tooltip("Température moyenne (°C):Q", format=".1f")]
        )
        st.altair_chart(chart_temp_clim, use_container_width=True)

        st.markdown("#### 🌧️ Précipitations mensuelles (modèle climatique)")
        chart_prec_clim = alt.Chart(df_2044_clim).mark_bar().encode(
            x=alt.X("Mois (nom):O", sort=ordre_mois),
            y=alt.Y("Précipitations totales (mm):Q"),
            tooltip=["Mois (nom)", alt.Tooltip("Précipitations totales (mm):Q", format=".1f")]
        )
        st.altair_chart(chart_prec_clim, use_container_width=True)

        # Export CSV
        csv_clim = df_2044_clim.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Télécharger la projection 2044 (modèle climatique)",
            data=csv_clim,
            file_name=f"projection_2044_CMIP6_{ssp}_{model_label}.csv",
            mime="text/csv"
        )
    else:
        st.info("Choisis l'Option A (téléchargement) ou l'Option B (import d'un NetCDF) pour afficher la projection 2044.")
