
#  Application Météo — Beauvais

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Réglages de base

print(" Initialisation de l'application ")

st.set_page_config(page_title="Climat Beauvais", layout="wide")
st.title("Climat de Beauvais")
st.write("On compare 2004 et 2024, et on fait une projection pour 2044.")

MOIS = ["Janvier","Février","Mars","Avril","Mai","Juin",
        "Juillet","Août","Septembre","Octobre","Novembre","Décembre"]


# 1) Télécharger et préparer une année

def charger_annee(annee):
    print(f"\n Téléchargement des données pour {annee} ")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 49.43,
        "longitude": 2.08,
        "start_date": f"{annee}-01-01",
        "end_date": f"{annee}-12-31",
        "daily": "temperature_2m_mean,precipitation_sum,et0_fao_evapotranspiration",
        "timezone": "Europe/Paris",
    }
    print(f"Appel de l’API Open-Meteo avec paramètres : {params}")
    r = requests.get(url, params=params, timeout=60)
    print(f"Statut de la requête : {r.status_code}")
    r.raise_for_status()
    data = r.json()["daily"]
    print("Données reçues")

    print("Création du DataFrame pandas")
    df = pd.DataFrame({
        "date": pd.to_datetime(data["time"]),
        "Température": data["temperature_2m_mean"],
        "Pluie (mm)": data["precipitation_sum"],
        "ET0 (mm)": data["et0_fao_evapotranspiration"],
    })
    print(f"Taille du tableau initial : {df.shape}")

    print("Ajout du numéro de mois et agrégation mensuelle")
    df["mois_num"] = df["date"].dt.month
    dfm = df.groupby("mois_num", as_index=False).agg({
        "Température": "mean",
        "Pluie (mm)": "sum",
        "ET0 (mm)": "sum"
    })
    dfm["Nom du Mois"] = dfm["mois_num"].apply(lambda m: MOIS[m-1])
    dfm["Pluie cumul (mm)"] = dfm["Pluie (mm)"].cumsum()
    dfm["ET0 cumul (mm)"] = dfm["ET0 (mm)"].cumsum()
    dfm = dfm.round(1)

    print(f"Année {annee} préparée avec succès — {len(dfm)} lignes")
    print(dfm.head())
    return dfm


# 2) Charger 2004 et 2024

print("\n Chargement des données 2004 et 2024 ")
with st.spinner(" Téléchargement des données 2004 et 2024"):
    df_2004 = charger_annee(2004)
    df_2024 = charger_annee(2024)
print("Données chargées avec succès pour 2004 et 2024")


# 3) Onglets

tab_comp, tab_annee, tab_proj = st.tabs([
    "Comparaison 2004 vs 2024",
    "Une seule année",
    "Projection 2044 "
])


# Comparaison 2004 vs 2024

with tab_comp:
    print("\n Onglet Comparaison ")
    c1, c2, c3 = st.columns(3)
    print("Calcul des métriques moyennes et totales")

    temp_diff = df_2024["Température"].mean() - df_2004["Température"].mean()
    pluie_diff = df_2024["Pluie (mm)"].sum() - df_2004["Pluie (mm)"].sum()
    et0_diff = df_2024["ET0 (mm)"].sum() - df_2004["ET0 (mm)"].sum()

    print(f"Diff Temp: {temp_diff:.2f} | Diff Pluie: {pluie_diff:.2f} | Diff ET0: {et0_diff:.2f}")

    c1.metric("Température moyenne 2024", f"{df_2024['Température'].mean():.1f} °C", f"{temp_diff:+.1f} vs 2004")
    c2.metric("Pluie totale 2024", f"{df_2024['Pluie (mm)'].sum():.0f} mm", f"{pluie_diff:+.0f} vs 2004")
    c3.metric("ET0 totale 2024", f"{df_2024['ET0 (mm)'].sum():.0f} mm", f"{et0_diff:+.0f} vs 2004")

    print("Création des graphiques")
    st.subheader("Températures mensuelles")
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.plot(df_2004["Nom du Mois"], df_2004["Température"], marker="o", label="2004")
    ax.plot(df_2024["Nom du Mois"], df_2024["Température"], marker="o", label="2024")
    ax.legend(); ax.set_xlabel("Mois"); ax.set_ylabel("°C"); plt.xticks(rotation=45)
    st.pyplot(fig, use_container_width=False, clear_figure=True)
    print("Graphique température affiché")

    st.subheader("Pluie mensuelle")
    fig, ax = plt.subplots(figsize=(4, 2.5))
    x = np.arange(12); width = 0.4
    ax.bar(x - width/2, df_2004["Pluie (mm)"], width=width, label="2004")
    ax.bar(x + width/2, df_2024["Pluie (mm)"], width=width, label="2024")
    ax.set_xticks(x, df_2024["Nom du Mois"]); plt.xticks(rotation=45)
    ax.legend(); ax.set_ylabel("mm")
    st.pyplot(fig, use_container_width=False, clear_figure=True)
    print("Graphique pluie affiché")

    print("Création du tableau comparatif")
    comp = pd.DataFrame({
        "Mois": df_2004["Nom du Mois"],
        "Temp 2004 (°C)": df_2004["Température"],
        "Temp 2024 (°C)": df_2024["Température"],
        "Δ Temp (°C)": (df_2024["Température"] - df_2004["Température"]).round(1),
        "Pluie 2004 (mm)": df_2004["Pluie (mm)"],
        "Pluie 2024 (mm)": df_2024["Pluie (mm)"],
        "Δ Pluie (mm)": (df_2024["Pluie (mm)"] - df_2004["Pluie (mm)"]).round(1),
        "ET0 2004 (mm)": df_2004["ET0 (mm)"],
        "ET0 2024 (mm)": df_2024["ET0 (mm)"],
        "Δ ET0 (mm)": (df_2024["ET0 (mm)"] - df_2004["ET0 (mm)"]).round(1),
    })
    print(" Tableau créé :")
    print(comp.head())
    st.dataframe(comp, use_container_width=True)


# Une seule année

with tab_annee:
    print("\n Onglet Une seule année ")
    an = st.radio("Choisis l'année :", [2004, 2024], horizontal=True)
    df_sel = df_2004 if an == 2004 else df_2024
    print(f"Année sélectionnée : {an}")
    st.dataframe(df_sel, use_container_width=True)

    st.markdown("Graphique températures")
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.plot(df_sel["Nom du Mois"], df_sel["Température"], marker="o")
    ax.set_xlabel("Mois"); ax.set_ylabel("°C"); plt.xticks(rotation=45)
    st.pyplot(fig, use_container_width=False, clear_figure=True)
    print("Graphique température année unique affiché")


# Projection 2044

with tab_proj:
    print("\n Onglet Projection ")
    st.write("on prolonge la tendance (2004→2024) jusqu’à 2044.")
    temp_2044 = (df_2024["Température"] + (df_2024["Température"] - df_2004["Température"])).round(1)
    pluie_2044 = (df_2024["Pluie (mm)"] + (df_2024["Pluie (mm)"] - df_2004["Pluie (mm)"])).round(1)
    et0_2044   = (df_2024["ET0 (mm)"] + (df_2024["ET0 (mm)"] - df_2004["ET0 (mm)"])).round(1)

    print("Calcul des valeurs projetées pour 2044 terminé")
    df_2044 = pd.DataFrame({
        "Nom du Mois": df_2024["Nom du Mois"],
        "Température": temp_2044,
        "Pluie (mm)": pluie_2044,
        "ET0 (mm)": et0_2044
    })
    df_2044["Pluie cumul (mm)"] = df_2044["Pluie (mm)"].cumsum().round(1)
    df_2044["ET0 cumul (mm)"] = df_2044["ET0 (mm)"].cumsum().round(1)
    print("Tableau 2044 prêt :")
    print(df_2044.head())

    st.dataframe(df_2044, use_container_width=True)

    st.markdown("Températures 2004 / 2024 / 2044")
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.plot(df_2004["Nom du Mois"], df_2004["Température"], label="2004")
    ax.plot(df_2024["Nom du Mois"], df_2024["Température"], label="2024")
    ax.plot(df_2044["Nom du Mois"], df_2044["Température"], label="2044 (proj.)")
    ax.legend(); plt.xticks(rotation=45)
    st.pyplot(fig, use_container_width=False, clear_figure=True)
    print("Graphique température projection affiché")

print("\n Fin du script !!! ")
