# covid19-nigeria-eda-dashboard
Visual EDA and Dashboard of COVID-19 in Nigeria (2020–2022) using Python, Plotly &amp; Streamlit
https://your-username-covid19-nigeria-eda-dashboard.streamlit.app
# 🇳🇬 COVID-19 in Nigeria — Visual EDA & Dashboard

📊 Exploratory Data Analysis (EDA) and Interactive Dashboard of COVID-19 in **Nigeria (2020–2022)**.  
Built using **Python, Pandas, Plotly, Seaborn, and Streamlit**.
pandas
numpy
matplotlib
seaborn
plotly
streamlit
geopandas
folium
pycountry
scipy
import streamlit as st
import pandas as pd
import plotly.express as px

st.title("COVID-19 — Nigeria Dashboard")

st.write("This is a minimal demo dashboard. Replace with your full app code.")

# Load data
try:
    df = pd.read_csv("nigeria_owid_clean.csv", parse_dates=["date"])
    st.line_chart(df.set_index("date")["new_cases"])
except Exception as e:
    st.error("Dataset not found. Please add nigeria_owid_clean.csv")
date,new_cases,total_cases,new_deaths,total_deaths
2020-02-01,1,1,0,0
2020-02-02,2,3,0,0
2020-02-03,3,6,1,1
2020-02-04,5,11,0,1
2020-02-05,8,19,1,2
2020-02-06,13,32,0,2
2020-02-07,21,53,2,4
2020-02-08,34,87,1,5
2020-02-09,55,142,0,5
2020-02-10,89,231,3,8
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# COVID-19 in Nigeria — EDA Notebook\n",
    "This notebook contains exploratory data analysis for COVID-19 in Nigeria.\n",
    "\n",
    "## Objectives:\n",
    "- Load and explore COVID-19 dataset (Nigeria only)\n",
    "- Perform data cleaning and preprocessing\n",
    "- Generate visualizations of trends\n",
    "- Provide insights from the analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import libraries\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "sns.set(style='darkgrid')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load dataset\n",
    "df = pd.read_csv('nigeria_owid_clean.csv', parse_dates=['date'])\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Basic info\n",
    "df.info()\n",
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Line plot of new cases\n",
    "plt.figure(figsize=(10,5))\n",
    "sns.lineplot(data=df, x='date', y='new_cases')\n",
    "plt.title('Daily New COVID-19 Cases in Nigeria')\n",
    "plt.xlabel('Date')\n",
    "plt.ylabel('New Cases')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Line plot of new deaths\n",
    "plt.figure(figsize=(10,5))\n",
    "sns.lineplot(data=df, x='date', y='new_deaths', color='red')\n",
    "plt.title('Daily New COVID-19 Deaths in Nigeria')\n",
    "plt.xlabel('Date')\n",
    "plt.ylabel('New Deaths')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cumulative cases vs deaths\n",
    "plt.figure(figsize=(10,5))\n",
    "sns.lineplot(data=df, x='date', y='total_cases', label='Total Cases')\n",
    "sns.lineplot(data=df, x='date', y='total_deaths', label='Total Deaths', color='red')\n",
    "plt.title('Cumulative COVID-19 Cases and Deaths in Nigeria')\n",
    "plt.xlabel('Date')\n",
    "plt.ylabel('Count')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
