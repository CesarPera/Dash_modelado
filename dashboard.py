















































































































































from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

# ===============================
# APP
# ===============================
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Análisis Predictivo de Flujo Vehicular"

# ===============================
# CARGAR DATOS
# ===============================
df = pd.read_excel(
    r"C:\Users\camila\Desktop\presentacion\Flujo_vehicular_2014_2024 (1).xlsx"
)

df = df.rename(columns={"A�O": "AÑO"})

# ===============================
# GRÁFICOS (SIN TÍTULO)
# ===============================

fig_anio = px.line(
    df.groupby("AÑO", as_index=False)["VEH_TOTAL"].sum(),
    x="AÑO",
    y="VEH_TOTAL",
    markers=True
)

fig_depa = px.bar(
    df.groupby("DEPARTAMENTO", as_index=False)["VEH_TOTAL"].sum(),
    x="DEPARTAMENTO",
    y="VEH_TOTAL"
)

# ===============================
# TORNEO DE MODELOS (CLASE)
# ===============================

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import pandas as pd
import plotly.express as px

variables = [
    'VEH_LIGEROS_TAR_DIF',
    'VEH_LIGEROS_AUTOMOVILES',
    'VEH_PESADOS_TAR_DIF',
    'VEH_PESADOS__2E',
    'VEH_PESADOS_3E',
    'VEH_PESADOS_4E',
    'VEH_PESADOS_5E',
    'VEH_PESADOS_6E',
    'VEH_PESADOS_7E'
]

X = df[variables]

df['flujo_clase'] = pd.qcut(
    df['VEH_TOTAL'],
    q=3,
    labels=['Bajo', 'Medio', 'Alto']
)

y = df['flujo_clase']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 🔹 Escalado (IMPORTANTE para SVM y LDA)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

modelos = {
    "Árbol de Decisión": DecisionTreeClassifier(max_depth=5),
    "SVM": SVC(kernel='rbf'),
    "Discriminante (LDA)": LinearDiscriminantAnalysis()
}

resultados = []

for nombre, modelo in modelos.items():
    
    # Árbol no necesita escalado
    if nombre == "Árbol de Decisión":
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
    else:
        modelo.fit(X_train_scaled, y_train)
        y_pred = modelo.predict(X_test_scaled)

    resultados.append({
        "Modelo": nombre,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "F1-Score": round(f1_score(y_test, y_pred, average='weighted'), 4)
    })

df_torneo = pd.DataFrame(resultados).sort_values(by="Accuracy", ascending=False)

fig_torneo = px.bar(
    df_torneo,
    x="Modelo",
    y="Accuracy",
    text="Accuracy",
    color="Modelo",
    title="🏆 Torneo de Modelos - Clasificación"
)

fig_torneo.update_traces(textposition="outside")
fig_torneo.update_layout(yaxis_range=[0,1])

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# VARIABLES DE REGRESIÓN
# -----------------------------
df_reg = df.copy()

# Convertir CODIGO_PEAJE a numérico
le = LabelEncoder()
df_reg["CODIGO_PEAJE"] = le.fit_transform(df_reg["CODIGO_PEAJE"])

x_reg = df_reg[["AÑO", "MES", "CODIGO_PEAJE"]]
y_reg = df_reg['VEH_LIGEROS_TOTAL']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    x_reg, y_reg, test_size=0.2, random_state=42
)

# -----------------------------
# REGRESIÓN LINEAL
# -----------------------------
lr = LinearRegression()
lr.fit(X_train_reg, y_train_reg)
y_pred_lr = lr.predict(X_test_reg)

mae_lr = mean_absolute_error(y_test_reg, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test_reg, y_pred_lr))
r2_lr = r2_score(y_test_reg, y_pred_lr)

# -----------------------------
# ÁRBOL DE REGRESIÓN
# -----------------------------
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train_reg, y_train_reg)
y_pred_tree = tree_reg.predict(X_test_reg)

mae_tree = mean_absolute_error(y_test_reg, y_pred_tree)
rmse_tree = np.sqrt(mean_squared_error(y_test_reg, y_pred_tree))
r2_tree = r2_score(y_test_reg, y_pred_tree)

import plotly.express as px
import pandas as pd

# =========================
# TORNEO REGRESIÓN DATA
# =========================

df_torneo_reg = pd.DataFrame({
    "Modelo": ["Regresión Lineal", "Árbol de Regresión"],
    "RMSE": [65143.67, 19322.51],
    "R2": [0.0161, 0.9134]
})

# Gráfico RMSE
fig_rmse = px.bar(
    df_torneo_reg,
    x="Modelo",
    y="RMSE",
    title="Comparación RMSE (Menor es Mejor)",
    text_auto=True
)

fig_rmse.update_layout(template="plotly_white")

# Gráfico R2
fig_r2 = px.bar(
    df_torneo_reg,
    x="Modelo",
    y="R2",
    title="Comparación R² (Mayor es Mejor)",
    text_auto=True
)

fig_r2.update_layout(template="plotly_white")

fig_anio.update_layout(title=None)
fig_depa.update_layout(title=None)

# ===============================
# LAYOUT
# ===============================
app.layout = dbc.Container([

    # --------- TÍTULO GENERAL ---------
    dbc.Row([
        dbc.Col([
            html.H1("Análisis Predictivo del Flujo Vehicular",
                    className="text-center text-primary mt-4"),
            html.P("Clasificación y regresión aplicadas a datos de peajes",
                   className="text-center text-muted")
        ])
    ]),

    html.Hr(),

    dbc.Tabs([

        # ===============================
        # TAB 1
        # ===============================
        dbc.Tab(label="Problema y Objetivo", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Problema"),
                    html.P(
                        "El aumento del flujo vehicular genera congestión en ciertos peajes, "
                        "afectando la movilidad y la gestión del transporte."
                    ),
                    html.H4("Objetivo"),
                    html.P(
                        "Analizar el flujo vehicular para identificar patrones de congestión "
                        "y predecir los peajes más saturados usando modelos de clasificación y regresión."
                    )
                ], md=10)
            ], className="mt-4")
        ]),

        # ===============================
        # TAB 2
        # ===============================
        dbc.Tab(label="Conociendo el Negocio", children=[

            dbc.Row([

                # ----- IZQUIERDA -----
                dbc.Col([

                    html.H4("Evolución del Flujo Vehicular por Año",
                            className="text-center"),

                    html.P(
                        "Se observa la tendencia anual del flujo vehicular total, "
                        "permitiendo identificar periodos de crecimiento y posibles "
                        "escenarios futuros de congestión."
                    ),

                    dcc.Graph(figure=fig_anio)

                ], md=6),

                # ----- DERECHA -----
                dbc.Col([

                    html.H4("Flujo Vehicular Total por Departamento",
                            className="text-center"),

                    html.P(
                        "El gráfico permite identificar los departamentos con mayor "
                        "concentración vehicular, facilitando la priorización de recursos "
                        "e inversiones estratégicas."
                    ),

                    dcc.Graph(figure=fig_depa)

                ], md=6)

            ], className="mt-4")

        ]),

        # ===============================
        # TAB 3
        # ===============================
        dbc.Tab(label="EDA", children=[
            html.Div("Aquí irá el análisis exploratorio detallado.",
                     className="m-4")
        ]),

        # ===============================
# TAB: LIMPIEZA Y TRANSFORMACIÓN
# ===============================
dbc.Tab(label="Limpieza y Transformación de Datos", children=[

    dbc.Row([
        dbc.Col([

            html.H4("Proceso de Limpieza y Transformación de Datos",
                    className="text-center"),

            html.Br(),

            html.P(
                "En esta etapa se realizó la depuración de datos para garantizar "
                "la calidad de la información utilizada en los modelos predictivos."
            ),

            html.Ul([
                html.Li("Corrección de nombres de columnas mal codificados (ej. AÑO)."),
                html.Li("Verificación y eliminación de valores nulos."),
                html.Li("Agrupación de datos por año y departamento."),
                html.Li("Transformación de variables para su uso en modelos de clasificación y regresión.")
            ]),

            html.P(
                "Estas transformaciones permiten mejorar la precisión de los modelos "
                "y asegurar resultados más confiables."
            )

        ], md=10)

    ], className="mt-4")

]),


       dbc.Tab(label="Torneo de Modelo", children=[

    dbc.Row([
        dbc.Col([
            html.H3("🏆 Torneo de Modelos de Clasificación"),
            html.P(
                "Se compararon los modelos enseñados en clase: "
                "Árbol de Decisión, SVM y Análisis Discriminante."
            )
        ])
    ], className="mt-4"),

    dbc.Row([
        dbc.Col([
            dbc.Table.from_dataframe(
                df_torneo,
                striped=True,
                bordered=True,
                hover=True,
                responsive=True
            )
        ])
    ], className="mt-3"),

    dbc.Row([
        dbc.Col([
            dcc.Graph(figure=fig_torneo)
        ])
    ], className="mt-4"),

    dbc.Row([
        dbc.Col([
            dbc.Alert(
                "El SVM obtuvo mayor exactitud (97%), sin embargo, se seleccionó el Árbol de Decisión debido a su interpretabilidad. Dado que el objetivo del negocio es identificar qué variables generan congestión, el árbol permite visualizar reglas claras de decisión que apoyan la planificación operativa. La diferencia de rendimiento fue mínima (2.8%), por lo que se priorizó la explicabilidad sobre una ligera mejora predictiva.",
                color="success"
            )
        ])
    ], className="mt-3"),

    html.Hr(),

html.H4("🏆 Torneo de Modelos de Regresión", className="mt-4"),

html.P("Comparación entre Regresión Lineal y Árbol de Regresión usando AÑO, MES y CODIGO_PEAJE."),

dbc.Table([
    html.Thead(html.Tr([
        html.Th("Modelo"),
        html.Th("MAE"),
        html.Th("RMSE"),
        html.Th("R²")
    ])),
    html.Tbody([
        html.Tr([
            html.Td("Regresión Lineal"),
            html.Td(f"{mae_lr:,.2f}"),
            html.Td(f"{rmse_lr:,.2f}"),
            html.Td(f"{r2_lr:.4f}")
        ]),
        html.Tr([
            html.Td("Árbol de Regresión"),
            html.Td(f"{mae_tree:,.2f}"),
            html.Td(f"{rmse_tree:,.2f}"),
            html.Td(f"{r2_tree:.4f}")
        ]),
    ])
], bordered=True, striped=True, hover=True),

dbc.Row([
    dbc.Col(dcc.Graph(figure=fig_rmse), md=6),
    dbc.Col(dcc.Graph(figure=fig_r2), md=6)
], className="mt-4"),

dbc.Alert(
    "Aunque inicialmente se consideró la Regresión Lineal por su simplicidad e interpretabilidad, el torneo de modelos demostró que el comportamiento del flujo vehicular no es lineal. El Árbol de Regresión obtuvo un R² de 0.91 frente a 0.01 de la regresión lineal, por lo que se seleccionó como modelo final al capturar mejor las relaciones no lineales entre año, mes y código de peaje.",
    color="info",
    className="mt-3"
),

]),

        # ===============================
        # TAB 4
        # ===============================
        dbc.Tab(label="Clasificación", children=[
            html.Div("Modelo Árbol de Decisión.",
                     className="m-4")
        ]),

        # ===============================
        # TAB 5
        # ===============================
        dbc.Tab(label="Regresión", children=[
            html.Div("Modelo de Regresión Lineal.",
                     className="m-4")
        ]),

        # ===============================
        # TAB 6
        # ===============================
        dbc.Tab(label="Conclusión", children=[
            html.Div(
                "El análisis permitió identificar tendencias de crecimiento "
                "y departamentos críticos, aportando información estratégica "
                "para mejorar la gestión del tráfico.",
                className="m-4"
            )
        ])

    ])

], fluid=True)

# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    app.run(debug=True, port=8060)
