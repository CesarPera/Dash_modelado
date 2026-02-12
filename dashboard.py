import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from dash import Dash, html, dcc
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, mean_absolute_error, mean_squared_error, r2_score, f1_score


app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Análisis Predictivo de Flujo Vehicular"


df = pd.read_excel("Flujo_vehicular_2014_2024 (1).xlsx")

df = df.rename(columns={"A�O": "AÑO"})

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

df_torneo = pd.DataFrame(resultados).sort_values(
    by="Accuracy", ascending=False)

fig_torneo = px.bar(
    df_torneo,
    x="Modelo",
    y="Accuracy",
    text="Accuracy",
    color="Modelo",
    title="🏆 Torneo de Modelos - Clasificación"
)

fig_torneo.update_traces(textposition="outside")
fig_torneo.update_layout(yaxis_range=[0, 1])


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
# CARGA Y PREPARACIÓN DE DATOS
# ===============================
df["AÑO"] = df["AÑO"].astype(int)

variables = [
    "VEH_LIGEROS_TAR_DIF",
    "VEH_LIGEROS_AUTOMOVILES",
    "VEH_PESADOS_TAR_DIF",
    "VEH_PESADOS__2E",
    "VEH_PESADOS_3E",
    "VEH_PESADOS_4E",
    "VEH_PESADOS_5E",
    "VEH_PESADOS_6E",
    "VEH_PESADOS_7E",
]

X = df[variables]

# Clases de flujo (Bajo, Medio, Alto) según terciles de VEH_TOTAL
df["flujo_clase"] = pd.qcut(
    df["VEH_TOTAL"], q=3, labels=["Bajo", "Medio", "Alto"]
)
y_clf = df["flujo_clase"]

# ===============================
# MODELO DE CLASIFICACIÓN
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_clf, test_size=0.3, random_state=42, stratify=y_clf
)

tree_model = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_model.fit(X_train, y_train)
y_pred = tree_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

importancias = (
    pd.Series(tree_model.feature_importances_, index=variables)
    .sort_values(ascending=False)
)

# ===============================
# GENERAR ÁRBOL DE DECISIÓN
# ===============================
try:
    from sklearn.tree import plot_tree
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    import io
    import base64

    # Crear figura de matplotlib
    fig_tree_mpl, ax = plt.subplots(figsize=(25, 15))
    plot_tree(
        tree_model,
        feature_names=variables,
        class_names=['Bajo', 'Medio', 'Alto'],
        filled=True,
        rounded=True,
        fontsize=10,
        ax=ax
    )
    plt.title("Árbol de Decisión – Clasificación del Flujo Vehicular",
              fontsize=20, pad=20)

    # Guardar en buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150,
                bbox_inches='tight', facecolor='white')
    buf.seek(0)
    tree_img_base64 = base64.b64encode(buf.read()).decode()
    plt.close()

    has_tree_image = True
except:
    has_tree_image = False
    tree_img_base64 = None

# --- COLORES PROFESIONALES ---
COLOR_PRIMARY = "#2E86DE"
COLOR_SUCCESS = "#10AC84"
COLOR_WARNING = "#F79F1F"
COLOR_DANGER = "#EE5A6F"
COLOR_INFO = "#00D2D3"
COLOR_PURPLE = "#A55EEA"

# --- Gráfico: Matriz de confusión ---
fig_cm = px.imshow(
    cm,
    labels=dict(x="Predicción", y="Real", color="Cantidad"),
    x=tree_model.classes_,
    y=tree_model.classes_,
    text_auto=True,
    color_continuous_scale=[[0, "#E3F2FD"], [0.5, "#42A5F5"], [1, "#0D47A1"]],
    title="<b>Matriz de Confusión</b>",
)
fig_cm.update_layout(
    height=450,
    font=dict(size=13),
    title_font_size=18,
    title_x=0.5,
    paper_bgcolor="white",
    plot_bgcolor="white"
)

# --- Gráfico: Importancia de variables ---
fig_importancia = go.Figure(go.Bar(
    x=importancias.values,
    y=importancias.index,
    orientation='h',
    marker=dict(
        color=importancias.values,
        colorscale=[[0, "#FFF3E0"], [0.5, "#FF9800"], [1, "#E65100"]],
        showscale=False
    ),
    text=[f'{val:.2%}' for val in importancias.values],
    textposition='outside'
))
fig_importancia.update_layout(
    title="<b>Importancia de Variables</b>",
    xaxis_title="Importancia",
    yaxis_title="",
    height=450,
    font=dict(size=13),
    title_font_size=18,
    title_x=0.5,
    paper_bgcolor="white",
    plot_bgcolor="white",
    yaxis={'categoryorder': 'total ascending'}
)

# --- Gráfico: métricas por clase ---
clases = ["Alto", "Bajo", "Medio"]
metricas_data = {
    "Clase": clases * 3,
    "Métrica": ["Precision"] * 3 + ["Recall"] * 3 + ["F1-Score"] * 3,
    "Valor": [
        report["Alto"]["precision"],
        report["Bajo"]["precision"],
        report["Medio"]["precision"],
        report["Alto"]["recall"],
        report["Bajo"]["recall"],
        report["Medio"]["recall"],
        report["Alto"]["f1-score"],
        report["Bajo"]["f1-score"],
        report["Medio"]["f1-score"],
    ],
}
df_metricas = pd.DataFrame(metricas_data)

fig_metricas_clf = go.Figure()
colors_metricas = [COLOR_PRIMARY, COLOR_SUCCESS, COLOR_WARNING]
for i, metrica in enumerate(["Precision", "Recall", "F1-Score"]):
    df_temp = df_metricas[df_metricas["Métrica"] == metrica]
    fig_metricas_clf.add_trace(go.Bar(
        name=metrica,
        x=df_temp["Clase"],
        y=df_temp["Valor"],
        marker_color=colors_metricas[i],
        text=[f'{val:.2%}' for val in df_temp["Valor"]],
        textposition='outside'
    ))

fig_metricas_clf.update_layout(
    title="<b>Métricas de Clasificación por Clase</b>",
    xaxis_title="Clase de Flujo",
    yaxis_title="Valor",
    barmode='group',
    height=450,
    font=dict(size=13),
    title_font_size=18,
    title_x=0.5,
    paper_bgcolor="white",
    plot_bgcolor="white",
    legend=dict(orientation="h", yanchor="bottom",
                y=1.02, xanchor="right", x=1)
)

# ===============================
# MODELO DE REGRESIÓN
# ===============================
x_reg = df[["AÑO", "MES", "CODIGO_PEAJE"]]
y_reg = df["VEH_LIGEROS_TOTAL"]
x_reg = pd.get_dummies(x_reg, columns=["CODIGO_PEAJE"], drop_first=True)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    x_reg, y_reg, test_size=0.3, random_state=42
)

reg_model = LinearRegression()
reg_model.fit(X_train_r, y_train_r)
y_pred_r = reg_model.predict(X_test_r)

mae = mean_absolute_error(y_test_r, y_pred_r)
mse = mean_squared_error(y_test_r, y_pred_r)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_r, y_pred_r)

# --- Distribución de residuales ---
residuales = y_test_r - y_pred_r
fig_residuales = go.Figure()
fig_residuales.add_trace(go.Histogram(
    x=residuales,
    nbinsx=50,
    marker_color=COLOR_INFO,
    opacity=0.8,
    name='Residuales'
))
fig_residuales.update_layout(
    title="<b>Distribución de Residuales</b>",
    xaxis_title="Residual (Real - Predicho)",
    yaxis_title="Frecuencia",
    height=450,
    font=dict(size=13),
    title_font_size=18,
    title_x=0.5,
    paper_bgcolor="white",
    plot_bgcolor="white",
    showlegend=False
)

# --- Valores reales vs predichos ---
fig_scatter = go.Figure()
fig_scatter.add_trace(go.Scatter(
    x=y_test_r,
    y=y_pred_r,
    mode='markers',
    marker=dict(
        color=COLOR_PRIMARY,
        size=6,
        opacity=0.6,
        line=dict(width=0.5, color='white')
    ),
    name='Predicciones'
))

min_val = min(y_test_r.min(), y_pred_r.min())
max_val = max(y_test_r.max(), y_pred_r.max())
fig_scatter.add_trace(go.Scatter(
    x=[min_val, max_val],
    y=[min_val, max_val],
    mode='lines',
    line=dict(color=COLOR_DANGER, dash='dash', width=3),
    name='Línea ideal'
))

fig_scatter.update_layout(
    title="<b>Valores Reales vs Predichos</b>",
    xaxis_title="Valor Real",
    yaxis_title="Valor Predicho",
    height=450,
    font=dict(size=13),
    title_font_size=18,
    title_x=0.5,
    paper_bgcolor="white",
    plot_bgcolor="white",
    legend=dict(orientation="h", yanchor="bottom",
                y=1.02, xanchor="right", x=1)
)

# --- Predicción 2025 y top peajes ---
df_2025 = df[df["AÑO"] == 2024].copy()
df_2025["AÑO"] = 2025
X_2025 = df_2025[["AÑO", "MES", "CODIGO_PEAJE"]]
X_2025 = pd.get_dummies(X_2025, columns=["CODIGO_PEAJE"], drop_first=True)
X_2025 = X_2025.reindex(columns=x_reg.columns, fill_value=0)
df_2025["VEH_LIGEROS_PRED_2025"] = reg_model.predict(X_2025)

top_peajes = (
    df_2025.groupby("CODIGO_PEAJE")["VEH_LIGEROS_PRED_2025"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

fig_top_peajes = go.Figure(go.Bar(
    x=top_peajes.index,
    y=top_peajes.values,
    marker=dict(
        color=top_peajes.values,
        colorscale=[[0, "#FFEBEE"], [0.5, "#EF5350"], [1, "#B71C1C"]],
        showscale=False
    ),
    text=[f'{val:,.0f}' for val in top_peajes.values],
    textposition='outside'
))

fig_top_peajes.update_layout(
    title="<b>Top 10 Peajes con Mayor Saturación Esperada (2025)</b>",
    xaxis_title="Código de Peaje",
    yaxis_title="Vehículos Ligeros Estimados",
    height=450,
    font=dict(size=13),
    title_font_size=18,
    title_x=0.5,
    paper_bgcolor="white",
    plot_bgcolor="white",
    xaxis_tickangle=-45
)


veh_cols = [
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

# ===============================
# HISTOGRAMAS (EDA)
# ===============================
hist_components = []

for col in veh_cols:
    fig = px.histogram(
        df,
        x=col,
        nbins=20,
        title=f"Distribución de {col}"
    )
    fig.update_layout(title_x=0.5)

    hist_components.append(
        dbc.Col(dcc.Graph(figure=fig), md=4)
    )

hist_rows = []
for i in range(0, len(hist_components), 3):
    hist_rows.append(
        dbc.Row(hist_components[i:i+3], className="mb-4")
    )

# ===============================
# MATRIZ DE CORRELACIÓN
# ===============================
corr = df[veh_cols].apply(pd.to_numeric, errors="coerce").corr()

fig_corr = px.imshow(
    corr,
    text_auto=".2f",
    aspect="auto",
    title="Matriz de Correlación – Variables Vehiculares"
)
fig_corr.update_layout(title_x=0.5)

# ===============================
# SCATTER MÁS CORRELACIONADO
# ===============================
corr_unstack = corr.unstack()
corr_unstack = corr_unstack[
    corr_unstack.index.get_level_values(
        0) != corr_unstack.index.get_level_values(1)
]
corr_unstack = corr_unstack.sort_values(ascending=False)

corr_unstack = corr_unstack[
    ~corr_unstack.index.map(lambda x: tuple(sorted(x))).duplicated()
]

top_pairs = corr_unstack.head(4).index.tolist()
scatter_figs = []

for var_x, var_y in top_pairs:
    fig = px.scatter(
        df,
        x=var_x,
        y=var_y,
        opacity=0.6,
        title=f"Relación entre {var_x} y {var_y}"
    )
    fig.update_layout(
        title_x=0.5,
        height=400
    )
    scatter_figs.append(fig)


def render_scatter_rows(figs):
    rows = []
    for i in range(0, len(figs), 2):
        rows.append(
            dbc.Row([
                dbc.Col(dcc.Graph(figure=figs[i]), md=6),
                dbc.Col(dcc.Graph(figure=figs[i+1]), md=6)
            ], className="mb-4")
        )
    return rows


scatter_rows = render_scatter_rows(scatter_figs)


# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


# ===============================
# LAYOUT
# ===============================
app.layout = dbc.Container([

    # --------- TÍTULO ---------
    dbc.Row([
        dbc.Col([
            html.H1("🚗 Análisis Predictivo del Flujo Vehicular",
                    className="text-center text-primary mt-4 mb-2",
                    style={'fontWeight': 'bold'}),
            html.P("Clasificación y regresión aplicadas a datos de peajes",
                   className="text-center text-muted",
                   style={'fontSize': '1.2rem'})
        ])
    ]),

    html.Hr(style={'borderWidth': '3px', 'borderColor': COLOR_PRIMARY}),

    # --------- TABS ---------
    dbc.Tabs([

        # ===============================
        # TAB 1: PROBLEMA Y OBJETIVO
        # ===============================
        dbc.Tab(label="🎯 Problema y Objetivo", tab_style={'fontWeight': 'bold'}, children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("🚨 Problema", className="text-danger mb-3"),
                            html.P(
                                "El aumento del flujo vehicular genera congestión en ciertos peajes, "
                                "afectando la movilidad y la gestión del transporte. La falta de predicción "
                                "adecuada dificulta la planificación de recursos.",
                                style={'fontSize': '1.1rem',
                                       'lineHeight': '1.8'}
                            ),
                        ])
                    ], className="shadow-sm mb-4", style={'borderLeft': f'5px solid {COLOR_DANGER}'}),

                    dbc.Card([
                        dbc.CardBody([
                            html.H4("🎯 Objetivo",
                                    className="text-success mb-3"),
                            html.P(
                                "Desarrollar modelos predictivos para:",
                                style={'fontSize': '1.1rem',
                                       'fontWeight': 'bold'}
                            ),
                            html.Ul([
                                html.Li("Clasificar el flujo vehicular en niveles: Bajo, Medio, Alto",
                                        style={'fontSize': '1.05rem', 'marginBottom': '10px'}),
                                html.Li("Predecir el volumen de vehículos ligeros en diferentes peajes",
                                        style={'fontSize': '1.05rem', 'marginBottom': '10px'}),
                                html.Li("Identificar los peajes con mayor saturación esperada para 2025",
                                        style={'fontSize': '1.05rem', 'marginBottom': '10px'}),
                                html.Li("Determinar las variables más influyentes en la congestión",
                                        style={'fontSize': '1.05rem', 'marginBottom': '10px'})
                            ])
                        ])
                    ], className="shadow-sm", style={'borderLeft': f'5px solid {COLOR_SUCCESS}'})
                ], md=10)
            ], className="mt-4", justify="center")
        ]),

        # ===============================
        # TAB 2: CONOCIENDO EL NEGOCIO
        # ===============================
        dbc.Tab(label="💼 Conociendo el Negocio", tab_style={'fontWeight': 'bold'}, children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("📊 Contexto del Negocio", className="mb-3",
                                    style={'color': COLOR_PRIMARY}),
                            html.P(
                                "Los peajes son puntos críticos de control vehicular en las carreteras. "
                                "Una mala gestión puede generar retrasos, pérdidas económicas "
                                "y malestar en los usuarios.",
                                style={'fontSize': '1.1rem',
                                       'lineHeight': '1.8'}
                            ),
                        ])
                    ], className="shadow-sm mb-4"),

                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("🚦 Beneficios Operacionales",
                                            className="text-center mb-3",
                                            style={'color': COLOR_INFO}),
                                    html.Ul([
                                        html.Li("Identificar peajes críticos", style={
                                                'fontSize': '1.05rem'}),
                                        html.Li("Optimizar recursos humanos", style={
                                                'fontSize': '1.05rem'}),
                                        html.Li("Reducir tiempos de espera", style={
                                                'fontSize': '1.05rem'})
                                    ])
                                ])
                            ], className="shadow-sm h-100", style={'borderTop': f'4px solid {COLOR_INFO}'})
                        ], md=6, className="mb-3"),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H5("📈 Beneficios Estratégicos",
                                            className="text-center mb-3",
                                            style={'color': COLOR_SUCCESS}),
                                    html.Ul([
                                        html.Li("Planificar infraestructura", style={
                                                'fontSize': '1.05rem'}),
                                        html.Li("Decisiones basadas en datos", style={
                                                'fontSize': '1.05rem'}),
                                        html.Li("Mejorar experiencia del usuario", style={
                                                'fontSize': '1.05rem'})
                                    ])
                                ])
                            ], className="shadow-sm h-100", style={'borderTop': f'4px solid {COLOR_SUCCESS}'})
                        ], md=6, className="mb-3")
                    ])
                ], md=10)
            ], className="mt-4", justify="center")
        ]),

        # ===============================
        # TAB 3: EDA
        # ===============================
        dbc.Tab(label="📊 EDA", tab_style={'fontWeight': 'bold'}, children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Análisis Exploratorio de Datos (EDA)"),
                    html.P(
                        "En esta etapa se analizan tendencias, distribuciones "
                        "y patrones del flujo vehicular por año, mes y tipo de vehículo."
                    ),
                    html.P("Aquí se incluirán gráficos descriptivos.")
                ])
            ], className="mt-4")
        ]),

        # ===============================
        # TAB 4: CLASIFICACIÓN
        # ===============================
        dbc.Tab(label="🌳 Clasificación", tab_style={'fontWeight': 'bold'}, children=[
            dbc.Container([
                # Título y descripción
                dbc.Row([
                    dbc.Col([
                        html.H3("Modelo de Clasificación – Árbol de Decisión",
                                className="text-center mb-4 mt-4",
                                style={'fontWeight': 'bold', 'color': COLOR_PRIMARY}),
                        html.P(
                            "El árbol de decisión clasifica el flujo vehicular en tres niveles según el volumen total de vehículos:",
                            className="text-center mb-3",
                            style={'fontSize': '1.1rem'}
                        ),
                        dbc.Row([
                            dbc.Col([
                                dbc.Badge("🟢 Bajo", color="success", className="p-2",
                                          style={'fontSize': '1rem'})
                            ], width="auto"),
                            dbc.Col([
                                dbc.Badge("🟡 Medio", color="warning", className="p-2",
                                          style={'fontSize': '1rem'})
                            ], width="auto"),
                            dbc.Col([
                                dbc.Badge("🔴 Alto", color="danger", className="p-2",
                                          style={'fontSize': '1rem'})
                            ], width="auto")
                        ], justify="center", className="mb-4")
                    ])
                ]),

                # Métricas principales
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H2(f"{accuracy:.1%}", className="text-center mb-0",
                                        style={'color': COLOR_SUCCESS, 'fontWeight': 'bold'}),
                                html.P(
                                    "Accuracy", className="text-center text-muted mb-0")
                            ])
                        ], className="shadow")
                    ], md=4),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H2(f"{report['macro avg']['precision']:.1%}",
                                        className="text-center mb-0",
                                        style={'color': COLOR_INFO, 'fontWeight': 'bold'}),
                                html.P("Precision",
                                       className="text-center text-muted mb-0")
                            ])
                        ], className="shadow")
                    ], md=4),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H2(f"{report['macro avg']['recall']:.1%}",
                                        className="text-center mb-0",
                                        style={'color': COLOR_WARNING, 'fontWeight': 'bold'}),
                                html.P(
                                    "Recall", className="text-center text-muted mb-0")
                            ])
                        ], className="shadow")
                    ], md=4)
                ], className="mb-5"),

                # Gráfico 1: Matriz de Confusión
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("📋 Descripción", className="mb-3",
                                        style={'color': COLOR_PRIMARY}),
                                html.P(
                                    "La matriz de confusión muestra cómo el modelo clasifica correctamente "
                                    "cada nivel de flujo vehicular. Los valores en la diagonal representan "
                                    "las predicciones correctas.",
                                    style={'fontSize': '1.05rem',
                                           'lineHeight': '1.7'}
                                ),
                                html.P(
                                    "✅ Valores altos en la diagonal = Buen desempeño",
                                    style={
                                        'fontSize': '1rem', 'color': COLOR_SUCCESS, 'fontWeight': 'bold'}
                                )
                            ])
                        ], className="shadow-sm h-100")
                    ], md=4),
                    dbc.Col([
                        dcc.Graph(figure=fig_cm, config={
                                  'displayModeBar': False})
                    ], md=8)
                ], className="mb-5", align="center"),

                # Gráfico 2: Métricas por Clase
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=fig_metricas_clf, config={
                                  'displayModeBar': False})
                    ], md=8),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("📊 Descripción", className="mb-3",
                                        style={'color': COLOR_PRIMARY}),
                                html.P(
                                    "Evaluación detallada del rendimiento del modelo para cada categoría de flujo vehicular:",
                                    style={
                                        'fontSize': '1.05rem', 'lineHeight': '1.7', 'marginBottom': '12px'}
                                ),
                                html.Div([
                                    html.P([
                                        html.Strong("🔵 Precision: ", style={
                                                    'color': COLOR_PRIMARY}),
                                        "De todas las predicciones de una clase, ¿cuántas fueron correctas? "
                                        "Alta precisión significa pocas falsas alarmas."
                                    ], style={'fontSize': '0.95rem', 'marginBottom': '10px', 'lineHeight': '1.6'}),
                                    html.P([
                                        html.Strong("🟢 Recall: ", style={
                                                    'color': COLOR_SUCCESS}),
                                        "De todos los casos reales de una clase, ¿cuántos detectó el modelo? "
                                        "Alto recall significa que no se pierden casos importantes."
                                    ], style={'fontSize': '0.95rem', 'marginBottom': '10px', 'lineHeight': '1.6'}),
                                    html.P([
                                        html.Strong(
                                            "🟡 F1-Score: ", style={'color': COLOR_WARNING}),
                                        "Promedio armónico entre Precision y Recall. "
                                        "Indica el balance general del modelo para cada clase."
                                    ], style={'fontSize': '0.95rem', 'marginBottom': '0px', 'lineHeight': '1.6'})
                                ])
                            ])
                        ], className="shadow-sm h-100")
                    ], md=4)
                ], className="mb-5", align="center"),

                # Gráfico 3: Árbol de Decisión Visual
                dbc.Row([
                    dbc.Col([
                        html.H4("🌳 Árbol de Decisión – Clasificación del Flujo Vehicular",
                                className="text-center mb-3",
                                style={'fontWeight': 'bold', 'color': COLOR_PRIMARY}),
                        html.P(
                            "Visualización completa del árbol de decisión entrenado. Cada nodo muestra la condición "
                            "de división, el índice Gini, el número de muestras y la clase predicha. "
                            "Los colores indican la clase mayoritaria: 🟢 Verde (Bajo), 🟡 Morado (Medio), 🔴 Naranja (Alto).",
                            className="text-center mb-4",
                            style={'fontSize': '1.05rem'}
                        ),
                        dbc.Card([
                            dbc.CardBody([
                                html.Img(
                                    src=f"data:image/png;base64,{tree_img_base64}" if has_tree_image else "",
                                    style={
                                        'width': '100%',
                                        'maxWidth': '100%',
                                        'height': 'auto',
                                        'display': 'block',
                                        'margin': 'auto'
                                    },
                                    className="img-fluid"
                                ) if has_tree_image else html.P(
                                    "⚠️ No se pudo generar la visualización del árbol. "
                                    "Asegúrate de tener matplotlib instalado.",
                                    className="text-center text-warning"
                                )
                            ])
                        ], className="shadow-lg")
                    ])
                ], className="mb-5"),

                # Gráfico 4: Importancia de Variables
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("🔍 Descripción", className="mb-3",
                                        style={'color': COLOR_PRIMARY}),
                                html.P(
                                    "Muestra qué tipo de vehículo tiene mayor impacto en la clasificación "
                                    "del flujo vehicular. Las variables con mayor importancia son las que "
                                    "más influyen en las decisiones del modelo.",
                                    style={'fontSize': '1.05rem',
                                           'lineHeight': '1.7'}
                                ),
                                html.P([
                                    html.Strong("Variable más importante: "),
                                    f"{importancias.index[0]} ({importancias.values[0]:.1%})"
                                ], style={'fontSize': '1rem', 'color': COLOR_WARNING, 'fontWeight': 'bold'})
                            ])
                        ], className="shadow-sm h-100")
                    ], md=4),
                    dbc.Col([
                        dcc.Graph(figure=fig_importancia, config={
                                  'displayModeBar': False})
                    ], md=8)
                ], className="mb-5", align="center")

            ], fluid=True)
        ]),

        # ===============================
        # TAB 5: REGRESIÓN
        # ===============================
        dbc.Tab(label="📈 Regresión", tab_style={'fontWeight': 'bold'}, children=[
            dbc.Container([
                # Título
                dbc.Row([
                    dbc.Col([
                        html.H3("Modelo de Regresión Lineal",
                                className="text-center mb-4 mt-4",
                                style={'fontWeight': 'bold', 'color': COLOR_PRIMARY}),
                        html.P(
                            "Predicción del volumen de vehículos ligeros para identificar peajes con mayor saturación",
                            className="text-center mb-4",
                            style={'fontSize': '1.1rem'}
                        )
                    ])
                ]),

                # Métricas principales
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H3(f"{mae:,.0f}", className="text-center mb-0",
                                        style={'color': COLOR_PRIMARY, 'fontWeight': 'bold'}),
                                html.P("MAE", className="text-center text-muted mb-0",
                                       style={'fontSize': '0.9rem'}),
                                html.Small("Mean Absolute Error",
                                           className="text-center d-block text-muted")
                            ])
                        ], className="shadow")
                    ], md=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H3(f"{mse:,.0f}", className="text-center mb-0",
                                        style={'color': COLOR_INFO, 'fontWeight': 'bold'}),
                                html.P("MSE", className="text-center text-muted mb-0",
                                       style={'fontSize': '0.9rem'}),
                                html.Small("Mean Squared Error",
                                           className="text-center d-block text-muted")
                            ])
                        ], className="shadow")
                    ], md=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H3(f"{rmse:,.0f}", className="text-center mb-0",
                                        style={'color': COLOR_WARNING, 'fontWeight': 'bold'}),
                                html.P("RMSE", className="text-center text-muted mb-0",
                                       style={'fontSize': '0.9rem'}),
                                html.Small("Root Mean Squared Error",
                                           className="text-center d-block text-muted")
                            ])
                        ], className="shadow")
                    ], md=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H3(f"{r2:.3f}", className="text-center mb-0",
                                        style={'color': COLOR_SUCCESS, 'fontWeight': 'bold'}),
                                html.P("R²", className="text-center text-muted mb-0",
                                       style={'fontSize': '0.9rem'}),
                                html.Small("Coef. Determinación",
                                           className="text-center d-block text-muted")
                            ])
                        ], className="shadow")
                    ], md=3)
                ], className="mb-5"),

                # Gráfico 1: Distribución de Residuales
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=fig_residuales, config={
                                  'displayModeBar': False})
                    ], md=8),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("📊 Descripción", className="mb-3",
                                        style={'color': COLOR_PRIMARY}),
                                html.P(
                                    "Los residuales representan la diferencia entre los valores reales "
                                    "y las predicciones del modelo.",
                                    style={'fontSize': '1.05rem',
                                           'lineHeight': '1.7'}
                                ),
                                html.P(
                                    "✅ Una distribución centrada en cero indica que el modelo no tiene "
                                    "sesgo sistemático.",
                                    style={'fontSize': '1rem',
                                           'color': COLOR_SUCCESS}
                                )
                            ])
                        ], className="shadow-sm h-100")
                    ], md=4)
                ], className="mb-5", align="center"),

                # Gráfico 2: Valores Reales vs Predichos
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("📈 Descripción", className="mb-3",
                                        style={'color': COLOR_PRIMARY}),
                                html.P(
                                    "Comparación visual entre los valores reales y las predicciones del modelo.",
                                    style={'fontSize': '1.05rem',
                                           'lineHeight': '1.7'}
                                ),
                                html.P(
                                    "🎯 Los puntos cercanos a la línea roja indican predicciones precisas. "
                                    "Mayor dispersión = menor precisión.",
                                    style={'fontSize': '1rem',
                                           'color': COLOR_DANGER}
                                )
                            ])
                        ], className="shadow-sm h-100")
                    ], md=4),
                    dbc.Col([
                        dcc.Graph(figure=fig_scatter, config={
                                  'displayModeBar': False})
                    ], md=8)
                ], className="mb-5", align="center"),

                # Gráfico 3: Top 10 Peajes
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=fig_top_peajes, config={
                                  'displayModeBar': False})
                    ], md=8),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("🚨 Descripción", className="mb-3",
                                        style={'color': COLOR_PRIMARY}),
                                html.P(
                                    "Predicción de los 10 peajes con mayor saturación esperada para 2025 "
                                    "basada en el modelo de regresión.",
                                    style={'fontSize': '1.05rem',
                                           'lineHeight': '1.7'}
                                ),
                                html.P([
                                    html.Strong("Peaje más crítico: "),
                                    f"{top_peajes.index[0]} ({top_peajes.values[0]:,.0f} vehículos)"
                                ], style={'fontSize': '1rem', 'color': COLOR_DANGER, 'fontWeight': 'bold'})
                            ])
                        ], className="shadow-sm h-100")
                    ], md=4)
                ], className="mb-5", align="center")

            ], fluid=True)
        ]),

        # ===============================
        # TAB 6: CONCLUSIÓN
        # ===============================
        dbc.Tab(label="✅ Conclusión", tab_style={'fontWeight': 'bold'}, children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3("🎯 Conclusión General", className="text-center mb-4",
                                    style={'fontWeight': 'bold', 'color': COLOR_PRIMARY}),

                            html.H5("📊 Hallazgos Principales:", className="mb-3",
                                    style={'color': COLOR_SUCCESS}),
                            html.Ul([
                                html.Li([
                                    html.Strong("Clasificación: "),
                                    f"El árbol de decisión alcanzó una precisión del {accuracy:.1%}, "
                                    "demostrando su capacidad para distinguir entre niveles de congestión."
                                ], style={'fontSize': '1.1rem', 'marginBottom': '15px'}),
                                html.Li([
                                    html.Strong("Variable Clave: "),
                                    f"Los vehículos ligeros ({importancias.index[0]}) son el factor "
                                    f"más determinante con una importancia del {importancias.values[0]:.1%}."
                                ], style={'fontSize': '1.1rem', 'marginBottom': '15px'}),
                                html.Li([
                                    html.Strong("Predicción: "),
                                    f"El modelo de regresión identificó los 10 peajes más críticos con un R² de {r2:.3f}."
                                ], style={'fontSize': '1.1rem', 'marginBottom': '15px'}),
                                html.Li([
                                    html.Strong("Peajes Críticos: "),
                                    f"{top_peajes.index[0]}, {top_peajes.index[1]} y {top_peajes.index[2]} "
                                    "requerirán atención prioritaria en 2025."
                                ], style={'fontSize': '1.1rem', 'marginBottom': '15px'})
                            ], className="mb-4"),

                            html.H5("💡 Recomendaciones:", className="mb-3",
                                    style={'color': COLOR_WARNING}),
                            dbc.Row([
                                dbc.Col([
                                    html.H6("⚡ Corto Plazo:",
                                            className="mb-2"),
                                    html.Ul([
                                        html.Li("Monitoreo en tiempo real", style={
                                                'fontSize': '1.05rem'}),
                                        html.Li("Aumentar personal en peajes críticos", style={
                                                'fontSize': '1.05rem'}),
                                        html.Li("Optimizar distribución de carriles", style={
                                                'fontSize': '1.05rem'})
                                    ])
                                ], md=6),
                                dbc.Col([
                                    html.H6("🚀 Largo Plazo:",
                                            className="mb-2"),
                                    html.Ul([
                                        html.Li("Planificar nueva infraestructura", style={
                                                'fontSize': '1.05rem'}),
                                        html.Li("Implementar tarifas dinámicas", style={
                                                'fontSize': '1.05rem'}),
                                        html.Li("Sistema de alertas tempranas", style={
                                                'fontSize': '1.05rem'})
                                    ])
                                ], md=6)
                            ])
                        ])
                    ], className="shadow-lg", style={'borderTop': f'5px solid {COLOR_PRIMARY}'})
                ], md=10)
            ], className="mt-4 mb-5", justify="center")
        ])

    ])
], fluid=True, style={'backgroundColor': '#f8f9fa'})


if __name__ == "__main__":
    app.run(debug=True, port=8060)
















































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

df = df.rename(columns={"A O": "AÑO"})

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
# GRÁFICO DE VALORES FALTANTES
# ===============================

missing_data = df.isnull().sum().reset_index()
missing_data.columns = ["Columna", "Valores Faltantes"]

fig_missing = px.bar(
    missing_data,
    x="Columna",
    y="Valores Faltantes",
    title="Valores Faltantes por Variable",
    text="Valores Faltantes"
)

fig_missing.update_layout(
    template="plotly_white",
    xaxis_tickangle=-45
)


# ===============================
# BOXPLOT VEH_TOTAL
# ===============================

fig_boxplot = px.box(
    df,
    y="VEH_TOTAL",
    title="Distribución y Outliers de VEH_TOTAL"
)

fig_boxplot.update_layout(template="plotly_white")

# ==============================
# TABLA: ESTADÍSTICOS VEH_TOTAL
# ==============================

stats = df["VEH_TOTAL"].describe().reset_index()
stats.columns = ["Estadístico", "Valor"]

# Redondear valores
stats["Valor"] = stats["Valor"].round(2)





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
# TAB 2 - CONOCIENDO EL NEGOCIO
# ===============================
dbc.Tab(label="Conociendo el Negocio", children=[

    html.Br(),

    html.H4(
        "Análisis Estratégico del Flujo Vehicular",
        className="text-center fw-bold"
    ),

    html.Br(),

    dbc.Row([

        # ===============================
        # GRÁFICO 1 - EVOLUCIÓN POR AÑO
        # ===============================
        dbc.Col(
            dbc.Card([
                dbc.CardBody([

                    html.H5(
                        "Evolución del Flujo Vehicular por Año",
                        className="text-center"
                    ),

                    dcc.Graph(figure=fig_anio),

                    html.P(
                        "El análisis anual permite observar la tendencia del flujo "
                        "vehicular a lo largo del tiempo. Se identifican periodos "
                        "de crecimiento sostenido, lo que puede indicar mayor "
                        "actividad económica y movilidad. Esta información es clave "
                        "para proyectar demanda futura y planificar infraestructura.",
                        className="mt-3",
                        style={"textAlign": "justify"}
                    )

                ])
            ], className="shadow-sm"),
            md=6
        ),

        # ===============================
        # GRÁFICO 2 - POR DEPARTAMENTO
        # ===============================
        dbc.Col(
            dbc.Card([
                dbc.CardBody([

                    html.H5(
                        "Flujo Vehicular Total por Departamento",
                        className="text-center"
                    ),

                    dcc.Graph(figure=fig_depa),

                    html.P(
                        "Este gráfico permite identificar los departamentos con "
                        "mayor concentración vehicular. Aquellos con niveles más "
                        "altos representan zonas estratégicas para inversión y "
                        "optimización de recursos, mientras que los de menor flujo "
                        "pueden indicar oportunidades de desarrollo o menor demanda.",
                        className="mt-3",
                        style={"textAlign": "justify"}
                    )

                ])
            ], className="shadow-sm"),
            md=6
        )

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
dbc.Tab(label="Limpieza y Transformación", children=[

    html.Br(),

    html.H4(
        "Calidad y Preparación de Datos",
        className="text-center fw-bold"
    ),

    html.Br(),

    # ===============================
    # FILA 1 → DOS GRÁFICOS LADO A LADO
    # ===============================
    dbc.Row([

        # ---- GRÁFICO 1: VALORES FALTANTES ----
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H5("Valores Faltantes", className="text-center"),
                    dcc.Graph(figure=fig_missing),

                    html.P(
                        "El análisis de valores faltantes muestra que el dataset "
                        "no presenta datos nulos en sus variables principales, "
                        "lo que garantiza integridad y consistencia en la información "
                        "para el análisis posterior.",
                        className="mt-3",
                        style={"textAlign": "justify"}
                    )
                ])
            ], className="shadow-sm"),
            md=6
        ),

        # ---- GRÁFICO 2: BOXPLOT ----
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H5("Detección de Outliers (VEH_TOTAL)", className="text-center"),
                    dcc.Graph(figure=fig_boxplot),

                    html.P(
                        "El boxplot evidencia la presencia de valores atípicos "
                        "en el flujo vehicular total. La amplitud de la caja y "
                        "la extensión de los puntos superiores indican una alta "
                        "variabilidad entre peajes, con registros que superan "
                        "considerablemente el comportamiento promedio.",
                        className="mt-3",
                        style={"textAlign": "justify"}
                    )
                ])
            ], className="shadow-sm"),
            md=6
        )

    ]),

    html.Br(),

    # ===============================
    # FILA 2 → TABLA CENTRADA CON INTERPRETACIÓN
    # ===============================
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H5("Estadísticos Descriptivos de VEH_TOTAL",
                            className="text-center"),

                    dbc.Table.from_dataframe(
                        stats,
                        striped=True,
                        bordered=True,
                        hover=True,
                        responsive=True,
                        className="mt-3"
                    ),

                    html.P(
                        "La media (78,384) es considerablemente mayor que la "
                        "mediana (41,815), lo que confirma una distribución "
                        "asimétrica hacia la derecha. La desviación estándar "
                        "(97,001) refleja una fuerte dispersión de los datos. "
                        "Además, el valor máximo (980,656) supera ampliamente "
                        "el tercer cuartil (97,636), evidenciando la existencia "
                        "de outliers que influyen en el promedio general.",
                        className="mt-3",
                        style={"textAlign": "justify"}
                    )
                ])
            ], className="shadow-sm"),
            md=8
        )
    ], justify="center")

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

