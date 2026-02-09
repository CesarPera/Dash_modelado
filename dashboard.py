from dash import Dash, html, dcc
import dash_bootstrap_components as dbc


app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Análisis Predictivo de Flujo Vehicular"

# ===============================
# LAYOUT
# ===============================
app.layout = dbc.Container([

    # --------- TÍTULO ---------
    dbc.Row([
        dbc.Col([
            html.H1("Análisis Predictivo del Flujo Vehicular",
                    className="text-center text-primary mt-4"),
            html.P("Clasificación y regresión aplicadas a datos de peajes",
                   className="text-center text-muted")
        ])
    ]),

    html.Hr(),

    # --------- TABS ---------
    dbc.Tabs([

        # ===============================
        # TAB 1: PROBLEMA Y OBJETIVO
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
        # TAB 2: CONOCIENDO EL NEGOCIO
        # ===============================
        dbc.Tab(label="Conociendo el Negocio", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Contexto del negocio"),
                    html.P(
                        "Los peajes son puntos críticos de control vehicular. "
                        "Una mala gestión puede generar retrasos, pérdidas económicas "
                        "y malestar en los usuarios."
                    ),

                    html.H4("Beneficio del análisis"),
                    html.Ul([
                        html.Li("Identificar peajes críticos"),
                        html.Li("Optimizar recursos y personal"),
                        html.Li("Apoyar la toma de decisiones estratégicas")
                    ])
                ], md=10)
            ], className="mt-4")
        ]),

        # ===============================
        # TAB 3: EDA
        # ===============================
        dbc.Tab(label="EDA", children=[
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
        dbc.Tab(label="Clasificación", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Modelo de Clasificación – Árbol de Decisión"),
                    html.P(
                        "El árbol de decisión se utilizó para clasificar el flujo vehicular "
                        "en niveles Bajo, Medio y Alto."
                    ),
                    html.P(
                        "Este modelo permite identificar qué tipo de vehículo "
                        "contribuye más a la congestión."
                    )
                ])
            ], className="mt-4")
        ]),

        # ===============================
        # TAB 5: REGRESIÓN
        # ===============================
        dbc.Tab(label="Regresión", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Modelo de Regresión Lineal"),
                    html.P(
                        "La regresión lineal se utilizó para predecir "
                        "qué peajes presentarán mayor saturación."
                    ),
                    html.P(
                        "Se analizaron variables temporales y tipos de vehículos "
                        "para estimar el flujo vehicular total."
                    )
                ])
            ], className="mt-4")
        ]),

        # ===============================
        # TAB 6: CONCLUSIÓN
        # ===============================
        dbc.Tab(label="Conclusión", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Conclusión General"),
                    html.P(
                        "El árbol de decisión permitió identificar los vehículos "
                        "con mayor impacto en la congestión, mientras que la regresión lineal "
                        "ayudó a predecir los peajes más saturados, aportando información clave "
                        "para la gestión del tráfico."
                    )
                ])
            ], className="mt-4")
        ])

    ])
], fluid=True)

if __name__ == "__main__":
    app.run(debug=True, port=8060)
