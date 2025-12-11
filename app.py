import pandas as pd
import numpy as np
from dash import Dash, html, dcc, Input, Output
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pymongo

# ---------------- MONGO ----------------
MONGO_URL = "mongodb+srv://MurielB:vHYfVK5HwilbMVb5@cluster0.uxcdlte.mongodb.net/Sirene?retryWrites=true&w=majority&appName=Cluster0"
client = pymongo.MongoClient(MONGO_URL)
db = client["Sirene"]
collection = db["Ocorrencia"]

# Carrega todos os dados
df_full = pd.DataFrame(list(collection.find()))
total_ocorrencia = len(df_full)  # Total real do banco

# Filtra dados válidos para gráficos
df = df_full.copy()
df['data_hora'] = pd.to_datetime(df['data_hora'].astype(str), errors='coerce')
df.dropna(subset=['data_hora', 'latitude', 'longitude'], inplace=True)

# ---------------- TRANSFORMAÇÕES ----------------
def hora_para_turno(h):
    if 6 <= h < 12:
        return 'Manhã'
    elif 12 <= h < 18:
        return 'Tarde'
    else:
        return 'Noite'

df['turno'] = df['data_hora'].dt.hour.apply(hora_para_turno)
df['dia_semana'] = df['data_hora'].dt.dayofweek
df['hora'] = df['data_hora'].dt.hour
df['mes'] = df['data_hora'].dt.month
df['semana_ano'] = df['data_hora'].dt.isocalendar().week.astype(int)
df['semana_sin'] = np.sin(2 * np.pi * df['semana_ano'] / 52)
df['semana_cos'] = np.cos(2 * np.pi * df['semana_ano'] / 52)
df['lat_grid'] = df['latitude'].round(2)
df['lon_grid'] = df['longitude'].round(2)

def mode_or_nan(x):
    try:
        return x.mode().iloc[0]
    except Exception:
        return np.nan

df_ml = df.groupby(['lat_grid', 'lon_grid']).agg(
    contagem_ocorrencia=('tipo_ocorrencia', 'count'),
    hora_media=('hora', 'mean'),
    dia_semana_moda=('dia_semana', lambda x: mode_or_nan(x)),
    semana_media=('semana_ano', 'mean'),
    semana_sin_media=('semana_sin', 'mean'),
    semana_cos_media=('semana_cos', 'mean'),
    cidade_moda=('cidade', lambda x: mode_or_nan(x)),
    prioridade_moda=('prioridade', lambda x: mode_or_nan(x))
).reset_index()

limiar_risco = df_ml['contagem_ocorrencia'].quantile(0.75)
df_ml['risco_alto'] = (df_ml['contagem_ocorrencia'] >= limiar_risco).astype(int)

# Label Encoding
le_cidade = LabelEncoder()
df_ml['cidade_moda_filled'] = df_ml['cidade_moda'].fillna("Desconhecida")
df_ml['cidade_encoded'] = le_cidade.fit_transform(df_ml['cidade_moda_filled'])

le_prior = LabelEncoder()
df_ml['prioridade_moda_filled'] = df_ml['prioridade_moda'].fillna("Desconhecida")
df_ml['prioridade_encoded'] = le_prior.fit_transform(df_ml['prioridade_moda_filled'])

features = [
    'lat_grid', 'lon_grid',
    'hora_media', 'dia_semana_moda',
    'semana_media', 'semana_sin_media', 'semana_cos_media',
    'cidade_encoded', 'prioridade_encoded'
]

for f in features:
    if df_ml[f].isna().any() and df_ml[f].dtype in ['float64', 'int64']:
        df_ml[f].fillna(df_ml[f].mean(), inplace=True)

X = df_ml[features].copy()
y = df_ml['risco_alto']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

df_ml['probabilidade_risco_base'] = model.predict_proba(df_ml[features])[:, 1]

df_counts = df.groupby('tipo_ocorrencia').size().reset_index(name='quantidade')

opcoes_tipo = ['Todas_Ocorrencia'] + df['tipo_ocorrencia'].unique().tolist()
opcoes_turno = ['Todos_Turnos'] + df['turno'].unique().tolist()
opcoes_cidade = ['Todas_Células'] + sorted(df_ml['cidade_moda_filled'].unique().tolist())

total_abertas = df[df["status"].isin(["Em andamento", "Em análise"])].shape[0] 
total_resolvidas = df[df["status"] == "Fechado"].shape[0]

map_center = {
    'lat': df_ml['lat_grid'].mean() if not df_ml['lat_grid'].isna().all() else -8.05,
    'lon': df_ml['lon_grid'].mean() if not df_ml['lon_grid'].isna().all() else -34.9
}

# ---------------- DASH ----------------
app = Dash(__name__)
server = app.server

card_style = {
    "background": "white",
    "padding": "20px",
    "borderRadius": "12px",
    "width": "250px",
    "boxShadow": "0 2px 10px rgba(0,0,0,0.05)",
    "marginRight": "15px"
}

chart_box_common = {
    "background": "white",
    "padding": "20px",
    "borderRadius": "12px",
    "boxShadow": "0 2px 10px rgba(0,0,0,0.08)",
}

chart_container = {
    "display": "flex",
    "gap": "20px",
    "marginTop": "30px",
    "width": "100%", 
}

app.layout = html.Div(style={"padding": "30px", "fontFamily": "Arial, sans-serif", "backgroundColor": "#F7F7F7"}, children=[

    html.H1("Dashboard Sirene", style={"marginBottom": "30px", "color": "#333"}),

    html.Div(style={"display": "flex", "gap": "20px", "marginBottom": "20px"}, children=[
        html.Div(style=card_style, children=[
            html.H4("Total de Ocorrências", style={"fontSize": "1.1em", "color": "#555"}),
            html.H1(total_ocorrencia, style={"fontSize": "2.5em", "fontWeight": "bold", "color": "#C90000"})
        ]),
        html.Div(style=card_style, children=[
            html.H4("Ocorrências Abertas", style={"fontSize": "1.1em", "color": "#555"}),
            html.H1(total_abertas, style={"fontSize": "2.5em", "fontWeight": "bold", "color": "#FF6347"})
        ]),
        html.Div(style=card_style, children=[
            html.H4("Fechadas", style={"fontSize": "1.1em", "color": "#555"}),
            html.H1(total_resolvidas, style={"fontSize": "2.5em", "fontWeight": "bold", "color": "#470404"})
        ]),
        html.Div(style=card_style, children=[
            html.H4("Tempo Médio de Resposta", style={"fontSize": "1.1em", "color": "#555"}),
            html.H1("—", style={"fontSize": "2.5em", "fontWeight": "bold", "color": "#333"})
        ]),
    ]),

    html.Div(style=chart_container, children=[
        html.Div(style={**chart_box_common, "width": "35%", "height": "450px"}, children=[
            html.H3("Ocorrências por Turno", style={"fontSize": "1.4em", "marginBottom": "15px"}),
            dcc.Dropdown(opcoes_turno, value="Todos_Turnos", id="dropdown_pizza", clearable=False),
            dcc.Graph(id="grafico_pizza", style={"height": "320px"})
        ]),
        html.Div(style={**chart_box_common, "width": "60%", "height": "450px"}, children=[
            html.H3("Ocorrências por Tipo", style={"fontSize": "1.4em", "marginBottom": "15px"}),
            dcc.Dropdown(opcoes_tipo, value="Todas_Ocorrencia", id="dropdown_barras", clearable=False),
            dcc.Graph(id="grafico_ocorrencia", style={"height": "320px"})
        ]),
    ]),

    html.Div(style={**chart_box_common, "marginTop": "30px", "width": "100%"}, children=[
        html.H3("Mapa de Risco Semanal (Previsão Random Forest)", style={"marginBottom": "8px", "color": "#222"}),

        html.Div(style={"display": "flex", "gap": "20px", "alignItems": "center", "marginBottom": "15px"}, children=[
            html.Div(children=[
                html.Label("Filtrar por cidade:"),
                dcc.Dropdown(opcoes_cidade, value="Todas_Células", id="dropdown_mapa_filtro", clearable=False, style={'width':'300px'})
            ]),
            html.Div(style={'flexGrow': '1'}, children=[
                html.Label("Semana do ano (1–52):"),
                dcc.Slider(id="slider_semana", min=1, max=52, step=1, 
                           value=int(df['semana_ano'].mode().iloc[0] if 'semana_ano' in df else 1),
                           marks={i: str(i) for i in range(1, 53, 5)}, 
                           tooltip={"placement": "bottom", "always_visible": True}, updatemode='mouseup')
            ])
        ]),

        dcc.Graph(id="grafico_mapa_calor_preditivo", style={"height": "550px"})
    ]),
])

# ---------------- CALLBACKS ----------------
@app.callback(
    Output('grafico_ocorrencia', 'figure'),
    Input('dropdown_barras', 'value')
)
def update_barras(value):
    if value == 'Todas_Ocorrencia':
        data_to_plot = df_counts
    else:
        temp = df[df['tipo_ocorrencia'] == value]
        data_to_plot = temp.groupby('tipo_ocorrencia').size().reset_index(name='quantidade')

    fig = px.bar(
        data_to_plot,
        x="tipo_ocorrencia",
        y="quantidade",
        color="tipo_ocorrencia",
        color_discrete_sequence=["#FA0505", "#A01E08", "#7E0C0C", "#5A0707", "#9C2B2B", "#6D2323"]
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor='white', paper_bgcolor='white', showlegend=False)
    fig.update_xaxes(title=None)
    fig.update_yaxes(title=None)
    return fig

@app.callback(
    Output('grafico_pizza', 'figure'),
    Input('dropdown_pizza', 'value')
)
def update_pizza(value):
    turno_count = df.groupby('turno').size().reset_index(name='quantidade')
    cores_originais = {'Manhã': "#E94B30", 'Tarde': '#B22222', 'Noite': "#5C0707"}
    DESTACADA = '#C90000'
    DESFOCADA = '#DDDDDD'

    if value == "Todos_Turnos":
        colors = [cores_originais.get(t, DESFOCADA) for t in turno_count['turno']]
    else:
        colors = [DESTACADA if t == value else DESFOCADA for t in turno_count['turno']]

    fig = px.pie(turno_count, names="turno", values="quantidade", hole=0.7,
                 color='turno', color_discrete_map=dict(zip(turno_count['turno'], colors)))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor='white', paper_bgcolor='white',
                      legend=dict(orientation="v", x=0.9, y=0.9, title=None, font=dict(size=15)))
    fig.update_traces(textinfo='label+percent', textposition='outside', marker=dict(line=dict(color='#F5F5F5', width=1)))
    return fig

@app.callback(
    Output('grafico_mapa_calor_preditivo', 'figure'),
    Input('dropdown_mapa_filtro', 'value'),
    Input('slider_semana', 'value')
)
def update_mapa(cidade_selecionada, semana_selecionada):
    df_plot = df_ml.copy()
    
    if cidade_selecionada != 'Todas_Células':
        df_plot = df_plot[df_plot['cidade_moda_filled'] == cidade_selecionada]

    if df_plot.empty:
        fig_empty = px.scatter_mapbox(pd.DataFrame({'lat':[], 'lon':[]}), lat='lat', lon='lon')
        fig_empty.update_layout(title="Nenhuma célula para o filtro selecionado.")
        return fig_empty

    semana_sin = np.sin(2 * np.pi * semana_selecionada / 52)
    semana_cos = np.cos(2 * np.pi * semana_selecionada / 52)

    X_plot = df_plot[features].copy()
    X_plot['semana_media'] = float(semana_selecionada)
    X_plot['semana_sin_media'] = semana_sin
    X_plot['semana_cos_media'] = semana_cos
    X_plot = X_plot.fillna(X_train.mean())

    df_plot['probabilidade_risco_semana'] = model.predict_proba(X_plot[features])[:, 1]

    center_lat = df_plot['lat_grid'].mean() if not df_plot.empty else map_center['lat']
    center_lon = df_plot['lon_grid'].mean() if not df_plot.empty else map_center['lon']

    fig = px.scatter_mapbox(df_plot, lat="lat_grid", lon="lon_grid",
                            color="probabilidade_risco_semana",
                            size="contagem_ocorrencia",
                            hover_name="cidade_moda_filled",
                            hover_data={'contagem_ocorrencia': True,
                                        'probabilidade_risco_semana': ':.2f',
                                        'lat_grid': False,
                                        'lon_grid': False},
                            color_continuous_scale=px.colors.sequential.Inferno,
                            zoom=9,
                            center=dict(lat=center_lat, lon=center_lon),
                            mapbox_style="carto-positron",
                            title=f"Probabilidade de Risco - Semana {semana_selecionada}")

    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), coloraxis_colorbar=dict(title="Probabilidade de Risco"))
    return fig

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 8050))  # pega a porta do Render ou 8050 localmente
    app.run(host="0.0.0.0", port=port, debug=True)