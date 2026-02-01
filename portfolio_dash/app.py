"""
James Callahan Portfolio - Dash version.
Run with: python app.py
"""
import os
import random
from pathlib import Path

import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
from dotenv import load_dotenv

load_dotenv()

# Paths relative to this app's directory
APP_DIR = Path(__file__).parent
CRASH_DATA_PATH = APP_DIR / "crash_data_prepped.csv"

app = dash.Dash(
    __name__,
    title="Callahan Portfolio",
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.CYBORG, "https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&display=swap"],
    assets_folder="assets",
)

# Sidebar — vertical nav with brand at top (plain div + dcc.Link so it always renders)
sidebar = html.Div(
    [
        html.Div(
            dcc.Link("Callahan Portfolio", href="/", className="sidebar-brand"),
            className="sidebar-header",
        ),
        html.Hr(className="sidebar-divider"),
        dcc.Link("Home", href="/", className="sidebar-link", id="sidebar-link-home"),
        dcc.Link("Clustering", href="/clustering", className="sidebar-link", id="sidebar-link-clustering"),
        dcc.Link("Toastmasters", href="/toastmasters", className="sidebar-link", id="sidebar-link-toastmasters"),
        dcc.Link("Car Prices", href="/car-prices", className="sidebar-link", id="sidebar-link-car-prices"),
    ],
    className="portfolio-sidebar",
)

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        html.Div(sidebar, className="sidebar-wrapper"),
        html.Div(
            html.Div(id="page-content", className="portfolio-page"),
            className="portfolio-main",
        ),
    ],
    className="portfolio-dark portfolio-with-sidebar",
)


# ---- Home page ----
def layout_home():
    return html.Div(
        [
            html.H1("Welcome to my portfolio!", className="mb-4"),
            html.P(
                "Hi, I'm James. I work as a data scientist and am passionate about the field. "
                "In my free time, I like to work out, travel, and latin dance! Working on cars and "
                "reading are some other things that bring me joy.",
                className="lead",
            ),
            html.Img(
                src=app.get_asset_url("IMG_5550.jpeg"),
                alt="James in Sri Lanka",
                className="portfolio-photo",
            ),
            html.P("A picture of me on a trip to Sri Lanka.", className="text-muted small"),
            html.H4("Resume", className="mt-4 mb-2 section-title"),
            html.Iframe(
                src=app.get_asset_url("resume.pdf"),
                className="resume-embed",
            ),
            html.Div(
                [
                    html.A(
                        "Download resume as PDF",
                        href=app.get_asset_url("resume.pdf"),
                        download="James_Callahan_Resume.pdf",
                        className="btn btn-resume mt-3",
                    ),
                ],
                className="mb-4",
            ),
            html.Hr(className="portfolio-hr"),
            html.P(
                [
                    "Go to my ",
                    html.A("LinkedIn", href="https://www.linkedin.com/in/jamesacallahan/", target="_blank", className="portfolio-link"),
                    " | ",
                    html.A("GitHub", href="https://github.com/jcallahan997", target="_blank", className="portfolio-link"),
                ],
                className="mb-2",
            ),
            html.P("Thank you for visiting my page!", className="text-muted"),
        ],
        className="page-container",
    )


# ---- Clustering page (lazy import to avoid loading heavy deps on home) ----
def layout_clustering():
    return html.Div(
        [
            html.H1("Hierarchical Clustering on Crash Data", className="mb-3"),
            html.P(
                [
                    "Data Source: ",
                    html.A(
                        "US Accidents (Kaggle)",
                        href="https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents/data",
                        target="_blank",
                        className="portfolio-link",
                    ),
                ]
            ),
            html.P(
                [
                    "Project repo: ",
                    html.A(
                        "Unsup_ML_Dockerized_App",
                        href="https://github.com/jcallahan997/Unsup_ML_Dockerized_App",
                        target="_blank",
                        className="portfolio-link",
                    ),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("State", className="fw-bold"),
                            dcc.Dropdown(
                                id="clustering-state",
                                options=STATE_OPTIONS,
                                value="CA",
                                clearable=False,
                                className="mb-3",
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            html.Label("Sample size", className="fw-bold"),
                            dcc.Dropdown(
                                id="clustering-sample-size",
                                options=[500, 1000, 2500, 5000, 7500, 10000],
                                value=2500,
                                clearable=False,
                                className="mb-3",
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            html.Label("\u00a0", className="d-block"),
                            dbc.Button("Resample data", id="clustering-resample", className="btn-clustering"),
                        ],
                        md=4,
                    ),
                ],
                className="mb-4",
            ),
            html.Label("Distance threshold", className="fw-bold"),
            dcc.Slider(
                id="clustering-distance",
                min=0,
                max=100,
                value=20,
                step=1,
                marks={0: "0", 25: "25", 50: "50", 75: "75", 100: "100"},
                className="mb-4",
            ),
            html.Hr(),
            html.H4("Correlation heatmap (sample of 100)"),
            dcc.Graph(id="clustering-heatmap"),
            html.P(
                "Note: Heatmap uses unscaled data. Correlation is not influenced by scale. "
                "Scaled data below uses median imputation for nulls.",
                className="small text-muted",
            ),
            html.H4("Scaled data (sample of 100)", className="mt-4"),
            html.Div(id="clustering-scaled-table"),
            html.H4("Dendrogram", className="mt-4"),
            dcc.Graph(id="clustering-dendrogram"),
            html.H4("Cluster summary", className="mt-4"),
            html.Div(id="clustering-summary-table"),
            html.H4("Clustered data (sample of 100)", className="mt-4"),
            html.Div(id="clustering-data-table"),
        ],
        className="page-container",
    )


STATE_DICT = {
    "AL": "Alabama", "AR": "Arkansas", "AZ": "Arizona", "CA": "California", "CO": "Colorado",
    "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia", "IA": "Iowa",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "KS": "Kansas", "KY": "Kentucky",
    "LA": "Louisiana", "MA": "Massachusetts", "MD": "Maryland", "ME": "Maine", "MI": "Michigan",
    "MN": "Minnesota", "MO": "Missouri", "MS": "Mississippi", "MT": "Montana", "NC": "North Carolina",
    "ND": "North Dakota", "NE": "Nebraska", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NV": "Nevada", "NY": "New York", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VA": "Virginia", "VT": "Vermont", "WA": "Washington", "WI": "Wisconsin",
    "WV": "West Virginia", "WY": "Wyoming",
}
STATE_OPTIONS = [{"label": f"{abbr} ({name})", "value": abbr} for abbr, name in STATE_DICT.items()]


# ---- Toastmasters page ----
def layout_toastmasters():
    return html.Div(
        [
            html.H1("Toastmasters 'Table Topic' Questions Generator", className="mb-3"),
            html.P(
                "Table Topics® is a long-standing Toastmasters tradition intended to help members "
                "develop their ability to organize their thoughts quickly and respond to an impromptu question or topic."
            ),
            html.P(
                "I created this app because occasionally I don't have time to prepare 10+ creative questions "
                "surrounding a preselected theme to ask fellow Toastmasters during the meeting."
            ),
            html.P(
                "This app uses an Azure OpenAI instance (gpt-35-turbo) as part of this Dash portfolio.",
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.InputGroup(
                                [
                                    dbc.Input(
                                        id="toastmasters-input",
                                        placeholder="Enter a theme or message...",
                                        type="text",
                                        debounce=True,
                                    ),
                                    dbc.Button("Send", id="toastmasters-send", className="btn-toastmasters"),
                                ],
                                className="mb-3",
                            ),
                            html.Div(id="toastmasters-messages", className="mt-3"),
                        ],
                        md=8,
                    ),
                ]
            ),
            dcc.Store(id="toastmasters-store", data=[]),
        ],
        className="page-container",
    )


# ---- Car Prices page ----
def layout_car_prices():
    return html.Div(
        [
            html.H1("RShiny App: Car Prices", className="mb-3"),
            html.P(
                "An RShiny app embedded below. It's hosted on Shinyapps and explores the relationship "
                "between car prices and mileage.",
                className="mb-4",
            ),
            html.Iframe(
                src="https://fxyqh7-james-callahan.shinyapps.io/car_shinyapp/",
                style={"width": "100%", "height": "1950px", "border": "none"},
            ),
        ],
        className="page-container",
    )


@callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname == "/clustering":
        return layout_clustering()
    if pathname == "/toastmasters":
        return layout_toastmasters()
    if pathname == "/car-prices":
        return layout_car_prices()
    return layout_home()


@callback(
    [
        Output("sidebar-link-home", "className"),
        Output("sidebar-link-clustering", "className"),
        Output("sidebar-link-toastmasters", "className"),
        Output("sidebar-link-car-prices", "className"),
    ],
    Input("url", "pathname"),
)
def set_active_sidebar_link(pathname):
    pathname = pathname or "/"
    base = "sidebar-link"
    return (
        f"{base} active" if pathname == "/" else base,
        f"{base} active" if pathname == "/clustering" else base,
        f"{base} active" if pathname == "/toastmasters" else base,
        f"{base} active" if pathname == "/car-prices" else base,
    )


# ---- Clustering callbacks (import here so they only run when needed) ----
@callback(
    [
        Output("clustering-heatmap", "figure"),
        Output("clustering-scaled-table", "children"),
        Output("clustering-dendrogram", "figure"),
        Output("clustering-summary-table", "children"),
        Output("clustering-data-table", "children"),
    ],
    [
        Input("clustering-state", "value"),
        Input("clustering-sample-size", "value"),
        Input("clustering-distance", "value"),
        Input("clustering-resample", "n_clicks"),
    ],
)
def update_clustering(state, sample_size, distance_threshold, n_clicks):
    import pandas as pd
    import pyarrow
    from pyarrow import csv
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import AgglomerativeClustering
    import idendrogram
    import plotly.express as px
    import plotly.graph_objects as go
    from dash import dash_table

    if not CRASH_DATA_PATH.exists():
        empty = go.Figure().add_annotation(text="Place crash_data_prepped.csv in the portfolio_dash folder.")
        return empty, html.P("No data."), empty, html.P("No data."), html.P("No data.")

    car_safety = csv.read_csv(str(CRASH_DATA_PATH))
    car_safety = car_safety.filter(pyarrow.compute.equal(car_safety["State"], state))
    n_rows = car_safety.num_rows
    k = min(sample_size, n_rows) if sample_size else n_rows
    seed = (n_clicks or 0) * 10000 + hash(state) % 10000 + sample_size
    random.seed(seed)
    indices = random.sample(range(n_rows), k=k)
    car_safety = car_safety.take(indices)

    columns_heatmap = [
        "Severity", "Temperature(F)", "Humidity(%)", "Visibility(mi)",
        "Wind_Speed(mph)", "Precipitation(in)",
    ]
    heatmap_data = car_safety.select(["ID"] + columns_heatmap)
    heatmap_df = heatmap_data.to_pandas()
    heatmap_df = heatmap_df.fillna(heatmap_df[columns_heatmap].median())

    corr = heatmap_df[columns_heatmap].corr()
    heatmap_fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            text=corr.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
        )
    )
    heatmap_fig.update_layout(
        title="Correlation heatmap",
        xaxis={"tickangle": -45},
        yaxis=dict(autorange="reversed"),
        margin=dict(l=120),
        height=500,
    )

    scaler = StandardScaler()
    scaled = heatmap_df[columns_heatmap].copy()
    scaled[:] = scaler.fit_transform(scaled)
    scaled.insert(0, "ID", heatmap_df["ID"])
    scaled_df = scaled.head(100)

    scaled_table = dash_table.DataTable(
        data=scaled_df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in scaled_df.columns],
        page_size=10,
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "8px"},
    )

    X = scaled[columns_heatmap]
    model = AgglomerativeClustering(
        distance_threshold=distance_threshold, n_clusters=None, compute_distances=True
    )
    model.fit(X)
    cl = model.fit_predict(X)

    idd = idendrogram.idendrogram()
    idd.set_cluster_info(idendrogram.ScikitLearnClusteringData(model))
    dendro_fig = idd.create_dendrogram().plot(backend="plotly", height=600, width=700)
    if hasattr(dendro_fig, "to_dict"):
        dendro_fig = dendro_fig
    else:
        dendro_fig = go.Figure()

    heatmap_df["Cluster"] = cl
    summary_df = heatmap_df.groupby("Cluster", as_index=False).agg(
        {c: "mean" for c in columns_heatmap} | {"ID": "count"}
    ).rename(columns={"ID": "Count"})
    summary_table = dash_table.DataTable(
        data=summary_df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in summary_df.columns],
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "8px"},
    )

    data_table = dash_table.DataTable(
        data=heatmap_df.head(100).to_dict("records"),
        columns=[{"name": c, "id": c} for c in heatmap_df.columns],
        page_size=10,
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "8px"},
    )

    return heatmap_fig, scaled_table, dendro_fig, summary_table, data_table


# ---- Toastmasters callback ----
@callback(
    [
        Output("toastmasters-messages", "children"),
        Output("toastmasters-store", "data"),
        Output("toastmasters-input", "value"),
    ],
    Input("toastmasters-send", "n_clicks"),
    [State("toastmasters-input", "value"), State("toastmasters-store", "data")],
    prevent_initial_call=True,
)
def toastmasters_chat(n_clicks, prompt, messages):
    if not prompt or not prompt.strip():
        return dash.no_update, dash.no_update, ""
    from openai import AzureOpenAI

    endpoint = os.getenv("endpoint")
    key = os.getenv("API_KEY")
    deployment_name = os.getenv("deployment_name")
    if not all([endpoint, key, deployment_name]):
        return (
            html.P("Set API_KEY, endpoint, and deployment_name in .env to use this page.", className="text-danger"),
            messages,
            "",
        )

    messages = list(messages) if messages else []
    messages.append({"role": "user", "content": prompt.strip()})

    try:
        client = AzureOpenAI(
            api_key=key,
            api_version="2023-03-15-preview",
            azure_endpoint=endpoint,
        )
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            stream=False,
        )
        full_response = response.choices[0].message.content or ""
    except Exception as e:
        full_response = f"Error calling API: {e}"
    messages.append({"role": "assistant", "content": full_response})

    children = []
    for m in messages:
        if m["role"] == "user":
            children.append(
                dbc.Card(
                    dbc.CardBody(html.P(m["content"], className="mb-0")),
                    className="mb-2 chat-card chat-card-user",
                    style={"marginLeft": "2rem"},
                )
            )
        else:
            children.append(
                dbc.Card(
                    dbc.CardBody(html.P(m["content"], className="mb-0")),
                    className="mb-2 chat-card chat-card-assistant",
                    style={"marginRight": "2rem"},
                )
            )

    return children, messages, ""


if __name__ == "__main__":
    app.run(debug=True, port=8050)
