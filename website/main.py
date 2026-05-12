import dash
from google.cloud import firestore
from dash import html, dcc, Input, Output, State, dash_table
import pandas as pd
import plotly.express as px
import json
import os

import requests

# Base URL for the Inference Cloud Run service (override per environment).
INFERENCE_SERVICE_URL = os.environ.get(
    "INFERENCE_SERVICE_URL",
    "https://transit-inference-service-428591046247.us-west2.run.app",
).rstrip("/")

db = firestore.Client(database='map-data')

def load_data_from_firestore():
    docs = db.collection('route_metrics').stream()
    data = [doc.to_dict() for doc in docs]
    return pd.DataFrame(data)

map_data = load_data_from_firestore()

fig = px.scatter_map(
    map_data,
    lat="stop_lat",
    lon="stop_lon",
    color="Cluster_Label",          # Color stops by their Route's Cluster
    size="Underserved_Score",       # Size of the bubble based on the Underserved Score
    hover_name="stop_name",         # Tooltip title
    hover_data={                    # Extra data in the tooltip
        "Route": True,
        "Underserved_Score": ':.3f',
        "Percent Low-Income": ':.2%',
        "Total Population": True,
        "stop_lat": False,          # Hide raw lat/lon in tooltip
        "stop_lon": False,
        "Cluster_Label": False
    },
    color_discrete_sequence=px.colors.qualitative.G10,
    zoom=10,
    map_style="carto-positron",  # Clean, light background map
    title="Interactive Route Metrics Map",
    height=600
)



app = dash.Dash(__name__, suppress_callback_exceptions=True)

server = app.server


@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/methods':
        return render_methods_page()
    elif pathname == '/findings':
        return render_findings_page()
    elif pathname == '/EDA':
        return render_EDA_page()
    else:
        # Default to landing page (Home)
        return render_landing_page()


@app.callback(
    Output("findings-inference-output", "children"),
    Input("findings-predict-btn", "n_clicks"),
    State("findings-route-dropdown", "value"),
    prevent_initial_call=True,
)
def fetch_cloud_run_prediction(n_clicks, route_id):
    if not route_id:
        return html.P("Select a route before running a prediction.", style={"color": "#666"})

    url = f"{INFERENCE_SERVICE_URL}/predict"
    try:
        resp = requests.get(url, params={"route_id": route_id}, timeout=45)
    except requests.RequestException as exc:
        return html.Div(
            [
                html.P("Could not reach the inference service.", style={"color": "#d9534f", "fontWeight": "600"}),
                html.Pre(str(exc), style={"fontSize": "13px", "whiteSpace": "pre-wrap"}),
            ]
        )

    if resp.status_code != 200:
        detail = (resp.text or "")[:500]
        err_children = [
            html.P(
                f"Inference service returned HTTP {resp.status_code}.",
                style={"color": "#d9534f", "fontWeight": "600"},
            ),
        ]
        if detail:
            err_children.append(
                html.Pre(detail, style={"fontSize": "13px", "whiteSpace": "pre-wrap"})
            )
        return html.Div(err_children)

    try:
        payload = resp.json()
    except ValueError:
        return html.P("Response was not valid JSON.", style={"color": "#d9534f"})

    if "error" in payload:
        return html.P(payload["error"], style={"color": "#d9534f"})

    underserved = payload.get("is_underserved_prediction")
    score = payload.get("underserved_score")
    forecast = payload.get("next_month_forecast")
    metrics = payload.get("metrics") or {}

    forecast_display = f"{forecast:.4f}" if forecast is not None else "— (no Prophet model for this route)"

    return html.Div(
        className="metrics-grid",
        children=[
            html.Div(
                className="metric-box",
                children=[
                    html.Span("Route", className="metric-label"),
                    html.P(str(payload.get("route_id", route_id))),
                ],
            ),
            html.Div(
                className="metric-box",
                children=[
                    html.Span("Underserved (RF)", className="metric-label"),
                    html.P("Yes" if underserved == 1 else "No" if underserved == 0 else str(underserved)),
                ],
            ),
            html.Div(
                className="metric-box",
                children=[
                    html.Span("Underserved score", className="metric-label"),
                    html.P(f"{float(score):.4f}" if score is not None else "—"),
                ],
            ),
            html.Div(
                className="metric-box",
                children=[
                    html.Span("Next month forecast (Prophet)", className="metric-label"),
                    html.P(forecast_display),
                ],
            ),
            html.Div(
                style={"gridColumn": "1 / -1", "marginTop": "8px"},
                children=[
                    html.Strong("Input features sent to the RF: "),
                    html.Pre(
                        json.dumps(metrics, indent=2),
                        style={
                            "fontSize": "13px",
                            "backgroundColor": "#f8f9fa",
                            "padding": "12px",
                            "borderRadius": "6px",
                            "border": "1px solid #dee2e6",
                        },
                    ),
                ],
            ),
        ],
    )


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),

    html.Nav([
        dcc.Link('Home', href='/'),
        dcc.Link('EDA', href='/EDA'),
        dcc.Link('Methodology', href='/methods'),
        dcc.Link('Findings', href='/findings'),
    ], style={'display': 'flex', 'gap': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa'}),

    html.Div(id='page-content', style={'padding': '20px'})
])


# --- Page Content Functions ---
def render_landing_page():
    return html.Main(className="container", children=[
        html.H1("Identifying Underserved Transit Routes", className="page-title"),

        html.Section(className="section", children=[
            html.H2("Project Summary"),
            html.P(
                f"This project analyzes Santa Clara county transit routes by combining ridership demand, service frequency, and neighborhood demographics."),
            html.P(
                "Through this analysis, this project aims to evaluate whether service levels align with rider demand and community need. From our metrics, unsupervised clustering models will group routes into interpretable categories such as high-demand low-service corridors or low-demand overserved routes, and predictive models such as time-forecast models also estimate which routes are most at risk of becoming underserved in the near future. The project aims to provide actionable insights for equitable and efficient transit planning.")
        ]),

        html.Section(className="section", children=[
            html.H2("Main Goals"),
            html.Ul([
                html.Li("Evaluate Service Levels: Determine if transit service aligns with rider demand and community need."),
                html.Li("Identify Transit Gaps: Pinpoint underserved routes in Santa Clara County, specifically those serving low-income or vehicle-less households."),
                html.Li("Predictive Planning: Use time-forecast models and feature importance to estimate which routes are at risk of becoming underserved in the near future and what factors are most likely causing it.")
            ])
        ]),

        html.Section(className="section", children=[
            html.H2("Broader Impacts"),
            html.P(
                f"By identifying routes where demand and community need exceed available service, this project can help support the allocation of limited transit resources. Routes serving low-income communities or households without private vehicles may be especially important to prioritize given their riders, as these populations are often more dependent on reliable transit service."),
            html.P(
                "Improved route-level planning can reduce overcrowding, shorten wait times, and increase accessibility to jobs, schools, healthcare, and commercial centers. More effective transit systems may also encourage greater public transit use, helping reduce traffic congestion and vehicle emissions. In addition, the framework developed in this project can be adapted to other transit systems seeking data-driven approaches to balancing efficiency and equity in service planning.")
        ]),

        html.Section(className="section", children=[
            html.H2("Data Sources"),
            html.Ul([
                html.Li(html.A("VTA Historical Ridership Data", href="https://data.vta.org/datasets/VTA::bus-and-light-rail-average-ridership-data-2025/about", target="_blank")),
                html.Li(html.A("October 2024 VTA Ridership by Stop Data", href="https://vta.maps.arcgis.com/sharing/rest/content/items/7c8dfd36a3f74b06b058b54a583c277e/data", target="_blank")),
                html.Li(html.A("MTA Communities of Concern Demographic Data", href="https://opendata.mtc.ca.gov/datasets/equity-priority-communities-plan-bay-area-2050/explore?location=37.878600%2C-122.370850%2C9", target="_blank")),
                html.Li(html.A("US Census Tract Geospatial Data", href="https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_06_tract_500k.zip", target="_blank")),
                html.Li(html.A("GTFS VTA Schedule Data", href="https://www.arcgis.com/home/item.html?id=47506a089a5146ca91f400ad9ee04ccf", target="_blank")),
            ])
        ]),
    ])

# Top 10 Busiest Stops
unique_locations = map_data.drop_duplicates(subset=['MAIN_CROSS_STREET'])
top_10_unique = unique_locations.nlargest(10, 'AVG_ACTIVITY')
fig_top_stops = px.bar(
    top_10_unique,
    x='AVG_ACTIVITY',
    y='MAIN_CROSS_STREET',
    orientation='h',
    title="Top 10 Busiest Transit Stops",
    labels={'AVG_ACTIVITY': 'Daily Activity', 'MAIN_CROSS_STREET': 'Stop Location'},
    color='AVG_ACTIVITY',
    color_continuous_scale='Blues'
).update_layout(yaxis={'categoryorder':'total ascending'})

# 2. Distribution of Ridership Activity
fig_activity_dist = px.histogram(
    map_data,
    x="AVG_ACTIVITY",
    nbins=50,
    title="Distribution of Daily Stop Activity",
    marginal="rug", # Adds a small distribution rug at the bottom
    color_discrete_sequence=['#007bff']
)

# 3. Demographic Snapshot: Low Income Percentages
fig_income_dist = px.histogram(
    map_data,
    x="Percent Low-Income",
    nbins=30,
    title="Distribution of Neighborhood Low-Income %",
    labels={'Percent Low-Income': 'Percentage of Low-Income Residents'},
    color_discrete_sequence=['#28a745']
)

def render_EDA_page():
    return html.Main(className="container", children=[
        html.H1("Exploratory Data Analysis", className="page-title"),
        html.P("A low-level overview of VTA ridership and Santa Clara County demographics.", className="group-info"),

        html.Section(className="section", children=[
            html.H2("Integrated Route Metrics"),
            html.P("""Below is a preview of our processed dataset. This table combines VTA 
                operational metrics with Census-level demographic indicators."""),

            dash_table.DataTable(
                data=map_data.to_dict('records'),
                columns=[
                    {"name": "Route", "id": "Route"},
                    {"name": "Stop Name", "id": "stop_name"},
                    {"name": "Activity (Board+Alight)", "id": "AVG_ACTIVITY"},
                    {"name": "Low-Income %", "id": "Percent Low-Income"},
                    {"name": "Zero-Vehicle %", "id": "Percent Zero-Vehicle Household"}
                ],
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'left',
                    'padding': '10px',
                    'fontFamily': 'Inter, sans-serif'
                },
                style_header={
                    'backgroundColor': '#f8f9fa',
                    'fontWeight': 'bold',
                    'border': '1px solid #dee2e6'
                },
                page_size=10,
                sort_action="native",
                filter_action="native",
            )
        ]),

        html.Section(className="section", children=[
            html.H2("Data Distribution Summary"),
            html.P("Our dataset consists of 672 unique route-stop observations across Santa Clara County."),

        ]),

        # Section 1: Volume Analysis
        html.Section(className="section", children=[
            html.H2("Transit Volume Overview"),
            html.P("These charts highlight where demand is concentrated across the network."),
            html.Div([dcc.Graph(figure=fig_top_stops)], className="six columns"),
            html.Div([dcc.Graph(figure=fig_activity_dist)], className="six columns"),
        ]),

        # Section 2: Demographic Indicators
        html.Section(className="section", children=[
            html.H2("Community Profiles"),
            html.P("Understanding the demographic makeup of the areas served by these transit stops."),
            dcc.Graph(figure=fig_income_dist)
        ]),

        # Section 3: Data Integrity Notes
        html.Section(className="section", children=[
            html.H2("Data Quality & Scope"),
            html.Ul([
                html.Li("Dataset contains over 3,200 unique stops."),
                html.Li("Ridership activity is averaged across October 2024 service periods."),
                html.Li("Census data is merged via spatial joins with VTA stop coordinates.")
            ])
        ])
    ])

ordered_columns = [
    "Route",
    "Stop_ID_Num",
    "stop_name",
    "AVG_BOARDINGS",
    "Total Population",
    "Equity Priority Community Class",
]
column_defs = [{"name": i, "id": i} for i in ordered_columns]
def render_methods_page():
    return html.Main(className="container", children=[
        html.H1("Analytical Methods", className="page-title"),
        html.P("Technical workflow for data integration, feature engineering, and route clustering.",
               className="group-info"),

        # Section 1: Data Integration & Processing
        html.Section(className="section", children=[
            html.H2("1. Data Pipeline & Spatial Integration"),
            html.P('''To create a route-level perspective of transit equity, we unified three 
                disparate data streams using spatial joins and ID matching:'''),
            html.Ul([
                html.Li([html.Strong("GTFS Mapping: "),
                         "Extracted geographic coordinates and stop sequences from VTA's static feed to establish the network backbone."]),
                html.Li([html.Strong("Ridership Merging: "),
                         "Merged October 2024 Ridership by Stop (RBS) data with the GTFS backbone by cleaning and aligning Stop IDs."]),
                html.Li([html.Strong("Demographic Overlay: "),
                         "Utilized a spatial join to link every transit stop to its corresponding Census Tract, enriching the data with MTC 'Equity Priority Community' indicators."]),
            ])
        ]),

        # Section 2: Feature Engineering (The Underserved Score)
        html.Section(className="section", children=[
            html.H2("2. Engineering the 'Underserved Score'"),
            html.P('''A composite 'Underserved Score' was developed to quantify the gap between 
                neighborhood need and current service levels. Key components include:'''),
            html.Div(className="metrics-grid", children=[
                html.Div(className="metric-box", children=[
                    html.Span("Demand", className="metric-label"),
                    html.P("Normalizaed sum of average daily boardings and alightings (AVG_ACTIVITY).")
                ]),
                html.Div(className="metric-box", children=[
                    html.Span("Need", className="metric-label"),
                    html.P("Normalized product of households that are low-income or have zero vehicles.")
                ]),
            ]),
            html.P(
                "The resulting score: (demand * need) / supply, quantifies routes lack of service level proportional to their need.",
                style={'marginTop': '15px'})
        ]),

        # Section 3: Machine Learning Implementation
        html.Section(className="section", children=[
            html.H2("3. Unsupervised Route Clustering"),
            html.P('''To move beyond simple ranking, we applied K-Means Clustering to group 
                routes into interpretable operational categories:'''),
            html.Ul([
                html.Li([html.Strong("Algorithm: "),
                         "K-Means (k=4) was selected based on the Elbow Method to maximize cluster cohesion."]),
                html.Li([html.Strong("Feature Set: "),
                         "Clustering was performed on normalized metrics of ridership volume, low-income percentage, and the composite need score."]),
                html.Li([html.Strong("Objective: "),
                         "This allows for the identification of 'Route Archetypes' like 'High-Demand Equity Corridors' vs 'Low-Density Neighborhood Feeders'."]),
            ]),

            html.Img(
                src='/assets/K-Mean-Vis.png',
                style={
                    'width': '100%',
                    'max-width': '800px',
                    'border': '1px solid #ddd',
                    'border-radius': '8px'
                }
            ),

            html.Img(
                src='/assets/K-Mean-Clust.png',
                style={
                    'width': '100%',
                    'max-width': '800px',
                    'border': '1px solid #ddd',
                    'border-radius': '8px'
                }
            )
        ]),

        dcc.Graph(figure=fig),

        html.Section(className="section", children=[
            html.H2("4. Classification"),
            html.P('''To validate our 'Underserved Score', we trained a Random Forest Classifier 
                    to predict whether a route would be flagged as high-priority based on hidden patterns 
                    in ridership and demographic features.'''),

            html.Div(className="metrics-grid", children=[
                html.Div(className="metric-box", children=[
                    html.Span("84%", className="metric-value"),
                    html.Span("Model Accuracy", className="metric-label")
                ]),
                html.Div(className="metric-box", children=[
                    html.Span("Features", className="metric-label"),
                    html.P("Boardings, Scheduled Trips, Low-Income %, Zero-Vehicle %")
                ]),
            ]),

            html.Img(
                src='/assets/Feat_Import_RF.png',
                style={
                    'width': '100%',
                    'max-width': '800px',
                    'border': '1px solid #ddd',
                    'border-radius': '8px'
                }
            ),

            html.Img(
                src='/assets/RF_Matrix.png',
                style={
                    'width': '100%',
                    'max-width': '800px',
                    'border': '1px solid #ddd',
                    'border-radius': '8px'
                }
            ),

            html.H3("Why Random Forest?"),
            html.P('''Random Forest was chosen for its ability to handle non-linear relationships 
                    and provide feature importance rankings. This allowed us to confirm that 
                    'Vehicle Ownership' was a more significant predictor of transit dependency 
                    than 'Income Level' alone.'''),
        ]),

        # Section 4: Technical References
        html.Section(className="section", children=[
            html.H2("Technical References"),
            html.P("Methodologies were informed by the following frameworks:"),
            html.Ul([
                html.Li(html.A("MTC Equity Priority Communities Framework",
                               href="https://mtc.ca.gov/planning/equity/equity-priority-communities", target="_blank")),
                html.Li(html.A("Scikit-Learn Clustering Documentation",
                               href="https://scikit-learn.org/stable/modules/clustering.html", target="_blank")),
                html.Li("VTA GTFS & Ridership Reporting Standards")
            ])
        ])
    ])

def render_findings_page():
    top_underserved = map_data.nlargest(5, 'Underserved_Score')[['Route', 'stop_name', 'Underserved_Score']]

    return html.Main(className="container", children=[
        html.H1("Major Findings: Transit Equity Gaps", className="page-title"),
        html.P("Final results from the K-Means clustering and equity gap analysis.", className="group-info"),

        # Section 1: The Primary Insight (The Clusters)
        html.Section(className="section", children=[
            html.H2("Route Segmentation Results"),
            html.P("Our K-Means model identified 3 primary clusters of transit service in Santa Clara County:"),

            html.Div(className="metrics-grid", children=[
                html.Div(className="metric-box", style={'border-top': '5px solid #d9534f'}, children=[
                    html.Span("Cluster 0", className="metric-label"),
                    html.P("High-Need / Low-Demand. Stops that service communities of need, but may not be high traffic areas")
                ]),
                html.Div(className="metric-box", style={'border-top': '5px solid #f0ad4e'}, children=[
                    html.Span("Cluster 1", className="metric-label"),
                    html.P("High-Need, High-Demand. Inner city transit stops which sees the majority of activity")
                ]),
                html.Div(className="metric-box", style={'border-top': '5px solid #5cb85c'}, children=[
                    html.Span("Cluster 2", className="metric-label"),
                    html.P("Low-Need / Low-Demand / High Service. Stops which see the ends of high traffic routes")
                ]),
            ])
        ]),

        html.Section(className="section", children=[
            html.H2("Results Interpretation"),
            html.P("From our results we notice that on most stops, service outpaces demand despite underserved score rising. This conclusion seems counter intuitive, until you account for how VTA has to address demand. From our EDA, we saw that most of the activity on our routes is concentrated on a few major stops, near the center of downtown. In order to keep service in line with the demand of those few major stops, the VTA can choose to run more trips of the same routes through those stops, or run new routes through those stops. Either way, it result in more trips to stops which have more than met their demand. This results in stops with much greater service than would be required in an isolated stop, drowning out our underserved metric."),
        ]),

        # Section 2: Interactive Data Explorer (Satisfies Rubric Item 7)
        html.Section(className="section", children=[
            html.H2("Detailed Route Metrics Explorer"),
            html.P("Search and sort through all analyzed routes to see their specific equity scores:"),

            dash_table.DataTable(
                id='map_data',
                columns=[
                    {"name": "Route", "id": "Route"},
                    {"name": "Location", "id": "stop_name"},
                    {"name": "Underserved Score", "id": "Underserved_Score"},
                    {"name": "Cluster Category", "id": "Cluster_Label"},
                    {"name": "Low-Income %", "id": "Percent Low-Income"}
                ],
                data=map_data.to_dict('records'),
                sort_action="native",
                filter_action="native",
                page_size=10,
                style_cell={'textAlign': 'left', 'padding': '10px'},
                style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{Underserved_Score} > 0.6'},
                        'backgroundColor': '#fff2f2',
                        'color': '#d9534f'
                    }
                ]
            )
        ]),

        # Section 3: Interpretation & Policy Recommendation
        html.Section(className="section", children=[
            html.H2("Analysis Conclusions"),
            html.P("Decouple Hub Reliance: To address the surplus in overserved areas, VTA could explore 'Short-Turning' (running extra trips only on the busiest segments of a route) to prevent unnecessary service in low-demand zones."),
            html.P("Targeted Equity Investment: Resources should be shifted from the 'Service Surplus' (Cluster 2) toward 'High Need Outliers' (Cluster 3), where ridership may be lower but transit dependency (low vehicle ownership) is highest."),
        ]),

        # Live inference from Cloud Run (same API as /predict on the inference service)
        html.Section(className="section", children=[
            html.H2("Live route predictions"),
            html.P(
                "The Random Forest and Prophet models deployed on Cloud Run can be queried "
                "for any route ID in the inference dataset. Select a route from the map data "
                "and run a prediction to see the model output.",
                style={"maxWidth": "720px"},
            ),
            dcc.Dropdown(
                id="findings-route-dropdown",
                options=[
                    {"label": f"Route {r}", "value": r}
                    for r in sorted(
                        map_data["Route"].dropna().astype(str).str.strip().unique()
                    )
                ],
                placeholder="Select a route ID…",
                clearable=True,
                style={"maxWidth": "420px", "marginBottom": "12px"},
            ),
            html.Button(
                "Run prediction",
                id="findings-predict-btn",
                n_clicks=0,
                style={
                    "padding": "8px 18px",
                    "cursor": "pointer",
                    "backgroundColor": "#007bff",
                    "color": "white",
                    "border": "none",
                    "borderRadius": "6px",
                    "fontWeight": "600",
                },
            ),
            html.Div(id="findings-inference-output", style={"marginTop": "20px"}),
        ]),

    ])

if __name__ == '__main__':
    app.run(debug=True)