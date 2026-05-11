import dash
from google.cloud import firestore
from dash import html, dcc, Input, Output, dash_table
import pandas as pd
import plotly.express as px
import os

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



app = dash.Dash(__name__)

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

        ]),

        dcc.Graph(figure=fig),

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
    return html.Div([
        html.H1("Major Findings"),
        dcc.Markdown('''
        ### Nothing Yet!    
        ''')
    ])


if __name__ == '__main__':
    app.run(debug=True)