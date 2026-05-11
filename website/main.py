import dash
from dash import html, dcc
import pandas as pd
import plotly.express as px
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, '..', 'data', 'route_metrics_map_data.csv')

map_data = pd.read_csv(data_path)

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

app.layout = html.Div([
    html.H1("Santa Clara Transit Analysis"),
    html.Div("Identifying Underserved Routes"),

dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run(debug=True)