import dash
from dash import html, dcc

# Initialize the Dash app
app = dash.Dash(__name__)

# EXPOSE THE SERVER: This is critical for App Engine!
server = app.server

app.layout = html.Div([
    html.H1("Santa Clara Transit Analysis"),
    html.Div("Identifying Underserved Routes"),
])

if __name__ == '__main__':
    app.run(debug=True)