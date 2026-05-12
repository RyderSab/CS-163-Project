import pandas as pd
from google.cloud import firestore

db = firestore.Client(project='cs-163-project-489017', database='map-data')

df = pd.read_csv('data/route_metrics_map_data.csv')

data_records = df.to_dict(orient='records')

for record in data_records:
    db.collection('route_metrics').add(record)