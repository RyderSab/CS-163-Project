FROM python:3.9-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
# scikit-learn is required to unpickle rf_model.joblib (RandomForestClassifier).
RUN pip install --no-cache-dir flask pandas joblib prophet gunicorn scikit-learn

# Run the web service on container startup.
# Use gunicorn as the production-grade web server.
# Cloud Run expects the container to listen on the port defined by the PORT env var (default 8080).
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 api:app
