"""Mule Network Explorer — Databricks App entry point."""
import dash
from fastapi import FastAPI
from starlette.middleware.wsgi import WSGIMiddleware

from layouts.graph_explorer import create_layout
from callbacks.graph_callbacks import register_callbacks

# Create Dash app
dash_app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    title="Mule Network Explorer",
    update_title=None,
)

dash_app.layout = create_layout()
register_callbacks(dash_app)

# Wrap Dash with FastAPI for Databricks Apps (expects ASGI)
fastapi_app = FastAPI()


@fastapi_app.get("/health")
def health():
    return {"status": "ok"}


# Mount Dash under FastAPI
fastapi_app.mount("/", WSGIMiddleware(dash_app.server))

# Export for uvicorn
server = fastapi_app
