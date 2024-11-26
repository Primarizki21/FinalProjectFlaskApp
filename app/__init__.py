from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Import routes and register them
    from .routes import app_routes
    app.register_blueprint(app_routes)

    return app
