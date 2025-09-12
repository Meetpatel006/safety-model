"""Modal Deployment for Weather Safety Prediction API (Modal v1.1)

This file has been migrated to Modal v1.1 APIs:
- use Image.add_local_dir / add_local_file / add_local_python_source instead of Mount
- set python_version on Image.debian_slim
- keep an ASGI wrapper function to expose the FastAPI app
"""

import modal
from pathlib import Path

# Name of the Modal app
app = modal.App("weather-safety-api")

# Build the container image (explicit python version)
image = (
    modal.Image.debian_slim(python_version="3.11").pip_install(
        [
            "fastapi==0.110.0",
            "uvicorn[standard]==0.29.0",
            "pydantic==1.10.14",
                "joblib",
                "pandas",
                "numpy",
                "requests",
                # Model runtime dependencies
                "xgboost>=1.7.0",
                "scikit-learn>=1.2",
        ]
    )
)

# Add local project files into the image. Replace the old Mount-based approach
# by explicitly adding the project directory and any model files we need.
project_root = Path(__file__).resolve().parent
image = image.add_local_dir(str(project_root), remote_path="/root/app")

# Explicitly include the trained model file to ensure it's available inside
# the Modal image regardless of how add_local_dir behaves for binary files.
model_file = project_root / "weather_safety_model_kaggle.pkl"
if model_file.exists():
    image = image.add_local_file(str(model_file), remote_path="/root/app/weather_safety_model_kaggle.pkl")
else:
    # Leave this as informative; the runtime will still log if the file is missing.
    pass

# If you prefer adding only python source directories, you can use:
# image = image.add_local_python_source("..")  # relative to project layout

# Expose FastAPI app via Modal
@app.function(image=image, min_containers=1, timeout=600)
@modal.asgi_app()
def fastapi_app():
    # Import the FastAPI app from the local api.py inside the image filesystem.
    # Because we added the project directory to the Image at /root/app, we
    # need to ensure Python can import it. We'll adjust sys.path at runtime.
    import sys
    from pathlib import Path as _P

    app_dir = _P("/root/app")
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))

    from api import app as fastapi_app  # type: ignore
    return fastapi_app
