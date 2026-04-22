import pytest
from fastapi.testclient import TestClient
from src.api.main import app
import io
from PIL import Image

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "version" in data
    assert "gpu" in data
    assert "available" in data["gpu"]


def test_predict_no_file():
    response = client.post("/predict")
    assert response.status_code == 422  # Validation error for missing file


def test_predict_invalid_file():
    response = client.post(
        "/predict", files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400
    assert "File must be an image" in response.json()["detail"]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_predict_success():
    # Create a dummy image
    img = Image.new("RGB", (256, 256), color="red")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr = img_byte_arr.getvalue()

    # We need to ensure the model is loaded or mock it.
    # Since startup_event loads the model, and TestClient handles lifespan in modern FastAPI,
    # but I used @app.on_event("startup"), it should work if we use the client.

    with client:  # Triggers startup events
        response = client.post(
            "/predict", files={"file": ("test.jpg", img_byte_arr, "image/jpeg")}
        )

    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 14
    assert "joint" in data["predictions"][0]
    assert "x" in data["predictions"][0]
    assert "y" in data["predictions"][0]
