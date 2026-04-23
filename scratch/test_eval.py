import requests
import json


def test_evaluate():
    url = "http://localhost:8000/evaluate"
    params = {"split": "valid"}
    try:
        response = requests.post(url, params=params)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Connection failed: {e}")


if __name__ == "__main__":
    test_evaluate()
