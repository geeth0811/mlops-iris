from fastapi.testclient import TestClient
from main import app
import datetime

# test to check the correct functioning of the /ping route
def test_ping():
    with TestClient(app) as client:
        response = client.get("/ping")
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"ping": "pong", "run_Time_timestamp":str(datetime.datetime.now().replace(microsecond=0))}


# test to check if Iris Virginica is classified correctly
def test_pred_virginica():
    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 3,
        "sepal_width": 5,
        "petal_length": 3.2,
        "petal_width": 4.4,
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"flower_class": "Iris Virginica", "run_Time_timestamp":str(datetime.datetime.now().replace(microsecond=0))}


#Task 2: Add 2 more unit tests of your choice to test_app.py and make sure they are passing.
# TC1: Here we checking if Iris Virginica for another data set class which is classified/predicted correctly
def test_pred_virginica_new_dataset():
    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 3,
        "sepal_width": 4,
        "petal_length": 5,
        "petal_width": 5,
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"flower_class": "Iris Virginica", "run_Time_timestamp":str(datetime.datetime.now().replace(microsecond=0))}

# TC2: Here we are checking correct functioning of feedback loop by providing a valid payload.
def test_feedback_loop():
    #defining a sample payload for the testcase
    payload = [{
        "sepal_length": 3,
        "sepal_width": 5,
        "petal_length": 3.2,
        "petal_width": 4.4,
        "flower_class": "Iris Virginica"
    }]
    with TestClient(app) as client:
        response = client.post("/feedback_loop", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"detail": "Feedback loop successful", "run_Time_timestamp":str(datetime.datetime.now().replace(microsecond=0))}