from fastapi.testclient import TestClient


from app import app

client = TestClient(app)

def test_chat():
    response = client.post("/chat", json={"question": "Who is Alan Turing?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    print(data["answer"])