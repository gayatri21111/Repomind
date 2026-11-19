from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


# ---- 1.1 ------------------------------------------------------------------
def test_get_all_employees():
    r = client.get("/employees")
    assert r.status_code == 200
    assert len(r.json()) >= 3


def test_get_employee_by_id():
    r = client.get("/employees/1")
    assert r.status_code == 200
    assert r.json() == {
        "id": 1, "name": "Rahul Sharma", "department": "Engineering", "status": "Active",
    }


def test_get_employee_not_found():
    r = client.get("/employees/999")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"]


# ---- 1.2 ------------------------------------------------------------------
def test_create_employee_defaults_to_active():
    r = client.post("/employees", json={"name": "Neha Singh", "department": "Finance"})
    assert r.status_code == 201
    body = r.json()
    assert body["name"] == "Neha Singh"
    assert body["department"] == "Finance"
    assert body["status"] == "Active"
    assert isinstance(body["id"], int)
    # it can be fetched afterwards
    assert client.get(f"/employees/{body['id']}").status_code == 200


def test_create_employee_empty_name_rejected():
    r = client.post("/employees", json={"name": "   ", "department": "HR"})
    assert r.status_code == 422


def test_create_employee_empty_department_rejected():
    r = client.post("/employees", json={"name": "Someone", "department": ""})
    assert r.status_code == 422


def test_create_employee_missing_field_rejected():
    r = client.post("/employees", json={"name": "Someone"})
    assert r.status_code == 422


# ---- 1.3 ------------------------------------------------------------------
def test_update_status():
    r = client.patch("/employees/2/status", json={"status": "Inactive"})
    assert r.status_code == 200
    assert r.json()["status"] == "Inactive"
    r = client.patch("/employees/2/status", json={"status": "Active"})
    assert r.json()["status"] == "Active"


def test_update_status_invalid_value():
    r = client.patch("/employees/1/status", json={"status": "Retired"})
    assert r.status_code == 422


def test_update_status_unknown_employee():
    r = client.patch("/employees/999/status", json={"status": "Active"})
    assert r.status_code == 404


def test_filter_by_department():
    r = client.get("/employees", params={"department": "Engineering"})
    assert r.status_code == 200
    data = r.json()
    assert len(data) >= 2
    assert all(e["department"] == "Engineering" for e in data)


def test_filter_unknown_department_returns_empty_list():
    r = client.get("/employees", params={"department": "Nope"})
    assert r.status_code == 200
    assert r.json() == []
