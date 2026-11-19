# Employee REST API

A simple Employee REST API built with **Python + FastAPI**. Data is kept in an in-memory list (no database), seeded with the three sample employees from the assessment.

## Requirements
- Python 3.9+

## Configure and run

```bash
# 1. (optional) create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. install dependencies
pip install -r requirements.txt

# 3. start the server
uvicorn main:app --reload
```

The API runs at **http://127.0.0.1:8000**.
Interactive Swagger docs (try every endpoint in the browser): **http://127.0.0.1:8000/docs**

## Run the tests

```bash
pytest -v
```

## Endpoints

| Question | Method | Path | Description |
|---|---|---|---|
| 1.1 | GET | `/employees` | All employees (optional `?department=` filter, 1.3) |
| 1.1 | GET | `/employees/{id}` | Employee by ID, `404` if not found |
| 1.2 | POST | `/employees` | Create employee (`name`, `department`); ID generated, status = `Active` |
| 1.3 | PATCH | `/employees/{id}/status` | Set status to `Active` or `Inactive` |

### Status codes
| Code | When |
|---|---|
| 200 | Successful GET / PATCH |
| 201 | Employee created |
| 404 | Employee ID does not exist |
| 422 | Validation error: empty/missing name or department, or status not `Active`/`Inactive` |

## Example requests and responses

**GET /employees**
```json
[
  {"id": 1, "name": "Rahul Sharma", "department": "Engineering", "status": "Active"},
  {"id": 2, "name": "Priya Patel",  "department": "HR",          "status": "Active"},
  {"id": 3, "name": "Amit Verma",   "department": "Engineering", "status": "Inactive"}
]
```

**GET /employees?department=Engineering** (only Engineering employees)
```json
[
  {"id": 1, "name": "Rahul Sharma", "department": "Engineering", "status": "Active"},
  {"id": 3, "name": "Amit Verma",   "department": "Engineering", "status": "Inactive"}
]
```

**GET /employees/999** → `404`
```json
{"detail": "Employee with id 999 not found"}
```

**POST /employees**
```json
{"name": "Neha Singh", "department": "Finance"}
```
→ `201`
```json
{"id": 4, "name": "Neha Singh", "department": "Finance", "status": "Active"}
```

**POST /employees** with empty name → `422`
```json
{"detail": [{"type": "value_error", "loc": ["body", "name"], "msg": "Value error, must not be empty"}]}
```

**PATCH /employees/1/status**
```json
{"status": "Inactive"}
```
→ `200`
```json
{"id": 1, "name": "Rahul Sharma", "department": "Engineering", "status": "Inactive"}
```

**PATCH /employees/1/status** with `{"status": "Retired"}` → `422`

**PATCH /employees/999/status** → `404`

## Postman
Import `postman_collection.json` into Postman. It uses a `baseUrl` variable (`http://127.0.0.1:8000`) and has a request for every endpoint, including the error cases.

## Notes
- The department filter is case-insensitive (`engineering` works too).
- Data resets when the server restarts (in-memory storage, as allowed by the assessment).

## Incomplete features
None. All requirements from 1.1, 1.2 and 1.3 are implemented.
