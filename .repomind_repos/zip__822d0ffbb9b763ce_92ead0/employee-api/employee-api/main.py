from enum import Enum
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, field_validator

app = FastAPI(
    title="Employee REST API",
    version="1.0.0",
)

class Status(str, Enum):
    ACTIVE = "Active"
    INACTIVE = "Inactive"


class Employee(BaseModel):
    id: int
    name: str
    department: str
    status: Status


# Request body for POST /employees (1.2)
class EmployeeCreate(BaseModel):
    name: str
    department: str

    @field_validator("name", "department")
    @classmethod
    def must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


# Request body for PATCH /employees/{id}/status (1.3)
class StatusUpdate(BaseModel):
    status: Status


employees: List[Employee] = [
    Employee(id=1, name="Rahul Sharma", department="Engineering", status=Status.ACTIVE),
    Employee(id=2, name="Priya Patel", department="HR", status=Status.ACTIVE),
    Employee(id=3, name="Amit Verma", department="Engineering", status=Status.INACTIVE),
]


def find_employee(employee_id: int) -> Employee:
    for emp in employees:
        if emp.id == employee_id:
            return emp
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Employee with id {employee_id} not found",
    )


def next_id() -> int:
    return max((e.id for e in employees), default=0) + 1

@app.get("/employees", response_model=List[Employee])
def get_employees(
    department: Optional[str] = Query(None, description="Filter by department, e.g. Engineering"),
):
    """1.1 return all employees; 1.3 optional ?department= filter (case-insensitive)."""
    if department is None:
        return employees
    wanted = department.strip().lower()
    return [e for e in employees if e.department.lower() == wanted]


@app.get("/employees/{employee_id}", response_model=Employee)
def get_employee(employee_id: int):
    """1.1 return a single employee, 404 if not found."""
    return find_employee(employee_id)


@app.post("/employees", response_model=Employee, status_code=status.HTTP_201_CREATED)
def create_employee(payload: EmployeeCreate):
    """1.2 create an employee. ID is generated, status defaults to Active."""
    employee = Employee(
        id=next_id(),
        name=payload.name,
        department=payload.department,
        status=Status.ACTIVE,
    )
    employees.append(employee)
    return employee


@app.patch("/employees/{employee_id}/status", response_model=Employee)
def update_employee_status(employee_id: int, payload: StatusUpdate):
    """1.3 update status to Active or Inactive.
    Unknown id -> 404. Invalid status value -> 422.
    """
    employee = find_employee(employee_id)
    employee.status = payload.status
    return employee
