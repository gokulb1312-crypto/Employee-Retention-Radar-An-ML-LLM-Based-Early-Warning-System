import logging
import re
import time
from collections import deque
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

LOG_PATH = Path(__file__).resolve().parent / "logs" / "retention_radar.log"
LOG_PATH.parent.mkdir(exist_ok=True)
logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("retention_radar")

class DashboardRequest(BaseModel):
    employee_id: int
    probability: float = Field(ge=0, le=1)
    age: int = Field(ge=18, le=100)
    department: str = Field(min_length=1, max_length=80)
    monthly_income: float = Field(gt=0, le=10_000_000)
    years_at_company: float = Field(ge=0, le=70)
    job_satisfaction: float = Field(ge=1, le=4)
    work_life_balance: float = Field(ge=1, le=4)
    overtime: str

    @field_validator("department")
    @classmethod
    def safe_department(cls, value: str) -> str:
        cleaned = re.sub(r"[\r\n<>]", " ", value).strip()
        if not cleaned:
            raise ValueError("Department is invalid.")
        return cleaned

    @field_validator("overtime")
    @classmethod
    def valid_overtime(cls, value: str) -> str:
        if value not in {"Yes", "No"}:
            raise ValueError("Overtime must be Yes or No.")
        return value

class InMemoryRateLimiter:
    """Demo limit; use Redis/shared storage in a multi-user deployment."""
    def __init__(self, limit: int = 5, window_seconds: int = 60):
        self.limit = limit
        self.window_seconds = window_seconds
        self.calls: deque[float] = deque()
    def allow(self) -> bool:
        now = time.monotonic()
        while self.calls and now - self.calls[0] > self.window_seconds:
            self.calls.popleft()
        if len(self.calls) >= self.limit:
            return False
        self.calls.append(now)
        return True

def audit(event: str, employee_id: int, detail: str = "") -> None:
    """Logs metadata only—never API keys, raw prompts, or complete employee records."""
    LOGGER.info("event=%s employee_id=%s %s", event, employee_id, detail[:160])

