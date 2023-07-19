import json

from pydantic import BaseModel


class ExceptionData(BaseModel):
    code: int
    msg: str


class TaskException(Exception):
    def __init__(self, data: ExceptionData) -> None:
        self.data = data
        super().__init__(self.data.dict())

    def __repr__(self) -> str:
        return json.dumps(self.data.dict())
