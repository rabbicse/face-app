from fastapi import Form
from pydantic import BaseModel


class Person(BaseModel):
    """
    ref: https://stackoverflow.com/questions/69292855/why-do-i-get-an-unprocessable-entity-error-while-uploading-an-image-with-fasta
    """
    person_id: int
    name: str
    email: str | None
    phone: str | None
    age: int | None
    description: str | None

    @classmethod
    def as_form(cls,
                person_id: int = Form(...),
                name: str = Form(...),
                email: str | None = Form(...),
                phone: str | None = Form(...),
                age: int | None = Form(...),
                description: str | None = Form(...),
                ) -> 'Person':
        return cls(person_id=person_id, name=name, email=email, phone=phone, age=age, description=description)
