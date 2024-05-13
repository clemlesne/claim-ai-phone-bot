from helpers.pydantic_types.phone_numbers import PhoneNumber
from pydantic import SecretStr, BaseModel


class CommunicationServicesModel(BaseModel):
    access_key: SecretStr
    endpoint: str
    phone_number: PhoneNumber
