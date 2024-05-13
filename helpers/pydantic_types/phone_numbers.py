from pydantic_extra_types.phone_numbers import PhoneNumber as PydanticPhoneNumber


class PhoneNumber(PydanticPhoneNumber):
    phone_format = "E164"  # E164 is standard accross all Microsoft services
