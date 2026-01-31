from typing import Optional, Literal
from pydantic import BaseModel, Field


class Resolution(BaseModel):
    result: str = Field(
        description="Empty string if the candidate record is unique, otherwise the matching record from the list."
    )


class Vehicle(BaseModel):
    brand: Optional[str] = Field(
        default=None, description="Vehicle manufacturer or marque"
    )

    model: Optional[str] = Field(
        default=None,
        description="Base vehicle model name, excluding submodel or trim details",
    )

    engine_size: Optional[float] = Field(
        default=None,
        description="Physical size or capacity of the engine, often measured in liters or cubic centimeters (cc)",
    )

    engine_power: Optional[int] = Field(
        default=None,
        description="Power output of the engine, often measured in horsepower (hp) or kilowatts (kW)",
    )

    engine_aspiration: Literal["natural", "turbo", "supercharger", None] = Field(
        default=None,
        description="Method of air intake (natural = naturally aspirated, turbo = turbocharged, supercharger = supercharged)",
    )

    injection_type: Literal["direct", "indirect", "dual", "other", None] = Field(
        default=None,
        description="Fuel injection method (direct = into cylinder, indirect = into intake, dual = both direct and port)",
    )

    submodel: Optional[str] = Field(
        default=None, description="Specific variant of the vehicle model"
    )

    trim_level: Optional[str] = Field(
        default=None,
        description="Package of features, styling, and equipment for the vehicle. Defines the specific configuration or edition",
    )

    cylinders: Optional[int] = Field(
        default=None, description="Number of cylinders in the engine"
    )

    body_type: Literal[
        "sedan",
        "hatchback",
        "suv",
        "crossover",
        "coupe",
        "convertible",
        "wagon",
        "pickup",
        "van",
        "minivan",
        "roadster",
        "other",
        None,
    ] = Field(default=None, description="Physical structure and shape of the vehicle")

    transmission_type: Literal["manual", "automatic", "semi-automatic", None] = Field(
        default=None, description="Type of transmission system in the vehicle"
    )

    transmission_technology: Literal["tc", "cvt", "sct", "dct", "other", None] = Field(
        default=None, description="Specific transmission mechanism"
    )

    transmission_gears: Optional[int] = Field(
        default=None,
        description="Number of gears or speeds in the transmission (e.g., 8 for an 8-speed transmission)",
    )
    energy_source: Literal["fossil", "hybrid", "electric", "other", None] = Field(
        description="Main energy source powering the vehicle"
    )

    fuel_type: Literal[
        "petrol",
        "diesel",
        "cng",
        "lpg",
        "lng",
        "methanol",
        "propane",
        "hydrogen",
        "other",
        None,
    ] = Field(default=None, description="Specific fuel used by the vehicle")

    propulsion_system: Literal[
        "icev",
        "ev",
        "pev",
        "bev",
        "hev",
        "mhv",
        "phev",
        "erev",
        "fcev",
        "pfcev",
        "other",
        None,
    ] = Field(default=None, description="Configuration of the propulsion system")

    drive_type: Literal["fwd", "rwd", "awd", "4wd", "other", None] = Field(
        default=None, description="Wheels receiving power from the drivetrain"
    )