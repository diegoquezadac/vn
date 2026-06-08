from typing import Optional, Literal, List
from pydantic import BaseModel, Field

EquipmentItem = Literal[
    # ── safety ────────────────────────────────────────────────────────────────
    "abs",                      # Anti-lock Braking System (ABS)
    "airbags",                  # any airbags (driver, passenger, side, curtain)
    "esp",                      # Electronic Stability Program / ESC / stability control
    "traction_control",         # Traction Control System (TCS) — distinct from full ESP
    "rear_camera",              # Rear / backup / reversing camera
    "parking_sensors",          # PDC / ultrasonic parking sensors (front or rear)
    "blind_spot_monitoring",    # Blind Spot Detection / Side Assist
    "lane_departure_warning",   # Lane Departure Warning / Lane Keep Assist
    "forward_collision_warning",# Forward Collision Warning / Autonomous Emergency Braking
    "adaptive_cruise_control",  # ACC / radar cruise control
    # ── comfort ───────────────────────────────────────────────────────────────
    "air_conditioning",         # AC / Climatisation / Klima (single-zone)
    "climate_control",          # Dual/automatic climate control (multi-zone)
    "heated_seats",             # Heated / seat warmer
    "ventilated_seats",         # Ventilated / cooled seats
    "leather_seats",            # Leather / leatherette / Leder / Cuir
    "electric_seats",           # Power / electric seat adjustment
    "memory_seats",             # Memory seat positions
    "sunroof",                  # Sunroof / moonroof / toit ouvrant
    "panoramic_roof",           # Panoramic / glass roof
    "heated_steering_wheel",    # Heated steering wheel
    # ── technology ────────────────────────────────────────────────────────────
    "navigation",               # Built-in GPS / navigation system / sat-nav
    "bluetooth",                # Bluetooth audio or hands-free (CD/MP3/Bluetooth counts)
    "dab_radio",                # DAB / digital radio (AutoTrader UK, AutoScout24 filter)
    "apple_carplay",            # Apple CarPlay
    "android_auto",             # Android Auto
    "premium_audio",            # Bose / Harman Kardon / Bang & Olufsen / upgraded speakers
    "head_up_display",          # HUD / head-up display
    "wireless_charging",        # Qi wireless phone charging
    "onboard_computer",         # Trip computer / onboard computer / MFD
    # ── convenience ───────────────────────────────────────────────────────────
    "cruise_control",           # Standard (non-adaptive) cruise control / régulateur
    "speed_limiter",            # Speed limiter / limiteur de vitesse
    "electric_windows",         # Power windows / electric windows
    "electric_mirrors",         # Power / electric wing mirrors
    "central_locking",          # Central locking / power locks / verrouillage centralisé
    "keyless_entry",            # Keyless entry / smart key / proximity key
    "push_start",               # Push-button / keyless start
    "remote_start",             # Remote engine start (CarGurus, AutoTrader US filter)
    "parking_assist",           # Automatic / assisted parking (not just sensors)
    "electric_tailgate",        # Power liftgate / electric boot / coffre électrique
    "tow_bar",                  # Tow bar / trailer hitch / attelage
    "power_steering",           # Power steering (relevant on older vehicles)
    "paddle_shifters",          # Steering-wheel paddle shifters
    # ── exterior ──────────────────────────────────────────────────────────────
    "alloy_wheels",             # Alloy / aluminium wheels (Alu, jantes alliage)
    "tinted_windows",           # Tinted / privacy windows
]


class Resolution(BaseModel):
    reasoning: str = Field(
        description="Brief explanation of whether both strings refer to the same real-world entity."
    )
    is_match: bool = Field(
        description="True if both strings refer to the same real-world entity, False otherwise."
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
        description="Numeric engine displacement value, in the unit given by engine_size_unit",
    )

    engine_size_unit: Literal["l", "cc", None] = Field(
        default=None,
        description="Unit for engine_size: liters (l) or cubic centimeters (cc)",
    )

    engine_power: Optional[int] = Field(
        default=None,
        description="Numeric power output value, in the unit given by engine_power_unit",
    )

    engine_power_unit: Literal["kw", "ps", "hp", None] = Field(
        default=None,
        description="Unit for engine_power: kilowatts (kw), metric horsepower (ps), or imperial horsepower (hp)",
    )

    engine_aspiration: Literal["natural", "turbo", "supercharger", None] = Field(
        default=None,
        description="Method of air intake (natural = naturally aspirated, turbo = turbocharged, supercharger = supercharged)",
    )

    injection_type: Literal["direct", "indirect", "dual", "other", None] = Field(
        default=None,
        description="Fuel injection method (direct = into cylinder, indirect = into intake, dual = both direct and port)",
    )

    year: Optional[int] = Field(
        default=None, description="Model year of the vehicle"
    )

    mileage: Optional[int] = Field(
        default=None, description="Numeric odometer reading, in the unit given by mileage_unit"
    )

    mileage_unit: Literal["km", "mi", None] = Field(
        default=None, description="Unit for mileage: kilometers (km) or miles (mi)"
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

    condition: Literal[
        "new", "excellent", "very good", "good", "fair", "poor", "salvage", None
    ] = Field(
        default=None,
        description="Vehicle condition as reported by the seller (new=unused, excellent=near-perfect, very good=minor wear, good=normal used, fair=noticeable wear, poor=significant issues, salvage=written-off or insurance total loss)",
    )

    color: Literal[
        "black", "white", "silver", "grey", "red", "blue", "green",
        "yellow", "orange", "brown", "beige", "gold", "purple", "other", None,
    ] = Field(
        default=None,
        description="Exterior paint or body color of the vehicle. Map manufacturer color names to the closest standard value (e.g. 'Midnight Black' → 'black', 'Pearl White' → 'white'). Use 'other' for colors that do not fit any standard value.",
    )

    doors: Optional[int] = Field(
        default=None,
        description="Number of doors on the vehicle body (typically 2, 3, 4, or 5)",
    )

    fuel_consumption: Optional[float] = Field(
        default=None,
        description="Numeric fuel consumption value, in the unit given by fuel_consumption_unit",
    )

    fuel_consumption_unit: Literal["l/100km", "mpg", "km/l", None] = Field(
        default=None,
        description="Unit for fuel_consumption: liters per 100 km (l/100km), miles per gallon (mpg), or kilometers per liter (km/l)",
    )

    equipment: Optional[List[EquipmentItem]] = Field(
        default=None,
        description=(
            "Standardized list of features and equipment explicitly mentioned in the listing. "
            "Map synonyms and translations to the closest standard term — examples: "
            "'backup camera' → 'rear_camera', 'stability control'/'ESC' → 'esp', "
            "'moonroof' → 'sunroof', 'Klima' → 'air_conditioning', 'Leder' → 'leather_seats', "
            "'Alufelgen' → 'alloy_wheels', 'CD/MP3/Bluetooth' → 'bluetooth', "
            "'Navigation System/GPS' → 'navigation', 'Central Locking' → 'central_locking', "
            "'Speed Limiter' → 'speed_limiter', 'Onboard Computer' → 'onboard_computer', "
            "'Power Windows' → 'electric_windows', 'Power Locks' → 'central_locking', "
            "'DAB' / 'digital radio' → 'dab_radio', 'remote start' / 'remote engine start' → 'remote_start'. "
            "Only include features explicitly stated — do not infer from trim level or model."
        ),
    )