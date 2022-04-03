from datetime import date

SENTINEL_2_BANDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B10",
    "B11",
    "B12",
]
SENTINEL_2_START_DATE = date(2017, 3, 28)
SENTINEL_2_ORBITAL_PERIOD_DAYS = 10

SENTINEL_1_BANDS = [
    'VV',
    'VH'
]
SENTINEL_1_START_DATE = date(2014, 10, 3)
SENTINEL_1_ORBITAL_PERIOD_DAYS = 12