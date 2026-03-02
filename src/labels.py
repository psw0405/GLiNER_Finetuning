# Exact 26 labels used for finetuning (spelling preserved as specified)
LABELS: list[str] = [
    "Person",
    "Location",
    "Organization",
    "Date",
    "Time",
    "Animal",
    "Quantity",
    "Event",
    "LocationCountry",
    "LocationCity",
    "Shop",
    "CultureSite",
    "Building",
    "DateDuraion",        # intentional spelling (non-time durations: days/weeks/months/years)
    "TimeDuration",
    "Sports",
    "Food",
    "CivilizationCurrency",
    "CivilizationLaw",
    "QuantityAge",
    "QunatityTemperature", # intentional spelling (temperature values)
    "QuantityPrice",
    "EventSports",
    "EventFestival",
    "TermMedical",
    "TermSports",
]

LABEL_SET: set[str] = set(LABELS)
