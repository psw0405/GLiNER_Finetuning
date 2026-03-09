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
    "Duraion",        # intentional spelling (non-time durations: days/weeks/months/years)
    "TimeDuration",
    "Sports",
    "Food",
    "Currency",
    "Law",
    "QuantityAge",
    "QunatityTemperature", # intentional spelling (temperature values)
    "QuantityPrice",
    "EventSports",
    "EventFestival",
    "TermMedical",
    "TermSports",
]

LABEL_SET: set[str] = set(LABELS)


LABEL_ALIASES: dict[str, str] = {
    "DateDuraion": "Duraion",
    "DateDuration": "Duraion",
    "QuantityMoney": "QuantityPrice",
    "QuantityTemperature": "QunatityTemperature",
    "TermDisease": "TermMedical",
    "CulturalAsset": "CultureSite",
    "CivilizationCurrency": "Currency",
    "CivilizationLaw": "Law",
}


def normalize_label(label: str) -> str:
    """Map legacy/alias labels to canonical labels used by this project."""
    return LABEL_ALIASES.get(label, label)
