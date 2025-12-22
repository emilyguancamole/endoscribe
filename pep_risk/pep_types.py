from enum import Enum
from typing import Dict


class PEPTherapy(str, Enum):
    NO_TREATMENT = "no_treatment"
    HYDRATION = "hydration"
    INDOMETHACIN = "indomethacin"
    PD_STENT = "pd_stent"
    HYDRATION_INDOMETHACIN = "hydration_indomethacin"
    INDOMETHACIN_PD = "indomethacin_pd"

THERAPY_LABELS: Dict[PEPTherapy, str] = {
    PEPTherapy.NO_TREATMENT: "No treatment",
    PEPTherapy.HYDRATION: "Aggressive hydration only",
    PEPTherapy.INDOMETHACIN: "Indomethacin only",
    PEPTherapy.PD_STENT: "PD stent only",
    PEPTherapy.HYDRATION_INDOMETHACIN: "Aggressive hydration and indomethacin",
    PEPTherapy.INDOMETHACIN_PD: "Indomethacin and PD stent",
}

_LABEL_TO_ID: Dict[str, PEPTherapy] = {v.lower(): k for k, v in THERAPY_LABELS.items()}

def therapy_id_from_label(label: str) -> PEPTherapy:
    if not label:
        return PEPTherapy.NO_TREATMENT
    return _LABEL_TO_ID.get(label.strip().lower(), PEPTherapy.NO_TREATMENT)


def therapy_label_from_id(tp: PEPTherapy) -> str:
    return THERAPY_LABELS.get(tp, "No treatment")
