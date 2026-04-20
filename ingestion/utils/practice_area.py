"""Practice area keyword classifier — shared between Stage 1 (hf_streaming) and Stage 2."""
from __future__ import annotations

PRACTICE_AREA_KEYWORDS: dict[str, list[str]] = {
    "employment": [
        "wrongful termination", "wrongful discharge", "retaliation",
        "discrimination", "hostile work environment", "wage", "overtime", "flsa",
        "fmla", "ada", "title vii", "eeoc", "labor code", "whistleblower", "nlra",
        "minimum wage", "unpaid wages", "at-will", "constructive discharge",
        "collective bargaining", "workers compensation",
    ],
    "tort": [
        "negligence", "personal injury", "premises liability", "slip and fall",
        "product liability", "defective product", "duty of care", "breach of duty",
        "proximate cause", "tort", "intentional infliction", "emotional distress",
        "battery", "assault", "trespass", "nuisance", "strict liability",
        "respondeat superior", "vicarious liability", "medical malpractice",
        "wrongful death",
    ],
    "family": [
        "divorce", "dissolution of marriage", "child custody", "child support",
        "spousal support", "alimony", "property division", "marital property",
        "domestic violence", "restraining order", "adoption", "guardianship",
        "paternity", "visitation", "parental rights", "family law",
        "community property",
    ],
    "criminal": [
        "criminal", "prosecution", "indictment", "felony", "misdemeanor",
        "fourth amendment", "search and seizure", "probable cause", "miranda",
        "due process", "excessive force", "police misconduct", "§1983",
        "civil rights violation", "false arrest", "unlawful detention",
        "habeas corpus", "sentencing", "plea bargain",
    ],
}


def classify_practice_area(text: str) -> list[str]:
    """Return list of matching practice area keys for the given lowercased text."""
    matched = []
    for area, keywords in PRACTICE_AREA_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            matched.append(area)
    return matched
