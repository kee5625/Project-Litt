"""
Statute parsing for three sources:
  1. Federal USC XML ZIPs  (raw/statutes/federal/usc{28,29,42}.zip)
  2. CA code HTML files    (raw/statutes/ca/{CODE}_{start}_{end}/section_XXXXX.html)
  3. pol us_bills JSONL    (raw/pol/us_bills.jsonl)
"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from html.parser import HTMLParser

RAW_STATUTES = Path("raw/statutes")
RAW_POL = Path("raw/pol")

TITLE_PRACTICE_AREA: dict[str, str] = {
    "28": "civil_rights",
    "29": "employment",
    "42": "civil_rights",
}

CA_DIR_META: dict[str, tuple[str, str, str]] = {
    # dir_prefix → (code_name, jurisdiction, practice_area)
    "LAB": ("CA Labor Code", "CA", "employment"),
    "FAM": ("CA Family Code", "CA", "family"),
    "PEN": ("CA Penal Code", "CA", "criminal"),
    "CIV": ("CA Civil Code", "CA", "tort"),
}


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

class _BodyExtractor(HTMLParser):
    """Strips tags; skips <script>, <style>, <head>. Captures text from
    the codeLawSectionNoHead div once found."""

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._in_target = False
        self._target_depth = 0
        self._depth = 0
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        self._depth += 1
        attr_dict = dict(attrs)
        if attr_dict.get("id") == "codeLawSectionNoHead":
            self._in_target = True
            self._target_depth = self._depth
        if tag in ("script", "style", "head"):
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "head") and self._skip_depth:
            self._skip_depth -= 1
        if self._in_target and self._depth == self._target_depth:
            self._in_target = False
        self._depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth or not self._in_target:
            return
        stripped = data.strip()
        if stripped:
            self.text.append(stripped)


def _extract_ca_text(html: str) -> str:
    p = _BodyExtractor()
    p.feed(html)
    text = "\n".join(p.text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ---------------------------------------------------------------------------
# Federal XML ZIPs
# ---------------------------------------------------------------------------

def parse_federal_zip(zip_path: Path) -> list[dict]:
    from xml.etree import ElementTree as ET

    title = zip_path.stem.replace("usc", "")
    practice_area = TITLE_PRACTICE_AREA.get(title, "civil_rights")
    ns = {"uslm": "http://xml.house.gov/schemas/uslm/1.0"}
    sections: list[dict] = []

    with zipfile.ZipFile(zip_path) as zf:
        xml_files = [n for n in zf.namelist() if n.endswith(".xml")]
        for xml_name in xml_files:
            with zf.open(xml_name) as f:
                try:
                    tree = ET.parse(f)
                except ET.ParseError:
                    continue

            root = tree.getroot()
            # Strip default namespace for findall
            tag_section = f"{{{ns['uslm']}}}section"
            tag_num = f"{{{ns['uslm']}}}num"
            tag_heading = f"{{{ns['uslm']}}}heading"

            for section_el in root.iter(tag_section):
                # Skip container sections that hold sub-sections
                has_child_sections = any(
                    child.tag == tag_section for child in section_el
                )
                if has_child_sections:
                    continue

                num_el = section_el.find(tag_num)
                heading_el = section_el.find(tag_heading)
                num = (num_el.text or "").strip() if num_el is not None else ""
                heading = (heading_el.text or "").strip() if heading_el is not None else ""

                # Gather all text recursively
                raw_text = " ".join(
                    (t.strip() for t in section_el.itertext() if t.strip())
                )
                raw_text = re.sub(r"\s+", " ", raw_text).strip()

                if len(raw_text) < 50:
                    continue

                safe_num = re.sub(r"[^a-zA-Z0-9_]", "_", num.lstrip("§").strip())
                section_id = f"fed_{title}_{safe_num}"
                code = f"{title} U.S.C. {num}".strip()

                sections.append({
                    "id": section_id,
                    "title": f"{heading} ({code})"[:512] if heading else code[:512],
                    "code": code,
                    "jurisdiction": "federal",
                    "practice_area": practice_area,
                    "section_text": raw_text[:2000],
                    "last_amended": 2024,
                })

    return sections


def parse_all_federal_zips() -> list[dict]:
    fed_dir = RAW_STATUTES / "federal"
    results: list[dict] = []
    for zip_path in sorted(fed_dir.glob("usc*.zip")):
        rows = parse_federal_zip(zip_path)
        results.extend(rows)
    return results


# ---------------------------------------------------------------------------
# CA HTML files
# ---------------------------------------------------------------------------

def parse_ca_section_file(html_path: Path, code_prefix: str, code_name: str,
                           practice_area: str) -> dict | None:
    # Filename: section_00098.html → section number 98
    stem = html_path.stem  # "section_00098"
    num_match = re.search(r"section_0*(\d+)", stem)
    if not num_match:
        return None
    section_num = num_match.group(1)

    try:
        html = html_path.read_text(errors="replace")
    except OSError:
        return None

    text = _extract_ca_text(html)
    if len(text) < 50:
        return None

    section_id = f"ca_{code_prefix}_{section_num}"
    code = f"{code_name} § {section_num}"

    # Title: first line of extracted text (usually includes section heading)
    first_line = text.split("\n")[0][:512]

    return {
        "id": section_id,
        "title": first_line or code,
        "code": code,
        "jurisdiction": "CA",
        "practice_area": practice_area,
        "section_text": text[:2000],
        "last_amended": 2024,
    }


def parse_all_ca_dirs() -> list[dict]:
    ca_root = RAW_STATUTES / "ca"
    results: list[dict] = []

    for subdir in sorted(ca_root.iterdir()):
        if not subdir.is_dir():
            continue
        # Directory name like LAB_98_132 or CIV_1714_1725
        dir_name = subdir.name
        code_prefix = dir_name.split("_")[0]
        meta = CA_DIR_META.get(code_prefix)
        if not meta:
            continue
        code_name, _jurisdiction, practice_area = meta

        for html_path in sorted(subdir.glob("section_*.html")):
            row = parse_ca_section_file(html_path, code_prefix, code_name, practice_area)
            if row:
                results.append(row)

    return results


# ---------------------------------------------------------------------------
# pol us_bills
# ---------------------------------------------------------------------------

def parse_us_bills(jsonl_path: Path | None = None) -> list[dict]:
    if jsonl_path is None:
        jsonl_path = RAW_POL / "us_bills.jsonl"
    if not jsonl_path.exists():
        return []

    results: list[dict] = []
    with open(jsonl_path) as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            row = _parse_one_bill(doc, idx)
            if row:
                results.append(row)
    return results


def _parse_one_bill(doc: dict, index: int) -> dict | None:
    url = doc.get("url", "")
    text = (doc.get("text") or "").strip()
    if len(text) < 100:
        return None

    bill_match = re.search(r"BILLS-(\w+)[/.]", url)
    bill_id = bill_match.group(1) if bill_match else str(index)

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    title = lines[0][:512] if lines else f"US Bill {bill_id}"

    ts = doc.get("created_timestamp", "")
    year_match = re.search(r"(\d{4})", ts)
    year = int(year_match.group(1)) if year_match else 2000

    areas: list[str] = doc.get("practice_area_matches", [])
    practice_area = areas[0] if areas else "civil_rights"

    section_text = "\n".join(lines[:50])[:2000]

    return {
        "id": f"usbill_{bill_id}",
        "title": title,
        "code": f"US Bill {bill_id}",
        "jurisdiction": "federal",
        "practice_area": practice_area,
        "section_text": section_text,
        "last_amended": year,
    }


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def parse_all_statutes() -> list[dict]:
    results: list[dict] = []
    results.extend(parse_all_federal_zips())
    results.extend(parse_all_ca_dirs())
    results.extend(parse_us_bills())
    # Deduplicate by id (should not happen, but safety net)
    seen: set[str] = set()
    deduped: list[dict] = []
    for row in results:
        if row["id"] not in seen:
            seen.add(row["id"])
            deduped.append(row)
    return deduped
