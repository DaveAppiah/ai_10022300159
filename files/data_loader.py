# src/data_loader.py
# Author: David Owusu Appiah | Index: 10022300159
# CS4241 - Introduction to Artificial Intelligence - 2026
"""
Part A: Data Engineering & Preparation
Handles loading and cleaning of:
  1. Ghana Election Results CSV
  2. Ghana 2025 Budget Statement PDF
"""

import re
import io
import requests
import pandas as pd
import fitz  # PyMuPDF


# ── Local file paths ────────────────────────────────────────────────────────────
import pathlib
# Project `data/` folder next to this module (not parent.parent, which breaks when
# data_loader lives at repo root instead of under src/).
DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"
LOCAL_CSV_PATH = DATA_DIR / "Ghana_Election_Result.csv"
LOCAL_PDF_PATH = DATA_DIR / "2025-Budget-Statement-and-Economic-Policy_v4.pdf"

# ── Fallback URLs (used only if local files are missing) ───────────────────────
CSV_URL = (
    "https://raw.githubusercontent.com/GodwinDansoAcity/acitydataset/"
    "main/Ghana_Election_Result.csv"
)
PDF_URL = (
    "https://mofep.gov.gh/sites/default/files/budget-statements/"
    "2025-Budget-Statement-andEconomic-Policy_v4.pdf"
)


# ── CSV Loader ────────────────────────────────────────────────────────────────
def load_election_csv(csv_path: pathlib.Path | str = LOCAL_CSV_PATH,
                      url: str = CSV_URL) -> pd.DataFrame:
    """
    Load the Ghana Election Results CSV.

    Order of precedence:
      1. Local file at csv_path (if exists)
      2. Download from url (fallback)
    """
    csv_path = pathlib.Path(csv_path)

    if csv_path.exists():
        print(f"[data_loader] Loading Election CSV from local file: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print(f"[data_loader] Local CSV not found, downloading from: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))

    # 1. Normalise column names
    df.columns = [c.strip().replace(" ", "_").upper() for c in df.columns]

    # 2. Strip string cells
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

    # 3. Drop completely empty rows
    df.dropna(how="all", inplace=True)

    # 4. Normalise numeric-looking columns
    for col in df.columns:
        if df[col].dtype == object:
            cleaned = df[col].str.replace(",", "", regex=False)
            try:
                df[col] = pd.to_numeric(cleaned)
            except (ValueError, TypeError):
                pass

    # 5. Fill remaining NaNs in string cols
    df[str_cols] = df[str_cols].fillna("Unknown")

    return df


def election_df_to_documents(df: pd.DataFrame) -> list[dict]:
    """
    Convert each CSV row into a plain-text document dict.

    Returns list of:
      {"text": str, "source": "election_csv", "metadata": {col: val, ...}}
    """
    docs = []
    for _, row in df.iterrows():
        # Build a human-readable sentence per row
        parts = [f"{col.replace('_', ' ').title()}: {val}" for col, val in row.items()]
        text = " | ".join(parts)
        docs.append({
            "text": text,
            "source": "election_csv",
            "metadata": row.to_dict(),
        })
    return docs


# ── PDF Loader ────────────────────────────────────────────────────────────────
def load_budget_pdf(pdf_path: pathlib.Path | str = LOCAL_PDF_PATH,
                    url: str = PDF_URL) -> str:
    """
    Load the 2025 Budget Statement PDF.

    Order of precedence:
      1. Local file at pdf_path (if exists)
      2. Download from url (fallback)

    Uses PyMuPDF (fitz) for text extraction.
    """
    pdf_path = pathlib.Path(pdf_path)

    if pdf_path.exists():
        print(f"[data_loader] Reading Budget PDF from local file: {pdf_path}")
        pdf_bytes = pdf_path.read_bytes()
        print(f"[data_loader] Read {len(pdf_bytes):,} bytes from disk")
    else:
        print(f"[data_loader] Local PDF not found, downloading from: {url}")
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            pdf_bytes = response.content
            print(f"[data_loader] Downloaded {len(pdf_bytes):,} bytes")
        except Exception as exc:
            print(f"[data_loader] WARNING: could not fetch PDF — {exc}")
            print("[data_loader] Returning empty string. Check URL/network.")
            return ""

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = []
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text("text")
            if page_text.strip():
                pages.append(f"[Page {page_num}]\n{page_text}")
        doc.close()
        print(f"[data_loader] Extracted {len(pages)} pages")
    except Exception as exc:
        print(f"[data_loader] ERROR parsing PDF: {exc}")
        return ""

    raw = "\n\n".join(pages)
    cleaned = _clean_pdf_text(raw)
    print(f"[data_loader] Final text length: {len(cleaned):,} chars")
    return cleaned


def _clean_pdf_text(text: str) -> str:
    """
    Clean extracted PDF text:
      - Remove repeated whitespace
      - Remove page-header/footer noise (heuristic: very short lines < 4 chars)
      - Collapse multiple blank lines to a single blank line
    """
    # Remove lines that are just numbers (page markers from PDF renderer)
    lines = text.splitlines()
    lines = [ln for ln in lines if not re.fullmatch(r"\s*\d{1,4}\s*", ln)]

    # Remove very short noise lines (e.g. lone dashes, bullets)
    lines = [ln for ln in lines if len(ln.strip()) >= 4 or ln.strip() == ""]

    text = "\n".join(lines)

    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def pdf_to_documents(text: str) -> list[dict]:
    """
    Wrap the full PDF text as a single document (chunker will split it).
    """
    if not text:
        return []
    return [{"text": text, "source": "budget_pdf", "metadata": {"title": "Ghana 2025 Budget Statement"}}]


# ── Combined Loader ───────────────────────────────────────────────────────────
def load_all_documents() -> list[dict]:
    """
    Load and return all raw documents from both datasets.
    Returns list of raw document dicts before chunking.
    """
    print("[data_loader] Loading Election CSV …")
    df = load_election_csv()
    election_docs = election_df_to_documents(df)
    print(f"[data_loader] {len(election_docs)} election rows loaded.")

    print("[data_loader] Loading Budget PDF …")
    pdf_text = load_budget_pdf()
    budget_docs = pdf_to_documents(pdf_text)
    print(f"[data_loader] Budget PDF loaded ({len(pdf_text):,} chars).")

    return election_docs + budget_docs
