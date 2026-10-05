"""Refresh StockShield's local medicine reference catalogue from current NLM RxNorm data.

The download is optional at runtime: the packaged 3,972-record reference seed keeps the
application usable offline. On a network-enabled machine this script can replace/expand
that seed using the current RxNorm prescribable release, then StockShield still performs
FDA/NDC and other live lookups for selected medicines.
"""
import csv, hashlib, io, os, re, tempfile, urllib.request, zipfile
from pathlib import Path

URL = "https://download.nlm.nih.gov/umls/kss/rxnorm/RxNorm_full_prescribe_current.zip"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "medicine_dataset.csv"
LIMIT = int(os.getenv("STOCKSHIELD_CATALOGUE_LIMIT", "10000"))


def clean(s):
    return re.sub(r"\\s+", " ", (s or "").strip())


def main():
    print(f"Downloading current RxNorm prescribable release: {URL}")
    with urllib.request.urlopen(URL, timeout=30) as r:
        blob = r.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        # RXNCONSO.RRF contains normalized RxNorm concepts and synonyms.
        name = next((n for n in z.namelist() if n.upper().endswith("RXNCONSO.RRF")), None)
        if not name:
            raise RuntimeError("RXNCONSO.RRF not found in RxNorm release")
        rows = {}
        with z.open(name) as fh:
            for raw in io.TextIOWrapper(fh, encoding="utf-8", errors="replace"):
                p = raw.rstrip("\n").split("|")
                if len(p) < 14 or p[11] != "RXNORM":
                    continue
                rxcui, tty, string = p[0], p[12], clean(p[14] if len(p) > 14 else "")
                # IN = ingredient; SCD/SBD = clinical drug / branded clinical drug.
                if tty not in {"IN", "SCD", "SBD"} or not string:
                    continue
                key = (rxcui, tty, string.lower())
                rows[key] = (rxcui, string, tty)
                if len(rows) >= LIMIT:
                    break
    out = []
    for rxcui, name, tty in rows.values():
        cid = hashlib.sha256(f"rxnorm|{rxcui}|{name}|{tty}".encode()).hexdigest()[:24]
        out.append([cid, name, "RxNorm", "Reference", "", "NLM RxNorm", "Medication reference", tty, "NLM RxNorm current release"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, newline="", encoding="utf-8", dir=OUT.parent) as f:
        w = csv.writer(f); w.writerow(["catalogue_id","Name","Category","Dosage Form","Strength","Manufacturer","Indication","Classification","Source"]); w.writerows(out); tmp=f.name
    os.replace(tmp, OUT)
    print(f"Wrote {len(out):,} local RxNorm catalogue records to {OUT}")


if __name__ == "__main__":
    main()
