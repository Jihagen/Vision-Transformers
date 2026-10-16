"""
Resolve the SMID images shown in the web transfer gallery to their individual
source and licence, and write attack/IV2_generalisation/smid_provenance.json.

The Hugging Face mirror the run downloaded from (AIML-TUDA/smid) is gated and
carries no per-image licence. The authority used here is the original SMID
release on OSF (https://osf.io/2rqad/), whose SMID_metadata.json gives author,
title, licence and source URL per image. Every source is a Wikimedia Commons
file, so each record is checked against the current Commons file page; the two
licence statements must agree for scripts/export_web_data.py to publish it.

Needs network access. Run it again whenever the gallery selection changes:

    python scripts/build_smid_provenance.py

Content warnings are written by hand after looking at the image (CONTENT_REVIEW
below). A selected image without an entry there is recorded as not reviewed and
is not published.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "attack" / "IV2_generalisation" / "smid_provenance.json"
MANIFEST_PATH = REPO_ROOT / "results" / "generalisation" / "moral_evaluation" / "sample_manifest.csv"
GALLERY_PATH = REPO_ROOT / "web_export" / "data" / "attack" / "transfer_gallery.json"

SMID_RELEASE_URL = "https://osf.io/2rqad/"
SMID_METADATA_URL = "https://osf.io/download/e7gej/"       # Data component: SMID_metadata.json
SMID_CITATION = ("Crone DL, Bode S, Murawski C, Laham SM (2018). The Socio-Moral Image Database "
                 "(SMID): A novel stimulus set for the study of social, moral and affective "
                 "processes. PLoS ONE 13(1): e0190954. https://doi.org/10.1371/journal.pone.0190954")
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "smid-provenance-check/1.0 (research project; per-image licence verification)"

# SMID image id -> content warning (None: nothing to warn about). Written after
# viewing the image; an id missing here counts as not reviewed.
CONTENT_REVIEW = {
    "b999_p484_15": None,                                           # clear-cut forest
    "b15_p349_17": "Shows a hanging noose.",
    "b999_p496_13": None,                                           # hands being washed
    "b6_p87_14": None,                                              # two gulls
    "b999_p482_20": None,                                           # Hall of Mirrors, Versailles
    "b14_p260_17": None,                                            # chocolate cake
    "b15_p293_18": "Shows a street protest with offensive placards.",
    "b999_p485_2": "Shows a war cemetery.",
    "b10_p127_20": None,                                            # officers talking
    "b15_p352_19": None,                                            # two rhinoceroses
    "b15_p360_3": "Shows homes destroyed by a tornado.",
    "b2_p41_5": "Shows an injured earthquake survivor being pulled from rubble.",
}

# Commons licence category -> public-domain basis (template name on Commons).
PD_BASIS = {
    "Category:PD US Army": "PD-USGov-Military-Army",
    "Category:PD US Navy": "PD-USGov-Military-Navy",
    "Category:PD-self": "PD-self",
    "Category:PD-user": "PD-user",
}


def _get(url: str, retries: int = 6) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == retries - 1:
                raise
            time.sleep(20 * (attempt + 1))
    raise RuntimeError(url)


def _plain(value: str | None) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", value or ""))).strip()


def commons_record(source_url: str) -> dict | None:
    """Current licence statement of the Commons file behind a SMID source URL."""
    filename = urllib.parse.unquote(source_url.rsplit("/", 1)[1])
    query = urllib.parse.urlencode({
        "action": "query", "format": "json", "redirects": 1, "titles": f"File:{filename}",
        "prop": "imageinfo|categories", "iiprop": "extmetadata|url|sha1", "cllimit": "max"})
    page = next(iter(json.loads(_get(f"{COMMONS_API}?{query}"))["query"]["pages"].values()))
    if "imageinfo" not in page:
        return None
    info = page["imageinfo"][0]
    ext = lambda key: _plain(info["extmetadata"].get(key, {}).get("value"))
    categories = [c["title"] for c in page.get("categories", [])]
    return {
        "file_page": info["descriptionurl"],
        "file_title": page["title"],
        "original_file_sha1": info["sha1"],
        "licence_short_name": ext("LicenseShortName"),
        "licence_code": ext("License"),
        "licence_url": ext("LicenseUrl") or None,
        "artist": ext("Artist"),
        "attribution_required": ext("AttributionRequired") == "true",
        "restrictions": ext("Restrictions") or None,
        "public_domain_basis": next((PD_BASIS[c] for c in categories if c in PD_BASIS), None),
    }


def selected_sample_ids() -> list[str]:
    gallery = json.loads(GALLERY_PATH.read_text())
    task = next(t for t in gallery["tasks"] if t["task"] == "moral_evaluation")
    return sorted({sid for x in task["targets"] for sid in x["selected_ids"]})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sample_ids", nargs="+", default=None,
                   help="Run sample ids (smid_00029 ...). Default: the samples currently "
                        "selected in web_export/data/attack/transfer_gallery.json.")
    args = p.parse_args()

    raw = _get(SMID_METADATA_URL)
    smid = json.loads(raw)
    manifest = pd.read_csv(MANIFEST_PATH).set_index("sample_id")
    samples = {}
    for sid in args.sample_ids or selected_sample_ids():
        smid_id = Path(manifest.loc[sid, "image_id"]).stem
        record = smid.get(smid_id)
        commons = commons_record(record["URL"]) if record else None
        time.sleep(4)   # the Commons API rate-limits bursts
        samples[smid_id] = {
            "sample_id": sid,
            "smid": ({"author": record["author"], "title": record["title"],
                      "licence": record["license"], "source_url": record["URL"]} if record else None),
            "commons": commons,
            "content_reviewed": smid_id in CONTENT_REVIEW,
            "content_warning": CONTENT_REVIEW.get(smid_id),
        }
        print(f"  {sid}  {smid_id}: SMID {record['license'] if record else 'no record'} | "
              f"Commons {commons['licence_short_name'] if commons else 'file not found'}")

    doc = {
        "dataset": {
            "name": "Socio-Moral Image Database (SMID)",
            "citation": SMID_CITATION,
            "release_url": SMID_RELEASE_URL,
            "metadata_file": {"name": "SMID_metadata.json", "url": SMID_METADATA_URL,
                              "sha256": hashlib.sha256(raw).hexdigest()},
        },
        "verified_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": "Each sample's SMID image id is looked up in the original release's "
                  "SMID_metadata.json; the Wikimedia Commons file it names is then queried "
                  "(imageinfo/extmetadata, following renames) for its current licence.",
        "samples": samples,
    }
    OUT_PATH.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(f"  wrote  {OUT_PATH.relative_to(REPO_ROOT)}  ({len(samples)} samples)")


if __name__ == "__main__":
    main()
