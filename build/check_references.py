#!/usr/bin/env python3
"""check_references.py — re-resolve every DOI in REFERENCES.md against Crossref.

    python3 build/check_references.py

Checks, per entry, that the DOI resolves and that the stored first-author
surname, year and volume match what Crossref returns. Catches the two things
that actually go wrong when citations are written from memory: an author list
that drifts, and a DOI that was assumed rather than looked up. Both happened
while assembling this file.

It does NOT check that the cited work supports the claim made about it. Only
reading it does that.

Exit code 1 on any mismatch, so it can gate a commit that touches citations.
"""
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFS = os.path.join(ROOT, "REFERENCES.md")
UA = {"User-Agent": "aif-course/1.0 (mailto:jon.shock@gmail.com)"}

DOI_RE = re.compile(r"doi:\[([^\]]+)\]")
YEAR_RE = re.compile(r"\((\d{4})[a-z]?\)")
# "Surname, A. B. & ..." — surnames may be multi-word ("Da Costa", "de Vries")
FIRST_AUTHOR_RE = re.compile(r"^\s*\|?\s*([^,|]+?),")
VOL_RE = re.compile(r"\*\*(\d+)\*\*")


def crossref(doi):
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=25) as r:
        return json.load(r)["message"]


def norm(s):
    return re.sub(r"[^a-z]", "", (s or "").lower())


def main():
    text = open(REFS, encoding="utf-8").read()
    problems, checked = [], 0

    for line in text.split("\n"):
        m = DOI_RE.search(line)
        if not m:
            continue
        doi = m.group(1).strip()
        cell = line.split("|")[1] if line.startswith("|") else line
        want_year = YEAR_RE.search(cell)
        want_auth = FIRST_AUTHOR_RE.search(cell)
        want_vol = VOL_RE.search(cell)
        label = (want_auth.group(1) if want_auth else "?") + " " + (want_year.group(1) if want_year else "?")

        try:
            mm = crossref(doi)
        except Exception as e:
            problems.append(f"{label}: DOI {doi} did not resolve ({e})")
            continue
        checked += 1

        got_auth = (mm.get("author") or [{}])[0].get("family", "")
        parts = mm.get("issued", {}).get("date-parts") or [[None]]
        got_year = parts[0][0]
        got_vol = mm.get("volume")
        got_title = (mm.get("title") or ["?"])[0]

        # A check that silently skips is worse than no check: say so out loud.
        if not want_auth:
            problems.append(f"{doi}: could not parse a first author from the entry, "
                            f"so the author was NOT checked (Crossref says '{got_auth}')")
        elif norm(got_auth) and norm(want_auth.group(1)) != norm(got_auth):
            problems.append(f"{label}: first author is '{got_auth}', file says "
                            f"'{want_auth.group(1)}'  ({doi})")
        if want_year and got_year and abs(int(want_year.group(1)) - int(got_year)) > 1:
            problems.append(f"{label}: year is {got_year}, file says {want_year.group(1)}  ({doi})")
        if want_vol and got_vol and want_vol.group(1) != str(got_vol):
            problems.append(f"{label}: volume is {got_vol}, file says {want_vol.group(1)}  ({doi})")

        print(f"  ok  {label:<22} {got_title[:58]}")
        time.sleep(0.35)

    print()
    for p in problems:
        print("  MISMATCH:", p)
    print(f"\n{checked} DOIs resolved · {len(problems)} mismatches")
    if not problems:
        print("Note: this confirms the citations exist and are described correctly.")
        print("It does not confirm they support the claims made about them.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
