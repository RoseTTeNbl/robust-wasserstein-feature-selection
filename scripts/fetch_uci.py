"""Download the two attributed public benchmarks; check the frozen zip hashes."""

import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np

SOURCES = [
    dict(
        name="wdbc",
        url="https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip",
        sha256="bc154869ef13f753f9e2b5a17e248cfe1ba4b6721db7c4da9f4880e40b05d3af",
        doi="https://doi.org/10.24432/C5DW2B",
        license="CC BY 4.0",
        rows=569,
        features=30,
        label_mapping={"0": "benign", "1": "malignant"},
        excluded="ID and diagnosis",
    ),
    dict(
        name="ionosphere",
        url="https://archive.ics.uci.edu/static/public/52/ionosphere.zip",
        sha256="4d218ece62756c99659011a13052d06464f13cb2b3d0410ce2aef16de1403860",
        doi="https://doi.org/10.24432/C5W01B",
        license="CC BY 4.0",
        rows=351,
        features=34,
        label_mapping={"0": "good", "1": "bad"},
        excluded="label only",
    ),
]


def main():
    target = Path(__file__).resolve().parents[1] / "data/raw/uci"
    target.mkdir(parents=True, exist_ok=True)
    for source in SOURCES:
        zip_path = target / (source["name"] + ".zip")
        blob = (
            zip_path.read_bytes()
            if zip_path.exists()
            else urlopen(source["url"], timeout=60).read()
        )
        if hashlib.sha256(blob).hexdigest() != source["sha256"]:
            raise RuntimeError(
                "Source hash changed: inspect the upstream data before replacing this fixture"
            )
        zip_path.write_bytes(blob)
        with ZipFile(io.BytesIO(blob)) as archive:
            data_file = next(
                n for n in archive.namelist() if n.endswith(source["name"] + ".data")
            )
            rows = [
                line.split(",")
                for line in archive.read(data_file).decode().strip().splitlines()
            ]
        if source["name"] == "wdbc":
            x = np.array([r[2:] for r in rows], float)
            y = np.array([int(r[1] == "M") for r in rows])
        else:
            x = np.array([r[:-1] for r in rows], float)
            y = np.array([int(r[-1] == "b") for r in rows])
        if x.shape != (source["rows"], source["features"]):
            raise RuntimeError(
                "The parsed benchmark dimensions differ from the frozen source"
            )
        np.savez_compressed(target / (source["name"] + ".npz"), x=x, y=y)
        print(source["name"], x.shape)
    (target / "manifest.json").write_text(
        json.dumps(SOURCES, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
