# -*- coding: utf-8 -*-
"""Fetch footwear from Google Scanned Objects (CC BY 4.0).

Why this dataset and not the product photos we started with:
  - every record is CC BY 4.0, so it can ship in a public demo with attribution
  - each model carries five pre-rendered 640x480 white-background views whose
    angles happen to line up with the vendor's camera plan (front/toe, lateral,
    rear/heel), which the free-form product photos never gave us
  - the meshes are METRIC, so a rendered view has a known pixel/mm scale and the
    measurements can finally be checked against ground truth instead of only
    against each other

Attribution required in the UI: Google Research / Open Robotics, CC BY 4.0.
"""
import os, sys, io, json, time, zipfile
import urllib.request
import urllib.parse

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "data", "gso")
API = "https://fuel.gazebosim.org/1.0"
UA = {"User-Agent": "vringon-qa/1.0 (research demo)"}

SHOE_HINT = ("shoe", "sneaker", "boot", "sandal", "loafer", "heel", "clog",
             "slipper", "trainer", "runner", "cleat", "moccasin", "oxford",
             "flip_flop", "pump")
N_ZIPS = int(os.environ.get("GSO_ZIPS", "36"))


def get(url, timeout=90):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def list_models():
    """Every GoogleResearch model, paginated."""
    out, page = [], 1
    while True:
        try:
            data = json.loads(get(f"{API}/GoogleResearch/models?per_page=100&page={page}"))
        except Exception as e:
            print("page", page, "failed:", e, flush=True)
            break
        if not data:
            break
        out += data
        print(f"  page {page}: +{len(data)} (total {len(out)})", flush=True)
        page += 1
        if page > 20:
            break
    return out


def is_footwear(m):
    n = m["name"].lower()
    cats = " ".join(c.get("name", "") if isinstance(c, dict) else str(c)
                    for c in (m.get("categories") or [])).lower()
    return any(k in n for k in SHOE_HINT) or "shoe" in cats


def main():
    os.makedirs(OUT, exist_ok=True)
    idx_path = os.path.join(OUT, "models.json")
    if os.path.exists(idx_path):
        models = json.load(open(idx_path, encoding="utf-8"))
        print(f"reusing model index: {len(models)}", flush=True)
    else:
        print("listing GoogleResearch models...", flush=True)
        models = list_models()
        json.dump(models, open(idx_path, "w", encoding="utf-8"))
    shoes = [m for m in models if is_footwear(m)]
    lic = {m.get("license_name") for m in shoes}
    print(f"total={len(models)} footwear={len(shoes)} licenses={lic}", flush=True)
    json.dump([{k: m[k] for k in ("name", "license_name", "license_url", "filesize")}
               for m in shoes],
              open(os.path.join(OUT, "footwear.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # 1) thumbnails for every footwear model - cheap, and already multi-view
    tdir = os.path.join(OUT, "thumbs")
    os.makedirs(tdir, exist_ok=True)
    got = 0
    for m in shoes:
        d = os.path.join(tdir, m["name"])
        if os.path.isdir(d) and len(os.listdir(d)) >= 4:
            continue
        os.makedirs(d, exist_ok=True)
        for i in range(5):
            p = os.path.join(d, f"{i}.jpg")
            if os.path.exists(p):
                continue
            url = f"{API}/GoogleResearch/models/{urllib.parse.quote(m['name'])}/tip/files/thumbnails/{i}.jpg"
            try:
                open(p, "wb").write(get(url, timeout=45))
            except Exception:
                break
        got += 1
        if got % 20 == 0:
            print(f"  thumbs {got}/{len(shoes)}", flush=True)
    print(f"thumbnails done for {len(shoes)} models", flush=True)

    # 2) full zips for a subset - meshes give ground-truth millimetres and let
    #    us render the top and bottom views the spec needs and no photo has
    zdir = os.path.join(OUT, "models")
    os.makedirs(zdir, exist_ok=True)
    subset = sorted(shoes, key=lambda m: -m.get("downloads", 0))[:N_ZIPS]
    for i, m in enumerate(subset):
        d = os.path.join(zdir, m["name"])
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "meshes", "model.obj")):
            continue
        url = f"{API}/GoogleResearch/models/{urllib.parse.quote(m['name'])}.zip"
        try:
            blob = get(url, timeout=240)
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                z.extractall(d)
            print(f"  [{i+1}/{len(subset)}] {m['name']} "
                  f"({len(blob)/1e6:.1f} MB)", flush=True)
        except Exception as e:
            print(f"  !! {m['name']}: {e}", flush=True)
    print("GSO DONE", flush=True)


if __name__ == "__main__":
    main()
