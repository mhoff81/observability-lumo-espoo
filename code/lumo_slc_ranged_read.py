#!/usr/bin/env python3
"""Reproduce a cached LUMO burst window/strip from the Sentinel-1 SLC it came from.

The Rust pipeline (`tower/backend`) anchors the LUMO mast coordinate on a burst's
geolocation grid, cuts a 41x41 `window.bin` and a 400x11 `strip.bin` out of the
measurement TIFF and caches them under `LUMO_BURST_CACHE_DIR`. This script does
the same reading **only the bytes it needs** out of the same CDSE object store -
the 7-byte TIFF header, the IFD arrays and the individual rows of the two
rectangles - and compares the result with the cached binaries sample by sample,
without ever downloading the multi-GB measurement TIFF.

Two anchor rules are scored, because the cache spans a behaviour change in the
Rust code (commit `fad0e46`, "location to insar pixel mapping fixed", 2026-09-10):

  * `current` (post-fad0e46) - `GeolocationGrid::interpolate_line_pixel`: the
    (line, pixel) of the mast coordinate by bilinear Newton inversion, clamped to
    the selected grid cell;
  * `legacy` (pre-fad0e46) - the crude inversion: lat bounds taken from the
    c00/c01 edge, lon bounds from the c00/c10 edge, u/v clamped to [0, 1].

Either rule can clamp silently, which is how an entry ends up anchored tens of km
from the mast; `--probe` prints the anchors, the selected cell, the nearest grid
node and both ground errors, so a clamp is visible rather than implied.
`lumo_burst_cache_audit.py` runs this rule set over a whole cache directory.

NETWORK: this is the one script in this folder that *does* need network access
and CDSE credentials. Credentials are read from the environment
(`COPERNICUS_CLIENT_ID`, `COPERNICUS_CLIENT_SECRET`, `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`) and, if you keep them in a file, from `--env-file` /
`$LUMO_SLC_ENV_FILE`; the environment wins. `--selftest` is fully offline.

INPUT: a cache directory written by the Rust pipeline - one sub-directory per
entry, each holding `meta.json`, `window.bin` and `strip.bin`:

  <cache>/<YYYY-MM>/<date>_<id>/meta.json

USAGE
  python3 code/lumo_slc_ranged_read.py --cache-dir DIR --entry 2021-04/2021-04-03_1e17aa73 --verify
  python3 code/lumo_slc_ranged_read.py --cache-dir DIR --entry MONTH/NAME --probe
  python3 code/lumo_slc_ranged_read.py --cache-dir DIR --entry MONTH/NAME --verify --samples-csv out.csv
  python3 code/lumo_slc_ranged_read.py --selftest

Exit codes: 0 ok, 1 unusable --cache-dir/--env-file or credentials, 2 cache entry
or measurement TIFF unusable, 3 verification mismatch, 4 selftest failure
(argparse exits 2 on bad usage).
"""
import argparse
import csv
import datetime
import hashlib
import hmac
import http.client
import json
import math
import os
import re
import struct
import sys
import tempfile
import urllib.parse
import urllib.request

# --- CDSE endpoints (public; no host, path or checkout is hard-coded below) ---
HOST = "eodata.dataspace.copernicus.eu"
REGION, SERVICE = "default", "s3"
TOKEN_URL = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
             "protocol/openid-connect/token")
CAT = "https://catalogue.dataspace.copernicus.eu/odata/v1"

# --- The mast, and the two rectangles the Rust pipeline cuts -----------------
# MAST_LAT/MAST_LON mirror LUMO_MAST_LATITUDE / LUMO_MAST_LONGITUDE, the
# compile-time constants of the Rust pipeline
# (input_proxy/insar_processor/constants.rs). Override with --lat/--lon.
MAST_LAT, MAST_LON = 52.38, 9.72
SEARCH_SIZE = 41                       # SEARCH_SIZE (window.bin)
STRIP_WIDTH = 11                       # TOWER_PHASE_STRIP_WIDTH
STRIP_LINES = 400                      # PHASE_STRIP_LINES

# --- Configuration: environment first, nothing hard-coded --------------------
CACHE_VAR = "LUMO_BURST_CACHE_DIR"     # the variable the Rust pipeline reads
ENV_FILE_VAR = "LUMO_SLC_ENV_FILE"     # optional key=value file with the secrets
LAT_VAR, LON_VAR = "LUMO_MAST_LATITUDE", "LUMO_MAST_LONGITUDE"
CRED_KEYS = ("COPERNICUS_CLIENT_ID", "COPERNICUS_CLIENT_SECRET",
             "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")

# fad0e46 "location to insar pixel mapping fixed" (Thu Sep 10 19:40:13 2026 UTC):
# the revision that replaced the crude clamped inversion with the bilinear
# Newton inversion. Cache entries written before it used the `legacy` rule.
FIX_TS = datetime.datetime(2026, 9, 10, 19, 40, 13,
                           tzinfo=datetime.timezone.utc).timestamp()

SAMPLE_COLUMNS = ("kind", "index", "row", "col", "line", "pixel", "i", "q",
                  "magnitude", "phase_rad", "i_cached", "q_cached", "equal")


def load_env(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def fetch_token(client_id, client_secret):
    form = urllib.parse.urlencode({
        "grant_type": "client_credentials",
        "client_id": client_id, "client_secret": client_secret}).encode()
    req = urllib.request.Request(TOKEN_URL, data=form)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())["access_token"]


def odata(url, token):
    req = urllib.request.Request(url)
    req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def credentials(env_file=None):
    """The four CDSE secrets: process environment first, then `env_file` (a
    `key=value` file such as the tower's `.env`). Raises `KeyError(list)` naming
    whatever is missing - the caller turns that into exit code 1."""
    env = dict(os.environ)
    if env_file:
        for key, val in load_env(env_file).items():
            env.setdefault(key, val)
    missing = [k for k in CRED_KEYS if not env.get(k)]
    if missing:
        raise KeyError(missing)
    return {k: env[k] for k in CRED_KEYS}


def load_entry(cache_dir, entry):
    """`MONTH/NAME` -> (month, name, entry directory, the parsed meta.json)."""
    month, _, name = entry.partition("/")
    path = os.path.join(cache_dir, month, name)
    with open(os.path.join(path, "meta.json")) as fh:
        return month, name, path, json.load(fh)


def pick_annotation(ann_paths, sw, pol):
    """The geolocation annotation of one sub-swath and polarisation. The manifest
    also lists `annotation/calibration/calibration-iw1-...` and
    `annotation/noise/noise-iw1-...`, whose file names carry the same sub-swath
    and polarisation words, so they have to be excluded by name."""
    usable = [p for p in ann_paths if "calibration" not in p.lower()
              and "noise" not in p.lower()]
    return next(p for p in usable if sw in p.lower() and pol in p.lower())


def burst_product(meta, token, s3, cache=None):
    """Resolve one cached `meta.json` to (product name, S3 root, annotation
    href, measurement href) through the CDSE OData catalogue and the SAFE
    manifest. `cache` (keyed by burst_id) spares the repeat lookups a burst
    would otherwise cost in every cache entry that reuses it."""
    bid = meta["burst_id"]
    if cache is not None and bid in cache:
        return cache[bid]
    burst = odata(f"{CAT}/Bursts({bid})", token)
    prod = odata(f"{CAT}/Products({burst['ParentProductId']})"
                 "?$select=Name,S3Path", token)
    root = prod["S3Path"].rstrip("/")
    manifest = s3.get(f"{root}/manifest.safe").decode("utf-8", "replace")
    ann_paths, meas_paths = parse_manifest(manifest)
    sw, pol = meta["subswath"].lower(), meta["polarisation"].lower()
    ann_rel = pick_annotation(ann_paths, sw, pol)
    meas_rel = next(p for p in meas_paths if sw in p.lower() and pol in p.lower())
    out = (prod["Name"], root, ann_rel, meas_rel)
    if cache is not None:
        cache[bid] = out
    return out


def annotation_grid(s3, root, ann_rel):
    """Fetch the burst annotation and parse its geolocation grid."""
    xml = s3.get(f"{root}/{ann_rel}").decode("utf-8", "replace")
    return xml, parse_geolocation_grid(xml)


def anchors(grid, lat, lon):
    """Both candidate (line, pixel) anchors of (lat, lon): the rule that was
    live before fad0e46 and the rule after it. None when the grid carries no
    usable cell for the coordinate."""
    return {"legacy": interpolate_line_pixel_pre_fad0e46(grid, lat, lon),
            "current": interpolate_line_pixel(grid, lat, lon)}


def forward_latlon(cell, line, pixel):
    """Bilinear (line, pixel) -> (lat, lon) inside a selected grid cell, the
    Rust `interpolate_lat_lon` direction. Used to measure anchor error."""
    l0, l1, p0, p1, c00, c10, c01, c11 = cell
    u = (line - l0) / (l1 - l0)
    v = (pixel - p0) / (p1 - p0)
    lat = ((1 - u) * (1 - v) * c00["lat"] + (1 - u) * v * c10["lat"]
           + u * (1 - v) * c01["lat"] + u * v * c11["lat"])
    lon = ((1 - u) * (1 - v) * c00["lon"] + (1 - u) * v * c10["lon"]
           + u * (1 - v) * c01["lon"] + u * v * c11["lon"])
    return lat, lon


def ground_distance_m(lat0, lon0, lat1, lon1):
    """Flat-earth distance between two coordinates, in metres: exact enough at
    these scales and the arithmetic every metre figure here is quoted in."""
    dy = (lat1 - lat0) * 111320.0
    dx = (lon1 - lon0) * 111320.0 * math.cos(math.radians(lat0))
    return math.hypot(dx, dy)


def ground_error_m(cell, line, pixel, lat, lon):
    """Ground distance between the coordinate an anchor maps back to and the
    requested (lat, lon): the size of the error a silent clamp leaves behind."""
    glat, glon = forward_latlon(cell, line, pixel)
    return ground_distance_m(lat, lon, glat, glon)


def tower_inside(bbox, lat, lon):
    """Is the coordinate inside a geolocation grid's bounding box?"""
    lat_min, lat_max, lon_min, lon_max = bbox
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def grid_stats(grid, lat, lon):
    """`(bbox, (nearest_distance_m, nearest_node))` of a geolocation grid.

    `bbox` is `(lat_min, lat_max, lon_min, lon_max)`. The nearest node is the
    closest of the grid's nodes to (lat, lon); for a burst that covers the mast
    even that distance is of the order of the grid step in range (~10 km), while
    for a burst that does not it runs to hundreds of km - which is what makes
    this the cheap containment test a clamped anchor fails."""
    lats = [p["lat"] for p in grid]
    lons = [p["lon"] for p in grid]
    near = min(grid, key=lambda p: ground_distance_m(lat, lon, p["lat"], p["lon"]))
    return ((min(lats), max(lats), min(lons), max(lons)),
            (ground_distance_m(lat, lon, near["lat"], near["lon"]), near))


def _sign(key, msg):
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


class S3:
    """SigV4 signed ranged GETs over one keep-alive HTTPS connection."""

    def __init__(self, ak, sk):
        self.ak, self.sk = ak, sk
        self.conn = None

    def _headers(self, path, byte_range=None):
        now = datetime.datetime.now(datetime.timezone.utc)
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = now.strftime("%Y%m%d")
        ph = "UNSIGNED-PAYLOAD"
        uri = urllib.parse.quote(path, safe="/-_.~")
        signed = f"host:{HOST}\nx-amz-content-sha256:{ph}\nx-amz-date:{amz_date}\n"
        sh = "host;x-amz-content-sha256;x-amz-date"
        canonical = "\n".join(["GET", uri, "", signed, sh, ph])
        scope = f"{date_stamp}/{REGION}/{SERVICE}/aws4_request"
        sts = "\n".join(["AWS4-HMAC-SHA256", amz_date, scope,
                         hashlib.sha256(canonical.encode()).hexdigest()])
        k = _sign(("AWS4" + self.sk).encode(), date_stamp)
        k = _sign(_sign(_sign(k, REGION), SERVICE), "aws4_request")
        sig = hmac.new(k, sts.encode(), hashlib.sha256).hexdigest()
        h = {"host": HOST, "x-amz-content-sha256": ph, "x-amz-date": amz_date,
             "authorization": f"AWS4-HMAC-SHA256 Credential={self.ak}/{scope}, "
                              f"SignedHeaders={sh}, Signature={sig}"}
        if byte_range:
            h["range"] = byte_range
        return h

    def _request(self, path, rng):
        if self.conn is None:
            self.conn = http.client.HTTPSConnection(HOST, timeout=120)
        self.conn.request("GET", path, headers=self._headers(path, rng))
        return self.conn.getresponse()

    def get(self, path, first=None, last=None, attempts=3):
        rng = None if first is None else f"bytes={first}-{last}"
        for attempt in range(attempts):
            try:
                resp = self._request(path, rng)
                body = resp.read()
                if resp.status not in (200, 206):
                    raise IOError(f"S3 {resp.status} for {path}: {body[:200]!r}")
                return body
            except Exception:
                try:
                    self.conn.close()
                except Exception:
                    pass
                self.conn = None
                if attempt + 1 >= attempts:
                    raise

    def size_of(self, path):
        resp = self._request(path, "bytes=0-0")
        resp.read()
        cr = resp.getheader("content-range")
        if not cr or "/" not in cr:
            raise IOError(f"no Content-Range for {path}: {cr!r}")
        return int(cr.rsplit("/", 1)[1])


def parse_manifest(xml):
    """Mirrors s3.rs parse_manifest_xml (annotation + measurement hrefs)."""
    ann, meas = [], []
    for blob in re.findall(r"<dataObject\b.*?</dataObject>", xml, re.S):
        m = re.search(r'<fileLocation[^>]*href="([^"]+)"', blob)
        if not m:
            continue
        href = m.group(1).strip().lstrip("./")
        if href.endswith(".xml") and "annotation" in href:
            ann.append(href)
        elif href.endswith(".tiff") and "measurement" in href:
            meas.append(href)
    return ann, meas


def _first_text(xml, tag):
    open_, close = f"<{tag}>", f"</{tag}>"
    i = xml.find(open_)
    if i < 0:
        raise ValueError(f"annotation: missing <{tag}>")
    j = xml.find(close, i)
    if j < 0:
        raise ValueError(f"annotation: missing </{tag}>")
    return xml[i + len(open_):j].strip()


def parse_geolocation_grid(xml):
    """Mirrors slc_decode.rs parse_geolocation_grid."""
    pts, pos = [], 0
    while True:
        i = xml.find("<geolocationGridPoint>", pos)
        if i < 0:
            break
        j = xml.find("</geolocationGridPoint>", i)
        if j < 0:
            raise ValueError("annotation: unterminated geolocationGridPoint")
        block = xml[i:j]
        pt = {"line": int(_first_text(block, "line")),
              "pixel": int(_first_text(block, "pixel")),
              "lat": float(_first_text(block, "latitude")),
              "lon": float(_first_text(block, "longitude")),
              "height": float(_first_text(block, "height"))}
        try:
            pt["incidence"] = float(_first_text(block, "incidenceAngle"))
        except ValueError:
            pt["incidence"] = None
        pts.append(pt)
        pos = j + len("</geolocationGridPoint>")
    if not pts:
        raise ValueError("annotation: no geolocationGridPoint entries")
    return pts


def select_cell(pts, lat, lon):
    """Cell selection of GeolocationGrid::interpolate_line_pixel (unchanged by
    fad0e46): the cell whose centre is nearest to the target, with the
    longitude difference scaled by cos(lat)."""
    lines = sorted({p["line"] for p in pts})
    pixels = sorted({p["pixel"] for p in pts})
    if len(lines) < 2 or len(pixels) < 2:
        return None
    by = {}
    for p in pts:
        by.setdefault((p["line"], p["pixel"]), p)

    best = None
    for li in range(len(lines) - 1):
        for pj in range(len(pixels) - 1):
            l0, l1 = lines[li], lines[li + 1]
            p0, p1 = pixels[pj], pixels[pj + 1]
            corners = (by.get((l0, p0)), by.get((l0, p1)),
                       by.get((l1, p0)), by.get((l1, p1)))
            if None in corners:
                return None
            c00, c10, c01, c11 = corners
            lat_c = (c00["lat"] + c01["lat"] + c10["lat"] + c11["lat"]) / 4.0
            lon_c = (c00["lon"] + c01["lon"] + c10["lon"] + c11["lon"]) / 4.0
            d_lat = lat_c - lat
            d_lon = (lon_c - lon) * math.cos(math.radians(lat))
            dist2 = d_lat * d_lat + d_lon * d_lon
            if best is None or dist2 < best[4]:
                best = (l0, l1, p0, p1, dist2)
    if best is None:
        return None
    l0, l1, p0, p1, _ = best
    corners = (by.get((l0, p0)), by.get((l0, p1)),
               by.get((l1, p0)), by.get((l1, p1)))
    if None in corners:
        return None
    c00, c10, c01, c11 = corners
    return (l0, l1, p0, p1, c00, c10, c01, c11)


def interpolate_line_pixel(pts, lat, lon):
    """Verbatim port of the CURRENT GeolocationGrid::interpolate_line_pixel
    (post-fad0e46): nearest-cell selection + bilinear Newton inversion."""
    cell = select_cell(pts, lat, lon)
    if cell is None:
        return None
    l0, l1, p0, p1, c00, c10, c01, c11 = cell

    line, pixel = 0.5 * (l0 + l1), 0.5 * (p0 + p1)
    for _ in range(10):
        u = (line - l0) / (l1 - l0)
        v = (pixel - p0) / (p1 - p0)
        glat = ((1 - u) * (1 - v) * c00["lat"] + (1 - u) * v * c10["lat"]
                + u * (1 - v) * c01["lat"] + u * v * c11["lat"])
        glon = ((1 - u) * (1 - v) * c00["lon"] + (1 - u) * v * c10["lon"]
                + u * (1 - v) * c01["lon"] + u * v * c11["lon"])
        dla_dl = ((1 - v) * c01["lat"] + v * c11["lat"]
                  - (1 - v) * c00["lat"] - v * c10["lat"]) / (l1 - l0)
        dla_dp = ((1 - u) * c10["lat"] + u * c11["lat"]
                  - (1 - u) * c00["lat"] - u * c01["lat"]) / (p1 - p0)
        dlo_dl = ((1 - v) * c01["lon"] + v * c11["lon"]
                  - (1 - v) * c00["lon"] - v * c10["lon"]) / (l1 - l0)
        dlo_dp = ((1 - u) * c10["lon"] + u * c11["lon"]
                  - (1 - u) * c00["lon"] - u * c01["lon"]) / (p1 - p0)
        det = dla_dl * dlo_dp - dla_dp * dlo_dl
        if abs(det) < 1e-15:
            break
        rla, rlo = lat - glat, lon - glon
        dl = (rla * dlo_dp - rlo * dla_dp) / det
        dp = (rlo * dla_dl - rla * dlo_dl) / det
        line += dl
        pixel += dp
        if abs(dl) < 1e-4 and abs(dp) < 1e-4:
            break
    return (min(max(line, l0), l1), min(max(pixel, p0), p1))


def interpolate_line_pixel_pre_fad0e46(pts, lat, lon):
    """Verbatim port of the anchor rule live until fad0e46 (2026-09-10), the
    revision that wrote the cached monthly_bursts bins: nearest-cell selection,
    then lat bounds taken from the c00/c01 edge and lon bounds from the c00/c10
    edge with u/v clamped to [0,1]."""
    cell = select_cell(pts, lat, lon)
    if cell is None:
        return None
    l0, l1, p0, p1, c00, c10, c01, c11 = cell

    lat_lo = min(c00["lat"], c01["lat"])
    lat_hi = max(c00["lat"], c01["lat"])
    lon_lo = min(c00["lon"], c10["lon"])
    lon_hi = max(c00["lon"], c10["lon"])

    if abs(lat_hi - lat_lo) < 1e-12:
        u = 0.0
    else:
        u = min(max((lat - lat_lo) / (lat_hi - lat_lo), 0.0), 1.0)
    if abs(lon_hi - lon_lo) < 1e-12:
        v = 0.0
    else:
        v = min(max((lon - lon_lo) / (lon_hi - lon_lo), 0.0), 1.0)

    return ((1.0 - u) * l0 + u * l1, (1.0 - v) * p0 + v * p1)


def _u16(b, o=0, little=True):
    return struct.unpack_from("<H" if little else ">H", b, o)[0]


def _u32(b, o=0, little=True):
    return struct.unpack_from("<I" if little else ">I", b, o)[0]


def rust_round(x):
    """f64::round(): half away from zero (Python's round() is banker's)."""
    return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)


def parse_tiff_header(s3, path):
    """Mirrors slc_decode.rs parse_tiff_header, over ranged reads."""
    size = s3.size_of(path)
    head = s3.get(path, 0, 7)
    little = head[:2] == b"II"
    if head[:2] not in (b"II", b"MM"):
        raise ValueError("TIFF: invalid byte order marker")
    if _u16(head, 2, little) != 42:
        raise ValueError("TIFF: invalid magic number")
    ifd = _u32(head, 4, little)
    if ifd + 2 > size:
        raise ValueError("TIFF: IFD offset out of bounds")
    n = _u16(s3.get(path, ifd, ifd + 1), 0, little)
    block = s3.get(path, ifd + 2, ifd + 2 + n * 12 - 1)

    def u16_arr(off, cnt):
        b = s3.get(path, off, off + cnt * 2 - 1)
        return [_u16(b, i * 2, little) for i in range(cnt)]

    def u32_arr(off, cnt):
        b = s3.get(path, off, off + cnt * 4 - 1)
        return [_u32(b, i * 4, little) for i in range(cnt)]

    width = length = bits = fmt = spp = None
    strip_offsets = strip_counts = None
    for i in range(n):
        o = i * 12
        if o + 12 > len(block):
            raise ValueError("TIFF: IFD entry out of bounds")
        tag = _u16(block, o, little)
        ftype = _u16(block, o + 2, little)
        count = _u32(block, o + 4, little)
        voff = _u32(block, o + 8, little)
        vfield = o + 8
        if tag == 256:
            width = ((_u16(block, vfield, little) if ftype == 3
                      else _u32(block, vfield, little)) if count == 1
                     else u32_arr(voff, count)[0])
        elif tag == 257:
            length = ((_u16(block, vfield, little) if ftype == 3
                       else _u32(block, vfield, little)) if count == 1
                      else u32_arr(voff, count)[0])
        elif tag == 258:
            bits = (_u16(block, vfield, little) if count == 1
                    else u16_arr(voff, 1)[0])
        elif tag == 273:
            strip_offsets = ([(_u16(block, vfield, little) if ftype == 3
                               else _u32(block, vfield, little))] if count == 1
                             else u32_arr(voff, count))
        elif tag == 277:
            spp = (_u16(block, vfield, little) if count == 1
                   else u16_arr(voff, 1)[0])
        elif tag == 279:
            strip_counts = ([(_u16(block, vfield, little) if ftype == 3
                              else _u32(block, vfield, little))] if count == 1
                            else u32_arr(voff, count))
        elif tag == 339:
            fmt = (_u16(block, vfield, little) if count == 1
                   else u16_arr(voff, 1)[0])

    if strip_offsets is None:
        raise ValueError("TIFF: StripOffsets tag not found")
    if strip_counts is None:
        raise ValueError("TIFF: StripByteCounts tag not found")
    if len(strip_offsets) != len(strip_counts):
        raise ValueError("TIFF: StripOffsets/StripByteCounts length mismatch")
    if width is None or length is None:
        raise ValueError("TIFF: ImageWidth/ImageLength tag not found")
    return {"size": size, "width": width, "height": length,
            "bits": bits if bits is not None else 16,
            "spp": spp if spp is not None else 2,
            "format": fmt if fmt is not None else 2,
            "strip_offsets": strip_offsets, "strip_counts": strip_counts}


def rect_origin(hdr, center_row, center_col, width, height):
    """Half-open [row_start, row_end) x [col_start, col_end) of the rectangle
    `extract_pixel_rect` cuts around a centre sample, clamped to the image. An
    overhanging centre collapses to an empty rectangle instead of inverting.
    Kept separate from the decoding so the CSV writer can name each sample's
    absolute (line, pixel) from the very same arithmetic."""
    half_w, half_h = width // 2, height // 2
    row_start = min(max(center_row - half_h, 0), hdr["height"])
    row_end = min(center_row + half_h + (height % 2), hdr["height"])
    col_start = min(max(center_col - half_w, 0), hdr["width"])
    col_end = min(center_col + half_w + (width % 2), hdr["width"])
    return (min(row_start, row_end), row_end,
            min(col_start, col_end), col_end)


def decode_rect(s3, path, hdr, center_row, center_col, width, height):
    """Verbatim port of extract_pixel_rect, over ranged reads."""
    is16 = hdr["bits"] == 16 and hdr["spp"] == 2
    is32 = hdr["bits"] == 32 and hdr["spp"] == 1
    if not (is16 or is32):
        raise ValueError(f"TIFF: unsupported SLC format (bits={hdr['bits']}, "
                         f"samples/pixel={hdr['spp']}, format={hdr['format']})")
    bpp = hdr["spp"] * (hdr["bits"] // 8)
    row_stride = hdr["width"] * bpp
    multi = len(hdr["strip_offsets"]) > 1

    row_start, row_end, col_start, col_end = rect_origin(
        hdr, center_row, center_col, width, height)

    out = []
    for row in range(row_start, row_end):
        si = min(row, len(hdr["strip_offsets"]) - 1) if multi else 0
        so, sc = hdr["strip_offsets"][si], hdr["strip_counts"][si]
        if so + sc > hdr["size"]:
            raise ValueError("TIFF: strip data out of bounds")
        # Offsets are relative to the strip start (Rust indexes the whole strip
        # slice); only the needed column span is fetched here.
        base = (col_start * bpp) if multi else (row * row_stride + col_start * bpp)
        end = (col_end * bpp) if multi else (row * row_stride + col_end * bpp)
        row_bytes = s3.get(path, so + base, so + end - 1)
        for col in range(col_start, col_end):
            strip_off = (col * bpp) if multi else (row * row_stride + col * bpp)
            if strip_off + bpp > sc:
                raise ValueError("TIFF: pixel data out of bounds")
            o = strip_off - base
            if is16:
                i, q = struct.unpack_from("<hh", row_bytes, o)
            else:
                packed = struct.unpack_from("<i", row_bytes, o)[0]
                i = struct.unpack("<h", struct.pack("<H", packed & 0xFFFF))[0]
                q = struct.unpack("<h",
                                  struct.pack("<H", (packed >> 16) & 0xFFFF))[0]
            out.append((float(i), float(q)))
    return out


def read_bin(path):
    b = open(path, "rb").read()
    w, h = struct.unpack_from("<II", b, 0)
    return w, h, [struct.unpack_from("<dd", b, 8 + 16 * i)
                  for i in range(w * h)]


def compare(label, mine, cache_path):
    w, h, ref = read_bin(cache_path)
    print(f"  {label}: decoded {len(mine)} vs cached {w}x{h}={len(ref)}")
    if len(mine) != len(ref):
        print("  !! length mismatch")
        return False
    bad, maxdiff = 0, 0.0
    for k, ((a, b), (c, d)) in enumerate(zip(mine, ref)):
        if a != c or b != d:
            bad += 1
            maxdiff = max(maxdiff, abs(a - c), abs(b - d))
            if bad <= 3:
                print(f"    idx {k}: mine ({a},{b}) cached ({c},{d})")
    print(f"  {label}: mismatches {bad}/{len(ref)}, max |diff| {maxdiff:g}")
    return bad == 0


def score(mine, ref):
    """Number of bit-identical (re, im) pairs."""
    return sum(1 for a, b in zip(mine, ref) if a == b)


def probe(meta, token, s3, product, root, ann_rel, tiff, hdr, grid, lat, lon):
    """Read-out that makes a clamped anchor visible rather than implied: the SAFE
    and burst record, the TIFF layout, and for both anchor rules the rounded
    centre, the nearest grid node and the ground error against (lat, lon)."""
    burst = odata(f"{CAT}/Bursts({meta['burst_id']})", token)
    print(f"SAFE {product}")
    print(f"  burst Lines={burst['Lines']} LinesPerBurst={burst['LinesPerBurst']} "
          f"SamplesPerBurst={burst['SamplesPerBurst']} "
          f"ByteOffset={burst['ByteOffset']} "
          f"orbit={burst.get('RelativeOrbitNumber')}/{burst.get('OrbitDirection')} "
          f"anx={burst.get('AzimuthAnxTime')}")
    print(f"  annotation {ann_rel}")
    print(f"  TIFF {hdr['width']}x{hdr['height']} bits={hdr['bits']} "
          f"spp={hdr['spp']} fmt={hdr['format']} "
          f"strips={len(hdr['strip_offsets'])} size={hdr['size']} "
          f"strip0={hdr['strip_offsets'][0]}")
    print(f"  grid points {len(grid)}; cached meta incidence "
          f"{meta['incidence_angle_deg']:.6f} deg")
    bbox, (near_m, near_pt) = grid_stats(grid, lat, lon)
    print(f"  grid bbox lat {bbox[0]:.4f}..{bbox[1]:.4f} "
          f"lon {bbox[2]:.4f}..{bbox[3]:.4f}; mast inside="
          f"{'yes' if tower_inside(bbox, lat, lon) else 'NO'}; nearest node "
          f"({near_pt['line']},{near_pt['pixel']}) {near_m / 1000:.1f} km away")
    cell = select_cell(grid, lat, lon)
    for rule, anchor in sorted(anchors(grid, lat, lon).items()):
        if anchor is None:
            print(f"  {rule}: the grid yields no anchor")
            continue
        near = min(grid, key=lambda p: (p["line"] - anchor[0]) ** 2
                   + (p["pixel"] - anchor[1]) ** 2)
        err = ground_error_m(cell, anchor[0], anchor[1], lat, lon)
        print(f"  {rule:8s}: (line,pixel)=({anchor[0]:.6f},{anchor[1]:.6f}) -> "
              f"centre ({rust_round(anchor[0])},{rust_round(anchor[1])}); "
              f"nearest node ({near['line']},{near['pixel']}) incidence "
              f"{near['incidence']}; ground error {err / 1000:.3f} km")
    if cell is not None:
        l0, l1, p0, p1, c00, c10, c01, c11 = cell
        print(f"  selected cell lines {l0}..{l1} pixels {p0}..{p1}")
        print(f"    (l0,p0)=({c00['lat']:.6f},{c00['lon']:.6f}) "
              f"(l1,p0)=({c01['lat']:.6f},{c01['lon']:.6f})")
        print(f"    (l0,p1)=({c10['lat']:.6f},{c10['lon']:.6f}) "
              f"(l1,p1)=({c11['lat']:.6f},{c11['lon']:.6f})")


def sample_rows(kind, decoded, cached, row_start, col_start, width):
    """One row per complex sample of a decoded rectangle, ready for the CSV.

    `i`/`q` use `%.17g`, which round-trips the f64 the cache stores for an int32
    sample exactly; `magnitude`/`phase_rad` are derived and rounded to 6
    decimals. `i_cached`/`q_cached` hold the cached pair for the same sample, so
    `equal` = 1 means this sample is bit-identical. `row`/`col` are relative to
    the rectangle (index = row * width + col, the order of the .bin file),
    `line`/`pixel` absolute in the burst."""
    rows = []
    for idx, ((i, q), (ci, cq)) in enumerate(zip(decoded, cached)):
        row, col = divmod(idx, width)
        rows.append({
            "kind": kind, "index": idx, "row": row, "col": col,
            "line": row_start + row, "pixel": col_start + col,
            "i": f"{i:.17g}", "q": f"{q:.17g}",
            "magnitude": f"{math.hypot(i, q):.6f}",
            "phase_rad": f"{math.atan2(q, i):.6f}",
            "i_cached": f"{ci:.17g}", "q_cached": f"{cq:.17g}",
            "equal": 1 if (i, q) == (ci, cq) else 0})
    return rows


def write_samples_csv(path, rows):
    """Write sample rows as CSV: LF line endings, no timestamp, rows in decode
    order, so two runs over the same entry `diff` cleanly."""
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(SAMPLE_COLUMNS),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def verify_entry(s3, meta, entry_dir, grid, tiff, hdr, lat, lon,
                 strip_lines=0, prefer=None, keep_samples=False):
    """Score both anchor rules of (lat, lon) against the cached window.bin.

    Returns `{"window_samples": n, "matched": [rules], "strip": ... or None,
    "rules": {rule: {anchor, center, exact[, samples]}}}`, `matched` listing the
    rules that reproduce every cached sample exactly. No I/Q swap is applied: a
    cache file whose halves were transposed would score 0 under both rules and be
    reported as unmatched rather than silently repaired.

    `strip_lines` > 0 also decodes the 400x11 strip at the matching anchor;
    `prefer` names the rule to take it from when both rules match (they can round
    to the same centre), `None` takes the first match. `keep_samples` fills in the
    CSV-ready sample rows of whichever rectangles were decoded.
    """
    _, _, w_ref = read_bin(os.path.join(entry_dir, "window.bin"))
    out = {"window_samples": len(w_ref), "matched": [], "strip": None, "rules": {}}
    for rule, anchor in anchors(grid, lat, lon).items():
        if anchor is None:
            out["rules"][rule] = {"anchor": None, "center": None, "exact": None}
            continue
        row, col = rust_round(anchor[0]), rust_round(anchor[1])
        win = decode_rect(s3, tiff, hdr, row, col, SEARCH_SIZE, SEARCH_SIZE)
        info = {"anchor": anchor, "center": (row, col), "decoded": win,
                "exact": score(win, w_ref)}
        if keep_samples:
            origin = rect_origin(hdr, row, col, SEARCH_SIZE, SEARCH_SIZE)
            info["samples"] = sample_rows("window", win, w_ref, origin[0],
                                          origin[2], SEARCH_SIZE)
        out["rules"][rule] = info
        if info["exact"] == len(w_ref):
            out["matched"].append(rule)
    if strip_lines and out["matched"]:
        rule = prefer if prefer in out["matched"] else out["matched"][0]
        row, col = out["rules"][rule]["center"]
        _, _, s_ref = read_bin(os.path.join(entry_dir, "strip.bin"))
        strip = decode_rect(s3, tiff, hdr, row, col, STRIP_WIDTH, STRIP_LINES)
        result = {"rule": rule, "samples": len(s_ref),
                  "exact": score(strip, s_ref)}
        if keep_samples:
            origin = rect_origin(hdr, row, col, STRIP_WIDTH, STRIP_LINES)
            result["sample_rows"] = sample_rows("strip", strip, s_ref, origin[0],
                                                origin[2], STRIP_WIDTH)
        out["strip"] = result
    return out


def _env_float(name, fallback):
    """A float from the environment, or `fallback` when unset or unparsable."""
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return fallback


# --- Fixtures for --selftest: no network, no cache, no credentials -----------
# A SAFE manifest as CDSE serves it, including the calibration and noise files
# that carry the same sub-swath and polarisation words as the annotation.
_MANIFEST_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<xfdu:XFDU xmlns:xfdu="urn:ccsds:schema:xfdu:1">
  <dataObjectSection>
    <dataObject id="annotation">
      <byteStream mimeType="text/xml" size="1">
        <fileLocation locatorType="URL"
            href="./annotation/s1a-iw1-slc-vv-x.xml"/>
      </byteStream>
    </dataObject>
    <dataObject id="calibration">
      <byteStream mimeType="text/xml" size="1">
        <fileLocation locatorType="URL"
            href="./annotation/calibration/calibration-s1a-iw1-slc-vv-x.xml"/>
      </byteStream>
    </dataObject>
    <dataObject id="noise">
      <byteStream mimeType="text/xml" size="1">
        <fileLocation locatorType="URL"
            href="./annotation/calibration/noise-s1a-iw1-slc-vv-x.xml"/>
      </byteStream>
    </dataObject>
    <dataObject id="measurement">
      <byteStream mimeType="application/octet-stream" size="1">
        <fileLocation locatorType="URL"
            href="./measurement/s1a-iw1-slc-vv-x.tiff"/>
      </byteStream>
    </dataObject>
  </dataObjectSection>
</xfdu:XFDU>
"""

# A 2x2 geolocation grid, skewed on purpose: no two corners share a latitude
# or a longitude, so the bilinear map is not separable and the two anchor rules
# really do disagree inside the cell.
_GRID_CORNERS = {(0, 0): (52.30, 9.60), (0, 1000): (52.31, 9.81),
                 (1000, 0): (52.52, 9.59), (1000, 1000): (52.50, 9.82)}


def _grid_fixture():
    """`_GRID_CORNERS` as a burst annotation, with the tags CDSE writes."""
    pts = []
    for (line, pixel), (lat, lon) in sorted(_GRID_CORNERS.items()):
        pts.append(f"""    <geolocationGridPoint>
      <azimuthTime>2021-04-03T05:51:31.000000</azimuthTime>
      <line>{line}</line>
      <pixel>{pixel}</pixel>
      <latitude>{lat:.8f}</latitude>
      <longitude>{lon:.8f}</longitude>
      <height>90.0</height>
      <incidenceAngle>38.9</incidenceAngle>
    </geolocationGridPoint>""")
    return ("<?xml version=\"1.0\"?>\n<product>\n  <geolocationGrid>\n"
            + "\n".join(pts) + "\n  </geolocationGrid>\n</product>\n")


class _BlobStore:
    """The two S3 calls the decoder makes, over one in-memory object."""

    def __init__(self, path, data):
        self.path, self.data = path, data

    def size_of(self, path):
        return len(self.data)

    def get(self, path, first=None, last=None):
        if first is None:
            return self.data
        return self.data[first:last + 1]


def _tiff_fixture():
    """A 4x4 single-band 32-bit complex TIFF, one strip per row, built the way
    the annotation files are laid out. Sample k (row-major) is (i, q) = (k+1,
    -k-1), packed as two int16 halves with i low."""
    w = h = 4
    ifd_at = 8
    arrays_at = ifd_at + 2 + 7 * 12 + 4
    entries = [(256, 3, 1, w), (257, 3, 1, h), (258, 3, 1, 32),
               (273, 4, h, arrays_at), (277, 3, 1, 1),
               (279, 4, h, arrays_at + 4 * h), (339, 3, 1, 2)]
    offsets = [arrays_at + 8 * h + r * 16 for r in range(h)]
    blob = bytearray(b"II" + struct.pack("<HI", 42, ifd_at))
    blob += struct.pack("<H", len(entries))
    for tag, ftype, count, value in entries:
        blob += struct.pack("<HHI", tag, ftype, count)
        blob += (struct.pack("<H", value) + b"\0\0" if count == 1
                 else struct.pack("<I", value))
    blob += struct.pack("<I", 0)                       # next IFD offset
    blob += struct.pack("<%dI" % h, *offsets)          # StripOffsets
    blob += struct.pack("<%dI" % h, *([16] * h))       # StripByteCounts
    assert len(blob) == offsets[0], (len(blob), offsets[0])
    for row in range(h):
        for col in range(w):
            k = row * w + col
            i, q = k + 1, -(k + 1)
            blob += struct.pack("<I", (i & 0xFFFF) | ((q & 0xFFFF) << 16))
    return bytes(blob)


def _tiff_store():
    """`(_BlobStore, blob)` over `_tiff_fixture()`."""
    blob = _tiff_fixture()
    return _BlobStore("synthetic.tiff", blob), blob


def _selftest_geometry():
    """The mirrored Rust arithmetic and the annotation parsing, all offline."""
    fails = []

    def check(name, got, want):
        if got != want:
            fails.append(f"{name}: got {got!r}, want {want!r}")

    def ok(name, cond):
        if not cond:
            fails.append(f"{name}: not satisfied")

    # f64::round() is half away from zero, unlike Python's round()
    check("rust_round(0.5)", rust_round(0.5), 1)
    check("rust_round(-0.5)", rust_round(-0.5), -1)
    check("rust_round(2.4)", rust_round(2.4), 2)

    # rectangle clamping, including a centre that overhangs the image
    hdr = {"width": 10, "height": 4}
    check("rect_origin inside the image", rect_origin(hdr, 1, 1, 3, 3),
          (0, 3, 0, 3))
    check("rect_origin at (0,0)", rect_origin(hdr, 0, 0, 3, 3), (0, 2, 0, 2))
    check("rect_origin at the far edge", rect_origin(hdr, 3, 9, 3, 3),
          (2, 4, 8, 10))
    check("rect_origin for a centre outside", rect_origin(hdr, 9, 9, 3, 3),
          (4, 4, 8, 10))

    # manifest: the calibration and noise XMLs carry the annotation's words too
    ann, meas = parse_manifest(_MANIFEST_FIXTURE)
    check("manifest annotation hrefs", ann,
          ["annotation/s1a-iw1-slc-vv-x.xml",
           "annotation/calibration/calibration-s1a-iw1-slc-vv-x.xml",
           "annotation/calibration/noise-s1a-iw1-slc-vv-x.xml"])
    check("manifest measurement hrefs", meas,
          ["measurement/s1a-iw1-slc-vv-x.tiff"])
    check("pick_annotation skips calibration and noise",
          pick_annotation(ann, "iw1", "vv"), "annotation/s1a-iw1-slc-vv-x.xml")

    # over a skewed cell, a coordinate built with the forward map must come back
    # from the Newton inversion - and must not from the legacy edge rule
    grid = parse_geolocation_grid(_grid_fixture())
    check("grid points", len(grid), 4)
    cell = select_cell(grid, 52.4288, 9.6828)
    ok("a grid cell was selected", cell is not None)
    if cell is None:
        return fails
    lat, lon = forward_latlon(cell, 600.0, 400.0)
    cur = interpolate_line_pixel(grid, lat, lon)
    old = interpolate_line_pixel_pre_fad0e46(grid, lat, lon)
    ok(f"current anchor ({cur[0]:.4f},{cur[1]:.4f}) inverts the forward map",
       abs(cur[0] - 600.0) < 0.01 and abs(cur[1] - 400.0) < 0.01)
    ok(f"legacy anchor ({old[0]:.4f},{old[1]:.4f}) drifts off it",
       abs(old[0] - 600.0) > 5.0 or abs(old[1] - 400.0) > 5.0)
    ok("current anchor ground error under 1 m",
       ground_error_m(cell, cur[0], cur[1], lat, lon) < 1.0)
    ok("legacy anchor ground error over 100 m",
       ground_error_m(cell, old[0], old[1], lat, lon) > 100.0)
    check("legacy clamps a longitude west of the cell",
          interpolate_line_pixel_pre_fad0e46(grid, 52.42, 9.0)[1], 0.0)
    check("legacy clamps a latitude north of the cell",
          interpolate_line_pixel_pre_fad0e46(grid, 53.0, 9.70)[0], 1000.0)

    # containment: the grid footprint and the distance to its nearest node are
    # what separate a burst that covers the mast from one that does not
    bbox, (near_m, near_pt) = grid_stats(grid, 52.4288, 9.6828)
    check("grid bbox", bbox, (52.30, 52.52, 9.59, 9.82))
    ok("the mast is inside the fixture grid", tower_inside(bbox, 52.4288, 9.6828))
    ok("a coordinate outside the fixture grid is not",
       not tower_inside(bbox, 53.0, 9.0))
    check("nearest fixture node", (near_pt["line"], near_pt["pixel"]), (1000, 0))
    ok("nearest fixture node distance in 11-13 km", 11000 < near_m < 13000)
    check("ground_distance_m for 1 degree of latitude",
          round(ground_distance_m(52.0, 9.0, 53.0, 9.0)), 111320)
    check("ground_distance_m for a point onto itself",
          ground_distance_m(52.0, 9.0, 52.0, 9.0), 0.0)
    return fails


def _selftest_io():
    """The ranged-read path, the cache readers and the CSV writer, over fixtures
    in a temporary directory that is removed again."""
    fails = []

    def check(name, got, want):
        if got != want:
            fails.append(f"{name}: got {got!r}, want {want!r}")

    def ok(name, cond):
        if not cond:
            fails.append(f"{name}: not satisfied")

    store, blob = _tiff_store()
    hdr = parse_tiff_header(store, store.path)
    check("fixture TIFF width/height", (hdr["width"], hdr["height"]), (4, 4))
    check("fixture TIFF bits/spp/format",
          (hdr["bits"], hdr["spp"], hdr["format"]), (32, 1, 2))
    check("fixture TIFF strips", len(hdr["strip_offsets"]), 4)
    check("fixture TIFF size", hdr["size"], len(blob))
    # centre (2,2) of size 2 covers rows 1..2 and columns 1..2, i.e. k = 5, 6,
    # 9, 10 - which is what the cached .bin order must agree with
    check("ranged decode of rows 1-2, cols 1-2",
          decode_rect(store, store.path, hdr, 2, 2, 2, 2),
          [(6.0, -6.0), (7.0, -7.0), (10.0, -10.0), (11.0, -11.0)])
    check("a rectangle wholly outside the image is empty",
          decode_rect(store, store.path, hdr, 99, 99, 2, 2), [])

    rows = sample_rows("window",
                       [(1.5, -2.5), (3.0, 4.0), (0.0, 0.0), (1e-7, 5.0)],
                       [(1.5, -2.5), (3.0, 4.0), (0.0, 1.0), (1e-7, 5.0)],
                       10, 20, 2)
    check("sample_rows row count", len(rows), 4)
    check("sample_rows absolute line/pixel",
          [(r["line"], r["pixel"]) for r in rows],
          [(10, 20), (10, 21), (11, 20), (11, 21)])
    check("sample_rows equal flags", [r["equal"] for r in rows], [1, 1, 0, 1])
    check("sample_rows i/q round-trip exactly",
          (float(rows[0]["i"]), float(rows[0]["q"])), (1.5, -2.5))
    check("sample_rows magnitude of (1.5,-2.5)", rows[0]["magnitude"], "2.915476")
    check("sample_rows phase of (0,0)", rows[2]["phase_rad"], "0.000000")

    # credentials: the KeyError has to name every missing secret, and an
    # unreadable --env-file has to surface as OSError (exit code 1 either way)
    saved = {k: os.environ.pop(k, None) for k in CRED_KEYS}
    try:
        credentials()
        fails.append("credentials: missing secrets did not raise KeyError")
    except KeyError as exc:
        check("credentials names the missing keys", sorted(exc.args[0]),
              sorted(CRED_KEYS))
    finally:
        for key, val in saved.items():
            if val is not None:
                os.environ[key] = val

    with tempfile.TemporaryDirectory() as tmp:
        try:
            credentials(os.path.join(tmp, "absent.env"))
            fails.append("credentials: an unreadable file did not raise OSError")
        except OSError:
            pass

        cache = os.path.join(tmp, "cache")
        entry_dir = os.path.join(cache, "2021-04", "2021-04-03_1e17aa73")
        os.makedirs(entry_dir)
        with open(os.path.join(entry_dir, "meta.json"), "w") as fh:
            fh.write('{"burst_id": 12345}')
        month, name, _, meta = load_entry(cache, "2021-04/2021-04-03_1e17aa73")
        check("load_entry month/name", (month, name),
              ("2021-04", "2021-04-03_1e17aa73"))
        check("load_entry meta", meta, {"burst_id": 12345})

        bin_path = os.path.join(entry_dir, "window.bin")
        with open(bin_path, "wb") as fh:
            fh.write(struct.pack("<IIdd", 2, 1, 1.0, -2.0))
            fh.write(struct.pack("<dd", 3.0, 4.0))
        check("read_bin", read_bin(bin_path), (2, 1, [(1.0, -2.0), (3.0, 4.0)]))

        csv_path = os.path.join(entry_dir, "samples.csv")
        check("write_samples_csv row count", write_samples_csv(csv_path, rows), 4)
        with open(csv_path, "rb") as fh:
            text = fh.read()
        lines = text.decode("utf-8").split("\n")
        check("csv header", lines[0], ",".join(SAMPLE_COLUMNS))
        check("csv body lines", len(lines[1:-1]), 4)
        check("csv fields per line", len(lines[1].split(",")), len(SAMPLE_COLUMNS))
        ok("csv uses LF line endings only", b"\r" not in text)
        ok("csv ends with a newline", text.endswith(b"\n"))
    return fails


def _selftest():
    """Every offline check; returns the failures, empty when all passed."""
    return _selftest_geometry() + _selftest_io()


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog=os.path.basename(__file__),
        description="Reproduce one cached LUMO burst window/strip from the "
                    "Sentinel-1 SLC on CDSE with ranged reads only, and score "
                    "both the pre- and the post-fad0e46 anchor rule against the "
                    "cached binaries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="exit codes: 0 ok, 1 unusable --cache-dir/--env-file or missing "
               "credentials, 2 cache entry or measurement TIFF unusable, "
               "3 verification mismatch, 4 selftest failure")
    ap.add_argument("--cache-dir", metavar="DIR",
                    default=os.environ.get(CACHE_VAR),
                    help="cache written by the Rust pipeline, one "
                         "<YYYY-MM>/<entry>/ sub-directory each "
                         "(default: $%s)" % CACHE_VAR)
    ap.add_argument("--entry", metavar="MONTH/NAME",
                    help="cache entry to read, e.g. 2021-04/2021-04-03_1e17aa73")
    ap.add_argument("--env-file", metavar="FILE",
                    default=os.environ.get(ENV_FILE_VAR),
                    help="key=value file holding the CDSE secrets; the process "
                         "environment wins (default: $%s)" % ENV_FILE_VAR)
    ap.add_argument("--lat", type=float, default=_env_float(LAT_VAR, MAST_LAT),
                    help="latitude of the mast (default: %(default)s)")
    ap.add_argument("--lon", type=float, default=_env_float(LON_VAR, MAST_LON),
                    help="longitude of the mast (default: %(default)s)")
    ap.add_argument("--verify", action="store_true",
                    help="decode both anchor rules and compare the result with "
                         "the cached window.bin (the default action)")
    ap.add_argument("--probe", action="store_true",
                    help="print the SAFE, grid, TIFF and anchor diagnostics")
    ap.add_argument("--strip", dest="strip_lines", type=int, default=0,
                    metavar="N",
                    help="also decode the 400x11 strip at the matching anchor")
    ap.add_argument("--samples-csv", metavar="FILE",
                    help="write the decoded window/strip samples of the matching "
                         "anchor to FILE (requires --verify)")
    ap.add_argument("--selftest", action="store_true",
                    help="check the arithmetic and the ranged-read path on "
                         "fixtures - no network, no cache, no credentials")
    args = ap.parse_args(argv)

    if args.selftest:
        print("lumo_slc_ranged_read selftest - no network, no cache, "
              "no credentials")
        fails = _selftest()
        if fails:
            for line in fails:
                print(f"  !! {line}", file=sys.stderr)
            print(f"  {len(fails)} check(s) FAILED", file=sys.stderr)
            return 4
        print("  all checks passed")
        return 0

    if not args.cache_dir:
        print(f"error: no cache directory - pass --cache-dir or set "
              f"${CACHE_VAR}", file=sys.stderr)
        return 1
    if not args.entry:
        print("error: --entry MONTH/NAME is required", file=sys.stderr)
        return 1

    try:
        month, name, entry_dir, meta = load_entry(args.cache_dir, args.entry)
    except (OSError, ValueError) as exc:
        print(f"error: cannot read cache entry {args.entry}: {exc}",
              file=sys.stderr)
        return 2
    try:
        creds = credentials(args.env_file)
    except KeyError as exc:
        print(f"error: credentials {exc.args[0]} are in neither the environment "
              f"nor {args.env_file or '$' + ENV_FILE_VAR}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: cannot read --env-file: {exc}", file=sys.stderr)
        return 1

    print(f"entry {month}/{name} - burst {meta['burst_id']} "
          f"{meta['subswath']}/{meta['polarisation']} {meta['acquisition_ts']}")
    token = fetch_token(creds["COPERNICUS_CLIENT_ID"],
                        creds["COPERNICUS_CLIENT_SECRET"])
    s3 = S3(creds["AWS_ACCESS_KEY_ID"], creds["AWS_SECRET_ACCESS_KEY"])
    try:
        product, root, ann_rel, meas_rel = burst_product(meta, token, s3)
        tiff = f"{root}/{meas_rel}"
        _, grid = annotation_grid(s3, root, ann_rel)
        hdr = parse_tiff_header(s3, tiff)
    except Exception as exc:
        print(f"error: cannot resolve the SLC of burst {meta['burst_id']}: {exc}",
              file=sys.stderr)
        return 2

    if args.probe:
        probe(meta, token, s3, product, root, ann_rel, tiff, hdr, grid,
              args.lat, args.lon)

    do_verify = args.verify or not args.probe
    if args.samples_csv and not do_verify:
        print("error: --samples-csv needs --verify", file=sys.stderr)
        return 1
    if not do_verify:
        return 0

    res = verify_entry(s3, meta, entry_dir, grid, tiff, hdr, args.lat, args.lon,
                       strip_lines=args.strip_lines,
                       keep_samples=bool(args.samples_csv))
    for rule in ("legacy", "current"):
        info = res["rules"][rule]
        if info["center"] is None:
            print(f"  {rule:7s}: the grid yields no anchor")
            continue
        print(f"  {rule:7s}: (line,pixel)=({info['anchor'][0]:.6f},"
              f"{info['anchor'][1]:.6f}) -> centre {info['center']} -> "
              f"{info['exact']}/{res['window_samples']} window samples bit-exact")
        if not info["exact"]:
            compare(rule, info["decoded"], os.path.join(entry_dir, "window.bin"))
    if res["strip"] is not None:
        print(f"  strip at the {res['strip']['rule']} anchor: "
              f"{res['strip']['exact']}/{res['strip']['samples']} samples "
              f"bit-exact")
    print("RESULT:", "IDENTICAL" if res["matched"] else "MISMATCH",
          f"(window.bin reproduced exactly by: "
          f"{', '.join(res['matched']) or 'no rule'})")

    if args.samples_csv:
        if not res["matched"]:
            print("error: no anchor reproduced window.bin, so there is no "
                  "matched sample set to write", file=sys.stderr)
            return 3
        rows = list(res["rules"][res["matched"][0]]["samples"])
        if res["strip"] is not None:
            rows += res["strip"]["sample_rows"]
        written = write_samples_csv(args.samples_csv, rows)
        print(f"  wrote {written} sample rows for the {res['matched'][0]} "
              f"anchor to {args.samples_csv}")
    return 0 if res["matched"] else 3


if __name__ == "__main__":
    sys.exit(main())




