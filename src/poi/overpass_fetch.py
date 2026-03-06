import threading
import queue
import random
import pandas as pd
import overpy
import time
import socket
import config
from pathlib import Path
from typing import Optional, List, Dict

import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union

from screens.shared.osm_extract_common import POINT_LAYER_NAMES, LINE_LAYER_NAMES

from .utils import clean_name, norm_str, parse_int_tag
from .filters import (
    is_excluded_park,
    is_excluded_golf_course,
    is_non_building_museum,
    is_private_hospital,
    is_excluded_hospital,
    merge_nearby_hospitals,
)

# ============================================================
# Timeout / error helpers
# ============================================================
def _is_timeout_error(e: Exception) -> bool:
    s = str(e).lower()
    return (
        isinstance(e, TimeoutError)
        or isinstance(e, socket.timeout)
        or "timed out" in s
        or "10060" in s
    )

def _is_overload_error(e: Exception) -> bool:
    s = str(e).lower()
    return (
        "server load too high" in s
        or "too busy" in s
        or "rate limit" in s
        or "429" in s
    )

def _is_blocked_or_bad_endpoint(e: Exception) -> bool:
    s = str(e).lower()
    return (
        "status code: 403" in s
        or "status code: 405" in s
        or " 403" in s
        or " 405" in s
        or "forbidden" in s
        or "method not allowed" in s
    )

def run_with_timeout(func, timeout=12):
    q = queue.Queue()

    def wrapper():
        try:
            q.put(func())
        except Exception as e:
            q.put(e)

    t = threading.Thread(target=wrapper, daemon=True)
    t.start()

    try:
        result = q.get(timeout=timeout)
    except queue.Empty:
        raise TimeoutError("Overpass request timed out")

    if isinstance(result, Exception):
        raise result

    return result


# ============================================================
# AOI / coverage helpers
# ============================================================
def area_clause_from_config():
    poly = getattr(config, "overpass_poly", None)
    if poly:
        return f'(poly:"{poly}")'

    bb = getattr(config, "bound_box", None) or getattr(config, "saved_bound_box", None)
    if bb:
        south, west, north, east = bb
        return f"({south},{west},{north},{east})"

    return None


def _aoi_geom_from_config():
    poly = getattr(config, "overpass_poly", None)
    if poly:
        parts = [p for p in str(poly).replace(",", " ").split() if p]
        if len(parts) >= 6 and len(parts) % 2 == 0:
            coords = []
            for i in range(0, len(parts), 2):
                lat = float(parts[i])
                lon = float(parts[i + 1])
                coords.append((lon, lat))
            if coords and coords[0] != coords[-1]:
                coords.append(coords[0])
            return Polygon(coords)

    bb = getattr(config, "bound_box", None) or getattr(config, "saved_bound_box", None)
    if bb:
        south, west, north, east = bb
        return box(west, south, east, north)

    return None


def _polygon_to_overpass_poly(p: Polygon) -> str:
    coords = list(p.exterior.coords)
    parts = [f"{y} {x}" for (x, y) in coords]
    return f'(poly:"{" ".join(parts)}")'


def _missing_area_clauses(missing) -> List[str]:
    if missing is None or missing.is_empty:
        return []
    if isinstance(missing, Polygon):
        return [_polygon_to_overpass_poly(missing)]
    if isinstance(missing, MultiPolygon):
        return [_polygon_to_overpass_poly(p) for p in missing.geoms]
    return []


# ============================================================
# Coverage sliver-tolerant missing computation
# ============================================================
def _compute_missing_with_tolerance(aoi, coverage, buffer_m=300, min_missing_km2=1.0, min_missing_ratio=0.005):
    if coverage is None or coverage.is_empty:
        return aoi

    try:
        aoi_m = gpd.GeoSeries([aoi], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
        cov_m = gpd.GeoSeries([coverage], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
    except Exception:
        m = aoi.difference(coverage)
        return None if m.is_empty else m

    cov_buf = cov_m.buffer(buffer_m)
    missing_m = aoi_m.difference(cov_buf)
    if missing_m.is_empty:
        return None

    aoi_area = float(aoi_m.area) if aoi_m.area else 0.0
    miss_area = float(missing_m.area) if missing_m.area else 0.0

    miss_km2 = miss_area / 1_000_000.0
    ratio = (miss_area / aoi_area) if aoi_area > 0 else 1.0

    if miss_km2 < float(min_missing_km2) or ratio < float(min_missing_ratio):
        return None

    try:
        return gpd.GeoSeries([missing_m], crs="EPSG:3857").to_crs("EPSG:4326").iloc[0]
    except Exception:
        m = aoi.difference(coverage)
        return None if m.is_empty else m


def _local_base_dir() -> Path:
    base = getattr(config, "LOCAL_DATA_DIR", None)
    if base:
        return Path(base)
    return Path("local_data_outputs")


def _datasets_intersecting_aoi(aoi) -> List[Path]:
    base = _local_base_dir()
    if not base.exists():
        return []

    out_dirs: List[Path] = []
    for cov_path in base.rglob("coverage.geojson"):
        try:
            gdf = gpd.read_file(cov_path)
            if gdf.empty:
                continue
            cov_geom = unary_union([g for g in gdf.geometry if g is not None])
            if cov_geom is None or cov_geom.is_empty:
                continue
            if cov_geom.intersects(aoi):
                out_dirs.append(cov_path.parent)
        except Exception:
            continue
    return out_dirs


def _coverage_union(out_dirs: List[Path]):
    geoms = []
    for d in out_dirs:
        cov = d / "coverage.geojson"
        try:
            gdf = gpd.read_file(cov)
            for g in gdf.geometry:
                if g is not None:
                    geoms.append(g)
        except Exception:
            pass
    return unary_union(geoms) if geoms else None


def _pick_gpkg(out_dir: Path) -> Optional[Path]:
    clean = out_dir / "layers_clean.gpkg"
    raw = out_dir / "layers.gpkg"
    if clean.exists():
        return clean
    if raw.exists():
        return raw
    return None


# ============================================================
# Local layer mapping (exact names from extractor)
# ============================================================
POI_TYPE_TO_LAYER: Dict[str, str] = {
    "Park": "poi_parks",
    "Mountain": "poi_mountains",
    "Hospital": "poi_hospitals",
    "Foreign mission": "poi_foreign_missions",
    "Cinema": "poi_cinemas",
    "Body of water": "poi_bodies_of_water",
    "Amusement park": "poi_amusement_parks",
    "Aquarium": "poi_aquariums",
    "Library": "poi_libraries",
    "Golf course": "poi_golf_courses",
    "Museum": "poi_museums",
}

LINE_TYPE_TO_LAYER: Dict[str, str] = {
    "Coastline": "lines_coastline",
    "Rivers": "lines_rivers",
    "Canals": "lines_canals",
    "Streams": "lines_streams",
}


def _local_fetch_points(gpkg: Path, layer: str, aoi) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = aoi.bounds
    try:
        gdf = gpd.read_file(gpkg, layer=layer, bbox=(minx, miny, maxx, maxy))
    except Exception:
        return gpd.GeoDataFrame(columns=["name", "geometry"])

    if gdf is None or gdf.empty or "geometry" not in gdf:
        return gpd.GeoDataFrame(columns=["name", "geometry"])

    try:
        gdf = gdf[gdf.geometry.intersects(aoi)]
    except Exception:
        pass

    return gdf


def _local_fetch_lines(gpkg: Path, layer: str, aoi) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = aoi.bounds
    try:
        gdf = gpd.read_file(gpkg, layer=layer, bbox=(minx, miny, maxx, maxy))
    except Exception:
        return gpd.GeoDataFrame(columns=["name", "geometry"])

    if gdf is None or gdf.empty or "geometry" not in gdf:
        return gpd.GeoDataFrame(columns=["name", "geometry"])

    try:
        gdf = gdf[gdf.geometry.intersects(aoi)]
    except Exception:
        pass

    return gdf


def _linestring_to_latlon_list(geom) -> Optional[list]:
    if geom is None:
        return None
    try:
        coords = list(geom.coords)
        pts = [(float(y), float(x)) for (x, y) in coords]  # (lon,lat)->(lat,lon)
        return pts if len(pts) >= 2 else None
    except Exception:
        return None


# ============================================================
# Main fetch (coverage-aware)
# ============================================================
def fetch_pois(osm_filter, type_name: str, status_label):
    aoi = _aoi_geom_from_config()
    area_clause = area_clause_from_config()
    if not area_clause or aoi is None:
        status_label.config(text="No area set. Go back and set a boundary first.")
        return None

    mirrors = list(getattr(config, "overpass_mirrors", []))
    if not mirrors:
        status_label.config(text="No Overpass mirrors configured.")
        return None

    random.shuffle(mirrors)

    def short_host(url: str) -> str:
        try:
            return url.split("/")[2] if "://" in url else url
        except Exception:
            return url

    type_key = " ".join(str(type_name).split()).lower()

    # Determine local coverage + missing polygons (SLIVER-TOLERANT)
    dirs = _datasets_intersecting_aoi(aoi)
    cov_union = _coverage_union(dirs) if dirs else None

    buffer_m = getattr(config, "COVERAGE_BUFFER_M", 300)
    min_km2 = getattr(config, "COVERAGE_MIN_MISSING_KM2", 1.0)
    min_ratio = getattr(config, "COVERAGE_MIN_MISSING_RATIO", 0.005)

    missing = _compute_missing_with_tolerance(
        aoi, cov_union,
        buffer_m=buffer_m,
        min_missing_km2=min_km2,
        min_missing_ratio=min_ratio,
    )
    missing_clauses = _missing_area_clauses(missing)

    # -------------------------
    # Local fetch
    # -------------------------
    local_df = None

    # Coastline (lines)
    if type_key == "coastline" and dirs:
        parts = []
        for d in dirs:
            gpkg = _pick_gpkg(d)
            if not gpkg:
                continue
            layer = "lines_coastline"
            if layer not in LINE_LAYER_NAMES:
                continue

            gdf = _local_fetch_lines(gpkg, layer, aoi)
            if gdf.empty:
                continue

            rows = []
            for _, row in gdf.iterrows():
                pts = _linestring_to_latlon_list(row.get("geometry"))
                if not pts:
                    continue
                rows.append({
                    "Name": "",
                    "Type": "Coastline",
                    "Kind": "coastline",
                    "Geometry": pts,
                })
            if rows:
                parts.append(pd.DataFrame(rows))

        if parts:
            local_df = pd.concat(parts, ignore_index=True)
            status_label.config(text=f"Local: {len(local_df)} coastline segments.")
        else:
            status_label.config(text="Local: 0 coastline segments.")

    # Body of water (points + lines)
    is_water = (type_key == "body of water")
    if not is_water:
        f_joined = " ".join(map(str, osm_filter if isinstance(osm_filter, (list, tuple)) else [osm_filter])).lower()
        if "waterway=" in f_joined or "natural=water" in f_joined or "water=" in f_joined:
            is_water = True

    if is_water and dirs:
        # points from poi_bodies_of_water
        point_parts = []
        for d in dirs:
            gpkg = _pick_gpkg(d)
            if not gpkg:
                continue
            layer = "poi_bodies_of_water"
            if layer not in POINT_LAYER_NAMES:
                continue

            gdf = _local_fetch_points(gpkg, layer, aoi)
            if gdf.empty:
                continue

            cent = gdf.geometry.centroid
            rows = []
            for i, row in gdf.iterrows():
                name = clean_name(str(row.get("name") or "").strip())
                if not name:
                    continue

                natural = norm_str(row.get("natural"))
                water = norm_str(row.get("water"))
                landuse = norm_str(row.get("landuse"))

                kind = "water"
                if landuse == "reservoir" or water == "reservoir":
                    kind = "reservoir"
                elif water in ("lake", "pond"):
                    kind = water
                elif natural == "water":
                    kind = water or "water"

                rows.append({
                    "Name": name,
                    "Type": "Body of water",
                    "Kind": kind,
                    "Latitude": float(cent.loc[i].y),
                    "Longitude": float(cent.loc[i].x),
                })
            if rows:
                point_parts.append(pd.DataFrame(rows))

        df_points = pd.concat(point_parts, ignore_index=True) if point_parts else pd.DataFrame()

        # lines from rivers/canals/streams
        line_parts = []
        for d in dirs:
            gpkg = _pick_gpkg(d)
            if not gpkg:
                continue

            for layer in ("lines_rivers", "lines_canals", "lines_streams"):
                if layer not in LINE_LAYER_NAMES:
                    continue
                gdf = _local_fetch_lines(gpkg, layer, aoi)
                if gdf.empty:
                    continue

                rows = []
                kind = layer.replace("lines_", "")
                for _, row in gdf.iterrows():
                    pts = _linestring_to_latlon_list(row.get("geometry"))
                    if not pts:
                        continue
                    name = clean_name(str(row.get("name") or "").strip())
                    if not name:
                        continue
                    rows.append({
                        "Name": name,
                        "Type": "Body of water",
                        "Kind": kind[:-1] if kind.endswith("s") else kind,
                        "Geometry": pts,
                    })
                if rows:
                    line_parts.append(pd.DataFrame(rows))

        df_lines = pd.concat(line_parts, ignore_index=True) if line_parts else pd.DataFrame()

        if df_points.empty and df_lines.empty:
            local_df = None
            status_label.config(text="Local: 0 water features.")
        else:
            if df_points.empty:
                local_df = df_lines
            elif df_lines.empty:
                local_df = df_points
            else:
                local_df = pd.concat([df_points, df_lines], ignore_index=True)
            status_label.config(text=f"Local: {len(local_df)} water features.")

    # Generic POI points
    if local_df is None and dirs:
        tn = " ".join(str(type_name).split()).strip()
        layer = None
        for k, v in POI_TYPE_TO_LAYER.items():
            if k.lower() == tn.lower():
                layer = v
                break

        if layer and layer in POINT_LAYER_NAMES:
            parts = []
            for d in dirs:
                gpkg = _pick_gpkg(d)
                if not gpkg:
                    continue
                gdf = _local_fetch_points(gpkg, layer, aoi)
                if gdf.empty:
                    continue

                cent = gdf.geometry.centroid
                rows = []

                for i, row in gdf.iterrows():
                    name = clean_name(str(row.get("name") or "").strip())
                    if not name:
                        continue

                    tags = row

                    if type_key == "park" and is_excluded_park(tags, name):
                        continue
                    if type_key == "golf course" and is_excluded_golf_course(tags, name):
                        continue
                    if type_key == "hospital":
                        if is_private_hospital(tags) or is_excluded_hospital(tags, name):
                            continue
                        beds = parse_int_tag(tags, "beds") or parse_int_tag(tags, "capacity")
                        rows.append({
                            "Name": name,
                            "Type": type_name,
                            "Latitude": float(cent.loc[i].y),
                            "Longitude": float(cent.loc[i].x),
                            "Beds": beds,
                        })
                    else:
                        rows.append({
                            "Name": name,
                            "Type": type_name,
                            "Latitude": float(cent.loc[i].y),
                            "Longitude": float(cent.loc[i].x),
                        })

                if rows:
                    parts.append(pd.DataFrame(rows))

            if parts:
                local_df = pd.concat(parts, ignore_index=True)
                if type_key == "hospital":
                    before = len(local_df)
                    local_df = merge_nearby_hospitals(local_df, radius_m=500.0)
                    merged = before - len(local_df)
                    if merged > 0:
                        status_label.config(text=f"Local: {len(local_df)} hospitals (merged {merged} nearby).")
                    else:
                        status_label.config(text=f"Local: {len(local_df)} hospitals.")
                else:
                    status_label.config(text=f"Local: {len(local_df)} {type_name}.")
            else:
                status_label.config(text=f"Local: 0 {type_name}.")

    # Fully covered -> skip overpass
    if dirs and not missing_clauses:
        status_label.config(text="Coverage: AOI treated as covered — skipping Overpass.")
        if local_df is not None and not local_df.empty:
            return local_df
        return None

    # Partial coverage -> overpass only for missing polygons
    if dirs and missing_clauses:
        if len(missing_clauses) > 8:
            status_label.config(text=f"Coverage partial ({len(missing_clauses)} parts) — using one Overpass call.")
            over_df = _fetch_pois_overpass(osm_filter, type_name, status_label, mirrors, short_host, area_clause)
            return _merge_local_overpass(local_df, over_df)

        status_label.config(text=f"Coverage partial — fetching missing area ({len(missing_clauses)} parts)…")
        over_parts = []
        for c in missing_clauses:
            df = _fetch_pois_overpass(osm_filter, type_name, status_label, mirrors, short_host, c)
            if df is not None and not df.empty:
                over_parts.append(df)
        over_df = pd.concat(over_parts, ignore_index=True) if over_parts else None
        return _merge_local_overpass(local_df, over_df)

    # No intersecting local coverage -> original overpass behaviour
    return _fetch_pois_overpass(osm_filter, type_name, status_label, mirrors, short_host, area_clause)


def _merge_local_overpass(local_df: Optional[pd.DataFrame], over_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if local_df is None or local_df.empty:
        return over_df if over_df is not None and not over_df.empty else None
    if over_df is None or over_df.empty:
        return local_df

    merged = pd.concat([local_df, over_df], ignore_index=True)

    if "Latitude" in merged.columns and "Longitude" in merged.columns:
        merged["_k"] = (
            merged.get("Latitude", pd.Series()).round(4).astype(str).fillna("")
            + "|"
            + merged.get("Longitude", pd.Series()).round(4).astype(str).fillna("")
            + "|"
            + merged.get("Name", pd.Series()).astype(str).fillna("")
            + "|"
            + merged.get("Type", pd.Series()).astype(str).fillna("")
            + "|"
            + merged.get("Kind", pd.Series()).astype(str).fillna("")
        )
    else:
        merged["_k"] = (
            merged.get("Name", pd.Series()).astype(str).fillna("")
            + "|"
            + merged.get("Type", pd.Series()).astype(str).fillna("")
            + "|"
            + merged.get("Kind", pd.Series()).astype(str).fillna("")
            + "|"
            + merged.get("Geometry", pd.Series()).astype(str).fillna("")
        )

    merged = merged.drop_duplicates("_k").drop(columns=["_k"]).reset_index(drop=True)
    return merged


# ============================================================
# Overpass portion (mostly original logic)
# ============================================================
def _fetch_pois_overpass(osm_filter, type_name: str, status_label, mirrors, short_host, area_clause: str):
    filters = osm_filter if isinstance(osm_filter, (list, tuple)) else [osm_filter]
    type_key = " ".join(str(type_name).split()).lower()

    is_water = (type_key == "body of water")
    if not is_water:
        f_joined = " ".join(map(str, filters)).lower()
        if "waterway=" in f_joined or "natural=water" in f_joined or "water=" in f_joined:
            is_water = True

    if is_water:
        return fetch_body_of_water(area_clause, status_label, mirrors, short_host)

    if type_key == "coastline":
        return fetch_coastline_lines(area_clause, status_label, mirrors, short_host)

    blocks = []
    for f in filters:
        blocks.append(f'node[{f}]{area_clause};')
        blocks.append(f'way[{f}]{area_clause};')
        blocks.append(f'relation[{f}]{area_clause};')

    query = f"""
    [out:json][timeout:50];
    (
      {"".join(blocks)}
    );
    out center;
    """

    for url in mirrors:
        try:
            status_label.config(text=f"Trying {short_host(url)}...")
            status_label.update_idletasks()

            api = overpy.Overpass(url=url)
            result = run_with_timeout(lambda: api.query(query), timeout=12)

            rows = []

            def maybe_add(name, tags, lat, lon):
                name = clean_name(name)

                if not name and type_key == "cinema":
                    name = clean_name(
                        tags.get("brand")
                        or tags.get("operator")
                        or tags.get("short_name")
                        or tags.get("name:en")
                        or tags.get("ref")
                    )
                    if not name:
                        name = "Cinema (unnamed)"

                if not name:
                    return

                name_l = norm_str(name)

                if type_key == "foreign mission":
                    if "residence of" in name_l or "ambassador's residence" in name_l:
                        return
                    BAD_KEYWORDS = (
                        "consular section", "consular department", "consulate general",
                        "consulate of", "visa office", "passport", "trade", "commercial",
                        "defence", "defense", "military", "attache", "education section",
                        "cultural", "medical office", "student department",
                        "science & technology", "naval", "delegation of",
                    )
                    if any(k in name_l for k in BAD_KEYWORDS):
                        return
                    ALLOWED_KEYWORDS = (
                        "embassy of", "high commission of", "royal embassy",
                        "delegation of the european union",
                    )
                    if not any(k in name_l for k in ALLOWED_KEYWORDS):
                        return

                if "house of " in name_l and name_l not in ("house of commons", "house of lords"):
                    return
                if "official residence of" in name_l:
                    return

                if type_key == "park" and is_excluded_park(tags, name):
                    return
                if type_key == "golf course" and is_excluded_golf_course(tags, name):
                    return
                if type_key == "museum":
                    if not norm_str(tags.get("building")) and not norm_str(tags.get("building:part")):
                        return
                    if is_non_building_museum(tags, name):
                        return
                if type_key == "hospital":
                    if is_private_hospital(tags):
                        return
                    if is_excluded_hospital(tags, name):
                        return
                    beds = parse_int_tag(tags, "beds") or parse_int_tag(tags, "capacity")
                    rows.append({
                        "Name": name,
                        "Type": type_name,
                        "Latitude": float(lat),
                        "Longitude": float(lon),
                        "Beds": beds,
                    })
                    return

                rows.append({
                    "Name": name,
                    "Type": type_name,
                    "Latitude": float(lat),
                    "Longitude": float(lon),
                })

            for n in result.nodes:
                maybe_add(n.tags.get("name"), n.tags, n.lat, n.lon)

            for w in result.ways:
                lat = getattr(w, "center_lat", None)
                lon = getattr(w, "center_lon", None)
                if lat is None or lon is None:
                    continue
                maybe_add(w.tags.get("name"), w.tags, lat, lon)

            for r in result.relations:
                lat = getattr(r, "center_lat", None)
                lon = getattr(r, "center_lon", None)
                if lat is None or lon is None:
                    continue
                maybe_add(r.tags.get("name"), r.tags, lat, lon)

            if not rows:
                status_label.config(text=f"No named {type_name} found.")
                return None

            df = pd.DataFrame(rows)

            if type_key == "hospital":
                before = len(df)
                df = merge_nearby_hospitals(df, radius_m=500.0)
                merged_n = before - len(df)
                if merged_n > 0:
                    status_label.config(text=f"Fetched {len(df)} hospitals (merged {merged_n} nearby).")

            if "Name" in df.columns and not df.empty:
                df = df[df["Name"].astype(str).str.strip().ne("")]
                df = df[df["Name"].astype(str).str.lower().ne("unnamed")]

            status_label.config(text=f"Fetched {len(df)} named {type_name}.")
            return df

        except TimeoutError:
            status_label.config(text=f"Timeout on {short_host(url)}")
            status_label.update_idletasks()
        except Exception as e:
            status_label.config(text=f"Error on {short_host(url)}: {e}")
            status_label.update_idletasks()

    status_label.config(text=f"Failed to fetch {type_name} (all servers).")
    return None


# ============================================================
# Water + Coastline (fixed + complete)
# ============================================================
def fetch_body_of_water(area_clause, status_label, mirrors, short_host):
    """
    Fetch water in two stages:
      1) named still-water points (lakes/ponds/reservoir/etc)
      2) named moving-water lines (rivers/streams/canals) as WAYS with geom
    If stage (2) fails, return stage (1) results.
    """
    df_points = fetch_water_points(area_clause, status_label, mirrors, short_host)
    if df_points is None:
        df_points = pd.DataFrame()

    df_lines = fetch_water_lines(area_clause, status_label, mirrors, short_host)
    if df_lines is None:
        return df_points if not df_points.empty else None

    if df_points.empty:
        return df_lines

    return pd.concat([df_points, df_lines], ignore_index=True)


def fetch_water_points(area_clause, status_label, mirrors, short_host):
    df = None  # will remain None unless a mirror succeeds

    q_points = f"""
    [out:json][timeout:80][maxsize:1073741824];
    (
      node[natural=water][name]{area_clause};
      way[natural=water][name]{area_clause};
      relation[natural=water][name]{area_clause};

      node[water~"^(lake|pond|reservoir)$"][name]{area_clause};
      way[water~"^(lake|pond|reservoir)$"][name]{area_clause};
      relation[water~"^(lake|pond|reservoir)$"][name]{area_clause};

      way[landuse=reservoir][name]{area_clause};
      relation[landuse=reservoir][name]{area_clause};
    );
    out body center;
    """

    for url in mirrors:
        try:
            status_label.config(text=f"Trying {short_host(url)} (water points)...")
            status_label.update_idletasks()

            api = overpy.Overpass(url=url)
            res = run_with_timeout(lambda: api.query(q_points), timeout=35)

            rows = []

            def add_point(tags, lat, lon):
                name = clean_name(tags.get("name") or tags.get("name:en"))
                if not name:
                    return

                natural = norm_str(tags.get("natural"))
                water = norm_str(tags.get("water"))
                landuse = norm_str(tags.get("landuse"))

                kind = "water"
                if landuse == "reservoir" or water == "reservoir":
                    kind = "reservoir"
                elif water in ("lake", "pond"):
                    kind = water
                elif natural == "water":
                    kind = water or "water"

                rows.append({
                    "Name": name,
                    "Type": "Body of water",
                    "Kind": kind,
                    "Latitude": float(lat),
                    "Longitude": float(lon),
                })

            for n in res.nodes:
                add_point(n.tags, n.lat, n.lon)

            for w in res.ways:
                lat = getattr(w, "center_lat", None)
                lon = getattr(w, "center_lon", None)
                if lat is None or lon is None:
                    continue
                add_point(w.tags, lat, lon)

            for r in res.relations:
                lat = getattr(r, "center_lat", None)
                lon = getattr(r, "center_lon", None)
                if lat is None or lon is None:
                    continue
                add_point(r.tags, lat, lon)

            if not rows:
                status_label.config(text="No named water points found.")
                return None

            df = pd.DataFrame(rows)
            status_label.config(text=f"Fetched {len(df)} water points.")
            return df

        except TimeoutError:
            status_label.config(text=f"Timeout on {short_host(url)} (water points)")
            status_label.update_idletasks()
            print(f"[WATER POINTS TIMEOUT] {short_host(url)}")
        except Exception as e:
            status_label.config(text=f"Error on {short_host(url)} (water points): {e}")
            status_label.update_idletasks()
            print(f"[WATER POINTS ERROR] {short_host(url)}: {e}")

    return df


def fetch_water_lines(area_clause, status_label, mirrors, short_host):
    """
    Rivers/streams/canals as LINES, staged for reliability:
      Stage 1: named rivers + named canals (WAYS only)
      Stage 2: named streams (WAYS only) - optional; only if stage 1 worked

    Geometry is rebuilt from returned nodes: (._;>;); out body;
    """

    def _fetch_lines_for_kinds(kinds, stage_label, timeout_s):
        """
        kinds: iterable like ("river","canal")
        Returns list[dict] rows or None if mirror loop totally failed.
        Returns [] if query succeeded but produced no rows.
        """
        kinds_re = "|".join(kinds)

        q = f"""
        [out:json][timeout:80][maxsize:1073741824];
        (
          way[waterway~"^({kinds_re})$"][name]{area_clause};
        );
        (._;>;);
        out body;
        """

        overload_backoffs = [1.5, 3.0, 5.0]
        backoff_i = 0

        last_err = None
        for round_i in range(2):
            for url in mirrors:
                host = short_host(url)
                try:
                    print(f"\n[WATER LINES:{stage_label}] Trying mirror: {host}")
                    status_label.config(text=f"Trying {host} ({stage_label})...")
                    status_label.update_idletasks()

                    api = overpy.Overpass(url=url)
                    res = run_with_timeout(lambda: api.query(q), timeout=timeout_s)

                    print(f"[WATER LINES:{stage_label}] Raw result:")
                    print(f"  Ways:  {len(res.ways)}")
                    print(f"  Nodes: {len(res.nodes)}")

                    node_ll = {}
                    for n in res.nodes:
                        try:
                            node_ll[int(n.id)] = (float(n.lat), float(n.lon))
                        except Exception:
                            continue

                    def way_to_geom(w):
                        pts = []
                        for n in getattr(w, "nodes", []) or []:
                            ll = node_ll.get(int(n.id))
                            if ll:
                                pts.append(ll)  # (lat,lon)
                        return pts if len(pts) >= 2 else None

                    rows = []
                    total = 0
                    ok = 0

                    for w in res.ways:
                        ww = norm_str(w.tags.get("waterway"))
                        if ww not in kinds:
                            continue
                        total += 1

                        geom = way_to_geom(w)
                        if not geom:
                            continue
                        ok += 1

                        name = clean_name(w.tags.get("name") or w.tags.get("name:en")) or ""
                        if not name:
                            continue

                        rows.append({
                            "Name": name,
                            "Type": "Body of water",
                            "Kind": ww,
                            "Geometry": geom,
                        })

                    print(f"[WATER LINES:{stage_label}] Ways found: {total}")
                    print(f"[WATER LINES:{stage_label}] Ways with rebuilt geometry: {ok}")
                    print(f"[WATER LINES:{stage_label}] Total segments collected: {len(rows)}")

                    return rows

                except Exception as e:
                    last_err = e

                    if _is_blocked_or_bad_endpoint(e):
                        print(f"[WATER LINES:{stage_label}] 🚫 Blocked/bad endpoint on {host}: {e}")
                        status_label.config(text=f"Blocked on {host} (skipping)")
                        status_label.update_idletasks()
                        continue

                    if _is_overload_error(e):
                        print(f"[WATER LINES:{stage_label}] 💤 Overloaded on {host}: {e}")
                        status_label.config(text=f"{host} overloaded (waiting...)")
                        status_label.update_idletasks()
                        time.sleep(overload_backoffs[min(backoff_i, len(overload_backoffs) - 1)])
                        backoff_i += 1
                        continue

                    if _is_timeout_error(e):
                        print(f"[WATER LINES:{stage_label}] ⏱ Timeout on {host}")
                        status_label.config(text=f"Timeout on {host} ({stage_label})")
                        status_label.update_idletasks()
                        continue

                    msg = str(e).lower()
                    if "unknown content type" in msg and "text/html" in msg:
                        print(f"[WATER LINES:{stage_label}] ⚠ HTML returned by {host} — skipping")
                        status_label.config(text=f"Mirror returned HTML (skipping {host})")
                        status_label.update_idletasks()
                        continue

                    print(f"[WATER LINES:{stage_label}] 💥 Error on {host}: {e}")
                    status_label.config(text=f"Error on {host} ({stage_label}): {e}")
                    status_label.update_idletasks()
                    continue

            print(f"[WATER LINES:{stage_label}] Round {round_i+1}/2 done, retrying mirrors...")

        print(f"[WATER LINES:{stage_label}] ❌ All mirrors failed. Last error: {last_err}")
        return None

    # Stage 1: rivers + canals
    stage1 = _fetch_lines_for_kinds(("river", "canal"), stage_label="water lines (rivers+canals)", timeout_s=25)
    if stage1 is None:
        return None
    if not stage1:
        status_label.config(text="No usable river/canal lines produced.")
        print("[WATER LINES] ❌ Stage 1 produced no usable geometry.")
        return None

    status_label.config(text=f"Fetched {len(stage1)} river/canal line segments. Fetching streams…")
    status_label.update_idletasks()

    # Stage 2: streams (optional)
    stage2 = _fetch_lines_for_kinds(("stream",), stage_label="water lines (streams)", timeout_s=30)
    if stage2 is None:
        print("[WATER LINES] ⚠ Stage 2 (streams) failed; returning rivers/canals only.")
        df = pd.DataFrame(stage1)
        status_label.config(text=f"Fetched {len(df)} water line segments (streams unavailable).")
        return df

    rows = stage1 + stage2
    if not rows:
        return None

    df = pd.DataFrame(rows)
    if "Name" in df.columns:
        df["Name"] = df["Name"].astype(str).str.strip()

    status_label.config(text=f"Fetched {len(df)} water line segments (rivers/canals + streams).")
    print(f"[WATER LINES] ✅ Returning {len(df)} line segments (stage1={len(stage1)}, stage2={len(stage2)})")
    return df


def fetch_coastline_lines(area_clause, status_label, mirrors, short_host):
    """
    Coastline as LINES.
    Rebuild geometry from nodes (same approach as water lines).
    Coastlines are usually unnamed, so we do NOT require name.
    """
    df = None

    q = f"""
    [out:json][timeout:80][maxsize:1073741824];
    (
      way[natural=coastline]{area_clause};
      relation[natural=coastline]{area_clause};
    );
    (._;>;);
    out body;
    """

    for url in mirrors:
        try:
            host = short_host(url)
            print(f"\n[COASTLINE] Trying mirror: {host}")
            status_label.config(text=f"Trying {host} (coastline)...")
            status_label.update_idletasks()

            api = overpy.Overpass(url=url)
            res = run_with_timeout(lambda: api.query(q), timeout=45)

            print(f"[COASTLINE] Raw result:")
            print(f"  Relations: {len(res.relations)}")
            print(f"  Ways:      {len(res.ways)}")
            print(f"  Nodes:     {len(res.nodes)}")

            node_ll = {}
            for n in res.nodes:
                try:
                    node_ll[int(n.id)] = (float(n.lat), float(n.lon))
                except Exception:
                    continue

            def way_to_geom(w):
                pts = []
                for n in getattr(w, "nodes", []) or []:
                    ll = node_ll.get(int(n.id))
                    if ll:
                        pts.append(ll)  # (lat,lon)
                return pts if len(pts) >= 2 else None

            rows = []
            total = 0
            ok = 0

            for w in res.ways:
                if norm_str(w.tags.get("natural")) != "coastline":
                    continue

                total += 1
                geom = way_to_geom(w)
                if not geom:
                    continue

                ok += 1
                rows.append({
                    "Name": "",
                    "Type": "Coastline",
                    "Kind": "coastline",
                    "Geometry": geom,
                })

            print(f"[COASTLINE] Ways found: {total}")
            print(f"[COASTLINE] Ways with rebuilt geometry: {ok}")
            print(f"[COASTLINE] Total coastline segments collected: {len(rows)}")

            if not rows:
                status_label.config(text="No usable coastline lines produced.")
                print("[COASTLINE] ❌ No usable coastline geometry produced.")
                return None

            df = pd.DataFrame(rows)
            status_label.config(text=f"Fetched {len(df)} coastline segments.")
            print(f"[COASTLINE] ✅ Returning {len(df)} coastline segments")
            return df

        except TimeoutError:
            print(f"[COASTLINE] ⏱ Timeout on {short_host(url)}")
            status_label.config(text=f"Timeout on {short_host(url)} (coastline)")
            status_label.update_idletasks()

        except Exception as e:
            print(f"[COASTLINE] 💥 Error on {short_host(url)}: {e}")
            status_label.config(text=f"Error on {short_host(url)} (coastline): {e}")
            status_label.update_idletasks()

    print("[COASTLINE] ❌ All mirrors failed")
    return df