"""Plot utilities for vmod deformation sources.

This module produces 2D map-view and 3D plots of a single ``vmod`` deformation
source (``Penny``, ``Yang``, or ``Mctigue``). It draws the source geometry and
can optionally overlay observed and modeled displacement vectors supplied in
vmod format. Plots are rendered in the local UTM coordinate frame with axis
labels expressed in kilometers.

Public API (implemented in later tasks):

- ``plot_mapview(...)``    -- the Mapview_Plotter (2D map-view plot).
- ``plot_source_3d(...)``  -- the Source_Plotter_3D (3D plot with topography).

Drafted with Kiro AI, results were checked for a number of
scenarios and looked fine, but these scenarios were not fully comprehensive,
and code itself was checked quickly so there might still be bugs buried.
"""

from __future__ import annotations

import json
import os
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import matplotlib

# Select a non-interactive backend when no display/GUI backend is active so the
# module can be imported and used in headless environments (e.g. test runs).
# This is a no-op if an interactive backend is already configured by the caller.
if matplotlib.get_backend().lower() == "agg":
    # Already headless-safe; nothing to do.
    pass

import matplotlib.pyplot as plt  # noqa: E402  (import after backend consideration)
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402  (registers 3D projection)

import rasterio  # noqa: E402
from rasterio.crs import CRS  # noqa: E402
from rasterio.warp import (  # noqa: E402
    Resampling,
    calculate_default_transform,
    reproject,
)


# ---------------------------------------------------------------------------
# Repository-root resolver
# ---------------------------------------------------------------------------
# This module lives at <repo_root>/LV_geodeticinv/plot_utilities.py, so the
# repository root is one directory above the package directory.
REPO_ROOT = Path(__file__).resolve().parents[1]


def get_repo_root() -> Path:
    """Return the absolute path to the repository root directory."""
    return REPO_ROOT


# ---------------------------------------------------------------------------
# Geometry discretization constants
# ---------------------------------------------------------------------------
N_SEGMENTS = 100          # 2D outline segments -> 101 vertices (closed polygon)
N_THETA = 20              # 3D horizontal-angle samples (0..360 degrees)
N_PHI = 10                # 3D vertical-angle samples (0..180 degrees)
PENNY_ASPECT = 0.05       # vertical aspect ratio for the thin oblate ellipsoid

# ---------------------------------------------------------------------------
# Shading constants
# ---------------------------------------------------------------------------
SHADE_MIN = 0.4           # brightness of a fully downward-facing facet
SHADE_MAX = 1.0           # brightness of a fully upward-facing facet

# Source-surface rendering emphasis. The base per-facet shading from
# surface_normal_shading spans [SHADE_MIN, SHADE_MAX]; for the 3D source it is
# remapped down to SOURCE_SHADE_MIN to exaggerate the top/bottom contrast, and
# thin black facet edges are drawn to make the surface curvature more legible.
SOURCE_SHADE_MIN = 0.12       # exaggerated darkest brightness for the source
SOURCE_WIREFRAME_LW = 0.3     # source facet-edge line width (thin black lines)

# ---------------------------------------------------------------------------
# Limit / margin constants
# ---------------------------------------------------------------------------
MARGIN = 0.10             # 10% auto-limit margin

# ---------------------------------------------------------------------------
# DEM constants
# ---------------------------------------------------------------------------
DEM_MAX_SAMPLES = 200     # max DEM samples per horizontal axis (map view)
# Cap for the 3D DEM surface. matplotlib renders one flat-colored quad per cell
# and 3D render cost grows with the cell count, so this trades detail vs. speed
# (~300 is a reasonable balance; raise for smoother topography, lower for speed).
DEM_3D_MAX_SAMPLES = 120
DEM_OPACITY = 0.5         # topographic surface opacity
DEM_CMAP = "gray_r"       # slope colormap (steeper -> darker), matches plot_gnss_vectors

# ---------------------------------------------------------------------------
# Figure output constants
# ---------------------------------------------------------------------------
FIG_SIZE = (8.0, 8.0)     # figure size in inches for both map-view and 3D plots
FIG_DPI = 400             # output resolution in dots per inch

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_COLOR = "orange"                          # default base color
DEFAULT_DEM = "data/maps/output_USGS30m.tif"      # default DEM (relative to repo root)
DEFAULT_FIG_DIR = "data/figures/"                 # default save directory (relative to repo root)

# ---------------------------------------------------------------------------
# Source parameter counts
# ---------------------------------------------------------------------------
SOURCE_PARAM_COUNTS = {"penny": 5, "yang": 8, "mctigue": 5}


# ---------------------------------------------------------------------------
# Source resolution and parameter handling
# ---------------------------------------------------------------------------
# Ordered parameter field names per source type. The ordering matches the
# ``parameters`` attribute of the corresponding vmod source class.
SOURCE_PARAM_FIELDS = {
    "penny": ("xcen", "ycen", "depth", "pressure", "radius"),
    "yang": ("xcen", "ycen", "depth", "pressure", "a", "b", "az", "dip"),
    "mctigue": ("xcen", "ycen", "depth", "radius", "dP"),
}


def resolve_source_type(source) -> str:
    """Resolve a source input to one of ``"penny"``, ``"yang"``, ``"mctigue"``.

    The ``source`` argument may be either:

    - a ``str`` naming the source type (matched case-insensitively), or
    - a vmod ``Source`` object, in which case its ``get_source_id()`` return
      value (falling back to its class name) is matched case-insensitively.

    Parameters:
        source: A source-type string or a vmod ``Source`` object.

    Returns:
        str: The canonical lowercase source type.

    Raises:
        ValueError: If the resolved identifier does not match a supported
            source type. The message names the unsupported type.
    """
    if isinstance(source, str):
        candidate = source
    else:
        # Treat as a vmod Source object. Prefer get_source_id(); fall back to
        # the class name if that method is unavailable or fails.
        candidate = None
        get_id = getattr(source, "get_source_id", None)
        if callable(get_id):
            try:
                candidate = get_id()
            except Exception:
                candidate = None
        if not isinstance(candidate, str) or not candidate:
            candidate = type(source).__name__

    key = candidate.strip().lower()
    if key in SOURCE_PARAM_COUNTS:
        return key

    supported = ", ".join(sorted(SOURCE_PARAM_COUNTS))
    raise ValueError(
        f"Unsupported source type {candidate!r}; expected one of: {supported}."
    )


def validate_parameters(source_type: str, params) -> None:
    """Validate that ``params`` has the expected length for ``source_type``.

    Parameters:
        source_type (str): A canonical source type ("penny"/"yang"/"mctigue").
        params: The ordered parameter vector for the source.

    Raises:
        ValueError: If ``len(params)`` differs from the required count for the
            source type. The message describes the expected and received counts.
    """
    expected = SOURCE_PARAM_COUNTS[source_type]
    received = len(params)
    if received != expected:
        raise ValueError(
            f"Source type {source_type!r} requires exactly {expected} "
            f"parameters, but received {received}."
        )


def unpack_parameters(source_type: str, params) -> dict:
    """Map an ordered parameter vector to named fields for ``source_type``.

    The parameter ordering follows the corresponding vmod source definition:

    - penny:   ``xcen, ycen, depth, pressure, radius``
    - yang:    ``xcen, ycen, depth, pressure, a, b, az, dip``
    - mctigue: ``xcen, ycen, depth, radius, dP``

    Parameters:
        source_type (str): A canonical source type ("penny"/"yang"/"mctigue").
        params: The ordered parameter vector for the source.

    Returns:
        dict: A mapping from field name to value, preserving parameter order.

    Raises:
        ValueError: If ``params`` does not have the expected length.
    """
    validate_parameters(source_type, params)
    fields = SOURCE_PARAM_FIELDS[source_type]
    return {name: value for name, value in zip(fields, params)}


# ---------------------------------------------------------------------------
# Coordinate transformer
# ---------------------------------------------------------------------------
# LV_info.json lives alongside this module in the LV_geodeticinv package
# directory (see the glossary entry "LV_Info": LV_geodeticinv/LV_info.json).
DEFAULT_LV_INFO = Path(__file__).resolve().parent / "LV_info.json"


def get_utm_zone(lv_info_path=None) -> tuple[int, bool]:
    """Determine the local UTM zone from the project centroid in LV_Info.

    Reads the project centroid ``[lat, lon]`` from ``LV_info.json`` (the
    default package-local path, or a caller-supplied override) and derives the
    UTM zone number from the centroid longitude along with a southern-hemisphere
    flag from the centroid latitude.

    Parameters:
        lv_info_path: Optional path to an ``LV_info.json`` file. When ``None``,
            the default package-local ``LV_info.json`` is used.

    Returns:
        tuple[int, bool]: ``(zone_number, southern)`` where
            ``zone_number = int((lon + 180) / 6) + 1`` and
            ``southern = lat < 0``.

    Raises:
        ValueError: If LV_Info cannot be read or does not contain a project
            centroid. The message identifies the problem and the offending path.
    """
    path = Path(lv_info_path) if lv_info_path is not None else DEFAULT_LV_INFO

    try:
        with open(path) as file:
            info = json.load(file)
    except FileNotFoundError as exc:
        raise ValueError(
            f"Could not read LV_info at {str(path)!r}: file not found."
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read LV_info at {str(path)!r}: {exc}."
        ) from exc

    if not isinstance(info, dict) or "centroid" not in info:
        raise ValueError(
            f"LV_info at {str(path)!r} does not contain a project 'centroid'."
        )

    centroid = info["centroid"]
    try:
        lat = float(centroid[0])
        lon = float(centroid[1])
    except (TypeError, IndexError, ValueError) as exc:
        raise ValueError(
            f"LV_info at {str(path)!r} has an invalid 'centroid'; expected "
            f"[lat, lon], got {centroid!r}."
        ) from exc

    zone_number = int((lon + 180) / 6) + 1
    southern = lat < 0
    return zone_number, southern


def utm_crs_from_zone(zone_number: int, southern: bool) -> str:
    """Return the EPSG code string for a WGS84 UTM zone.

    Northern-hemisphere zones map to EPSG ``326xx`` and southern-hemisphere
    zones to EPSG ``327xx`` (where ``xx`` is the two-digit zone number). This is
    the destination CRS used when reprojecting a geographic DEM into the local
    UTM frame the rest of the module operates in.

    Parameters:
        zone_number (int): UTM zone number (1-60).
        southern (bool): ``True`` for the southern hemisphere.

    Returns:
        str: An EPSG code string such as ``"EPSG:32611"``.
    """
    base = 32700 if southern else 32600
    return f"EPSG:{base + int(zone_number)}"


def depth_to_elevation(depth_m: float, reference_elevation: float = 0.0) -> float:
    """Convert a positive-down source depth to an elevation.

    Uses the relation ``elevation = reference_elevation - depth``, where
    ``depth`` is measured positive-down from the reference elevation.

    Parameters:
        depth_m (float): Source depth in meters (positive-down).
        reference_elevation (float): Elevation, in meters, from which depth is
            measured downward. Defaults to 0.0.

    Returns:
        float: The elevation in meters.
    """
    return reference_elevation - depth_m


def km_axis_formatter():
    """Return a matplotlib ``FuncFormatter`` that labels axes in kilometers.

    The formatter divides axis tick values (in meters) by 1000 so displayed
    axis labels read in kilometers. It is shared by both the map-view and 3D
    plotters for the easting, northing, and (in 3D) elevation axes.

    Returns:
        matplotlib.ticker.FuncFormatter: A formatter converting meters to km.
    """
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda value, _pos: f"{value / 1000.0:g}")


# ---------------------------------------------------------------------------
# Geometry builders (pure)
# ---------------------------------------------------------------------------
def _yang_basis_vectors(az: float, dip: float) -> tuple:
    """Return the Yang prolate spheroid's orthonormal (E, N, Up) basis vectors.

    ``az`` and ``dip`` must already be in radians. The major axis (length
    ``a``) points along azimuth ``az`` (clockwise from north) tilted downward
    from horizontal by ``dip``; the two minor axes (length ``b`` each) complete
    a right-handed orthonormal frame. This basis is shared by
    :func:`build_2d_outline` (for the projected footprint) and
    :func:`build_3d_surface` (for the 3D surface) so both stay consistent.

    Parameters:
        az (float): Azimuth in radians, clockwise from north.
        dip (float): Dip in radians, from horizontal.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: The
            ``(e_major, e_minor1, e_minor2)`` unit vectors, each a length-3
            ``(E, N, Up)`` array.
    """
    sin_az, cos_az = np.sin(az), np.cos(az)
    sin_dip, cos_dip = np.sin(dip), np.cos(dip)
    e_major = np.array([cos_dip * sin_az, cos_dip * cos_az, -sin_dip])
    e_minor1 = np.array([cos_az, -sin_az, 0.0])
    e_minor2 = np.array([-sin_dip * sin_az, -sin_dip * cos_az, -cos_dip])
    return e_major, e_minor1, e_minor2


def build_2d_outline(source_type: str, fields: dict) -> np.ndarray:
    """Build the closed 2D map-view outline of a source footprint.

    The outline is expressed in local UTM meters as ``(E, N)`` vertices and is
    discretized with exactly ``N_SEGMENTS`` (100) line segments, producing 101
    vertices whose final vertex coincides exactly with the first (a closed
    polygon).

    Geometry per source type:

    - ``penny`` / ``mctigue``: a circle centered at ``(xcen, ycen)`` with a
      radius equal to the source ``radius`` in meters.
    - ``yang``: the exact horizontal (orthogonal) projection of the dipped,
      rotated prolate spheroid onto the map-view plane.

    Parameters:
        source_type (str): A canonical source type ("penny"/"yang"/"mctigue").
        fields (dict): The unpacked source fields from ``unpack_parameters``.

    Returns:
        numpy.ndarray: A ``(101, 2)`` array of ``(E, N)`` vertices in UTM meters
            with ``result[-1] == result[0]``.

    Raises:
        ValueError: If the source ``radius`` (penny/mctigue) or either semi-axis
            ``a`` or ``b`` (yang) is less than or equal to zero.
    """
    xcen = float(fields["xcen"])
    ycen = float(fields["ycen"])

    # Parameter t spans a full revolution; 100 segments -> 101 vertices.
    t = np.linspace(0.0, 2.0 * np.pi, N_SEGMENTS + 1)

    if source_type in ("penny", "mctigue"):
        radius = float(fields["radius"])
        if radius <= 0:
            raise ValueError(
                f"Source type {source_type!r} requires a positive radius, "
                f"but received radius={radius}."
            )
        east = xcen + radius * np.cos(t)
        north = ycen + radius * np.sin(t)

    elif source_type == "yang":
        a = float(fields["a"])
        b = float(fields["b"])
        if a <= 0 or b <= 0:
            raise ValueError(
                f"Source type 'yang' requires positive semi-axes, but received "
                f"a={a}, b={b}."
            )
        az = np.deg2rad(float(fields["az"]))
        dip = np.deg2rad(float(fields["dip"]))

        # The map-view footprint is the horizontal (orthogonal) projection of the
        # dipping prolate spheroid, which is itself an ellipse - build 3x3
        # semi-axis matrix A (columns = the world-frame semi-axis vectors, same
        # basis as build_3d_surface), then the shadow of the solid ellipsoid
        # {center + A u : |u| <= 1} under (E, N) projection is the solid ellipse
        # {center + (P A) v : |v| <= 1}, whose shape matrix is
        # M = (P A)(P A)^T (P = the projection onto the first two rows).
        e_major, e_minor1, e_minor2 = _yang_basis_vectors(az, dip)
        semi_axis_matrix = np.column_stack(
            (a * e_major, b * e_minor1, b * e_minor2)
        )
        proj = semi_axis_matrix[:2, :]  # (E, N) rows only
        shadow = proj @ proj.T  # 2x2 shape matrix of the projected ellipse

        # eigh returns ascending eigenvalues for a symmetric matrix; semi_axes
        # are the projected ellipse's semi-axis lengths and eigvecs their
        # (E, N) directions.
        eigvals, eigvecs = np.linalg.eigh(shadow)
        semi_axes = np.sqrt(np.maximum(eigvals, 0.0))

        local_u = semi_axes[-1] * np.cos(t)
        local_v = semi_axes[0] * np.sin(t)
        east = xcen + eigvecs[0, -1] * local_u + eigvecs[0, 0] * local_v
        north = ycen + eigvecs[1, -1] * local_u + eigvecs[1, 0] * local_v

    else:
        # resolve_source_type guarantees a valid type upstream; guard anyway.
        supported = ", ".join(sorted(SOURCE_PARAM_COUNTS))
        raise ValueError(
            f"Unsupported source type {source_type!r}; expected one of: {supported}."
        )

    outline = np.column_stack((east, north))
    # Force exact closure (row[-1] == row[0]) despite floating-point rounding.
    outline[-1] = outline[0]
    return outline


def build_3d_surface(
    source_type: str, fields: dict, reference_elevation: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the 3D ellipsoidal surface of a source.

    Returns ``(X, Y, Z)`` meshgrids, each of shape ``(N_THETA, N_PHI)`` =
    ``(100, 100)``, sampling the horizontal angle ``theta`` over 0..360 degrees
    (the row/first axis) and the vertical angle ``phi`` over 0..180 degrees (the
    column/second axis). The surface is centered horizontally at
    ``(xcen, ycen)`` and vertically at ``elevation = reference_elevation -
    depth`` (via :func:`depth_to_elevation`). ``X`` and ``Y`` are UTM eastings
    and northings in meters; ``Z`` is elevation above sea level in meters.

    Geometry per source type:

    - ``penny``: a thin oblate ellipsoid whose two horizontal semi-axes each
      equal the source ``radius`` and whose vertical semi-minor axis equals
      ``radius * PENNY_ASPECT`` (0.05).
    - ``mctigue``: a sphere with all semi-axes equal to the source ``radius``.
    - ``yang``: a prolate spheroid with major semi-axis ``a`` and two minor
      semi-axes ``b``, oriented so that the major axis points along the azimuth
      ``az`` (clockwise from north) and dips downward from horizontal by ``dip``
      degrees. The orientation is consistent with :func:`build_2d_outline`,
      whose horizontal major direction is ``(sin az, cos az)`` in ``(E, N)`` and
      whose horizontal minor direction is ``(cos az, -sin az)``.

    Parameters:
        source_type (str): A canonical source type ("penny"/"yang"/"mctigue").
        fields (dict): The unpacked source fields from ``unpack_parameters``.
        reference_elevation (float): Elevation, in meters, from which the source
            depth is measured downward (positive-down). Defaults to 0.0.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: The ``(X, Y, Z)``
            meshgrids, each of shape ``(N_THETA, N_PHI)``.

    Raises:
        ValueError: If the source ``radius`` (penny/mctigue) or either semi-axis
            ``a`` or ``b`` (yang) is less than or equal to zero, or if the
            source type is unsupported.
    """
    xcen = float(fields["xcen"])
    ycen = float(fields["ycen"])
    depth = float(fields["depth"])
    elevation = depth_to_elevation(depth, reference_elevation)

    # theta -> horizontal angle 0..360 deg (first axis, N_THETA samples)
    # phi   -> vertical angle 0..180 deg (second axis, N_PHI samples)
    theta = np.linspace(0.0, 2.0 * np.pi, N_THETA)
    phi = np.linspace(0.0, np.pi, N_PHI)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

    # Unit-sphere direction cosines (shape (N_THETA, N_PHI)).
    unit_x = np.sin(phi_grid) * np.cos(theta_grid)
    unit_y = np.sin(phi_grid) * np.sin(theta_grid)
    unit_z = np.cos(phi_grid)

    if source_type in ("penny", "mctigue"):
        radius = float(fields["radius"])
        if radius <= 0:
            raise ValueError(
                f"Source type {source_type!r} requires a positive radius, "
                f"but received radius={radius}."
            )
        semi_z = radius * PENNY_ASPECT if source_type == "penny" else radius
        east = xcen + radius * unit_x
        north = ycen + radius * unit_y
        up = elevation + semi_z * unit_z

    elif source_type == "yang":
        a = float(fields["a"])
        b = float(fields["b"])
        if a <= 0 or b <= 0:
            raise ValueError(
                f"Source type 'yang' requires positive semi-axes, but received "
                f"a={a}, b={b}."
            )
        az = np.deg2rad(float(fields["az"]))
        dip = np.deg2rad(float(fields["dip"]))

        # Orthonormal spheroid basis in world (E, N, Up) coordinates, shared
        # with build_2d_outline so the 3D surface and the projected footprint
        # stay consistent.
        e_major, e_minor1, e_minor2 = _yang_basis_vectors(az, dip)

        # Scale the unit-sphere coords by the semi-axes (a along major, b along
        # each minor) and rotate into world coordinates.
        p = a * unit_x
        q = b * unit_y
        r = b * unit_z
        east = xcen + p * e_major[0] + q * e_minor1[0] + r * e_minor2[0]
        north = ycen + p * e_major[1] + q * e_minor1[1] + r * e_minor2[1]
        up = elevation + p * e_major[2] + q * e_minor1[2] + r * e_minor2[2]

    else:
        supported = ", ".join(sorted(SOURCE_PARAM_COUNTS))
        raise ValueError(
            f"Unsupported source type {source_type!r}; expected one of: {supported}."
        )

    return east, north, up


def surface_normal_shading(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray
) -> np.ndarray:
    """Compute a per-facet brightness scaling factor for downward lighting.

    Given the ``(X, Y, Z)`` meshgrids of a 3D surface (as produced by
    :func:`build_3d_surface`, shape ``(N_THETA, N_PHI)`` = ``(100, 100)``),
    this returns a brightness scaling factor for every quad facet formed
    between adjacent grid cells. The result therefore has shape
    ``(N_THETA - 1, N_PHI - 1)`` = ``(99, 99)``.

    For each facet the unit surface normal is computed from the cross product
    of the facet's two edge vectors. The normal is oriented outward (away from
    the surface centroid) so that upward-facing facets have a positive vertical
    component. The vertical component ``nz`` lies in ``[-1, 1]`` (``+1`` fully
    upward-facing, ``-1`` fully downward-facing) and is mapped linearly to a
    brightness scale::

        scale = SHADE_MIN + (SHADE_MAX - SHADE_MIN) * (nz + 1) / 2
              = 0.4 + 0.6 * (nz + 1) / 2

    so that ``nz = +1`` -> ``1.0`` and ``nz = -1`` -> ``0.4``. The scale is
    clamped to the inclusive range ``[SHADE_MIN, SHADE_MAX]`` = ``[0.4, 1.0]``
    and is monotonically non-decreasing in ``nz``.

    Parameters:
        X (numpy.ndarray): Easting meshgrid of the surface, shape ``(m, n)``.
        Y (numpy.ndarray): Northing meshgrid of the surface, shape ``(m, n)``.
        Z (numpy.ndarray): Elevation meshgrid of the surface, shape ``(m, n)``.

    Returns:
        numpy.ndarray: A ``(m - 1, n - 1)`` array of brightness scaling factors
            in the inclusive range ``[0.4, 1.0]``, one per quad facet.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=float)

    def corners(A):
        # (i,j), (i+1,j), (i,j+1), (i+1,j+1) corners of each quad facet.
        return A[:-1, :-1], A[1:, :-1], A[:-1, 1:], A[1:, 1:]

    x00, x10, x01, x11 = corners(X)
    y00, y10, y01, y11 = corners(Y)
    z00, z10, z01, z11 = corners(Z)

    # Two edge vectors of each quad emanating from the (i, j) corner.
    edge1 = np.stack((x10 - x00, y10 - y00, z10 - z00), axis=-1)
    edge2 = np.stack((x01 - x00, y01 - y00, z01 - z00), axis=-1)
    normal = np.cross(edge1, edge2)  # shape (m-1, n-1, 3)

    # Orient each facet normal outward (away from the surface centroid) so that
    # upward-facing facets get a positive vertical component. The surface is a
    # convex ellipsoid, so pointing away from the centroid is the outward sense.
    facet_center = np.stack(
        (
            (x00 + x10 + x01 + x11) / 4.0,
            (y00 + y10 + y01 + y11) / 4.0,
            (z00 + z10 + z01 + z11) / 4.0,
        ),
        axis=-1,
    )
    centroid = np.array([X.mean(), Y.mean(), Z.mean()])
    outward = facet_center - centroid
    inward = np.sum(normal * outward, axis=-1) < 0
    normal[inward] = -normal[inward]

    magnitude = np.linalg.norm(normal, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        nz = np.where(magnitude > 0, normal[..., 2] / magnitude, 0.0)

    scale = SHADE_MIN + (SHADE_MAX - SHADE_MIN) * (nz + 1.0) / 2.0
    return np.clip(scale, SHADE_MIN, SHADE_MAX)


# ---------------------------------------------------------------------------
# Limit calculator (pure)
# ---------------------------------------------------------------------------
def compute_horizontal_limits(geometry_pts, vector_pts):
    """Compute auto horizontal limits enclosing geometry and vector points.

    The limits enclose every supplied ``(E, N)`` point (source geometry plus any
    displayed displacement-vector positions) and then expand each axis by a
    margin equal to :data:`MARGIN` (10%) of that axis's enclosed extent on each
    side.

    Parameters:
        geometry_pts: An array-like of source-geometry ``(E, N)`` points in UTM
            meters, shape ``(n, 2)``. Must contain at least one point.
        vector_pts: An array-like of displacement-vector ``(E, N)`` station
            positions in UTM meters, shape ``(m, 2)``, or ``None``/empty when no
            vectors are displayed.

    Returns:
        tuple[tuple[float, float], tuple[float, float]]: The limits as
            ``((e_lo, e_hi), (n_lo, n_hi))`` in UTM meters.

    Raises:
        ValueError: If no geometry points are supplied.
    """
    geometry = np.asarray(geometry_pts, dtype=float).reshape(-1, 2)
    if geometry.size == 0:
        raise ValueError(
            "compute_horizontal_limits requires at least one geometry point."
        )

    points = geometry
    if vector_pts is not None:
        vectors = np.asarray(vector_pts, dtype=float).reshape(-1, 2)
        if vectors.size > 0:
            points = np.vstack((geometry, vectors))

    e_lo, n_lo = points.min(axis=0)
    e_hi, n_hi = points.max(axis=0)

    e_margin = MARGIN * (e_hi - e_lo)
    n_margin = MARGIN * (n_hi - n_lo)

    easting_limits = (float(e_lo - e_margin), float(e_hi + e_margin))
    northing_limits = (float(n_lo - n_margin), float(n_hi + n_margin))
    return easting_limits, northing_limits


def compute_depth_limit(fields: dict) -> float:
    """Compute an auto positive-down depth limit enclosing the source.

    The limit encloses the source's deepest vertical extent -- the source
    ``depth`` plus the source's vertical half-extent -- and then adds a margin
    of :data:`MARGIN` (10%) of that enclosed depth extent below it.

    The vertical half-extent depends on the source geometry, matching
    :func:`build_3d_surface`:

    - ``mctigue`` (sphere): half-extent = ``radius``.
    - ``penny`` (thin oblate ellipsoid): half-extent = ``radius * PENNY_ASPECT``.
    - ``yang`` (dipped prolate spheroid): half-extent =
      ``sqrt((a * sin(dip))**2 + (b * cos(dip))**2)``, the vertical support of
      the oriented spheroid.

    The source type is inferred from the field names present (``fields`` as
    produced by :func:`unpack_parameters`): a ``yang`` source has ``a``/``b``/
    ``dip``; a ``mctigue`` source has ``dP``; otherwise the source is ``penny``.

    Parameters:
        fields (dict): The unpacked source fields from :func:`unpack_parameters`.

    Returns:
        float: The positive-down depth limit in meters.
    """
    depth = float(fields["depth"])

    if "a" in fields and "b" in fields and "dip" in fields:
        a = float(fields["a"])
        b = float(fields["b"])
        dip = np.deg2rad(float(fields["dip"]))
        half_extent = float(
            np.hypot(a * np.sin(dip), b * np.cos(dip))
        )
    elif "dP" in fields:
        # mctigue sphere
        half_extent = float(fields["radius"])
    else:
        # penny thin oblate ellipsoid
        half_extent = float(fields["radius"]) * PENNY_ASPECT

    max_depth = depth + half_extent
    return max_depth + MARGIN * max_depth


def validate_limits(easting_limits, northing_limits, depth_limit) -> None:
    """Validate caller-supplied plot limits.

    Any argument may be ``None`` (not supplied), in which case it is skipped.
    Supplied horizontal limit pairs must have a lower bound strictly less than
    their upper bound, and a supplied depth limit must be strictly positive.

    Parameters:
        easting_limits: A supplied ``(lo, hi)`` easting pair in UTM meters, or
            ``None``.
        northing_limits: A supplied ``(lo, hi)`` northing pair in UTM meters, or
            ``None``.
        depth_limit: A supplied positive-down depth limit in meters, or ``None``.

    Raises:
        ValueError: If a supplied horizontal limit pair has ``lo >= hi``, or if a
            supplied ``depth_limit`` is less than or equal to zero. The message
            identifies the invalid limit.
    """
    if easting_limits is not None:
        lo, hi = easting_limits
        if lo >= hi:
            raise ValueError(
                f"Invalid easting_limits {(lo, hi)!r}: lower bound must be less "
                f"than upper bound."
            )

    if northing_limits is not None:
        lo, hi = northing_limits
        if lo >= hi:
            raise ValueError(
                f"Invalid northing_limits {(lo, hi)!r}: lower bound must be less "
                f"than upper bound."
            )

    if depth_limit is not None:
        if depth_limit <= 0:
            raise ValueError(
                f"Invalid depth_limit {depth_limit!r}: must be greater than zero."
            )


# ---------------------------------------------------------------------------
# DEM loader (I/O)
# ---------------------------------------------------------------------------
# Return contract shared by ``downsample_dem`` and ``clip_dem_to_limits``
# (and consumed by ``plot_source_3d`` in task 11.1):
#
#     dem_grid = (elevation, easting, northing)
#
# where
#     - ``elevation`` is a 2D ``numpy.ndarray`` of shape ``(nrows, ncols)`` of
#       elevations in meters (invalid cells are NaN),
#     - ``easting`` is a 1D ``numpy.ndarray`` of length ``ncols`` giving the
#       cell-center UTM easting (meters) for each column, and
#     - ``northing`` is a 1D ``numpy.ndarray`` of length ``nrows`` giving the
#       cell-center UTM northing (meters) for each row.
#
# Thus ``elevation[i, j]`` sits at horizontal position
# ``(easting[j], northing[i])``. The 1D coordinate arrays make it trivial to
# build a meshgrid for ``Axes3D.plot_surface`` and to clip by horizontal limits.


def resolve_dem_path(dem_path=None) -> Path:
    """Resolve the DEM path, falling back to the project default.

    Parameters:
        dem_path: Optional path to a DEM GeoTIFF. When ``None``, the default
            :data:`DEFAULT_DEM` (``data/maps/output_USGS30m.tif``) relative to
            the repository root is used.

    Returns:
        pathlib.Path: The resolved DEM path.

    Raises:
        FileNotFoundError: If no file exists at the resolved path. The message
            names the missing DEM path.
    """
    if dem_path is not None:
        path = Path(dem_path)
    else:
        path = REPO_ROOT / DEFAULT_DEM

    if not path.exists():
        raise FileNotFoundError(f"DEM file not found: {path}")

    return path


@contextmanager
def _proj_network_off():
    """Temporarily disable PROJ's network grid downloads.

    Some datum transforms (e.g. NAD83 -> WGS84 UTM, used when reprojecting the
    default geographic DEM into the local UTM frame) make PROJ try to download a
    datum-shift grid from ``cdn.proj.org``. On networks that block that download
    PROJ raises a hard error. Disabling the network makes PROJ fall back to a
    sub-meter "ballpark" transform, which is more than accurate enough for
    plotting. The previous ``PROJ_NETWORK`` value is restored on exit.
    """
    previous = os.environ.get("PROJ_NETWORK")
    os.environ["PROJ_NETWORK"] = "OFF"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("PROJ_NETWORK", None)
        else:
            os.environ["PROJ_NETWORK"] = previous


def load_dem(dem_path, dst_crs=None) -> tuple[np.ndarray, "rasterio.Affine"]:
    """Load the elevation band and affine transform from a DEM raster.

    Follows the DEM-loading convention established in ``plot_gnss_vectors.py``:
    open the raster with ``rasterio`` and read the first band as the elevation
    grid together with the raster's affine ``transform``. Cells equal to the
    raster's nodata value are converted to NaN so downstream code can detect
    the absence of valid elevation data.

    When ``dst_crs`` is supplied and differs from the raster's own CRS, the DEM
    is reprojected into ``dst_crs`` before returning. This is used to warp a
    geographic (lat/lon) DEM into the local UTM frame the rest of the module
    operates in, so the default project DEM (EPSG:4269) can be used directly.

    Parameters:
        dem_path: Path to the DEM GeoTIFF to open for reading.
        dst_crs: Optional destination CRS (anything ``rasterio`` accepts, e.g.
            an ``"EPSG:32611"`` string). When ``None`` (the default), the DEM is
            returned in its native CRS with no reprojection.

    Returns:
        tuple[numpy.ndarray, rasterio.Affine]: ``(elevation, transform)`` where
            ``elevation`` is the (possibly reprojected) first band as a float
            ``(nrows, ncols)`` array (nodata -> NaN) and ``transform`` is the
            corresponding affine transform mapping pixel ``(col, row)`` to CRS
            ``(x, y)``.
    """
    # Disable PROJ's network grid downloads for the whole open/reproject block.
    # rasterio snapshots the PROJ/GDAL config when the dataset Env is entered
    # (at ``rasterio.open``), so the setting must be in place *before* the file
    # is opened -- not merely around the warp calls -- for it to take effect.
    with _proj_network_off():
        with rasterio.open(dem_path) as src:
            src_crs = src.crs
            want_reproject = (
                dst_crs is not None
                and src_crs is not None
                and CRS.from_user_input(dst_crs) != src_crs
            )

            if want_reproject:
                dst_transform, width, height = calculate_default_transform(
                    src_crs, dst_crs, src.width, src.height, *src.bounds
                )
                elevation = np.full((height, width), np.nan, dtype=float)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=elevation,
                    src_transform=src.transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    src_nodata=src.nodata,
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                )
                return elevation, dst_transform

            elevation = src.read(1).astype(float)
            transform = src.transform
            nodata = src.nodata

    if nodata is not None:
        elevation[elevation == nodata] = np.nan

    return elevation, transform


def _axis_coordinates(transform, nrows: int, ncols: int):
    """Return (easting, northing) cell-center coordinate arrays for a raster.

    Assumes a north-up (unrotated) affine transform, which is the standard case
    for the project DEMs. Cell centers are offset by half a pixel from the pixel
    edges given by the transform.
    """
    # Affine: x = a*col + b*row + c ; y = d*col + e*row + f.
    # For a north-up raster b == d == 0, so easting depends only on column and
    # northing only on row.
    a = transform.a
    c = transform.c
    e = transform.e
    f = transform.f

    cols = np.arange(ncols)
    rows = np.arange(nrows)
    easting = c + a * (cols + 0.5)
    northing = f + e * (rows + 0.5)
    return easting, northing


def downsample_dem(dem, transform):
    """Downsample a DEM so each axis has at most :data:`DEM_MAX_SAMPLES` samples.

    A uniform integer stride is chosen per axis so that neither the number of
    rows nor the number of columns exceeds :data:`DEM_MAX_SAMPLES` (200). The
    elevation grid is strided and matching cell-center easting/northing
    coordinate arrays are computed from ``transform``.

    Parameters:
        dem (numpy.ndarray): The full-resolution elevation grid of shape
            ``(nrows, ncols)`` (as returned by :func:`load_dem`).
        transform (rasterio.Affine): The raster affine transform mapping pixel
            ``(col, row)`` to CRS ``(x, y)`` (as returned by :func:`load_dem`).

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: The downsampled
            ``(elevation, easting, northing)`` bundle described in the module's
            DEM return-contract note. ``elevation`` has shape
            ``(nrows_ds, ncols_ds)`` with each dimension ``<= DEM_MAX_SAMPLES``;
            ``easting`` has length ``ncols_ds`` and ``northing`` length
            ``nrows_ds``.
    """
    dem = np.asarray(dem, dtype=float)
    nrows, ncols = dem.shape

    easting, northing = _axis_coordinates(transform, nrows, ncols)

    # Ceil-division stride guarantees the strided count stays <= max samples.
    row_stride = int(np.ceil(nrows / DEM_MAX_SAMPLES)) if nrows > 0 else 1
    col_stride = int(np.ceil(ncols / DEM_MAX_SAMPLES)) if ncols > 0 else 1
    row_stride = max(row_stride, 1)
    col_stride = max(col_stride, 1)

    elevation_ds = dem[::row_stride, ::col_stride]
    northing_ds = northing[::row_stride]
    easting_ds = easting[::col_stride]

    return elevation_ds, easting_ds, northing_ds


def full_dem_bundle(dem, transform):
    """Wrap a full-resolution DEM as an ``(elevation, easting, northing)`` bundle.

    Computes cell-center easting/northing coordinate arrays from ``transform``
    without any striding, so the DEM can be clipped to the plot window at full
    resolution before being downsampled. Clipping first, then downsampling the
    (usually much smaller) window keeps the topography within the plot detailed
    instead of coarsening the whole DEM up front -- which is what previously made
    small-extent 3D plots look blocky.
    """
    dem = np.asarray(dem, dtype=float)
    nrows, ncols = dem.shape
    easting, northing = _axis_coordinates(transform, nrows, ncols)
    return dem, easting, northing


def downsample_bundle(dem_grid, max_samples=DEM_MAX_SAMPLES):
    """Downsample an ``(elevation, easting, northing)`` bundle to <= max samples.

    Companion to :func:`downsample_dem` that operates on an already-clipped
    bundle (rather than a full raster plus transform), striding each axis so
    neither exceeds ``max_samples`` (defaults to :data:`DEM_MAX_SAMPLES`).
    """
    elevation, easting, northing = dem_grid
    elevation = np.asarray(elevation, dtype=float)
    nrows, ncols = elevation.shape

    row_stride = max(int(np.ceil(nrows / max_samples)) if nrows > 0 else 1, 1)
    col_stride = max(int(np.ceil(ncols / max_samples)) if ncols > 0 else 1, 1)

    return (
        elevation[::row_stride, ::col_stride],
        np.asarray(easting, dtype=float)[::col_stride],
        np.asarray(northing, dtype=float)[::row_stride],
    )


def prepare_dem_bundle(
    dem_path, dst_crs, easting_limits, northing_limits, max_samples=DEM_MAX_SAMPLES
):
    """Load a DEM and return a clipped, downsampled ``(elev, easting, northing)``.

    Loads (optionally reprojecting), clips to the plot limits at full resolution,
    then downsamples the clipped window to at most ``max_samples`` per axis. This
    ordering preserves detail within small plot windows.
    """
    resolved_dem = resolve_dem_path(dem_path)
    elevation, transform = load_dem(resolved_dem, dst_crs=dst_crs)
    bundle = full_dem_bundle(elevation, transform)
    bundle = clip_dem_to_limits(bundle, easting_limits, northing_limits)
    return downsample_bundle(bundle, max_samples=max_samples)


def clip_dem_to_limits(dem_grid, easting_limits, northing_limits):
    """Clip a DEM bundle to the given horizontal limits.

    Keeps only the cells whose cell-center easting lies within
    ``easting_limits`` and whose cell-center northing lies within
    ``northing_limits`` (both inclusive).

    Parameters:
        dem_grid: A ``(elevation, easting, northing)`` bundle following the
            module's DEM return-contract (as produced by :func:`downsample_dem`).
        easting_limits (tuple[float, float]): ``(e_lo, e_hi)`` horizontal easting
            limits in the same coordinate frame as ``dem_grid`` easting.
        northing_limits (tuple[float, float]): ``(n_lo, n_hi)`` horizontal
            northing limits in the same coordinate frame as ``dem_grid``
            northing.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: The clipped
            ``(elevation, easting, northing)`` bundle.

    Raises:
        ValueError: If no cells fall within the limits, or if every retained
            cell has an invalid (NaN) elevation -- i.e. no valid elevation data
            remain within the plot limits.
    """
    elevation, easting, northing = dem_grid
    elevation = np.asarray(elevation, dtype=float)
    easting = np.asarray(easting, dtype=float)
    northing = np.asarray(northing, dtype=float)

    e_lo, e_hi = min(easting_limits), max(easting_limits)
    n_lo, n_hi = min(northing_limits), max(northing_limits)

    col_mask = (easting >= e_lo) & (easting <= e_hi)
    row_mask = (northing >= n_lo) & (northing <= n_hi)

    clipped = elevation[np.ix_(row_mask, col_mask)]

    if clipped.size == 0 or not np.any(np.isfinite(clipped)):
        raise ValueError(
            "No valid DEM elevation data within the plot horizontal limits "
            f"easting={tuple(easting_limits)!r}, northing={tuple(northing_limits)!r}."
        )

    return clipped, easting[col_mask], northing[row_mask]


# ---------------------------------------------------------------------------
# Vector rendering
# ---------------------------------------------------------------------------
# Default displacement-vector colors. These are
# applied without caller configuration. ``horizontal`` is the color of the
# per-station horizontal (ux, uy) vector; ``up``/``down`` are the colors of the
# vertical (uz) bar classified by the sign of uz.
VECTOR_COLORS = {
    "observed": {"horizontal": "black", "up": "red", "down": "blue"},
    "modeled": {"horizontal": "green", "up": "magenta", "down": "cyan"},
}

# In the 3D plot the vertical and horizontal components are combined into one
# 3-component vector per station, so a single color per kind is used instead of
# the separate horizontal/up/down colors of the 2D map view. Observed (blue)
# vectors are drawn slightly thinner than in the map view but still thicker than
# modeled (red) vectors.
VECTOR_3D_COLORS = {"observed": "blue", "modeled": "red"}
VECTOR_3D_LINEWIDTHS = {"observed": 1.6, "modeled": 1.0}

# Per-kind line widths. Observed vectors are drawn thicker than modeled vectors
# so that, where the two overlap at shared stations, both remain visible even
# though modeled vectors are drawn on top.
VECTOR_LINEWIDTHS = {"observed": 2.2, "modeled": 1.0}

# Fraction of the plot horizontal extent that the largest displacement magnitude
# should occupy when a common scale is computed automatically.
VECTOR_EXTENT_FRACTION = 0.15

# The station-position and displacement-component keys every Vmod_Vector_Set
# must provide, plus the optional uncertainty keys.
_VECTOR_REQUIRED_KEYS = ("x", "y", "ux", "uy", "uz")
_VECTOR_UNCERTAINTY_KEYS = ("errx", "erry", "errz")


def compute_common_scale(vector_sets, extent, fraction=VECTOR_EXTENT_FRACTION):
    """Compute a single common scale factor shared by all displacement vectors.

    One scale factor is derived from the union of the observed and modeled
    displacement magnitudes (both horizontal ``sqrt(ux**2 + uy**2)`` and vertical
    ``|uz|``) together with the plot horizontal ``extent`` so that the largest
    magnitude present occupies ``fraction`` (default 15%) of the extent. The same
    returned factor is then passed to every :func:`render_vectors` call so that
    observed and modeled vectors share one consistent scale.

    Parameters:
        vector_sets: An iterable of Vmod_Vector_Set dicts (or ``None`` entries,
            which are ignored). Each dict may contain ``ux``, ``uy``, ``uz``
            arrays.
        extent (float): The characteristic plot horizontal extent in UTM meters
            (e.g. the larger of the easting and northing ranges).
        fraction (float): The fraction of ``extent`` the largest magnitude should
            occupy. Defaults to :data:`VECTOR_EXTENT_FRACTION`.

    Returns:
        float: The common scale factor. Returns ``1.0`` when no positive
            displacement magnitude is present or ``extent`` is non-positive, so
            callers always receive a usable value.
    """
    extent = float(extent)
    max_mag = 0.0
    for vector_set in vector_sets:
        if vector_set is None:
            continue
        ux = np.asarray(vector_set.get("ux", []), dtype=float).ravel()
        uy = np.asarray(vector_set.get("uy", []), dtype=float).ravel()
        uz = np.asarray(vector_set.get("uz", []), dtype=float).ravel()

        if ux.size and uy.size and ux.size == uy.size:
            horiz = np.sqrt(ux ** 2 + uy ** 2)
            if horiz.size:
                candidate = np.nanmax(horiz)
                if np.isfinite(candidate):
                    max_mag = max(max_mag, float(candidate))
        if uz.size:
            candidate = np.nanmax(np.abs(uz))
            if np.isfinite(candidate):
                max_mag = max(max_mag, float(candidate))

    if max_mag <= 0.0 or extent <= 0.0:
        return 1.0
    return fraction * extent / max_mag


def _validate_vector_set(vector_set) -> dict:
    """Validate a Vmod_Vector_Set and return its arrays as a normalized dict.

    Ensures the required station-position and displacement-component arrays are
    present and that every present array (including any supplied uncertainty
    arrays) shares the same length. This validation runs before any plotting so
    that a mismatch is reported without partially drawing the figure.

    Parameters:
        vector_set: A Vmod_Vector_Set dict with keys ``x``, ``y``, ``ux``,
            ``uy``, ``uz`` and optional ``errx``, ``erry``, ``errz``.

    Returns:
        dict: A dict mapping each present key to a 1D ``float`` ``numpy.ndarray``.

    Raises:
        ValueError: If a required key is missing, or if the station-position and
            displacement-component arrays (or any supplied uncertainty arrays) do
            not all share the same length.
    """
    missing = [key for key in _VECTOR_REQUIRED_KEYS if vector_set.get(key) is None]
    if missing:
        raise ValueError(
            f"Vmod_Vector_Set is missing required key(s): {', '.join(missing)}."
        )

    arrays = {}
    lengths = {}
    for key in _VECTOR_REQUIRED_KEYS:
        arr = np.asarray(vector_set[key], dtype=float).ravel()
        arrays[key] = arr
        lengths[key] = arr.size

    for key in _VECTOR_UNCERTAINTY_KEYS:
        value = vector_set.get(key)
        if value is not None:
            arr = np.asarray(value, dtype=float).ravel()
            arrays[key] = arr
            lengths[key] = arr.size

    unique_lengths = set(lengths.values())
    if len(unique_lengths) > 1:
        detail = ", ".join(f"{key}={size}" for key, size in lengths.items())
        raise ValueError(
            "Vmod_Vector_Set arrays have mismatched lengths: " + detail + "."
        )

    return arrays


def _has_uncertainty(vector_set) -> bool:
    """Return True when all horizontal/vertical uncertainty arrays are present."""
    return all(vector_set.get(key) is not None for key in _VECTOR_UNCERTAINTY_KEYS)


def render_vectors(ax, vector_set, *, kind, scale, show_uncertainty) -> None:
    """Draw one observed or modeled displacement Vmod_Vector_Set on ``ax``.

    For each station this draws a horizontal displacement vector originating at
    the station position ``(x, y)`` (UTM meters) oriented along ``(ux, uy)`` and
    a vertical bar whose length represents ``uz``, applying the single common
    ``scale`` to every vector so observed and modeled sets share one scale
    Colors follow the fixed defaults:

    - horizontal vectors: observed = black, modeled = green;
    - vertical bar, ``uz > 0`` (up): observed = red, modeled = magenta;
    - vertical bar, ``uz < 0`` (down): observed = blue, modeled = cyan.

    When ``show_uncertainty`` is true and the set includes ``errx``/``erry``/
    ``errz``, a horizontal uncertainty ellipse (from the eigendecomposition of
    the horizontal covariance ``diag(errx**2, erry**2)``) is drawn at the
    horizontal vector tip and a vertical uncertainty bar (from ``errz``) is drawn
    centered at the vertical bar tip. When ``show_uncertainty``
    is false, no uncertainty indicators are drawn. When
    ``show_uncertainty`` is true but the uncertainties are absent, the vectors are
    drawn without indicators and a warning is emitted noting that uncertainty data
    was unavailable.

    On a 2D map-view axes the vertical ``uz`` bar is drawn northward from the
    station as a proxy for elevation (matching the convention in
    ``plot_gnss_vectors.py``). When ``ax`` is a 3D axes (``ax.name == "3d"``),
    the east/north/vertical components are instead combined into a single
    3-component vector per station, drawn from the station's ground elevation
    (taken from an optional ``z`` array in ``vector_set``, defaulting to 0) and
    colored by kind (observed = blue, modeled = red). Uncertainty indicators are
    only drawn on the 2D map view, never in 3D.

    Parameters:
        ax: A matplotlib 2D ``Axes`` or 3D ``Axes3D`` to draw on.
        vector_set: A Vmod_Vector_Set dict (see the module data model).
        kind (str): Either ``"observed"`` or ``"modeled"``.
        scale (float): The common scale factor (see :func:`compute_common_scale`).
        show_uncertainty (bool): Whether to draw uncertainty indicators.

    Returns:
        None

    Raises:
        ValueError: If ``kind`` is not ``"observed"`` or ``"modeled"``, or if the
            Vmod_Vector_Set arrays have mismatched lengths (validated before any
            drawing occurs).
    """
    if kind not in VECTOR_COLORS:
        raise ValueError(
            f"Invalid kind {kind!r}; expected 'observed' or 'modeled'."
        )

    # Validate lengths BEFORE any plotting.
    arrays = _validate_vector_set(vector_set)

    colors = VECTOR_COLORS[kind]
    vec3d_color = VECTOR_3D_COLORS[kind]
    scale = float(scale)
    is_3d = getattr(ax, "name", "") == "3d"

    # Line widths differ between the 2D map view and the 3D combined vectors.
    lw = VECTOR_3D_LINEWIDTHS[kind] if is_3d else VECTOR_LINEWIDTHS[kind]
    unc_lw = lw + 1.5  # (2D only) uncertainty bars a bit thicker than the vector

    x = arrays["x"]
    y = arrays["y"]
    ux = arrays["ux"]
    uy = arrays["uy"]
    uz = arrays["uz"]

    # Station elevations for the 3D case (optional; default to zero).
    if is_3d:
        z_value = vector_set.get("z")
        if z_value is not None:
            z = np.asarray(z_value, dtype=float).ravel()
        else:
            z = np.zeros_like(x)

    # Resolve uncertainty availability and warn if requested but unavailable.
    draw_uncertainty = False
    if show_uncertainty:
        if _has_uncertainty(vector_set):
            draw_uncertainty = True
        else:
            warnings.warn(
                f"Uncertainty display requested but the {kind} Vmod_Vector_Set "
                "does not include errx/erry/errz; drawing vectors without "
                "uncertainty indicators.",
                stacklevel=2,
            )

    if draw_uncertainty:
        errx = arrays["errx"]
        erry = arrays["erry"]
        errz = arrays["errz"]

    for i in range(x.size):
        # --- 3D: one combined 3-component vector per station ---
        # The east/north/vertical components are drawn as a single vector from
        # the station's ground position, colored by kind (observed=blue,
        # modeled=red). Uncertainties are not drawn in 3D.
        if is_3d:
            tip_e = x[i] + ux[i] * scale
            tip_n = y[i] + uy[i] * scale
            tip_z = z[i] + uz[i] * scale
            ax.plot(
                [x[i], tip_e], [y[i], tip_n], [z[i], tip_z],
                color=vec3d_color, linewidth=lw, zorder=5,
            )
            continue

        # --- 2D map view: separate horizontal vector + vertical bar ---
        # --- Horizontal displacement vector ---
        tip_e = x[i] + ux[i] * scale
        tip_n = y[i] + uy[i] * scale
        ax.plot(
            [x[i], tip_e], [y[i], tip_n],
            color=colors["horizontal"], linewidth=lw, zorder=5,
        )

        # --- Vertical (uz) bar drawn northward as an elevation proxy ---
        if uz[i] != 0 and np.isfinite(uz[i]):
            vcolor = colors["up"] if uz[i] > 0 else colors["down"]
            bar_length = uz[i] * scale
            vtip_e, vtip_n = x[i], y[i] + bar_length
            ax.plot(
                [x[i], vtip_e], [y[i], vtip_n],
                color=vcolor, linewidth=lw, zorder=5,
            )

            # --- Vertical uncertainty bar centered at the vertical tip ---
            if draw_uncertainty and np.isfinite(errz[i]) and errz[i] > 0:
                unc_half = errz[i] * scale / 2.0
                ax.plot(
                    [vtip_e, vtip_e],
                    [vtip_n - unc_half, vtip_n + unc_half],
                    color=vcolor, linewidth=unc_lw, alpha=0.3, zorder=3,
                )

        # --- Horizontal uncertainty ellipse at the horizontal tip ---
        if draw_uncertainty and np.isfinite(errx[i]) and np.isfinite(erry[i]):
            # Horizontal covariance from the east/north uncertainties. Only the
            # diagonal terms are available, so cross-correlation is treated as 0.
            covmat = np.array(
                [[errx[i] ** 2, 0.0], [0.0, erry[i] ** 2]], dtype=float
            )
            # eigh returns ascending eigenvalues for a symmetric matrix.
            eigenvalues, eigenvectors = np.linalg.eigh(covmat)
            semi_axes = np.sqrt(np.maximum(eigenvalues, 0.0)) * scale
            angle_rad = np.arctan2(eigenvectors[1, -1], eigenvectors[0, -1])

            theta = np.linspace(0.0, 2.0 * np.pi, 60)
            ell_major = semi_axes[-1] * np.cos(theta)
            ell_minor = semi_axes[0] * np.sin(theta)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            ell_e = cos_a * ell_major - sin_a * ell_minor + tip_e
            ell_n = sin_a * ell_major + cos_a * ell_minor + tip_n

            ax.plot(
                ell_e, ell_n,
                color=colors["horizontal"], linewidth=0.5, alpha=0.5,
                zorder=4,
            )


# ---------------------------------------------------------------------------
# Color resolution and figure saving
# ---------------------------------------------------------------------------
def resolve_base_color(base_color: str = DEFAULT_COLOR) -> tuple:
    """Validate and normalize a base color to an RGBA tuple.

    Any matplotlib color specification is accepted, including named colors
    (e.g. ``"orange"``), hex strings (e.g. ``"#ff8800"``), RGB(A) tuples, and
    grayscale strings. Validation is delegated to
    :func:`matplotlib.colors.to_rgba`.

    Parameters:
        base_color: A matplotlib color specification. Defaults to
            :data:`DEFAULT_COLOR` (``"orange"``) when not supplied.

    Returns:
        tuple: The color as an ``(r, g, b, a)`` tuple with components in
            ``[0, 1]``, used for both the 2D source outline and the 3D source
            surface rendering.

    Raises:
        ValueError: If ``base_color`` is not a valid matplotlib color
            specification. The message names the invalid color.
    """
    import matplotlib.colors as mcolors

    try:
        return mcolors.to_rgba(base_color)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Invalid base color {base_color!r}; expected a valid matplotlib "
            f"color specification."
        ) from exc


def save_figure(fig, save_path=None, dpi=FIG_DPI) -> None:
    """Save a matplotlib figure to a PNG file without closing it.

    The save path is resolved as follows:

    - When ``save_path`` is supplied, the figure is saved at that path.
    - When ``save_path`` is ``None``, the figure is saved into the default
      figures directory (:data:`DEFAULT_FIG_DIR`, i.e. ``data/figures/``
      relative to the repository root) under an auto-generated, timestamped
      filename of the form ``plot_utilities_YYYYmmdd_HHMMSS_ffffff.png``. The
      timestamp (including microseconds) keeps successive auto-named saves from
      clobbering one another.

    An existing file at the resolved path is overwritten.
    The figure is never closed, so the caller can continue to modify it after
    saving.

    Only the resolved PNG file is created or modified; no directories are
    created. If the parent directory of the resolved save path does not exist,
    a ``FileNotFoundError`` is raised and no file is written.

    Parameters:
        fig (matplotlib.figure.Figure): The figure to save.
        save_path: Optional path (str or ``pathlib.Path``) for the output PNG.
            When ``None``, a timestamped file is written into the default
            figures directory.
        dpi: Output resolution in dots per inch. Defaults to :data:`FIG_DPI`
            (400).

    Raises:
        FileNotFoundError: If the parent directory of the resolved save path
            does not exist. The message identifies the missing directory.
    """
    if save_path is not None:
        path = Path(save_path)
    else:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"plot_utilities_{timestamp}.png"
        path = REPO_ROOT / DEFAULT_FIG_DIR / filename

    parent = path.parent
    if not parent.is_dir():
        raise FileNotFoundError(
            f"Cannot save figure: the parent directory {str(parent)!r} does "
            f"not exist. No file was created."
        )

    fig.savefig(path, format="png", dpi=dpi, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Public API: Mapview_Plotter
# ---------------------------------------------------------------------------
def _vector_station_points(*vector_sets):
    """Return an ``(m, 2)`` array of station ``(E, N)`` positions.

    Gathers the station positions from every supplied (non-``None``)
    Vmod_Vector_Set for use when auto-computing horizontal limits. Returns an
    empty ``(0, 2)`` array when no vector sets are supplied.
    """
    points = []
    for vector_set in vector_sets:
        if vector_set is None:
            continue
        x = np.asarray(vector_set["x"], dtype=float).ravel()
        y = np.asarray(vector_set["y"], dtype=float).ravel()
        points.append(np.column_stack((x, y)))
    if not points:
        return np.empty((0, 2), dtype=float)
    return np.vstack(points)


# ---------------------------------------------------------------------------
# Topography (slope) shading -- shared by the map-view and 3D plotters
# ---------------------------------------------------------------------------
def compute_slope(elevation, easting, northing) -> np.ndarray:
    """Compute the topographic slope magnitude of a DEM bundle.

    Mirrors the slope calculation in ``plot_gnss_vectors.py`` (centered finite
    differences of elevation with respect to the horizontal coordinates), so the
    map-view and 3D topography are shaded consistently with that script. The
    slope magnitude is ``sqrt((dz/de)**2 + (dz/dn)**2)`` in rise-over-run units.

    Parameters:
        elevation (numpy.ndarray): The ``(nrows, ncols)`` elevation grid in
            meters (as produced by :func:`clip_dem_to_limits`). May contain NaN.
        easting (numpy.ndarray): The length-``ncols`` cell-center easting
            coordinates in meters.
        northing (numpy.ndarray): The length-``nrows`` cell-center northing
            coordinates in meters.

    Returns:
        numpy.ndarray: A ``(nrows, ncols)`` slope-magnitude grid. Cells whose
            slope cannot be evaluated (NaN neighbors) are returned as NaN.
    """
    elevation = np.asarray(elevation, dtype=float)
    easting = np.asarray(easting, dtype=float)
    northing = np.asarray(northing, dtype=float)

    if elevation.ndim != 2 or elevation.shape[0] < 2 or elevation.shape[1] < 2:
        return np.zeros_like(elevation)

    # np.gradient with coordinate arrays handles non-uniform and descending
    # (north-up) spacing; only the slope magnitude is used, so the sign of the
    # spacing is irrelevant. axis 0 -> northing (rows), axis 1 -> easting (cols).
    dz_dn, dz_de = np.gradient(elevation, northing, easting)
    return np.sqrt(dz_de ** 2 + dz_dn ** 2)


def slope_to_rgba(slope, alpha=1.0) -> np.ndarray:
    """Map a slope-magnitude grid to ``gray_r`` RGBA colors (steeper -> darker).

    Parameters:
        slope (numpy.ndarray): A slope-magnitude grid (as produced by
            :func:`compute_slope`). May contain NaN.
        alpha (float): The opacity applied to every color's alpha channel.

    Returns:
        numpy.ndarray: An ``(..., 4)`` RGBA array with the same leading shape as
            ``slope``. NaN cells are colored as the colormap's zero value.
    """
    from matplotlib.colors import Normalize

    slope = np.asarray(slope, dtype=float)
    finite = slope[np.isfinite(slope)]
    vmax = float(np.max(finite)) if finite.size else 0.0
    norm = Normalize(vmin=0.0, vmax=vmax if vmax > 0 else 1.0)
    cmap = plt.get_cmap(DEM_CMAP)
    rgba = cmap(norm(np.nan_to_num(slope, nan=0.0)))
    rgba[..., 3] = alpha
    return rgba


def _load_topography(dem_path, dst_crs, e_limits, n_limits):
    """Load, downsample, and clip a DEM, returning a slope-shaded bundle.

    Returns ``(elevation, easting, northing, slope)`` clipped to the plot
    limits, or ``None`` (with a warning) if the DEM cannot be loaded or holds no
    valid data within the limits -- so topography is best-effort and never
    prevents a plot from being produced.
    """
    try:
        dem_elev, dem_e, dem_n = prepare_dem_bundle(
            dem_path, dst_crs, e_limits, n_limits
        )
    except (FileNotFoundError, ValueError) as exc:
        warnings.warn(
            f"Topography could not be rendered ({exc}); plotting without it.",
            stacklevel=2,
        )
        return None
    slope = compute_slope(dem_elev, dem_e, dem_n)
    return dem_elev, dem_e, dem_n, slope


def _max_vector_magnitude(vector_sets) -> float:
    """Return the largest displacement magnitude (meters) across vector sets.

    Considers both horizontal (``sqrt(ux**2 + uy**2)``) and vertical (``|uz|``)
    magnitudes. Returns ``0.0`` when no finite positive magnitude is present.
    """
    max_mag = 0.0
    for vector_set in vector_sets:
        if vector_set is None:
            continue
        ux = np.asarray(vector_set.get("ux", []), dtype=float).ravel()
        uy = np.asarray(vector_set.get("uy", []), dtype=float).ravel()
        uz = np.asarray(vector_set.get("uz", []), dtype=float).ravel()
        if ux.size and uy.size and ux.size == uy.size:
            horiz = np.sqrt(ux ** 2 + uy ** 2)
            if horiz.size:
                candidate = np.nanmax(horiz)
                if np.isfinite(candidate):
                    max_mag = max(max_mag, float(candidate))
        if uz.size:
            candidate = np.nanmax(np.abs(uz))
            if np.isfinite(candidate):
                max_mag = max(max_mag, float(candidate))
    return max_mag


def _draw_vector_scale_bar(ax, scale, max_mag_m) -> None:
    """Draw a labelled displacement reference (scale) bar in the lower-left.

    A horizontal black bar whose length equals a round-number reference
    displacement (scaled by the shared vector ``scale``) is drawn in axes-limit
    coordinates, matching the displacement-magnitude reference in
    ``plot_gnss_vectors.py``. No bar is drawn when there is no positive
    displacement to scale against.
    """
    if not np.isfinite(max_mag_m) or max_mag_m <= 0 or scale <= 0:
        return

    # Pick a nice round reference magnitude (mm) near half the max magnitude.
    max_mm = max_mag_m * 1000.0
    nice_mm = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    target = max_mm * 0.5
    ref_mm = min(nice_mm, key=lambda v: abs(v - target))
    bar_len = (ref_mm / 1000.0) * scale  # plot meters

    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    x0 = x_lo + 0.05 * (x_hi - x_lo)
    y0 = y_lo + 0.14 * (y_hi - y_lo)

    ax.plot([x0, x0 + bar_len], [y0, y0], color="black", linewidth=2.0, zorder=8)
    ax.text(
        x0 + bar_len / 2.0,
        y0 - 0.02 * (y_hi - y_lo),
        f"{ref_mm:g} mm",
        ha="center", va="top", fontsize=9, zorder=8,
    )


def _add_vector_legend(ax, observed, modeled) -> None:
    """Add a legend describing the observed/modeled vector color scheme.

    Only entries for the supplied (non-``None``) vector sets are included. The
    vertical entries note the up/down color convention in their labels.
    """
    from matplotlib.lines import Line2D

    handles = []
    if observed is not None:
        lw = VECTOR_LINEWIDTHS["observed"]
        handles.append(
            Line2D([0], [0], color="black", linewidth=lw,
                   label="Observed horizontal")
        )
        handles.append(
            Line2D([0], [0], color="red", linewidth=lw,
                   label="Observed vertical (up=red, down=blue)")
        )
    if modeled is not None:
        lw = VECTOR_LINEWIDTHS["modeled"]
        handles.append(
            Line2D([0], [0], color="green", linewidth=lw,
                   label="Modeled horizontal")
        )
        handles.append(
            Line2D([0], [0], color="magenta", linewidth=lw,
                   label="Modeled vertical (up=magenta, down=cyan)")
        )
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.85)


def plot_mapview(
    source,
    params,
    *,
    observed=None,
    modeled=None,
    show_uncertainty=False,
    base_color=DEFAULT_COLOR,
    easting_limits=None,
    northing_limits=None,
    crop_inset=None,
    reference_elevation=0.0,
    dem_path=None,
    reproject_dem=False,
    show_topography=True,
    save_path=None,
    lv_info_path=None,
):
    """Produce a 2D map-view plot of a single vmod deformation source.

    Orchestrates the pure-computation and rendering helpers of this module to
    draw the source footprint outline (and, optionally, observed and modeled
    displacement vectors) in the local UTM coordinate frame, with axis labels
    expressed in kilometers. The figure is saved as a PNG and returned open so
    the caller can further customize it.

    The pipeline is: resolve/validate/unpack the source, determine the UTM zone
    from ``LV_info`` (:func:`get_utm_zone`), build the 2D outline
    (:func:`build_2d_outline`), validate any supplied limits
    (:func:`validate_limits`) or compute them automatically
    (:func:`compute_horizontal_limits`), optionally overlay displacement vectors
    with a single common scale (:func:`render_vectors`), draw the outline in the
    resolved base color, label the axes in kilometers, and save the figure
    (:func:`save_figure`).

    Parameters:
        source: A vmod ``Source`` object or a source-type string
            ("penny"/"yang"/"mctigue").
        params: The ordered :term:`Parameter_Vector` for the source.
        observed: An optional observed Vmod_Vector_Set to overlay.
        modeled: An optional modeled Vmod_Vector_Set to overlay.
        show_uncertainty (bool): Whether to draw uncertainty indicators for the
            supplied vectors. Defaults to ``False``.
        base_color: A matplotlib color spec for the source outline. Defaults to
            :data:`DEFAULT_COLOR` (``"orange"``).
        easting_limits: An optional ``(lo, hi)`` easting pair in UTM meters. When
            ``None``, limits are computed automatically.
        northing_limits: An optional ``(lo, hi)`` northing pair in UTM meters.
            When ``None``, limits are computed automatically.
        crop_inset: Optional inset, in meters, subtracted from each side of the
            auto-computed horizontal limits to crop the view more tightly (e.g.
            ``5000`` trims ~5 km from every edge). Applied only to axes whose
            limits are auto-computed, and skipped for an axis if it would
            collapse the limits. Defaults to ``None`` (no extra crop).
        reference_elevation (float): Elevation (meters) from which the source
            depth is measured. Retained for API symmetry with
            :func:`plot_source_3d`; the map view is plan-view only. Defaults to
            0.0.
        dem_path: Optional path to a DEM GeoTIFF used for the slope-shaded
            topographic background. When ``None``, the default project DEM is
            used.
        reproject_dem (bool): When ``True``, reproject the DEM into the local UTM
            zone before shading. Use this when the DEM is stored in a geographic
            (lat/lon) CRS -- such as the default project DEM -- since the map
            view operates in UTM meters. Defaults to ``False``.
        show_topography (bool): When ``True`` (the default), draw a slope-shaded
            (``gray_r``) topographic background clipped to the plot limits,
            matching ``plot_gnss_vectors.py``. Topography rendering is
            best-effort: if the DEM is missing or holds no data within the
            limits, a warning is emitted and the plot is produced without it.
        save_path: Optional output PNG path. When ``None``, a timestamped file is
            written into the default figures directory.
        lv_info_path: Optional override for the ``LV_info.json`` path.

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: The figure and its
            2D axes. The figure is left open.

    Raises:
        ValueError: For an unsupported source type, a wrong parameter count,
            invalid source dimensions, invalid supplied limits, mismatched
            Vmod_Vector_Set arrays, an invalid base color, or unreadable
            ``LV_info``.
        FileNotFoundError: If the parent directory of the resolved save path does
            not exist.
    """
    # --- Source resolution and parameter handling ---
    source_type = resolve_source_type(source)
    fields = unpack_parameters(source_type, params)

    # --- Coordinate frame: determine the UTM zone from LV_info
    # Raises if LV_info is missing/invalid before any
    # plotting occurs. The zone also defines the target CRS when reprojecting a
    # geographic DEM into the local UTM frame for the topographic background.
    zone_number, southern = get_utm_zone(lv_info_path)

    # --- Validate any caller-supplied limits before touching the figure
    # No depth limit applies to the map view.
    validate_limits(easting_limits, northing_limits, None)

    # --- Validate supplied vector sets up front so a length mismatch is
    # reported before a plot is produced.
    if observed is not None:
        _validate_vector_set(observed)
    if modeled is not None:
        _validate_vector_set(modeled)

    # --- Resolve the base color before rendering.
    color = resolve_base_color(base_color)

    # --- Build the 2D source footprint outline. ---
    outline = build_2d_outline(source_type, fields)

    # --- Determine whether any vectors are displayed. ---
    has_vectors = observed is not None or modeled is not None
    vector_pts = _vector_station_points(observed, modeled)

    # --- Horizontal limits: use supplied values or compute automatically
    # Auto limits enclose the geometry plus any
    # displayed vector positions with a 10% margin per side.
    auto_easting, auto_northing = compute_horizontal_limits(outline, vector_pts)
    e_limits = easting_limits if easting_limits is not None else auto_easting
    n_limits = northing_limits if northing_limits is not None else auto_northing

    # --- Optionally crop the auto-computed limits inward for a tighter view. ---
    if crop_inset:
        if easting_limits is None:
            cropped = (e_limits[0] + crop_inset, e_limits[1] - crop_inset)
            if cropped[0] < cropped[1]:
                e_limits = cropped
        if northing_limits is None:
            cropped = (n_limits[0] + crop_inset, n_limits[1] - crop_inset)
            if cropped[0] < cropped[1]:
                n_limits = cropped

    # --- Create the figure and axes. ---
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # --- Slope-shaded topographic background (borrowed from
    # plot_gnss_vectors.py). Drawn beneath everything else. Best-effort: a
    # missing DEM or empty clip only emits a warning. ---
    if show_topography:
        dst_crs = utm_crs_from_zone(zone_number, southern) if reproject_dem else None
        topo = _load_topography(dem_path, dst_crs, e_limits, n_limits)
        if topo is not None:
            dem_elev, dem_e, dem_n, slope = topo
            # imshow extent = (left, right, bottom, top); the northing axis of a
            # north-up DEM is descending, so origin="upper" places row 0 at top.
            img_extent = [dem_e[0], dem_e[-1], dem_n[-1], dem_n[0]]
            ax.imshow(
                slope,
                origin="upper",
                extent=img_extent,
                cmap=DEM_CMAP,
                alpha=0.7,
                zorder=0,
            )

    # --- Optional displacement vectors, sharing a single common scale
    # The vector renderer is skipped entirely when no
    # vector sets are supplied. Observed vectors are drawn
    # first (thicker) and modeled vectors on top (thinner) so both stay visible.
    scale = None
    max_mag = 0.0
    if has_vectors:
        extent = max(e_limits[1] - e_limits[0], n_limits[1] - n_limits[0])
        scale = compute_common_scale((observed, modeled), extent)
        max_mag = _max_vector_magnitude((observed, modeled))
        if observed is not None:
            render_vectors(
                ax, observed, kind="observed", scale=scale,
                show_uncertainty=show_uncertainty,
            )
        if modeled is not None:
            render_vectors(
                ax, modeled, kind="modeled", scale=scale,
                show_uncertainty=show_uncertainty,
            )

    # --- Draw the source outline in the resolved base color
        ax.plot(outline[:, 0], outline[:, 1], color=color, linewidth=1.5, zorder=6)

    # --- Apply limits and km axis labels. ---
    ax.set_xlim(e_limits)
    ax.set_ylim(n_limits)
    ax.set_aspect("equal", adjustable="box")

    formatter = km_axis_formatter()
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.set_xlabel("UTM Easting (km)")
    ax.set_ylabel("UTM Northing (km)")

    # --- Vector displacement scale bar and legend (drawn after limits are set
    # so the corner placement uses the final axis extents). ---
    if has_vectors:
        _draw_vector_scale_bar(ax, scale, max_mag)
        _add_vector_legend(ax, observed, modeled)

    # --- Save the PNG and leave the figure open ---
    save_figure(fig, save_path)

    return fig, ax

# ---------------------------------------------------------------------------
# Public API: Source_Plotter_3D
# ---------------------------------------------------------------------------
def _nearest_indices(coords, values):
    """Return, for each value, the index of the nearest entry in ``coords``.

    Both inputs are treated as 1D. Works regardless of whether ``coords`` is
    ascending or descending (e.g. a north-up DEM's northing axis), which is why
    an ``argmin`` over absolute differences is used rather than ``searchsorted``.

    Parameters:
        coords: 1D array-like of monotonic coordinate values (length ``L``).
        values: 1D array-like of query positions (length ``M``).

    Returns:
        numpy.ndarray: Integer indices into ``coords`` of shape ``(M,)``.
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float).ravel()
    if coords.size == 0:
        return np.zeros(values.shape, dtype=int)
    return np.argmin(np.abs(coords[np.newaxis, :] - values[:, np.newaxis]), axis=1)


def _sample_dem_elevation(dem_bundle, east_pts, north_pts):
    """Sample DEM ground elevations at the given ``(E, N)`` positions.

    Uses nearest-neighbor lookup against the clipped DEM cell-center coordinate
    arrays. Positions outside the DEM extent are clamped to the nearest cell.

    Parameters:
        dem_bundle: A ``(elevation, easting, northing)`` DEM bundle following the
            module's DEM return-contract (as produced by
            :func:`clip_dem_to_limits`).
        east_pts: Array-like of query eastings in UTM meters.
        north_pts: Array-like of query northings in UTM meters.

    Returns:
        numpy.ndarray: Ground elevations in meters, one per query position
            (invalid DEM cells yield NaN).
    """
    elevation, easting, northing = dem_bundle
    elevation = np.asarray(elevation, dtype=float)
    col_idx = _nearest_indices(easting, east_pts)
    row_idx = _nearest_indices(northing, north_pts)
    return elevation[row_idx, col_idx]


def _convex_hull_2d(points):
    """Return the closed convex-hull polygon of 2D ``points``.

    Uses Andrew's monotone-chain algorithm (numpy only). The returned array is
    the ordered hull vertices with the first vertex repeated at the end so the
    polygon is closed. Used to draw the source's projected silhouette on the
    vertical bounding walls of the 3D plot.
    """
    pts = np.asarray(points, dtype=float)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    # Remove duplicate points to keep the cross-product tests well behaved.
    keep = np.ones(len(pts), dtype=bool)
    keep[1:] = np.any(np.diff(pts, axis=0) != 0, axis=1)
    pts = pts[keep]
    if len(pts) <= 2:
        return np.vstack([pts, pts[:1]]) if len(pts) else pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.array(lower[:-1] + upper[:-1])
    return np.vstack([hull, hull[:1]])


def plot_source_3d(
    source,
    params,
    *,
    observed=None,
    modeled=None,
    base_color=DEFAULT_COLOR,
    easting_limits=None,
    northing_limits=None,
    depth_limit=None,
    reference_elevation=0.0,
    dem_path=None,
    reproject_dem=False,
    save_path=None,
    lv_info_path=None,
):
    """Produce a 3D plot of a single vmod deformation source with topography.

    Orchestrates the module's pure-computation and rendering helpers to draw the
    source as a shaded ellipsoidal surface below the topographic (DEM) surface,
    with the source footprint outline drawn on the ground surface and optional
    observed/modeled displacement vectors overlaid. The plot is rendered in the
    local UTM coordinate frame using the ``mpl_toolkits.mplot3d`` backend, with
    all three axes labelled in kilometers. The figure is saved as a PNG and
    returned open so the caller can further customize it.

    The pipeline is: resolve/validate/unpack the source, determine the UTM zone
    from ``LV_info`` (:func:`get_utm_zone`), validate any supplied limits
    (:func:`validate_limits`), resolve the base color, resolve the DEM path,
    build the 3D source surface (:func:`build_3d_surface`) and its downward
    shading (:func:`surface_normal_shading`), build the footprint outline
    (:func:`build_2d_outline`), compute auto horizontal/depth limits when not
    supplied (:func:`compute_horizontal_limits`, :func:`compute_depth_limit`),
    load/downsample/clip the DEM (:func:`load_dem`, :func:`downsample_dem`,
    :func:`clip_dem_to_limits`), draw the shaded source surface, the ground
    footprint outline, optional displacement vectors with a single common scale,
    and the semi-transparent DEM surface, set an equal aspect ratio across the
    easting/northing/elevation axes, span the vertical axis from the maximum DEM
    elevation down to the depth limit, label the axes in kilometers, and save the
    figure (:func:`save_figure`).

    Parameters:
        source: A vmod ``Source`` object or a source-type string
            ("penny"/"yang"/"mctigue").
        params: The ordered :term:`Parameter_Vector` for the source.
        observed: An optional observed Vmod_Vector_Set to overlay (blue).
        modeled: An optional modeled Vmod_Vector_Set to overlay (red).
        base_color: A matplotlib color spec for the source surface and outline.
            Defaults to :data:`DEFAULT_COLOR` (``"orange"``).
        easting_limits: An optional ``(lo, hi)`` easting pair in UTM meters. When
            ``None``, limits are computed automatically.
        northing_limits: An optional ``(lo, hi)`` northing pair in UTM meters.
            When ``None``, limits are computed automatically.
        depth_limit: An optional positive-down depth limit in meters. When
            ``None``, it is computed automatically from the source geometry.
        reference_elevation (float): Elevation (meters) from which the source
            depth is measured downward (positive-down). Defaults to 0.0.
        dem_path: Optional path to a DEM GeoTIFF. When ``None``, the default
            :data:`DEFAULT_DEM` relative to the repository root is used.
        reproject_dem (bool): When ``True``, reproject the DEM into the local
            UTM zone (derived from ``LV_info``) before rendering. Use this when
            the DEM is stored in a geographic (lat/lon) CRS -- such as the
            default project DEM -- since the rest of the plot operates in UTM
            meters. Defaults to ``False`` (DEM assumed already in the UTM frame).
        save_path: Optional output PNG path. When ``None``, a timestamped file is
            written into the default figures directory.
        lv_info_path: Optional override for the ``LV_info.json`` path.

    Returns:
        tuple[matplotlib.figure.Figure, mpl_toolkits.mplot3d.axes3d.Axes3D]: The
            figure and its 3D axes. The figure is left open.

    Raises:
        ValueError: For an unsupported source type, a wrong parameter count,
            invalid source dimensions, invalid supplied limits, mismatched
            Vmod_Vector_Set arrays, an invalid base color, unreadable
            ``LV_info``, or no valid DEM data within the plot limits.
        FileNotFoundError: If the DEM file is missing, or if the parent directory
            of the resolved save path does not exist.
    """
    # --- Source resolution and parameter handling ---
    source_type = resolve_source_type(source)
    fields = unpack_parameters(source_type, params)

    # --- Coordinate frame: determine the UTM zone from LV_info
    # Raises if LV_info is missing/invalid before any
    # plotting occurs. The zone also defines the target CRS when reprojecting a
    # geographic DEM into the local UTM frame.
    zone_number, southern = get_utm_zone(lv_info_path)

    # --- Validate any caller-supplied limits before touching the figure ---
    validate_limits(easting_limits, northing_limits, depth_limit)

    # --- Validate supplied vector sets up front so a length mismatch is
    # reported before a plot is produced. ---
    if observed is not None:
        _validate_vector_set(observed)
    if modeled is not None:
        _validate_vector_set(modeled)

    # --- Resolve the base color before rendering.
    color = resolve_base_color(base_color)

    # --- Resolve the DEM path up front so a missing DEM is reported before a
    # plot is produced. ---
    resolved_dem = resolve_dem_path(dem_path)

    # --- Build the 3D source surface and its downward-lighting shading ---
    surf_x, surf_y, surf_z = build_3d_surface(source_type, fields, reference_elevation)
    shading = surface_normal_shading(surf_x, surf_y, surf_z)

    # --- Build the source footprint outline. ---
    outline = build_2d_outline(source_type, fields)

    # --- Horizontal limits: use supplied values or compute automatically
    # Auto limits enclose the source footprint plus
    # any displayed vector positions with a 10% margin per side. ---
    has_vectors = observed is not None or modeled is not None
    vector_pts = _vector_station_points(observed, modeled)
    auto_easting, auto_northing = compute_horizontal_limits(outline, vector_pts)
    e_limits = easting_limits if easting_limits is not None else auto_easting
    n_limits = northing_limits if northing_limits is not None else auto_northing

    # --- Depth limit: use supplied value or compute automatically
    # Positive-down meters. ---
    d_limit = depth_limit if depth_limit is not None else compute_depth_limit(fields)

    # --- Load, downsample, and clip the DEM to the horizontal limits
    # Clipping raises if no valid data remain. When
    # requested, reproject the DEM into the local UTM zone so a geographic DEM
    # (e.g. the default project DEM) aligns with the UTM-meter plot frame. ---
    dst_crs = utm_crs_from_zone(zone_number, southern) if reproject_dem else None
    dem_bundle = prepare_dem_bundle(
        resolved_dem, dst_crs, e_limits, n_limits,
        max_samples=DEM_3D_MAX_SAMPLES,
    )
    dem_elev, dem_easting, dem_northing = dem_bundle

    # --- Vertical axis extent: from the maximum DEM elevation within the limits
    # down to the (positive-down) depth limit. ---
    max_dem_elev = float(np.nanmax(dem_elev))
    z_bottom = depth_to_elevation(d_limit, reference_elevation)
    z_top = max_dem_elev

    # --- Slope shading for the DEM surface (borrowed from
    # plot_gnss_vectors.py): steeper terrain is darker via the gray_r colormap.
    dem_slope = compute_slope(dem_elev, dem_easting, dem_northing)

    # --- Create the 3D figure/axes using the mpl_toolkits.mplot3d backend. ---
    fig = plt.figure(figsize=FIG_SIZE)
    ax3d = fig.add_subplot(projection="3d")

    # --- Keep the 3D axis gridlines (helps read positions along each axis),
    # while the DEM surface itself is still drawn without its own wireframe
    # mesh (edgecolor="none" on that plot_surface call). ---
    ax3d.grid(True)

    # --- Shaded source surface in the resolved base color
    # The per-facet brightness scales the base color's RGB. The base
    # shading spans [SHADE_MIN, SHADE_MAX]; remap it down to SOURCE_SHADE_MIN to
    # exaggerate the top-lit/bottom-dark contrast of the source. ---
    shading = SOURCE_SHADE_MIN + (shading - SHADE_MIN) * (
        (SHADE_MAX - SOURCE_SHADE_MIN) / (SHADE_MAX - SHADE_MIN)
    )
    shading = np.clip(shading, SOURCE_SHADE_MIN, SHADE_MAX)

    base_rgb = np.asarray(color[:3], dtype=float)
    base_alpha = float(color[3]) if len(color) > 3 else 1.0
    facecolors = np.empty(shading.shape + (4,), dtype=float)
    facecolors[..., :3] = np.clip(
        base_rgb[np.newaxis, np.newaxis, :] * shading[..., np.newaxis], 0.0, 1.0
    )
    facecolors[..., 3] = base_alpha
    # A single shaded surface with thin black facet edges (rather than a separate
    # wireframe overlay, which rendered inconsistently). The coarse source mesh
    # (N_THETA x N_PHI) keeps the edge grid legible without crowding.
    ax3d.plot_surface(
        surf_x, surf_y, surf_z,
        facecolors=facecolors, shade=False,
        edgecolor="black", linewidth=SOURCE_WIREFRAME_LW,
        antialiased=True, zorder=4,
    )

    # --- Source footprint outline drawn on the topographic ground surface
    # vertically above the footprint. ---
    outline_z = _sample_dem_elevation(dem_bundle, outline[:, 0], outline[:, 1])
    ax3d.plot(
        outline[:, 0], outline[:, 1], outline_z,
        color=color, linewidth=1.5, zorder=6,
    )

    # --- Source silhouettes projected onto the four vertical bounding walls
    # (east+/east-/north+/north-), in addition to the ground-surface outline.
    # The silhouette is the convex hull of the 3D surface points projected onto
    # each wall plane. ---
    surf_pts_e = np.column_stack((surf_x.ravel(), surf_z.ravel()))  # (E, Z)
    surf_pts_n = np.column_stack((surf_y.ravel(), surf_z.ravel()))  # (N, Z)
    hull_nz = _convex_hull_2d(surf_pts_n)  # for east walls: (N, Z)
    hull_ez = _convex_hull_2d(surf_pts_e)  # for north walls: (E, Z)
    for x_wall in (e_limits[0], e_limits[1]):
        ax3d.plot(
            np.full(hull_nz.shape[0], x_wall), hull_nz[:, 0], hull_nz[:, 1],
            color=color, linewidth=1.0, alpha=0.8, zorder=3,
        )
    for y_wall in (n_limits[0], n_limits[1]):
        ax3d.plot(
            hull_ez[:, 0], np.full(hull_ez.shape[0], y_wall), hull_ez[:, 1],
            color=color, linewidth=1.0, alpha=0.8, zorder=3,
        )

    # --- Optional displacement vectors, sharing a single common scale
    # Vectors originate at the station ground
    # elevation sampled from the DEM. A black dot marks each station, drawn last
    # (high zorder, depthshade off) so it sits in front of the vectors. ---
    if has_vectors:
        extent = max(e_limits[1] - e_limits[0], n_limits[1] - n_limits[0])
        scale = compute_common_scale((observed, modeled), extent)
        for vector_set, kind in ((observed, "observed"), (modeled, "modeled")):
            if vector_set is None:
                continue
            station_z = _sample_dem_elevation(
                dem_bundle, vector_set["x"], vector_set["y"]
            )
            placed = dict(vector_set)
            placed["z"] = station_z
            render_vectors(
                ax3d, placed, kind=kind, scale=scale,
                show_uncertainty=False,
            )
        # Station location dots, drawn after the vectors so they render on top.
        for vector_set in (observed, modeled):
            if vector_set is None:
                continue
            sx = np.asarray(vector_set["x"], dtype=float).ravel()
            sy = np.asarray(vector_set["y"], dtype=float).ravel()
            sz = _sample_dem_elevation(dem_bundle, sx, sy)
            ax3d.scatter(
                sx, sy, sz, color="black", s=12, depthshade=False, zorder=10,
            )

    # --- Semi-transparent, slope-shaded DEM topographic surface
    # Faces are colored by slope magnitude (gray_r, steeper
    # -> darker) to match plot_gnss_vectors.py, with no wireframe mesh drawn on
    # the surface (edgecolor="none", linewidth=0, antialiased=False). ---
    dem_ee, dem_nn = np.meshgrid(dem_easting, dem_northing)
    dem_facecolors = slope_to_rgba(dem_slope, alpha=DEM_OPACITY)
    # matplotlib's plot_surface defaults to rcount=ccount=50, which downsamples
    # the DEM to a 50x50 grid regardless of its true size (this is why the 3D
    # topography looked far coarser than the imshow-based map view). Pass
    # rcount/ccount equal to the grid shape so it renders at full resolution.
    dem_rows, dem_cols = dem_elev.shape
    ax3d.plot_surface(
        dem_ee, dem_nn, dem_elev,
        facecolors=dem_facecolors, shade=False,
        rcount=dem_rows, ccount=dem_cols,
        edgecolor="none", linewidth=0, antialiased=True, zorder=2,
    )

    # --- Apply limits. ---
    ax3d.set_xlim(e_limits)
    ax3d.set_ylim(n_limits)
    ax3d.set_zlim(z_bottom, z_top)

    # --- Equal aspect ratio across easting/northing/elevation so a given
    # distance in km occupies the same displayed length on each axis. ---
    e_range = e_limits[1] - e_limits[0]
    n_range = n_limits[1] - n_limits[0]
    z_range = z_top - z_bottom
    ax3d.set_box_aspect((e_range, n_range, z_range))

    # --- Kilometer axis labels on all three axes. ---
    formatter = km_axis_formatter()
    ax3d.xaxis.set_major_formatter(formatter)
    ax3d.yaxis.set_major_formatter(formatter)
    ax3d.zaxis.set_major_formatter(formatter)
    ax3d.set_xlabel("UTM Easting (km)")
    ax3d.set_ylabel("UTM Northing (km)")
    ax3d.set_zlabel("Elevation (km)")

    # --- Save the PNG and leave the figure open. ---
    save_figure(fig, save_path)

    return fig, ax3d