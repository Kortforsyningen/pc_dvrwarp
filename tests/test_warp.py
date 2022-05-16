import pytest
from laspy.lasdata import LasData
from laspy.vlrs.known import LasZipVlr, WktCoordinateSystemVlr
from osgeo import osr
import numpy as np

from pathlib import Path
import subprocess
import json

# Maximum allowed deviations from expected data (in georeferenced units,
# meters) in output.
OUTPUT_XY_TOLERANCE = 0.01
OUTPUT_Z_TOLERANCE = 0.05 # some slack in case DVR90 should change slightly...

EXPECTED_SRS_EPSG = 7416
EXPECTED_SRS = osr.SpatialReference()
EXPECTED_SRS.ImportFromEPSG(EXPECTED_SRS_EPSG)

# We should be writing LAS/LAZ 1.4 files
EXPECTED_MAJOR_VERSION = 1
EXPECTED_MINOR_VERSION = 4

# Modern LAS 1.4 PDRFs without and with RGB color, respectively
EXPECTED_PDRF_WITHOUT_COLOR = 6
EXPECTED_PDRF_WITH_COLOR = 7

def assert_las_version_correct(actual_lasdata: LasData) -> None:
    assert actual_lasdata.header.major_version == EXPECTED_MAJOR_VERSION
    assert actual_lasdata.header.minor_version == EXPECTED_MINOR_VERSION

def assert_pdrf_equal(actual_lasdata: LasData, expected_format_id: int) -> None:
    assert actual_lasdata.point_format.id == expected_format_id

def assert_xy_approx_equal(actual_lasdata: LasData, expected_lasdata: LasData) -> None:
    actual_x = actual_lasdata.points.x
    actual_y = actual_lasdata.points.y

    expected_x = expected_lasdata.points.x
    expected_y = expected_lasdata.points.y

    actual_xy = np.column_stack([actual_x, actual_y])
    expected_xy = np.column_stack([expected_x, expected_y])

    np.testing.assert_allclose(actual_xy, expected_xy, rtol=0, atol=OUTPUT_XY_TOLERANCE)

def assert_z_approx_equal(actual_lasdata: LasData, expected_lasdata: LasData) -> None:
    np.testing.assert_allclose(actual_lasdata.points.z, expected_lasdata.points.z, rtol=0, atol=OUTPUT_Z_TOLERANCE)

def assert_rgb_equal(actual_lasdata: LasData, expected_lasdata: LasData) -> None:
    actual_red = actual_lasdata.points.red
    actual_green = actual_lasdata.points.green
    actual_blue = actual_lasdata.points.blue

    expected_red = expected_lasdata.points.red
    expected_green = expected_lasdata.points.green
    expected_blue = expected_lasdata.points.blue

    actual_rgb = np.column_stack([actual_red, actual_green, actual_blue])
    expected_rgb = np.column_stack([expected_red, expected_green, expected_blue])

    np.testing.assert_equal(actual_rgb, expected_rgb)

def assert_is_laz(path: Path, expected_is_laz: bool) -> None:
    laszip_user_id = LasZipVlr.official_user_id()
    
    # A bit hackish - Laspy seems to hide the presence of LASzip compression from the user, so inspect with PDAL instead
    pdal_info_output = subprocess.check_output(['pdal', 'info', '--metadata', str(path)])
    info_obj = json.loads(pdal_info_output)
    metadata = info_obj['metadata']
    vlrs = [metadata[key] for key in metadata if key.startswith('vlr_')]
    vlr_user_ids = [vlr['user_id'] for vlr in vlrs]

    has_laszip_vlr = laszip_user_id in vlr_user_ids
    assert has_laszip_vlr == expected_is_laz

def assert_srs_correct(actual_lasdata: LasData) -> None:
    coordinate_system_user_id = WktCoordinateSystemVlr.official_user_id()
    
    actual_wkt_vlr = actual_lasdata.vlrs.get_by_id(user_id=coordinate_system_user_id)[0]
    actual_srs = osr.SpatialReference()
    actual_srs.ImportFromWkt(actual_wkt_vlr.string)

    assert actual_srs.IsSame(EXPECTED_SRS)

def assert_extradim_descriptors_equal(actual_lasdata: LasData, expected_lasdata: LasData) -> None:
    # The `extra_dimensions` fields are generators of
    # laspy.point.dims.DimensionInfo elements. Convert to lists so we can
    # compare them with `==`.
    expected_extradims = list(expected_lasdata.point_format.extra_dimensions)
    actual_extradims = list(actual_lasdata.point_format.extra_dimensions)

    assert actual_extradims == expected_extradims

def get_extradim_raw_values(lasdata: LasData, extradim_name: str) -> np.ndarray:
    extradim = lasdata.points[extradim_name]

    # lasdata.points[extradim_name] may be either a ScaledArrayView or a
    # simple NumPy array, depending on the data type etc. Make sure we handle
    # both cases.
    try:
        # If it's a ScaledArrayView, we can grab a NumPy array of raw values
        # from the .array attribute...
        raw_values = extradim.array
    except AttributeError:
        # ...otherwise, the dimension is simply represented by a NumPy array
        raw_values = extradim

    return raw_values

def assert_extradim_raw_values_equal(actual_lasdata: LasData, expected_lasdata: LasData) -> None:
    expected_extradim_names = expected_lasdata.point_format.extra_dimension_names

    for extradim_name in expected_extradim_names:
        expected_extradim_values = get_extradim_raw_values(expected_lasdata, extradim_name)
        actual_extradim_values = get_extradim_raw_values(actual_lasdata, extradim_name)
        np.testing.assert_equal(actual_extradim_values, expected_extradim_values)

def assert_extradim_scaled_values_approx_equal(actual_lasdata: LasData, expected_lasdata: LasData) -> None:
    expected_extradim_names = expected_lasdata.point_format.extra_dimension_names

    for extradim_name in expected_extradim_names:
        # .points[extradim_name] may be either a ScaledArrayView or a simple
        # NumPy array, but can always be converted to NumPy arrays
        expected_values = np.array(expected_lasdata.points[extradim_name])
        actual_values = np.array(actual_lasdata.points[extradim_name])
        np.testing.assert_allclose(actual_values, expected_values)

def test_xy(output_lasdata, expected_lasdata):
    assert_xy_approx_equal(output_lasdata, expected_lasdata)

def test_z(output_lasdata, expected_lasdata):
    assert_z_approx_equal(output_lasdata, expected_lasdata)

def test_srs(output_lasdata):
    assert_srs_correct(output_lasdata)

def test_las_version(output_lasdata):
    assert_las_version_correct(output_lasdata)

def test_compression(output_path, write_compressed):
    assert_is_laz(output_path, write_compressed)

def test_pdrf(output_lasdata, colorization_raster):
    if colorization_raster is None:
        assert_pdrf_equal(output_lasdata, EXPECTED_PDRF_WITHOUT_COLOR)
    else:
        assert_pdrf_equal(output_lasdata, EXPECTED_PDRF_WITH_COLOR)

def test_rgb(output_lasdata, expected_lasdata, colorization_raster):
    if colorization_raster is None:
        pytest.skip('no colorization applied')
    
    assert_rgb_equal(output_lasdata, expected_lasdata)

def test_srs(output_lasdata):
    assert_srs_correct(output_lasdata)

# Extrabyte tests below.
#
# The "expected" values for the extrabyte dimensions are a verbatim
# carryover from the input data. PDAL tends to change datatype into float64
# and make other changes to extrabyte dimensions, so for the moment we only
# test that the ScaledArrayView representation of each dimension is the same
# (to within rounding errors).
#
# Once the implementation is ready, the more stringent extrabyte tests below
# should be used rather than the approximate test on scaled values.

# TODO This test is redundant once the more stringent extrabyte tests pass
def test_extradim_scaled_values(input_lasdata, output_lasdata, retain_extra_dims):
    if retain_extra_dims:
        assert_extradim_scaled_values_approx_equal(output_lasdata, input_lasdata)
    else:
        pytest.skip('extrabyte dimensions not retained')

# TODO Make this test pass
@pytest.mark.skip('currently unsupported')
def test_extradim_descriptors(input_lasdata, output_lasdata, retain_extra_dims):
    if retain_extra_dims:
        assert_extradim_descriptors_equal(output_lasdata, input_lasdata)
    else:
        pytest.skip('extrabyte dimensions not retained')

# TODO Make this test pass
@pytest.mark.skip('currently unsupported')
def test_extradim_raw_values(output_lasdata, input_lasdata, retain_extra_dims):
    if retain_extra_dims:
        assert_extradim_raw_values_equal(output_lasdata, input_lasdata)
    else:
        pytest.skip('extrabyte dimensions not retained')
