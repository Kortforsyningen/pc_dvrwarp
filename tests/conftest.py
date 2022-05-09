import pytest
import laspy
from pathlib import Path

_TEST_DIR = Path(__file__).parent
_TEST_DATA_DIR = _TEST_DIR.joinpath('data')
INPUT_FILE_DIR = _TEST_DATA_DIR.joinpath('input')
EXPECTED_FILE_DIR = _TEST_DATA_DIR.joinpath('expected')

COLORIZATION_RASTER_PATH = INPUT_FILE_DIR.joinpath('color.tif')
LAS_WITH_OLD_EB_NO_SRS_PATH = INPUT_FILE_DIR.joinpath('las_with_old_eb_no_srs.las')
LAZ_WITH_NEW_EB_PATH = INPUT_FILE_DIR.joinpath('laz_with_new_eb.laz')
INPUT_PC_PATHS = [
    LAS_WITH_OLD_EB_NO_SRS_PATH,
    LAZ_WITH_NEW_EB_PATH,
]

EXPECTED_COLORIZED_PC_PATH = EXPECTED_FILE_DIR.joinpath('expected_colorized.las')

@pytest.fixture
def read_expected_las():
    lasdata = laspy.read(EXPECTED_COLORIZED_PC_PATH)
    return lasdata
