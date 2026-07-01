import pytest

@pytest.fixture
def if_debug():
    # set True to explore specific case (specific test class)
    # set False to run all tests
    return False