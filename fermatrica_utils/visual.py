"""
Utilities for visual elements / tasks
"""


def hex_to_rgb(hex_code: str) -> tuple:
    """
    Convert HEX colour code to RGB

    :param hex_code:
    :return:
    """

    hex_code = hex_code.replace("#", "")
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
