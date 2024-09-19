"""
Utilities for numerics (floats and integers) manipulation
"""


from line_profiler_pycharm import profile


@profile
def int_to_roman(number: int | float):
    """
    Convert integer or float number to Roman numerals. Floats to be rounded to integers before.
    Use to create neat numerical outputs for GUI, presentations etc.

    :param number:
    :return:
    """

    if not isinstance(number, int):
        number = int(round(number))

    roman_dict = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I")
    ]

    rtrn = []

    for (remainder, roman) in roman_dict:

        # division with remainder:
        #
        #   1. use the quotient as times to repeat roman literal, e.g. 28 // 10 = 2,
        #      so 'X' * 2 = ['X', 'X']
        #   2. the remainder replaces the original number for new iteration, e.g.
        #      28 % 10 = 8, so next iteration to use 8 as number

        (quotient, number) = divmod(number, remainder)
        rtrn.append(roman * quotient)

        if number == 0:
            break

    rtrn = "".join(rtrn)

    return rtrn

