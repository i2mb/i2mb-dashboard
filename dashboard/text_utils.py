import string


def text_wrapper(text: str, num_lines=2, min_width=None):
    """Wrap text into num_lines. Text must be at least min_width before wrapping into num_lines"""
    if min_width is not None and len(text) < min_width:
        return [text]

    if num_lines <= 1:
        return [text]

    if num_lines >= len(text.split()):
        return text.split()

    mid_point = len(text) // num_lines
    left_space = text[:mid_point].rfind(" ", )
    right_space = text.find(" ", mid_point)

    lines = []
    wrapping_point = left_space
    if (mid_point - left_space) >= (right_space - mid_point):
        wrapping_point = right_space

    lines.append(text[:wrapping_point])
    lines.extend(text_wrapper(text[wrapping_point+1:], num_lines=num_lines-1, min_width=min_width))
    return lines


def text_fill(text: str, num_lines=2, min_width=None):
    return "\n".join(text_wrapper(text, num_lines=num_lines, min_width=min_width))


# Formating helper
class ExtendedFormatter(string.Formatter):
    """
    Extends the Formatter class to include the conversion of text to an upper case.
    """
    def convert_field(self, value, conversion):
        if conversion == "t":
            return value.title()

        return string.Formatter.convert_field(self, value, conversion)


class Template:
    """Compatibility object that can replace a template string where the format method will be called, to ensure that
    the call gets routed through the ExtendedFormatter Formatter"""
    def __init__(self, format_string):
        self.format_string = format_string
        self.formatter = ExtendedFormatter()

    def format(self, *args, **kwargs):
        return self.formatter.format(self.format_string, *args, **kwargs)
