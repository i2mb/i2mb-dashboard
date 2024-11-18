import matplotlib as mpl


def figure_size_4K_UHD():
    """ Based on the current dpi, returns the figure size in inches corresponding to 4K (UHD)"""
    dpi = mpl.rcParams["figure.dpi"]
    return 3840 / dpi, 2160 / dpi


def figure_size_4K_DCI():
    """ Based on the current dpi, returns the figure size in inches corresponding to DCI-4K """
    dpi = mpl.rcParams["figure.dpi"]
    return 4096 / dpi, 2160 / dpi


def figure_size_HD():
    """ Based on the current dpi, returns the figure size in inches corresponding to HD"""
    dpi = mpl.rcParams["figure.dpi"]
    return 1920 / dpi, 1080 / dpi


def figure_size_8K():
    """ Based on the current dpi, returns the figure size in inches corresponding to 8K (UHD)"""
    dpi = mpl.rcParams["figure.dpi"]
    return 7680 / dpi, 4320 / dpi


def figure_size_HD720():
    """ Based on the current dpi, returns the figure size in inches corresponding to 8K (UHD)"""
    dpi = mpl.rcParams["figure.dpi"]

    # Adjust font size to fit better
    mpl.rcParams["font.size"] = 8
    return 1280 / dpi, 720 / dpi

