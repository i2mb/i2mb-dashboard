from matplotlib.colors import to_rgba


def rgba_to_rgb(c, alpha=None, background_color="white"):

    bgc_r, bgc_g, bgc_b, bgc_alpha = to_rgba(background_color)
    sc_r, sc_g, sc_b, sc_alpha = to_rgba(c, alpha)
    r = ((1 - sc_alpha) * bgc_r) + (sc_alpha * sc_r)
    g = ((1 - sc_alpha) * bgc_g) + (sc_alpha * sc_g)
    b = ((1 - sc_alpha) * bgc_b) + (sc_alpha * sc_b)
    return (r, g, b)
