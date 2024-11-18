from IPython.utils.wildcard import is_type
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.offsetbox import DrawingArea, TextArea, HPacker, VPacker
from matplotlib.text import Text
from matplotlib.textpath import TextPath


def table_legend(rows, cols, check_matrix, parent, check_mark=".", uncheck_mark="", combined_mark="X",
                 combined_value=-1,
                 check_mark_props=None,
                 uncheck_mark_props=None, **kwarg):
    # Create a new legend
    legend: Legend = parent.legend([], **kwarg)
    fontsize = legend._fontsize
    descent = 0.35 * fontsize * (legend.handleheight - 0.7)  # heuristic.
    height = fontsize*.75 * legend.handleheight - descent

    if check_mark_props is None:
        check_mark_props = {}

    text_props = dict(verticalalignment='center',
                      horizontalalignment='center',
                      fontproperties=legend.prop)

    # Create handler column
    header_row = HPacker(sep=legend.handletextpad * fontsize,
                         children=[DrawingArea(width=legend.handlelength * fontsize,
                                               height=height, xdescent=0., ydescent=descent)] + [
                                      TextArea(col, textprops=text_props) for col in cols])

    table_rows = [header_row]
    for handle, check_row in zip(rows, check_matrix):
        handlebox = DrawingArea(width=legend.handlelength * fontsize,
                                height=height,
                                xdescent=0., ydescent=descent)

        table_row = [handlebox]
        if isinstance(handle, Text):
            width = legend.handlelength * fontsize * 5 + legend.handletextpad * fontsize *4
            text_area = DrawingArea( width=width, height=height, xdescent=0., ydescent=descent)
            handle.set(**text_props)
            handle.set_x(width/2)
            text_area.add_artist(handle)
            table_rows.append(HPacker(sep=legend.handletextpad * fontsize, mode="expand", children=[text_area]))
            continue

        handler = legend.get_legend_handler(legend.get_legend_handler_map(), handle)
        handler.legend_artist(legend, handle, fontsize, handlebox)
        for col, checked in zip(cols, check_row):
            if isinstance(checked, str):
                mk_handlebox = TextArea(checked, textprops=text_props)
            else:
                mk_handlebox = DrawingArea(width=legend.handlelength * fontsize,
                                       height=height,
                                       xdescent=0., ydescent=descent)

                marker = (check_mark if checked == 1 else uncheck_mark if checked == 0 else combined_mark
                            if checked == -1 else TextPath([0,0], checked,))
                mk_handle = Line2D([], [], marker=marker,
                                   ls="", **check_mark_props)
                mk_handler = legend.get_legend_handler(legend.get_legend_handler_map(), mk_handle)
                mk_handler.legend_artist(legend, mk_handle, fontsize, mk_handlebox)

            table_row.append(mk_handlebox)

        table_rows.append(HPacker(sep=legend.handletextpad * fontsize, mode="expand", children=table_row))

    vp = legend.get_children()[0]
    vp.get_children().extend(table_rows)
    vp.set_figure(legend.figure)

    return legend


def table_legend_column_wise(rows, cols, check_matrix, parent, **kwarg):
    # Create a new legend
    legend: Legend = parent.legend([], **kwarg)
    fontsize = legend._fontsize
    descent = 0.35 * fontsize * (legend.handleheight - 0.7)  # heuristic.
    height = fontsize * legend.handleheight - descent

    # Create handler column
    column = [DrawingArea(width=legend.handlelength * fontsize,
                          height=height,
                          xdescent=0., ydescent=descent)]
    for handle in rows:
        handlebox = DrawingArea(width=legend.handlelength * fontsize,
                                height=height,
                                xdescent=0., ydescent=descent)
        handler = legend.get_legend_handler(legend.get_legend_handler_map(), handle)
        handler.legend_artist(legend, handle, fontsize, handlebox)
        column.append(handlebox)

    text_props = dict(
        verticalalignment='baseline',
        horizontalalignment='center',
        fontproperties=legend.prop)
    table = HPacker(children=[VPacker(children=column),
                              VPacker(children=[TextArea("Col 1", textprops=text_props)] + [
                                  TextArea(str(i), textprops=text_props) for i in range(len(rows))]),
                              VPacker(children=[TextArea("Col 2", textprops=text_props)] + [
                                  TextArea(str(i), textprops=text_props) for i in range(len(rows))])
                              ],
                    sep=legend.handletextpad * fontsize)
    vp = legend.get_children()[0]

    vp._children.extend([table])

    vp.set_figure(legend.figure)

    return legend
