from collections import namedtuple
from platform import python_version_tuple
import re


if python_version_tuple()[0] < "3":
    from itertools import izip_longest
    from functools import partial
    _none_type = type(None)
    _int_type = int
    _float_type = float
    _text_type = str
    _binary_type = str
else:
    from itertools import zip_longest as izip_longest
    from functools import reduce, partial
    _none_type = type(None)
    _int_type = int
    _float_type = float
    _text_type = str
    _binary_type = bytes


__all__ = ["tabulate", "tabulate_formats", "simple_separated_format"]
__version__ = "0.7.2"


Line = namedtuple("Line", ["begin", "hline", "sep", "end"])
DataRow = namedtuple("DataRow", ["begin", "sep", "end"])
TableFormat = namedtuple("TableFormat", ["lineabove", "linebelowheader",
                                         "linebetweenrows", "linebelow",
                                         "headerrow", "datarow",
                                         "padding", "with_header_hide"])


def _pipe_segment_with_colons(align, colwidth):
    w = colwidth
    if align in ["right", "decimal"]:
        return ('-' * (w - 1)) + ":"
    elif align == "center":
        return ":" + ('-' * (w - 2)) + ":"
    elif align == "left":
        return ":" + ('-' * (w - 1))
    else:
        return '-' * w


def _pipe_line_with_colons(colwidths, colaligns):
    segments = [_pipe_segment_with_colons(a, w) for a, w in zip(colaligns, colwidths)]
    return "|" + "|".join(segments) + "|"


def _mediawiki_row_with_attrs(separator, cell_values, colwidths, colaligns):
    alignment = { "left":    '',
                  "right":   'align="right"| ',
                  "center":  'align="center"| ',
                  "decimal": 'align="right"| ' }

    values_with_attrs = [' ' + alignment.get(a, '') + c + ' '
                         for c, a in zip(cell_values, colaligns)]
    colsep = separator*2
    return (separator + colsep.join(values_with_attrs)).rstrip()


def _latex_line_begin_tabular(colwidths, colaligns):
    alignment = { "left": "l", "right": "r", "center": "c", "decimal": "r" }
    tabular_columns_fmt = "".join([alignment.get(a, "l") for a in colaligns])
    return "\\begin{tabular}{" + tabular_columns_fmt + "}\n\hline"


_table_formats = {"simple":
                  TableFormat(lineabove=Line("", "-", "  ", ""),
                              linebelowheader=Line("", "-", "  ", ""),
                              linebetweenrows=None,
                              linebelow=Line("", "-", "  ", ""),
                              headerrow=DataRow("", "  ", ""),
                              datarow=DataRow("", "  ", ""),
                              padding=0,
                              with_header_hide=["lineabove", "linebelow"]),
                  "plain":
                  TableFormat(lineabove=None, linebelowheader=None,
                              linebetweenrows=None, linebelow=None,
                              headerrow=DataRow("", "  ", ""),
                              datarow=DataRow("", "  ", ""),
                              padding=0, with_header_hide=None),
                  "grid":
                  TableFormat(lineabove=Line("+", "-", "+", "+"),
                              linebelowheader=Line("+", "=", "+", "+"),
                              linebetweenrows=Line("+", "-", "+", "+"),
                              linebelow=Line("+", "-", "+", "+"),
                              headerrow=DataRow("|", "|", "|"),
                              datarow=DataRow("|", "|", "|"),
                              padding=1, with_header_hide=None),
                  "pipe":
                  TableFormat(lineabove=_pipe_line_with_colons,
                              linebelowheader=_pipe_line_with_colons,
                              linebetweenrows=None,
                              linebelow=None,
                              headerrow=DataRow("|", "|", "|"),
                              datarow=DataRow("|", "|", "|"),
                              padding=1,
                              with_header_hide=["lineabove"]),
                  "orgtbl":
                  TableFormat(lineabove=None,
                              linebelowheader=Line("|", "-", "+", "|"),
                              linebetweenrows=None,
                              linebelow=None,
                              headerrow=DataRow("|", "|", "|"),
                              datarow=DataRow("|", "|", "|"),
                              padding=1, with_header_hide=None),
                  "rst":
                  TableFormat(lineabove=Line("", "=", "  ", ""),
                              linebelowheader=Line("", "=", "  ", ""),
                              linebetweenrows=None,
                              linebelow=Line("", "=", "  ", ""),
                              headerrow=DataRow("", "  ", ""),
                              datarow=DataRow("", "  ", ""),
                              padding=0, with_header_hide=None),
                  "mediawiki":
                  TableFormat(lineabove=Line("{| class=\"wikitable\" style=\"text-align: left;\"",
                                             "", "", "\n|+ <!-- caption -->\n|-"),
                              linebelowheader=Line("|-", "", "", ""),
                              linebetweenrows=Line("|-", "", "", ""),
                              linebelow=Line("|}", "", "", ""),
                              headerrow=partial(_mediawiki_row_with_attrs, "!"),
                              datarow=partial(_mediawiki_row_with_attrs, "|"),
                              padding=0, with_header_hide=None),
                  "latex":
                  TableFormat(lineabove=_latex_line_begin_tabular,
                              linebelowheader=Line("\\hline", "", "", ""),
                              linebetweenrows=None,
                              linebelow=Line("\\hline\n\\end{tabular}", "", "", ""),
                              headerrow=DataRow("", "&", "\\\\"),
                              datarow=DataRow("", "&", "\\\\"),
                              padding=1, with_header_hide=None),
                  "tsv":
                  TableFormat(lineabove=None, linebelowheader=None,
                              linebetweenrows=None, linebelow=None,
                              headerrow=DataRow("", "\t", ""),
                              datarow=DataRow("", "\t", ""),
                              padding=0, with_header_hide=None)}


tabulate_formats = list(sorted(_table_formats.keys()))


_invisible_codes = re.compile("\x1b\[\d*m")
_invisible_codes_bytes = re.compile(b"\x1b\[\d*m") 


def simple_separated_format(separator):
    return TableFormat(None, None, None, None,
                       headerrow=DataRow('', separator, ''),
                       datarow=DataRow('', separator, ''),
                       padding=0, with_header_hide=None)


def _isconvertible(conv, string):
    try:
        n = conv(string)
        return True
    except ValueError:
        return False


def _isnumber(string):
    return _isconvertible(float, string)


def _isint(string):
    return type(string) is int or \
           (isinstance(string, _binary_type) or isinstance(string, _text_type)) and \
           _isconvertible(int, string)


def _type(string, has_invisible=True):
    if has_invisible and \
       (isinstance(string, _text_type) or isinstance(string, _binary_type)):
        string = _strip_invisible(string)

    if string is None:
        return _none_type
    elif hasattr(string, "isoformat"):
        return _text_type
    elif _isint(string):
        return int
    elif _isnumber(string):
        return float
    elif isinstance(string, _binary_type):
        return _binary_type
    else:
        return _text_type


def _afterpoint(string):
    if _isnumber(string):
        if _isint(string):
            return -1
        else:
            pos = string.rfind(".")
            pos = string.lower().rfind("e") if pos < 0 else pos
            if pos >= 0:
                return len(string) - pos - 1
            else:
                return -1
    else:
        return -1


def _padleft(width, s, has_invisible=True):
    iwidth = width + len(s) - len(_strip_invisible(s)) if has_invisible else width
    fmt = "{0:>%ds}" % iwidth
    return fmt.format(s)


def _padright(width, s, has_invisible=True):
    iwidth = width + len(s) - len(_strip_invisible(s)) if has_invisible else width
    fmt = "{0:<%ds}" % iwidth
    return fmt.format(s)


def _padboth(width, s, has_invisible=True):
    iwidth = width + len(s) - len(_strip_invisible(s)) if has_invisible else width
    fmt = "{0:^%ds}" % iwidth
    return fmt.format(s)


def _strip_invisible(s):
    if isinstance(s, _text_type):
        return re.sub(_invisible_codes, "", s)
    else:
        return re.sub(_invisible_codes_bytes, "", s)


def _visible_width(s):
    if isinstance(s, _text_type) or isinstance(s, _binary_type):
        return len(_strip_invisible(s))
    else:
        return len(_text_type(s))


def _align_column(strings, alignment, minwidth=0, has_invisible=True):
    if alignment == "right":
        strings = [s.strip() for s in strings]
        padfn = _padleft
    elif alignment == "center":
        strings = [s.strip() for s in strings]
        padfn = _padboth
    elif alignment == "decimal":
        decimals = [_afterpoint(s) for s in strings]
        maxdecimals = max(decimals)
        strings = [s + (maxdecimals - decs) * " "
                   for s, decs in zip(strings, decimals)]
        padfn = _padleft
    elif not alignment:
        return strings
    else:
        strings = [s.strip() for s in strings]
        padfn = _padright

    if has_invisible:
        width_fn = _visible_width
    else:
        width_fn = len

    maxwidth = max(max(list(map(width_fn, strings))), minwidth)
    padded_strings = [padfn(maxwidth, s, has_invisible) for s in strings]
    return padded_strings


def _more_generic(type1, type2):
    types = { _none_type: 0, int: 1, float: 2, _binary_type: 3, _text_type: 4 }
    invtypes = { 4: _text_type, 3: _binary_type, 2: float, 1: int, 0: _none_type }
    moregeneric = max(types.get(type1, 4), types.get(type2, 4))
    return invtypes[moregeneric]


def _column_type(strings, has_invisible=True):
    types = [_type(s, has_invisible) for s in strings ]
    return reduce(_more_generic, types, int)


def _format(val, valtype, floatfmt, missingval=""):
    if val is None:
        return missingval

    if valtype in [int, _text_type]:
        return "{0}".format(val)
    elif valtype is _binary_type:
        return _text_type(val, "ascii")
    elif valtype is float:
        return format(float(val), floatfmt)
    else:
        return "{0}".format(val)


def _align_header(header, alignment, width):
    if alignment == "left":
        return _padright(width, header)
    elif alignment == "center":
        return _padboth(width, header)
    elif not alignment:
        return "{0}".format(header)
    else:
        return _padleft(width, header)


def _normalize_tabular_data(tabular_data, headers):

    if hasattr(tabular_data, "keys") and hasattr(tabular_data, "values"):
        if hasattr(tabular_data.values, "__call__"):
            keys = list(tabular_data.keys())
            rows = list(zip_longest(*list(tabular_data.values())))
        elif hasattr(tabular_data, "index"):
            keys = list(tabular_data.keys())
            vals = tabular_data.values
            names = tabular_data.index
            rows = [[v]+list(row) for v,row in zip(names, vals)]
        else:
            raise ValueError("tabular data doesn't appear to be a dict or a DataFrame")

        if headers == "keys":
            headers = list(map(_text_type,keys))

    else:
        rows = list(tabular_data)

        if (headers == "keys" and
            hasattr(tabular_data, "dtype") and
            getattr(tabular_data.dtype, "names")):
            headers = tabular_data.dtype.names
        elif (headers == "keys"
              and len(rows) > 0
              and isinstance(rows[0], tuple)
              and hasattr(rows[0], "_fields")):
            headers = list(map(_text_type, rows[0]._fields))
        elif headers == "keys" and len(rows) > 0:
            headers = list(map(_text_type, list(range(len(rows[0])))))

    if headers == "firstrow" and len(rows) > 0:
        headers = list(map(_text_type, rows[0]))
        rows = rows[1:]

    headers = list(headers)
    rows = list(map(list,rows))

    if headers and len(rows) > 0:
       nhs = len(headers)
       ncols = len(rows[0])
       if nhs < ncols:
           headers = [""]*(ncols - nhs) + headers

    return rows, headers


def tabulate(tabular_data, headers=[], tablefmt="simple",
             floatfmt="g", numalign="decimal", stralign="left",
             missingval=""):
    list_of_lists, headers = _normalize_tabular_data(tabular_data, headers)

    plain_text = '\n'.join(['\t'.join(map(_text_type, headers))] + \
                            ['\t'.join(map(_text_type, row)) for row in list_of_lists])
    has_invisible = re.search(_invisible_codes, plain_text)
    if has_invisible:
        width_fn = _visible_width
    else:
        width_fn = len

    cols = list(zip(*list_of_lists))
    coltypes = list(map(_column_type, cols))
    cols = [[_format(v, ct, floatfmt, missingval) for v in c]
             for c,ct in zip(cols, coltypes)]

    aligns = [numalign if ct in [int,float] else stralign for ct in coltypes]
    minwidths = [width_fn(h)+2 for h in headers] if headers else [0]*len(cols)
    cols = [_align_column(c, a, minw, has_invisible)
            for c, a, minw in zip(cols, aligns, minwidths)]

    if headers:
        minwidths = [max(minw, width_fn(c[0])) for minw, c in zip(minwidths, cols)]
        headers = [_align_header(h, a, minw)
                   for h, a, minw in zip(headers, aligns, minwidths)]
        rows = list(zip(*cols))
    else:
        minwidths = [width_fn(c[0]) for c in cols]
        rows = list(zip(*cols))

    if not isinstance(tablefmt, TableFormat):
        tablefmt = _table_formats.get(tablefmt, _table_formats["simple"])

    return _format_table(tablefmt, headers, rows, minwidths, aligns)


def _build_simple_row(padded_cells, rowfmt):
    begin, sep, end = rowfmt
    return (begin + sep.join(padded_cells) + end).rstrip()


def _build_row(padded_cells, colwidths, colaligns, rowfmt):
    if not rowfmt:
        return None
    if hasattr(rowfmt, "__call__"):
        return rowfmt(padded_cells, colwidths, colaligns)
    else:
        return _build_simple_row(padded_cells, rowfmt)


def _build_line(colwidths, colaligns, linefmt):
    if not linefmt:
        return None
    if hasattr(linefmt, "__call__"):
        return linefmt(colwidths, colaligns)
    else:
        begin, fill, sep,  end = linefmt
        cells = [fill*w for w in colwidths]
        return _build_simple_row(cells, (begin, sep, end))


def _pad_row(cells, padding):
    if cells:
        pad = " "*padding
        padded_cells = [pad + cell + pad for cell in cells]
        return padded_cells
    else:
        return cells


def _format_table(fmt, headers, rows, colwidths, colaligns):
    lines = []
    hidden = fmt.with_header_hide if (headers and fmt.with_header_hide) else []
    pad = fmt.padding
    headerrow = fmt.headerrow

    padded_widths = [(w + 2*pad) for w in colwidths]
    padded_headers = _pad_row(headers, pad)
    padded_rows = [_pad_row(row, pad) for row in rows]

    if fmt.lineabove and "lineabove" not in hidden:
        lines.append(_build_line(padded_widths, colaligns, fmt.lineabove))

    if padded_headers:
        lines.append(_build_row(padded_headers, padded_widths, colaligns, headerrow))
        if fmt.linebelowheader and "linebelowheader" not in hidden:
            lines.append(_build_line(padded_widths, colaligns, fmt.linebelowheader))

    if padded_rows and fmt.linebetweenrows and "linebetweenrows" not in hidden:
        for row in padded_rows[:-1]:
            lines.append(_build_row(row, padded_widths, colaligns, fmt.datarow))
            lines.append(_build_line(padded_widths, colaligns, fmt.linebetweenrows))
        lines.append(_build_row(padded_rows[-1], padded_widths, colaligns, fmt.datarow))
    else:
        for row in padded_rows:
            lines.append(_build_row(row, padded_widths, colaligns, fmt.datarow))

    if fmt.linebelow and "linebelow" not in hidden:
        lines.append(_build_line(padded_widths, colaligns, fmt.linebelow))

    return "\n".join(lines)
