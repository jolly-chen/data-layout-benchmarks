import csv


def read_columns(path, columns, input_size=None):
    """
    Read the given columns from a dataset with a header row, as lists of floats.

    :param path: Dataset file; the first row names the columns, e.g. "pt,eta,phi,e"
    :param columns: Names of the columns to read
    :param input_size: Number of rows to read (default: all)
    """
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        missing = [c for c in columns if c not in header]
        if missing:
            raise ValueError(
                f"{path} has no column(s) {', '.join(missing)} in its header row {','.join(header)}"
            )
        idx = [header.index(c) for c in columns]
        values = [[] for _ in columns]
        for i, row in enumerate(reader):
            if input_size is not None and i >= input_size:
                break
            for v, j in zip(values, idx):
                v.append(float(row[j]))
    return values
