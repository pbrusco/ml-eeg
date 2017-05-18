import numpy as np


def get_column(data, column_name):
    column_id = [a[0] for a in data["attributes"]].index(column_name)
    values = np.array([float(d[column_id]) for d in data["data"]])
    return values
