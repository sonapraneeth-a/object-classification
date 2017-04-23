from sklearn.preprocessing import StandardScaler, MinMaxScaler
from library.preprocessing import ZCA


def transform(input, transform_method='StandardScaler'):
    if transform_method == 'StandardScaler':
        ss = StandardScaler()
    elif transform_method == 'MinMaxScaler':
        ss = MinMaxScaler()
    else:
        data = input
    if transform_method != '':
        data = ss.fit_transform(input)
    return data

