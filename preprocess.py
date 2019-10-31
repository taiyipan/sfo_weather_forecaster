import pandas as pd

def preprocess(infile_name, outfile_name):
    # load data
    df = pd.read_csv(infile_name)
    # print(df.head(25))
    print('dataframe size: {}'.format(df.size))
    print('dataframe shape" {}'.format(df.shape))
    print('dataframe ndim: {}'.format(df.ndim))

    # drop null
    df = df.dropna()
    print('dataframe size: {}'.format(df.size))
    print('dataframe shape" {}'.format(df.shape))
    print('dataframe ndim: {}'.format(df.ndim))

    # save processed data
    df.to_csv(outfile_name, index = False)

infile_name = 'data/lax_tmpc.txt'
outfile_name = 'data/lax_tmpc.csv'
preprocess(infile_name, outfile_name)
