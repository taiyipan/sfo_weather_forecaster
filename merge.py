import pandas as pd

features = ['station', 'valid', 'tmpc']

df1 = pd.read_csv('data/sfo_tmpc.csv')
df1 = df1[features]
# df1.index = df['valid']

df2 = pd.read_csv('data/oak_tmpc.csv')
df2 = df2[features]
# df2.index = df['valid']

df3 = pd.read_csv('data/lax_tmpc.csv')
df3 = df3[features]
# df3.index = df['valid']

# df = pd.concat([df1, df2, df3], axis = 1, join = 'inner')

df = df1.set_index('valid').join(df2.set_index('valid'), how = 'inner', lsuffix = '_sfo', rsuffix = '_oak')
df = df.join(df3.set_index('valid'), how = 'inner', rsuffix = '_lax')

df.to_csv('data/sfo_oak_lax.csv', index = True)
