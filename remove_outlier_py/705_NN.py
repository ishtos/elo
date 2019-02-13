from keras.models import Model, load_model
from keras.layers import Input, Activation, Dense, LSTM, TimeDistributed, Lambda, Reshape, RepeatedVector, Permute, Multiply, Add, Concatenate
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.engine.topology import Container
from keras import backend as K

# =============================================================================
# train and test
# =============================================================================
features = []

features += [f'f10{i}.pkl' for i in (2, 3, 4, 5, 6, 7, 8)]
features += [f'f11{i}_{j}.pkl' for i in (1,) 
                               for j in ('Y', 'N')]
features += [f'f13{i}.pkl' for i in (1, 3, 4)]

features += [f'f20{i}.pkl' for i in (2, 3, 4, 5, 6, 7)]
# features += [f'f23{i}.pkl' for i in (1, 3)]

features += [f'f30{i}.pkl' for i in (2, )]

# =============================================================================
# train and test
# =============================================================================
train = pd.read_csv(os.path.join(PATH, 'train.csv'))
test = pd.read_csv(os.path.join(PATH, 'test.csv'))

for f in tqdm(features):
    t = pd.read_pickle(os.path.join('..', 'remove_outlier_feature', f))
    train = pd.merge(train, t, on=KEY, how='left')
    test = pd.merge(test, t, on=KEY, how='left')

df = pd.concat([train, test], axis=0)
df['first_active_month'] = pd.to_datetime(df['first_active_month'])

date_features = [
    'hist_purchase_date_max','hist_purchase_date_min',
    'new_purchase_date_max', 'new_purchase_date_min',
    'Y_hist_auth_purchase_date_max', 'N_hist_auth_purchase_date_max',
    'Y_hist_auth_purchase_date_min', 'N_hist_auth_purchase_date_min'
]

for f in date_features:
    df[f] = pd.to_datetime(df[f])

df['hist_first_buy'] = (df['hist_purchase_date_min'].dt.date - df['first_active_month'].dt.date).dt.days
df['hist_last_buy'] = (df['hist_purchase_date_max'].dt.date - df['first_active_month'].dt.date).dt.days
df['new_first_buy'] = (df['new_purchase_date_min'].dt.date - df['first_active_month'].dt.date).dt.days
df['new_last_buy'] = (df['new_purchase_date_max'].dt.date - df['first_active_month'].dt.date).dt.days

for f in date_features:
    df[f] = df[f].astype(np.int64) * 1e-9

df['card_id_cnt_total'] = df['new_card_id_count'] + df['hist_card_id_count']
df['purchase_amount_total'] = df['new_purchase_amount_sum'] + df['hist_purchase_amount_sum']
df['purchase_amount_mean'] = df['new_purchase_amount_mean'] + df['hist_purchase_amount_mean']
df['purchase_amount_max'] = df['new_purchase_amount_max'] + df['hist_purchase_amount_max']
df['purchase_amount_min'] = df['new_purchase_amount_min'] + df['hist_purchase_amount_min']
df['sum_new_CLV'] = df['new_card_id_count'] * df['new_purchase_amount_sum'] / df['new_month_diff_mean']
df['sum_hist_CLV'] = df['hist_card_id_count'] * df['hist_purchase_amount_sum'] / df['hist_month_diff_mean']
df['sum_CLV_ratio'] = df['sum_new_CLV'] / df['sum_hist_CLV']

df['nans'] = df.isnull().sum(axis=1)

train = df[df['target'].notnull()]
test = df[df['target'].isnull()]
y = train['target']

categorical_features = ['feature_1', 'feature_2', 'feature_3']
pca = PCA(n_components=1)
pca.fit(train[categorical_features])
pca_train_values = pca.transform(train[categorical_features])
pca_test_values = pca.transform(test[categorical_features])

pca_train_values = np.transpose(pca_train_values, (1, 0))
pca_test_values = np.transpose(pca_test_values, (1, 0))

for e, (pca_train, pca_test) in enumerate(zip(pca_train_values, pca_test_values)):
    train[f'pca_feature_{e}'] = pca_train
    test[f'pca_feature_{e}'] = pca_test

del df
gc.collect()
