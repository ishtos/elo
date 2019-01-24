import os
import gc
import utils

import pandas as pd
import numpy as np
import pickle as pkl
from datetime import date

from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, Flatten, Concatenate, Dot, Reshape, Add, Subtract
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import l2

utils.start(__file__)
#==============================================================================
PREF = 'f503'

KEY = 'card_id'

SEED = 18
np.random.seed(SEED)

# =============================================================================
# def
# =============================================================================
def get_embed(x_input, x_size, k_latent):
    if x_size > 0:  
        embed = Embedding(x_size, k_latent, input_length=1,
                          embeddings_regularizer=l2(embedding_reg))(x_input)
        embed = Flatten()(embed)
    else:
        embed = Dense(k_latent, kernel_regularizer=l2(embedding_reg))(x_input)
    return embed


def build_model_1(X, fsize):
    dim_input = len(fsize)

    input_x = [Input(shape=(1,)) for i in range(dim_input)]

    biases = [get_embed(x, size, 1) for (x, size) in zip(input_x, fsize)]

    factors = [get_embed(x, size, k_latent)
               for (x, size) in zip(input_x, fsize)]

    s = Add()(factors)

    diffs = [Subtract()([s, x]) for x in factors]

    dots = [Dot(axes=1)([d, x]) for d, x in zip(diffs, factors)]

    x = Concatenate()(biases + dots)
    x = BatchNormalization()(x)
    output = Dense(1, activation='relu', kernel_regularizer=l2(kernel_reg))(x)
    model = Model(inputs=input_x, outputs=[output])
    opt = Adam(clipnorm=0.5)
    model.compile(optimizer=opt, loss='mean_squared_error')
    output_f = factors + biases
    model_features = Model(inputs=input_x, outputs=output_f)

    return model, model_features

# =============================================================================
# features
# =============================================================================
features = []

features += [f'f10{i}.pkl' for i in (2, )]
features += [f'f11{i}_{j}.pkl' for i in (1, 2)
             for j in ('Y', 'N')]
features += [f'f12{i}.pkl' for i in (1, 2)]
features += [f'f13{i}.pkl' for i in (1, 2)]

features += [f'f20{i}.pkl' for i in (2, 3)]
features += [f'f21{i}_{j}.pkl' for i in (1, 2)
             for j in ('Y', 'N')]
features += [f'f23{i}.pkl' for i in (1, 2)]

# features += [f'f40{i}.pkl' for i in (2, 3)]
# features += [f'f41{i}_{j}.pkl' for i in (1, 2)
#                                for j in ('Y', 'N')]
# features += [f'f42{i}.pkl' for i in (1, 2)]

# features += [f'f50{i}.pkl' for i in (2, )]

# features = os.listdir('../remove_outlier_feature')

# =============================================================================
# read data and features
# =============================================================================
train = pd.read_csv(os.path.join(PATH, 'train.csv'))
test = pd.read_csv(os.path.join(PATH, 'test.csv'))

for f in tqdm(features):
    t = pd.read_pickle(os.path.join('..', 'remove_outlier_feature', f))
    train = pd.merge(train, t, on=KEY, how='left')
    test = pd.merge(test, t, on=KEY, how='left')

# =============================================================================
# change date to int
# =============================================================================
cols = train.columns.values
for f in [
    'new_purchase_date_max', 'new_purchase_date_min',
    'hist_purchase_date_max', 'hist_purchase_date_min',
    'Y_hist_auth_purchase_date_max', 'Y_hist_auth_purchase_date_min',
    'N_hist_auth_purchase_date_max', 'N_hist_auth_purchase_date_min',
    'Y_new_auth_purchase_date_max', 'Y_new_auth_purchase_date_min',
    'N_new_auth_purchase_date_max', 'N_new_auth_purchase_date_min',
]:
    if f in cols:
        train[f] = train[f].astype(np.int64) * 1e-9
        test[f] = test[f].astype(np.int64) * 1e-9

# =============================================================================
# concat train and test
# =============================================================================
df = pd.concat([train, test], axis=0, sort=False)
del train, test
gc.collect()

# =============================================================================
# main
# =============================================================================

features = ['feature_1', 'feature_2', 'feature_3']
fsize = [int(df[f].max()) + 1 for f in features]

X = df.groupby(features)['card_id'].count()

X = X.unstack().fillna(0)
X = X.stack().astype('float32')
X = np.log1p(X).reset_index()
X.columns = features + ['num']

X_train = [X[f].values for f in features]
y_train = (X[['num']].values).astype('float32')

k_latent = 1
embedding_reg = 0.0002
kernel_reg = 0.1

model, model_features = build_model_1(X_train, fsize)

n_epochs = 1000

batch_size = 2 ** 17
model, model_features = build_model_1(X_train, fsize)
earlystopper = EarlyStopping(patience=0, verbose=50)

history = model.fit(
    X_train,  y_train,
    epochs=n_epochs, batch_size=batch_size, verbose=1, shuffle=True,
    validation_data=(X_train, y_train),
    callbacks=[earlystopper],
)
model.save('weights/{}_weights.h5'.format(str(date.today()).replace('-', '')))

X_pred = model_features.predict(X_train, batch_size=batch_size)

factors = X_pred[:len(features)]

biases = X_pred[len(features):2*len(features)]

for f, X_p in zip(features, factors):
    for i in range(k_latent):
        X['%s_fm_factor_%d' % (f, i)] = X_p[:, i]

for f, X_p in zip(features, biases):
    X['%s_fm_bias' % (f)] = X_p[:, 0]

df = pd.merge(df, X, on=features, how='left')
df = df.drop(features, axis=1)
df.to_pickle(f'../feature/{PREF}.pkl')

#==============================================================================
utils.end(__file__)
