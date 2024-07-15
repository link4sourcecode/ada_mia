import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import train_test_split
from numpy import loadtxt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale, MinMaxScaler, StandardScaler
from tqdm import tqdm


dataset = loadtxt('./Logging/InferenceData.csv', delimiter=",")
data = dataset[:, :-1]
data = scale(data)
label = dataset[:, -1]
# the ratio of training samples to testing samples is 4:1
train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.2)
dtrain = xgb.DMatrix(train_data, label=train_label)
dtest = xgb.DMatrix(test_data, label=test_label)

# a rough example of binary supervised inference method
params={'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        # setting leaves
        'gamma': 0.125,
        # max depth of tree - main
        'max_depth': 2,
        # L2 regularization
        'lambda': 0.15,
        # L1 regularization
        'alpha': 0.2,
        # use percentage of training data
        'subsample': 1,
        # feature random sampling percentage
        'colsample_bytree': 1,
        # if leave < weight stop training
        'min_child_weight': 1,
        # learning rate - main
        'eta': 0.0035,
        # 'seed': 0,
        'nthread': -1,
        # 'silent': 1,
        }

# watchlist = [(dtrain, 'train'), (dtest, 'test')]
watchlist = [(dtrain, 'train'), (dtest, 'test')]
# folds = KFold(n_splits=5, shuffle=True, random_state=2019)

model = xgb.train(params,
                  dtrain,
                  num_boost_round=2000,
                  evals=watchlist,
                  verbose_eval=False,
                  # early_stopping_rounds=1000,
                  )

ypred = model.predict(dtest)
y_pred = (ypred >= 0.5) * 1
print('Accuracy: %.4f' % metrics.accuracy_score(test_label, y_pred))
print('Precesion: %.4f' % metrics.precision_score(test_label, y_pred))
print('Recall: %.4f' % metrics.recall_score(test_label, y_pred))
print('F1-score: %.4f' % metrics.f1_score(test_label, y_pred))
print('AUC: %.4f' % metrics.roc_auc_score(test_label, ypred))
