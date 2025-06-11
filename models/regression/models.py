SEED = 42


def train_linear_regression(X_train, y_train):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=SEED)
    model.fit(X_train, y_train)
    return model


def train_support_vector_regression(X_train, y_train):
    from sklearn.svm import SVR
    model = SVR(kernel='linear')
    model.fit(X_train, y_train)
    return model

def train_ridge_regression(X_train, y_train):
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1.0, random_state=SEED)
    model.fit(X_train, y_train)
    return model

def train_lasso_regression(X_train, y_train):
    from sklearn.linear_model import Lasso
    model = Lasso(alpha=0.1, random_state=SEED)
    model.fit(X_train, y_train)
    return model

def train_elastic_net(X_train, y_train):
    from sklearn.linear_model import ElasticNet
    model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=SEED)
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting_regressor(X_train, y_train):
    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor(n_estimators=100, random_state=SEED)
    model.fit(X_train, y_train)
    return model

def train_knn_regressor(X_train, y_train, n_neighbors):
    from sklearn.neighbors import KNeighborsRegressor
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

def train_decision_tree_regressor(X_train, y_train):
    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor(random_state=SEED)
    model.fit(X_train, y_train)
    return model

def train_adaboost_regressor(X_train, y_train):
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.tree import DecisionTreeRegressor
    base_estimator = DecisionTreeRegressor(max_depth=3, random_state=SEED)
    model = AdaBoostRegressor(estimator=base_estimator, n_estimators=100, random_state=SEED)
    model.fit(X_train, y_train)
    return model

def train_bayesian_ridge_regression(X_train, y_train):
    from sklearn.linear_model import BayesianRidge
    model = BayesianRidge()
    model.fit(X_train, y_train)
    return model

def train_xgboost_regressor(X_train, y_train):
    from xgboost import XGBRegressor
    model = XGBRegressor(n_estimators=100, random_state=SEED)
    model.fit(X_train, y_train)
    return model

def train_lightgbm_regressor(X_train, y_train):
    from lightgbm import LGBMRegressor
    model = LGBMRegressor(n_estimators=100, random_state=SEED)
    model.fit(X_train, y_train)
    return model

def train_catboost_regressor(X_train, y_train):
    from catboost import CatBoostRegressor
    model = CatBoostRegressor(iterations=100, random_state=SEED, verbose=0)
    model.fit(X_train, y_train)
    return model

