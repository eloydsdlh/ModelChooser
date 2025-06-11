SEED = 42


def train_logistic_regression_model(X_train, y_train):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_random_forest_model(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=SEED)
    model.fit(X_train, y_train)
    return model


def train_support_vector_machine(X_train, y_train):
    from sklearn.svm import SVC
    model = SVC(kernel='linear', random_state=SEED)
    model.fit(X_train, y_train)
    return model

def train_knn_classifier(X_train, y_train, n_neighbors):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

def train_decision_tree_classifier(X_train, y_train):
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(random_state=SEED)
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting_classifier(X_train, y_train):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(random_state=SEED)
    model.fit(X_train, y_train)
    return model

def train_adaboost_classifier(X_train, y_train):
    from sklearn.ensemble import AdaBoostClassifier
    model = AdaBoostClassifier(random_state=SEED)
    model.fit(X_train, y_train)
    return model

def train_naive_bayes_classifier(X_train, y_train):
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def train_xgboost_classifier(X_train, y_train):
    from xgboost import XGBClassifier
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=SEED)
    model.fit(X_train, y_train)
    return model

def train_lightgbm_classifier(X_train, y_train):
    from lightgbm import LGBMClassifier
    model = LGBMClassifier(random_state=SEED)
    model.fit(X_train, y_train)
    return model

def train_catboost_classifier(X_train, y_train):
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(verbose=0, random_state=SEED)
    model.fit(X_train, y_train)
    return model

