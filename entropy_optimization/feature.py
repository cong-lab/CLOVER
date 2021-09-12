from sklearn.preprocessing import PolynomialFeatures


def get_feature(intervention):
    feature_func = PolynomialFeatures(degree=2, interaction_only=True)
    if len(intervention.shape) == 1:
        intervention = intervention.reshape(1, -1)
    feature = feature_func.fit_transform(intervention)
    return feature
