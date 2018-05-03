from sklearn.decomposition import PCA


def transform(pca, data):
    return pca.transform(data)


def fit_transform(n_components, training_data):
    pca = PCA(n_components=n_components)
    return [pca, pca.fit_transform(training_data)]


def get_pca(n_components, images):
    pca = PCA(n_components=n_components)
    pca.fit(images)
    return pca