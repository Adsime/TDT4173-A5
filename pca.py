import image
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





def run(ai_model, methods, alphabet, n_components):
    train_data, train_targets = alphabet.get_all_train_data(methods)
    pca, train_data = fit_transform(n_components, train_data)

    ai_model.train(train_data, train_targets)

    test_data, test_targets = alphabet.get_all_test_data(methods)
    test_data = transform(pca, test_data)
    error = 0
    elements = 0

    for i, img in enumerate(test_data):
        error += 0 if ai_model.predict(img.reshape(1, -1)) == test_targets[i] else 1
        elements += 1

    print("Error: " + (error/elements).__str__())