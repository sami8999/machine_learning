from string import punctuation, digits
import numpy as np
import random


# ==============================================================================
# ===  PART I  =================================================================
# ==============================================================================


def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices


def hinge_loss_single(feature_vector, label, theta, theta_0):
    parameter = (np.dot(feature_vector, theta) + theta_0) * label
    if parameter >= 1:
        return 0
    else:
        return 1 - parameter


def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the hinge loss for given classification parameters averaged over a
    given dataset

    feature_vector = np.array([[1, 2], [1, 2]])
    label, theta, theta_0 = np.array([1, 1]), np.array([-1, 1]), -0.2

    Args:
        `feature_matrix` - numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `theta` - numpy array describing the linear classifier.
        `theta_0` - real valued number representing the offset parameter.
    Returns:
        the hinge loss, as a float, associated with the given dataset and
        parameters.  This number should be the average hinge loss across all of
    """

    y = np.dot(feature_matrix, theta) + theta_0
    losses = np.maximum(0.0, 1 - y * labels)
    return np.mean(losses)
    #
    # NOTE: the above gives the same result as the following line.  However, we
    # prefer to avoid doing linear algebra and other iteration in pure Python,
    # since Python lists and loops are slow.
    #
    # return np.mean([hinge_loss_single(feature_vector, label, theta, theta_0)
    #                 for (feature_vector, label) in zip(feature_matrix, labels)])
    #


def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    y = (np.dot(feature_vector, current_theta) + current_theta_0) * label
    if y <= 0:
        new_theta = current_theta + label * feature_vector
        new_theta_0 = current_theta_0 + label
        return new_theta, new_theta_0
    else:
        return current_theta, current_theta_0


def perceptron(feature_matrix, labels, T):
    theta = np.zeros(len(feature_matrix[0]))
    theta_0 = 0

    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_0)

    return theta, theta_0


def average_perceptron(feature_matrix, labels, T):
    theta = np.zeros(len(feature_matrix[0]))
    theta_0 = 0
    theta_array = []
    theta_0_array = np.array([])

    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_0)
            theta_array.append(theta)
            theta_0_array = np.append(theta_0_array, theta_0)

    return np.mean(theta_array, axis=0), np.mean(theta_0_array)


def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        theta,
        theta_0):
    y = (np.dot(feature_vector, theta) + theta_0) * label
    if y <= 1:
        # new_theta = theta + label * feature_vector
        theta = (1 - eta * L) * theta + eta * label * feature_vector
        theta_0 = theta_0 + eta * label

    else:
        theta = (1 - eta * L) * theta

    return theta, theta_0


def pegasos(feature_matrix, labels, T, L):
    theta = np.zeros(len(feature_matrix[0]))
    theta_0 = 0

    counter = 1
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            theta, theta_0 = pegasos_single_step_update(feature_matrix[i], labels[i], L, 1 / counter ** (1 / 2), theta,
                                                        theta_0)
            counter = counter + 1

    return theta, theta_0


# ==============================================================================
# ===  PART II  ================================================================
# ==============================================================================


##  #pragma: coderesponse answer
#  def decision_function(feature_vector, theta, theta_0):
#      return np.dot(theta, feature_vector) + theta_0
#  def classify_vector(feature_vector, theta, theta_0):
#      return 2*np.heaviside(decision_function(feature_vector, theta, theta_0), 0)-1
#  #pragma: coderesponse end


def classify(feature_matrix, theta, theta_0):
    epsilon = 1e-8
    y = np.dot(feature_matrix, theta) + theta_0
    rounded_y = np.where(np.abs(y) < epsilon, -1, np.sign(y))
    return rounded_y


def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    theta, theta_0 = classifier(train_feature_matrix, train_labels, **kwargs)

    preds_train = classify(train_feature_matrix, theta, theta_0)
    preds_valid = classify(val_feature_matrix, theta, theta_0)

    return accuracy(preds_train, train_labels), accuracy(preds_valid, val_labels)


def extract_words(text):
    """
    Helper function for `bag_of_words(...)`.
    Args:
        a string `text`.
    Returns:
        a list of lowercased words in the string, where punctuation and digits
        count as their own words.
    """
    for c in punctuation + digits:
        text = text.replace(c, ' ' + c + ' ')
    return text.lower().split()


def bag_of_words(texts, remove_stopword=True):
    """
    NOTE: feel free to change this code as guided by Section 3 (e.g. remove
    stopwords, add bigrams etc.)

    Args:
        `texts` - a list of natural language strings.
    Returns:
        a dictionary that maps each word appearing in `texts` to a unique
        integer `index`.
    """
    # # Your code here
    # raise NotImplementedError
    with open('stopwords.txt', 'r') as f:
        stopword = [line.strip() for line in f.readlines()]

    indices_by_word = {}  # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word in indices_by_word: continue
            if word in stopword: continue
            indices_by_word[word] = len(indices_by_word)

    return indices_by_word


def extract_bow_feature_vectors(reviews, indices_by_word, binarize=True):
    """
    Args:
        `reviews` - a list of natural language strings
        `indices_by_word` - a dictionary of uniquely-indexed words.
    Returns:
        a matrix representing each review via bag-of-words features.  This
        matrix thus has shape (n, m), where n counts reviews and m counts words
        in the dictionary.
    """

    feature_matrix = np.zeros([len(reviews), len(indices_by_word)], dtype=np.float64)
    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word not in indices_by_word: continue
            count = word_list.count(word)
            feature_matrix[i, indices_by_word[word]] = count

    return feature_matrix


def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the fraction of predictions that are correct.
    """
    return (preds == targets).mean()
