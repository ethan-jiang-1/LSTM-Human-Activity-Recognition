# the matrix used in other app (which is better than below one)
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
from sklearn import metrics


def plot_confusion(pred_test, y_test, labels):
    print(classification_report(pred_test, y_test, target_names=[lb for lb in labels]))
    conf_mat = confusion_matrix(pred_test, y_test)

    plt.style.use('bmh')
    fig = plt.figure(figsize=(6,6))
    # width = np.shape(conf_mat)[1]
    # height = np.shape(conf_mat)[0]

    res = plt.imshow(np.array(conf_mat), cmap=plt.cm.summer, interpolation='nearest')
    for i, row in enumerate(conf_mat):
        for j, c in enumerate(row):
            if c > 0:
                plt.text(j-.2, i+.1, c, fontsize=16)

    fig.colorbar(res)
    plt.title('Confusion Matrix')
    _ = plt.xticks(range(6), [lb for lb in labels], rotation=90)
    _ = plt.yticks(range(6), [lb for lb in labels])
    plt.show()

def print_accuracy(final_accuracy, pred_test, y_test, X_test):
    print("Testing Accuracy: {}%".format(100*final_accuracy))

    print("")
    print("Precision: {}%".format(100*metrics.precision_score(y_test, pred_test, average="weighted")))
    print("Recall: {}%".format(100*metrics.recall_score(y_test, pred_test, average="weighted")))
    print("f1_score: {}%".format(100*metrics.f1_score(y_test, pred_test, average="weighted")))

    print("")
    print("Confusion Matrix:")
    confusion_matrix = metrics.confusion_matrix(y_test, pred_test)
    print(confusion_matrix)
    normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100

    print("")
    print("Confusion matrix (normalised to {} of total test data):".format(len(X_test)))
    print(normalised_confusion_matrix)
    print("Note: training and testing data is not equally distributed amongst classes, ")
    print("so it is normal that more than a 6th of the data is correctly classifier in the last category.")


def plot_traning(batch_size, train_losses, train_accuracies, m_training_iters,
                 test_losses, test_accuracies, m_display_iter):
    font = {
        'family' : 'Bitstream Vera Sans',
        'weight' : 'bold',
        'size'   : 12
    }
    matplotlib.rc('font', **font)

    width = 9
    height = 6
    plt.figure(figsize=(width, height))

    indep_train_axis = np.array(range(batch_size, (len(train_losses)+1)*batch_size, batch_size))
    plt.plot(indep_train_axis, np.array(train_losses), "b--", label="Train losses")
    plt.plot(indep_train_axis, np.array(train_accuracies), "g--", label="Train accuracies")

    indep_test_axis = np.append(
        np.array(range(batch_size, len(test_losses)*m_display_iter, m_display_iter)[:-1]),
        [m_training_iters]
    )
    plt.plot(indep_test_axis, np.array(test_losses), "b-", label="Test losses")
    plt.plot(indep_test_axis, np.array(test_accuracies), "g-", label="Test accuracies")

    plt.title("Training session's progress over iterations")
    plt.legend(loc='upper right', shadow=True)
    plt.ylabel('Training Progress (Loss or Accuracy values)')
    plt.xlabel('Training iteration')

    plt.show()    