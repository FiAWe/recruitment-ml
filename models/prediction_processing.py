from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd


def post_process(model_vars, model_name, verbose=True):
    
    # Extract the variables
    X_test = model_vars['X_test']
    y_test = model_vars['y_test']
    y_pred_nb = model_vars['y_pred_nb']
    y_pred_nb_proba = model_vars['y_pred_nb_proba']
    le = model_vars['le']

    # Print the classification report
    if verbose:
        print(classification_report(y_test, y_pred_nb, digits=4))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred_nb))
        print("AUC-ROC:", roc_auc_score(y_test, y_pred_nb_proba))

    # Save the predictions
    y_pred_nb = le.inverse_transform(y_pred_nb)
    y_test = le.inverse_transform(y_test)

    # Join into a single dataframe
    df = pd.DataFrame({'text': X_test, 'label': y_test, 'prediction': y_pred_nb, 'prediction_proba': y_pred_nb_proba})
    df.to_csv(f'results/{model_name}.csv', index=False)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_nb_proba)
    plt.plot(fpr, tpr,  marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')

    # Add a line for a random classifier
    plt.plot([0, 1], [0, 1], linestyle='--')

    # Remove right and upper borders
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    plt.savefig(f'results/{model_name}_roc_curve.png', dpi=300)
    plt.clf()

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_nb_proba)
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')

    # Add a line for a random classifier
    positive_ratio = y_test.mean()
    plt.plot([0, 1], [positive_ratio, positive_ratio], linestyle='--')

    # Remove right and upper borders
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    plt.savefig(f'results/{model_name}_precision_recall_curve.png', dpi=300)
    plt.clf()

def save_meta_data(timings:dict, model_vars:dict, model_name:str):

    data_size = {
        'train': len(model_vars['X_train']),
        'test': len(model_vars['X_test'])
    }

    with open(f'results/{model_name}_metadata.txt', 'w') as f:
        f.write('Timings\n')
        f.write('-------\n')
        for key, value in timings.items():
            f.write(f'{key}: {value}\n')
        f.write('\nData Size\n')
        f.write('---------\n')
        for key, value in data_size.items():
            f.write(f'{key}: {value}\n')