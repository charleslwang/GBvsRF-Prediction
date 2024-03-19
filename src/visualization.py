import matplotlib.pyplot as plt

def plot_predictions(y_true, gb_predictions, rf_predictions, title):
    """
    Plot the true values and predictions of Gradient Boosting and Random Forest models.

    Parameters:
    y_true (numpy.ndarray): True target values.
    gb_predictions (numpy.ndarray): Predictions of Gradient Boosting model.
    rf_predictions (numpy.ndarray): Predictions of Random Forest model.
    title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='True Values', color='blue')
    plt.plot(gb_predictions, label='Gradient Boosting Predictions', linestyle='--', color='green')
    plt.plot(rf_predictions, label='Random Forest Predictions', linestyle='--', color='red')
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()
