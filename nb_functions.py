import pandas as pd


# Pass in X_train and y_train
# Returns tuple with X_train_resampled and y_train_resampled
def resample_fraud_dataset(X_train: pd.DataFrame, y_train: pd.DataFrame) -> tuple:
    from sklearn.utils import resample
    # Balancing data set with resampling

    # Separate majority and minority classes in training set
    X_train_majority = X_train[y_train["Class"] == 0]
    y_train_majority = y_train[y_train["Class"] == 0]
    X_train_minority = X_train[y_train["Class"] == 1]
    y_train_minority = y_train[y_train["Class"] == 1]

    # X_train_majority.head()

    # Oversample minority class, matching number of Fraud and not-Fraud rows with duplication
    X_minority_upsampled, y_minority_upsampled = resample(
        X_train_minority, y_train_minority,
        replace=True,  # sample with replacement
        n_samples=len(y_train_majority),  # match majority class
        random_state=1
    )

    # Combine majority and upsampled minority
    X_train_resampled = pd.concat((X_train_majority, X_minority_upsampled))
    y_train_resampled = pd.concat((y_train_majority, y_minority_upsampled))

    return X_train_resampled, y_train_resampled


