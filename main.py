import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import resample

# Models
from sklearn.linear_model import SGDClassifier # SGD model
from sklearn.ensemble import RandomForestClassifier # Random Forest model
from sklearn.neural_network import MLPClassifier # Neural network

# If the dataset is too sparse, try performance of these instead
# from sklearn.svm import LinearSVC, SVC
# from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

def main() -> None:
    # Reading into data frame from file
    df = pd.read_csv('./creditcard.csv')

    # Data to fit and targets
    feature_set = ["Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14",
                   "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
                   "Amount"]
    target_set = ["Class"]

    X = df[feature_set]
    y = df[target_set]

    # Splits data into 80-20 train-test
    # Prevents data leakage (where training and test sets influence each other in scaling)
    # Stratify y so we get a balance of fraud and not-fraud in the split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # Scaling with each method, preserving unscaled values for comparison
    standard_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()

    # Standard scaling
    X_train_standard_scaled = standard_scaler.fit_transform(X_train)
    X_test_standard_scaled = standard_scaler.transform(X_test)
    # Minmax Scaling
    X_train_minmax_scaled = minmax_scaler.fit_transform(X_train)
    X_test_minmax_scaled = minmax_scaler.transform(X_test)

    # Initialize model(s)
    SGD_model = SGDClassifier(class_weight='balanced', random_state=1)  # Apparently prefers standard scaling
    # RF_model = RandomForestClassifier(random_state=1) # Theoretically needs no scaling
    # MLP_model = MLPClassifier(random_state=1)

    # Does cross validation with 5 folds using our training data post split
    kf = KFold(n_splits=5)
    cv_scores = cross_val_score(SGD_model, X_train, y_train, cv=kf, scoring="accuracy")
    print("Cross validation scores:", cv_scores)
    print(f"Mean Validation Score: {np.mean(cv_scores)}")

if __name__ == "__main__":
    main()