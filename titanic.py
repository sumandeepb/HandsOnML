import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score


DATA_PATH = os.path.join("/home/sumandeep/Data/HandsOnML", "titanic")


def load_titanic_data(file_name):
    csv_path = os.path.join(DATA_PATH, file_name)
    return pd.read_csv(csv_path)


def prepare_titanic_data(train, test):
    train = train[['Pclass', 'Age', 'SibSp',
                   'Parch', 'Fare', 'Sex', 'Embarked']]
    test = test[['Pclass', 'Age', 'SibSp',
                 'Parch', 'Fare', 'Sex', 'Embarked']]

    numeric_pipeline = Pipeline([
        ('numeric_imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler())
    ])

    sex_pipeline = Pipeline([
        ('sex_encoder', OrdinalEncoder())
    ])

    embarked_pipeline = Pipeline([
        ('embarked_imputer', SimpleImputer(strategy="most_frequent")),
        ('embarked_encoder', OneHotEncoder())
    ])

    numeric_attribs = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    sex_attribs = ['Sex']
    embarked_attribs = ['Embarked']

    full_pipeline = ColumnTransformer([
        ('num1', numeric_pipeline, numeric_attribs),
        ('cat1', sex_pipeline, sex_attribs),
        ('cat2', embarked_pipeline, embarked_attribs)
    ])

    full_pipeline.fit(train)
    return full_pipeline.transform(train), full_pipeline.transform(test)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b--', label="Precision")
    plt.plot(thresholds, recalls[:-1], 'g-', label="Recall")
    plt.legend(loc="lower right")
    plt.show()


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, linewidth=2, label="fpr vs tpr")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend(loc="lower right")
    #plt.title("Precision vs Recall")
    plt.show()


def main():
    # raw dataframe
    titanic_train = load_titanic_data("train.csv")
    titanic_test = load_titanic_data("test.csv")

    # training labels
    titanic_train_labels = titanic_train['Survived'].copy()

    # prepare training input
    titanic_train_prepared, titanic_test_prepared = prepare_titanic_data(
        titanic_train, titanic_test)

    # initialize classifier
    classifier = RandomForestClassifier(random_state=42)

    # model training performance metrics
    titanic_scores = cross_val_predict(
        classifier, titanic_train_prepared, titanic_train_labels, cv=3, method="predict_proba")

    precisions, recalls, thresholds = precision_recall_curve(
        titanic_train_labels, titanic_scores[:, 1])
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

    fpr, tpr, thresholds = roc_curve(
        titanic_train_labels, titanic_scores[:, 1])
    plot_roc_curve(fpr, tpr)

    print("roc_auc_score", roc_auc_score(
        titanic_train_labels, titanic_scores[:, 1]))

    # train model
    classifier.fit(titanic_train_prepared, titanic_train_labels)

    # predict test labels
    titanic_test_pred = classifier.predict(titanic_test_prepared)
    print("titanic_test_pred", titanic_test_pred.dtype, titanic_test_pred.shape)

    # build submission dataframe
    titanic_test_solution = titanic_test[['PassengerId']]
    titanic_test_pred_df = pd.DataFrame(titanic_test_pred,
                                        columns=['Survived'], index=titanic_test_solution.index)
    titanic_test_solution = titanic_test_solution.join(titanic_test_pred_df)
    csv_path = os.path.join(DATA_PATH, 'submission.csv')
    titanic_test_solution.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
