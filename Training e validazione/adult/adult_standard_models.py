from utils import utils
from datetime import datetime
from time import sleep

from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import StandardScaler
from aif360.metrics import BinaryLabelDatasetMetric,ClassificationMetric
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import StandardDataset

from codecarbon import track_emissions
from codecarbon import OfflineEmissionsTracker

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.svm import LinearSVC
import pickle
import gc
import multiprocessing

def load_processed_dataset():
    """
    Load the processed dataset.
    """
    df = utils.open_dataset("src/datasets/Adult Dataset/adult_dataset_processed.csv")
    
    # Drop ID from the dataset
    df.drop('ID', inplace=True, axis=1)

    return df

def run_model_training():
    """
    Execute the training process for the model.
    """
    df = load_processed_dataset()
    for i in range(30):
        print(f'########################### {i+1} execution ###########################')
        
        manager = multiprocessing.Manager()
        results = manager.dict()
        p = multiprocessing.Process(target=training_model, args=(df, i, results))
        p.start()
        p.join()
        
        for keys, item in results.items():
            utils.to_csv("src/output/reports/adult/adult_reports.csv", item, "adult")
        if i < 29:
            print('########################### IDLE TIME START ###########################')
            sleep(60)
            print('########################### IDLE TIME FINISH ###########################')
    return

#@track_emissions(country_iso_code='ITA',offline=True)
def training_model(dataset, index, process_results):
    """
    Perform training of the model.
    """
    tracker = OfflineEmissionsTracker(country_iso_code="ITA")

    # Initialize results dictionaries for each model
    lr_results = {'dataset': 'adult', 'iteration': index + 1, 'model': 'lr', 'type': 'Baseline', 'sol_name': 'Baseline'}
    rf_results = {'dataset': 'adult', 'iteration': index + 1, 'model': 'rf', 'type': 'Baseline', 'sol_name': 'Baseline'}
    svm_results = {'dataset': 'adult', 'iteration': index + 1, 'model': 'svm', 'type': 'Baseline', 'sol_name': 'Baseline'}
    xgb_results = {'dataset': 'adult', 'iteration': index + 1, 'model': 'xgb', 'type': 'Baseline', 'sol_name': 'Baseline'}

    # Get the feature names from the dataset
    features = dataset.columns.tolist()

    # Remove the target feature from the feature list
    features.remove('salary')

    # Set the target feature
    target = ['salary']

    fair_results = {}

    # Set dataset features
    X = dataset[features]

    # Set dataset target feature
    y = dataset[target]

    # Construct standard models using pipelines containing a scaler for data normalization and a regressor
    lr_model_pipeline = make_pipeline(StandardScaler(), LogisticRegression())
    rf_model_pipeline = make_pipeline(StandardScaler(), RandomForestClassifier())
    svm_model_pipeline = make_pipeline(StandardScaler(), LinearSVC())
    xgb_model_pipeline = make_pipeline(StandardScaler(), xgb.XGBClassifier())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=index)

    # Train the baseline model on the training set
    print(f'######### Training models #########')

    tracker.start()
    start_lr = datetime.now()
    lr_model_pipeline.fit(X_train, y_train.values.ravel())
    end_lr = datetime.now()
    tracker.stop()

    tracker.start()
    start_rf = datetime.now()
    rf_model_pipeline.fit(X_train, y_train.values.ravel())
    end_rf = datetime.now()
    tracker.stop()

    tracker.start()
    start_svm = datetime.now()
    svm_model_pipeline.fit(X_train, y_train.values.ravel())
    end_svm = datetime.now()
    tracker.stop()

    tracker.start()
    start_xgb = datetime.now()
    xgb_model_pipeline.fit(X_train, y_train.values.ravel())
    end_xgb = datetime.now()
    tracker.stop()

    print(f'######### Saving models #########')
    pickle.dump(lr_model_pipeline, open(f'src/output/standard_models/adult/lr_model_{index}.sav', 'wb'))
    pickle.dump(rf_model_pipeline, open(f'src/output/standard_models/adult/rf_model_{index}.sav', 'wb'))
    pickle.dump(xgb_model_pipeline, open(f'src/output/standard_models/adult/xgb_model_{index}.sav', 'wb'))
    pickle.dump(svm_model_pipeline, open(f'src/output/standard_models/adult/svm_model_{index}.sav', 'wb'))

    # Calculate elapsed time and size for each model
    lr_results = {'dataset': 'adult', 'iteration': index + 1, 'model': 'lr', 'type': 'Baseline', 'sol_name': 'Baseline',
                  'elapsed_time': (end_lr - start_lr).total_seconds(), 'size': utils.get_mb_size(f"src/output/standard_models/adult/lr_model_{index}.sav")}
    rf_results = {'dataset': 'adult', 'iteration': index + 1, 'model': 'rf', 'type': 'Baseline', 'sol_name': 'Baseline',
                  'elapsed_time': (end_rf - start_rf).total_seconds(), 'size': utils.get_mb_size(f"src/output/standard_models/adult/rf_model_{index}.sav")}
    svm_results = {'dataset': 'adult','iteration': index + 1, 'model': 'svm', 'type': 'Baseline', 'sol_name': 'Baseline',
                   'elapsed_time': (end_svm - start_svm).total_seconds(), 'size': utils.get_mb_size(f"src/output/standard_models/adult/svm_model_{index}.sav")}
    xgb_results = {'dataset': 'adult', 'iteration': index + 1, 'model': 'xgb', 'type': 'Baseline', 'sol_name': 'Baseline',
                   'elapsed_time': (end_xgb - start_xgb).total_seconds(), 'size': utils.get_mb_size(f"src/output/standard_models/adult/xgb_model_{index}.sav")}

    # Test models and calculate evaluation metrics
    print(f'######### Testing models #########')
    utils.validate(lr_model_pipeline, X_test, y_test, lr_results)
    utils.validate(rf_model_pipeline, X_test, y_test, rf_results)
    utils.validate(svm_model_pipeline, X_test, y_test, svm_results)
    utils.validate(xgb_model_pipeline, X_test, y_test, xgb_results)

    X_test_df = X_test.copy(deep=True)
    X_test_df['salary'] = y_test

    lr_pred = X_test_df.copy(deep=True)
    lr_pred['salary'] = lr_model_pipeline.predict(X_test)

    rf_pred = X_test_df.copy(deep=True)
    rf_pred['salary'] = rf_model_pipeline.predict(X_test)

    svm_pred = X_test_df.copy(deep=True)
    svm_pred['salary'] = svm_model_pipeline.predict(X_test)

    xgb_pred = X_test_df.copy(deep=True)
    xgb_pred['salary'] = xgb_model_pipeline.predict(X_test)

    eq_odds_fair_report(X_test_df, lr_pred, lr_results)
    eq_odds_fair_report(X_test_df, rf_pred, rf_results)
    eq_odds_fair_report(X_test_df, svm_pred, svm_results)
    eq_odds_fair_report(X_test_df, xgb_pred, xgb_results)

    process_results["lr"] = lr_results
    process_results["rf"] = rf_results
    process_results["svm"] = svm_results
    process_results["xgb"] = xgb_results

    del tracker
    gc.collect()
    print(f'######### OPERATION COMPLETED SUCCESSFULLY #########')


def eq_odds_fair_report(dataset, prediction, results):
    """
    Generate fairness report using Equal Odds.
    """
    sex_features = ['sex_Male', 'sex_Female']

    aif_sex_dataset = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['salary'],
        protected_attribute_names=sex_features,
        privileged_protected_attributes=['sex_Male'],
    )

    aif_sex_pred = BinaryLabelDataset(
        df=prediction,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['salary'],
        protected_attribute_names=sex_features,
        privileged_protected_attributes=['sex_Male'],
    )

    sex_privileged_groups = [{'sex_Male': 1}]
    sex_unprivileged_groups = [{'sex_Female': 1}]

    metrics = ClassificationMetric(dataset=aif_sex_dataset, classified_dataset=aif_sex_pred, unprivileged_groups=sex_unprivileged_groups, privileged_groups=sex_privileged_groups)

    results['sex_mean_diff'] = round(metrics.mean_difference(), 3)
    results['sex_eq_opp_diff'] = round(metrics.equal_opportunity_difference(), 3)
    results['sex_avg_odds_diff'] = round(metrics.average_odds_difference(), 3)

    race_feature = ['race']

    aif_race_dataset = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['salary'],
        protected_attribute_names=race_feature,
    )

    aif_race_pred = BinaryLabelDataset(
        df=prediction,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['salary'],
        protected_attribute_names=race_feature,
    )

    race_privileged_groups = [{'race': 1}]
    race_unprivileged_groups = [{'race': 0}]

    metrics = ClassificationMetric(dataset=aif_race_dataset, classified_dataset=aif_race_pred, unprivileged_groups=race_unprivileged_groups, privileged_groups=race_privileged_groups)

    results['race_mean_diff'] = round(metrics.mean_difference(), 3)
    results['race_eq_opp_diff'] = round(metrics.equal_opportunity_difference(), 3)
    results['race_avg_odds_diff'] = round(metrics.average_odds_difference(), 3)

    protected_features = ['sex_Male', 'sex_Female', 'race']

    aif_overall_dataset = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['salary'],
        protected_attribute_names=protected_features,
    )

    aif_overall_pred = BinaryLabelDataset(
        df=prediction,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['salary'],
        protected_attribute_names=protected_features,
    )

    privileged_groups = [{'sex_Male': 1} | {'race': 1}]
    unprivileged_groups = [{'sex_Female': 1, 'race': 0}]

    metrics = ClassificationMetric(dataset=aif_overall_dataset, classified_dataset=aif_overall_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

    results['overall_mean_diff'] = round(metrics.mean_difference(), 3)
    results['overall_eq_opp_diff'] = round(metrics.equal_opportunity_difference(), 3)
    results['overall_avg_odds_diff'] = round(metrics.average_odds_difference(), 3)
