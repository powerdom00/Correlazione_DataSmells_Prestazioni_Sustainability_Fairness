from utils import utils
from datetime import datetime
from time import sleep

from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import StandardScaler

from aif360.datasets import BinaryLabelDataset
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric,ClassificationMetric

from codecarbon import track_emissions
from codecarbon import OfflineEmissionsTracker

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.svm import LinearSVC
import pickle
import gc
import multiprocessing



# Load the processed dataset
def load_processed_dataset():
    df = utils.open_dataset("src/datasets/German Credit Dataset/german_dataset_processed.csv")
    
    # Replace '2' with '0' in the 'Target' column
    df['Target'] = df['Target'].replace(2, 0)
    
    return df

# Function to run model training
def run_model_training():
    # Load the processed dataset
    df = load_processed_dataset()
    
    # Loop for model training
    for i in range(30):
        print(f'########################### {i+1} execution ###########################')
        
        # Initialize multiprocessing manager and results dictionary
        manager = multiprocessing.Manager()
        results = manager.dict()
        
        # Create a new process for training the model
        p = multiprocessing.Process(target=training_model, args=(df, i, results))
        p.start()
        p.join()
        
        # Save results to CSV
        for keys, item in results.items():
            utils.to_csv("src/output/reports/german/german_reports.csv", item, "german")
        
        # Pause execution if not the last iteration
        if i < 29:
            print('########################### IDLE TIME START ###########################')
            sleep(60)
            print('########################### IDLE TIME FINISH ###########################')
    
    return

# Function for training the model
def training_model(dataset, index, process_results):
    tracker = OfflineEmissionsTracker(country_iso_code="ITA")
    
    # Model learning function on the dataset

    # Set feature names
    features = dataset.columns.tolist()
    features.remove('Target')  # Remove the target feature name from the feature list
    target = ['Target']  # Set the target feature name

    # Set dataset features and target
    X = dataset[features]
    y = dataset[target]

    # Build standard models using pipelines
    lr_model_pipeline = make_pipeline(StandardScaler(), LogisticRegression())
    rf_model_pipeline = make_pipeline(StandardScaler(), RandomForestClassifier())
    svm_model_pipeline = make_pipeline(StandardScaler(), LinearSVC())
    xgb_model_pipeline = make_pipeline(StandardScaler(), xgb.XGBClassifier())

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=index)

    # Training the model
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

    # Save the trained models
    print(f'######### Saving models #########')
    pickle.dump(lr_model_pipeline, open(f'src/output/standard_models/german/lr_model_{index}.sav', 'wb'))
    pickle.dump(rf_model_pipeline, open(f'src/output/standard_models/german/rf_model_{index}.sav', 'wb'))
    pickle.dump(xgb_model_pipeline, open(f'src/output/standard_models/german/xgb_model_{index}.sav', 'wb'))
    pickle.dump(svm_model_pipeline, open(f'src/output/standard_models/german/svm_model_{index}.sav', 'wb'))

    # Define results for each model
    lr_results = {'dataset': 'german', 'iteration': index + 1, 'model': 'lr', 'type': 'Baseline', 'sol_name':'Baseline',
                  'elapsed_time': (end_lr - start_lr).total_seconds(), 'size': utils.get_mb_size(f"src/output/standard_models/german/lr_model_{index}.sav")}
    rf_results = {'dataset': 'german', 'iteration': index + 1, 'model': 'rf', 'type': 'Baseline','sol_name':'Baseline',
                  'elapsed_time': (end_rf - start_rf).total_seconds(), 'size': utils.get_mb_size(f"src/output/standard_models/german/rf_model_{index}.sav")}
    svm_results = {'dataset': 'german', 'iteration': index + 1, 'model': 'svm', 'type': 'Baseline','sol_name':'Baseline',
                   'elapsed_time': (end_svm - start_svm).total_seconds(), 'size': utils.get_mb_size(f"src/output/standard_models/german/svm_model_{index}.sav")}
    xgb_results = {'dataset': 'german', 'iteration': index + 1, 'model': 'xgb', 'type': 'Baseline','sol_name':'Baseline',
                   'elapsed_time': (end_xgb - start_xgb).total_seconds(), 'size': utils.get_mb_size(f"src/output/standard_models/german/xgb_model_{index}.sav")}

    # Evaluate the model
    print(f'######### Testing models #########')
    utils.validate(lr_model_pipeline, X_test, y_test, lr_results)
    utils.validate(rf_model_pipeline, X_test, y_test, rf_results)
    utils.validate(svm_model_pipeline, X_test, y_test, svm_results)
    utils.validate(xgb_model_pipeline, X_test, y_test, xgb_results)

    # Predictions
    X_test_df = X_test.copy(deep=True)
    X_test_df['Target'] = y_test

    lr_pred = X_test_df.copy(deep=True)
    lr_pred['Target'] = lr_model_pipeline.predict(X_test)

    rf_pred = X_test_df.copy(deep=True)
    rf_pred['Target'] = rf_model_pipeline.predict(X_test)

    svm_pred = X_test_df.copy(deep=True)
    svm_pred['Target'] = svm_model_pipeline.predict(X_test)

    xgb_pred = X_test_df.copy(deep=True)
    xgb_pred['Target'] = xgb_model_pipeline.predict(X_test)

    # Fairness report
    eq_odds_fair_report(X_test_df, lr_pred, lr_results)
    eq_odds_fair_report(X_test_df, rf_pred, rf_results)
    eq_odds_fair_report(X_test_df, svm_pred, svm_results)
    eq_odds_fair_report(X_test_df, xgb_pred, xgb_results)

    # Store results
    process_results["lr"] = lr_results
    process_results["rf"] = rf_results
    process_results["svm"] = svm_results
    process_results["xgb"] = xgb_results

    del tracker
    gc.collect()
    print(f'######### OPERATION COMPLETED SUCCESSFULLY #########')


# Fairness report function
def eq_odds_fair_report(dataset, prediction, results):
    sex_features = ['sex_A91', 'sex_A92', 'sex_A93', 'sex_A94']

    aif_sex_dataset = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=sex_features,
        privileged_protected_attributes=['sex_A93'],
    )

    aif_sex_pred = BinaryLabelDataset(
        df=prediction,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=sex_features,
        privileged_protected_attributes=['sex_A93'],
    )

    sex_privileged_groups = [{'sex_A93': 1}]
    sex_unprivileged_groups = [{'sex_A93': 0}]

    metrics = ClassificationMetric(dataset=aif_sex_dataset, classified_dataset=aif_sex_pred,
                                   unprivileged_groups=sex_unprivileged_groups, privileged_groups=sex_privileged_groups)

    results['sex_mean_diff'] = round(metrics.mean_difference(), 3)
    results['sex_eq_opp_diff'] = round(metrics.equal_opportunity_difference(), 3)
    results['sex_avg_odds_diff'] = round(metrics.average_odds_difference(), 3)

    age_feature = ['Age in years']

    aif_age_dataset = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=age_feature,
    )

    aif_age_pred = BinaryLabelDataset(
        df=prediction,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=age_feature,
    )

    age_privileged_groups = [{'Age in years': 1}]
    age_unprivileged_groups = [{'Age in years': 0}]

    metrics = ClassificationMetric(dataset=aif_age_dataset, classified_dataset=aif_age_pred,
                                   unprivileged_groups=age_unprivileged_groups, privileged_groups=age_privileged_groups)

    results['age_mean_diff'] = round(metrics.mean_difference(), 3)
    results['age_eq_opp_diff'] = round(metrics.equal_opportunity_difference(), 3)
    results['age_avg_odds_diff'] = round(metrics.average_odds_difference(), 3)

    protected_features = [
        'sex_A91', 'sex_A92', 'sex_A93', 'sex_A94', 'Age in years'
    ]

    aif_overall_dataset = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=protected_features,
    )

    aif_overall_pred = BinaryLabelDataset(
        df=prediction,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Target'],
        protected_attribute_names=protected_features,
    )

    privileged_groups = [{'sex_A93': 1} | {'Age in years': 1}]
    unprivileged_groups = [{'sex_A93': 0, 'Age in years': 0}]

    metrics = ClassificationMetric(dataset=aif_overall_dataset, classified_dataset=aif_overall_pred,
                                   unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

    results['overall_mean_diff'] = round(metrics.mean_difference(), 3)
    results['overall_eq_opp_diff'] = round(metrics.equal_opportunity_difference(), 3)
    results['overall_avg_odds_diff'] = round(metrics.average_odds_difference(), 3)