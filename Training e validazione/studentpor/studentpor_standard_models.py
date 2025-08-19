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


def load_processed_dataset():
    """
    Load the processed dataset.
    """
    # Load the dataset using a utility function
    df = utils.open_dataset("src/datasets/studentpor/student-por.csv")
    return df

def run_model_training():
    """
    Run the model training process.
    """
    # Load the processed dataset
    df = load_processed_dataset()
    
    # Loop through 30 iterations
    for i in range(30):
        print(f'########################### {i+1} execution ###########################')
        
        # Create a multiprocessing manager
        manager = multiprocessing.Manager()
        results = manager.dict()
        
        # Create a multiprocessing process for training_model function
        p = multiprocessing.Process(target=training_model, args=(df, i, results))
        p.start()
        p.join()
        
        # Save results to a CSV file
        for keys, item in results.items():
            utils.to_csv("src/output/reports/studentpor/studentpor_reports.csv", item, "heart_disease")
        
        # Add an idle time between iterations
        if i < 29:
            print('########################### IDLE TIME START ###########################')
            sleep(60)
            print('########################### IDLE TIME FINISH ###########################')

#@track_emissions(country_iso_code='ITA',offline=True)
def training_model(dataset, index, process_results):
    """
    Train the model using the dataset.
    """
    # Initialize an offline emissions tracker
    tracker = OfflineEmissionsTracker(country_iso_code="ITA")
    
    # Get the features and target from the dataset
    features = dataset.columns.tolist()
    features.remove('G3_high')
    target = ['G3_high']
    
    # Initialize a dictionary for storing results
    fair_results = {}
    
    # Set dataset features and target
    X = dataset[features]
    y = dataset[target]

    # Build standard models with different algorithms
    lr_model_pipeline = make_pipeline(StandardScaler(), LogisticRegression())
    rf_model_pipeline = make_pipeline(StandardScaler(), RandomForestClassifier())
    svm_model_pipeline = make_pipeline(StandardScaler(), LinearSVC())
    xgb_model_pipeline = make_pipeline(StandardScaler(), xgb.XGBClassifier())

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=index)

    # Train the models
    print(f'######### Training models #########')

    # Train LR model
    tracker.start()
    start_lr = datetime.now()
    lr_model_pipeline.fit(X_train, y_train.values.ravel())
    end_lr = datetime.now()
    tracker.stop()

    # Train RF model
    tracker.start()
    start_rf = datetime.now()
    rf_model_pipeline.fit(X_train, y_train.values.ravel())
    end_rf = datetime.now()
    tracker.stop()

    # Train SVM model
    tracker.start()
    start_svm = datetime.now()
    svm_model_pipeline.fit(X_train, y_train.values.ravel())
    end_svm = datetime.now()
    tracker.stop()

    # Train XGB model
    tracker.start()
    start_xgb = datetime.now()
    xgb_model_pipeline.fit(X_train, y_train.values.ravel())
    end_xgb = datetime.now()
    tracker.stop()

    # Save trained models
    print(f'######### Saving models #########')
    pickle.dump(lr_model_pipeline, open(f'src/output/standard_models/studentpor/lr_model_{index}.sav', 'wb'))
    pickle.dump(rf_model_pipeline, open(f'src/output/standard_models/studentpor/rf_model_{index}.sav', 'wb'))
    pickle.dump(xgb_model_pipeline, open(f'src/output/standard_models/studentpor/xgb_model_{index}.sav', 'wb'))
    pickle.dump(svm_model_pipeline, open(f'src/output/standard_models/studentpor/svm_model_{index}.sav', 'wb'))

    # Record model results
    lr_results = {
        'dataset': 'student-por',
        'iteration': index + 1,
        'model': 'lr',
        'type': 'Baseline',
        'sol_name': 'Baseline',
        'elapsed_time': (end_lr - start_lr).total_seconds(),
        'size': utils.get_mb_size(f"src/output/standard_models/studentpor/lr_model_{index}.sav")
    }
    rf_results = {
        'dataset': 'student-por',
        'iteration': index + 1,
        'model': 'rf',
        'type': 'Baseline',
        'sol_name': 'Baseline',
        'elapsed_time': (end_rf - start_rf).total_seconds(),
        'size': utils.get_mb_size(f"src/output/standard_models/studentpor/rf_model_{index}.sav")
    }
    svm_results = {
        'dataset': 'student-por',
        'iteration': index + 1,
        'model': 'svm',
        'type': 'Baseline',
        'sol_name': 'Baseline',
        'elapsed_time': (end_svm - start_svm).total_seconds(),
        'size': utils.get_mb_size(f"src/output/standard_models/studentpor/svm_model_{index}.sav")
    }
    xgb_results = {
        'dataset': 'student-por',
        'iteration': index + 1,
        'model': 'xgb',
        'type': 'Baseline',
        'sol_name': 'Baseline',
        'elapsed_time': (end_xgb - start_xgb).total_seconds(),
        'size': utils.get_mb_size(f"src/output/standard_models/studentpor/xgb_model_{index}.sav")
    }

    # Test the models
    print(f'######### Testing models #########')
    utils.validate(lr_model_pipeline, X_test, y_test, lr_results)
    utils.validate(rf_model_pipeline, X_test, y_test, rf_results)
    utils.validate(svm_model_pipeline, X_test, y_test, svm_results)
    utils.validate(xgb_model_pipeline, X_test, y_test, xgb_results)

    # Prepare predictions for fairness assessment
    X_test_df = X_test.copy(deep=True)
    X_test_df['G3_high'] = y_test

    lr_pred = X_test_df.copy(deep=True)
    lr_pred['G3_high'] = lr_model_pipeline.predict(X_test)

    rf_pred = X_test_df.copy(deep=True)
    rf_pred['G3_high'] = rf_model_pipeline.predict(X_test)

    svm_pred = X_test_df.copy(deep=True)
    svm_pred['G3_high'] = svm_model_pipeline.predict(X_test)

    xgb_pred = X_test_df.copy(deep=True)
    xgb_pred['G3_high'] = xgb_model_pipeline.predict(X_test)

    
    # Generate fairness reports
    eq_odds_fair_report(X_test_df, lr_pred, lr_results)
    eq_odds_fair_report(X_test_df, rf_pred, rf_results)
    eq_odds_fair_report(X_test_df, svm_pred, svm_results)
    eq_odds_fair_report(X_test_df, xgb_pred, xgb_results)
    

    # Record results
    process_results["lr"] = lr_results
    process_results["rf"] = rf_results
    process_results["svm"] = svm_results
    process_results["xgb"] = xgb_results

    # Clean up
    del tracker
    gc.collect()
    print(f'######### OPERATIONS SUCCESSFULLY COMPLETED #########')

def eq_odds_fair_report(dataset, prediction, results):
    """
    Generate fairness reports based on predictions.
    """
    # Define protected features
    sex_features = ['sex_F','sex_M']

    # Create BinaryLabelDataset for sex
    aif_sex_dataset = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['G3_high'],
        protected_attribute_names=sex_features,
    )

    aif_sex_pred = BinaryLabelDataset(
        df=prediction,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['G3_high'],
        protected_attribute_names=sex_features,
    )

    sex_privileged_groups = [{'sex_M': 1.0}]
    sex_unprivileged_groups = [{'sex_M': 0.0}]

    # Calculate fairness metrics for sex
    metrics = ClassificationMetric(dataset=aif_sex_dataset, classified_dataset=aif_sex_pred,
                                   unprivileged_groups=sex_unprivileged_groups, privileged_groups=sex_privileged_groups)

    results['sex_mean_diff'] = round(metrics.mean_difference(), 3)
    results['sex_eq_opp_diff'] = round(metrics.equal_opportunity_difference(), 3)
    results['sex_avg_odds_diff'] = round(metrics.average_odds_difference(), 3)
    results['overall_mean_diff'] = round(metrics.mean_difference(), 3)
    results['overall_eq_opp_diff'] = round(metrics.equal_opportunity_difference(), 3)
    results['overall_avg_odds_diff'] = round(metrics.average_odds_difference(), 3)

