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
    df = utils.open_dataset("src/datasets/MEP Dataset/mep1_processed.csv")

    return df

def run_model_training():
    df = load_processed_dataset()
    for i in range(30):
        print(f'########################### {i+1} execution ###########################')
        manager = multiprocessing.Manager()
        results = manager.dict()
        p = multiprocessing.Process(target=training_model, args = (df,i, results))
        p.start()
        p.join()
        #results = training_model(df,i)
        
        for keys,item in results.items():
            
            utils.to_csv("src/output/reports/mep/mep_reports.csv", item,"mep")
        if(i < 29):
            print('########################### IDLE TIME START ###########################')
            sleep(60)
            print('########################### IDLE TIME FINISH ###########################')
    return

#@track_emissions(country_iso_code='ITA',offline=True)
def training_model(dataset,index, process_results):
    tracker = OfflineEmissionsTracker(country_iso_code="ITA")
    # setting dataset feature names
    features = dataset.columns.tolist()

    # we remove the target feature name from the feature name list
    features.remove('Probability')

    # setting target feature name
    target = ['Probability']

    # setting dataset features
    X = dataset[features]

    # setting dataset target feature
    y = dataset[target]


    # we build the standard model via pipeline containing a scaler for data normalization and a regressor
    lr_model_pipeline = make_pipeline(StandardScaler(),LogisticRegression())
    rf_model_pipeline = make_pipeline(StandardScaler(),RandomForestClassifier())
    svm_model_pipeline = make_pipeline(StandardScaler(),LinearSVC())
    xgb_model_pipeline = make_pipeline(StandardScaler(), xgb.XGBClassifier())

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=index)

    print(f'######### Training models #########')

    tracker.start()
    start_lr = datetime.now()
    lr_model_pipeline.fit(X_train,y_train.values.ravel())
    end_lr = datetime.now()
    tracker.stop()

    
    tracker.start()
    start_rf = datetime.now()
    rf_model_pipeline.fit(X_train,y_train.values.ravel())
    end_rf = datetime.now()
    tracker.stop()

    
    tracker.start()
    start_svm = datetime.now()
    svm_model_pipeline.fit(X_train,y_train.values.ravel())
    end_svm = datetime.now()
    tracker.stop()

    tracker.start()
    start_xgb = datetime.now()
    xgb_model_pipeline.fit(X_train,y_train.values.ravel())
    end_xgb = datetime.now()
    tracker.stop()


    print(f'######### Saving models #########')
    pickle.dump(lr_model_pipeline,open(f'src/output/standard_models/mep/lr_model_{index}.sav','wb'))
    pickle.dump(rf_model_pipeline,open(f'src/output/standard_models/mep/rf_model_{index}.sav','wb'))
    pickle.dump(xgb_model_pipeline,open(f'src/output/standard_models/mep/xgb_model_{index}.sav','wb'))
    pickle.dump(svm_model_pipeline,open(f'src/output/standard_models/mep/svm_model_{index}.sav','wb'))



    lr_results = {'dataset':'mep','iteration': index+1,'model':'lr', 'type': 'Baseline', 'sol_name':'Baseline',
                  'elapsed_time':(end_lr - start_lr).total_seconds(), 'size': utils.get_mb_size(f"src/output/standard_models/mep/lr_model_{index}.sav")}
    rf_results = {'dataset':'mep','iteration': index+1,'model':'rf', 'type': 'Baseline', 'sol_name':'Baseline',
                  'elapsed_time':(end_rf - start_rf).total_seconds(), 'size': utils.get_mb_size(f"src/output/standard_models/mep/rf_model_{index}.sav")}
    svm_results = {'dataset':'mep','iteration': index+1,'model':'svm', 'type': 'Baseline', 'sol_name':'Baseline',
                   'elapsed_time':(end_svm - start_svm).total_seconds(), 'size': utils.get_mb_size(f"src/output/standard_models/mep/svm_model_{index}.sav")}
    xgb_results = {'dataset':'mep','iteration': index+1,'model':'xgb', 'type': 'Baseline', 'sol_name':'Baseline',
                   'elapsed_time':(end_xgb - start_xgb).total_seconds(), 'size': utils.get_mb_size(f"src/output/standard_models/mep/xgb_model_{index}.sav")}

        

    print(f'######### Testing models #########')
    utils.validate(lr_model_pipeline,X_test,y_test,lr_results)
    utils.validate(rf_model_pipeline,X_test,y_test,rf_results)
    utils.validate(svm_model_pipeline,X_test,y_test,svm_results)
    utils.validate(xgb_model_pipeline,X_test,y_test,xgb_results)

    X_test_df = X_test.copy(deep=True)
    X_test_df['Probability'] = y_test

    lr_pred = X_test_df.copy(deep=True)
    lr_pred['Probability'] = lr_model_pipeline.predict(X_test)

    rf_pred =  X_test_df.copy(deep=True)
    rf_pred['Probability'] = rf_model_pipeline.predict(X_test)

    svm_pred =  X_test_df.copy(deep=True)
    svm_pred['Probability'] = svm_model_pipeline.predict(X_test)

    xgb_pred =  X_test_df.copy(deep=True)
    xgb_pred['Probability'] = xgb_model_pipeline.predict(X_test)

    eq_odds_fair_report(X_test_df,lr_pred,lr_results)
    eq_odds_fair_report(X_test_df,rf_pred,rf_results)
    eq_odds_fair_report(X_test_df,svm_pred,svm_results)
    eq_odds_fair_report(X_test_df,xgb_pred,xgb_results)



    process_results["lr"] = lr_results 
    process_results["rf"] = rf_results
    process_results["svm"] = svm_results
    process_results["xgb"] = xgb_results

    del tracker
    gc.collect()
    print(f'######### OPERATIONS SUCCESSFULLY COMPLETED #########')


def eq_odds_fair_report(dataset,prediction,results):

    sex_features = ['sex']

    aif_sex_dataset = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Probability'],
        protected_attribute_names=sex_features,
    )

    aif_sex_pred = BinaryLabelDataset(
        df=prediction,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Probability'],
        protected_attribute_names=sex_features,
    )

    sex_privileged_groups = [{'sex': 1}]
    sex_unprivileged_groups = [{'sex': 0}]

    metrics = ClassificationMetric(dataset=aif_sex_dataset,classified_dataset=aif_sex_pred,unprivileged_groups=sex_unprivileged_groups,privileged_groups=sex_privileged_groups)

    results['sex_mean_diff'] = round(metrics.mean_difference(),3)
    results['sex_eq_opp_diff'] = round(metrics.equal_opportunity_difference(),3)
    results['sex_avg_odds_diff'] = round(metrics.average_odds_difference(),3)

    race_feature = ['race']

    aif_race_dataset = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Probability'],
        protected_attribute_names=race_feature,
    )

    aif_race_pred = BinaryLabelDataset(
        df=prediction,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Probability'],
        protected_attribute_names=race_feature,
    )

    race_privileged_groups = [{'race': 1}]
    race_unprivileged_groups = [{'race': 0}]

    metrics = ClassificationMetric(dataset=aif_race_dataset,classified_dataset=aif_race_pred,unprivileged_groups=race_unprivileged_groups,privileged_groups=race_privileged_groups)

    results['race_mean_diff'] = round(metrics.mean_difference(),3)
    results['race_eq_opp_diff'] = round(metrics.equal_opportunity_difference(),3)
    results['race_avg_odds_diff'] = round(metrics.average_odds_difference(),3)

    protected_features = ['sex','race']

    aif_overall_dataset = BinaryLabelDataset(
        df=dataset,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Probability'],
        protected_attribute_names=protected_features,
    )

    aif_overall_pred = BinaryLabelDataset(
        df=prediction,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['Probability'],
        protected_attribute_names=protected_features,
    )

    privileged_groups = [{'sex': 1} | {'race': 1}]
    unprivileged_groups = [{'sex': 0, 'race': 0}]

    metrics = ClassificationMetric(dataset=aif_overall_dataset,classified_dataset=aif_overall_pred,unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)

    results['overall_mean_diff'] = round(metrics.mean_difference(),3)
    results['overall_eq_opp_diff'] = round(metrics.equal_opportunity_difference(),3)
    results['overall_avg_odds_diff'] = round(metrics.average_odds_difference(),3)
