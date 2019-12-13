# Copyright 2019 H2O.ai; Proprietary License;  -*- encoding: utf-8 -*-

import pandas as pd
import numpy as np
from numpy import nan
from scipy.special._ufuncs import expit
from scoring_h2oai_experiment_825d312a_1d57_11ea_b321_0242ac110002 import Scorer

#
# The format of input record to the Scorer.score() method is as follows:
#

# ---------------------------------------------------------
# Name                       Type      Range               
# ---------------------------------------------------------
# patient_nbr                float32   [135.0, 189502624.0]
# race                       object    -                   
# age                        object    -                   
# weight                     object    -                   
# admission_type_id          float32   [1.0, 8.0]          
# discharge_disposition_id   float32   [1.0, 28.0]         
# admission_source_id        float32   [1.0, 25.0]         
# time_in_hospital           float32   [1.0, 14.0]         
# payer_code                 object    -                   
# medical_specialty          object    -                   
# num_lab_procedures         float32   [1.0, 132.0]        
# num_procedures             float32   [0.0, 6.0]          
# num_medications            float32   [1.0, 81.0]         
# number_outpatient          float32   [0.0, 42.0]         
# number_emergency           float32   [0.0, 76.0]         
# number_inpatient           float32   [0.0, 21.0]         
# diag_1                     object    -                   
# diag_2                     object    -                   
# diag_3                     object    -                   
# number_diagnoses           float32   [1.0, 16.0]         
# max_glu_serum              object    -                   
# A1Cresult                  object    -                   
# metformin                  object    -                   
# nateglinide                object    -                   
# chlorpropamide             object    -                   
# glimepiride                object    -                   
# acetohexamide              object    -                   
# tolbutamide                object    -                   
# pioglitazone               object    -                   
# rosiglitazone              object    -                   
# miglitol                   object    -                   
# insulin                    object    -                   
# glyburide-metformin        object    -                   
# glimepiride-pioglitazone   object    -                   
# metformin-rosiglitazone    object    -                   
# metformin-pioglitazone     object    -                   
# diabetesMed                object    -                   
# ---------------------------------------------------------


#
# Create a singleton Scorer instance.
# For optimal performance, create a Scorer instance once, and call score() or score_batch() multiple times.
#
scorer = Scorer()


#
# To score one row at a time, use the Scorer.score() method (this can seem slow due to one-time overhead):
#

print('---------- Score Row ----------')
print(scorer.score([
    '5220.0',  # patient_nbr
    'AfricanAmerican',  # race
    '[20-30)',  # age
    '[25-50)',  # weight
    '2.0',  # admission_type_id
    '2.0',  # discharge_disposition_id
    '6.0',  # admission_source_id
    '10.0',  # time_in_hospital
    'BC',  # payer_code
    'Family/GeneralPractice',  # medical_specialty
    '6.0',  # num_lab_procedures
    '5.0',  # num_procedures
    '5.0',  # num_medications
    '3.0',  # number_outpatient
    '5.0',  # number_emergency
    '2.0',  # number_inpatient
    '157',  # diag_1
    '202',  # diag_2
    '139',  # diag_3
    '9.0',  # number_diagnoses
    '>200',  # max_glu_serum
    'Norm',  # A1Cresult
    'Steady',  # metformin
    'No',  # nateglinide
    'No',  # chlorpropamide
    'Steady',  # glimepiride
    'No',  # acetohexamide
    'No',  # tolbutamide
    'Steady',  # pioglitazone
    'Up',  # rosiglitazone
    'No',  # miglitol
    'Up',  # insulin
    'No',  # glyburide-metformin
    'No',  # glimepiride-pioglitazone
    'No',  # metformin-rosiglitazone
    'No',  # metformin-pioglitazone
    'Yes',  # diabetesMed
]))
print(scorer.score([
    '13041.0',  # patient_nbr
    'Asian',  # race
    '[50-60)',  # age
    '[175-200)',  # weight
    '6.0',  # admission_type_id
    '3.0',  # discharge_disposition_id
    '8.0',  # admission_source_id
    '6.0',  # time_in_hospital
    'HM',  # payer_code
    'Emergency/Trauma',  # medical_specialty
    '1.0',  # num_lab_procedures
    '1.0',  # num_procedures
    '4.0',  # num_medications
    '3.0',  # number_outpatient
    '0.0',  # number_emergency
    '6.0',  # number_inpatient
    '196',  # diag_1
    '112',  # diag_2
    '112',  # diag_3
    '5.0',  # number_diagnoses
    'Norm',  # max_glu_serum
    'Norm',  # A1Cresult
    'Down',  # metformin
    'No',  # nateglinide
    'No',  # chlorpropamide
    'Up',  # glimepiride
    'No',  # acetohexamide
    'No',  # tolbutamide
    'No',  # pioglitazone
    'Up',  # rosiglitazone
    'No',  # miglitol
    'Steady',  # insulin
    'No',  # glyburide-metformin
    'No',  # glimepiride-pioglitazone
    'No',  # metformin-rosiglitazone
    'No',  # metformin-pioglitazone
    'No',  # diabetesMed
]))
print(scorer.score([
    '135.0',  # patient_nbr
    'AfricanAmerican',  # race
    '[60-70)',  # age
    '[175-200)',  # weight
    '6.0',  # admission_type_id
    '6.0',  # discharge_disposition_id
    '8.0',  # admission_source_id
    '2.0',  # time_in_hospital
    'DM',  # payer_code
    'Cardiology',  # medical_specialty
    '6.0',  # num_lab_procedures
    '5.0',  # num_procedures
    '4.0',  # num_medications
    '3.0',  # number_outpatient
    '1.0',  # number_emergency
    '5.0',  # number_inpatient
    '154',  # diag_1
    '211',  # diag_2
    '204',  # diag_3
    '7.0',  # number_diagnoses
    'Norm',  # max_glu_serum
    '>8',  # A1Cresult
    'No',  # metformin
    'No',  # nateglinide
    'No',  # chlorpropamide
    'Steady',  # glimepiride
    'No',  # acetohexamide
    'No',  # tolbutamide
    'No',  # pioglitazone
    'Down',  # rosiglitazone
    'No',  # miglitol
    'Down',  # insulin
    'No',  # glyburide-metformin
    'No',  # glimepiride-pioglitazone
    'No',  # metformin-rosiglitazone
    'No',  # metformin-pioglitazone
    'No',  # diabetesMed
]))
print(scorer.score([
    '6228.0',  # patient_nbr
    'Other',  # race
    '[10-20)',  # age
    '[75-100)',  # weight
    '4.0',  # admission_type_id
    '1.0',  # discharge_disposition_id
    '20.0',  # admission_source_id
    '10.0',  # time_in_hospital
    'CM',  # payer_code
    'InfectiousDiseases',  # medical_specialty
    '5.0',  # num_lab_procedures
    '1.0',  # num_procedures
    '5.0',  # num_medications
    '3.0',  # number_outpatient
    '8.0',  # number_emergency
    '0.0',  # number_inpatient
    '180',  # diag_1
    '162',  # diag_2
    '196',  # diag_3
    '3.0',  # number_diagnoses
    '>300',  # max_glu_serum
    '>8',  # A1Cresult
    'Down',  # metformin
    'No',  # nateglinide
    'Steady',  # chlorpropamide
    'Down',  # glimepiride
    'No',  # acetohexamide
    'No',  # tolbutamide
    'No',  # pioglitazone
    'Up',  # rosiglitazone
    'No',  # miglitol
    'Down',  # insulin
    'No',  # glyburide-metformin
    'No',  # glimepiride-pioglitazone
    'No',  # metformin-rosiglitazone
    'No',  # metformin-pioglitazone
    'No',  # diabetesMed
]))
print(scorer.score([
    '27936.0',  # patient_nbr
    'Caucasian',  # race
    '[90-100)',  # age
    '[0-25)',  # weight
    '5.0',  # admission_type_id
    '2.0',  # discharge_disposition_id
    '7.0',  # admission_source_id
    '9.0',  # time_in_hospital
    'OG',  # payer_code
    'Emergency/Trauma',  # medical_specialty
    '1.0',  # num_lab_procedures
    '5.0',  # num_procedures
    '2.0',  # num_medications
    '3.0',  # number_outpatient
    '0.0',  # number_emergency
    '3.0',  # number_inpatient
    '180',  # diag_1
    '204',  # diag_2
    '198',  # diag_3
    '8.0',  # number_diagnoses
    'Norm',  # max_glu_serum
    '>8',  # A1Cresult
    'Up',  # metformin
    'No',  # nateglinide
    'Steady',  # chlorpropamide
    'Up',  # glimepiride
    'No',  # acetohexamide
    'No',  # tolbutamide
    'No',  # pioglitazone
    'Up',  # rosiglitazone
    'No',  # miglitol
    'Down',  # insulin
    'No',  # glyburide-metformin
    'No',  # glimepiride-pioglitazone
    'No',  # metformin-rosiglitazone
    'No',  # metformin-pioglitazone
    'No',  # diabetesMed
]))


#
# To score a batch of rows, use the Scorer.score_batch() method (much faster than repeated one-row scoring):
#
print('---------- Score Frame ----------')
columns = [
    pd.Series(['5220.0', '13041.0', '135.0', '6228.0', '27936.0', '27936.0', '5220.0', '12105.0', '6228.0', '5220.0'], name='patient_nbr', dtype='float32'),
    pd.Series(['AfricanAmerican', 'Asian', 'AfricanAmerican', 'Other', 'Caucasian', 'Hispanic', 'Caucasian', 'Caucasian', 'Caucasian', 'Hispanic'], name='race', dtype='object'),
    pd.Series(['[20-30)', '[50-60)', '[60-70)', '[10-20)', '[90-100)', '[10-20)', '[90-100)', '[60-70)', '[10-20)', '[30-40)'], name='age', dtype='object'),
    pd.Series(['[25-50)', '[175-200)', '[175-200)', '[75-100)', '[0-25)', '[0-25)', '[100-125)', '[125-150)', '[150-175)', '[100-125)'], name='weight', dtype='object'),
    pd.Series(['2.0', '6.0', '6.0', '4.0', '5.0', '8.0', '3.0', '2.0', '4.0', '4.0'], name='admission_type_id', dtype='float32'),
    pd.Series(['2.0', '3.0', '6.0', '1.0', '2.0', '3.0', '6.0', '7.0', '3.0', '1.0'], name='discharge_disposition_id', dtype='float32'),
    pd.Series(['6.0', '8.0', '8.0', '20.0', '7.0', '7.0', '17.0', '6.0', '5.0', '7.0'], name='admission_source_id', dtype='float32'),
    pd.Series(['10.0', '6.0', '2.0', '10.0', '9.0', '1.0', '1.0', '2.0', '7.0', '10.0'], name='time_in_hospital', dtype='float32'),
    pd.Series(['BC', 'HM', 'DM', 'CM', 'OG', 'OG', 'MD', 'HM', 'CM', 'CH'], name='payer_code', dtype='object'),
    pd.Series(['Family/GeneralPractice', 'Emergency/Trauma', 'Cardiology', 'InfectiousDiseases', 'Emergency/Trauma', 'InfectiousDiseases', 'Endocrinology', 'Dentistry', 'Gastroenterology', 'Endocrinology'], name='medical_specialty', dtype='object'),
    pd.Series(['6.0', '1.0', '6.0', '5.0', '1.0', '1.0', '12.0', '5.0', '8.0', '11.0'], name='num_lab_procedures', dtype='float32'),
    pd.Series(['5.0', '1.0', '5.0', '1.0', '5.0', '3.0', '3.0', '4.0', '6.0', '0.0'], name='num_procedures', dtype='float32'),
    pd.Series(['5.0', '4.0', '4.0', '5.0', '2.0', '6.0', '10.0', '1.0', '5.0', '1.0'], name='num_medications', dtype='float32'),
    pd.Series(['3.0', '3.0', '3.0', '3.0', '3.0', '3.0', '0.0', '2.0', '1.0', '0.0'], name='number_outpatient', dtype='float32'),
    pd.Series(['5.0', '0.0', '1.0', '8.0', '0.0', '2.0', '0.0', '5.0', '5.0', '2.0'], name='number_emergency', dtype='float32'),
    pd.Series(['2.0', '6.0', '5.0', '0.0', '3.0', '3.0', '1.0', '3.0', '3.0', '4.0'], name='number_inpatient', dtype='float32'),
    pd.Series(['157', '196', '154', '180', '180', '11', '185', '185', '157', '151'], name='diag_1', dtype='object'),
    pd.Series(['202', '112', '211', '162', '204', '162', '204', '162', '197', '205'], name='diag_2', dtype='object'),
    pd.Series(['139', '112', '204', '196', '198', '198', '197', '197', '153', '162'], name='diag_3', dtype='object'),
    pd.Series(['9.0', '5.0', '7.0', '3.0', '8.0', '5.0', '5.0', '9.0', '2.0', '6.0'], name='number_diagnoses', dtype='float32'),
    pd.Series(['>200', 'Norm', 'Norm', '>300', 'Norm', 'Norm', 'Norm', 'Norm', '>200', '>300'], name='max_glu_serum', dtype='object'),
    pd.Series(['Norm', 'Norm', '>8', '>8', '>8', '>7', 'Norm', 'Norm', '>8', '>7'], name='A1Cresult', dtype='object'),
    pd.Series(['Steady', 'Down', 'No', 'Down', 'Up', 'Steady', 'Steady', 'No', 'Steady', 'Down'], name='metformin', dtype='object'),
    pd.Series(['No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No'], name='nateglinide', dtype='object'),
    pd.Series(['No', 'No', 'No', 'Steady', 'Steady', 'Steady', 'No', 'No', 'Steady', 'Steady'], name='chlorpropamide', dtype='object'),
    pd.Series(['Steady', 'Up', 'Steady', 'Down', 'Up', 'No', 'No', 'Down', 'Steady', 'Up'], name='glimepiride', dtype='object'),
    pd.Series(['No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No'], name='acetohexamide', dtype='object'),
    pd.Series(['No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No'], name='tolbutamide', dtype='object'),
    pd.Series(['Steady', 'No', 'No', 'No', 'No', 'Up', 'Up', 'Steady', 'No', 'Up'], name='pioglitazone', dtype='object'),
    pd.Series(['Up', 'Up', 'Down', 'Up', 'Up', 'No', 'No', 'Up', 'Down', 'No'], name='rosiglitazone', dtype='object'),
    pd.Series(['No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No'], name='miglitol', dtype='object'),
    pd.Series(['Up', 'Steady', 'Down', 'Down', 'Down', 'Down', 'Steady', 'Up', 'Down', 'Down'], name='insulin', dtype='object'),
    pd.Series(['No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No'], name='glyburide-metformin', dtype='object'),
    pd.Series(['No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No'], name='glimepiride-pioglitazone', dtype='object'),
    pd.Series(['No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No'], name='metformin-rosiglitazone', dtype='object'),
    pd.Series(['No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'No'], name='metformin-pioglitazone', dtype='object'),
    pd.Series(['Yes', 'No', 'No', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'Yes'], name='diabetesMed', dtype='object'),
]
df = pd.concat(columns, axis=1)
print(scorer.score_batch(df))

##  Recommended workflow with datatable (fast and consistent with training):
import datatable as dt
dt.Frame(df).to_csv("test.csv")          # turn above dummy frame into a CSV (for convenience)
test_dt = dt.fread("test.csv", na_strings=['', '?', 'None', 'nan', 'NA', 'N/A', 'unknown', 'inf', '-inf', '1.7976931348623157e+308', '-1.7976931348623157e+308'])           # parse test set CSV file into datatable (with consistent NA handling)
preds_df = scorer.score_batch(test_dt)   # make predictions (pandas frame)
dt.Frame(preds_df).to_csv("preds.csv")   # save pandas frame to CSV using datatable


#
# The following lines demonstrate how to obtain per-feature prediction contributions per row. These can be
# very helpful in interpreting the model's predictions for individual observations (rows).
# Note that the contributions are in margin space (link space), so for binomial models the application of the
# final logistic function is omitted, while for multinomial models, the application of the final softmax function is
# omitted and for regression models the inverse link function is omitted (such as exp/square/re-normalization/etc.).
# This ensures that we can provide per-feature contributions that add up to the model's prediction.
# To simulate the omission of the transformation from margin/link space back to the probability or target space,
# and to get the predictions in the margin/link space, enable the output_margin flag. To get the prediction
# contributions, set pred_contribs=True. Note that you cannot provide both flags at the same time.
#

print('---------- Get Per-Feature Prediction Contributions for Row ----------')
print(scorer.score([
    '5220.0',  # patient_nbr
    'AfricanAmerican',  # race
    '[20-30)',  # age
    '[25-50)',  # weight
    '2.0',  # admission_type_id
    '2.0',  # discharge_disposition_id
    '6.0',  # admission_source_id
    '10.0',  # time_in_hospital
    'BC',  # payer_code
    'Family/GeneralPractice',  # medical_specialty
    '6.0',  # num_lab_procedures
    '5.0',  # num_procedures
    '5.0',  # num_medications
    '3.0',  # number_outpatient
    '5.0',  # number_emergency
    '2.0',  # number_inpatient
    '157',  # diag_1
    '202',  # diag_2
    '139',  # diag_3
    '9.0',  # number_diagnoses
    '>200',  # max_glu_serum
    'Norm',  # A1Cresult
    'Steady',  # metformin
    'No',  # nateglinide
    'No',  # chlorpropamide
    'Steady',  # glimepiride
    'No',  # acetohexamide
    'No',  # tolbutamide
    'Steady',  # pioglitazone
    'Up',  # rosiglitazone
    'No',  # miglitol
    'Up',  # insulin
    'No',  # glyburide-metformin
    'No',  # glimepiride-pioglitazone
    'No',  # metformin-rosiglitazone
    'No',  # metformin-pioglitazone
    'Yes',  # diabetesMed
], pred_contribs=True))


print('---------- Get Per-Feature Prediction Contributions for Frame ----------')
pred_contribs = scorer.score_batch(df, pred_contribs=True)  # per-feature prediction contributions
print(pred_contribs)


#
# The following lines demonstrate how to perform feature transformations without scoring.
# You can use this capability to transform input rows and fit models on the transformed frame
#   using an external ML tool of your choice, e.g. Sparkling Water or H2O.
#

#
# To transform a batch of rows (without scoring), use the Scorer.fit_transform_batch() method:
# This method fits the feature engineering pipeline on the given training frame, and applies it on the validation set,
# and optionally also on a test set.
#

# Transforms given datasets into enriched datasets with Driverless AI features')
#    train - for model fitting (do not use parts of this frame for parameter tuning)')
#    valid - for model parameter tuning')
#    test  - for final model testing (optional)')

print('---------- Transform Frames ----------')

# The target column 'readmitted' has to be present in all provided frames.
df['readmitted'] = pd.Series(['>30', '<30', '>30', '<30', 'NO', 'NO', '<30', '<30', 'NO', '>30'], dtype='object')

#  For demonstration only, do not use the same frame for train, valid and test!
train_munged, valid_munged, test_munged = \
  scorer.fit_transform_batch(train_frame=df, valid_frame=df, test_frame=df)
print(train_munged)  # for model fitting (use entire frame, no cross-validation)
print(valid_munged)  # for model validation (parameter tuning)
print(test_munged)   # for final pipeline testing (one time)

#
# To retrieve the original feature column names, use the Scorer.get_column_names() method:
# This method retrieves the input column names 
#

print('---------- Retrieve column names ----------')
print(scorer.get_column_names())

#
# To retrieve the transformed column names, use the Scorer.get_transformed_column_names() method:
# This method retrieves the transformed column names
#

print('---------- Retrieve transformed column names ----------')
print(scorer.get_transformed_column_names())

