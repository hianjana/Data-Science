# Importing libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import gc
import pickle
import csv

############# Data load ###################

train_dataset = pd.read_csv('/home/ubuntu/recommendations/data/sorted_rollup_5mths.csv', sep=',')
train_dataset = train_dataset.rename(columns={'e-account': 'e_account'})
train_dataset = train_dataset.rename(columns={'past_5mths_e-account': 'past_5mths_e_account'})
train_dataset = train_dataset.fillna(0)

################ Data Analysis and Pre-Processing ################

# Store all customer_codes
customer_codes = []
for each in train_dataset.cust_code:
        customer_codes.append(each)

# Convert customer_status into binary
customer_status = []
for each in train_dataset.customer_status:
    if each == 1:
        customer_status.append(1)
    else:
        customer_status.append(0)

cust_reltn_type_beg_of_month_A = []
for each in train_dataset.cust_reltn_type_beg_of_month:
    if each == 'A':
        cust_reltn_type_beg_of_month_A.append(1)
    else:
        cust_reltn_type_beg_of_month_A.append(0)

cust_reltn_type_beg_of_month_P = []
for each in train_dataset.cust_reltn_type_beg_of_month:
    if each == 'P':
        cust_reltn_type_beg_of_month_P.append(1)
    else:
        cust_reltn_type_beg_of_month_P.append(0)

cust_reltn_type_beg_of_month_R = []
for each in train_dataset.cust_reltn_type_beg_of_month:
    if each == 'R':
        cust_reltn_type_beg_of_month_R.append(1)
    else:
        cust_reltn_type_beg_of_month_R.append(0)

cust_reltn_type_beg_of_month_I = []
for each in train_dataset.cust_reltn_type_beg_of_month:
    if each == 'I':
        cust_reltn_type_beg_of_month_I.append(1)
    else:
        cust_reltn_type_beg_of_month_I.append(0)

cust_reltn_type_beg_of_month_other = []
for each in train_dataset.cust_reltn_type_beg_of_month:
    if each != 'A' and each != 'P' and each != 'R' and each != 'I':
        cust_reltn_type_beg_of_month_other.append(1)
    else:
        cust_reltn_type_beg_of_month_other.append(0)

# Convert gender into binary
gender = []
for each in train_dataset.gender:
    if each == 'V':
        gender.append(1)
    else:
        gender.append(0)

employment_status_A =[]
for each in train_dataset.employment_status:
    if each == 'A':
        employment_status_A.append(1)
    else:
        employment_status_A.append(0)

employment_status_B =[]
for each in train_dataset.employment_status:
    if each == 'B':
        employment_status_B.append(1)
    else:
        employment_status_B.append(0)

employment_status_F =[]
for each in train_dataset.employment_status:
    if each == 'F':
        employment_status_F.append(1)
    else:
        employment_status_F.append(0)

employment_status_N =[]
for each in train_dataset.employment_status:
    if each == 'N':
        employment_status_N.append(1)
    else:
        employment_status_N.append(0)

employment_status_P =[]
for each in train_dataset.employment_status:
    if each == 'P':
        employment_status_P.append(1)
    else:
        employment_status_P.append(0)

employment_status_S =[]
for each in train_dataset.employment_status:
    if each == 'S':
        employment_status_S.append(1)
    else:
        employment_status_S.append(0)

employment_status_other =[]
for each in train_dataset.employment_status:
    if each != 'A' and each != 'B' and each != 'F' and each != 'N' and each != 'P' and each != 'S':
        employment_status_other.append(1)
    else:
        employment_status_other.append(0)         

# Convert foreigner_flg into binary
foreigner_flg = []
for each in train_dataset.foreigner_flg:
    if each == 'S':
        foreigner_flg.append(1)
    else:
        foreigner_flg.append(0)

# Convert spouse_flg into binary
spouse_flg = []
for each in train_dataset.spouse_flg:
    if each == 'S':
        spouse_flg.append(1)
    else:
        spouse_flg.append(0)

# Convert residence_flg into binary
residence_flg = []
for each in train_dataset.residence_flg:
    if each == 'S':
        residence_flg.append(1)
    else:
        residence_flg.append(0)

segmentation_01 =[]
for each in train_dataset.segmentation:
    if each == '01 - TOP':
        segmentation_01.append(1)
    else:
        segmentation_01.append(0)

segmentation_02 =[]
for each in train_dataset.segmentation:
    if each == '02 - PARTICULARES':
        segmentation_02.append(1)
    else:
        segmentation_02.append(0)

segmentation_03 =[]
for each in train_dataset.segmentation:
    if each == '03 - UNIVERSITARIO':
        segmentation_03.append(1)
    else:
        segmentation_03.append(0)

segmentation_other = []
for each in train_dataset.segmentation:
    if each != '01 - TOP' and each != '02 - PARTICULARES' and each != '03 - UNIVERSITARIO':
        segmentation_other.append(1)
    else:
        segmentation_other.append(0)

# Convert spouse_flg into binary
deceased_index = []
for each in train_dataset.deceased_index:
    if each == 'S':
        deceased_index.append(1)
    else:
        deceased_index.append(0)

# Standardize value for cust_type_beg_of_month
cust_type_beg_of_month = []
for each in train_dataset.cust_type_beg_of_month:
    if each == 'P':
        int_each = 5
    else:
        try:
            flt_each = float(each)
            int_each = int(flt_each)
        except:
            int_each = 9
        
    if int_each == 1:
        cust_type_beg_of_month.append(1)
    elif int_each == 2:
        cust_type_beg_of_month.append(2)
    elif int_each == 3:
        cust_type_beg_of_month.append(3)
    elif int_each == 4:
        cust_type_beg_of_month.append(4)    
    elif int_each == 4:
        cust_type_beg_of_month.append(5) 
    else:
        cust_type_beg_of_month.append(int_each)

cust_type_beg_of_month_1     = []
for each in cust_type_beg_of_month:
    if each == 1:
        cust_type_beg_of_month_1.append(1)
    else:
        cust_type_beg_of_month_1.append(0)

cust_type_beg_of_month_2     = []
for each in cust_type_beg_of_month:
    if each == 2:
        cust_type_beg_of_month_2.append(1)
    else:
        cust_type_beg_of_month_2.append(0)

cust_type_beg_of_month_3     = []
for each in cust_type_beg_of_month:
    if each == 3:
        cust_type_beg_of_month_3.append(1)
    else:
        cust_type_beg_of_month_3.append(0)

cust_type_beg_of_month_4     = []
for each in cust_type_beg_of_month:
    if each == 4:
        cust_type_beg_of_month_4.append(1)
    else:
        cust_type_beg_of_month_4.append(0)

cust_type_beg_of_month_other     = []
for each in cust_type_beg_of_month:
    if each != 1 and each != 2 and each != 3 and each != 4:
        cust_type_beg_of_month_other.append(1)
    else:
        cust_type_beg_of_month_other.append(0)

# Convert new_customer_index into binary
new_customer_index = []
for each in train_dataset.new_customer_index:
    if each == 1:
        new_customer_index.append(1)
    else:
        new_customer_index.append(0)

# Convert country into binary for train
country = []
for each in train_dataset.country:
    if each == 'ES':
        country.append(1)
    else:
        country.append(0)

# Standardize values for seniority
seniority = []
for each in train_dataset.seniority:
    try:
        int_each = int(each)
        if int_each > 0:
            seniority.append(int_each)
        else:
            seniority.append(0)
    except:
        seniority.append(0)        

## Outlier detection
arr = seniority
elements = np.array(seniority)
mean_sen = np.mean(elements, axis=0)
sd_sen = np.std(elements, axis=0)
sen_below_mean_2sd = mean_sen - (2 * sd_sen)
sen_above_mean_2sd = mean_sen + (2 * sd_sen)

## Fixing outliers
seniority_no_outliers = []
for each in seniority:
    if each > sen_above_mean_2sd:        
        seniority_no_outliers.append(sen_above_mean_2sd)
    elif each < sen_below_mean_2sd: 
        seniority_no_outliers.append(sen_below_mean_2sd)
    else:
        seniority_no_outliers.append(each)    

# Convert string to integer for age
age = []    
for each in train_dataset.age:
    try:
        age.append(int(each))
    except:
        age.append(40)        

## Outlier detection
arr = age
elements = np.array(age)
mean_age = np.mean(elements, axis=0)
sd_age = np.std(elements, axis=0)
age_below_mean_3sd = mean_age - (3 * sd_age)
age_above_mean_3sd = mean_age + (3 * sd_age)

## Fixing outliers
age_no_outliers = []
for each in age:
    if each > age_above_mean_3sd:        
        age_no_outliers.append(age_above_mean_3sd)
    elif each < age_below_mean_3sd: 
        age_no_outliers.append(age_below_mean_3sd)
    else:
        age_no_outliers.append(each)  

# Standardize the column gross_income for train. In case of 'NA', replace it with the mean salary of Spain
gross_income = []
for each in train_dataset.gross_income:
    try:
        flt_each = float(each)
        int_each = int(flt_each)
        gross_income.append(int_each)   
    except:
        ##X_gross_income.append(-99999999)            
        gross_income.append(15816)

################# End of data preprocessing of regular predictors ################

############################# Add past 5 months predictors #############################

# Create seperate lists for each account type
past_5mths_savings_account = []
past_5mths_savings_account = [int(each) for each in train_dataset.past_5mths_savings_account]

past_5mths_particular_plus_account = []
past_5mths_particular_plus_account = [int(each) for each in train_dataset.past_5mths_particular_plus_account]

past_5mths_guarantees = []
past_5mths_guarantees = [int(each) for each in train_dataset.past_5mths_guarantees]

past_5mths_current_account = []
past_5mths_current_account = [int(each) for each in train_dataset.past_5mths_current_account]

past_5mths_derivada_account = []
past_5mths_derivada_account = [int(each) for each in train_dataset.past_5mths_derivada_account]

past_5mths_payroll_account = []
past_5mths_payroll_account = [int(each) for each in train_dataset.past_5mths_payroll_account]

past_5mths_junior_account = []
past_5mths_junior_account = [int(each) for each in train_dataset.past_5mths_junior_account]

past_5mths_mas_particular_account = []
past_5mths_mas_particular_account = [int(each) for each in train_dataset.past_5mths_mas_particular_account]

past_5mths_particular_account = []
past_5mths_particular_account = [int(each) for each in train_dataset.past_5mths_particular_account]

past_5mths_short_term_deposits = []
past_5mths_short_term_deposits = [int(each) for each in train_dataset.past_5mths_short_term_deposits]

past_5mths_medium_term_deposits = []
past_5mths_medium_term_deposits = [int(each) for each in train_dataset.past_5mths_medium_term_deposits]

past_5mths_long_term_deposits = []
past_5mths_long_term_deposits = [int(each) for each in train_dataset.past_5mths_long_term_deposits]

past_5mths_e_account = []
past_5mths_e_account = [int(each) for each in train_dataset.past_5mths_e_account]

past_5mths_funds = []
past_5mths_funds = [int(each) for each in train_dataset.past_5mths_funds]

past_5mths_mortgage = []
past_5mths_mortgage = [int(each) for each in train_dataset.past_5mths_mortgage]

past_5mths_pensions = []
past_5mths_pensions = [int(each) for each in train_dataset.past_5mths_pensions]

past_5mths_loans = []
past_5mths_loans = [int(each) for each in train_dataset.past_5mths_loans]

past_5mths_taxes = []
past_5mths_taxes = [int(each) for each in train_dataset.past_5mths_taxes]

past_5mths_credit_card = []
past_5mths_credit_card = [int(each) for each in train_dataset.past_5mths_credit_card]

past_5mths_securities = []
past_5mths_securities = [int(each) for each in train_dataset.past_5mths_securities]

past_5mths_home_account = []
past_5mths_home_account = [int(each) for each in train_dataset.past_5mths_home_account]

past_5mths_payroll = []
past_5mths_payroll = [int(each) for each in train_dataset.past_5mths_payroll]

past_5mths_plan_fin = []
past_5mths_plan_fin = [int(each) for each in train_dataset.past_5mths_plan_fin]

past_5mths_direct_debit = []
past_5mths_direct_debit = [int(each) for each in train_dataset.past_5mths_direct_debit]

############################# End of past 5 months predictors #############################

############################# Y creation ###################################################

# Create seperate lists for each account type
savings_account = []
savings_account = [int(each) for each in train_dataset.savings_account]

particular_plus_account = []
particular_plus_account = [int(each) for each in train_dataset.particular_plus_account]

guarantees = []
guarantees = [int(each) for each in train_dataset.guarantees]

current_account = []
current_account = [int(each) for each in train_dataset.current_account]

derivada_account = []
derivada_account = [int(each) for each in train_dataset.derivada_account]

payroll_account = []
payroll_account = [int(each) for each in train_dataset.payroll_account]

junior_account = []
junior_account = [int(each) for each in train_dataset.junior_account]

mas_particular_account = []
mas_particular_account = [int(each) for each in train_dataset.mas_particular_account]

particular_account = []
particular_account = [int(each) for each in train_dataset.particular_account]

short_term_deposits = []
short_term_deposits = [int(each) for each in train_dataset.short_term_deposits]

medium_term_deposits = []
medium_term_deposits = [int(each) for each in train_dataset.medium_term_deposits]

long_term_deposits = []
long_term_deposits = [int(each) for each in train_dataset.long_term_deposits]

e_account = []
e_account = [int(each) for each in train_dataset.e_account]

funds = []
funds = [int(each) for each in train_dataset.funds]

mortgage = []
mortgage = [int(each) for each in train_dataset.mortgage]

pensions = []
pensions = [int(each) for each in train_dataset.pensions]

loans = []
loans = [int(each) for each in train_dataset.loans]

taxes = []
taxes = [int(each) for each in train_dataset.taxes]

credit_card = []
credit_card = [int(each) for each in train_dataset.credit_card]

securities = []
securities = [int(each) for each in train_dataset.securities]

home_account = []
home_account = [int(each) for each in train_dataset.home_account]

payroll = []
payroll = [int(each) for each in train_dataset.payroll]

plan_fin = []
plan_fin = [int(each) for each in train_dataset.plan_fin]

direct_debit = []
direct_debit = [int(each) for each in train_dataset.direct_debit]

############### Y addition complete #########################


############# Base table creation #######################

# Non_Min_Max dataset - Combine all those columns which are either binary or in a standardised form as they do not need minmax transformation
non_min_max = np.array(list(zip(customer_status,gender,foreigner_flg,spouse_flg,new_customer_index,deceased_index,residence_flg,country,segmentation_01, segmentation_02, segmentation_03, segmentation_other,cust_reltn_type_beg_of_month_A, cust_reltn_type_beg_of_month_P, cust_reltn_type_beg_of_month_R, cust_reltn_type_beg_of_month_I, cust_reltn_type_beg_of_month_other,employment_status_A, employment_status_B, employment_status_F, employment_status_N, employment_status_P, employment_status_S, employment_status_other,cust_type_beg_of_month_1, cust_type_beg_of_month_2, cust_type_beg_of_month_3, cust_type_beg_of_month_4, cust_type_beg_of_month_other, past_5mths_savings_account, past_5mths_particular_plus_account, past_5mths_guarantees, past_5mths_current_account, past_5mths_derivada_account, past_5mths_payroll_account, past_5mths_junior_account, past_5mths_mas_particular_account, past_5mths_particular_account, past_5mths_short_term_deposits, past_5mths_medium_term_deposits, past_5mths_long_term_deposits, past_5mths_e_account, past_5mths_funds, past_5mths_mortgage, past_5mths_pensions, past_5mths_loans, past_5mths_taxes, past_5mths_credit_card, past_5mths_securities, past_5mths_home_account, past_5mths_payroll, past_5mths_plan_fin, past_5mths_direct_debit)))

# Min_Max dataset  - Combine all those columns which are continuous
min_max = np.array(list(zip(age_no_outliers, gross_income, seniority_no_outliers)))

# Train - Scaling down all the features to a same range
min_max_train=MinMaxScaler()
min_max_train.fit(min_max)
minmax_scaled=min_max_train.transform(min_max)

# Final dataset - Append both numpy arrays to form one numpy array
full_data = ([])
full_data = np.append(minmax_scaled, non_min_max, axis =1)

list_of_accounts = ['savings_account', 'guarantees', 'current_account', 'derivada_account', 'payroll_account', 'junior_account', 'mas_particular_account', 'particular_account', 'particular_plus_account', 'short_term_deposits', 'medium_term_deposits', 'long_term_deposits', 'e_account', 'funds', 'mortgage', 'plan_fin', 'loans', 'taxes', 'credit_card', 'securities', 'home_account', 'payroll', 'pensions', 'direct_debit']
scores = [[0 for x in range(2)] for y in range(24)]
i = -1

for account_nm in list_of_accounts:
    i += 1
    account_data = ([])
    account_data = np.append(full_data, np.array(list(zip(globals()[account_nm]))), axis =1)
    pd_account_data = pd.DataFrame(account_data, columns=["age", "gross_income", "seniority", "customer_status", "gender", "foreigner_flg", "spouse_flg", "new_customer_index", "deceased_index", "residence_flg", "country", "segmentation_01", "segmentation_02", "segmentation_03", "segmentation_other", "cust_reltn_type_beg_of_month_A", "cust_reltn_type_beg_of_month_P", "cust_reltn_type_beg_of_month_R", "cust_reltn_type_beg_of_month_I", "cust_reltn_type_beg_of_month_other", "employment_status_A", "employment_status_B", "employment_status_F", "employment_status_N", "employment_status_P", "employment_status_S", "employment_status_other", "cust_type_beg_of_month_1", "cust_type_beg_of_month_2", "cust_type_beg_of_month_3", "cust_type_beg_of_month_4", "cust_type_beg_of_month_other", "past_5mths_savings_account", "past_5mths_particular_plus_account", "past_5mths_guarantees", "past_5mths_current_account", "past_5mths_derivada_account", "past_5mths_payroll_account", "past_5mths_junior_account", "past_5mths_mas_particular_account", "past_5mths_particular_account", "past_5mths_short_term_deposits", "past_5mths_medium_term_deposits", "past_5mths_long_term_deposits", "past_5mths_e_account", "past_5mths_funds", "past_5mths_mortgage", "past_5mths_pensions", "past_5mths_loans", "past_5mths_taxes", "past_5mths_credit_card", "past_5mths_securities", "past_5mths_home_account", "past_5mths_payroll", "past_5mths_plan_fin", "past_5mths_direct_debit", account_nm])
    
    data_final_vars=pd_account_data.columns.values.tolist()
    y_cols=[account_nm]
    X_cols=[i for i in data_final_vars if i not in y_cols]
    
    X = pd_account_data[X_cols]
    y_data = pd_account_data[y_cols]
    y_ravel = y_data.values.ravel()
    Y = np.array(y_ravel).astype(int)
 
    # split data into train and test sets
    seed = 7
    test_size = 0.30
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    
    model_var = 'xgb_' + account_nm
    globals()[model_var] = XGBClassifier(nthread=4, objective="multi:softprob", eta=0.08, max_depth=7, eval_metric="mlogloss", min_child_weight=1, seed=25, num_rounds=100, gamma=0.15, num_class=22, silent=1, subsample=0.9, colsample_bytree=0.9)
    globals()[model_var].fit(X_train, y_train)    
    y_pred = globals()[model_var].predict(X_test)
       
    # make predictions for test data
    y_pred = globals()[model_var].predict(X_test)    
    
    score = accuracy_score(y_test, y_pred)
    scores[i][0] = account_nm
    scores[i][1] = score
    
    
    pickle.dump(globals()[model_var], open('/home/ubuntu/recommendations/models/' + model_var + '.p', 'wb'))
        
    X_train = ''
    X_test = ''
    y_train =''
    y_test = ''
    account_data = ''
    pd_account_data = ''
    X = ''
    Y = ''
    y_data = ''
    y_ravel = ''
    globals()[model_var] = ''
    gc.collect()

print('Models completed!!!')

#################################### Test data #################################

#### Data Load #####
test_dataset = pd.read_csv('/home/ubuntu/recommendations/data/tst_rollup_dataset.csv', sep=',')
test_dataset = test_dataset.rename(columns={'e-account': 'e_account'})
test_dataset = test_dataset.rename(columns={'past_5mths_e-account': 'past_5mths_e_account'})
test_dataset = test_dataset.fillna(0)

######## Data extraction and data preparation ########

# Store all customer_codes
customer_codes_tst = []
for each in test_dataset.cust_code:
        customer_codes_tst.append(each)

# Convert customer_status into binary
customer_status_tst = []
for each in test_dataset.customer_status:
    if each == 1:
        customer_status_tst.append(1)
    else:
        customer_status_tst.append(0)

cust_reltn_type_beg_of_month_A_tst = []
for each in test_dataset.cust_reltn_type_beg_of_month:
    if each == 'A':
        cust_reltn_type_beg_of_month_A_tst.append(1)
    else:
        cust_reltn_type_beg_of_month_A_tst.append(0)

cust_reltn_type_beg_of_month_P_tst = []
for each in test_dataset.cust_reltn_type_beg_of_month:
    if each == 'P':
        cust_reltn_type_beg_of_month_P_tst.append(1)
    else:
        cust_reltn_type_beg_of_month_P_tst.append(0)

cust_reltn_type_beg_of_month_R_tst = []
for each in test_dataset.cust_reltn_type_beg_of_month:
    if each == 'R':
        cust_reltn_type_beg_of_month_R_tst.append(1)
    else:
        cust_reltn_type_beg_of_month_R_tst.append(0)

cust_reltn_type_beg_of_month_I_tst = []
for each in test_dataset.cust_reltn_type_beg_of_month:
    if each == 'I':
        cust_reltn_type_beg_of_month_I_tst.append(1)
    else:
        cust_reltn_type_beg_of_month_I_tst.append(0)

cust_reltn_type_beg_of_month_other_tst = []
for each in test_dataset.cust_reltn_type_beg_of_month:
    if each != 'A' and each != 'P' and each != 'R' and each != 'I':
        cust_reltn_type_beg_of_month_other_tst.append(1)
    else:
        cust_reltn_type_beg_of_month_other_tst.append(0)

# Convert gender into binary
gender_tst = []
for each in test_dataset.gender:
    if each == 'V':
        gender_tst.append(1)
    else:
        gender_tst.append(0)

employment_status_A_tst =[]
for each in test_dataset.employment_status:
    if each == 'A':
        employment_status_A_tst.append(1)
    else:
        employment_status_A_tst.append(0)

employment_status_B_tst =[]
for each in test_dataset.employment_status:
    if each == 'B':
        employment_status_B_tst.append(1)
    else:
        employment_status_B_tst.append(0)

employment_status_F_tst =[]
for each in test_dataset.employment_status:
    if each == 'F':
        employment_status_F_tst.append(1)
    else:
        employment_status_F_tst.append(0)

employment_status_N_tst =[]
for each in test_dataset.employment_status:
    if each == 'N':
        employment_status_N_tst.append(1)
    else:
        employment_status_N_tst.append(0)

employment_status_P_tst =[]
for each in test_dataset.employment_status:
    if each == 'P':
        employment_status_P_tst.append(1)
    else:
        employment_status_P_tst.append(0)

employment_status_S_tst =[]
for each in test_dataset.employment_status:
    if each == 'S':
        employment_status_S_tst.append(1)
    else:
        employment_status_S_tst.append(0)

employment_status_other_tst =[]
for each in test_dataset.employment_status:
    if each != 'A' and each != 'B' and each != 'F' and each != 'N' and each != 'P' and each != 'S':
        employment_status_other_tst.append(1)
    else:
        employment_status_other_tst.append(0)

# Convert foreigner_flg into binary
foreigner_flg_tst = []
for each in test_dataset.foreigner_flg:
    if each == 'S':
        foreigner_flg_tst.append(1)
    else:
        foreigner_flg_tst.append(0)

# Convert spouse_flg into binary
spouse_flg_tst = []
for each in test_dataset.spouse_flg:
    if each == 'S':
        spouse_flg_tst.append(1)
    else:
        spouse_flg_tst.append(0)

# Convert residence_flg into binary
residence_flg_tst = []
for each in test_dataset.residence_flg:
    if each == 'S':
        residence_flg_tst.append(1)
    else:
        residence_flg_tst.append(0)

segmentation_01_tst =[]
for each in test_dataset.segmentation:
    if each == '01 - TOP':
        segmentation_01_tst.append(1)
    else:
        segmentation_01_tst.append(0)

segmentation_02_tst =[]
for each in test_dataset.segmentation:
    if each == '02 - PARTICULARES':
        segmentation_02_tst.append(1)
    else:
        segmentation_02_tst.append(0)

segmentation_03_tst =[]
for each in test_dataset.segmentation:
    if each == '03 - UNIVERSITARIO':
        segmentation_03_tst.append(1)
    else:
        segmentation_03_tst.append(0)

segmentation_other_tst = []
for each in test_dataset.segmentation:
    if each != '01 - TOP' and each != '02 - PARTICULARES' and each != '03 - UNIVERSITARIO':
        segmentation_other_tst.append(1)
    else:
        segmentation_other_tst.append(0)

# Convert spouse_flg into binary
deceased_index_tst = []
for each in test_dataset.deceased_index:
    if each == 'S':
        deceased_index_tst.append(1)
    else:
        deceased_index_tst.append(0)

# Standardize value for cust_type_beg_of_month
cust_type_beg_of_month_tst = []
for each in test_dataset.cust_type_beg_of_month:
    if each == 'P':
        int_each = 5
    else:
        try:
            flt_each = float(each)
            int_each = int(flt_each)
        except:
            int_each = 9
  
    if int_each == 1:
        cust_type_beg_of_month_tst.append(1)
    elif int_each == 2:
        cust_type_beg_of_month_tst.append(2)
    elif int_each == 3:
        cust_type_beg_of_month_tst.append(3)
    elif int_each == 4:
        cust_type_beg_of_month_tst.append(4)    
    elif int_each == 4:
        cust_type_beg_of_month_tst.append(5) 
    else:
        cust_type_beg_of_month_tst.append(int_each)

cust_type_beg_of_month_1_tst     = []
for each in cust_type_beg_of_month_tst:
    if each == 1:
        cust_type_beg_of_month_1_tst.append(1)
    else:
        cust_type_beg_of_month_1_tst.append(0)

cust_type_beg_of_month_2_tst     = []
for each in cust_type_beg_of_month_tst:
    if each == 2:
        cust_type_beg_of_month_2_tst.append(1)
    else:
        cust_type_beg_of_month_2_tst.append(0)

cust_type_beg_of_month_3_tst     = []
for each in cust_type_beg_of_month_tst:
    if each == 3:
        cust_type_beg_of_month_3_tst.append(1)
    else:
        cust_type_beg_of_month_3_tst.append(0)

cust_type_beg_of_month_4_tst     = []
for each in cust_type_beg_of_month_tst:
    if each == 4:
        cust_type_beg_of_month_4_tst.append(1)
    else:
        cust_type_beg_of_month_4_tst.append(0)

cust_type_beg_of_month_other_tst     = []
for each in cust_type_beg_of_month_tst:
    if each != 1 and each != 2 and each != 3 and each != 4:
        cust_type_beg_of_month_other_tst.append(1)
    else:
        cust_type_beg_of_month_other_tst.append(0)

# Convert new_customer_index into binary
new_customer_index_tst = []
for each in test_dataset.new_customer_index:
    if each == 1:
        new_customer_index_tst.append(1)
    else:
        new_customer_index_tst.append(0)

# Convert country into binary for train
country_tst = []
for each in test_dataset.country:
    if each == 'ES':
        country_tst.append(1)
    else:
        country_tst.append(0)        

# Standardize values for seniority
seniority_tst = []
for each in test_dataset.seniority:
    try:
        int_each = int(each)
        if int_each > 0:
            seniority_tst.append(int_each)
        else:
            seniority_tst.append(0)
    except:
        seniority_tst.append(0) 

## Fixing outliers
seniority_no_outliers_tst = []
for each in seniority_tst:
    if each > sen_above_mean_2sd:        
        seniority_no_outliers_tst.append(sen_above_mean_2sd)
    elif each < sen_below_mean_2sd: 
        seniority_no_outliers_tst.append(sen_below_mean_2sd)
    else:
        seniority_no_outliers_tst.append(each)         

# Convert string to integer for age
age_tst = []    
for each in test_dataset.age:
    try:
        age_tst.append(int(each))
    except:
        age_tst.append(40)        

## Fixing outliers
age_no_outliers_tst = []
for each in age_tst:
    if each > age_above_mean_3sd:        
        age_no_outliers_tst.append(age_above_mean_3sd)
    elif each < age_below_mean_3sd: 
        age_no_outliers_tst.append(age_below_mean_3sd)
    else:
        age_no_outliers_tst.append(each) 

# Standardize the column gross_income for train. In case of 'NA', replace it with the mean salary of Spain
gross_income_tst = []
for each in test_dataset.gross_income:
    try:
        flt_each = float(each)
        int_each = int(flt_each)
        gross_income_tst.append(int_each)   
    except:       
        gross_income_tst.append(15816)        

# Create seperate lists for each account type
past_5mths_savings_account_tst = []
past_5mths_savings_account_tst = [int(each) for each in test_dataset.past_5mths_savings_account]

past_5mths_particular_plus_account_tst = []
past_5mths_particular_plus_account_tst = [int(each) for each in test_dataset.past_5mths_particular_plus_account]

past_5mths_guarantees_tst = []
past_5mths_guarantees_tst = [int(each) for each in test_dataset.past_5mths_guarantees]

past_5mths_current_account_tst = []
past_5mths_current_account_tst = [int(each) for each in test_dataset.past_5mths_current_account]

past_5mths_derivada_account_tst = []
past_5mths_derivada_account_tst = [int(each) for each in test_dataset.past_5mths_derivada_account]

past_5mths_payroll_account_tst = []
past_5mths_payroll_account_tst = [int(each) for each in test_dataset.past_5mths_payroll_account]

past_5mths_junior_account_tst = []
past_5mths_junior_account_tst = [int(each) for each in test_dataset.past_5mths_junior_account]

past_5mths_mas_particular_account_tst = []
past_5mths_mas_particular_account_tst = [int(each) for each in test_dataset.past_5mths_mas_particular_account]

past_5mths_particular_account_tst = []
past_5mths_particular_account_tst = [int(each) for each in test_dataset.past_5mths_particular_account]

past_5mths_short_term_deposits_tst = []
past_5mths_short_term_deposits_tst = [int(each) for each in test_dataset.past_5mths_short_term_deposits]

past_5mths_medium_term_deposits_tst = []
past_5mths_medium_term_deposits_tst = [int(each) for each in test_dataset.past_5mths_medium_term_deposits]

past_5mths_long_term_deposits_tst = []
past_5mths_long_term_deposits_tst = [int(each) for each in test_dataset.past_5mths_long_term_deposits]

past_5mths_e_account_tst = []
past_5mths_e_account_tst = [int(each) for each in test_dataset.past_5mths_e_account]

past_5mths_funds_tst = []
past_5mths_funds_tst = [int(each) for each in test_dataset.past_5mths_funds]

past_5mths_mortgage_tst = []
past_5mths_mortgage_tst = [int(each) for each in test_dataset.past_5mths_mortgage]

past_5mths_pensions_tst = []
past_5mths_pensions_tst = [int(each) for each in test_dataset.past_5mths_pensions]

past_5mths_loans_tst = []
past_5mths_loans_tst = [int(each) for each in test_dataset.past_5mths_loans]

past_5mths_taxes_tst = []
past_5mths_taxes_tst = [int(each) for each in test_dataset.past_5mths_taxes]

past_5mths_credit_card_tst = []
past_5mths_credit_card_tst = [int(each) for each in test_dataset.past_5mths_credit_card]

past_5mths_securities_tst = []
past_5mths_securities_tst = [int(each) for each in test_dataset.past_5mths_securities]

past_5mths_home_account_tst = []
past_5mths_home_account_tst = [int(each) for each in test_dataset.past_5mths_home_account]

past_5mths_payroll_tst = []
past_5mths_payroll_tst = [int(each) for each in test_dataset.past_5mths_payroll]

past_5mths_plan_fin_tst = []
past_5mths_plan_fin_tst = [int(each) for each in test_dataset.past_5mths_plan_fin]

past_5mths_direct_debit_tst = []
past_5mths_direct_debit_tst = [int(each) for each in test_dataset.past_5mths_direct_debit]

############# Base table creation #######################

# Non_Min_Max dataset - Combine all those columns which are either binary or in a standardised form as they do not need minmax transformation
non_min_max_tst = np.array(list(zip(customer_status_tst, gender_tst, foreigner_flg_tst, spouse_flg_tst, new_customer_index_tst, deceased_index_tst, residence_flg_tst, country_tst, segmentation_01_tst,  segmentation_02_tst,  segmentation_03_tst,  segmentation_other_tst, cust_reltn_type_beg_of_month_A_tst,  cust_reltn_type_beg_of_month_P_tst,  cust_reltn_type_beg_of_month_R_tst,  cust_reltn_type_beg_of_month_I_tst,  cust_reltn_type_beg_of_month_other_tst, employment_status_A_tst,  employment_status_B_tst,  employment_status_F_tst,  employment_status_N_tst,  employment_status_P_tst,  employment_status_S_tst,  employment_status_other_tst, cust_type_beg_of_month_1_tst,  cust_type_beg_of_month_2_tst,  cust_type_beg_of_month_3_tst,  cust_type_beg_of_month_4_tst,  cust_type_beg_of_month_other_tst, past_5mths_savings_account_tst, past_5mths_particular_plus_account_tst, past_5mths_guarantees_tst, past_5mths_current_account_tst, past_5mths_derivada_account_tst, past_5mths_payroll_account_tst, past_5mths_junior_account_tst, past_5mths_mas_particular_account_tst, past_5mths_particular_account_tst, past_5mths_short_term_deposits_tst, past_5mths_medium_term_deposits_tst, past_5mths_long_term_deposits_tst, past_5mths_e_account_tst, past_5mths_funds_tst, past_5mths_mortgage_tst, past_5mths_pensions_tst, past_5mths_loans_tst, past_5mths_taxes_tst, past_5mths_credit_card_tst, past_5mths_securities_tst, past_5mths_home_account_tst, past_5mths_payroll_tst, past_5mths_plan_fin_tst, past_5mths_direct_debit_tst)))

# Min_Max dataset  - Combine all those columns which are continuous
min_max_tst = np.array(list(zip(age_no_outliers_tst, gross_income_tst, seniority_no_outliers_tst)))

# Scaling down all the features to a same range
minmax_scaled_tst=min_max_train.transform(min_max_tst)


# Final dataset - Append both numpy arrays to form one numpy array
full_data_tst = ([])
full_data_tst = np.append(minmax_scaled_tst, non_min_max_tst, axis =1)
pd_tst = pd.DataFrame(full_data_tst, columns=["age", "gross_income", "seniority", "customer_status", "gender", "foreigner_flg", "spouse_flg", "new_customer_index", "deceased_index", "residence_flg", "country", "segmentation_01", "segmentation_02", "segmentation_03", "segmentation_other", "cust_reltn_type_beg_of_month_A", "cust_reltn_type_beg_of_month_P", "cust_reltn_type_beg_of_month_R", "cust_reltn_type_beg_of_month_I", "cust_reltn_type_beg_of_month_other", "employment_status_A", "employment_status_B", "employment_status_F", "employment_status_N", "employment_status_P", "employment_status_S", "employment_status_other", "cust_type_beg_of_month_1", "cust_type_beg_of_month_2", "cust_type_beg_of_month_3", "cust_type_beg_of_month_4", "cust_type_beg_of_month_other", "past_5mths_savings_account", "past_5mths_particular_plus_account", "past_5mths_guarantees", "past_5mths_current_account", "past_5mths_derivada_account", "past_5mths_payroll_account", "past_5mths_junior_account", "past_5mths_mas_particular_account", "past_5mths_particular_account", "past_5mths_short_term_deposits", "past_5mths_medium_term_deposits", "past_5mths_long_term_deposits", "past_5mths_e_account", "past_5mths_funds", "past_5mths_mortgage", "past_5mths_pensions", "past_5mths_loans", "past_5mths_taxes", "past_5mths_credit_card", "past_5mths_securities", "past_5mths_home_account", "past_5mths_payroll", "past_5mths_plan_fin", "past_5mths_direct_debit"])


list_of_accounts = ['savings_account', 'guarantees', 'current_account', 'derivada_account', 'payroll_account', 'junior_account', 'mas_particular_account', 'particular_account', 'particular_plus_account', 'short_term_deposits', 'medium_term_deposits', 'long_term_deposits', 'e_account', 'funds', 'mortgage', 'plan_fin', 'loans', 'taxes', 'credit_card', 'securities', 'home_account', 'payroll', 'pensions', 'direct_debit']
for account_nm in list_of_accounts:
    model_nm_var = 'xgb_' + account_nm
    globals()[model_nm_var] = pickle.load(open('/home/ubuntu/recommendations/models/' + model_nm_var + '.p', 'rb'))
    tst_pred_var = 'tst_pred_' + account_nm
    globals()[tst_pred_var] = globals()[model_nm_var].predict_proba(pd_tst)
    pickle.dump(globals()[tst_pred_var], open('/home/ubuntu/recommendations/results/' + tst_pred_var + '.p', 'wb'))
    ## print(model_nm_var)
    print(tst_pred_var)    
    globals()[model_nm_var] = ''
    globals()[tst_pred_var] = ''

########## Part of part 2 #######################

np_customer_codes_tst = np.array(customer_codes_tst)

pd_final_result = pd.DataFrame({'cust_code': np_customer_codes_tst, 'savings_account': tst_pred_savings_account[:,1], 'particular_plus_account': tst_pred_particular_plus_account[:,1], 'guarantees': tst_pred_guarantees[:,1], 'current_account': tst_pred_current_account[:,1], 'derivada_account': tst_pred_derivada_account[:,1], 'payroll_account': tst_pred_payroll_account[:,1], 'junior_account': tst_pred_junior_account[:,1], 'mas_particular_account': tst_pred_mas_particular_account[:,1], 'particular_account': tst_pred_particular_account[:,1], 'short_term_deposits': tst_pred_short_term_deposits[:,1], 'medium_term_deposits': tst_pred_medium_term_deposits[:,1], 'long_term_deposits': tst_pred_long_term_deposits[:,1], 'e_account': tst_pred_e_account[:,1], 'funds': tst_pred_funds[:,1], 'mortgage': tst_pred_mortgage[:,1], 'pensions': tst_pred_pensions[:,1], 'loans': tst_pred_loans[:,1], 'taxes': tst_pred_taxes[:,1], 'credit_card': tst_pred_credit_card[:,1], 'securities': tst_pred_securities[:,1], 'home_account': tst_pred_home_account[:,1], 'payroll': tst_pred_payroll[:,1], 'plan_fin': tst_pred_plan_fin[:,1], 'direct_debit': tst_pred_direct_debit[:,1] })
pickle.dump(pd_final_result, open('/home/ubuntu/recommendations/pd_final_result.p', 'wb'))

results={}
for idx, row in pd_final_result.iterrows():
    cust_code = row['cust_code']
    results[cust_code] = {}
    
    credit_card =row['credit_card']
    results[cust_code]['credit_card'] = credit_card
    
    current_account = row['current_account']
    results[cust_code]['current_account'] = current_account
    
    derivada_account= row['derivada_account']
    results[cust_code]['derivada_account'] = derivada_account
    
    direct_debit =row['direct_debit']
    results[cust_code]['direct_debit'] = direct_debit
    
    e_account =row['e_account']
    results[cust_code]['e_account'] = e_account
    
    funds= row['funds']
    results[cust_code]['funds'] = funds
    
    guarantees= row['guarantees']
    results[cust_code]['guarantees'] = guarantees
    
    home_account= row['home_account']
    results[cust_code]['home_account'] = home_account
    
    junior_account = row['junior_account']
    results[cust_code]['junior_account'] = junior_account
    
    loans = row['loans']
    results[cust_code]['loans'] = loans
    
    long_term_deposits= row['long_term_deposits']
    results[cust_code]['long_term_deposits'] = long_term_deposits
    
    mas_particular_account= row['mas_particular_account']
    results[cust_code]['mas_particular_account'] = mas_particular_account
    
    medium_term_deposits= row['medium_term_deposits']
    results[cust_code]['medium_term_deposits'] = medium_term_deposits
    
    mortgage = row['mortgage']
    results[cust_code]['mortgage'] = mortgage
    
    particular_account = row['particular_account']
    results[cust_code]['particular_account'] = particular_account
    
    particular_plus_account =row['particular_plus_account']
    results[cust_code]['particular_plus_account'] = particular_plus_account
    
    payroll =row['payroll']
    results[cust_code]['payroll'] = payroll
    
    payroll_account =row['payroll_account']
    results[cust_code]['payroll_account'] = payroll_account
    
    pensions =row['pensions']
    results[cust_code]['pensions'] = pensions
    
    plan_fin =row['plan_fin']
    results[cust_code]['plan_fin'] = plan_fin
    
    savings_account = row['savings_account']
    results[cust_code]['savings_account'] = savings_account
    
    securities =row['securities']
    results[cust_code]['securities'] = securities
    
    short_term_deposits =row['short_term_deposits']
    results[cust_code]['short_term_deposits'] = short_term_deposits
    
    taxes=row['taxes']
    results[cust_code]['taxes'] = taxes

#pickle.dump(results, open('/home/ubuntu/recommendations/results.p', 'wb'))
## results = pickle.load(open( "/home/ubuntu/recommendations/results.p", "rb" ) )

sorted_results = {}
from operator import itemgetter
for each in results:
    temp = {}
    temp = results[each]
    sorted_results[each] = []
    sorted_results[each] = sorted(temp, key=temp.get, reverse=True)


pickle.dump(sorted_results, open('/home/ubuntu/recommendations/sorted_results.p', 'wb'))
## sorted_results = pickle.load(open( "/home/ubuntu/recommendations/sorted_results.p", "rb" ) )

translator= {'savings_account': 'ind_ahor_fin_ultl',
'guarantees': 'ind_aval_fin_ult1',
'current_account': 'ind_cco_fin_ult1',
'derivada_account': 'ind_cder_fin_ult1',
'payroll_account': 'ind_cno_fin_ult1',
'junior_account': 'ind_ctju_fin_ult1',
'mas_particular_account': 'ind_ctma_fin_ult1',
'particular_account': 'ind_ctop_fin_ult1',
'particular_plus_account': 'ind_ctpp_fin_ult1',
'short_term_deposits': 'ind_deco_fin_ult1',
'medium_term_deposits': 'ind_deme_fin_ult1',
'long_term_deposits': 'ind_dela_fin_ult1',
'e_account': 'ind_ecue_fin_ult1',
'funds': 'ind_fond_fin_ult1',
'mortgage': 'ind_hip_fin_ult1',
'plan_fin': 'ind_plan_fin_ult1',
'loans': 'ind_pres_fin_ult1',
'taxes': 'ind_reca_fin_ult1',
'credit_card': 'ind_tjcr_fin_ult1',
'securities': 'ind_valo_fin_ult1',
'home_account': 'ind_viv_fin_ult1',
'payroll': 'ind_nomina_ult1',
'pensions': 'ind_nom_pens_ult1',
'direct_debit': 'ind_recibo_ult1'}


######################## Excluding customer's last month products ########################

customer_latest_products = pickle.load(open( "/home/ubuntu/recommendations/customer_latest_products.p", "rb" ) )

######################## Excluding unwanted products ########################

products_to_exclude = ['guarantees', 'short_term_deposits', 'long_term_deposits', 'loans']

####################### Recommendations ######################################

# Recommend a new product per customer. 
recommendations = {}
for each in sorted_results:
    cust_code = each
    products = sorted_results[cust_code]
    last_used_prdts = []
    last_used_prdts = customer_latest_products.get(cust_code)
    recommendations[cust_code] = []
    i = 0
    for prd in products:
        if i < 7:
            if prd not in last_used_prdts and (len(last_used_prdts) > 12 or (prd not in products_to_exclude)):
                recommendations[cust_code].append(prd)
                i += 1
    listrec = recommendations[cust_code]
    if len(listrec) < 7:
        print(cust_code)


f = open('/home/ubuntu/recommendations/brand_new_recommendations.csv', 'w')
headers = 'ncodpers,added_products'
f.write(headers)
f.write('\n')
for each in recommendations:
    product0 = recommendations[each][0]
    spanish_nm0 = translator[product0]
    product1 = recommendations[each][1]
    spanish_nm1 = translator[product1]
    product2 = recommendations[each][2]
    spanish_nm2 = translator[product2]
    product3 = recommendations[each][3]
    spanish_nm3 = translator[product3]
    product4 = recommendations[each][4]
    spanish_nm4 = translator[product4]
    product5 = recommendations[each][5]
    spanish_nm5 = translator[product5]
    product6 = recommendations[each][6]
    spanish_nm6 = translator[product6]    
    
    str_result = str(int(each)) + "," + spanish_nm0 + " " + spanish_nm1 + " " + spanish_nm2 + " " + spanish_nm3 + " " + spanish_nm4 + " " + spanish_nm5 + " " + spanish_nm6
    f.write(str_result)
    f.write('\n')    
f.close()

