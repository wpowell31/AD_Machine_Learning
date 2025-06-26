import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
warnings.filterwarnings('ignore')
print("Imports complete")

path = '/storage1/fs1/aditigupta/Active/Will/20241128_MDC_datapull'
diagnoses = pd.read_csv(path + '/diagnoses.csv')
demographics = pd.read_csv(path + '/demographics.csv')
demographics = demographics.rename(columns={'EPIC_mrn': "EPIC_MRN"})
med_orders = pd.read_csv(path + '/med_orders.csv')
encounters = pd.read_csv(path + '/encounters.csv')

###############################################################################################################################
################################## First AD CDR ########################################################
###############################################################################################################################

# Function to process the data for a given year
def process_sb_data(year, path):
    # Load the data
    flowsheets = pd.read_csv(f"{path}/flowsheets_{year}.csv")

    # Filter based on FLOWSHEET_NAME
    cdr_df = flowsheets[flowsheets['FLOWSHEET_NAME'] == 'MDC CDR/Diagnosis New']

    # Further filter for MEASURE_NAME 'Global Score'
    global_score_df = cdr_df[cdr_df['MEASURE_NAME'] == 'Global Score']
    
    global_score_df['PERFORMED'] = pd.to_datetime(global_score_df['PERFORMED'], errors='coerce')

    # Convert the 'VALUE' column to numeric
    global_score_df['VALUE'] = pd.to_numeric(global_score_df['VALUE'], errors='coerce')

    # Filter rows where 'VALUE' is greater than/equal to 1
    global_score_df['VALUE_GS_1'] = (global_score_df['VALUE'] >= 1).astype(int)
    
    global_score_df = global_score_df[['EPIC_MRN', 'PERFORMED', 'VALUE', 'VALUE_GS_1']]

    return global_score_df


# Define the years
years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

# Create a list to hold the filtered data for different years
filtered_data_list = []

# Process data for each year and store the result
for year in years:
    filtered_data = process_sb_data(year, path)
    filtered_data_list.append(filtered_data)

# Combine all filtered DataFrames into a single DataFrame
cdr_data = pd.concat(filtered_data_list, ignore_index=True) 
cdr_data['PERFORMED'] = pd.to_datetime(cdr_data['PERFORMED'])

cdr_data = cdr_data[cdr_data['VALUE'].isin([0.0, 0.5, 1.0, 2.0, 3.0])]
# Filter the DataFrame where VALUE >= 1.0
filtered_data = cdr_data[cdr_data['VALUE'] >= 1.0]

# Sort the filtered data by EPIC_MRN and PERFORMED date
sorted_data = filtered_data.sort_values(by=['EPIC_MRN', 'PERFORMED'])

# Get the first PERFORMED date for each EPIC_MRN
first_ad_cdr = sorted_data.groupby('EPIC_MRN').first().reset_index()
first_ad_cdr = first_ad_cdr.rename(columns={'PERFORMED': 'FIRST_AD_CDR_DATE'})
print("First AD from CDR complete")

###############################################################################################################################
################################## First AD from Flowsheets ########################################################
###############################################################################################################################
# Function to process the data for a given year
def process_flowsheet_data(year, path):
    # Load the data
    flowsheets = pd.read_csv(f"{path}/flowsheets_{year}.csv")

    # Filter based on FLOWSHEET_NAME
    ad_flowsheet_df = flowsheets[flowsheets['FLOWSHEET_NAME'] == 'MDC CDR/Diagnosis New'] # Look for non alzheimers dementias

    # Further filter for MEASURE_NAME 'Global Score'
    ad_flowsheet_df = ad_flowsheet_df[ad_flowsheet_df['MEASURE_NAME'] == 'AD Dementia']

    return ad_flowsheet_df[['EPIC_MRN', 'PERFORMED', 'VALUE']]

# Define the years you want to process
years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

# Create a list to hold the filtered data for different years
ad_data_list = []

# Process data for each year and store the result
for year in years:
    ad_data = process_flowsheet_data(year, path)
    # Add a column for the year to keep track of the source
    ad_data['YEAR'] = year
    ad_data_list.append(ad_data)

# Combine all filtered DataFrames into a single DataFrame
flowsheets_ad_data = pd.concat(ad_data_list, ignore_index=True)
flowsheets_ad_data['PERFORMED'] = pd.to_datetime(flowsheets_ad_data['PERFORMED'])
print(flowsheets_ad_data['EPIC_MRN'].nunique())

# Sort the filtered data by EPIC_MRN and PERFORMED date
sorted_data = flowsheets_ad_data.sort_values(by=['EPIC_MRN', 'PERFORMED'])

# Get the first PERFORMED date for each EPIC_MRN
first_ad_flowsheet = sorted_data.groupby('EPIC_MRN').first().reset_index()
first_ad_flowsheet = first_ad_flowsheet.rename(columns={'PERFORMED': 'FIRST_AD_FLOWSHEET_DATE'})
print("First AD from flowsheet complete")


###############################################################################################################################
################################## First AD from ICD Diagnoses ########################################################
###############################################################################################################################
# Dictionary of conditions and their corresponding ICD codes
conditions = {
    'AD_diagnosis': ['G30', 'G30.0', 'G30.1', 'G30.8', 'G30.9']  # Added Alzheimer's Disease
}

# Filter diagnoses to only include rows with AD diagnosis codes
ad_diagnoses = diagnoses[diagnoses['ICD_CODES'].isin(conditions['AD_diagnosis'])]

# Convert RECORDED_DATE column to datetime format
ad_diagnoses['RECORDED_DATE'] = pd.to_datetime(ad_diagnoses['RECORDED_DATE'])

# Sort AD diagnoses by MRN and RECORDED_DATE
ad_diagnoses_sorted = ad_diagnoses.sort_values(by=['EPIC_MRN', 'RECORDED_DATE'])

# Drop duplicates to keep the first occurrence of AD diagnosis for each MRN
first_ad_diagnosis = ad_diagnoses_sorted.drop_duplicates(subset=['EPIC_MRN'], keep='first')
first_ad_diagnosis = first_ad_diagnosis[['EPIC_MRN', 'RECORDED_DATE', 'ICD_CODES', 'DIAGNOSIS']]
first_ad_diagnosis = first_ad_diagnosis.rename(columns={'RECORDED_DATE': 'FIRST_AD_DIAGNOSIS_DATE'})

print("First AD from diag complete")
###############################################################################################################################
################################## First AD from Medications ########################################################
###############################################################################################################################


###############################################################################################################################
################################## see if anyone lecanemab before anything else ##################################################
###############################################################################################################################


# Step 2: String search for lecanemab and donanemab
lecanemab_med_orders = med_orders[med_orders['MED_NAME'].str.contains('lecanemab', case=False)]
donanemab_med_orders = med_orders[med_orders['MED_NAME'].str.contains('donanemab', case=False)]

# Combine all filtered records
combined_med_orders = pd.concat([lecanemab_med_orders, donanemab_med_orders]).drop_duplicates()

# Step 4: Group by EPIC_MRN and find the earliest MED_ORDER_DATE
first_ad_med_orders = combined_med_orders.groupby('EPIC_MRN').agg(FIRST_AD_MED_DATE=('MED_ORDER_DATE', 'min')).reset_index()

print("First AD from meds complete")

###############################################################################################################################
################################## Combine for overall first AD ########################################################
###############################################################################################################################

# Convert date columns to datetime
first_ad_cdr['FIRST_AD_CDR_DATE'] = pd.to_datetime(first_ad_cdr['FIRST_AD_CDR_DATE']) # get rid of this for definition
first_ad_flowsheet['FIRST_AD_FLOWSHEET_DATE'] = pd.to_datetime(first_ad_flowsheet['FIRST_AD_FLOWSHEET_DATE'])
first_ad_diagnosis['FIRST_AD_DIAGNOSIS_DATE'] = pd.to_datetime(first_ad_diagnosis['FIRST_AD_DIAGNOSIS_DATE']) # don't do AD ICD diagnosis alone potentially?
first_ad_med_orders['FIRST_AD_MED_DATE'] = pd.to_datetime(first_ad_med_orders['FIRST_AD_MED_DATE'])


# Extract unique MRN values
unique_mrns = set(first_ad_cdr['EPIC_MRN']).union(set(first_ad_flowsheet['EPIC_MRN']), set(first_ad_diagnosis['EPIC_MRN']), set(first_ad_med_orders['EPIC_MRN']))

# Create a new dataframe with unique MRNs
result_df = pd.DataFrame(unique_mrns, columns=['EPIC_MRN'])

# Merge date columns into the result dataframe
result_df = result_df.merge(first_ad_cdr[['EPIC_MRN', 'FIRST_AD_CDR_DATE']], on='EPIC_MRN', how='left')
result_df = result_df.merge(first_ad_flowsheet[['EPIC_MRN', 'FIRST_AD_FLOWSHEET_DATE']], on='EPIC_MRN', how='left')
result_df = result_df.merge(first_ad_diagnosis[['EPIC_MRN', 'FIRST_AD_DIAGNOSIS_DATE']], on='EPIC_MRN', how='left')
result_df = result_df.merge(first_ad_med_orders[['EPIC_MRN', 'FIRST_AD_MED_DATE']], on='EPIC_MRN', how='left')



# Calculate the earliest overall date for each MRN
result_df['EARLIEST_OVERALL_DATE'] = result_df[['FIRST_AD_CDR_DATE', 'FIRST_AD_FLOWSHEET_DATE', 'FIRST_AD_DIAGNOSIS_DATE', 'FIRST_AD_MED_DATE']].min(axis=1) # take out CDR from this

# Keeping information on all 4 possible types
# However, if only FIRST_AD_CDR_DATE is available, do not count that patient as having AD and set EARLIEST_OVERALL_DATE to NaN
result_df.loc[result_df[['FIRST_AD_FLOWSHEET_DATE', 'FIRST_AD_DIAGNOSIS_DATE', 'FIRST_AD_MED_DATE']].isna().all(axis=1), 'EARLIEST_OVERALL_DATE'] = pd.NaT


###############################################################################################################################
################################## Get controls, encounter window ########################################################
###############################################################################################################################
# Convert ENC_ADMIT_DTTM to datetime
encounters['ENC_ADMIT_DTTM'] = pd.to_datetime(encounters['ENC_ADMIT_DTTM'])

# Group by EPIC_MRN and calculate first, last encounter timestamps, and time differences
result = encounters.groupby('EPIC_MRN').agg(
    FIRST_ENC=('ENC_ADMIT_DTTM', 'min'),
    LAST_ENC=('ENC_ADMIT_DTTM', 'max')
).reset_index()

# Calculate the difference in years between first and last encounter
result['DIFF_YEARS'] = result['LAST_ENC'].dt.year - result['FIRST_ENC'].dt.year

# Calculate INDEX_TIME, which is 1 year before LAST_ENC
result['INDEX_TIME_ENC'] = result['LAST_ENC'] - pd.DateOffset(years=1)

result = result[result['DIFF_YEARS'] >= 3]
print(f"encounters greater than 3 year: {result.shape}")

ad_dates = result_df

merged = result.merge(ad_dates, on='EPIC_MRN', how='left')

merged_df = merged

# Create INDEX_DATE as EARLIEST_OVERALL_DATE if it's not null, otherwise use INDEX_TIME
merged_df['INDEX_DATE'] = np.where(merged_df['EARLIEST_OVERALL_DATE'].notna(), 
                                    merged_df['EARLIEST_OVERALL_DATE'], 
                                    merged_df['INDEX_TIME_ENC'])

# Create label column: 1 if EARLIEST_OVERALL_DATE is not na, else 0
merged_df['AD_label'] = np.where(merged_df['EARLIEST_OVERALL_DATE'].notna(), 1, 0)

#merged_df['MDC_days'] = merged_df['INDEX_DATE'].dt.day - merged_df['FIRST_ENC'].dt.day

print("controlscomplete")

###############################################################################################################################
################################## Merge demographics and conditions ########################################################
###############################################################################################################################

demographics_clean = demographics.drop_duplicates(subset='EPIC_MRN', keep='first')

# Apply the race grouping
def categorize_race(race):
    if race == 'White':
        return 'White'
    elif race == 'Black or African American':
        return 'Black or African American'
    elif race == 'Asian':
        return 'Asian'
    else:
        return 'Unknown/Other'

demographics_clean['PT_RACE'] = demographics_clean['PT_RACE'].apply(categorize_race)

ad_demo_data = merged_df.merge(demographics_clean, on='EPIC_MRN', how='left')

# Dictionary of conditions and their corresponding ICD codes
conditions = {
    'Hypertension': ['I10', 'I15', 'I15.0', 'I15.1', 'I15.2', 'I15.8', 'I15.9'],
    'Hyperlipidemia': ['E78.5', 'E78.4', 'E78.4.x', 'E78.2', 'E78.2.x'],
    'Diabetes': ['E08.x', 'E09.x', 'E10.x', 'E11.x', 'E13.x'],
    'Cerebrovascular disease': ['G45.x', 'G46.x', 'H34.0', 'I60.x', 'I61.x', 'I62.x',
                                'I63.x', 'I64.x', 'I65.x', 'I66.x', 'I67.x', 'I68.x', 'I69.x'],
    'Myocardial infarction': ['I21.x', 'I22.x', 'I25.2'],
    'Chronic Kidney Disease': ['N18', 'N18.1', 'N18.2', 'N18.3', 'N18.30', 'N18.31', 'N18.32', 'N18.4', 'N18.5', 'N18.6', 'N18.9'],
    'Liver cirrhosis': ['K70.3', 'K74.6', 'K74.60', 'K74.69', 'K74.0', 'K74', 
                        'K74.2', 'K74.3', 'K74.4', 'K74.5', 'K71.7', 'K70.2', 'K74.72'],
     'Central Nervous System (CNS) conditions': ['G00.x', 'G01.x', 'G02.x', 'G03.x', 'G04.x', 'G05.x', 'G06.x', 'G07.x', 'G08.x', 'G09.x',
                                                'G10.x', 'G11.x', 'G12.x', 'G13.x', 'G14.x', 'G15.x', 'G16.x', 'G17.x', 'G18.x', 'G19.x',
                                                'G20.x', 'G21.x', 'G22.x', 'G23.x', 'G24.x', 'G25.x', 'G26.x', 'G27.x', 'G28.x', 'G29.x',
                                                'G30.x', 'G31.x', 'G32.x', 'G33.x', 'G34.x', 'G35.x', 'G36.x', 'G37.x', 'G38.x', 'G39.x',
                                                'G40.x', 'G41.x', 'G42.x', 'G43.x', 'G44.x', 'G45.x', 'G46.x', 'G47.x', 'G48.x', 'G49.x',
                                                'G50.x', 'G51.x', 'G52.x', 'G53.x', 'G54.x', 'G55.x', 'G56.x', 'G57.x', 'G58.x', 'G59.x',
                                                'G60.x', 'G61.x', 'G62.x', 'G63.x', 'G64.x', 'G65.x', 'G66.x', 'G67.x', 'G68.x', 'G69.x',
                                                'G70.x', 'G71.x', 'G72.x', 'G73.x', 'G74.x', 'G75.x', 'G76.x', 'G77.x', 'G78.x', 'G79.x',
                                                'G80.x', 'G81.x', 'G82.x', 'G83.x', 'G84.x', 'G85.x', 'G86.x', 'G87.x', 'G88.x', 'G89.x',
                                                'G90.x', 'G91.x', 'G92.x', 'G93.x', 'G94.x', 'G95.x', 'G96.x', 'G97.x', 'G98.x', 'G99.x'],
    'Polyneuropathy': ['G60.0', 'G60.1', 'G60.2', 'G60.3', 'G60.8', 'G60.9', 
                       'G61.0', 'G61.1', 'G61.8', 'G61.9', 'G62.0', 'G62.1', 
                       'G62.2', 'G62.8', 'G62.9']
}

# potentially - get rid of conditions and just do phecodes alone
# Dictionary to store results
results = {}

# Ensure there are no NaNs in the relevant columns
diagnoses['ICD_CODES'] = diagnoses['ICD_CODES'].fillna('')
diagnoses['DIAGNOSIS'] = diagnoses['DIAGNOSIS'].fillna('')

# filter for diagnosis only before CDR Final score date
diagnoses_filtered = pd.merge(diagnoses, ad_demo_data, on='EPIC_MRN', how='inner')
diagnoses_filtered = diagnoses_filtered[diagnoses_filtered['RECORDED_DATE'] < diagnoses_filtered['INDEX_DATE']]

# Function to find matching EPIC_MRNs, unique ICD_CODES, and DIAGNOSIS strings
def find_matches(condition_name, icd_codes_list):
    matched_mrns = set()
    matched_icd_codes = set()
    matched_diagnoses = set()
    for code in icd_codes_list:
        if code.endswith('.x'):
            base_code = code[:-2]
            matches = diagnoses_filtered[diagnoses_filtered['ICD_CODES'].str.startswith(base_code, na=False)]
        else:
            matches = diagnoses_filtered[diagnoses_filtered['ICD_CODES'] == code]
        matched_mrns.update(matches['EPIC_MRN'])
        matched_icd_codes.update(matches['ICD_CODES'])
        matched_diagnoses.update(matches['DIAGNOSIS'])
    return list(matched_mrns), list(matched_icd_codes), list(matched_diagnoses)

# Loop through each condition and its ICD codes
for condition, icd_codes in conditions.items():
    epic_mrns, icd_codes, diagnoses_str = find_matches(condition, icd_codes)
    results[condition] = {
        'EPIC_MRN': epic_mrns,
        'ICD_CODES': icd_codes,
        'DIAGNOSIS': diagnoses_str
    }

# Add columns for each condition to demographics_clean
for condition in conditions.keys():
    ad_demo_data[condition] = ad_demo_data['EPIC_MRN'].isin(results[condition]['EPIC_MRN']).astype(int)


ad_demo_data['PT_RACE'] = ad_demo_data['PT_RACE'].fillna('unknown')
ad_demo_data['PT_BIRTH_DTTM'] = pd.to_datetime(ad_demo_data['PT_BIRTH_DTTM'])
ad_demo_data['age_at_index'] = (pd.to_datetime(ad_demo_data['INDEX_DATE'] )- pd.to_datetime(ad_demo_data['PT_BIRTH_DTTM'])).dt.days // 365
ad_demo_data['PT_GENDER'] = ad_demo_data['PT_GENDER'].map({'Male': 1, 'Female': 0})
ad_demo_data = pd.get_dummies(ad_demo_data, columns = ['PT_RACE'])


# Ensure your datetime columns are in datetime format
ad_demo_data['FIRST_ENC'] = pd.to_datetime(ad_demo_data['FIRST_ENC'], format='%Y-%m-%d %H:%M:%S', errors='coerce').fillna(pd.to_datetime(ad_demo_data['FIRST_ENC'], format='%Y-%m-%d', errors='coerce'))
ad_demo_data['INDEX_DATE'] = pd.to_datetime(ad_demo_data['INDEX_DATE'], format='%Y-%m-%d %H:%M:%S', errors='coerce').fillna(pd.to_datetime(ad_demo_data['INDEX_DATE'], format='%Y-%m-%d', errors='coerce'))

# Now check if there are any remaining parsing issues
assert ad_demo_data['FIRST_ENC'].notnull().all(), "There are some unparsable dates in FIRST_ENC"
assert ad_demo_data['INDEX_DATE'].notnull().all(), "There are some unparsable dates in INDEX_DATE"

# Calculate the time difference in years
ad_demo_data['MDC_years'] = (ad_demo_data['INDEX_DATE'] - ad_demo_data['FIRST_ENC']).dt.days / 365

# Filter rows where the difference is at least 3 years
#filtered_data = ad_demo_data[ad_demo_data['diff_years'] >= 3] unnecessary

# Drop the 'diff_years' column if you don't need it
#filtered_data = filtered_data.drop(columns=['diff_years'])
#ad_demo_data = filtered_data

print("merged with demographics and conditions")

# save checkpoint
ad_demo_data.to_csv('/storage1/fs1/aditigupta/Active/Will/ML_LLM_AD/AD_prediction/checkpoint.csv', index=False)
diagnoses_filtered.to_csv('/storage1/fs1/aditigupta/Active/Will/ML_LLM_AD/AD_prediction/diagnoses_filtered.csv')


###############################################################################################################################
################################## Phecodes ###################################################################################
###############################################################################################################################
phecode_map = pd.read_csv('/storage1/fs1/aditigupta/Active/Will/Phecode_map_v1_2_icd9_icd10cm_09_30_2024.csv')
phecode_map = phecode_map.rename(columns={'ICD': 'ICD_CODES'})

# Mergin the diagnoses with the phecodes
diag_phecd = pd.merge(diagnoses_filtered, phecode_map, how="left", on="ICD_CODES").dropna(subset=['Phecode'])

# filtering to make sure all diagnoses epic_mrns only those included in the study
diag_phecd = diag_phecd[diag_phecd['EPIC_MRN'].isin(ad_demo_data['EPIC_MRN'])]

# Merge the dataframes on 'EPIC_MRN'
merged_df = pd.merge(ad_demo_data, diag_phecd, on='EPIC_MRN', how='inner') # is this the issue? should this just be a left merge? - no, inner fine here

# Pivot the filtered dataframe
pivoted_df = merged_df.pivot_table(
    index='EPIC_MRN',
    columns='PhecodeString',
    values='RECORDED_DATE',
    aggfunc='count',  # You can use other aggregation functions if needed
    fill_value=0
).reset_index()


# Convert the relevant part of the dataframe to binary (where any number greater than 0 becomes 1)
binary_pivoted_values = pivoted_df.iloc[:, 2:].applymap(lambda x: 1 if x > 0 else 0)

# Combine the first two columns with the binarized values
binary_pivoted_df = pd.concat([pivoted_df.iloc[:, :2], binary_pivoted_values], axis=1)
binary_pivoted_df = binary_pivoted_df.rename(columns={'CDR_ID_x': 'CDR_ID'})

# Note - error here - investigate - later
ad_demo_phecode = pd.merge(ad_demo_data, binary_pivoted_df, on='EPIC_MRN', how='left') #.dropna()
ad_demo_phecode
ad_demo_phecode.to_csv('/storage1/fs1/aditigupta/Active/Will/ML_LLM_AD/AD_prediction/checkpoint2.csv', index=False)

###############################################################################################################################
################################## Medications ################################################################################
###############################################################################################################################

# Load your new data
df1 = pd.read_csv('/storage1/fs1/aditigupta/Active/Will/NovoNordisk_Analysis/beers_meds_df.csv')
df2 = pd.read_csv('/storage1/fs1/aditigupta/Active/Will/NovoNordisk_Analysis/df_med_names.csv')
df_med_names = pd.concat([df1, df2], axis=0).drop(columns=['Unnamed: 0'])
med_orders = pd.read_csv(path + '/med_orders.csv')

# Create a dictionary mapping each MED_CLASS to a list of MED_NAMEs
med_class_to_names = df_med_names.groupby('MED_CLASS')['MED_NAME'].apply(list).to_dict()

# Ensure 'MED_ORDER_DATE' is in datetime format
med_orders['MED_ORDER_DATE'] = pd.to_datetime(med_orders['MED_ORDER_DATE'])

# Step 1: Merge the dataframes on MED_NAME
merged_df = pd.merge(med_orders, df_med_names, on='MED_NAME', how='left')

# Step 2: Filter out med_orders that only occur in a 'Hospital Encounter' MED_PT_TYPE
# Group by EPIC_MRN and MED_NAME and check if any other MED_PT_TYPE exists for the same MED_NAME
has_other_pt_type = merged_df.groupby(['EPIC_MRN', 'MED_NAME'])['MED_PT_TYPE'].transform(lambda x: 'Hospital Encounter' not in x.unique() or len(x.unique()) > 1)

filtered_df = merged_df[has_other_pt_type]

# Step 3: Exclude rows where MED_PT_TYPE is 'ANESTHETICS'
med_orders_filtered = filtered_df[filtered_df['MED_PT_TYPE'] != 'ANESTHETICS']


# Create an empty DataFrame to store the medication flags for each patient
med_condition_columns = pd.DataFrame(0, index=ad_demo_phecode.index, columns=med_class_to_names.keys())

# Iterate through each record in ad_demo_phecode
for idx, row in ad_demo_phecode.iterrows():
    epic_mrn = row['EPIC_MRN']
    performed_date = row['INDEX_DATE']
    
    # Filter medication orders for this patient and where the order date is before the performed date
    patient_med_orders = med_orders_filtered[(med_orders_filtered['EPIC_MRN'] == epic_mrn) & (med_orders_filtered['MED_ORDER_DATE'] < performed_date)]
    
    for med_class, med_names in med_class_to_names.items():
        # Check if any medication order matches the med class
        has_med_class = patient_med_orders['MED_NAME'].isin(med_names).any()
        # Set the med_class column to 1 if the med class is present, otherwise 0
        med_condition_columns.at[idx, med_class] = int(has_med_class)

# Concatenate the original DataFrame with the medication columns
ad_demo_phecode = pd.concat([ad_demo_phecode, med_condition_columns], axis=1)
ad_demo_phecode
ad_demo_phecode.to_csv('/storage1/fs1/aditigupta/Active/Will/ML_LLM_AD/AD_prediction/checkpoint3.csv', index=False)
###############################################################################################################################
################################## Number of Medications ######################################################################
###############################################################################################################################

###############################################################################################################################
################################## exclude hospital and anesthesa here too #################################################################
###############################################################################################################################

###############################################################################################################################
################################## make this a count of class #################################################################
############################################################################################################################### med_orders['THERAPEUTIC_CLASS_NAME'].value_counts()
############################################################################################################################### or last 6 months
############################################################################################################################### total number and compare performance

# Convert columns to datetime if they are not already
ad_demo_phecode['INDEX_DATE'] = pd.to_datetime(ad_demo_phecode['INDEX_DATE'])
med_orders_filtered = med_orders_filtered[med_orders_filtered['EPIC_MRN'].isin(ad_demo_phecode['EPIC_MRN'])]
print('initial_complete')

# Merge the DataFrames on EPIC_MRN
merged_df = pd.merge( ad_demo_phecode[['EPIC_MRN', 'INDEX_DATE']], med_orders_filtered, on='EPIC_MRN', suffixes=('', '_order'))
print('merged_df')

# Filter rows where MED_ORDER_DATE is before INDEX_DATE
filtered_df = merged_df[merged_df['MED_ORDER_DATE'] < merged_df['INDEX_DATE']]
print('filtered_rows')

# Count occurrences for each EPIC_MRN and INDEX_DATE date
counts = filtered_df.groupby(['EPIC_MRN', 'INDEX_DATE']).size().reset_index(name='NUMBER_OF_MEDS_BEFORE_INDEX') # Is this correct, I doubt it
print('counts complete')

# Merge the counts back into the original df
ad_demo_phecode = ad_demo_phecode.merge(counts, how='left', on='EPIC_MRN')

# Fill NaN values with 0 where there are no matching orders before performed date
ad_demo_phecode['NUMBER_OF_MEDS_BEFORE_INDEX'].fillna(0, inplace=True)
ad_demo_phecode.to_csv('/storage1/fs1/aditigupta/Active/Will/ML_LLM_AD/AD_prediction/checkpoint4.csv', index=False)
###############################################################################################################################
################################## Biomarkers #######################################################################
###############################################################################################################################

###############################################################################################################################
################################## Filter Labs - need to be after date #######################################################################
############################################################################################################################### Filter LAbs
print("Adding biomarkers")

def read_and_combine_labs():
    # Define the years we are interested in
    years = range(2018, 2025)
    
    # Initialize an empty list to store individual dataframes
    dfs = []
    
    # Loop through each year, read the corresponding CSV and append to the list
    for year in years:
        file_path = f'/storage1/fs1/aditigupta/Active/Will/20241128_MDC_datapull/labs_{year}.csv'
        df = pd.read_csv(file_path)
        df = df[df['LAB_TEST'] == 'P-TAU/ABETA42']
        dfs.append(df)
    
    # Combine all the dataframes into one
    combined_df = pd.concat(dfs, ignore_index=True)
    
    return combined_df

# Example usage
labs = read_and_combine_labs()

labs['LAB_RSLT_COLLECTION_DTTM'] = pd.to_datetime(labs['LAB_RSLT_COLLECTION_DTTM'])
ad_demo_phecode['INDEX_DATE_x'] = pd.to_datetime(ad_demo_phecode['INDEX_DATE_x'])

# Merge the dataframes on 'EPIC_MRN'
merged_df = pd.merge(labs, ad_demo_phecode[['EPIC_MRN', 'INDEX_DATE_x']], on='EPIC_MRN', how='left')

# Filter the rows where LAB_RSLT_COLLECTION_DTTM is before INDEX_DATE
filtered_labs = merged_df[merged_df['LAB_RSLT_COLLECTION_DTTM'] < merged_df['INDEX_DATE_x']]


ptau_abeta_labs = filtered_labs[filtered_labs['LAB_TEST'] == 'P-TAU/ABETA42']

# Remove any rows that contain the character '<' in LAB_RSLT_VALUE
ptau_abeta_labs = ptau_abeta_labs[~ptau_abeta_labs['LAB_RSLT_VALUE'].str.contains('<', na=False)]

# Convert the LAB_RSLT_VALUE column to numeric
ptau_abeta_labs['LAB_RSLT_VALUE'] = pd.to_numeric(ptau_abeta_labs['LAB_RSLT_VALUE'], errors='coerce')

# Setting column for positive ptau/abeta > 0.028 test
positive_patu_abeta_mrns = ptau_abeta_labs[ptau_abeta_labs['LAB_RSLT_VALUE'] >= 0.028]['EPIC_MRN']

ad_demo_phecode['POSITIVE_PTAU_ABETA'] = ad_demo_phecode['EPIC_MRN'].isin(positive_patu_abeta_mrns).astype(int)

positive_pet_mrns = pd.read_csv('/storage1/fs1/aditigupta/Active/Will/ML_LLM_AD/LLM_extract/rapid_progression/positive_pet_mrns.csv')['EPIC_MRN']
ad_demo_phecode['POSITIVE_PET'] = ad_demo_phecode['EPIC_MRN'].isin(positive_pet_mrns).astype(int)


ad_demo_phecode.to_csv('/storage1/fs1/aditigupta/Active/Will/ML_LLM_AD/AD_prediction/checkpoint5.csv', index=False)

###############################################################################################################################
################################## Notes #######################################################################
###############################################################################################################################
note_ids = pd.read_csv('/storage1/fs1/aditigupta/Active/Will/ML_LLM_AD/LLM_extract/llama_modeling/clinical_note_ids.csv')
df1 = pd.read_csv('/storage1/fs1/aditigupta/Active/Will/ML_LLM_AD/LLM_extract/llama_modeling/results_llama3.2-120k_htn_dept_atr_infarct.csv')
df2 = pd.read_csv('/storage1/fs1/aditigupta/Active/Will/ML_LLM_AD/LLM_extract/llama_modeling/results_llama3.2-120k_htn_dept_atr_infarct2.csv')
df3 = pd.read_csv('/storage1/fs1/aditigupta/Active/Will/ML_LLM_AD/LLM_extract/llama_modeling/results_llama3.2-120k_htn_dept_atr_infarct3.csv')
df4 = pd.read_csv('/storage1/fs1/aditigupta/Active/Will/ML_LLM_AD/LLM_extract/llama_modeling/results_llama3.2-120k_htn_dept_atr_infarct4.csv')
df = pd.concat([df1.iloc[:1879, :], df2.iloc[1879:2900, :], df3.iloc[2900:3170, :], df4.iloc[3170:, :] ], axis=0)
df['EPIC_MRN'] = df['EPIC_MRN'].astype(int)
df['NOTE_ID'] = df['NOTE_ID'].astype(int)
df = pd.merge(df, note_ids, how='inner', on='NOTE_ID')
df = df.rename(columns={'EPIC_MRN_x': 'EPIC_MRN'})

ad_demo_phecode = ad_demo_phecode.rename(columns={'INDEX_DATE_x': 'INDEX_DATE'})
ad_demo_phecode = ad_demo_phecode.drop(columns=['INDEX_DATE_y'])

# Ensure the date columns are in datetime format
ad_demo_phecode['INDEX_DATE'] = pd.to_datetime(ad_demo_phecode['INDEX_DATE'])
df['NOTE_DOS'] = pd.to_datetime(df['NOTE_DOS'])

# Convert the date columns to just the date part (ignoring the time part)
ad_demo_phecode['INDEX_DATE'] = ad_demo_phecode['INDEX_DATE'].dt.date
df['NOTE_DOS'] = df['NOTE_DOS'].dt.date

# Initialize the new columns in ad_demo_phecode to be 0
ad_demo_phecode['htn'] = 0 # these are or
ad_demo_phecode['depr'] = 0 # or with phenotyles
ad_demo_phecode['infarcts'] = 0
ad_demo_phecode['atrophy'] = 0

# Loop through each row of ad_demo_phecode
for i, cdr_row in ad_demo_phecode.iterrows():
    #print(i)
    epic_mrn = cdr_row['EPIC_MRN']
    performed_date = cdr_row['INDEX_DATE']

    # Subset df for rows matching the current EPIC_MRN
    subset_df = df[df['EPIC_MRN'] == epic_mrn]

    # Check the condition on the subset of df
    for j, df_row in subset_df.iterrows():

        if df_row['NOTE_DOS'] <= performed_date and df_row['htn'] == 'yes':
            # Set htn to 1 if condition is met
            ad_demo_phecode.at[i, 'htn'] = 1

        if df_row['NOTE_DOS'] <= performed_date and df_row['depr'] == 'yes':
            # Set htn to 1 if condition is met
            ad_demo_phecode.at[i, 'depr'] = 1

        if df_row['NOTE_DOS'] <= performed_date and df_row['infarcts'] == 'yes':
            # Set htn to 1 if condition is met
            ad_demo_phecode.at[i, 'infarcts'] = 1

        if df_row['NOTE_DOS'] <= performed_date and df_row['atrophy'] == 'yes':
            # Set htn to 1 if condition is met
            ad_demo_phecode.at[i, 'atrophy'] = 1



df1 = pd.read_csv('/storage1/fs1/aditigupta/Active/Will/ML_LLM_AD/LLM_extract/llama_modeling/results_llama3.2-120k_llama_repeat_misplace_famhx.csv')
df2 = pd.read_csv('/storage1/fs1/aditigupta/Active/Will/ML_LLM_AD/LLM_extract/llama_modeling/results_llama3.2-120k_llama_repeat_misplace_famhx2.csv')
df3 = pd.read_csv('/storage1/fs1/aditigupta/Active/Will/ML_LLM_AD/LLM_extract/llama_modeling/results_llama3.2-120k_llama_repeat_misplace_famhx3.csv')
df = pd.concat([df1.iloc[:2072, :], df2.iloc[2071:3375, :], df3.iloc[3377:, :] ], axis=0)
df['EPIC_MRN'] = df['EPIC_MRN'].astype(int)
df['NOTE_ID'] = df['NOTE_ID'].astype(int)
df = pd.merge(df, note_ids, how='inner', on='NOTE_ID')
df = df.rename(columns={'EPIC_MRN_x': 'EPIC_MRN'})

df['NOTE_DOS'] = pd.to_datetime(df['NOTE_DOS'])
df['NOTE_DOS'] = df['NOTE_DOS'].dt.date

# Initialize the new columns in ad_demo_phecode to be 0
ad_demo_phecode['repeat'] = 0
ad_demo_phecode['misplaces_objects'] = 0
ad_demo_phecode['famhx'] = 0

# Loop through each row of ad_demo_phecode
for i, cdr_row in ad_demo_phecode.iterrows():
    #print(i)
    epic_mrn = cdr_row['EPIC_MRN']
    performed_date = cdr_row['INDEX_DATE']

    # Subset df for rows matching the current EPIC_MRN
    subset_df = df[df['EPIC_MRN'] == epic_mrn]

    # Check the condition on the subset of df
    for j, df_row in subset_df.iterrows():

        if df_row['NOTE_DOS'] <= performed_date and df_row['repeat'] == 'yes':
            # Set htn to 1 if condition is met
            ad_demo_phecode.at[i, 'repeat'] = 1

        if df_row['NOTE_DOS'] <= performed_date and df_row['misplaces_objects'] == 'yes':
            # Set htn to 1 if condition is met
            ad_demo_phecode.at[i, 'misplaces_objects'] = 1

        if df_row['NOTE_DOS'] <= performed_date and df_row['famhx'] == 'yes':
            # Set htn to 1 if condition is met
            ad_demo_phecode.at[i, 'famhx'] = 1

###############################################################################################################################
################################## Drop or Combine Similar Phecodes ########################################################
###############################################################################################################################

# Update 'Hypertension' column where 'htn' is 1
ad_demo_phecode.loc[ad_demo_phecode['htn'] == 1, 'Hypertension'] = 1
ad_demo_phecode.loc[ad_demo_phecode['depr'] == 1, 'Major depressive disorder'] = 1

# Drop the 'htn' column
ad_demo_phecode = ad_demo_phecode.drop(columns=['htn', 'depr'])

            
###############################################################################################################################
################################## Saving to CSV #######################################################################
############################################################################################################################### 
ad_demo_phecode.to_csv("/storage1/fs1/aditigupta/Active/Will/ML_LLM_AD/AD_prediction/df_extract_ad_prediction2.csv", index=False)
