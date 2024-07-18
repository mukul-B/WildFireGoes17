import pandas as pd

def compare_csv(requested_file, summary_file):
    # Load CSV files into DataFrames
    df1 = pd.read_csv(requested_file)
    df2 = pd.read_csv(summary_file)
    df1.rename(columns={'bright_ti4': 'brightness'}, inplace=True)
    df1.rename(columns={'bright_ti5': 'bright_t31'}, inplace=True)
    df2.rename(columns={'bright_ti4': 'brightness'}, inplace=True)
    df2.rename(columns={'bright_ti5': 'bright_t31'}, inplace=True)
    
    grouping_columns = ['latitude', 'longitude', 'acq_date','acq_time','scan','track']
    matching_columns = ['latitude', 'longitude', 'acq_date','acq_time']
    matching_value_col = 'brightness'
    df1 = df1.groupby(grouping_columns, as_index=False)[matching_value_col].sum()
    df2 = df2.groupby(grouping_columns, as_index=False)[matching_value_col].sum()
    # matching_columns = ['latitude', 'longitude', 'acq_date','acq_time']
    # matching_value_col = 'acq_time'
    # Columns to be used for matching
    
    merged_df_2 = pd.merge(df1, df2, on=['latitude', 'longitude'], how='outer', suffixes=('_file1', '_file2'))
    merged_df_2 = merged_df_2[merged_df_2['acq_date_file1'] != merged_df_2['acq_date_file2']]
    merged_df_2 = merged_df_2[(merged_df_2['acq_date_file1'].notna()) & (merged_df_2['acq_date_file2'].notna())]

    merged_df = pd.merge(df1, df2, on=['latitude', 'longitude', 'acq_date'], how='outer', suffixes=('_file1', '_file2'))

    
    merged_df_3 = pd.merge(df1, df2, on=['latitude', 'longitude', 'acq_date','acq_time','scan','track'], how='inner', suffixes=('_file1', '_file2'))
    merged_df_3 = merged_df_3[merged_df_3['brightness_file1'] == merged_df_3['brightness_file2']]

    print(merged_df_3.shape[0])
    print(merged_df_3)
    print("---")
    
    # Filter rows where 'acq_time' is not the same
    acq_time_diff_rows = merged_df[merged_df['acq_time_file1'] != merged_df['acq_time_file2']]
    acq_time_same_rows = merged_df[merged_df['acq_time_file1'] == merged_df['acq_time_file2']]
    
    acq_time_diff_rows = acq_time_diff_rows.assign(
        acq_time_file1=acq_time_diff_rows['acq_time_file1'].fillna('0000').astype(int),
        acq_time_file2=acq_time_diff_rows['acq_time_file2'].fillna('0000').astype(int),
        time_difference=lambda x: (x['acq_time_file1'] - x['acq_time_file2']).astype(int),
        brightness_difference=lambda x: (x[matching_value_col+'_file1'] - x[matching_value_col+'_file2'])
    )

    acq_time_same_rows = acq_time_same_rows.assign(
        acq_time_file1=acq_time_same_rows['acq_time_file1'].fillna('0000').astype(int),
        acq_time_file2=acq_time_same_rows['acq_time_file2'].fillna('0000').astype(int),
        time_difference=lambda x: (x['acq_time_file1'] - x['acq_time_file2']).astype(int),
        brightness_difference=lambda x: (x[matching_value_col+'_file1'] - x[matching_value_col+'_file2'])
    )

    same_time_same_TrackScan = acq_time_same_rows[(acq_time_same_rows['track_file1'] == acq_time_same_rows['track_file2'] ) 
                                                  & (acq_time_same_rows['scan_file1'] == acq_time_same_rows['scan_file2'] )]

    same_time_different_bright_records = acq_time_same_rows[acq_time_same_rows['brightness_difference'] != 0]

    Common_records_different_time_notconsider_scanTrack = acq_time_diff_rows[acq_time_diff_rows['brightness_difference'].notna()]

    Common_records_different_time = Common_records_different_time_notconsider_scanTrack
    # Common_records_different_time = Common_records_different_time_notconsider_scanTrack[(Common_records_different_time_notconsider_scanTrack['track_file1'] == Common_records_different_time_notconsider_scanTrack['track_file2'] ) & (Common_records_different_time_notconsider_scanTrack['scan_file1'] == Common_records_different_time_notconsider_scanTrack['scan_file2'] )]

    requested_file_missing = acq_time_diff_rows[(acq_time_diff_rows[matching_value_col+'_file1'].isna())]
    summary_file_missing = acq_time_diff_rows[(acq_time_diff_rows[matching_value_col+'_file2'].isna())]

    Common_records_with_same_brigthness = Common_records_different_time[Common_records_different_time['brightness_difference'] == 0]

    requested_file_later = Common_records_with_same_brigthness [Common_records_with_same_brigthness['time_difference'] >0]
    
    requested_file_timeDifference_by_number  = Common_records_different_time['time_difference'].value_counts().reset_index()

    print(f"\nFor requested_file: {requested_file} , summary_file: {summary_file}")
    print(f"Number of rows in requested_file: {df1.shape[0]}")
    print(f"Number of rows in summary_file: {df2.shape[0]}")
    print(f"Number of rows with the same latitude, longitude but different acq_date : {merged_df_2.shape[0]}")
    print(f"number of common rows (considering latitude, longitude, acq_date) : {acq_time_diff_rows.shape[0]}")
    print(f"Number of rows missing in requested_file (considering latitude, longitude, acq_date): {requested_file_missing.shape[0]}")
    print(f"Number of rows missing in summary_file (considering latitude, longitude, acq_date): {summary_file_missing.shape[0]}")
    print(f"Check : {df2.shape[0]} + {summary_file_missing.shape[0]} = {df2.shape[0] + summary_file_missing.shape[0]} = {df1.shape[0]}")
    print(f"Check : {df1.shape[0]} + {requested_file_missing.shape[0]} = {df1.shape[0] + requested_file_missing.shape[0]} = {df2.shape[0]}")
    print(f"Number of rows with the same latitude, longitude, acq_date, and acq_time : {acq_time_same_rows.shape[0]}")
    print(f"Number of rows with the same latitude, longitude, acq_date, and acq_time (same track & scan) : {same_time_same_TrackScan.shape[0]}")
    
    print(f"Number of rows with the same latitude, longitude, acq_date, and acq_time but different brightness: {same_time_different_bright_records.shape[0]}")
    print(f"Number of rows with the same latitude, longitude, acq_date, but different acquisition time (not considering scan and track): {Common_records_different_time_notconsider_scanTrack.shape[0]}")
    print(f"Number of rows with the same latitude, longitude, acq_date, but different acquisition time: {Common_records_different_time.shape[0]}")
    print(f"Number of common rows with the same brightness: {Common_records_with_same_brigthness.shape[0]}")
    print(f"Number of rows where requested_file has a later acquisition time: {requested_file_later.shape[0]}")
    print(requested_file_timeDifference_by_number)
    
    
    
# Usage example
# compare_csv("VIIRS_Source/fire_archive_SV-C2_478216.csv", 'VIIRS_Source/viirs-snpp_2020_United_States.csv')
# compare_csv("VIIRS_Source/fire_archive_SV-C2_480397.csv", 'VIIRS_Source/viirs-snpp_2021_United_States.csv')
# compare_csv("VIIRS_Source/fire_archive_SV-C2_480488.csv", 'VIIRS_Source/viirs-snpp_2022_United_States.csv')
# compare_csv("VIIRS_Source/fire_nrt_SV-C2_480488.csv", 'VIIRS_Source/viirs-snpp_2022_United_States.csv')
# compare_csv("VIIRS_Source/fire_nrt_SV-C2_480488.csv", 'VIIRS_Source/fire_archive_SV-C2_480488.csv')
# compare_csv("VIIRS_Source/viirs-snpp_2020_United_States.csv", 'VIIRS_Source/viirs-snpp_2020_United_States_2.csv')
year = '2022'
# compare_csv( f'VIIRS_Source_new/fire_archive_SV-C2_{year}.csv',f'VIIRS_Source/viirs-snpp_{year}_United_States.csv')
# compare_csv( f'VIIRS_Source_new/fire_archive_SV-C2_{year}.csv',f'VIIRS_Source_new/fire_nrt_SV-C2_{year}.csv')
compare_csv(f'VIIRS_Source_new/fire_nrt_SV-C2_{year}.csv', f'VIIRS_Source_new/fire_nrt_J1V-C2_{year}.csv')
# compare_csv( f'VIIRS_Source_new/fire_archive_SV-C2_{year}.csv',f'VIIRS_Source_new/fire_nrt_J1V-C2_{year}.csv')

# latitude   longitude  bright_ti4  scan  track    acq_date  acq_time satellite instrument confidence  version  bright_ti5    frp daynight  type
#  latitude   longitude  brightness  scan  track    acq_date  acq_time satellite instrument confidence  version  bright_t31    frp daynight  type


# year , snpp  , noaa-20
# 2019 , sp    , NRT
# 2020 , sp    , NRT
# 2021 , sp    , NRT
# 2022 , (sp,NRT)    , NRT
# 2023 , NRT    , NRT