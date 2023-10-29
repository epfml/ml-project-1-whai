import numpy as np

#define a few functions used to clean and scale the data properly
def frequency_scaler(df, col, names_map):
    """
    Converts the different frequency scales to one common scale, and asigns lacking answers to NaN
    """
    times_per_day   = np.where(((df[:, names_map[col]]>=101)*(df[:, names_map[col]]<=199))==True)
    times_per_week  = np.where(((df[:, names_map[col]]>=201)*(df[:, names_map[col]]<=299))==True)
    times_per_month = np.where(((df[:, names_map[col]]>=301)*(df[:, names_map[col]]<=399))==True)
    none_indices    = np.where(((df[:, names_map[col]] == 777) + (df[:, names_map[col]] == 999))==True)

    df[:, names_map[col]][times_per_day]                = df[:, names_map[col]][times_per_day]%100
    df[:, names_map[col]][times_per_week]               = (df[:, names_map[col]][times_per_week]%200)/7
    df[:, names_map[col]][times_per_month]              = (df[:, names_map[col]][times_per_month]%300)/30
    df[:, names_map[col]][df[:, names_map[col]]==300]   = 1/30
    df[:, names_map[col]][df[:, names_map[col]]==555]   = 0
    df[:, names_map[col]][none_indices]                 = np.nan

def weekly_frequency_scaler(df, col, names_map):
    """
    Converts the different frequency scales to one common scale, and asigns lacking answers to NaN
    """
    times_per_week  = np.where(((df[:, names_map[col]]>=101)*(df[:, names_map[col]]<=199))==True)
    times_per_month = np.where(((df[:, names_map[col]]>=201)*(df[:, names_map[col]]<=299))==True)
    none_indices    = np.where(((df[:, names_map[col]] == 777) + (df[:, names_map[col]] == 999))==True)

    df[:, names_map[col]][times_per_week]                   = (df[:, names_map[col]][times_per_week]%100)/7
    df[:, names_map[col]][times_per_month]                  = (df[:, names_map[col]][times_per_month]%200)/30
    df[:, names_map[col]][df[:, names_map[col]]==888]       = 0
    df[:, names_map[col]][none_indices]                     = np.nan

def hours_to_minutes(df, col, names_map):
    """
    Converts the different time scales to one common scale, and asigns lacking answers to NaN
    """
    hour_indices    = np.where(( (df[:, names_map[col]]>=1)*(df[:, names_map[col]]<=759) + (df[:, names_map[col]]>=800)*(df[:, names_map[col]]<=959))==True)
    none_indices    = np.where(((df[:, names_map[col]]==777) + (df[:, names_map[col]]==999))==True)
    
    df[:, names_map[col]][hour_indices] = 60*df[:, names_map[col]][hour_indices]//100 + df[:, names_map[col]][hour_indices]%100                                                 
    df[:, names_map[col]][none_indices] = np.nan

#function to NaN in a column of a pandas dataframe
def replace_nan(df, column, value, names_map):
    """
    replaces nans in a column by a given value
    """
    
    nan_indices = np.where(np.isnan(df[:, names_map[column]]) + (df[:, names_map[column]]==np.nan))
    df[:, names_map[column]][nan_indices] = value


def clean_data(names_map, x_raw, y_raw=None, is_y=False, is_train_data=True, mean_dico=None, median_dico=None):
    """
    Preprocessing of the data for the feature matrix, and the output vector.
    
    Input:
        names_map (dictionnary)         : dictionnary mapping the features to their index
        x_raw (N,D)-array               : raw input data
        y_raw (N,)-array                : raw output data
        is_y (boolean)                  : True if we have the output feature y, False otherwise
        is_train_data (boolean)         : True if the data is the training data, False otherwise
        new_mean_dico (dictionnary)     : means of the features in the training data if provided
        new_median_dico (dictionnary)   : medians of the features in the training data if provided

    Output:
        x                               : input data with the preprocessing applied
        y                               : output data with the preprocessing applied
        new_mean_dico (dictionnary)     : means of the features in the training data if training data, empty-set otherwise
        new_median_dico (dictionnary)   : medians of the features in the training data if training data, empty-set otherwise
    """
    x   = x_raw.copy()
    try:
        y = y_raw.copy()
    except:
        y = None
    
    #replaces -1 by 0 in the output feature
    if is_y:
        y = np.where(y == -1, 0, y)

    #Converting the weights to kg, and asigning lacking answers to NaN
    array = x[:, names_map["WEIGHT2"]]
    pounds_indices  = np.where((array >= 50)*(array <= 999))
    kg_indices      = np.where((array >= 9000)*(array <= 9998))
    none_indices    = np.where((array == 7777) + (array == 9998))

    x[:, names_map["WEIGHT2"]][pounds_indices] = 0.453592 * x[:, names_map["WEIGHT2"]][pounds_indices]
    x[:, names_map["WEIGHT2"]][kg_indices] = x[:, names_map["WEIGHT2"]][kg_indices]%9000
    x[:, names_map["WEIGHT2"]][none_indices] = np.nan

    #converting the height to meters, and asigning lacking answers to NaN
    array = x[:, names_map["HEIGHT3"]]
    imperial_indices    = np.where((array >= 200)*(array <= 711))
    cm_indices          = np.where((array >= 9000)*(array <= 9998))
    none_indices        = np.where((array == 9998) + (array == 7777))

    #conversion to cm
    x[:, names_map["HEIGHT3"]][imperial_indices] = x[:, names_map["HEIGHT3"]][imperial_indices]//100 * 30.48 + x[:, names_map["HEIGHT3"]][imperial_indices]%100 * 2.54
    x[:, names_map["HEIGHT3"]][cm_indices] = x[:, names_map["HEIGHT3"]][cm_indices]%9000
    x[:, names_map["HEIGHT3"]][none_indices] = np.nan

    #converting FLSHTMY2 to months, and asigning lacking answers to NaN
    array = x[:, names_map["FLSHTMY2"]]
    none_indices = np.where((array == 777777) + (array == 999999))
    days_indices = np.where((array >= 12014)*(array <= 122015))

    x[:, names_map["FLSHTMY2"]][none_indices] = np.nan
    x[:, names_map["FLSHTMY2"]][days_indices] = x[:, names_map["FLSHTMY2"]][days_indices]//10000 + 12*x[:, names_map["FLSHTMY2"]][days_indices]%10000

    #fix the frequency scales of the following columns
    frequency_scaler(x, "FRUITJU1", names_map)
    frequency_scaler(x, "FRUIT1", names_map)
    frequency_scaler(x, "FVBEANS", names_map)
    frequency_scaler(x, "FVGREEN", names_map)
    frequency_scaler(x, "FVORANG", names_map)
    frequency_scaler(x, "VEGETAB1", names_map)

    hours_to_minutes(x, "EXERHMM1", names_map)
    hours_to_minutes(x, "EXERHMM2", names_map)

    weekly_frequency_scaler(x, "ALCDAY5", names_map)
    weekly_frequency_scaler(x, "EXEROFT1", names_map)
    weekly_frequency_scaler(x, "EXEROFT2", names_map)
    weekly_frequency_scaler(x, "STRENGTH", names_map)

    #Transformation hashmap
    dico_transfos={
                "GENHLTH":{7:np.nan,8:np.nan,9:np.nan},"POORHLTH":{88:0,77:np.nan,99:np.nan},"HLTHPLN1":{7:np.nan,9:np.nan},"CHECKUP1":{8:15,7:np.nan,9:np.nan},
                "BPMEDS":{7:np.nan,9:np.nan}, "TOLDHI2":{7:np.nan,9:np.nan}, "PHYSHLTH":{88:0,77:np.nan,99:np.nan},
                "MENTHLTH":{88:0,77:np.nan,99:np.nan}, "CVDSTRK3":{7:np.nan, 9:np.nan}, "HLTHPLN1":{9:np.nan}, "CHCOCNCR":{7:np.nan, 9:np.nan},
                "HAVARTH3":{7:np.nan, 9:np.nan}, "CHCKIDNY":{7:np.nan, 9:np.nan}, "DIABETE3":{7:np.nan, 9:np.nan}, "CHCCOPD1":{7:np.nan, 9:np.nan},
                "ASTHMA3":{7:np.nan, 9:np.nan}, "ASTHNOW":{7:np.nan, 9:np.nan}, "CHCSCNCR":{7:np.nan, 9:np.nan}, "ADDEPEV2":{7:np.nan, 9:np.nan},
                "DIABAGE2":{98:np.nan, 99:np.nan}, "EDUCA":{9:np.nan}, "INCOME2":{77:np.nan, 99:np.nan}, "QLACTLM2":{7:np.nan, 9:np.nan},
                "USEEQUIP":{7:np.nan, 9:np.nan}, "BLIND":{7:np.nan, 9:np.nan}, "DECIDE":{7:np.nan, 9:np.nan}, "DIFFWALK":{7:np.nan, 9:np.nan},
                "DIFFDRES":{7:np.nan, 9:np.nan}, "DIFFALON":{7:np.nan, 9:np.nan}, "SMOKE100":{7:np.nan, 9:np.nan}, "SMOKDAY2":{7:np.nan, 9:np.nan},
                "USENOW3":{7:np.nan, 9:np.nan}, "AVEDRNK2":{77:np.nan, 99:np.nan}, "DRNK3GE5":{77:np.nan, 88:np.nan, 99:np.nan}, "MAXDRNKS":{77:np.nan, 99:np.nan},
                "EXERANY2":{7:np.nan, 9:np.nan}, "EXERHMM1":{777:np.nan, 999:np.nan}, "SEATBELT":{7:np.nan, 8:np.nan, 9:np.nan}, "PNEUVAC3":{7:np.nan, 9:np.nan},
                "ARTHDIS2":{7:np.nan, 9:np.nan}, "ARTHSOCL":{7:np.nan, 9:np.nan}, "JOINPAIN":{77:np.nan, 99:np.nan}, "ARTHEDU":{7:np.nan, 9:np.nan}, "FLUSHOT6":{7:np.nan, 9:np.nan},
                "DOCTDIAB":{88:0, 77:np.nan, 99:np.nan}, "DIABEYE":{7:np.nan, 9:np.nan}, "CRGVMST2":{7:np.nan, 9:np.nan}, "VIDFCLT2":{7:np.nan}, "VIREDIF3":{7:np.nan},
                "VICTRCT4":{7:np.nan}, "VIGLUMA2":{7:np.nan}, "VIMACDG2":{7:np.nan}, "CIMEMLOS":{7:np.nan, 9:np.nan}, "CDSOCIAL":{7:np.nan, 9:np.nan}, "DRADVISE":{7:np.nan, 9:np.nan},
                "ASTHMAGE":{97:6, 98:np.nan, 99:np.nan}, "ASERVIST":{88:0}, "CVDASPRN":{7:np.nan, 9:np.nan}, "RDUCHART":{7:np.nan, 9:np.nan}, "ARTHEXER":{7:2, 9:2},
                "HPVADVC2":{7:np.nan, 9:np.nan}, "HPVADSHT":{77:np.nan, 99:np.nan}, "PCPSARE1":{7:np.nan, 9:np.nan}, "MISTMNT":{7:np.nan, 9:np.nan},
                "_CHISPNC":{9:np.nan}, "_RFCHOL":{9:np.nan}, "_LTASTH1":{9:1}, "_CASTHM1":{9:0}, "_ASTHMS1":{9:3}, "_HISPANC":{9:2}, "_AGEG5YR":{14:np.nan}, "_CHLDCNT":{9:np.nan},
                "_EDUCAG":{9:np.nan}, "_INCOMG":{9:np.nan}, "_SMOKER3":{9:4}, "_RFSMOK3":{9:1}, "DRNKANY5":{9:1, 7:1}, "DROCDY3_":{900:np.nan},
                "_TOTINDA":{9:np.nan}, "_LMTSCL1":{9:np.nan}, "_RFSEAT2":{9:0}, "_PASTRNG":{9:2}, "_PACAT1":{9:2}, "STRFREQ_":{99000:np.nan}
                }

    #applying the transformations
    for col in dico_transfos:
        for key, value in dico_transfos[col].items():
            x[:, names_map[col]][np.where(x[:, names_map[col]] == key)] = value

    #list of features to replace NaN with a specific value
    CRGVMST2_value    = 6
    VICTRCT4_value    = 3
    ARTHEXER_value    = 2
    HPVADSHT_value    = 0
    PCPSARE1_value    = 2

    #list of features to replace NaN with the mean
    mean_features   = ["POORHLTH", "PHYSHLTH", "MENTHLTH", "WEIGHT2", "DIABAGE2", "HEIGHT3", "FRUITJU1", "FRUIT1", "FVBEANS", "FVGREEN", "FVORANG", "VEGETAB1",
                    "EXERHMM1", "FLSHTMY2", "FTJUDA1_", "FRUTDA1_", "BEANDAY_", "GRENDAY_", "ORNGDAY_", "VEGEDA1_", "STRFREQ_"]

    #list of features to replace NaN with the median
    median_features = ["GENHLTH", "HLTHPLN1", "CHECKUP1", "BPMEDS", "TOLDHI2", "CVDSTRK3", "CHCKIDNY", "CHCOCNCR", "HAVARTH3", "DIABETE3", "CHCCOPD1", "ASTHMA3", "ASTHNOW", "CHCSCNCR",
                    "ADDEPEV2", "EDUCA", "INCOME2", "QLACTLM2", "USEEQUIP", "BLIND", "DECIDE", "DIFFWALK", "DIFFDRES", "DIFFALON", "SMOKE100", "SMOKDAY2", "USENOW3", "ALCDAY5", "AVEDRNK2",
                    "DRNK3GE5", "MAXDRNKS", "EXERANY2", "SEATBELT", "PNEUVAC3", "ARTHDIS2", "ARTHSOCL", "JOINPAIN", "FLUSHOT6", "DOCTDIAB", "VIREDIF3", "VIGLUMA2", "VIMACDG2", "CIMEMLOS",
                    "CDSOCIAL", "DRADVISE", "HPVADVC2", "_CHISPNC", "_DRDXAR1", "_AGEG5YR", "DROCDY3_", "_CHLDCNT", "_TOTINDA", "_LMTSCL1", "ARTHEDU", "_INCOMG", "_EDUCAG", "_RFCHOL", 
                    "MISTMNT", "DIABEYE", "_BMI5"]

    #list of specific features
    value_features  = ["CRGVMST2", "VICTRCT4", "ARTHEXER", "HPVADSHT", "PCPSARE1"]

    #replace the NaN with a specific value
    for feature in value_features:
        replace_nan(x, feature, eval(feature + "_value"), names_map)

    new_mean_dico, new_median_dico = {}, {}
    
    #if we are cleaning the training data, we save in dictionnaries the relevant means and features.
    if is_train_data:
        #replace the NaN with the mean
        for feature in mean_features:
            feature_mean = np.nanmean(x[:, names_map[feature]])
            replace_nan(x, feature, feature_mean, names_map)
            new_mean_dico[feature] = feature_mean

        #replace the NaN with the median
        for feature in median_features:
            feature_median = np.nanmedian(x[:, names_map[feature]])
            replace_nan(x, feature, feature_median, names_map)
            new_median_dico[feature] = feature_median

    else:
        #replace the NaN with the mean
        for feature in mean_features:
            replace_nan(x, feature, mean_dico[feature], names_map)

        #replace the NaN with the median
        for feature in median_features:
            replace_nan(x, feature, median_dico[feature], names_map)
    interesting_features=list(dico_transfos.keys())
    return x, y, new_mean_dico, new_median_dico,interesting_features


def scale_data(x, is_train_data=True, train_mean=None, train_std=None):
    """
   Scales the data to have a mean of 0 and a standard deviation of 1.
   
   Input:
       x (N,D)-array               : input data
       is_train_data (boolean)     : True if the data is the training data, False otherwise
       train_mean (D,)-array       : means of the features in the training data if provided
       train_std (D,)-array        : standard deviations of the features in the training data if provided
    
    Output:
        x_scaled (N,D)-array        : scaled input data
        new_train_mean (D,)-array   : means of the features in the training data if training data, None otherwise
        new_train_std (D,)-array    : standard deviations of the features in the training data if training data, None otherwise
    """
    
    x_scaled = x.copy()

    if is_train_data:
        new_train_mean = np.nanmean(x, axis=0)
        new_train_std  = np.nanstd(x, axis=0)

        nonzero_std_indices = np.where(new_train_std > 0)
        zero_std_indices  = np.where(new_train_std == 0)

        x_scaled[:, nonzero_std_indices] =  (x[:, nonzero_std_indices] - new_train_mean[nonzero_std_indices])/new_train_std[nonzero_std_indices]
        x_scaled[:, zero_std_indices]   =   (x[:, zero_std_indices] - new_train_mean[zero_std_indices])

    else:
        nonzero_std_indices = np.where(train_std > 0)
        zero_std_indices    = np.where(train_std == 0)
        
        x_scaled[:, nonzero_std_indices] =  (x[:, nonzero_std_indices] - new_train_mean[nonzero_std_indices])/new_train_std[nonzero_std_indices]
        x_scaled[:, zero_std_indices]   =   (x[:, zero_std_indices] - new_train_mean[zero_std_indices])

        new_train_mean=None
        new_train_std=None
    
    return x_scaled, new_train_mean, new_train_std