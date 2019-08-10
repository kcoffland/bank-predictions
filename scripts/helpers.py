import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
import os
from imblearn.over_sampling import SMOTENC


def load_data(path, file, verbose=False, sep=',', index=None):
    """Reads in the data at the given path/filename. Currently file is assumed to be csv 
    Args:
        path: string, path to file containing data
        file: string, name of file containing data that will be merged together
        verbose: boolean, If the caller wanted information about the df to be displayed
    Returns: 
        df, holds data in the given csv files
    """
    
    df = pd.read_csv(path+file, index_col=index, sep=sep)
    
    if verbose:
        shape = f'{df.shape}'
        dtypes = f'{df.dtypes[:30]}'
        head = f'{df.head()[:10]}'
        name = file.split('.')[0]
        
        print(f'{name} shape'.center(80, '-'))
        print(shape.center(80))
        print()
        print()
        print(f"{name}'s column types".center(80, '-'))
        print(dtypes)
        print()
        print()
        print(f"{name} first five rows".center(80, '-'))
        print(head)
    
    return df

def mean_squared_error(predictions, actual):
    """Computes the MSE for a given prediction and observed data
    Args:
         predictions: Series or float/int, holding predicted target value
         actual: Series, holding actual target value
    """

    return ((predictions-actual)**2).sum() / len(actual)

def find_outliers(data, method='iqr'):
    """Finds the values for which outliers begin
    Args:
        data: DataFrame or Series, holds data you want to find the outliers for
        method: string, Method used to calculate an outlier
    Returns:
        lower, upper: floats or Series of floats, values less than lower are outliers and
                              values larger than upper are outliers. A Series of 
                              floats corresponding to the lower and upper values
                              of the columns will be returned if data is a DataFrame
    """

    if type(data) != pd.core.frame.DataFrame or type(data) != pd.core.series.Series:
        raise TypeError("data must be either a DataFrame or Series")

    if method=='iqr':
        # Finding the interquartile range
        q1 = data.quantile(.25)
        q3 = data.quantile(.75)
        iqr = q3-q1

        upper = q3 + iqr*1.5
        lower = q1 - iqr*1.5
    elif method=='std':
        std = data.std()
        lower = data.mean() - 3*std
        upper = data.mean() + 3*std
    else:
        raise ValueError("Invalid value for 'method' passed")


    return lower, upper

def classifier_categorical_variance(df, cat_cols, target_col):
    """Calculates the variance of the target column due to changes in categorical columns.
       This method calculates how the target_col is split based on the cat_cols, normalizes 
       the result based on the total count, then calculates the variance in the normalized 
       target values. A higher variance corresponds to a feature that can better describe
       changes in rates of the target variable. For example, if the number of rejections 
       varies a lot from month to month, then that can help us separate the data
       
    Args:
        df: DataFrame, Data you want to find the variance on
        cat_cols: list of strings, columns in the dataframe which are categorical
        target_col: string, The column which we want to measure how it varies based
                            on changes due to the different categories
    Returns:
        variances: DataFrame, contains the calculated variances for each given cat_col
                              sorted from highest to lowest
    """
    variances = pd.DataFrame(columns=df[target_col].unique())
    for col in cat_cols:
        counts = df.groupby(col)[target_col].value_counts()
        total_counts = counts.groupby(col).sum()
        scaled_counts = counts / total_counts
        for df_col in variances.columns:
            variances.loc[col, df_col] = scaled_counts.xs(df_col, level=target_col).var()
    
    return variances.sort_values('yes', ascending=False)


def plot_success_rates(df, cat_cols, target_col, num_cols=2):
    """Plots the success rates of the given categorical columns. This is only to be used 
       with a Classification problem

    Args:
        df: DataFrame, holds all the data, assumes target_col is in the data
        cat_cols: list of strings, all the names of columns which the user wants to plot
        target_col: string, name of the target column
        num_cols: int, number of columns in the final figure
    
    Returns:
        Nothing
    """
    n = len(cat_cols)
    num_rows = ceil(n/num_cols)
    fig, axes = plt.subplots(nrows=ceil(n/num_cols), 
                           ncols=num_cols,
                           sharey='row', 
                           figsize=(4*num_cols, 5*num_rows)
                          )
    num_plots = 0
    
    for cat_col, ax in zip(cat_cols, axes.flatten()):
        counts = df.groupby(cat_col)[target_col].value_counts()
        total_counts = counts.groupby(cat_col).sum()
        norm_counts = counts / total_counts

        norm_counts.unstack().plot(kind='bar', ax=ax)
        ax.set_ylabel('Success Rate')
        ax.set_title(f'Comparing {cat_col} Success Rates')
        num_plots += 1
    
    # Removing any extra plots
    for i in range(n, len(axes.flatten())):
        fig.delaxes(axes.flatten()[i])
        
    plt.tight_layout()
    plt.show()

def get_smotenc(X, y):
    """Upsamples categorical and non-categorical data using SMOTENC package
    Args:
        X: DataFrame, feature data that needs to be upsampled
        y: Series, tags corresponding to the given features
    
    Returns:
        us_X: DataFrame, upsampled feature data
        us_y: Series, upsampled target data
    """
    categorical_mask = [index for index, col in enumerate(X) if X[col].dtype==object]
    X_dtypes = X.dtypes

    smote = SMOTENC(categorical_mask)

    upsampled_data, upsampled_results = smote.fit_resample(X, y)

    # Converting the numpy arrays back to dataframes and series objects
    us_X = pd.DataFrame(upsampled_data, columns=X.columns)
    us_y = pd.Series(upsampled_results)

    # The data types are all defaulted to 'object' so I am fixing that
    for col in us_X:
        us_X[col] = us_X[col].astype(X_dtypes[col])
    
    return us_X, us_y