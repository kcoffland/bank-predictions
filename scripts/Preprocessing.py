import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder,\
    LabelBinarizer, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scripts.helpers import check_type


class Preprocessing:
    """
    Processes the data and ensures that the test data cannot learn
    more information that what the training data will give (e.g. new
    columns will be filled with 0s to give no additional information).
    """

    def __init__(self, cols_to_filter=None, ordinal_cols=None, binned_cols=None,
                 bin_type='default', inplace=False, classification=False, 
                 polynomial_features=0, pca=0):
        """
        Args:
            cols_to_filter: list of strings, names columns which need
                            to be removed from the dataset
                            Default: empty list
            ordinal_cols: dict {string: list of strings or list of ints}, the key is the ordinal
                            column and the value is the list of categories in
                            order of importance. ORDER OF IMPORTANCE WILL MATTER
                            Default: empty dict
            binned_cols: dict {string: int}, the key is the column which
                            the user wants to be binned. The int will be the number of
                            bins that will be created. The new column will be added to
                            the dataframe as 'given_binned' where "given" is the key of
                            the dict. If the user wants the original column removed, then
                            they can add it to the columns to filter. Note: some engineered
                            features require the binning of certain columns and will
                            not work if they have not been binned. The type of binning
                            will be specified with bin_type
                            Default: empty dict
            inplace: bool, Whether the transformations to the data are made in place or on
                           a copy of the given data
            
        """

        # stores whether the model preprocessor has been fit to the data
        self.is_fit = False
        self.is_transformed = False

        if cols_to_filter is None:
            self.cols_to_filter = []
        else:
            self.cols_to_filter = cols_to_filter

        self.categorical_cols = pd.Index([])
        self.transformed_cat_cols = pd.Index([])
        self.one_hot_encoder = OneHotEncoder(sparse=False)

        self.numerical_cols = pd.Index([])
        self.transformed_num_cols = pd.Index([])

        # Setting up the binarizer if it is a classification problem
        self.classification = classification
        if self.classification:
            self.label_binarizer = LabelBinarizer()

        # Setting up ordinal columns encoding
        if ordinal_cols is None:
            self.ordinal_cols = {}
            self.ordinal_enc = None
        else:
            self.ordinal_cols = ordinal_cols
            categories = list(ordinal_cols.values())
            self.ordinal_enc = OrdinalEncoder(categories=categories)

        # scaler to make numerical columns between 0 and 1
        self.standard_scaler = StandardScaler()

        # Setting up the polynomial features
        self.polynomial_features = polynomial_features
        if self.polynomial_features:
            self.poly_creater = PolynomialFeatures(self.polynomial_features)

        # Setting up binned columns
        if binned_cols is None:
            self.binned_cols = {}
        else:
            self.binned_cols = binned_cols
            self.bin_type = bin_type
            self.bins = [[] for i in range(len(self.binned_cols))]

        # Giving the user the option to transform data inplace
        self.inplace = inplace
        
        self.pca = pca
        if self.pca:
            self.pca_calc = PCA(self.pca, random_state=44)

    def fit(self, X, y=None):
        """
        Finds out what which columns need to be converted to dummy
        values and stores it as an attribute to be used later in
        the transform method.
        """
        self.is_fit = True

        # Creating the bins from the training data to cut on later
        if self.binned_cols:
            for i, col in enumerate(self.binned_cols.keys()):
                assert (col in X.columns)
                if self.bin_type == 'default':
                    _, self.bins[i] = pd.cut(X[col], self.binned_cols[col],
                                            labels=False, retbins=True, 
                                            include_lowest=True)
                elif self.bin_type == 'quartile':
                    _, self.bins[i] = pd.qcut(X[col], self.binned_cols[col],
                                            labels=False, retbins=True)
                else:
                    raise ValueError("bin_type is not valid")
        
        # Saving the categorical columns of the data
        self.categorical_cols = pd.Index([col for col in X.columns
                                          if X[col].dtype == object
                                          ]
                                         )

        self.numerical_cols = pd.Index([col for col in X.columns \
                                    if (col not in self.categorical_cols) \
                                        and (col not in self.ordinal_cols.keys()) \
                                        and (col not in self.binned_cols.keys())]
                                        )

        # Removing filtered columns from established numeric and categorical columns
        if self.cols_to_filter:
            for col in self.cols_to_filter:
                if col in self.categorical_cols:
                    self.categorical_cols = self.categorical_cols.drop([col])
                elif col in self.numerical_cols:
                    self.numerical_cols = self.numerical_cols.drop([col])

        # Finding all columns that are dummied, note this is added after columns
        # are filtered out of self.categorical_cols
        if not self.categorical_cols.empty:
            # Ensuring ordinal cols aren't dummied
            if self.ordinal_cols:
                nominal = set(self.categorical_cols) - set(self.ordinal_cols.keys())
            else:
                nominal = set(self.categorical_cols)
            self.one_hot_encoder.fit(X.loc[:, nominal])
            self.categorical_cols = pd.Index(nominal)

        # Creating ordinal columns
        if self.ordinal_cols:
            for col in self.ordinal_cols.keys():
                if col not in X.columns:
                    raise ValueError(f"{col} not found in given data")
            self.ordinal_enc.fit(X[list(self.ordinal_cols.keys())])

        # Fitting the scaler to the training data's numerical columns
        if not self.numerical_cols.empty:
            # Creating the polynomial features of the test set so that the 
            if self.polynomial_features:
                temp_data = self.poly_creater.fit_transform(X[self.numerical_cols].copy())
                temp_data = self.standard_scaler.fit_transform(temp_data)
                if self.pca:
                    self.pca_calc.fit(temp_data)
                del temp_data
            else:
                # If no polynomial features are created, then the numerical 
                # columns of the original data can be used
                temp_data = self.standard_scaler.fit_transform(X[self.numerical_cols])
                if self.pca:
                    self.pca_calc.fit(temp_data)
                del temp_data

        if  y is not None and self.classification:
            self.label_binarizer.fit(y)
        
        return self

    def transform(self, X, y=None):
        # Ensuring data has been fit before this method can be used
        if not self.is_fit:
            raise RuntimeError("Fit method must be called before data can be transformed")
        self.is_transformed = True

        # Maybe this naming convention is why software engineers hate Data 
        # Scientists. Maybe it's the stupid comments....
        if self.inplace:
            X_new = X
        else:
            X_new = X.copy()

        # Setting all values for columns not in the training data to 0
        # so no additional information can be learned outside of engineering
        # new features available in the training data
        new_cols = set(X_new.columns) - set(self.numerical_cols) - \
                   set(self.categorical_cols) - set(list(self.ordinal_cols.keys())) - \
                   set(list(self.binned_cols.keys())) - set(self.cols_to_filter)
        for col in new_cols:
            X_new[col] = 0

        ################### Transforming Data ########################
        # Encoding Ordinal columns as well as ensuring they're scaled properly
        if self.ordinal_cols:
            X_new[list(self.ordinal_cols.keys())] = self.ordinal_enc.transform(
                                                        X_new[list(self.ordinal_cols.keys())]
                                                    )
            X_new[list(self.ordinal_cols.keys())] /= X_new[list(self.ordinal_cols.keys())].max()

        # OneHotEncoding the Categorical columns
        if not self.categorical_cols.empty:
            # Transforming the data
            ohe_data = self.one_hot_encoder.transform(
                X_new.loc[:, self.categorical_cols])
            ohe_columns = self.one_hot_encoder.get_feature_names(self.categorical_cols)
            # Creating a new dataframe of the transformed data and new columns
            ohe_df = pd.DataFrame(ohe_data, columns=ohe_columns, index=X_new.index)
            # Removing not transformed columns
            X_new.drop(self.categorical_cols, inplace=True, axis=1)
            # Merging the transformed data back into the original dataframe
            X_new = X_new.join(ohe_df)
            self.transformed_cat_cols = ohe_columns
            # Cleaning up stored data
            del ohe_data
            del ohe_columns
            del ohe_df


        # Scale the numerical columns
        if not self.numerical_cols.empty:
            if self.polynomial_features:
                # Transforming all the numerical columns
                poly_data = self.poly_creater.transform(X_new[self.numerical_cols])
                poly_cols = self.poly_creater.get_feature_names(self.numerical_cols)
                # Creating new dataframe
                poly_df = pd.DataFrame(poly_data, columns=poly_cols, index=X_new.index)
                # Removing old numerical columns
                X_new.drop(self.numerical_cols, inplace=True, axis=1)
                # Merging the transformed data
                X_new = X_new.join(poly_df)
                # Scaling all the numerical columns
                scaled_temp = self.standard_scaler.transform(X_new[poly_cols])
                # Performing PCA after scaling
                if self.pca:
                    pca_data = self.pca_calc.transform(scaled_temp)
                    pca_df = pd.DataFrame(pca_data, index=X_new.index)
                    X_new.drop(poly_cols, inplace=True, axis=1)
                    X_new = X_new.join(pca_df)
                    self.transformed_num_cols = pca_df.columns
                    del pca_data
                    del pca_df
                else:
                    X_new[poly_cols] = scaled_temp
                    self.transformed_num_cols = poly_df.columns
                # Cleaning up stored data
                del poly_data
                del poly_cols
                del poly_df
            else:
                X_new[self.numerical_cols] = self.standard_scaler.transform(
                    X_new[self.numerical_cols])
                self.transformed_num_cols = self.numerical_cols.copy()
                # Performing PCA after scaling
                if self.pca:
                    pca_data = self.pca_calc.transform(X_new[self.numerical_cols])
                    pca_df = pd.DataFrame(pca_data, index=X_new.index)
                    X_new.drop(self.numerical_cols, inplace=True, axis=1)
                    X_new = X_new.join(pca_df)
                    self.transformed_num_cols = pca_df.columns
                    del pca_data
                    del pca_df
        
        ################ Creating/Removing Columns ##################
        # Putting the removal of columns at the very end of this function allows
        # for columns that are created by ohe and poly_features to be dropped 
        # automatically instead of manually later

        # Binning appropriate columns
        if self.binned_cols:
            for i, col in enumerate(self.binned_cols.keys()):
                # I cast to int so that I don't have to worry about the Category dtype
                # If I don't include_lowest, then objects with value 0 will not be included
                X_new[f'{col}_binned'] = pd.cut(X_new[col], self.bins[i],
                                                labels=False, include_lowest=True)\
                                                .astype(int).copy()
                X_new.loc[:, f'{col}_binned'] /= X_new.loc[:, f'{col}_binned'].max()
                self.cols_to_filter.append(col)

        # Removing columns that the user wants to filter
        if self.cols_to_filter:
            for col in self.cols_to_filter:
                if col not in X_new.columns:
                    raise ValueError(f"Given column to filter, {col}, not in data")
                # Dropping columns from the indexes that stored them
                if col in self.transformed_cat_cols:
                    self.transformed_cat_cols = self.transformed_cat_cols.drop([col])
                elif col in self.transformed_num_cols:
                    self.transformed_num_cols = self.transformed_num_cols.drop([col])

            X_new.drop(self.cols_to_filter, inplace=True, axis=1)
        
        # If there is a y column given to preprocess
        if check_type(y, 'SERIES') and self.classification:
            for val in y.unique():
                if val not in self.label_binarizer.classes_:
                    raise ValueError(f"{val} is not in the data which the target "
                        f"column was fit to: {self.label_binarizer.classes_}. "
                        "This may occur when the target column given has already "
                        "been transformed. Ensure that the values you want "
                        f"transformed are in fact {y.unique()}")
            y_new = self.label_binarizer.transform(y.copy())
            y_new = pd.Series(y_new.ravel(), name=y.name, index=y.index)
            return X_new, y_new

        return X_new

    def fit_transform(self, X, y=None):

        return self.fit(X, y=y).transform(X, y=y)

    def get_cat_cols(self):
        if not self.is_transformed:
            raise RuntimeError("Data must be transformed before accessing categorical colums")
        return self.transformed_cat_cols
    
    def get_num_cols(self):
        if not self.is_transformed:
            raise RuntimeError("Data must be transformed before accessing numerical colums")
        return self.transformed_num_cols

