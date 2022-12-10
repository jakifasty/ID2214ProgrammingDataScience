import numpy as np
import pandas as pd
import time

# rklearn
import sklearn
from sklearn.tree import DecisionTreeClassifier


# rdkit
from rdkit import Chem
import rdkit.Chem.rdMolDescriptors as d
import rdkit.Chem.Fragments as f
import rdkit.Chem.Lipinski as l


# Reused functions from Assignment 1 for preprocessing
def column_filter(df):
    filtered_df = df.copy()  # copy input dataframe

    # iterate through all columns and consider to drop a column only if it is not labeled "CLASS" or "ID"
    # you may check the number of unique (non-missing) values in a column by applying the pandas functions
    # dropna and unique to drop missing values and get the unique (remaining) values
    filtered_df = filtered_df.dropna(how='all', axis=1)
    for col in filtered_df.columns:
        if col != "CLASS" and col != "ID":
            if filtered_df[col].dropna().unique().size == 1:
                filtered_df = filtered_df.drop(col, axis=1)

    column_filter = filtered_df.columns  # list of the names of the remaining columns, including "CLASS" and "ID"

    return filtered_df, column_filter


def apply_column_filter(df, column_filter):
    filtered_new_df = df.copy()  # copy input dataframe

    # drop each column that is not included in column_filter
    for col in filtered_new_df.columns:
        if col not in column_filter:
            filtered_new_df = filtered_new_df.drop(col, axis=1)

    return filtered_new_df


def imputation(df):
    df_temp = df.copy()
    values = {}
    for column in df_temp:
        columnSeriesObj = df_temp[column]
        if columnSeriesObj.dtype == int or columnSeriesObj.dtype == float:
            values[column] = columnSeriesObj.mean()
        elif columnSeriesObj.dtype == object:
            values[column] = columnSeriesObj.mode()[0]

    df_temp.fillna(value=values, inplace=True)

    return df_temp, values


def apply_imputation(df, imputation):
    df_temp = df.copy()
    values = imputation
    df_temp.fillna(value=values, inplace=True)
    return df_temp


def one_hot(df):
    new_df = df.copy()  # copy input dataframe

    one_hot = {}  # a mapping (dictionary) from column name to a set of categories (possible values for the feature)

    for col in new_df.columns:
        if (new_df[col].dtype == "object" or new_df[col].dtype == "category") and col != "CLASS" and col != "ID":
            one_hot[col] = set(new_df[col])
            for value in one_hot[col]:
                new_df[col + "_" + value] = (new_df[col] == value).astype(float)
            new_df = new_df.drop(col, axis=1)

    return new_df, one_hot


def apply_one_hot(df, one_hot):
    new_df = df.copy()  # copy input dataframe

    for col in new_df.columns:
        if (new_df[col].dtype == "object" or new_df[col].dtype == "category") and col != "CLASS" and col != "ID":
            for value in one_hot[col]:
                new_df[col + "_" + value] = (new_df[col] == value).astype(float)
            new_df = new_df.drop(col, axis=1)

    return new_df




#Approach one:
# SMILE Class
class SMILEActive:
    def __init__(self):
        column_filter = None
        imputation = None
        one_hot = None
        labels = None
        model = None

    def featureExtraction(self, df):
        #Involve features:
        # Chem
        # rdMolDescriptors:
        # Fragments:
        # Lipinski:
        df_copy = df.copy()
        df_feature = df.copy()

        # Adding columns for desired feature
        df_feature["NumAtoms"] = np.nan
        df_feature["MolWt"] = np.nan
        df_feature["AroRing"] = np.nan
        df_feature["AmideBond"] = np.nan
        df_feature["AL_COO"] = np.nan
        df_feature["HeavyAtom"] = np.nan

        # Loop through index, calculate and adding features to the feature dataframe
        for i in df_copy.index:
            m = Chem.MolFromSmiles(df_copy['SMILES'][i])
            numatoms = m.GetNumAtoms()  # numbers of atoms
            molwt = d.CalcExactMolWt(m)  # molecule's exact molecular weight
            aroring = d.CalcNumAromaticRings(m)  # number of aromatic rings for a molecule
            amidebond = d.CalcNumAmideBonds(m)  # number of amide bonds in a molecule
            alcoo = f.fr_Al_COO(m)  #Number of aliphatic carboxylic acids
            heavyatom = l.HeavyAtomCount(m)  #Number of heavy atoms a molecule

            df_feature.loc[i, 'NumAtoms'] = numatoms
            df_feature.loc[i, 'MolWt'] = molwt
            df_feature.loc[i, 'AroRing'] = aroring
            df_feature.loc[i, 'AmideBond'] = amidebond
            df_feature.loc[i, 'AL_COO'] = alcoo
            df_feature.loc[i, 'HeavyAtom'] = heavyatom


        # move ACTIVE to the end of the dataframe
        tempcolumn = df_feature.columns.tolist()
        activeindex = df_feature.columns.get_loc("ACTIVE")
        newcolumn = tempcolumn[0:activeindex] + tempcolumn[activeindex + 1:] + tempcolumn[activeindex:activeindex + 1]
        print(newcolumn)
        df_feature = df_feature[newcolumn]

        return df_feature


    def fit(self, df, no_trees=100):
        df_copy = df.copy()  # make a copy of the dataframe

        filtered_df, self.column_filter = column_filter(df)  # apply column filter
        df_temp, self.imputation = imputation(filtered_df)  # apply imputation
        new_df, self.one_hot = one_hot(df_temp)  # apply one-hot encoding

        training_labels = df["CLASS"].astype("category")
        self.labels = list(training_labels.cat.categories)  # get values of class labels
        # print(self.labels)

        # here we generate the random forest. Uses no_trees and df_onehot to generate a forest of trees
        random_forest = []  # list of trees
        for tree in range(no_trees):
            # generate the indices for the bootstrap sample
            rows = [idx for idx in range(len(new_df))]  # list of row indices
            dflength = len(new_df)  # number of instances in the bootstrap

            randomsamples = np.random.choice(rows, size=dflength,
                                             replace=True)  # generate indices of the bootstrap sample

            inputlabel = new_df[
                "CLASS"].values  # get class labels for the bootstrap sample as the values of the "CLASS" column
            inputlabel = inputlabel[randomsamples]  # get class labels for the bootstrap sample

            inputX = new_df.drop(columns=["CLASS"]).values  # get the instances for the bootstrap sample
            inputX = inputX[randomsamples, :]  # get the instances for the bootstrap sample

            # generate the tree
            clf = DecisionTreeClassifier(max_features=int(
                np.log2(inputX.shape[1])))  # with max_features as number of features to be evaluated in each node
            # print(bootstrap_instances)
            # bootstrap_instances_onehot = apply_one_hot(new_df, self.one_hot) #apply one-hot to the bootstrap instances
            clf.fit(inputX, inputlabel)  # fit the tree to the bootstrap sample
            random_forest.append(clf)  # add the generated tree to the forest

        self.model = random_forest

    def predict(self, df):
        df_copy = df.copy()  # copy the dataframe to a new dataframe (as done in Assignment 1 and 2)

        df_drop = df_copy.drop(["CLASS"],
                               axis=1)  # dropping the CLASS column (we can't drp ID because we don't have that column on this dataset)
        filtered_df = apply_column_filter(df_drop, self.column_filter)  # applying the column filter
        imputated_df = apply_imputation(filtered_df, self.imputation)  # applying the imputation
        onehot_df = apply_one_hot(imputated_df, self.one_hot)  # applying the one-hot
        input_data_values = onehot_df.values

        predictions = np.zeros((input_data_values.shape[0], len(self.labels)),
                               dtype="float64")  # list of predictions for each tree in the forest and list of average predictions for each tree in the forest
        num_trees = 0  # counter for the number of trees in the forest
        for clf in (self.model):  # iterating over the trees in the forest
            X = input_data_values
            # print("Shape of X: ", np.shape(X))
            # print("Shape of predictions: ", np.shape(predictions))
            # print("Shape of predictions2: ", np.shape(clf.predict_proba(X)))
            # print(predictions)

            # 2a main implementation part
            roughprediction = clf.predict_proba(X)
            treelabel = clf.classes_  # get the labels for tree
            df_prediction = pd.DataFrame(roughprediction,
                                         columns=treelabel)  # create a dataframe for tree output prediction, easier to add column

            for i in self.labels:
                if i in df_prediction.columns:
                    pass
                else:  # each class label that is not included
                    df_prediction.insert(self.labels.index(i), i, np.zeros(input_data_values.shape[0]),
                                         True)  # assigned zero probability, hint 1 and hint3
            # print("Shape of prediction: ", np.shape(predictions))
            # print("Shape of df_predictions.values: ", np.shape(df_prediction.values))
            predictions = predictions + df_prediction.values  # appending the predictions of each tree in the forest, change df back to ndarrays
            num_trees = num_trees + 1  # incrementing the number of trees in the forest

        predictions = predictions / num_trees  # dividing the predictions by the number of trees in the forest to get the average predictions
        predictions = pd.DataFrame(predictions,
                                   columns=self.labels)  # averaging the predictions of each tree in the forest

        return predictions




train_df = pd.read_csv("training_smiles.csv")

test_df = pd.read_csv("test_smiles.csv")

smile = SMILEActive()

t0 = time.perf_counter()
feature_df = smile.featureExtraction(train_df)
print(feature_df)
#smile.fit(feature_df)
print("Training time: {:.2f} s.".format(time.perf_counter()-t0))

