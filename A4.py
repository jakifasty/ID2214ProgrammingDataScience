import numpy as np
import pandas as pd
import time

# rklearn
import sklearn
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics


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


def auc_score(df, correctlabels):
    df_temp = df.copy()
    values = {}
    auc_percolumn = 0
    df_truepositive = pd.DataFrame(np.zeros((len(df_temp), len(df_temp.columns))), columns=df_temp.columns)
    for i in range(len(correctlabels)):
        df_truepositive.loc[i, correctlabels[i]] = 1
    # print("df_truepositive: ", df_truepositive)

    for column in df_temp:
        columnseriesobj = df_temp[column]
        columntruepositive = df_truepositive[column]
        columnfalsepositive = columntruepositive.copy()
        for i in range(len(columntruepositive)):
            if columntruepositive[i] == 0:
                columnfalsepositive[i] = 1
            elif columntruepositive[i] == 1:
                columnfalsepositive[i] = 0
        df_auc_temp = pd.DataFrame({"s": columnseriesobj, "tp": columntruepositive, "fp": columnfalsepositive})

        agg_functions = {'tp': 'sum', 'fp': 'sum'}
        df_auc_temp = df_auc_temp.groupby(df_auc_temp['s']).aggregate(agg_functions)
        df_auc_temp = df_auc_temp.sort_values(by='s', ascending=False)

        # return auc_score
        AUC = 0
        cov_tp = 0
        tot_tp = np.sum(columntruepositive)
        tot_fp = np.sum(columnfalsepositive)

        for idx in df_auc_temp.index:
            if df_auc_temp["fp"][idx] == 0:
                cov_tp += df_auc_temp["tp"][idx]
            elif df_auc_temp["tp"][idx] == 0:
                AUC += (cov_tp / tot_tp) * (df_auc_temp["fp"][idx] / tot_fp)
            else:
                AUC += (cov_tp / tot_tp) * (df_auc_temp["fp"][idx] / tot_fp) + (df_auc_temp["tp"][idx] / tot_tp) * (
                        df_auc_temp["fp"][idx] / tot_fp) / 2
                cov_tp += df_auc_temp["tp"][idx]

        # AUC = metrics.roc_auc_score(columntruepositive, columnseriesobj)
        values[column] = AUC

    auc_score = 0
    for i in values:
        count = 0
        for output in correctlabels:
            if i == output:
                count += 1
        frequency = count / len(correctlabels)
        auc_score += values[i] * frequency
    return auc_score









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
        # basic info
        df_feature["NumAtoms"] = np.nan
        df_feature["MolWt"] = np.nan
        #df_feature["HeavyAtom"] = np.nan
        # specific features or fragment
        df_feature["AroRing"] = np.nan
        df_feature["AmideBond"] = np.nan
        df_feature["RotatableBond"] = np.nan
        df_feature["SaturatedRing"] = np.nan
        df_feature["AL_COO"] = np.nan
        df_feature["Benzene"] = np.nan

        # Loop through index, calculate and adding features to the feature dataframe
        for i in df_copy.index:
            m = Chem.MolFromSmiles(df_copy['SMILES'][i])
            numatoms = m.GetNumAtoms()  # numbers of atoms
            molwt = d.CalcExactMolWt(m)  # molecule's exact molecular weight
            heavyatom = l.HeavyAtomCount(m)  # Number of heavy atoms a molecule
            aroring = d.CalcNumAromaticRings(m)  # number of aromatic rings for a molecule, they are very stable and do not break apart easily
            amidebond = d.CalcNumAmideBonds(m)  # number of amide bonds in a molecule
            rotabond= d.CalcNumRotatableBonds(m)  # number of rotatable bonds for a molecule
            saturatedring = d.CalcNumSaturatedRings(m)  # returns the number of saturated rings for a molecule
            alcoo = f.fr_Al_COO(m)  # Number of aliphatic carboxylic acids
            benzene = f.fr_benzene(m) # Number of benzene rings

            df_feature.loc[i, 'NumAtoms'] = numatoms
            df_feature.loc[i, 'MolWt'] = molwt
            df_feature.loc[i, 'HeavyAtom'] = heavyatom
            df_feature.loc[i, 'AroRing'] = aroring
            df_feature.loc[i, 'AmideBond'] = amidebond
            df_feature.loc[i, 'RotatableBond'] = rotabond
            df_feature.loc[i, 'SaturatedRing'] = saturatedring
            df_feature.loc[i, 'AL_COO'] = alcoo
            df_feature.loc[i, 'Benzene'] = benzene


        # move ACTIVE to the end of the dataframe
        tempcolumn = df_feature.columns.tolist()
        activeindex = df_feature.columns.get_loc("ACTIVE")
        newcolumn = tempcolumn[0:activeindex] + tempcolumn[activeindex + 1:] + tempcolumn[activeindex:activeindex + 1]
        print(newcolumn)
        df_feature = df_feature[newcolumn]

        return df_feature


    def preprocess(self, df):
        df_copy = df.copy()  # make a copy of the dataframe



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


def split(df):
    df_copy = df.copy()
    y_total = df_copy["ACTIVE"].values
    x_total = df_copy.drop(columns=["ACTIVE", "INDEX", "HeavyAtom", "SMILES"]).values
    x_train, x_val, y_train, y_val = model_selection.train_test_split(x_total, y_total, test_size=0.2, random_state=1)
    return x_train, x_val, y_train, y_val

def split_skf(df):
    df_copy = df.copy()
    y_total = df_copy["ACTIVE"].values
    x_total = df_copy.drop(columns=["ACTIVE", "INDEX", "HeavyAtom", "SMILES"]).values
    skf = model_selection.StratifiedKFold(n_splits = 5)
    skf.get_n_splits(x_total, y_total)
    return skf, x_total, y_total


def mlp(x_train, x_val,  y_train, y_val):
    # hyper parameter list
    max_iter = [10, 20, 30, 40]
    hidden_layer_sizes = [(10,), (15,), (20,), (25,) ]
    solver = ['sgd', 'adam']
    activation = ['relu', 'identity', 'logistic', 'tanh']
    learning_rate = ['constant', 'adaptive']
    param_grid = {'max_iter': max_iter, 'hidden_layer_sizes': hidden_layer_sizes, 'solver': solver, 'activation': activation, 'learning_rate': learning_rate}
    mlp = MLPClassifier()
    clf = GridSearchCV(estimator=mlp, param_grid=param_grid,
                       scoring='roc_auc', n_jobs=4)
    clf.fit(x_train, y_train)
    print("Best Score: ")
    print(clf.best_score_)
    print("Best Estimator: ")
    print(clf.best_estimator_)
    best_params = clf.best_params_
    model = MLPClassifier(**best_params)
    model.fit(x_train, y_train)
    prediction = model.predict_proba(x_val)
    auc = metrics.roc_auc_score(y_val, prediction[:, 1])
    score = model.score(x_val, y_val)
    print("auc:", auc)
    print("score:", score)

    return model, auc, prediction, score


train_df = pd.read_csv("training_smiles.csv")

test_df = pd.read_csv("test_smiles.csv")

smile = SMILEActive()

t0 = time.perf_counter()
feature_df = smile.featureExtraction(train_df)
#print(feature_df)
#x_train, x_val, y_train, y_val = split(feature_df)
skf, x_total,y_total = split_skf(feature_df)
for i, (train_index, val_index) in enumerate(skf.split(x_total, y_total)):
    print(f"Fold {i}:")
    x_train, x_val = x_total[train_index], x_total[val_index]
    y_train, y_val = y_total[train_index], y_total[val_index]
    model, auc, prediction, score = mlp(x_train, x_val, y_train, y_val)


#model, auc, prediction, score = mlp(x_train, x_val, y_train, y_val)

print("Training time: {:.2f} s.".format(time.perf_counter()-t0))


