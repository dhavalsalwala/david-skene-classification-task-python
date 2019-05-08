import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

class DawidSkene:

    #Get input data source from csv generated via train_mv.py
    def get_input_data_source(self):
        ds_input = pd.read_csv("data/input_ds.csv", delimiter=",")
        ds_input = ds_input.loc[:, ['id', 'annotator', 'class']]
        ds_input.columns = ['movieid', 'workerid', 'class']
        return ds_input
        #print (ds_input.head())

    # Get mturk sample (along with annotator id and class) data source from csv generated via train_mv.py
    def get_polarity_data_source(self):
        mturk_sample_final = pd.read_csv("data/mturk_sample_ds.csv", delimiter=",")
        ds_pol = mturk_sample_final.loc[:, ['id', 'class']]
        ds_pol['neg'] = np.where(ds_pol['class'] == 1, 0, 1)
        ds_pol.columns = ['movieid', 'pos', 'neg']
        cols = ['pos', 'neg']
        ds_pol[cols] = ds_pol[cols].apply(pd.to_numeric, errors='coerce', axis=1)
        ds_pol = ds_pol.astype({"movieid": int, "pos": float, "neg": float})
        return ds_pol
        #print (ds_pol.head())

    #Get mturk sample data source to prepare worker accuracy matrix.
    def get_accuracy_data_set(self):
        df = pd.DataFrame(columns=["MoiveID", "Worker", "TP", "TN", "FP", "FN"])
        ds = pd.read_csv('data/mturk_sample.csv')
        movie_ids = list(ds['id'])
        workers = list(ds['annotator'])
        df['MoiveID'] = movie_ids
        df['Worker'] = workers
        df['TP'] = 0
        df['TN'] = 0
        df['FP'] = 0
        df['FN'] = 0
        return df
        #print (df.head())

    @staticmethod
    def convert_class_into_numerical(ds):
        ds.loc[ds['class'] == "pos", "class"] = 1
        ds.loc[ds['class'] == "neg", "class"] = 0
        return ds

    #Calculate and display accuracy
    def evaluate_performance(self, y_test, y_pred, output_file_name):

        y_pred = np.array(y_pred)
        y_test = np.array(y_test)

        d = {'Actual Value': y_test, 'Predicted Value': y_pred}
        df = pd.DataFrame(data=d)

        f = open("data/"+output_file_name, "w+")

        df_confusion = self._confusion_matrix(y_test, y_pred)
        f.write("The confusion matrix:\n\n")
        f.write(str(df_confusion))
        TP = df_confusion.iloc[0][0]
        TN = df_confusion.iloc[1][1]
        FP = df_confusion.iloc[0][1]
        FN = df_confusion.iloc[1][0]
        f.write("\n\nAccuracy: %.2f" % ((TP + TN) / (TP + TN + FP + FN)))
        f.write("\nPrecision: %.2f" % (TP / (TP + FP)))
        f.write("\nF-score: %.2f" % (2 / ((1 / (TP / (TP + FP))) + (1 / (TP / (TP + FN))))))
        FPR, TPR, threshold = metrics.roc_curve(y_test, y_pred)
        f.write('\nArea Under Curve = %0.2f' % metrics.auc(FPR, TPR))
        f.write('\nprediction probabilities:\n')
        f.write(df.to_string(index=False))
        f.close()

        fig = plt.figure()
        plt.title('Model training using CrowdSourced:Amazon mturks\n data with Majority Voting')
        plt.plot(FPR, TPR, 'b', label='AUC = %0.2f' % metrics.auc(FPR, TPR))
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        fig.savefig('data/'+'David_Skene.png', dpi=fig.dpi)

    @staticmethod
    def _confusion_matrix(y_test, y_pred):
        y_test = pd.Series(y_test, name='Gold')
        y_pred = pd.Series(y_pred, name='Predicted')
        return pd.crosstab(y_test, y_pred, margins=True)

#Initialise class object
ds=DawidSkene()

#Get all data sources from using class object.
input_ds=ds.get_input_data_source()
pol_ds=ds.get_polarity_data_source()
accuracy_ds=ds.get_accuracy_data_set()

# unique movie ids
movId = input_ds['movieid'].unique().tolist()
mw = dict.fromkeys(movId, [])

#Prepare dictionary with key=movieid and value=list of workerids.
rows = []
for k in mw.keys():
    s=input_ds[input_ds['movieid']==k]['workerid'].values.tolist()
    mw[k]=s

#No of iterations = 50
for i in range(50):

    print ("Iteration "+str(i+1))

    # Iterating over movieId dictionary
    for k,v in mw.items():
        for worker in v:
            sub_ip = input_ds[(input_ds.movieid==k) & (input_ds.workerid==worker)]
            sub_pol = pol_ds[pol_ds.movieid==k]
            pol = sub_pol.iloc[0]["pos"]>sub_pol.iloc[0]["neg"]
            sub_df = accuracy_ds[(accuracy_ds.MoiveID==k) & (accuracy_ds.Worker==worker)]
            idx = accuracy_ds.index[(accuracy_ds.MoiveID==k) & (accuracy_ds.Worker==worker)].tolist()[0]

            #If class type in the input data set is positive and true polarity is also positive then update the existing accuracy.
            if(sub_ip.iloc[0]["class"]==1.0 and pol):
                tmp =  accuracy_ds.iloc[idx]["TP"]
                accuracy_ds.set_value(idx,"TP", tmp+1)
            # If class type in the input data set is positive and true polarity is negative then update the existing accuracy.
            elif (sub_ip.iloc[0]["class"]==1.0 and not pol):
                tmp =  accuracy_ds.iloc[idx]["FN"]
                accuracy_ds.set_value(idx,"FN",tmp+1)
            # If class type in the input data set is negative and true polarity is positive then update the existing accuracy.
            elif (sub_ip.iloc[0]["class"]==0.0 and pol):
                tmp =  accuracy_ds.iloc[idx]["TN"]
                accuracy_ds.set_value(idx,"TN",tmp+1)
            # If class type in the input data set is negative and true polarity is also negative then update the existing accuracy.
            elif(sub_ip.iloc[0]["class"]==0.0 and not pol):
                tmp =  accuracy_ds.iloc[idx]["FP"]
                accuracy_ds.set_value(idx,"FP",tmp+1)

    pos_sum = np.sum(pol_ds["pos"].values)
    neg_sum = np.sum(pol_ds["neg"].values)

    #compute aggregation of accuracy against true polarities
    accuracy_ds["TP"] = accuracy_ds["TP"]/pos_sum
    accuracy_ds["FP"] = accuracy_ds["FP"]/pos_sum
    accuracy_ds["TN"] = accuracy_ds["TN"]/neg_sum
    accuracy_ds["FN"] = accuracy_ds["FN"]/neg_sum

    movie_ids = pol_ds.iloc[:,0].tolist()

    #Iterating over movie ids
    for movieid in movie_ids:
        input_sub_df = input_ds[input_ds['movieid'] == movieid]
        row_index = pol_ds.index[pol_ds['movieid'] == movieid].tolist()[0]

        for index, row in input_sub_df.iterrows():
            accur_sub_ds = accuracy_ds.loc[(accuracy_ds['MoiveID'] == movieid) & (accuracy_ds['Worker'] == row['workerid'])]

            # If class type in the input data set is positive then update the true polarity by comparing with accuracy data source.
            if row['class'] == 1:
                pos_sum = pol_ds.iloc[row_index]['pos']
                neg_sum = pol_ds.iloc[row_index]['neg']
                pol_ds.set_value(row_index, 'pos', pos_sum + accur_sub_ds.iloc[0]['TP'])
                pol_ds.set_value(row_index, 'neg', neg_sum + accur_sub_ds.iloc[0]['FN'])

            # If class type in the input data set is negative then update the true polarity by comparing with accuracy data source.
            if row['class'] == 0:
                pos_sum = pol_ds.iloc[row_index]['pos']
                neg_sum = pol_ds.iloc[row_index]['neg']
                pol_ds.set_value(row_index, 'pos', pos_sum + accur_sub_ds.iloc[0]['FP'])
                pol_ds.set_value(row_index, 'neg', neg_sum + accur_sub_ds.iloc[0]['TN'])

    #Compute average of true polarity movieid wise.
    updated_ds_pol=pol_ds
    for index, row in pol_ds.iterrows():
        sum = row['pos']+row['neg']
        updated_ds_pol.set_value(index, 'pos', row['pos']/sum)
        updated_ds_pol.set_value(index, 'neg', row['neg']/sum)

    pol_ds=updated_ds_pol

# Training and testing our data set with Decision Tree Classifier
data=pd.read_csv("data/mturk_sample_ds.csv", delimiter=",")
updated_data=data
for index, row in data.iterrows():
    pol=pol_ds[pol_ds['movieid']==row['id']]
    updated_data.set_value(index, 'class', np.where(pol.iloc[0]['pos'] >= pol.iloc[0]['neg'], 1, 0))

data=updated_data
data=data.drop('id', axis=1)

#Getting test.csv for testing on trained model
test = pd.read_csv("data/test.csv")
test = ds.convert_class_into_numerical(test)

test = test.drop('id', axis=1)

# creating instance of ID3 Classifier with parameters
dtree = DecisionTreeClassifier(criterion='entropy', random_state=100, max_depth=100)

X_train = data.drop('class', axis=1)
y_train = data['class']

X_test = test.drop('class', axis=1)
y_test = test['class']

dtree.fit(X_train, y_train)  # fit model to data set
y_pred = dtree.predict(X_test)  # predict for outcomes

ds.evaluate_performance(y_test, y_pred, output_file_name='train_ds.txt')
