# Julian Velez
# CSC 481: Artificial Intelligence
#
# Predicting Country Happiness Using Global Development Indicators

from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import math

happy = []
unhappy = []
test = []

# read data into a happy list, unhappy list, and test list
def init_data():
    for i in range(1,16):
        happy.append(np.genfromtxt("DataFiles/WDI/Happy/happy"+str(i)+".csv"))
        unhappy.append(np.genfromtxt("DataFiles/WDI/Unhappy/unhappy"+str(i)+".csv"))
        test.append(np.genfromtxt("DataFiles/WDI/Test/test"+str(i)+".csv"))
   
    # normalize the data
    norm = []
    for i in range(0,15):
        for j in range(0,15):
            norm.append(happy[j][i])
        for k in range(0,15):
            norm.append(unhappy[k][i])
        for l in range(0, 15):
            norm.append(test[l][i])
        normd_val = preprocessing.normalize(norm)
        
        for m in range(0,45):
            if m < 15:
                happy[m][i] = normd_val[0][m]
            elif m < 30:
                unhappy[m-15][i] = normd_val[0][m]
            else:
                test[m-30][i] = normd_val[0][m]
        del norm[:]

def init_data_edu():
    del happy[:]
    del unhappy[:]
    del test[:]

    for i in range(1,16):
        happy.append(np.genfromtxt("DataFiles/WDI/Happy/happy"+str(i)+"edu.csv"))
        unhappy.append(np.genfromtxt("DataFiles/WDI/Unhappy/unhappy"+str(i)+"edu.csv"))
        test.append(np.genfromtxt("DataFiles/WDI/Test/test"+str(i)+"edu.csv"))
   
    # normalize the data
    norm = []
    for i in range(0,4):
        for j in range(0,15):
            norm.append(happy[j][i])
        for k in range(0,15):
            norm.append(unhappy[k][i])
        for l in range(0, 15):
            norm.append(test[l][i])
        normd_val = preprocessing.normalize(norm)
        
        for m in range(0,45):
            if m < 15:
                happy[m][i] = normd_val[0][m]
            elif m < 30:
                unhappy[m-15][i] = normd_val[0][m]
            else:
                test[m-30][i] = normd_val[0][m]
        del norm[:]

def init_data_urb():
    del happy[:]
    del unhappy[:]
    del test[:]

    for i in range(1,16):
        happy.append(np.genfromtxt("DataFiles/WDI/Happy/happy"+str(i)+"urb.csv"))
        unhappy.append(np.genfromtxt("DataFiles/WDI/Unhappy/unhappy"+str(i)+"urb.csv"))
        test.append(np.genfromtxt("DataFiles/WDI/Test/test"+str(i)+"urb.csv"))
   
    # normalize the data
    norm = []
    for i in range(0,6):
        for j in range(0,15):
            norm.append(happy[j][i])
        for k in range(0,15):
            norm.append(unhappy[k][i])
        for l in range(0, 15):
            norm.append(test[l][i])
        normd_val = preprocessing.normalize(norm)
        
        for m in range(0,45):
            if m < 15:
                happy[m][i] = normd_val[0][m]
            elif m < 30:
                unhappy[m-15][i] = normd_val[0][m]
            else:
                test[m-30][i] = normd_val[0][m]
        del norm[:]

def init_data_tech():
    del happy[:]
    del unhappy[:]
    del test[:]

    for i in range(1,16):
        happy.append(np.genfromtxt("DataFiles/WDI/Happy/happy"+str(i)+"tech.csv"))
        unhappy.append(np.genfromtxt("DataFiles/WDI/Unhappy/unhappy"+str(i)+"tech.csv"))
        test.append(np.genfromtxt("DataFiles/WDI/Test/test"+str(i)+"tech.csv"))
   
    # normalize the data
    norm = []
    for i in range(0,5):
        for j in range(0,15):
            norm.append(happy[j][i])
        for k in range(0,15):
            norm.append(unhappy[k][i])
        for l in range(0, 15):
            norm.append(test[l][i])
        normd_val = preprocessing.normalize(norm)
        
        for m in range(0,45):
            if m < 15:
                happy[m][i] = normd_val[0][m]
            elif m < 30:
                unhappy[m-15][i] = normd_val[0][m]
            else:
                test[m-30][i] = normd_val[0][m]
        del norm[:]

# run a knn algorithm
def knn(list1, list2, list3):
    train = []
    for i in range(len(list1)):
        train.append(list1[i])
    train.extend(list2)
    test = list3
    y = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,]
    neigh = KNeighborsClassifier(n_neighbors=6)
    neigh.fit(train, y)

    x = neigh.predict(test)

    return x

# run a kmeans algorithm
def kmean(list1, list2, list3):
    train = []
    for i in range(len(list1)):
        train.append(list1[i])
    train.extend(list2)
    test = list3
    cluster = KMeans(n_clusters=2, random_state=0).fit(train)

    x = cluster.predict(test)

    return x

# run a classification neural network
def nn_net(list1, list2, list3):
    train = []
    for i in range(len(list1)):
        train.append(list1[i])
    train.extend(list2)
    test = list3
    y = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    net = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=0)
    net.fit(train, y)

    x = net.predict(test)

    return x

# plotting results 
def plot(actual, predicted):
    fpr, tpr, thresholds = roc_curve(actual,predicted)
    roc_auc = auc(fpr,tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b',
    label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

# main function
def main():
    # initalize our data and present the true class of our test values
    init_data()
    actual = [0,1,0,0,1,1,1,0,0,0,0,0,0,0,1]

    # test k-nearest neighbor
    knn_result = knn(happy, unhappy, test)
    knn_conf_matrix = confusion_matrix(actual,knn_result)
    print(knn_result)
    print(knn_conf_matrix)
    plot(actual,knn_result)

    # test k-means
    kmean_result = kmean(happy, unhappy, test)
    kmean_conf_matrix = confusion_matrix(actual,kmean_result)
    print(kmean_result)
    print(kmean_conf_matrix)
    plot(actual,kmean_result)

    # test neural network
    nn_result = nn_net(happy, unhappy, test)
    nn_conf_matrix = confusion_matrix(actual,nn_result)
    print(nn_result)
    print(nn_conf_matrix)
    plot(actual,nn_result)

    # test only the edu features
    init_data_edu()
    knn_result = knn(happy, unhappy, test)
    knn_conf_matrix = confusion_matrix(actual,knn_result)
    print(knn_result)
    print(knn_conf_matrix)
    plot(actual,knn_result)
    kmean_result = kmean(happy, unhappy, test)
    kmean_conf_matrix = confusion_matrix(actual,kmean_result)
    print(kmean_result)
    print(kmean_conf_matrix)
    plot(actual,kmean_result)
    nn_result = nn_net(happy, unhappy, test)
    nn_conf_matrix = confusion_matrix(actual,nn_result)
    print(nn_result)
    print(nn_conf_matrix)
    plot(actual,nn_result)

    # test only the urb features
    init_data_urb()
    knn_result = knn(happy, unhappy, test)
    knn_conf_matrix = confusion_matrix(actual,knn_result)
    print(knn_result)
    print(knn_conf_matrix)
    plot(actual,knn_result)
    kmean_result = kmean(happy, unhappy, test)
    kmean_conf_matrix = confusion_matrix(actual,kmean_result)
    print(kmean_result)
    print(kmean_conf_matrix)
    plot(actual,kmean_result)
    nn_result = nn_net(happy, unhappy, test)
    nn_conf_matrix = confusion_matrix(actual,nn_result)
    print(nn_result)
    print(nn_conf_matrix)
    plot(actual,nn_result)

    # test only the tech features
    init_data_tech()
    knn_result = knn(happy, unhappy, test)
    knn_conf_matrix = confusion_matrix(actual,knn_result)
    print(knn_result)
    print(knn_conf_matrix)
    plot(actual,knn_result)
    kmean_result = kmean(happy, unhappy, test)
    kmean_conf_matrix = confusion_matrix(actual,kmean_result)
    print(kmean_result)
    print(kmean_conf_matrix)
    plot(actual,kmean_result)
    nn_result = nn_net(happy, unhappy, test)
    nn_conf_matrix = confusion_matrix(actual,nn_result)
    print(nn_result)
    print(nn_conf_matrix)
    plot(actual,nn_result)

main()