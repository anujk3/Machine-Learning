
# coding: utf-8

# # Home Work-5: Machine Learning
# -Vinayak Bakshi

# In[432]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as nplg
import scipy.sparse.linalg
get_ipython().magic(u'matplotlib inline')


# In[433]:

# Read all files
scores_df = pd.read_csv("CFB2016_scores.csv",delimiter = ",",header = None,names = ["A_index", "A_points", "B_index", "B_points"])
scores_mat = np.genfromtxt("CFB2016_scores.csv", delimiter = ",")
teams_df = pd.read_table("TeamNames.txt",header = None, names=["Team_Name"])


# In[434]:

# Initialize M matrix

M = np.zeros((teams_df.shape[0],teams_df.shape[0]))

# Create M 
for i in xrange(scores_mat.shape[0]):
    
    M[int(scores_mat[i][0])-1][int(scores_mat[i][0])-1] = M[int(scores_mat[i][0])-1][int(scores_mat[i][0])-1] + int(scores_mat[i][1] > scores_mat[i][3]) + (1.0*scores_mat[i][1]/(scores_mat[i][1] + scores_mat[i][3]))
    M[int(scores_mat[i][2])-1][int(scores_mat[i][2])-1] = M[int(scores_mat[i][2])-1][int(scores_mat[i][2])-1] + int(scores_mat[i][1] < scores_mat[i][3]) + (1.0*scores_mat[i][3]/(scores_mat[i][1] + scores_mat[i][3]))
    M[int(scores_mat[i][0])-1][int(scores_mat[i][2])-1] = M[int(scores_mat[i][0])-1][int(scores_mat[i][2])-1] + int(scores_mat[i][1] < scores_mat[i][3]) + (1.0*scores_mat[i][3]/(scores_mat[i][1] + scores_mat[i][3]))
    M[int(scores_mat[i][2])-1][int(scores_mat[i][0])-1] = M[int(scores_mat[i][2])-1][int(scores_mat[i][0])-1] + int(scores_mat[i][1] > scores_mat[i][3]) + (1.0*scores_mat[i][1]/(scores_mat[i][1] + scores_mat[i][3]))

# Normalize M
for i in xrange(M.shape[0]):
    M[i] = M[i]/np.sum(M,axis = 1)[i]
    
# Find states
w_all = []
w_10000 = [] 

for t in [10,100,1000,10000]:
    
    w = (1.0/760)*np.ones(760)
    
    if t == 10000:
        for i in xrange(t):
            w = w.dot(M)
            w_10000.append(w)
        w_all.append(w)
        
    else:
        for i in xrange(t):
            w=w.dot(M)
        w_all.append(w)


# ## Solution 1(a)

# In[435]:

# Get top Team names
Results_df = pd.DataFrame()
for i in range(4): 
    Results_df = pd.concat([Results_df,teams_df.iloc[np.argsort(w_all[i])[::-1][:25],:].reset_index().Team_Name],axis = 1)
    Results_df = pd.concat([Results_df,pd.Series(w_all[i][np.argsort(w_all[i])[::-1][:25]])],axis = 1)

Results_df.columns = ['t=10','weight','t=100','weight','t=1000','weight','t=10000','weight']
Results_df.index+=1
Results_df


# ## Solution 1(b)

# In[436]:

# Computing Eigen Vectors

w_inf = scipy.sparse.linalg.eigs(M.T,k=1,sigma=1.0)[1]
w_inf = w_inf/np.sum(w_inf)
w_inf_rep = np.tile(w_inf.reshape(760),(10000,1))
norms = nplg.norm(w_10000 - w_inf_rep,1,axis = 1)

_=plt.figure(figsize = (10,8))
_=plt.plot(range(10000),norms)
_=plt.xlim([-100,10100])
_=plt.title("Variation of $|w_\infty - w_t |_1$ with t", fontsize = 18)
_=plt.xlabel("t",fontsize = 14)
_=plt.ylabel("$|w_\infty - w_t |_1$",fontsize = 14)


# ## Solution - 2

# In[437]:

# Parse File
file = open("nyt_data.txt","r") 
X = np.zeros((3012,8447))
c1 =0

for line in file.readlines():
    curr_line = line.split(",")
    
    for ele in curr_line:
        X[int(ele.split(":")[0])-1][c1] = ele.split(":")[1]
    c1 += 1


# In[438]:

# Function to calculate objective function
def objective():
    err = (10**(-16))
    WH = W.dot(H)
    I = np.ones(WH.shape)
    obj = X*np.log(I/(WH + err)) + WH
    return np.sum(obj)


# In[439]:

#Initialize W and H
W = []
H= []

for i in range(X.shape[0]):
    W.append(np.random.uniform(1,2,25))
W = np.array(W)

for i in range(X.shape[1]):
    H.append(np.random.uniform(1,2,25))
H = (np.array(H)).T

obj_all=[]
err = (10**(-16))
for iter in range(100):
    H = H * ((W.T/np.sum(W.T,axis =1)[:,np.newaxis]).dot(X/(W.dot(H)+err)))
    W = W *((X/(W.dot(H)+err)).dot(H.T/np.sum(H.T,axis =0)[np.newaxis,:]))
    obj_all.append(objective())    


# ## Solution 2(a)

# In[440]:

_=plt.figure(figsize = (10,8))
_=plt.title("Variation of Divergence Penalty with itertions", fontsize = 18)
_=plt.xlabel("iter",fontsize = 14)
_=plt.ylabel("Divergence Penalty",fontsize = 14)
_=plt.plot(range(100),obj_all)


# ## Solution 2 (b)

# In[441]:

W_norm = (W/np.sum(W,axis =0)[np.newaxis,:])

# Get Top Weights
weights_ind = np.around(np.sort(W_norm,axis =0)[-10:,:].T,decimals = 6)

# Get Top weighted Words' index
words_ind = np.argsort(W_norm,axis = 0)[-10:,:].T

# Read vocab
vocab = pd.read_table("nyt_vocab.dat",delimiter = ",",header = None)


# In[442]:

# Create 5 x 5 matrix of topics
mat_topics = [] 
temp = []
for i in range(25):
    b = list(np.fliplr(weights_ind)[i])
    c = list((np.char.array(vocab.iloc[np.fliplr(words_ind)[i],:])).reshape(10,))
    topic = [m+'-'+str(n) for m,n in zip(c,b)]
    temp.append(','.join(topic))
    if len(temp)==5:
        mat_topics.append(temp)
        temp = []

(pd.DataFrame(np.array(mat_topics)))

