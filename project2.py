#!/usr/bin/env python
# coding: utf-8

# # Movie Recommendation System
# ## Web Information Management: Project II #
# In this project, I will develop different algorithms to make recommendations for movies.
# <hr>

# ### Data import and export functions:

# In[2]:


import numpy as np
import pandas as pd

UPDATE_INT = 25

def fetch_train():
    data = pd.read_csv('data/train.txt', delimiter='\t', header=None, names=np.arange(1, 1001), dtype=int)
    return data

def fetch_test(fn):
    data = pd.read_csv('data/'+fn, delimiter=' ', header=None, names=['U','M','R'], dtype=int)
    return data

def write_test(data, fn):
    data.to_csv('result/'+fn, sep=' ', header=False, index=False)
    print('> Results written to {}\n'.format(fn))


# ### Helper functions:

# In[12]:


def remove_zeros(a, b):
    assert len(a)==len(b), "{} != {}".format(len(a), len(b))
    ra = np.array([])
    rb = np.array([])
    for x1, x2 in zip(a,b):
        if x1 and x2:
            ra = np.append(ra, x1)
            rb = np.append(rb, x2)
    return ra, rb

def cos_sim(a, b):
    assert a.shape == b.shape, "{} != {}".format(a.shape, b.shape)
    if np.sum(b)==0:
        return 0
    
    # remove 0's
    ta, tb = remove_zeros(a, b)
    if len(ta)<2 or len(tb)<2:
        return 0
    
    # cosine similarity
    num = ta.dot(tb)
    den = np.linalg.norm(ta)*np.linalg.norm(tb)
    return num/den

def pea_cor(a, b):
    assert a.shape == b.shape, "{} != {}".format(a.shape, b.shape)
    
    # remove 0's
    ta, tb = remove_zeros(a, b)
    
    # remove 1 element arrays?
    if len(ta)<2 or len(tb)<2:
        return 0
    
    # subtract average
    ta = ta - np.mean(ta)
    tb = tb - np.mean(tb)
    
    # cosine similarity
    num = ta.dot(tb)
    den = np.linalg.norm(ta)*np.linalg.norm(tb)
    return (num/den) if den else 0

def weighted_avg(w, r, absval=False):
    assert w.shape == r.shape, "{} != {}".format(w.shape, r.shape)
    if np.sum(w) == 0:
        return 0
    if absval:
        return np.sum(w*r)/np.sum(np.absolute(w))
    return np.sum(w*r)/np.sum(w)

def count_col(arr, target, column):
    for t in target:
        count = arr[arr[column]==t].count()
        print("! {} appears {} times in column {}".format(t, count[column], column))
    return 0

def round_result(val, lower=1, upper=5):
    if val == 0:
        return (upper+lower)/2
    elif val < lower:
        return lower
    elif val > upper:
        return upper
    else:
        return round(val)

def constrain_array(arr, col, lower=1, upper=5):
    arr[arr[col]<lower] = lower
    arr[arr[col]>upper] = upper
    return arr
    


# ### Problem 2 Functions:
# Here, I implemented several user-based collaborative filtering algorithms, including modifications such as cosine similarity, Pearson correlation, inverse user frequency, and case modification.
# - Cosine Similarity
# - Pearson Correlation
# - Pearson Correlation w/ Inverse User Frequency
# - Pearson Correlation w/ Case Modification

# #### 1) Cosine Similarity
# > Using cosine similarity, the user similarity is calculated using the following formula:
# >
# > $
# \begin{align}
# \cos\theta = \frac{\vec{A} \cdot \vec{B}}{||\vec{A}|| \times ||\vec{B}||}
# \end{align}
# $

# In[17]:


## Cosine Similarity
def problem2_cs(testfile, outfile=None, k=None, t=0.8):
    
    # fetch training/testing data
    train = fetch_train()
    test = fetch_test(testfile)
    test['R2'] = 0.0
    
    # create rng, a list of users to solve for
    rng = (test.U.min(), test.U.max()+1)
    results = pd.DataFrame(columns=test.columns)
    
    # loop through each user
    for i in range(rng[0], rng[1]):
        
        # separate known and unknown ratings
        ratings = test[test.U==i]
        known = ratings[ratings.R!=0]
        unknown = ratings[ratings.R==0]
        
        # calculate USER similarity by comparing each rating R in 'known' against every other movie rating
        user_sim = train.apply(lambda x: cos_sim(known.R.values, x[known.M].values), axis=1)

        # rating prediction
        for j, r in unknown.iterrows():
            rs, rr = remove_zeros(user_sim, train[r.M])
            knn = pd.DataFrame({'S':rs, 'R':rr}).sort_values(by='S', ascending=False)
            knn = knn.iloc[:k]
            r.R2 = weighted_avg(knn['S'], knn['R'])
            r.R = round_result(r.R2)
            
            results = results.append(r, ignore_index=True)
        
        # print update
        if (i%UPDATE_INT == 0):
            print("User {} ({} predictions)...".format(i, len(unknown.M)))
    
    # remove 0's and 6's
    count_col(results, [0,6], 'R')
        
    if outfile:
        write_test(results[['U','M','R']].astype(int), outfile)
        
    return results

test01 = problem2_cs('test5.txt', 'result5_cs.txt')
test02 = problem2_cs('test10.txt', 'result10_cs.txt')
test03 = problem2_cs('test20.txt', 'result20_cs.txt')


# #### 2) Pearson Correlation
# > Using pearson correlation, the user similarity is calculated using the following formula:
# >
# > $
# \begin{align}
# \cos\theta = \frac{(A-\overline{A}) \cdot (B-\overline{B})}{||(A-\overline{A})|| \times ||(B-\overline{B})||}
# \end{align}
# $

# In[22]:


def problem2_pc(testfile, outfile=None, k=None):
    
    # fetch training/testing data
    train = fetch_train()
    test = fetch_test(testfile)
    test['R2'] = 0.0
    
    # create rng, a list of users to solve for
    rng = (test.U.min(), test.U.max()+1)
    results = pd.DataFrame(columns=test.columns)
    
    # loop through each user
    for i in range(rng[0], rng[1]):
        
        # separate known and unknown ratings
        ratings = test[test.U==i]
        known = ratings[ratings.R!=0]
        unknown = ratings[ratings.R==0]
        
        # calculate USER similarity by comparing each rating R in 'known' against every other movie rating
        user_sim = train.apply(lambda x: pea_cor(known.R.values, x[known.M].values), axis=1)
        
        avg_rating = np.mean(known.R)
        
        # rating prediction
        for j, r in unknown.iterrows():
            rs, rr = remove_zeros(user_sim, train[r.M])
            rr = rr - np.mean(rr)
            knn = pd.DataFrame({'S':rs, 'R':rr}).sort_values(by='S', ascending=False)
            knn = knn.iloc[:k]
            r.R = round_result(avg_rating + weighted_avg(knn['S'], knn['R'], True))
            
            results = results.append(r, ignore_index=True)
        
        # print update
        if (i%UPDATE_INT == 0):
            print("User {} ({} predictions)...".format(i, len(unknown.M)))
    
    # remove 0's and 6's
    count_col(results, [0,6], 'R')
        
    if outfile:
        write_test(results[['U','M','R']].astype(int), outfile)
        
    return results

test04 = problem2_pc('test5.txt', 'result5_pc.txt')
test05 = problem2_pc('test10.txt', 'result10_pc.txt')
test06 = problem2_pc('test20.txt', 'result20_pc.txt')


# #### 3) Pearson Correlation w/ Inverse User Frequency
# > Inverse user frequency uses a log function to apply a larger weight to "unpopular" movies, or movies with less ratings. The assumnption is that popular movies will all receive similar positive ratings, so less popular movies should have a larger impact on estimated ratings.
# >
# > $
# \begin{align}
# iuf(j) = \log{(\frac{m}{m_j})}
# \end{align}
# $

# In[21]:


def problem2_pciuf(testfile, outfile=None, k=None):
    
    # fetch training/testing data
    train = fetch_train()
    test = fetch_test(testfile)
    
    # compute iuf
    m = len(train)
    iuf = []
    for i,c in train.iteritems():
        mj = c[c!=0].count()
        iuf.append(np.log(m/mj) if mj else 0.0)
    train_iuf = train*iuf
    
    # create rng, a list of users to solve for
    rng = (test.U.min(), test.U.max()+1)
    results = pd.DataFrame(columns=test.columns)
    
    # loop through each user
    for i in range(rng[0], rng[1]):
        
        # separate known and unknown ratings
        ratings = test[test.U==i]
        known = ratings[ratings.R!=0]
        unknown = ratings[ratings.R==0]
        
        # calculate USER similarity by comparing each rating R in 'known' against every other movie rating
        user_sim = train_iuf.apply(lambda x: pea_cor(known.R.values, x[known.M].values), axis=1)
        
        avg_rating = np.mean(known.R)
        
        # rating prediction
        for j, r in unknown.iterrows():
            rs, rr = remove_zeros(user_sim, train[r.M])
            rr = rr - np.mean(rr)
            knn = pd.DataFrame({'S':rs, 'R':rr}).sort_values(by='S', ascending=False)
            knn = knn.iloc[:k]
            r.R = round_result(avg_rating + weighted_avg(knn['S'], knn['R'], True))
            
            results = results.append(r, ignore_index=True)
        
        # print update
        if (i%UPDATE_INT == 0):
            print("User {} ({} predictions)...".format(i, len(unknown.M)))
    
    # remove 0's and 6's
    count_col(results, [0,6], 'R')
        
    if outfile:
        write_test(results, outfile)
        
    return results

test07 = problem2_pciuf('test5.txt', 'result5_pciuf.txt')
test08 = problem2_pciuf('test10.txt', 'result10_pciuf.txt')
test09 = problem2_pciuf('test20.txt', 'result20_pciuf.txt')


# #### 4) Pearson Correlation w/ Case Modification
# > Case modification applies an exponent to each value in the similarity matrix after it is generated. I used the common value of 2.5, which will have little effect on high similarity values, while greatly reducing smaller similarity values.
# >
# > $
# \begin{align}
# w' = w \cdot |w^\rho|
# \end{align}
# $

# In[50]:


def problem2_pccm(testfile, outfile=None, k=None, p=2.5):
    
    # fetch training/testing data
    train = fetch_train()
    test = fetch_test(testfile)
    test['R2'] = 0.0
    
    # create rng, a list of users to solve for
    rng = (test.U.min(), test.U.max()+1)
    results = pd.DataFrame(columns=test.columns)
    
    # loop through each user
    for i in range(rng[0], rng[1]):
        
        # separate known and unknown ratings
        ratings = test[test.U==i]
        known = ratings[ratings.R!=0]
        unknown = ratings[ratings.R==0]
        
        # calculate USER similarity by comparing each rating R in 'known' against every other movie rating
        user_sim = train.apply(lambda x: pea_cor(known.R.values, x[known.M].values), axis=1)
        
        # apply case modification
        user_sim = user_sim * (user_sim ** p)
        
        avg_rating = np.mean(known.R)
        
        # rating prediction
        for j, r in unknown.iterrows():
            rs, rr = remove_zeros(user_sim, train[r.M])
            rr = rr - np.mean(rr)
            knn = pd.DataFrame({'S':rs, 'R':rr}).sort_values(by='S', ascending=False)
            knn = knn.iloc[:k]
            r.R = round_result(avg_rating + weighted_avg(knn['S'], knn['R'], True))
            
            results = results.append(r, ignore_index=True)
        
        # print update
        if (i%UPDATE_INT == 0):
            print("User {} ({} predictions)...".format(i, len(unknown.M)))
    
    # remove 0's and 6's
    count_col(results, [0,6], 'R')
        
    if outfile:
        write_test(results[['U','M','R']].astype(int), outfile)
        
    return results

test10 = problem2_pccm('test5.txt', 'result5_pccm.txt')
test11 = problem2_pccm('test10.txt', 'result10_pccm.txt')
test12 = problem2_pccm('test20.txt', 'result20_pccm.txt')


# ### Problem 3 Function:
# > Here I implemented a basic item-based collaborative filtering algorithm. This function simply generates a similarity matrix between different items (movies) rather than between users.

# In[22]:


def problem3(testfile, outfile=None, k=None, t=0.8):
    
    # fetch training/testing data
    train = fetch_train()
    test = fetch_test(testfile)
    test['R2'] = 0.0
    
    results = pd.DataFrame(columns=test.columns)
    unknown = test[test.R==0]
    ct = 0
    
    # iterate through all unranked movies
    for i,r in unknown.iterrows():
        known = test[(test.R!=0) & (test.U==r.U)]
        
        # generate and sort the ITEM similarity matrix (using cos_sim)
        item_sim = train[known.M].apply(lambda x: cos_sim(train[r.M], x), axis=0)
        rs, rr = remove_zeros(item_sim, known.R)
        knn = pd.DataFrame({'S':rs, 'R':rr}).sort_values(by='S', ascending=False)
        knn = knn.iloc[:k]
        
        # compute ratings
        r.R2 = weighted_avg(knn['S'], knn['R'])
        r.R = round_result(r.R2)
             
        results = results.append(r, ignore_index=True)
        
        # progress update
        ct = ct + 1
        if ct%1000 == 0:
            print("Completed {} iterations...".format(ct))
            
    count_col(results, [0,6], 'R')
        
    if outfile:
        write_test(results[['U','M','R']].astype(int), outfile)
    
    return results

test13 = problem3('test5.txt', 'result5_ibcf.txt')
test14 = problem3('test10.txt', 'result10_ibcf.txt')
test15 = problem3('test20.txt', 'result20_ibcf.txt')


# ### Problem 4 Function:
# > This is a personal algorithm I created to try to achieve better MAE than the previous methods. In this method, I used cosine similarity as a foundation, since I achieved the best results with it so far. Then, I took all cases where the similarity matrix was empty, and rather than fill with 0 (which gets constrained to 1), I filled these values with the average movie rating among all users. Failing this, I simply used the mean value of all ratings, 3.

# In[48]:


def problem4(testfile, outfile=None, k=None, t=0.8):
    
    # fetch training/testing data
    train = fetch_train()
    test = fetch_test(testfile)
    
    # create rng, a list of users to solve for
    rng = (test.U.min(), test.U.max()+1)
    results = pd.DataFrame(columns=test.columns)
    
    # loop through each user
    for i in range(rng[0], rng[1]):
        
        # separate known and unknown ratings
        ratings = test[test.U==i]
        known = ratings[ratings.R!=0]
        unknown = ratings[ratings.R==0]
        
        # calculate USER similarity by comparing each rating R in 'known' against every other movie rating
        user_sim = train.apply(lambda x: cos_sim(known.R.values, x[known.M].values), axis=1)

        # rating prediction
        for j, r in unknown.iterrows():
            rs, rr = remove_zeros(user_sim, train[r.M])
            knn = pd.DataFrame({'S':rs, 'R':rr}).sort_values(by='S', ascending=False)
            knn = knn.iloc[:k]
            r.R = round(weighted_avg(knn['S'], knn['R']))
            
            if r.R == 0:
                all_rat = train[r.M]
                r.R = round(np.mean(all_rat[all_rat>0])) if (all_rat != 0).any() else 3
            
            results = results.append(r, ignore_index=True)
        
        # print update
        if (i%UPDATE_INT == 0):
            print("User {} ({} predictions)...".format(i, len(unknown.M)))
    
    count_col(results, [0,6], 'R')
        
    if outfile:
        write_test(results[['U','M','R']].astype(int), outfile)
        
    return results

test16 = problem4('test5.txt', 'result5_me.txt')
test17 = problem4('test10.txt', 'result10_me.txt')
test18 = problem4('test20.txt', 'result20_me.txt')


# PIN: 9423572820721098
