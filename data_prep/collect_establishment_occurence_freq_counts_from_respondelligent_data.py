#!/usr/bin/env python
# coding: utf-8

# In[33]:


import sys
import pandas as pd
import numpy as np
from collections import Counter


# In[3]:


in_data = sys.argv[1]
outfile = sys.argv[2]

df = pd.read_pickle(in_data)


# In[59]:


## plotting
# print(len(df.establishment.unique()))
# df.establishment.value_counts().plot(kind='barh', figsize=(16, 20), fontsize=10)
# print(df.establishment.value_counts())


# In[60]:


establishment_counts = df.establishment.value_counts()


# In[61]:


with open(outfile, 'w', encoding='utf8') as outf:
    for i, (k, v) in enumerate(establishment_counts.iteritems(), 1):
        print(k, v)
        outf.write(f"{k}\t{v}\t<{k.replace(' ', '_')}>\t<est_{str(i)}>\n")


# In[ ]:





# In[ ]:




