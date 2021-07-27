
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv(r'C:\Users\yu_zhou\Desktop\earphone_sentiment.csv')
subject = dict(Counter(list(data['subject'])))
content = data['content']
sentiment_word = data['sentiment_word']
sentiment_value = data['sentiment_value']

# labels = subject.keys()
# x = np.arange(0,7,1)
# y = subject.values()
#
# fig, ax = plt.subplots()
#
# reacts = ax.bar(x,y)
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.set_xlabel('我是')
# ax.bar_label(reacts,padding=3)

# plt.figure(figsize=(9,9))
# sns.countplot('subject',data=data)


# plt.show()
import pandas as pd
df = pd.DataFrame([  
            ['green' , 'A'],   
            ['red'   , 'B'],   
            ['blue'  , 'A']])  

df.columns = ['color',  'class'] 
print(pd.get_dummies(df))
a= 0
b= 1
c = 10 if b==2 else 20
print(c)
