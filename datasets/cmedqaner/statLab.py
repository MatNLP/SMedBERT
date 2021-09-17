import pandas as pd
from collections import Counter


table=pd.read_csv('new_dev.txt',sep=' ',header=None)
stat_list=[ i if len(i)==1 else i[2:]  for i in table[1]]
stat_counter=Counter(stat_list)
