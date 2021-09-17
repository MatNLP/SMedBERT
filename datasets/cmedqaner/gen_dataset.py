import json
from random import randint
prefix='dev'
js_file=open(prefix+'.json','r',encoding='utf-8')
out_file=open(prefix+'_new.txt','w+',encoding='utf-8')

max_len=0
allowed_maxlen=510

for line in js_file:
    js_obj=json.loads(line)
    text=list(js_obj['text'])
    max_len=max(max_len,len(text))
    labels=['O']*len(text)
    for m in js_obj['mention_data']:
        offset=int(m['offset'])
        mention=m['mention']
        label=m['type']
        labels[offset]='B-'+label
        labels[offset+1:offset+len(mention)]=['I-'+label]*(len(mention)-1)
    to_split=1 if len(text)<allowed_maxlen else (len(text)//allowed_maxlen + 1) 
    to_write=[]
    # print(len(text))
    if(to_split==1):
        to_write.append(labels)
    else:
        split_point=[0]
        offset=randint(1,5)
        basic_len=len(text)//to_split
        for i in range(1,to_split):
            start=i*basic_len
            while(True):
                tmp_set=set(labels[start:start+offset])
                if(len(tmp_set)==1 and 'O' in tmp_set):
                    split_point.append(start)
                    print(start)
                    break
                start+=1
        split_point.append(len(labels))
        print(split_point)
        for i in range(len(split_point)-1):
            to_write.append( labels[split_point[i]:split_point[i+1]] )  
    i=0
    # print(text)
    for w in to_write:
        # print(len(w))
        assert(len(w)<=allowed_maxlen and len(w)>=min(10,len(text)))
        for l in w:
            out_file.write('{} {}\n'.format(text[i],l))
            i+=1
        out_file.write('\n')

print(max_len)
    
