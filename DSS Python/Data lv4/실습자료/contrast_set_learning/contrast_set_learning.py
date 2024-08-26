#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
import itertools as it
from scipy.stats import chi2_contingency, ttest_1samp
logging.getLogger().setLevel(logging.INFO)

class ContrastSetLearner:
    def __init__(self, num_parts=4, cset_max_length = 3,
                 min_group_sup = 0.1, min_total_sup = 0.05, max_total_sup = 0.9,
                 alpha = 0.05, min_dev = 0.05, min_corr = 0.1,
                 min_subset_sup=0.01, min_subset_corr=0.01 ):
        
        """
        num_parts: 연속형 입력변수의 분할 수
        cset_max_length: 생성된 cset의 최대 길이
        
        min_group_sup: 허용 가능한 특정 그룹 내 cset의 minimum support 
        min_total_sup: 허용 가능한 전체 데이터 내 cset의 minimum support 
        max_total_sup: 허용 가능한 전체 데이터 내 cset의 maximum support
        
        alpha: 독립성 검정을 위한 유의수준 기준
        min_dev: 허용 가능한 그룹 간 cset의 support 차이 
        min_corr: 허용 가능한 cset과 출력변수와의 minimum correlation

        min_subset_sup: 허용 가능한 부분 집합과의 최소 support 차이 
        min_subset_corr: 허용 가능한 부분 집합과의 최소 correlation 차이
        
        """


        try:
            if num_parts<1:        logging.error("num_parts가 1보다 작습니다.")
            if cset_max_length<1:  logging.error("cset_max_length가 1보다 작습니다.")
                
            if min_group_sup<=0:   logging.error("min_group_sup가 0 보다 같거나 작습니다")
            if min_total_sup<=0:   logging.error("min_total_sup가 0 보다 같거나 작습니다")
            if max_total_sup>=1:   logging.error("max_total_sup이 1 보다 같거나 큽니다")
                        
            if alpha<=0:           logging.error("alpha가 0보다 같거나 작습니다.")
            if min_dev<=0:         logging.error("min_dev가 0보다 같거나 작습니다.")
            if min_corr<=0:        logging.error("min_corr이 0보다 같거나 작습니다.")
                
            if min_subset_sup<=0:  logging.error("min_subset_sup가 0 보다 같거나 작습니다.")
            if min_subset_corr<=0: logging.error("min_subset_corr이 0 보다 같거나 작습니다.")
                
        except TypeError:
            logging.error("숫자형 인자값을 넣어야 합니다.")
            
        self.num_parts=num_parts
        self.cset_max_length=cset_max_length
        
        self.min_group_sup=min_group_sup
        self.min_total_sup=min_total_sup
        self.max_total_sup=max_total_sup
        
        self.alpha=alpha
        self.min_dev=min_dev
        self.min_corr=min_corr
        
        self.min_subset_sup=min_subset_sup
        self.min_subset_corr=min_subset_corr
        
    def transform(self,data_frame,group_variable):   
        logging.info("transform 전 데이터 크기는 %s 입니다"%(list(data_frame.shape)))
        
        # 결측치 제거 
        if (data_frame.isnull().sum().sum())>0: 
            data_frame=data_frame.dropna(axis=0)
            logging.info("결측 데이터가 제거되었습니다.") 
            
        # 출력변수 확인       
        try:
            Y=data_frame[group_variable]
            X=data_frame.drop(group_variable,axis=1)
            logging.info("'group_variable'은 %s 입니다"%(group_variable))
        except KeyError:
            logging.error("'group_variable'이 유효하지 않습니다. 올바르게 설정해주세요.")      
        self.group_variable=group_variable
        
        # 출력변수 전처리 
        if (Y.dtypes=="float64")|(Y.dtypes=="int64"):
            print("'group_variable'이 연속형이므로 범주화 작업을 수행합니다.")
            Y=pd.cut(Y,2,labels=["Pass","Fault"])
            
        # 입력변수 전처리
        for i in X.columns:
            if (X.dtypes.loc[i]=="float64")|(X.dtypes.loc[i]=="int64"):
                print("입력변수 %s는 연속형 변수이므로 %d등분 합니다." %(i,num_parts))
                X[i]=pd.qcut(X[i],num_parts,duplicates="drop")
            X[i]=i+"_"+X[i].astype("str")
                
        data_frame_new=pd.concat([X,Y],axis=1)
        logging.info("transform 후 데이터 크기는 %s 입니다"%(list(data_frame_new.shape)))
        return data_frame_new
    
    
    
    def fit(self,transform_data):
        
        # 주요 함수 정의
        def build_candidate(gen_candidate, cset_length):
            columns = ['result_group', 'group_sup', 'total_sup', 'corr', 'significant', 'large','g_flag', 'lift']
            columns_add = ['candidate_'+str(i) for i in list(range(1, cset_length+1))]
            candidate_df = pd.DataFrame(columns=columns+columns_add)
            if cset_length == 1: candidate_df['candidate_1'] = gen_candidate; return candidate_df
            stacks = [stack(x, index_dict_keys.index(x[-1]), candidate_df.columns, corr) for x, corr in zip(gen_candidate.loc[:, 'candidate_1':].values, gen_candidate['corr'].values)]
            candidate_df = pd.concat(stacks)
            return candidate_df
        
        def build_deviation(candidate_df, cset_length):
            aggregation = candidate_df.iloc[:, -cset_length:].agg(' & '.join, axis=1)
            dev_df = candidate_df.drop(candidate_df.columns[-cset_length:], axis=1)
            dev_df['cset'] = aggregation
            if cset_length == 1: dev_df['total_sup'] = [len(logic([x[0]]))/len(transform_data) for x in dev_df.loc[:, 'cset':].values]
            dev_df = dev_df.reset_index(drop=True)
            dev_df = dev_df.drop(dev_df.index[dev_df['result_group']=='nan'])
            dev_df = dev_df.dropna()
            return dev_df
         
        def build_generation(gen_candidate, cset_length):
            aggregation = gen_candidate.iloc[:, 1-cset_length:].agg(' & '.join, axis=1)
            gen_candidate = gen_candidate.drop(gen_candidate.columns[1-cset_length:], axis=1)
            gen_candidate['cset'] = aggregation
            return gen_candidate
        
        def check_corr(x, dev, gen):
            if len(dev) == 1: return 0
            check_list = []
            for i in range(len(x)-1, 0, -1): check_list += list(it.combinations(x[:-1], i))
            check_list1 = [list(z) for z in check_list if len(z) == len(x)-1]
            check_list2 = [list(z)+[x[-1]] for z in check_list if len(z) < len(x)-1]
            check_list = check_list1 + check_list2
            check_list = [' & '.join(x) for x in check_list]
            check_list += [x[-1]]
            corr_list = [corr(x, dev, gen) for x in check_list]
            max_corr = np.max(corr_list)
            return max_corr
        
        def check_group_support(x, dev, gen):
            if len(dev) == 1: return 1+self.min_subset_sup
            check_list = []
            for i in range(len(x)-1, 0, -1): check_list += list(it.combinations(x[:-1], i))
            check_list1 = [list(z) for z in check_list if len(z) == len(x)-1]
            check_list2 = [list(z)+[x[-1]] for z in check_list if len(z) < len(x)-1]
            check_list = check_list1 + check_list2
            check_list = [' & '.join(x) for x in check_list]
            check_list += [x[-1]]
            group_support_list = [group_support(x, dev, gen) for x in check_list]
            min_group_support = np.min(group_support_list)
            return min_group_support

        def corr(x, dev, gen):
            if len(dev[dev['cset']==x]) > 0: return float(dev[dev['cset']==x]['corr'].values[0])
            elif len(gen[gen['cset']==x]) > 0: return float(gen[gen['cset']==x]['corr'].values[0])
            else: return 0         
 
        def fill(x, max_corr, min_group_support, candi_size, cset_length):
            fill_data = np.full((7), np.nan).tolist()
            group_2_x = group_dict[group_2].intersection(x)
            group_1_x = group_dict[group_1].intersection(x)
            if max(support(group_2_x, group_dict[group_2]),support(group_1_x, group_dict[group_1])) < self.min_group_sup: return fill_data
            if abs(support(group_2_x, group_dict[group_2])-support(group_1_x, group_dict[group_1])) < self.min_dev: return fill_data
            if len(x)*min(len(group_dict[group_2]), len(group_dict[group_1]))/len(transform_data) < 5: return fill_data
            fill_data[0] = max(support(group_2_x, group_dict[group_2]), support(group_1_x, group_dict[group_1]))
            if fill_data[0] > min_group_support-self.min_subset_sup: return fill_data
            chi_matrix = np.array([[len(group_2_x), len(group_1_x)], [len(group_dict[group_2])-len(group_2_x), len(group_dict[group_1])-len(group_1_x)]])
            fill_data[2] = chi2_contingency(chi_matrix)[1]
            fill_data[3] = np.sqrt(chi2_contingency(chi_matrix)[0]/len(transform_data))
            fill_data[6] =abs(support(group_2_x, group_dict[group_2])-support(group_1_x, group_dict[group_1]))
            if min(len(group_2_x), len(group_1_x)) > 0: fill_data[1] = 1
            else: fill_data[1] = 0
                
            if fill_data[2] > (self.alpha/(2**cset_length)): return fill_data
            if fill_data[3] < self.min_corr: return fill_data
            if fill_data[3] < max_corr+self.min_subset_corr: return fill_data

            if support(group_2_x, group_dict[group_2])-support(group_1_x, group_dict[group_1]) > 0:
                fill_data[4] = group_2
                fill_data[5] = lift(group_2_x, group_dict[group_2], x)
            else:
                fill_data[4] = group_1
                fill_data[5] = lift(group_1_x, group_dict[group_1], x)
            return fill_data


        def fill_candidate(candidate_df,cset_length):
            condition = [logic(x) for x in candidate_df.loc[:, 'candidate_1':].values]
            fills = np.array([fill(x, max_corr, min_group_support,len(candidate_df),cset_length) for x, max_corr, min_group_support in zip(condition, max_corr_list,min_group_support_list)])
            candidate_df['group_sup'] = fills[:, 0]
            candidate_df['g_flag'] = fills[:, 1]
            candidate_df['significant'] = fills[:, 2]
            candidate_df['corr'] = fills[:, 3]
            candidate_df['result_group'] = fills[:, 4]
            candidate_df['lift'] = fills[:, 5]
            candidate_df['large'] = fills[:, 6]
            
            return candidate_df 

        def group_support(x, dev, gen):
            if len(dev[dev['cset']==x]) > 0: return float(dev[dev['cset']==x]['group_sup'].values[0])
            elif len(gen[gen['cset']==x]) > 0: return float(gen[gen['cset']==x]['group_sup'].values[0])
            else: return 0 

        def lift(cset, group, x): return ((len(cset)/len(transform_data))/((len(group)/len(transform_data))*len(x)/len(transform_data)))

        def logic(x):
            x = [index_dict[x] for x in x]
            z = x[0]
            for i in x[1:]: z = z.intersection(i) 
            return z

        def stack(x, index, columns, corr):
            df = pd.DataFrame(index=range(index+1, len(index_dict_keys)), columns=columns)
            for i in range(0, len(x)): df['candidate_%d'%(i+1)] = x[i]
            df['candidate_%d'%(len(x)+1)] = index_dict_keys[index+1:len(index_dict_keys)]
            df['corr'] = corr
            df['total_sup'] = [len(logic(x))/len(transform_data) for x in df.loc[:, 'candidate_1':].values]
            df = df[df['total_sup']>self.min_total_sup]
            df = df[df['total_sup']<self.max_total_sup]
            return df

        def support(cset, group): return len(cset)/len(group)

                 
        ################# 코드 시작 
        
        trans_Y=transform_data[self.group_variable]
        trans_X=transform_data.drop(self.group_variable,axis=1)
        
        # 각 인자별 index 추출
        index_dict = {}
        group_dict = {}
        
        # 입력변수 index 추출
        for i in trans_X.columns:
            for j in trans_X[i].unique():
                if (len(trans_X.index[trans_X[i]==j])>= np.shape(transform_data)[0]*self.min_total_sup)|(len(trans_X.index[trans_X[i]==j])<np.shape(transform_data)[0]*self.max_total_sup):
                    index_dict[j]=trans_X.index[trans_X[i]==j] 
                    
        # 출력변수 index 추출            
        for k in trans_Y.unique():
            group_dict[k]=trans_Y.index[trans_Y==k]
        
        index_dict_keys = list(index_dict.keys())
        group_dict_keys = list(group_dict.keys())
        
        # Contrast set learning 
        if index_dict_keys == 0: print('# PATTERN NOT FOUND.'); return 0 

        group_1=group_dict_keys[0]
        group_2=group_dict_keys[1]
        dev_list = []
        gen_list = []

        for cset_length in range(1, (self.cset_max_length)+1):
            print('# CSET LENGTH %d START.'%(cset_length))
            print('# building candidate... (1/4)')
            if cset_length == 1: gen_candidate = index_dict_keys
            else: gen_candidate = candidate_df[candidate_df['g_flag']=='1']
            if len(gen_candidate) == 0: print('# NO MORE CONSIDERABLE CANDIDATE.'); break
            candidate_df = build_candidate(gen_candidate, cset_length)
            if cset_length != 1:
                gen_candidate = build_generation(gen_candidate, cset_length)
                gen_list.append(gen_candidate)
                dev = pd.concat(dev_list)
                gen = pd.concat(gen_list)
            else: dev, gen = [0], [0]

            print('# checking correlation... (2/4)')
            max_corr_list = [check_corr(x, dev, gen) for x in candidate_df.loc[:, 'candidate_1':].values]

            print('# checking group support... (3/4)')
            min_group_support_list = [check_group_support(x, dev, gen) for x in candidate_df.loc[:, 'candidate_1':].values]

            print('# calculating value... (4/4) \n\n')
            candidate_df = fill_candidate(candidate_df,cset_length)
            deviation = build_deviation(candidate_df, cset_length)
            dev_list.append(deviation)

        self.rules=pd.concat(dev_list)
        print('# DONE.')                


# In[ ]:




