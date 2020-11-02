import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('career_wanted_list.csv')

count_vector = CountVectorizer()
#해당 직종코드가 각 기업에 있으면 1, 없으면 0
count_vector_jobsCd = count_vector.fit_transform(df['jobsCd'])

#코사인 유사도를 구한 벡터를 미리 저장
jobsCd_sim = cosine_similarity(count_vector_jobsCd, count_vector_jobsCd).argsort()[:,::-1]

def get_recommend_job_list(df, company):
    #입력받은 기업의 정보 추출
    target_job_index = df[df['company'] == company].index.values

    #비슷한 코사인 유사도를 가진 정보 추출
    sim_index = jobsCd_sim[target_job_index].reshape(-1)

    #data frame으로 만들고 score로 정렬
    result = df.iloc[sim_index].sort_values('score', ascending = False)[:10]
    return result
