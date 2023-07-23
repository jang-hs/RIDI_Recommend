제가 애용하는 서비스인 리디셀렉트의 도서 소개 데이터를 크롤링하여 책 추천, 작가 추천을 구현하였습니다.

TF-IDF 행렬을 활용하여 코사인 유사도를 산출하고, 입력한 항목과 가장 유사도가 높은 항목을 추천해주는 방식입니다.

노트북 램 용량의 한계로 장르별로만 매트릭스를 생성하였습니다.

코드는 아래에 첨부하였습니다.

```python
'''
@ create on 2020.09.01
'''
# =========================================================
# 00. Package Load
# =========================================================

from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

import pandas as pd
import numpy as np
okt = Okt()

# stopwords
with open("stopwords.txt", "r") as f:

    stopwords = f.readlines()
stopwords = ",".join(stopwords).replace('\n', '').split(sep = ',')
```

```python
# Tokenizer & Vectorizer
def tokenizer(raw, pos=["Noun"], stopword=[]):
    return [
        word for word, tag in okt.pos(
            raw, 
            norm=True,   # normalize
            stem=True    # stemming
            )
            if len(word) > 1 and tag in pos and word not in stopword
        ]

def vectorizer():
    vectorize = TfidfVectorizer(
    tokenizer=tokenizer,
    min_df=2,
    ngram_range = (1,2), # ngram 범위
    sublinear_tf=True  # TF가 무한정 커지는 것을 막음
    )
    return vectorize
```

```python
# =========================================================
# 01. Load Dataset / mode option
# =========================================================
title_dataset = pd.read_csv('title_dataset.csv', encoding = 'utf8') # 책 제목 / 책 소개 데이터셋
author_dataset = pd.read_csv('authort_dataset.csv', encoding = 'utf8') # 저자 / 저자 설명 데이터셋
theme_no = 0
# vetorizer 선언
vectorize = vectorizer()


# mode가 title일 경우 책 데이터셋 / author일 경우 저자 데이터셋의 파라미터 지정
mode = 'title'
if mode == 'title':
    dataset = title_dataset
    dup_check_col = ['책 제목','출간일','평점']
    book_theme = ['소설','경영/경제','인문/사회/역사','자기계발','만화 단행본',
                  '에세이/시','가정/생활','과학','어린이/청소년','건강/다이어트',
                  '외국어','여행','잡지']
    tfidf_col = '책 소개'
    tfidf_name = '책 제목'

else :
    dataset = author_dataset
    dup_check_col = ['저자','저자 소개','출간일','평점']
    book_theme = ['소설','경영/경제','인문/사회/역사','자기계발','만화 단행본',
                  '에세이/시','가정/생활','과학','어린이/청소년','건강/다이어트',
                  '외국어','여행']
    tfidf_col = '저자 소개'
    tfidf_name = '저자'
```

```python
title_dataset.head()
```

```python
author_dataset.head()
```

|   | CATEGORY\_CD | CATEGORY\_NM1 | CATEGORY\_NM2 | 책 제목 | 저자 | 저자 소개 | 출간일 | 평점 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 101 | 소설 | 한국소설 | 달리는 조사관 | 송시우 | 저자 - 송시우 2008년 단편소설 ＜좋은 친구＞로 계간 미스터리 신인상을 수상하면... | 2015.10.22. | 4.5 |
| 1 | 101 | 소설 | 한국소설 | 검은 개가 온다 | 송시우 | 대전에서 태어났다. 고려대학교 철학과를 졸업했다. 2008년 《계간 미스터리》 겨울... | 2018.07.20. | 4.3 |
| 2 | 101 | 소설 | 한국소설 | 280일 | 전혜진 | 지은이: 전혜진 글 쓰고 만화 만들고 컴퓨터와 잘 노는 사람. 퇴근 후에는 “성실한... | 2019.06.25. | 4.7 |
| 3 | 101 | 소설 | 한국소설 | 404 이름을 찾을 수 없습니다. | 무명 | 필명은 무명이다. 한 때, 모든 공중파와 종편의 메인 뉴스, 인터넷 포털의 실시간 ... | 2020.01.17. | 3.6 |
| 4 | 101 | 소설 | 한국소설 | 철수 이야기 1권 | 상수탕 | 춘천에서 태어났다. 홍익대학교 미술대학을 졸업하고 여러 학교를 중퇴했다. 부엽토 깔... | 2020.02.21. | 4.8 |

```python
# =========================================================
# 02. Create Similarity Matrix
# =========================================================
def create_sim_matrix(dataset, dup_check_col, book_theme, tfidf_col, tfidf_name):

    # 중복된 행은 제거한 뒤 select_dataset 생성
    dataset_ = dataset.copy()[dup_check_col]
    drop_duplicated_index = dataset_.drop_duplicates().index
    dataset_df = dataset.loc[drop_duplicated_index].reset_index(drop = True)
    select_dataset = dataset_df[dataset_df['CATEGORY_NM1']==book_theme[theme_no]].reset_index(drop=True)

    # TF-IDF 매트릭스 생성
    tfidf_matrix = vectorize.fit_transform(select_dataset[tfidf_col])
    features = vectorize.get_feature_names()

    # 코사인 유사도 산출
    from sklearn.metrics.pairwise import linear_kernel
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    target_list = list(select_dataset[tfidf_name])

    indices = pd.Series(select_dataset.index, index=select_dataset[tfidf_name]).drop_duplicates()
#     doc_similarities = tfidf_matrix *tfidf_matrix.T
#     doc_sim_df = pd.DataFrame(doc_similarities.toarray(), columns =target_list, index = target_list)

    return indices, target_list, select_dataset, cosine_sim
```

```python
# 인덱스, 추천 대상리스트, 정제된 데이터셋, 코사인 유사도 생성
indices, target_list, select_dataset, cosine_sim = create_sim_matrix(dataset, dup_check_col, book_theme, tfidf_col, tfidf_name)
```

```python
# =========================================================
# 03. Get Recommand
# =========================================================
def get_recommendations(dataset, title, cosine_sim = cosine_sim):
    idx = indices[title]

    # 모든 책에 대해서 지정된 책의 유사도 산출
    sim_scores = list(enumerate(cosine_sim[idx])) 
    # 유사도 순으로 정렬
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # 상위 10개 항목을 저장
    sim_scores = sim_scores[1:11]
    # 가장 유사한 10개의 인덱스 저장
    book_indices = [i[0] for i in sim_scores]

    # 상위 10개 유사 항목 리턴
    return dataset[tfidf_name].iloc[book_indices]
```

```python
# 책 제목을 input으로 받고 추천 항목 안에 없을 경우 메시지 리턴
search_t = input(f'{book_theme[theme_no]} 장르의 책 제목을 입력하세요 - ')
if search_t in target_list:
    pass
else:
    print(f'{book_theme[theme_no]} 장르에는 책 {search_t}이(가) 없습니다.')
```

```
소설 장르의 책 제목을 입력하세요 - 280일
```

```python
# 추천 항목 얻기 
get_recommendations(dataset = select_dataset, title = search_t)
```

```python
542            아침이 온다
287         나는 언제나 옳다
114    무소의 뿔처럼 혼자서 가라
340            언틸유아마인
341           개정판 | 룸
24             세계의 호수
417              13시간
286           다크 플레이스
295               시스터
169          몸을 긋는 소녀
Name: 책 제목, dtype: object
```
