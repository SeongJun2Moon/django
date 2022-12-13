import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, font_manager, rc
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.under_sampling import RandomUnderSampler
import seaborn as sns
font_path = "C:/Windows/Fonts/malgunbd.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

stroke_meta = {
    'id':'아이디', 'gender':'성별', 'age':'나이',
    'hypertension':'고혈압',
    'heart_disease':'심장병',
    'ever_married':'기혼여부',
    'work_type':'직종',
    'Residence_type':'거주형태',
    'avg_glucose_level':'평균혈당',
    'bmi':'체질량지수',
    'smoking_status':'흡연여부',
    'stroke':'뇌졸중'
}

'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5110 entries, 0 to 5109
Data columns (total 12 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   id                 5110 non-null   int64  
 1   gender             5110 non-null   object 
 2   age                5110 non-null   float64
 3   hypertension       5110 non-null   int64  
 4   heart_disease      5110 non-null   int64  
 5   ever_married       5110 non-null   object 
 6   work_type          5110 non-null   object 
 7   Residence_type     5110 non-null   object 
 8   avg_glucose_level  5110 non-null   float64
 9   bmi                4909 non-null   float64
 10  smoking_status     5110 non-null   object 
 11  stroke             5110 non-null   int64  
dtypes: float64(3), int64(4), object(5)
memory usage: 479.2+ KB
None
'''

class Iris:
    data_path = "C:/Users/MSJ/AIA/djangoProject/shop/iris/data"

    def __init__(self):
        self.iris = pd.read_csv(f'{self.data_path}/Iris.csv')

    '''
    1.스펙보기
    '''
    def spec(self):
        print(" --- 1.Shape ---")
        print(self.iris.shape)
        print(" --- 2.Features ---")
        print(self.iris.columns)
        print(" --- 3.Info ---")
        print(self.iris.info())
        print(" --- 4.Case Top1 ---")
        print(self.iris.head(1))
        print(" --- 5.Case Bottom1 ---")
        print(self.iris.tail(3))
        print(" --- 6.Describe ---")
        print(self.iris.describe())
        print(" --- 7.Describe All ---")
        print(self.iris.describe(include='all'))

    '''
    shape(150, 6)
    ['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm',
       'Species']
    '''


    '''
    2.한글 메타데이터
    '''
    def rename_meta(self):
        pass

    '''
    3.타깃변수(=종속변수 dependent, Y값) 설정
    입력변수(=설명변수, 확률변수, X값)
    타깃변수명: stroke (=뇌졸중)
    타깃변수값: 과거에 한 번이라도 뇌졸중이 발병했으면 1, 아니면 0
    인터벌 = ['나이','평균혈당','체질량지수']
    '''
    def interval(self):
        pass

    '''
    4.범주형 = ['성별', '심장병', '기혼여부', '직종', '거주형태','흡연여부', '고혈압']
    '''

    def ratio(self): # 해당 컬럼이 없음
        pass

    def norminal(self):
        pass

    def ordinal(self): # 해당 컬럼이 없음
        pass

    '''
    데이터프레임을 데이터 파티션하기 전에 타깃변수와 입력변수를 
    target 과 data 에 분리하여 저장한다.
    '''
    def targeting(self):
        pass

    def partition(self):
        pass

    def learning(self, flag):
        pass

    def hook(self):
        self.spec()

iris_menu = ["Exit", #0
                "Hook",#1
]#8
iris_lambda = {
    "1" : lambda x: x.hook()
}
if __name__ == '__main__':
    iris = Iris()
    while True:
        [print(f"{i}. {j}") for i, j in enumerate(iris_menu)]
        menu = input('메뉴선택: ')
        if menu == '0':
            print("종료")
            break
        else:
            try:
                iris_lambda[menu](iris)
            except KeyError as e:
                if 'some error message' in str(e):
                    print('Caught error message')
                else:
                    print("Didn't catch error message")
