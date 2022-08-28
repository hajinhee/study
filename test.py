from icecream import ic
import pandas as pd
import numpy as np
import random
import string

def non_static_method22(self):
        ic('non_static_method')

class Test:
    def __init__(self) -> None:
        pass

    def non_static_method(self):
        ic('non_static_method')

    @staticmethod
    def static_method():
        ic('static_method')

if __name__ == '__main__':
    test_object = Test()
    test_object.non_static_method()
    test_object.static_method()

    test_object2 = Test()
    test_object2.non_static_method()
    test_object2.static_method()

    Test.static_method()
    Test().static_method()
    # Test.non_static_method() 
    Test().non_static_method()


'''
 Q3 두자리 정수를 랜덤으로 2행 3열 데이터프레임을 생성
        ic| df3:     0   1   2
                 0  95  25  74
                 1  44  24  97
'''

def quiz31_rand_2_by_3(self) -> str:
    '''
    l1 = [myRandom(10, 99) for i in range(3)]
    l2 = [myRandom(10, 99) for i in range(3)]
    l3 = list([myRandom(10, 99) for i in range(3)] for i in range(2))
    df = pd.DataFrame(l3, index=range(2), columns=range(3))
    '''
    # numpy 사용한 예제
    df = pd.DataFrame(np.random.randint(10, 99, size=(2, 3)))
    ic(df)
    return None

'''
데이터프레임 문제 Q04.
국어, 영어, 수학, 사회 4과목을 시험치른 10명의 학생들의 성적표 작성.
단 점수 0 ~ 100이고 학생은 랜덤 알파벳 5자리 ID 로 표기
ic| df4:        국어  영어  수학  사회
        lDZid   57   90   55    24
        Rnvtg   12   66   43    11
        ljfJt   80   33   89    10
        ZJaje   31   28   37    34
        OnhcI   15   28   89    19
        claDN   69   41   66    74
        LYawb   65   16   13    20
        QDBCw   44   32    8    29
        PZOTP   94   78   79    96
        GOJKU   62   17   75    49
'''

@staticmethod
def id(chr_size) -> str: return ''.join([random.choice(string.ascii_letters) for i in range(chr_size)])

def quiz32_df_grade(self) -> str:
    data = np.random.randint(0, 101, (10, 4))  # 과목 점수 (10, 4)
    idx = [self.id(5) for i in range(10)]  # 아이디 인덱스
    col = ['국어', '영어', '수학', '사회']  # 컬럼

    df_list = pd.DataFrame(data, idx, col)
    ic(df_list)

    df_dict = pd.DataFrame.from_dict(dict(zip(idx, data)), orient='index', columns=col)
    ic(df_dict)
    return None