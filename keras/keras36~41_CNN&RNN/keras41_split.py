#  RNN 쓰기 위해 데이터 timestep대로 데이터를 잘라서 만들어주는 함수. + 해석
  
import numpy as np

a = np.array(range(1, 15))                              # 데이터로드

size = 5                                                # timestep 값 

def split_x(dataset, size):                             # 함수선언 + 데이터,timestep 값 입력.
    aaa = []                                            # aaa라는 리스트 선언
    for i in range(len(dataset) - size + 1):            # 데이터셋(a)의 길이 - timesteps값 + 1만큼의 횟수만큼 반복하겠다
        subset = dataset[i : (i + size)]                # subset이 뭐냐면 -> 로드한 dataset을 슬라이싱해서 그 부분만큼만 불러옴 리스트형태로. -> 그걸 subset에 저장
        aaa.append(subset)                              # aaa라는 리스트에 subset을 추가하겠다.
    return np.array(aaa)                                # 구해낸 aaa값을 반환해준다.

cc = split_x(a, size)

#print(cc.shape)  #(6, 5)

x = cc[:,:4]       #행과 열을 각각 슬라이싱해줘야 한다. ,로 구분하고 :만 쓰는건 전체를 다 쓰겠다는 뜻.
y = cc[:, 4]

print(x)      # (6, 4)
print(y)      # (6,)
