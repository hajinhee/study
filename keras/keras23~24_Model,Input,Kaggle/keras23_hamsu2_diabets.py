from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
import numpy as np

#1.데이터
datasets = load_diabetes()
x = datasets.data  # (442, 10)
y = datasets.target
# print(x.shape)
# print(np.unique(y))
'''
[ 25.  31.  37.  39.  40.  42.  43.  44.  45.  47.  48.  49.  50.  51.
  52.  53.  54.  55.  57.  58.  59.  60.  61.  63.  64.  65.  66.  67.
  68.  69.  70.  71.  72.  73.  74.  75.  77.  78.  79.  80.  81.  83.
  84.  85.  86.  87.  88.  89.  90.  91.  92.  93.  94.  95.  96.  97.
  98.  99. 100. 101. 102. 103. 104. 107. 108. 109. 110. 111. 113. 114.
 115. 116. 118. 120. 121. 122. 123. 124. 125. 126. 127. 128. 129. 131.
 132. 134. 135. 136. 137. 138. 139. 140. 141. 142. 143. 144. 145. 146.
 147. 148. 150. 151. 152. 153. 154. 155. 156. 158. 160. 161. 162. 163.
 164. 166. 167. 168. 170. 171. 172. 173. 174. 175. 177. 178. 179. 180.
 181. 182. 183. 184. 185. 186. 187. 189. 190. 191. 192. 195. 196. 197.
 198. 199. 200. 201. 202. 206. 208. 209. 210. 212. 214. 215. 216. 217.
 219. 220. 221. 222. 225. 229. 230. 232. 233. 235. 236. 237. 241. 242.
 243. 244. 245. 246. 248. 249. 252. 253. 257. 258. 259. 261. 262. 263.
 264. 265. 268. 270. 272. 273. 274. 275. 276. 277. 279. 280. 281. 283.
 288. 292. 293. 295. 296. 297. 302. 303. 306. 308. 310. 311. 317. 321.
 332. 336. 341. 346.]
'''
x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.9, shuffle=True, random_state=42) 

#scaler = MinMaxScaler()   
#scaler = StandardScaler()
#scaler = RobustScaler()
scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)   
x_test = scaler.transform(x_test)    


#2. 모델구성,모델링
input1 = Input(shape=(10,))
dense1 = Dense(100)(input1)
dense2 = Dense(80,activation="relu")(dense1)
dense3 = Dense(60)(dense2)
dense4 = Dense(40,activation="relu")(dense3)
dense5 = Dense(20)(dense4)
output1 = Dense(1)(dense5)
model = Model(inputs=input1,outputs=output1)

# model = Sequential()
# model.add(Dense(100, input_dim=10))
# model.add(Dense(80,activation='relu')) #
# model.add(Dense(60))
# model.add(Dense(40,activation='relu')) #
# model.add(Dense(20))
# model.add(Dense(1))
# model.summary()


#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam') 
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, baseline=None, restore_best_weights=True)
model.fit(x_train, y_train, epochs=10000, batch_size=10, validation_split=0.1111111, callbacks=[es])

model.save("./_save/keras25_2_save_diabets.h5")
#model = load_model("./_save/keras25_2_save_diabets.h5")

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ', r2)


'''
loss: 2737.8281
r2스코어 :  0.5524145457365297
'''