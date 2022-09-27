from bayes_opt import BayesianOptimization

def black_box_function(x,y):
    return -x **2 - (y-1) ** 2 + 1   # -x제곱 - (y-1)제곱 + 1

pbounds = {'x' : (2,4), 'y' : (-3, 3)}

optimizer = BayesianOptimization(
    f = black_box_function,
    pbounds = pbounds,
    random_state = 66)

# f에 모델을 넣고, pbonds에 파라미터

optimizer.maximize(
    init_points = 2, 
    n_iter = 15
)

'''
|   iter    |  target   |     x     |     y     |
-------------------------------------------------
|  1        | -14.56    |  2.309    | -2.198    |
|  2        | -6.433    |  2.725    |  1.075    |
|  3        | -14.75    |  3.369    | -1.097    |
|  4        | -6.331    |  2.489    |  2.065    |
|  5        | -19.0     |  4.0      |  3.0      |
|  6        | -3.117    |  2.0      |  1.341    |
|  7        | -3.634    |  2.0      |  0.2036   |
=================================================
'''