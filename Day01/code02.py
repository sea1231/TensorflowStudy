
# 경사하강법 Gradient Descent
# 변수가 2개 일때

import numpy as np

def f(x):
    return 2*x[0]**2 + 2*x[0]*x[1] + x[1]**2

def df(x):
    dx = 4*x[0] + 2*x[1]
    dy = 2*x[0] + 2*x[1]
    return np.array([dx,dy])

rho = 0.005
precision = 0.0000001
diff = 100

x = np.random.rand(2) # x, y 값 랜덤으로 생성

while diff > precision: # 두 점의 거리가 precision보다 클 동안에만 실행
    dr = df(x) # x, y 의 기울기 구하기
    prev_x = x # 현재 x,y 값 저장
    x = x - rho * dr # 다음점 x1,y1 은 현재점-상수(rho)*현재점의 기울기
    diff = np.dot(x-prev_x, x-prev_x)
    print("x = {}, df = {}, f(x) = {:f}".format(np.array2string(x), np.array2string(dr), f(x)))