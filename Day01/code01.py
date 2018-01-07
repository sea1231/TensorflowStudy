
# 경사하강법 Gradient Descent
# 변수가 1개 일때

from random import random

# 함수 정의
def f(x):
    return x**4 - 12.0*x**2 - 12.0*x

# 함수 f(x)를 미분한 함수 정의
def df(x):
    return 4*x**3 - 24 *x -12

rho = 0.005
precision = 0.000000001
difference = 100
x = random() # x값은 랜덤으로 선택

while difference > precision: # 두 점의 거리가 precision보다 클 동안에만 실행
    dr = df(x) # dr은 현재 x점의 기울기
    prev_x = x # 현재점 x0 을 prev_x 에 저장
    x = x - rho * dr # 다음점 x1 은 현재점-상수(rho)*현재점의 기울기
    difference = abs(prev_x-x) # 두 점들의 거리를 구함
    print("x = {:f}, df = {:10.6f}, f(x) = {:f}".format(x, dr, f(x)))
