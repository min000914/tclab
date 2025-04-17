import tclab
import numpy as np
import time
import csv
import matplotlib.pyplot as plt
from APMonitor.apm import *
from mpc_lib import *

# 실험 파라미터
EPISODES = 100
EP_DURATION = 600  # 초
sleep_max = 3.0
st_temp = 27.0
s = 'http://byu.apmonitor.com'
c = 'my_MPC'

def generate_random_tsp(length, name='TSP'):
    tsp = np.zeros(length)
    i = 0
    print(f'[{name} 설정 정보]')
    while i < length:
        duration = int(np.clip(np.random.normal(240, 50), 80, 400))
        temp = np.random.uniform(25, 65)
        end = min(i + duration, length)
        tsp[i:end] = temp
        print(f'  구간: {i:>3} ~ {end - 1:>3}, 목표 온도: {temp:.2f}°C')
        i = end
    return tsp


# 장치 연결
a = tclab.TCLab()

for epi in range(1, EPISODES + 1):
    print(f"\n===== Episode {epi} 시작 =====")

    # 안전 온도 확인
    a.Q1(0)
    a.Q2(0)
    while a.T1 >= st_temp or a.T2 >= st_temp:
        print(f'Time: {i} T1: {a.T1} T2: {a.T2}')
        i += 20
        time.sleep(20)

    # 초기값 할당
    tm = np.zeros(EP_DURATION)
    T1 = np.ones(EP_DURATION) * a.T1
    T2 = np.ones(EP_DURATION) * a.T2
    Q1 = np.zeros(EP_DURATION)
    Q2 = np.zeros(EP_DURATION)
    Tsp1 = generate_random_tsp(EP_DURATION, 'TSP1')
    Tsp2 = generate_random_tsp(EP_DURATION, 'TSP2')

    mpc_init()

    start_time = time.time()
    prev_time = start_time
    dt_error = 0.0
    
    filename = f'mpc_episode_{epi}_data.csv'
    with open(filename, 'w', newline='') as fid:
        writer = csv.writer(fid)
        writer.writerow(['EPI_Num', 'Time', 'Q1', 'Q2', 'T1', 'T2', 'TSP1', 'TSP2'])

        for i in range(1, EP_DURATION):
            sleep = sleep_max - (time.time() - prev_time) - dt_error
            if sleep >= 1e-4:
                time.sleep(sleep - 1e-4)
            else:
                print('exceeded max cycle time by ' + str(abs(sleep)) + ' sec')
                time.sleep(1e-4)

            t = time.time()
            dt = t - prev_time
            if (sleep>=1e-4):
                dt_error = dt-sleep_max+0.009
            else:
                dt_error = 0.0
            prev_time = t
            tm[i] = t - start_time
            
            # Read temperatures in Kelvin 
            T1[i] = a.T1
            T2[i] = a.T2

            # MPC 계산
            Q1[i], Q2[i] = mpc(T1[i], Tsp1[i], T2[i], Tsp2[i])
            a.Q1(Q1[i])
            a.Q2(Q2[i])

            # 로그 출력
            print(f"{tm[i]:5.1f} s | T1: {T1[i]:.2f} / {Tsp1[i]:.2f}, T2: {T2[i]:.2f} / {Tsp2[i]:.2f} | Q1: {Q1[i]:.1f}, Q2: {Q2[i]:.1f}")

            # CSV 기록
            writer.writerow([
                epi,
                f"{tm[i]:.2f}",
                f"{Q1[i]:.2f}",
                f"{Q2[i]:.2f}",
                f"{T1[i]:.2f}",
                f"{T2[i]:.2f}",
                f"{Tsp1[i]:.2f}",
                f"{Tsp2[i]:.2f}"
            ])

    # 플롯 저장
    plt.figure(figsize=(10, 7))
    ax = plt.subplot(2, 1, 1)
    ax.grid()
    plt.plot(tm, Tsp1, 'k--', label='T1 setpoint')
    plt.plot(tm, T1, 'b-', label='T1 measured')
    plt.plot(tm, Tsp2, 'g--', label='T2 setpoint')
    plt.plot(tm, T2, 'r-', label='T2 measured')
    plt.ylabel('Temperature (°C)')
    plt.title(f'MPC Episode {epi}')
    plt.legend()

    ax = plt.subplot(2, 1, 2)
    ax.grid()
    plt.plot(tm, Q1, 'b-', label='Q1')
    plt.plot(tm, Q2, 'r-', label='Q2')
    plt.ylabel('Heater Output (%)')
    plt.xlabel('Time (sec)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'mpc_episode_{epi}_plot.png')
    plt.close()

# 종료
a.Q1(0)
a.Q2(0)
a.close()
print("모든 에피소드 종료됨.")
