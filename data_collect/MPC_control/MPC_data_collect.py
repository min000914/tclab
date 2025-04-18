import tclab
import numpy as np
import time
import csv
import matplotlib.pyplot as plt
from APMonitor.apm import *
from .mpc_lib import *

import os


csv_dir = '../data/PID2MPC/MPC2/csv'
png_dir = '../data/PID2MPC/MPC2/png'
os.makedirs(csv_dir, exist_ok=True)
os.makedirs(png_dir, exist_ok=True)
# 실험 파라미터
EPISODES = 100
EP_DURATION = 240  # 초
sleep_max = 5.0
st_temp = 29.0
s = 'http://byu.apmonitor.com'
c = 'my_MPC'

def generate_random_tsp(length, name='TSP'):
    tsp = np.zeros(length)
    i = 0
    print(f'[{name} 설정 정보]')
    while i < length:
        duration = int(np.clip(np.random.normal(48, 10), 16, 80))
        temp = np.random.uniform(25, 65)
        end = min(i + duration, length)
        tsp[i:end] = temp
        print(f'  구간: {i:>3} ~ {end - 1:>3}, 목표 온도: {temp:.2f}°C')
        i = end
    return tsp


def mpc_main():
    # 장치 연결
    a = tclab.TCLab()

    for epi in range(1, EPISODES + 1):
        print(f"\n===== Episode {epi} 시작 =====")

        # 안전 온도 확인
        a.Q1(0)
        a.Q2(0)
        i=0
        print(f'Check that temperatures are < {st_temp} degC before starting')
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
        iae = 0.0
        csv_filename = os.path.join(csv_dir,f'mpc_episode_{epi}_data.csv')
        with open(csv_filename, 'w', newline='') as fid:
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
                
                iae += np.abs(Tsp1[i]-T1[i]) + np.abs(Tsp2[i]-T2[i])

                # MPC 계산
                Q1[i], Q2[i] = mpc(T1[i], Tsp1[i], T2[i], Tsp2[i])
                a.Q1(Q1[i])
                a.Q2(Q2[i])

                print("{:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}".format(
                    'Time', 'Tsp1', 'T1', 'Q1', 'Tsp2', 'T2', 'Q2', 'IAE'
                ))
                print(('{:6.1f} {:6.2f} {:6.2f} ' + \
                        '{:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f}').format( \
                            tm[i],Tsp1[i],T1[i],Q1[i],Tsp2[i],T2[i],Q2[i],iae))
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
        plt.plot(tm,Tsp1,'k--',label=r'$T_1$ set point')
        plt.plot(tm,T1,'b.',label=r'$T_1$ measured')
        plt.plot(tm,Tsp2,'k-',label=r'$T_2$ set point')
        plt.plot(tm,T2,'r.',label=r'$T_2$ measured')
        plt.ylabel(r'Temperature ($^oC$)')
        plt.title(f'Episode {epi}')
        plt.legend(loc='best')

        ax = plt.subplot(2, 1, 2)
        ax.grid()
        plt.plot(tm,Q1,'b-',label=r'$Q_1$')
        plt.plot(tm,Q2,'r:',label=r'$Q_2$')
        plt.ylabel('Heater Output (%)')
        plt.xlabel('Time (sec)')
        plt.legend(loc='best')

        plt.tight_layout()
        png_filename=os.path.join(png_dir,f'mpc_episode_{epi}_plot.png')
        plt.savefig(png_filename)
        plt.close()

    # 종료
    a.Q1(0)
    a.Q2(0)
    a.close()
    print("모든 에피소드 종료됨.")
