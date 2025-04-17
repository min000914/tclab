import tclab
import numpy as np
import time
import csv
import matplotlib.pyplot as plt

# PID 상수
Kc   = 9.24
tauI = 126.6 # sec
tauD = 8.90  # sec
Kff  = -0.66
st_temp = 27.0

# 에피소드 루프
EPISODES = 100
EP_DURATION = 600
sleep_max = 1.0


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


def pid(sp, pv, pv_last, ierr, dt, d, cid):
    if cid == 1:
        KP = Kc
        Kf = Kff
    else:
        KP = Kc * 2.0
        Kf = Kff * 2.0
    KI = Kc / tauI
    KD = Kc * tauD
    op0 = 0
    ophi = 100
    oplo = 0
    error = sp - pv
    ierr += KI * error * dt
    dpv = (pv - pv_last) / dt
    P = KP * error
    I = ierr
    D = -KD * dpv
    FF = Kf * d
    op = op0 + P + I + D + FF
    if op < oplo or op > ophi:
        I -= KI * error * dt
        op = max(oplo, min(ophi, op))
    return op, P, I, D, FF

# 초기화
a = tclab.TCLab()

for epi in range(1, EPISODES + 1):

    # 쿨다운
    a.Q1(0)
    a.Q2(0)
    print('Check that temperatures are < 25 degC before starting')
    i = 0

    
    while a.T1 >= st_temp or a.T2 >= st_temp:
        print(f'Time: {i} T1: {a.T1} T2: {a.T2}')
        i += 20
        time.sleep(20)

    tm = np.zeros(EP_DURATION)
    T1 = np.ones(EP_DURATION) * a.T1
    T2 = np.ones(EP_DURATION) * a.T2
    Q1 = np.zeros(EP_DURATION)
    Q2 = np.zeros(EP_DURATION)

    # 랜덤한 구간 및 온도로 setpoint 설정
    Tsp1 = generate_random_tsp(EP_DURATION, 'TSP1')
    Tsp2 = generate_random_tsp(EP_DURATION, 'TSP2')


    print(f'===== Episode {epi} Start =====')
    start_time = time.time()
    prev_time = start_time
    dt_error = 0.0
    # Integral error
    ierr1 = 0.0
    ierr2 = 0.0
    # Integral absolute error
    iae = 0.0

    filename = f'episode_{epi}_data.csv'
    
    
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

            # Disturbances
            d1 = T1[i] - 23.0
            d2 = T2[i] - 23.0

            # Integral absolute error
            iae += np.abs(Tsp1[i]-T1[i]) + np.abs(Tsp2[i]-T2[i])

            # Calculate PID output
            Q1[i],P,ierr1,D,FF = pid(Tsp1[i],T1[i],T1[i-1],ierr1,dt,d2,1)
            Q2[i],P,ierr2,D,FF = pid(Tsp2[i],T2[i],T2[i-1],ierr2,dt,d1,2)

            # Write output (0-100)
            a.Q1(Q1[i])
            a.Q2(Q2[i])
            
            print("{:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}".format(
                'Time', 'Tsp1', 'T1', 'Q1', 'Tsp2', 'T2', 'Q2', 'IAE'
            ))
            print(('{:6.1f} {:6.2f} {:6.2f} ' + \
                    '{:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f}').format( \
                        tm[i],Tsp1[i],T1[i],Q1[i],Tsp2[i],T2[i],Q2[i],iae))

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
    # 에피소드 종료 후 그래프 저장
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
    plt.savefig(f'episode_{epi}_plot.png')
    plt.close()

# 종료
a.Q1(0)
a.Q2(0)
a.close()
print("All episodes finished.")
