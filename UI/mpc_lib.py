
from APMonitor.apm import *

_APM_SERVER = 'http://byu.apmonitor.com'
_APM_APP    = 'my_MPC'

def _mpc_solve(T1_meas, T1_sp, T2_meas, T2_sp):
    # 측정값 입력
    apm_meas(_APM_SERVER, _APM_APP, 'TC1', T1_meas)
    apm_meas(_APM_SERVER, _APM_APP, 'TC2', T2_meas)

    DT = 0.1
    apm_option(_APM_SERVER, _APM_APP, 'TC1.sphi', T1_sp + DT)
    apm_option(_APM_SERVER, _APM_APP, 'TC1.splo', T1_sp - DT)
    apm_option(_APM_SERVER, _APM_APP, 'TC2.sphi', T2_sp + DT)
    apm_option(_APM_SERVER, _APM_APP, 'TC2.splo', T2_sp - DT)

    output = apm(_APM_SERVER, _APM_APP, 'solve')

    if apm_tag(_APM_SERVER, _APM_APP, 'apm.appstatus') == 1:
        Q1 = apm_tag(_APM_SERVER, _APM_APP, 'Q1.Newval')
        Q2 = apm_tag(_APM_SERVER, _APM_APP, 'Q2.Newval')
    else:
        print(output)
        Q1, Q2 = 0, 0

    return Q1, Q2

class MPCController:
    def __init__(self, history_horizon: int = 600):
        apm(_APM_SERVER, _APM_APP, 'clear all')
        apm_load(_APM_SERVER, _APM_APP, 'control.apm')
        csv_load(_APM_SERVER, _APM_APP, 'control.csv')

        apm_info(_APM_SERVER, _APM_APP, 'MV', 'Q1')
        apm_info(_APM_SERVER, _APM_APP, 'MV', 'Q2')
        apm_info(_APM_SERVER, _APM_APP, 'CV', 'TC1')
        apm_info(_APM_SERVER, _APM_APP, 'CV', 'TC2')

        apm_option(_APM_SERVER, _APM_APP, 'apm.imode',    6)
        apm_option(_APM_SERVER, _APM_APP, 'apm.solver',   3)
        apm_option(_APM_SERVER, _APM_APP, 'apm.hist_hor', history_horizon)

        apm_option(_APM_SERVER, _APM_APP, 'Q1.dcost', 1.0e-4)
        apm_option(_APM_SERVER, _APM_APP, 'Q1.cost',   0.0)
        apm_option(_APM_SERVER, _APM_APP, 'Q1.dmax',   50)
        apm_option(_APM_SERVER, _APM_APP, 'Q1.upper', 100)
        apm_option(_APM_SERVER, _APM_APP, 'Q1.lower',   0)

        apm_option(_APM_SERVER, _APM_APP, 'Q2.dcost', 1.0e-4)
        apm_option(_APM_SERVER, _APM_APP, 'Q2.cost',   0.0)
        apm_option(_APM_SERVER, _APM_APP, 'Q2.dmax',   50)
        apm_option(_APM_SERVER, _APM_APP, 'Q2.upper', 100)
        apm_option(_APM_SERVER, _APM_APP, 'Q2.lower',   0)

        apm_option(_APM_SERVER, _APM_APP, 'TC1.tau',     10)
        apm_option(_APM_SERVER, _APM_APP, 'TC2.tau',     10)
        apm_option(_APM_SERVER, _APM_APP, 'TC1.tr_init',  1)
        apm_option(_APM_SERVER, _APM_APP, 'TC2.tr_init',  1)

        apm_option(_APM_SERVER, _APM_APP, 'Q1.status',     1)
        apm_option(_APM_SERVER, _APM_APP, 'Q2.status',     1)
        apm_option(_APM_SERVER, _APM_APP, 'TC1.status',    1)
        apm_option(_APM_SERVER, _APM_APP, 'TC2.status',    1)

        apm_option(_APM_SERVER, _APM_APP, 'Q1.fstatus',    0)
        apm_option(_APM_SERVER, _APM_APP, 'Q2.fstatus',    0)
        apm_option(_APM_SERVER, _APM_APP, 'TC1.fstatus',   1)
        apm_option(_APM_SERVER, _APM_APP, 'TC2.fstatus',   1)

        apm_option(_APM_SERVER, _APM_APP, 'apm.web_plot_freq', 3)

    def step(self, current_temp: list[float], target_temp: list[float], dt: float):
      
        return _mpc_solve(
            T1_meas=current_temp[0], T1_sp=target_temp[0],
            T2_meas=current_temp[1], T2_sp=target_temp[1]
        )


def mpc_init(history_horizon: int = 600) -> MPCController:

    return MPCController(history_horizon)