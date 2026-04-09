from numba import njit, prange
from qkd_1decoy_analysis_numba import * 
import numpy as np
from  matplotlib import pyplot as pl
from matplotlib.ticker import AutoMinorLocator, MultipleLocator


@njit(parallel=True)
def scan_loop(d_arr, Edet, mu2_scan, mu1_scan, pZ_scan, p1_scan):
    n_d = len(d_arr)
    n_e = len(Edet)
    
    # preallocate 2D arrays
    Mu1_opt = np.zeros((n_d, n_e))
    Mu2_opt = np.zeros((n_d, n_e))
    P_mu1_opt = np.zeros((n_d, n_e))
    PZ_opt = np.zeros((n_d, n_e))
    Skr = np.zeros((n_d, n_e))
    
    
    for di in prange(len(d_arr)):
        d = d_arr[di]
        for ei in range(len(Edet)):
            e = Edet[ei]

            skr_max = 0.0
            mu1_opt = 0.0
            mu2_opt = 0.0
            p_mu1_opt = 0.0
            pZ_opt = 0.0

            for mu2_v in mu2_scan:
                for mu1_v in mu1_scan:
                    if mu1_v <= mu2_v:
                        continue
                    for pZ_v in pZ_scan:
                        for p1_v in p1_scan:
                            skr_tmp = compute_all_fast(d, e, p1_v, mu1_v, mu2_v, pZ_v)
                            if skr_tmp > skr_max:
                                skr_max = skr_tmp
                                mu1_opt = mu1_v
                                mu2_opt = mu2_v
                                p_mu1_opt = p1_v
                                pZ_opt = pZ_v

            Mu1_opt[di,ei] = mu1_opt
            Mu2_opt[di,ei] = mu2_opt
            P_mu1_opt[di,ei] = p_mu1_opt
            PZ_opt[di,ei] = pZ_opt
            Skr[di,ei] = skr_max
            

    return Mu1_opt, Mu2_opt, P_mu1_opt, PZ_opt, Skr


mu2_scan = np.linspace(0.01, 0.9, 50)
mu1_scan = np.linspace(0.01, 1, 50)
p1_scan = np.linspace(0.01, 0.99, 50)
pZ_scan = np.linspace(0.01, 0.99, 50)
Edet = np.linspace(0,0.08,100)


Mu1_opt, Mu2_opt, P_mu1_opt, PZ_opt, Skr = scan_loop(d_arr, Edet, mu2_scan, mu1_scan, pZ_scan, p1_scan)

print(d_arr)
print(Edet)
print(Mu1_opt)
print(Mu2_opt)
print(P_mu1_opt) 
print(PZ_opt)
print(Skr)


for i in range(len(d_arr)):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.plot(Edet,Mu2_opt[i], label="mu2_opt")
    ax.plot(Edet,Mu1_opt[i], label="mu1_opt")
    ax.plot(Edet,P_mu1_opt[i], label="p_mu1_opt")
    ax.plot(Edet,PZ_opt[i], label="pZ_opt")


    ax.minorticks_on()  # activate minor ticks

    #ax.xaxis.set_minor_locator(AutoMinorLocator())
    #ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(MultipleLocator(0.005)) 
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    ax.grid(which='major', linestyle='-')
    ax.grid(which='minor', linestyle='--', linewidth=0.5)

    ax.set_xlabel("edet")

    plt.title(f"Optimal parameters for d={round(d_arr[i],2)} and nZ=10^{np.log10(nZ)} using protocol symmetyric: {Protocol_symmetric}")
    plt.legend(loc="upper right")
    plt.show()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

for i in range(len(d_arr)):
    ax.plot(Edet,Skr[i], label=f"skr at d={round(d_arr[i],2)} km" )
    ax.set_xlabel("edet")
    ax.set_ylabel("skr (bit/s)")
    ax.set_yscale('log')

plt.title(f"Secret Key Rate with optimal parameters for nZ=10^{np.log10(nZ)} using protocol symmetyric: {Protocol_symmetric}")
plt.legend(loc="upper right")
plt.show()


# for i in range(len(Edet[::2])):
#     e=Edet[2*i]
#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1)


#     ax.plot(d_arr,Mu2_opt[:,2*i], label="mu2_opt")
#     ax.plot(d_arr,Mu1_opt[:,2*i], label="mu1_opt")
#     ax.plot(d_arr,P_mu1_opt[:,2*i], label="p_mu1_opt")
#     ax.plot(d_arr,PZ_opt[:,2*i], label="pZ_opt")


#     ax.minorticks_on()  # activate minor ticks

#     #ax.xaxis.set_minor_locator(AutoMinorLocator())
#     #ax.yaxis.set_minor_locator(AutoMinorLocator())
#     #ax.xaxis.set_minor_locator(MultipleLocator(0.005)) 
#     ax.yaxis.set_minor_locator(MultipleLocator(0.1))

#     ax.grid(which='major', linestyle='-')
#     ax.grid(which='minor', linestyle='--', linewidth=0.5)

#     ax.set_xlabel("distance (km)")


#     plt.title(f"Optimal parameters for edet={round(e,3)} and nZ=10^{np.log10(nZ)}")
#     plt.legend(loc="upper right")
#     plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# for i in range(len(Edet[::2])):
#     e=Edet[2*i]
#     ax.plot(d_arr,Skr[:,2*i], label=f"skr at edet={round(e,3)}" )
#     ax.set_xlabel("distance (km)")
#     ax.set_ylabel("skr (bit/s)")
#     ax.set_yscale('log')

# plt.title(f"Secret Key Rate with optimal parameters for nZ=10^{np.log10(nZ)}")
# plt.legend(loc="upper right")
# plt.show()
