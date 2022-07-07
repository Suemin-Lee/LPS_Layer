
import numpy as np 
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from LPS_function import *

import warnings
warnings.filterwarnings("ignore")
import os
if not os.path.exists('plots/'):
    os.mkdir('plots')




#   varying (Mg),varying Brush-pep attraction, 
#   varying area per chain, varying chain length 

#  --------------Variables -------------------------\
NA = 6.023 *10**23                   #  Avagadro's number
V =  10**(+24)                       #  Total volume 1L to nm**3
conv = NA *10**(-6)* 10**(-24)       #  convert microMolar to molecules/nm**3
convMg = 0.602 *10**(-3)             #  convert miliMolar to molecules/nm**3
convCt = 10**(-21)                   #  cells/mL to cells/nm**3
lB = 0.7                             #  Bejurium Length (nm)
aLPS = 1.66                          #  cross sectional area of LPS (nm)**2               
a = 0.64                             #  Site length in nm
Q = 4                                #  Peptide's charge number
KA = 200/4.114                       #  bilayer LPS is 2*120 pN/nm and also 1kbT = 4.114 pN/nm  Smart used 4.414
n1mol = 0.1                          #  Salt concentration in Molar unit
n1 = n1mol *0.602                    #  Salt concentration in molecule/nm**3
del1 = 0.3                           #  Gap between ion and lipid interface (nm)
del2 = 0.25                          #  Gap between divalent ion and lipid interface (nm)
delp = 0.4                           #  Gap between peptide and lipid interface (nm)
r1 = 0.34                            #  hydrated radius monovalent (nm)
r2 = 0.43                            #  hydrated radius divalent (nm)
v1 = 4/3*np.pi*(r1)**3               #  free ion volume (nm)**3
v2 = 4/3*np.pi*(r2)**3               #  free ion volume (nm)**3
esp =  25/4/np.pi                    #  Shape parameter for excluded volume entropy
H = -10                              #  Hydrophobic energy magenin II
Ab =  2 *6 *10**(-12) *10**18        #   2 times of E-coli's area in nm**2

#  ------------ PEPTIDE Properties:Maganin--------------  
vp = 2.5                             #  volume of free peptide coil (nm)**3  
Rp = 0.6                             #  radius of the cap of cylinder peptide  
Lp = 2.2                             #  length of cylinder peptide  
Npp = 23                             #  number of amino acids  
dp = 0.35                            #  diameter of amino acids (nm)  
Q = 4        



def vpHlx(Rp, Lp): return np.pi* Rp**2 *Lp     #  cylinder volume alpha helical magainin nm**3  
def ApHlx(Rp, Lp): return 2 *np.pi *Rp *Lp + 2 *np.pi *Rp**2    #   cylinder area alpha helical magainin (nm**3)  

#  ---------------------- BRUSH properties -----------------------------------  
def vpSph(Rp): return (4/3) *np.pi *Rp**3    #  volume of sphere-maginin within brush  
def ApSph(Rp): return 4 *np.pi *Rp**2    #  area of sphere-magainin within brush  
alpha = 3/5    #  exponent scale,relation btw. hight and grafting density  
db = 0.85    #  diameter of every monosaccharide unit (nm)  

def vb(db): return db**3   #  monomer volume nm**3  
def Nb(nr): return 4 *nr + 8 + 4    #  monomer number units of LPS Brush  
def Ach(S, xp): return S *a**2* (1 + Q *xp)     #  area per brush chain  
def sigmaB(S, xp): return db**2/Ach(S, xp)    #  Grafting density  
#  sigmaB(S,xp)=1/Ach(S,xp)    
def H0(S, nr, xp): return  Nb(nr) *db *sigmaB(S, xp)**alpha    #  equilibrium height  
def phiB0(S, xp): return sigmaB(S, xp)**(1 - alpha) #  dimentionless brush monomer volume fraction  
def phipB(S, nr, Rp, xp, xpB): return (vpSph(Rp)* xpB)/(H0(S, nr, xp) * a**2 * (1 + Q*xp))   #  dimentionless peptide volume fraction within brush  


#  ---------------- Free Parameters (Unspecified parameter values)----------------
n2 = .1 * convMg   #   Mg2+ concentration in molecule/nm**3  
EpepB=-0.05     #  Pep-Brush weak attraction in kbT  
S = 4   #  number of sites every single brush chain occupies  
nr=20000     #  repeat unit of Oantigen in wild LPS  
npmol=1000 * 10**(-6)    #  peptide concentration in Molar  
p_con= npmol * 0.602    #  peptide concentration in molecule/nm**3    

Ct = 1 *10**10 * convCt   #  number of total cells in mL  

p_con_range = np.r_[0.001*conv:20.0*conv:0.5*conv]
p_con_range = p_con_range[1:]
tolerance =0.000000001


# # EpepB variation

# ----------- Varying Brush-pep attration(EpepB) -----------

    
EpepB_variations ={'LPS_Mg_fixed':[],'LPS_Brush_Mg_EpepB':[]}

# p_con_range = np.r_[0.001*conv:20.0*conv:10*conv]
ans = []
for p_con in p_con_range:
    bnds = ((None, 1), (None, 1/2), (None, 1/Q))
    sol = opt.minimize(lambda x: Ftotal1(p_con, n2, x[0], x[1], x[2], Ct),(0.2,0.1,0.001),
                        tol=tolerance,method='SLSQP',bounds=bnds)
    EpepB_variations['LPS_Mg_fixed'].append(sol.x)
    print(sol.x)

EpepB_range=[-0.1,-1,-1.5,-2]

# EpepB_range=[-0.1]
# ,-1,-1.5,-2]

for EpepB in EpepB_range: 
    ans = []
    for i, p_con in enumerate(p_con_range):
        if i<10: 
            bnds = ((None, 1), (None, 1/2), (None, 1/Q),(0, 0.01))
            sol = opt.minimize(lambda x: Ftotal2(S, nr, EpepB, p_con, n2, x[0], x[1], x[2],x[3], Rp, Ct),
                                (0.02,0.2,0.0001,0.0001),tol=tolerance,method='SLSQP',bounds=bnds)        
        else: 
            bnds = ((None, 1), (None, 1/2), (None, 1/Q),(0, 0.5))
            sol = opt.minimize(lambda x: Ftotal2(S, nr, EpepB, p_con, n2, x[0], x[1], x[2],x[3], Rp, Ct),
                                (0.02,0.2,0.01,0.001),tol=tolerance,method='SLSQP',bounds=bnds)                                  
        ans.append(sol.x)
        print(sol.x)
    EpepB_variations['LPS_Brush_Mg_EpepB'].append(ans)

new_p_con_range = p_con_range/conv

# Plot figure(5)

plt.figure(figsize=(15,4.5))

cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']


EpepB_fixed_np=[]
EpepB_fixed_n2=[]
for i,p in enumerate(new_p_con_range):
    tmp1 = Q*EpepB_variations['LPS_Mg_fixed'][i][2]
    tmp2 = 2*EpepB_variations['LPS_Mg_fixed'][i][1]
    EpepB_fixed_np.append(tmp1)   
    EpepB_fixed_n2.append(tmp2) 
mg = n2/convMg
plt.subplot(1,3,1)    
plt.plot(new_p_con_range,EpepB_fixed_np,label=r'No Brush: $\mathregular{Mg}^{2+}$ = %1.1f mM'%mg)     




# plt.subplot(1,3,2)    
# plt.plot(new_p_con_range,EpepB_fixed_n2,label='Fixed Mg = %1.0i mM'%mg)       


plt.subplot(1,3,1)    
for j,EpepB in enumerate(EpepB_range):
    EpepB_result =[]
    for i,p in enumerate(new_p_con_range):
        tmp2 = Q*EpepB_variations['LPS_Brush_Mg_EpepB'][j][i][2]
        EpepB_result.append(tmp2)       
    plt.plot(new_p_con_range,EpepB_result,label=r'Brush: $\epsilon_\mathregular{att}$ = %1.2f $k_BT$'%EpepB,color=cycle[j+1])
    #              ,color=plt.gca().lines[-1].get_color())
    plt.ylabel('$QN_P/N_0$',fontsize='14')
    plt.xlabel('AMP [$\mu M$]',fontsize='14')
    plt.legend()

plt.subplot(1,3,2)   
for j,EpepB in enumerate(EpepB_range):
    B_Mg_trapped_brush =[]
    for i,p in enumerate(new_p_con_range):
        tmp_xp = EpepB_variations['LPS_Brush_Mg_EpepB'][j][i][2]
        tmp_xpB = EpepB_variations['LPS_Brush_Mg_EpepB'][j][i][3]
        tmp = phipB(S, nr, Rp, tmp_xp, tmp_xpB)*H0(S, nr, tmp_xp)
        B_Mg_trapped_brush.append(tmp) 
    plt.plot(new_p_con_range,B_Mg_trapped_brush,'-',label=r'Brush: $\epsilon_\mathregular{att}$ = %1.2f $k_BT$'%EpepB,color=cycle[j+1])

    plt.ylabel('$\Phi_P(N_p) \, H_\mathregular{Brush}$',fontsize='14')
    plt.xlabel('AMP [$\mu M$]',fontsize='14')
    plt.legend()  

plt.subplot(1,3,3)   
for j,EpepB in enumerate(EpepB_range):
    B_Mg_trapped_brush =[]
    for i,p in enumerate(new_p_con_range):
        tmp_xp = EpepB_variations['LPS_Brush_Mg_EpepB'][j][i][2]
        tmp_xpB = EpepB_variations['LPS_Brush_Mg_EpepB'][j][i][3]
        tmp = phipB(S, nr, Rp, tmp_xp, tmp_xpB)*H0(S, nr, tmp_xp)/tmp_xp
        B_Mg_trapped_brush.append(tmp) 
    plt.plot(new_p_con_range,B_Mg_trapped_brush,'-',label=r'Brush: $\epsilon_\mathregular{att}$ = %1.2f $k_BT$'%EpepB,color=cycle[j+1])

    plt.ylabel('$\Phi_P(N_p) \, H_\mathregular{Brush}/N_p$',fontsize='14')
    plt.xlabel('AMP [$\mu M$]',fontsize='14')
    plt.legend()  

plt.tight_layout()
plt.savefig('plots/'+'EpepB_variations_plot'+'.pdf',bbox_inches='tight')


