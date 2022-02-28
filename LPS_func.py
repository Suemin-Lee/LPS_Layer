#!/usr/bin/env python
# coding: utf-8

import numpy as np 
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.integrate import dblquad

import warnings
warnings.filterwarnings("ignore")
import os
if not os.path.exists('plots/'):
    os.mkdir('plots')

# import time

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
KA = 240/4.114                       #  bilayer LPS is 2*120 pN/nm and also 1kbT = 4.114 pN/nm  Smart used 4.414
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

#  ---------------------- BRUSH properties -------------------  
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



#  Which results do we want? 
#   varying (Mg),varying Brush-pep attraction, 
#   varying area per chain, varying chain length 
#   The one we examine should be equal to 1 and the rest zero   
Mg = 0  
BpAtt = 0 
area = 0 
chain = 0 
crossover = 1 
radius = 0 
radius2 = 0 
cell = 0 
  #  --------Free Parameters  
n2 = 5 * convMg   #   Mg2+ concentration in molecule/nm**3  
#  EpepB=-0.05     #  Pep-Brush weak attraction in kbT  
S = 4   #  number of sites every single brush chain occupies  
#  nr=15     #  repeat unit of Oantigen in wild LPS  
npmol=1000 * 10**(-6)    #  peptide concentration in Molar  
p_con= npmol * 0.602    #  peptide concentration in molecule/nm**3    

Ct = 1 *10**5 * convCt   #  number of total cells in mL  



#  ---------------------constants depends on n2----------------------  
def kp(n2): return np.sqrt(4 *np.pi *lB*(2*n1 + (2**2 + 2)* n2)) #  Inverse Debye length(nm) 
def Dell(n2): return 2 * (1/40 + kp(n2) *40)/(2/40 + kp(n2) * 40)   #  Dielectric discontinuity  
#  Free energy components ------- per number of sites (energy/N0)  

# def M1(n2): 
#     f1 = lambda y, x: np.exp(-kp(n2)* np.sqrt(x**2 + y**2))/np.sqrt(x**2 + y**2)
#     ans = dblquad(f1, -a/2, a/2, lambda x: -a/2, lambda x: a/2,epsabs=1.49e-08, epsrel=1.49e-08)
#     return ans[0]

def M1(n2): 
    f1 = lambda y, x: np.exp(-kp(n2)* np.sqrt(x**2 + y**2))/np.sqrt(x**2 + y**2)
    ans1 = dblquad(f1, -a/2, 0, lambda x: -a/2, lambda x: 0)
    ans2 = dblquad(f1, 0, a/2, lambda x: 0, lambda x: a/2)
    ans = ans1[0]+ans2[0]
    return ans*2

# def Mp(n2):
#     f1 = lambda y, x: np.exp(-kp(n2)* np.sqrt(x**2 + y**2))/np.sqrt(x**2 + y**2)
#     ans = dblquad(f1, -Q *a /2, Q *a/2, lambda x: -a/2, lambda x: a/2)
#     return ans[0]

def Mp(n2):
    f1 = lambda y, x: np.exp(-kp(n2)* np.sqrt(x**2 + y**2))/np.sqrt(x**2 + y**2)
    ans1 = dblquad(f1,  -Q *a /2, 0, lambda x: -a/2, lambda x: 0)
    ans2 = dblquad(f1, 0, Q *a/2, lambda x: 0, lambda x: a/2)
    ans = ans1[0]+ans2[0]
    return ans*2


def Fmean(n2, x1, x2, xp): 
    y = -lB* Dell(n2)*(Mp(n2)-M1(n2))/2 *Q*xp*  (x1+ 2*x2 +Q*xp)/(1+Q*xp)/a**2 +lB*Dell(n2)*(np.pi/kp(n2)- M1(n2)/2)*        (1-x1- 2*x2- Q*xp)**2/(a**2*(1 + Q*xp))
    return y

def Entro(n2, x1, x2, xp): 
    y1 = x1 * np.log(x1/(1 + Q * xp)/(n1*  v1)) +   x2* np.log(x2/(1 + Q * xp)/(n2*  v2))
    y2 = xp * np.log(Q*  xp/(1 + Q * xp)) 
    y3 =(1 + Q * xp - x1 - x2 - Q * xp)* np.log(1 - (x1 + x2 + Q*  xp)/(1 + Q * xp)) 
    y4 = (1 - Q)/ Q * (1 + Q*  xp - Q * xp)*  np.log(1 - Q * xp/(1 + Q * xp))  
    y5 = -xp *  (esp + 1) - 1/Q * (1 + Q*  xp)*  np.log( 1 - Q*  xp/(1 + Q*  xp)) 
    y6 = ((1 + Q * xp)/Q)*   esp/(1 - Q * xp/(1 + Q*  xp))
    y =y1+y2+y3+y4+y5+y6
    return  y


def Fbulk(n2, x1, x2, xp):
    y =   x1 * lB * (Dell(n2) - 1)/2/del1 + x1 * lB * kp(n2)/(1 + kp(n2) * r1)/2          + x2 *2**2 *lB * (Dell(n2) - 1)/2/del2          + x2 *2**2 *lB *kp(n2)/(1 + kp(n2) * r2)/2          + xp *Q *lB * (Dell(n2) - 1)* (1/delp + (Mp(n2) - M1(n2))/a**2)/2          + xp *Q *lB * kp(n2)/(1 + kp(n2) * r1)/2 
    return y


def sum_fun(n2):
    x1_i_ls =[]
    for i in range(1,11):
        for j in range(0,i):
            x1_i = 1/2 *(-1)**(i + j - 1)*np.exp(-kp(n2) *a *np.sqrt(i**2 + j**2))/(a* np.sqrt(i**2 + j**2))* 8
            x1_i_ls.append(x1_i)
    
    x1_j_ls =[]
    for i in range(1,11):
        x1_j = 1/2 *(-1)**(i - 1)*np.exp(-kp(n2)* a *i)/(a *i)* 4 
        x1_j_ls.append(x1_j)

    x1_k_ls =[]                
    for i in range(1,11):   
        for j in range(1,i):
            x1_k = 1/2 *(-1)**(i + i - 1)*np.exp(-kp(n2)* a *np.sqrt(2) *i)/(a *np.sqrt(2)* i)* 4
            x1_k_ls.append(x1_k)
                
    y =  sum(x1_i_ls) - sum(x1_j_ls)- sum(x1_k_ls)
    return y

def Flateral(n2, x1, x2, xp): 
    y = -lB * Dell(n2) *2 *x2 *(1 + Q *xp - x1 - x2 - Q *xp)/(1 + Q *xp)* sum_fun(n2)
    return y

def Ftrans(n2, x1, x2, xp): 
    y =  -lB * Dell(n2)* x1/del1 - lB * Dell(n2) *2* x2/del2 - lB* Dell(n2)* Q* xp/delp
    return y



#=========================================================================
#*********************** Brush Free Energy ******************************
#=========================================================================


# (*primary*)
def FBrushS(S, EpepB, xp):
    y = (xp/2)* ( phiB0(S, xp)**3/vb(db) *vpHlx(Rp, Lp)                
        + phiB0(S, xp)**2/vb(db)**(2/3) *ApHlx(Rp, Lp)              
        + phiB0(S, xp)/vb(db)**(2/3) *ApHlx(Rp, Lp) *EpepB )
    return y

# (*ternary*)
def FBrush(S, EpepB, xpB, xp, Rp):
    y = xpB *(phiB0(S, xp)**3/vb(db)* vpSph(Rp)         + phiB0(S, xp)/vb(db)*(2/3)* ApSph(Rp)* EpepB         + phiB0(S, xp)**2/vb(db)**(2/3) * ApSph(Rp))
    return y

# (*ternary entropy*)
def EntB(S, nr, Rp, xpB, xp):
    y  = xpB * np.log(phipB(S, nr, Rp, xp, xpB)) + (H0(S, nr, xp)* a**2* (1 + Q * xp) - vpSph(Rp)*  xpB) * np.log(1 - phipB(S, nr, Rp, xp, xpB)) 
    return y



# Free peptide's entropy*)
def EntFreeP1(p_con, xp, Ct):
    y  = 1/(Ct* V) *(((p_con* V*a**2)/Ab - Ct*V*xp)
        *np.log(vp *(p_con - (Ab/a**2) *Ct * xp)) 
        - ((p_con*V*a**2)/Ab - Ct*V*xp))
    return y

# (*used for Smart's LPS*)
def EntFreeP(p_con, xp, xpB, Ct):
    y = 1/(Ct* V)* (((p_con* V* a**2)/Ab - Ct *V* (xp + xpB))*np.log(vp*(p_con - (Ab/a**2)* Ct* (xp + xpB))) - ((p_con*V*a**2)/Ab - Ct *V *(xp + xpB)))
    return y

# (*Total Free energy*)
# (* LPS *)
def Ftotal1(p_con, n2, x1, x2, xp, Ct):
    y =  Entro(n2, x1, x2, xp) + Fmean(n2, x1, x2, xp)     + Flateral(n2, x1, x2, xp) + Ftrans(n2, x1, x2, xp)     + Fbulk(n2, x1, x2, xp) + H *xp + a**2* KA *(Q *xp)**2/2     + EntFreeP1(p_con, xp, Ct)
    return y
# (*LPS + Brush*)
def Ftotal2(S, nr, EpepB, p_con, n2, x1, x2, xp, xpB, Rp, Ct): 
    y = Entro(n2, x1, x2, xp) + Fmean(n2, x1, x2, xp)         + Flateral(n2, x1, x2, xp) + Ftrans(n2, x1, x2, xp)         + Fbulk(n2, x1, x2, xp) + H *xp + a**2* KA *(Q *xp)**2/2         + FBrushS(S, EpepB, xp) + FBrush(S, EpepB, xpB, xp, Rp)         + EntB(S, nr, Rp, xpB, xp) + EntFreeP(p_con, xp, xpB, Ct)
    return y



# LPS no brush Mg variations
# LPS + brush with Mg variations
# # (* Absorption Vs. Cp (smart model) \*)
mg_range =[1,5,10]

Mg_variations ={'No_Brush_Mg':[],'LPS_Brush_Mg':[]}
p_con_range = np.r_[0.001*conv:20.0*conv:0.1*conv]
# n2_range = [[1* convMg, 5 *convMg, 10 *convMg ]]
n2_range = [mg*convMg for mg in mg_range]

tolerance =0.00001

# LPS no brush Mg variations
for i,n2 in enumerate(n2_range): 
    ans = []
    for p_con in p_con_range:
        bnds = ((None, 1), (None, 1/2), (None, 1/Q))
        # sol = opt.minimize(lambda x: Ftotal1(p_con, n2, x[0], x[1], x[2], Ct),[x1_b, x2_b, xp_b], method='SLSQP',bounds=bnds)
        sol = opt.minimize(lambda x: Ftotal1(p_con, n2, x[0], x[1], x[2], Ct),(0.2,0.1,0.01),tol=tolerance,method='SLSQP',bounds=bnds)
        ans.append(sol.x)
        print(sol.x)
    Mg_variations['No_Brush_Mg'].append(ans)

# LPS + brush with Mg variations
nr, EpepB =15,-0.1;
for n2 in n2_range: 
    ans = []
    for p_con in p_con_range:
        bnds = ((None, 1), (None, 1/2), (None, 1/Q),(0, 0.5))
        sol = opt.minimize(lambda x: Ftotal2(S, nr, EpepB, p_con, n2, x[0], x[1], x[2],x[3], Rp, Ct),
                           (0.2,0.1,0.01,0.000001),tol=tolerance,method='SLSQP',bounds=bnds)        
        ans.append(sol.x)
        print(sol.x)
    Mg_variations['LPS_Brush_Mg'].append(ans)
    



# plot for figure(4) with no brush term
# Combined into one figure 

new_p_con_range = p_con_range/conv

plt.figure(figsize=(10,4.5))
# QNp/N0
plt.subplot(1,2,1)
for j,mg in enumerate(mg_range):
    noB_Mg_np =[]
    B_Mg_np =[]
    for i,p in enumerate(new_p_con_range):
        tmp1 = Q*Mg_variations['No_Brush_Mg'][j][i][2]
        tmp2 = Q*Mg_variations['LPS_Brush_Mg'][j][i][2]
        noB_Mg_np.append(tmp1) 
        B_Mg_np.append(tmp2) 
    plt.plot(new_p_con_range,noB_Mg_np,':o',label='No Brush Mg = %1.0i mM'%mg)        
    plt.plot(new_p_con_range,B_Mg_np,'-*',label='Brush :Mg = %1.0i mM'%mg,color=plt.gca().lines[-1].get_color())
    
    plt.ylabel('$QN_P/N_0$',fontsize='14')
    plt.xlabel('AMP [$\mu M$]',fontsize='14')
    plt.legend()

# 2N_2/N0
plt.subplot(1,2,2)    
for j,mg in enumerate(mg_range):
    noB_Mg_np =[]
    B_Mg_np =[]
    for i,p in enumerate(new_p_con_range):
        tmp1 = Q*Mg_variations['No_Brush_Mg'][j][i][1]
        tmp2 = Q*Mg_variations['LPS_Brush_Mg'][j][i][1]
        noB_Mg_np.append(tmp1) 
        B_Mg_np.append(tmp2) 
    plt.plot(new_p_con_range,noB_Mg_np,':o',label='No Brush Mg = %1.0i mM'%mg)        
    plt.plot(new_p_con_range,B_Mg_np,'-*',label='Brush :Mg = %1.0i mM'%mg,color=plt.gca().lines[-1].get_color())

    plt.ylabel('$2N_2/N_0$',fontsize='14')
    plt.xlabel('AMP [$\mu M$]',fontsize='14')
    plt.legend()

plt.tight_layout()
plt.savefig('plots/'+('Mg_variation_plot')+'.pdf',bbox_inches='tight')



