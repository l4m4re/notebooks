from math import sqrt, pi

from scipy import optimize
from scipy.optimize import Bounds

# Constants
c     = 299792458.0            # [m/s] - speed of light
k     = c**2                   # dynamic viscosity [m^2/s]
h     = 6.62607015e-34         # Planck's constant [J-s], [kg-m^2/s]
m     = h / k                  # elemental mass
q     = 1.602176634e-19        # Elementary charge (C)
k_air = 1.00058986
e     = q  # * sqrt(k_air)     # Elementary charge (C)
eta   = 1 / (4 * pi * 1e-7)    # viscosity [Pa-s], [kg/m-s], [N-s/m^2], [J-s/m^3]
rho   = eta/k                  # 8.854187817620389e-12

"""

https://en.wikipedia.org/wiki/Fine-structure_constant

Further refinement of the experimental value was published by the end of 2020,
giving the value ⁠1/α⁠ = 137.035999206(11),

with a relative accuracy of 8.1×10−11, which has a significant discrepancy from
the previous experimental value

"""

#alpha_codata = 0.007297352564311 # fine structure constant
alpha_codata = 1/137.03599920611

# Experimental values https://en.wikipedia.org/wiki/Anomalous_magnetic_dipole_moment
a_e  = 0.00115965218059    # electron magnetic moment anomaly
a_mu = 0.00116592059       # muon magnetic moment anomaly
#mma  = 1.001165923         # Stowe's magnetic moment anomaly
#mma  = 1.001159652181      # ChatGPT magnetic moment anomaly
mma  = 1 + a_e


# Define initial beta using high precision
#initial_beta = (m * c * eta) / (4 * pi**2 * sqrt(3) * e**2) - 1
#initial_beta = 0.002333205376442 # Stowe's mmma^2 - 1
#initial_beta = 0.003

vcmb = 620*1000
#vcmb = 611.1*1000
initial_beta = vcmb/c

print(f"Initial beta            : {(initial_beta)}")

#he galaxy group that includes our own Milky Way galaxy — appears to be moving at 620±15 km/s
#beta_upper = sqrt(((620+15)*1000)**2/c**2)
#beta_lower = sqrt(((620-15)*1000)**2/c**2)

beta_upper = (635*1000)/c
beta_lower = (605*1000)/c

#print(f"beta_lower              : {(beta_lower)}")
#print(f"beta_upper              : {(beta_upper)}")


#k_air = 3/( 1/(2*alpha_codata * (2 *pi * mma)**2) )**2

upper_k = 1.00058986+0.00000050
lower_k = 1.00058986-0.00000050

#print(f"upper_k                 : {(upper_k)}")
#print(f"lower_k                 : {(lower_k)}")

def check_k(k_air):
    #print(f"k_air                   : {(k_air)}")
    #print(f"sqrt(k_air)             : {(sqrt(k_air))}")
    #print(f"k_air**2                : {(k_air**2)}")

    if k_air > upper_k:
        print(f"k_air > upper_k")

    if k_air < lower_k:
        print(f"k_air < lower_k")

check_k(k_air)



print(f"-------------------------------------------------")
print(f"- First step: get relativistic correction right -")
print(f"-       Optimizing beta and R                   -")
print(f"-------------------------------------------------")


# e*e = rho * k_air * k * pi * R* 1/sqrt(1 - beta**2)  * (m * c * alpha * sqrt(1 + beta**2)) / (pi * R )

# (1 - beta) / sqrt(1 - beta^2) =  sqrt(1 - beta)  / sqrt(1 + beta) = sqrt((1-beta)/(1+beta))

# (1 - beta)   sqrt(1 + beta)   =  sqrt(1 - beta^2)  sqrt(1 - beta)

# 1 /( 1 + beta) = sqrt(1 - beta) / sqrt(1 - beta^2)  = sqrt((1-beta)/(1+beta))

# alpha = 1 / (8 * pi**2 * sqrt(3/k_air) * (1 + beta))

# base1: return rho * k_air * k * pi * 1/sqrt( 1 - beta**2 )
# base2: return (m * c) / (4 * pi**3 * pi/2 * 1/sqrt(1 + beta**2) )


def f(alpha,beta, R, k_air, return_all=False):
    #ee = ( (2 * alpha * rho * k_air * h * c ) )
    ee = ( (2 * alpha * rho * h * c ) )

    #e1 = (       rho * k       *  pi * R  ) *   sqrt(1 - beta**2)
    #e2 = ((2 * m * c * alpha ) / (pi * R) ) /   sqrt(1 - beta**2)

    e1 = (     eta               *  pi * R  ) *   sqrt(1 - beta**2)
    e2 = ((2 * h/k * c * alpha ) / (pi * R) ) /   sqrt(1 - beta**2)

    #ret = abs(e*e - e1 * e2)
    #ret = abs(ee - e1 * e2)
    ret = abs(2*e*e - ee - e1 * e2)
    
    #print(f"ee: {(ee)}, sqrt(ee): {(sqrt(ee))}, e1: {(e1)}, e2: {(e2)}")
    #print(f"alpha: {(alpha)}, beta: {(beta)}, R: {(R)}, k_air: {(k_air)}, ret: {(ret)}")

    if return_all:
        return ee, e1, e2
    return ret

def target(x):
    return f(alpha=x[0],beta=x[1], R=x[2], k_air=x[3])

R_initial = 6.405e-26

first_guess = [alpha_codata, initial_beta, R_initial, 1.0005898]

#print(f"First guess             : {(first_guess)}")

#fac = 1+1.6e-10
fac = 1+1.8e-11

bounds = Bounds(
    [alpha_codata/fac, beta_lower, 6.3e-26, 1.00058986 - 0.00000050],
    [alpha_codata*fac, beta_upper, 6.5e-26, 1.00058986 + 0.00000050]
)


res = optimize.minimize( target, x0=first_guess, bounds=bounds, method='nelder-mead', tol=1e-15,
                         options={'disp': False, 'maxiter': 100000, 'adaptive': True })

optimized_alpha = res.x[0]
optimized_beta  = res.x[1]
optimized_R     = res.x[2]
optimized_k_air = res.x[3]


# Print results
print(f"Optimized beta          : {(optimized_beta)}")

ee, e1, e2 = f(alpha=optimized_alpha,beta=optimized_beta, R=optimized_R, k_air=optimized_k_air, return_all=True)

print(f"q                       : {sqrt(ee)}")
print(f"e/q                     : {e/sqrt(ee)}")
print(f"q_linear                : {e1}")
print(f"q_angular               : {e2}")
print(f"Optimized R             : {(optimized_R)}")

vcmb = optimized_beta * c
print(f"vcmb                    : {(vcmb / 1000)} km/s")

# The Earth's velocity relative to the CMB varies between ∼340 km/s∼340km/s and
# ∼400 km/s∼400km/s, depending on the time of year. 

# The Sun is moving at v⃗Sun→CMB = 369.82±0.11 km/s towards the constellation
# Leo.

print(f"Optimized alpha         : {(optimized_alpha)}")
print(f"alpha / alpha_codata    : {(optimized_alpha / alpha_codata)}")
#print(f"alpha_codata / alpha    : {(alpha_codata / optimized_alpha)}")


# \[
#   1 + \beta = \sqrt{1 - \beta} \cdot \sqrt{1 + \beta}
# \]

#  1 + beta = sqrt(1 - beta) * sqrt(1 + beta)

# e*e = rho * k * pi * R* sqrt(1 - beta)  * (m * c * alpha) / (pi * R * sqrt(1 + beta) )


# alpha_stowe = 1/(2*sqrt(3/k_air)*(2*pi*mma)**2)


# The way we come to these equation is to start from the definition of the fine
# structure constant within our model:

# e*e = 2 alpha rho h c

# multiply by m/m, pi/pi and R/R and re-arrange:

# e*e = (rho h/m * pi * R) * ( 2 * alpha * m * c / (pi * R)).

# substitute h/m = k and apply relativistic corrections:

# e*e =  (eta * pi * R) * sqrt(1 - beta**2) * ((2 * h/k * c * alpha )/ (pi * R) ) / sqrt(1 - beta**2)

# (eta * pi * R) * sqrt(1 - beta**2) = ((2 * h/k * c * alpha )/ (pi * R) ) / sqrt(1 - beta**2)

#   e1 = (     eta               *  pi * R  ) *   sqrt(1 - beta**2)
#   e2 = ((2 * h/k * c * alpha ) / (pi * R) ) /   sqrt(1 - beta**2)

print(f"-------------------------------------------------")
print(f"Refinement...")
print(f"-------------------------------------------------")

fac1 = sqrt(3)
fac2 = (pi * (32 / 17)) * ((sqrt(2) - 1) / sqrt(2))
fac3 = 2*pi*((sqrt(2) - 1) / sqrt(2)) 

print(f"fac1                   : {(fac1)}")
print(f"fac2                   : {(fac2)}")
print(f"fac2 / fac1            : {(fac2 / fac1)}")
print(f"fac1 / fac2            : {(fac1 / fac2)}")

geom_fac = 4 * pi**2 * fac2


def Alpha(beta):
    #return sqrt(1-beta**2) / (8 * pi**2 * sqrt(3))
    #return 1 / (8 * pi**2 * sqrt(3) * sqrt(1-beta**2) )
    #return 1 / (8 * pi**2 * sqrt(3) )
    #return 1 / (8 * pi**2 * sqrt(3) * (beta + 1))

    # with this, the initial value for beta results in alpha being very close to the CODATA value
    #return 1 / (8 * pi**2 * sqrt(3) * (1 + beta))
    #return 1 / (8 * pi**2 * sqrt(3/k_air) * (1 + beta))

    # with this, the optimized value for beta results in alpha being very close to the CODATA value
    #return 1 / (8 * pi**2 * sqrt(3) * (1 + beta)/sqrt(1-beta**2) )

    #return 1 / (8 * pi**2 * sqrt(3/k_air) * (1 + beta)/sqrt(1-beta**2) )
    #return 1 / (8 * pi**2 * sqrt(3/k_air) / sqrt(1-beta**2) )
    #return 1 / (8 * pi**2 * sqrt(3/k_air) * (1+beta) )
    #return 1 / (8 * pi**2 * sqrt(3/k_air) * (1+beta) )
    #return 1 / (8 * pi**2 * pi/2 / sqrt(1-beta**4) )
    #return 1 / (8 * pi**2 * sqrt(3) / sqrt(1-beta**4) )
    #return base1(beta,k_air) / base2(beta,k_air)

    #return (1 - beta**2) / (8 * pi**2 * vl_vt )   
    #return 1 / (2 * geom_fac * (1 + beta) )  

    #corr = sqrt(1 - beta**2)
    # alpha / alpha_codata    : 1.0020405478631944
    # alpha_codata / alpha    : 0.99796360749318

    #corr = sqrt(1 + beta**2)
    # alpha / alpha_codata    : 1.0020360041360523
    # alpha_codata / alpha    : 0.9979681327540643

    #corr = (1 + beta)**2 
    # alpha / alpha_codata    : 0.9977843219661409
    # alpha_codata / alpha    : 1.002220598164434

    #corr = (1 + beta) 
    # alpha / alpha_codata    : 0.9999090367601923
    # alpha_codata / alpha    : 1.0000909715148714

    #corr = (1 + beta) / sqrt(1 - beta**2)
    # alpha / alpha_codata    : 0.9999067697215523
    # alpha_codata / alpha    : 1.0000932389711428

    #corr = (1 + beta) 
    # alpha / alpha_codata    : 0.9999090367601923
    # alpha_codata / alpha    : 1.0000909715148714

    #corr = (1 - beta)/sqrt(1+beta**2)
    # alpha / alpha_codata    : 1.0041788794281883
    # alpha_codata / alpha    : 0.995838510932865

    #corr = sqrt((1 + beta**2) /(1 - beta**2))
    # alpha / alpha_codata    : 1.0020337322750563
    # alpha_codata / alpha    : 0.997970395397329

    #corr = ((1 + beta**2) /(1 - beta**2))
    # alpha / alpha_codata    : 1.0020291885788193
    # alpha_codata / alpha    : 0.9979749206889899

    #corr = ((1 + beta) /(1 - beta))
    # alpha / alpha_codata    : 0.9977797975284877
    # alpha_codata / alpha    : 1.0022251427389208


    #corr = 1/(1 - beta)
    # alpha / alpha_codata    : 0.9999045026880526
    # alpha_codata / alpha    : 1.000095506432555

    #orr = sqrt( (1 + beta) / (1 - beta) )

    corr = (1 - beta**2)

    return 1 / (2 * geom_fac * corr )  


#   e1 = (     eta               *  pi * R  ) *   sqrt(1 - beta**2)
#   e2 = ((2 * h/k * c * alpha ) / (pi * R) ) /   sqrt(1 - beta**2)
#   e1/e2 - 1


def base1(beta, k_air=k_air):
    #return eta * pi * sqrt(1 - beta)
    #return eta * pi * ( 1 - sqrt(beta) )
    #return eta * pi * ( 1 - beta )
    #return rho * k_air * k * pi * 1/sqrt( 1 - beta**2 )
    #return rho * k_air * k * pi * ( 1 - sqrt(beta) )
    #return rho * k_air * k * pi * sqrt(1 - beta**2)
    return rho * k * pi # * sqrt(1 - beta**2)


def base2(beta, k_air=k_air):
    #return (m * c) / (4 * pi**3 * sqrt(3) * sqrt(1 + beta))
    #return (m * c) / (4 * pi**3 * sqrt(3) * (1 + sqrt(beta)) )
    #return (m * c) / (4 * pi**3 * sqrt(3) * (1 + beta) )
    #return (m * c) / (4 * pi**3 * sqrt(3) * sqrt(1 + beta) )
    #return (m * c) / (4 * pi**3 * sqrt(3/k_air) * sqrt(1 + beta) )
    #return (m * c) / (4 * pi**3 * sqrt(3/k_air) * (1 + sqrt(beta)))
    #return (m * c) / (4 * pi**3 * pi/2 * 1/sqrt(1 + beta**2) )
    return (m * c) / (pi * geom_fac) # / sqrt(1 - beta**2)

def R_1(beta):
    return e / base1(beta)

def R_2(beta):
    return base2(beta)/ e

def Q1(beta, R, k_air=k_air ):
    return base1(beta,k_air=k_air) * R

def Q2(beta, R, k_air=k_air):
    return base2(beta,k_air) / R

def QQ(beta, k_air=k_air):
    return base1(beta,k_air=k_air) * base2(beta,k_air) 



def radii_difference(beta):
    #print(f"beta: {(beta)}", end="")
    #print(f"beta: {(beta)}", end="")
    #ret = R_1(beta) / R_2(beta) - 1
    #print(f", ret: {(ret)}") 
    #return ret
    return (R_1(beta) - R_2(beta))

def q_term_difference(beta,R, k_air=k_air):
    ret = abs(2*e*e - QQ(beta,k_air) - Q1(beta,R,k_air) * Q2(beta,R,k_air))
    #print(f"beta: {(beta)}, R: {(R)}, ret: {(ret)}")
    return ret

alpha = Alpha(optimized_beta)

print(f"alpha                   : {(alpha)}")
print(f"alpha / alpha_codata    : {(alpha / alpha_codata)}")
print(f"alpha_codata / alpha    : {(alpha_codata / alpha)}")

q1 = Q1(optimized_beta, optimized_R)
q2 = Q2(optimized_beta, optimized_R)

qq = q1 * q2

print(f"q                       : {sqrt(qq)}")
print(f"e/q                     : {e/sqrt(qq)}")
print(f"q_linear                : {q1}")
print(f"q_angular               : {q2}")
print(f"Optimized R             : {(optimized_R)}")


# Optimize
print(f"-------------------------------------------------")
print(f"Optimizing beta...")
print(f"-------------------------------------------------")


def objective(x):
    #return radii_difference(x[0])
    return q_term_difference(beta=x[0],R=x[1])

beta_lower = (300*1000)/c
beta_upper = (900*1000)/c

print(f"beta_lower              : {(beta_lower)}")
print(f"beta_upper              : {(beta_upper)}")

initial_beta = optimized_beta
initial_R    = optimized_R

first_guess  = [initial_beta, initial_R]

bounds = Bounds(
    [beta_lower, 6e-26 ],
    [beta_upper, 8e-26 ]
)

#geom_fac = 4 * pi**2 * sqrt(3)
def optimize_beta():
    #print(f"First guess             : {(first_guess)}")

    res = optimize.minimize( objective, x0=first_guess, bounds=bounds, method='nelder-mead', tol=1e-15,
                             options={'disp': False, 'maxiter': 100000, 'adaptive': True })

    optimized_beta  = res.x[0]
    optimized_R     = res.x[1]


    # Print results
    print(f"Optimized beta          : {(optimized_beta)}")
    print(f"Initial beta / Optimized: {(initial_beta / optimized_beta)}")
    print(f"1 / sqrt(1 - beta)      : {(1 / sqrt(1 - optimized_beta))}")
    print(f"1/sqrt(1 + beta)        : {(1 / sqrt(1 + optimized_beta))}")
    print(f"sqrt((1 + beta**2)/(1 - beta**2)): {(sqrt((1 + optimized_beta**2)/(1 - optimized_beta**2)))}")

    print(f"Optimized R             : {(optimized_R)}")

    q1 = Q1(optimized_beta, optimized_R)
    q2 = Q2(optimized_beta, optimized_R)
    qq = QQ(optimized_beta)

    print(f"q                       : {sqrt(qq)}")
    print(f"q/e                     : {e/sqrt(qq)}")
    print(f"q_linear                : {q1}")
    print(f"q_angular               : {q2}")
    print(f"Optimized R             : {(optimized_R)}")


    vcmb = sqrt(optimized_beta**2 * c**2)
    print(f"vcmb                    : {(vcmb / 1000)} km/s")

    gamma = 1 / sqrt(1 - optimized_beta**2)
    alpha = Alpha(optimized_beta)

    return alpha

alpha = optimize_beta()

print(f"alpha                   : {(alpha)}")
print(f"alpha / alpha_codata    : {(alpha / alpha_codata)}")
print(f"alpha_codata / alpha    : {(alpha_codata / alpha)}")




print(f"-------------------------------------------------")
print(f"Other stuff...")
print(f"-------------------------------------------------")


print(f"mma                     : {(mma)}")
print(f"mma^2                   : {(mma**2)}")
print(f"a_e + 1                 : {(a_e + 1)}")
print(f"a_mu + 1                : {(a_mu + 1)}")

"""


# http://www.tuks.nl/pdf/Reference_Material/Paul_Stowe/The%20Fine%20structure%20constant%20-%20Paul%20Stowe.pdf 
#
# >> And, an alternate theory to QED (classical Continuum Mechanics based)
# >> suggests a variable FSC, given by the equation:
# >>
# >>           ___
# >>  1       / 3    /      \2
# >> --- = 2 /  -   | 2piMMA |
# >> FSC    V   k    \      /
# >>
# >> Where MMA is the Magnetic Moment Anomaly (1.001165923) and k is the
# >> dielectric constant of the bulk material in which its measurement is
# >> made. With air, k ~= 1.0006, thus we get:
# >>
# >>
# >> 2 sqrt(3/1.0006)(2pi[1.001165923])^2 = 137.03523
# >>
# >> Accurate to 0.0005%. The remaining inaccuracy can be attributed to
# >> that of the measurement of k...

# k_air = mpf('1.0006') # stowe
# 1.00058986±0.00000050 (at STP, 900 kHz), 
# https://doi.org/10.1063%2F1.1745374
# "A dielectric constant of 1.00058986±0.00000050 is found for air at normal pressure and temperature"
#k_air = mpf('1.00058986')

alpha_stowe = 1/(2*sqrt(3/k_air)*(2*pi*mma)**2)

print(f"alpha_stowe               : {(alpha_stowe)}")
print(f"alpha_stowe / alpha_codata: {(alpha_stowe / alpha_codata)}")
print(f"alpha_codata / alpha_stowe: {(alpha_codata / alpha_stowe)}")
"""

"""
So, it seems that from the Rankine vortex model, we can derive the following:

v_t = ω R = k/(2 pi R)

k = 2 π R^2 ω_t

k = 2 π R c



v = k/(4*pi)) 



hc = 2 * pi * optimized_R * eta



print(f"h                       : {(h)}")
print(f"hc                      : {(hc)}")
print(f"hc / h                  : {(hc / h)}")


"""