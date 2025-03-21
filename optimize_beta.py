from math import sqrt, pi

from scipy import optimize
from scipy.optimize import Bounds

# Constants

c     = 299792458.0          # [m/s] - speed of light
c_l   = c * (pi/2)           # [m/s] - speed of longitudinal (sound) waves
#mu    = 4*pi*1e-7           # mu_0 - 1/viscosity
mu    = 4*pi*1.00000000082e-7# mu_0 - 1/viscosity - 2019 redefinition
eta   = 1/(4*pi*1e-7)        # viscosity     [Pa-s], [kg/m-s],  [N-s/m^2], [J-s/m^3]
rho   = eta/c**2             # 8.854187817620389e-12 [kg/m^3]
k     = eta/rho              # dynamic viscosity     [m^2/s]
h     = 6.62607015e-34       # Planck's constant     [kg-m^2/s], [J-s]
q     = 1.602176634e-19      # elemental charge      [kg/s], [A-s], [C]
m     = h/k                  # elemental mass        [kg]
rho_q = q/m * rho            # charge density 1.9241747011042014e+20  [kg/m^3-s]

k_air = 1.00058986
e     = q  # * sqrt(k_air)      # Elementary charge (C)

r_e   = 2.8179403227e-15        # Classical electron radius
r_b   = 5.29177210903e-11       # Bohr radius (hydrogen atom)

N_a   = 6.02214076e23        # Avrogado's constant. 

m_e   = 9.1093837015e-31     # mass_electron
m_p   = 1.6726231e-27        # mass proton                 = 1836.152672   * m_e
m_n   = 1.674927471e-27      # mass neutron                = 1838.68363205 * m_e
m_u   = 1.66053906660e-27    # Atomic mass constant [kg]   = 1822.8884     * m_e
m_neu = 2.14e-37             # Mass neutrino
m_he4 = 4.002602*m_u         # Mass Hellium 4              = 6.64647698905e-27 [kg]



delta = eta * 4 * pi
print(f"delta                                   : {(delta)}")


def getOmegaR(m):
    R = (3 * k * m) ** (1/3)
    omega = (6*h) / (4 * pi * m * R**2)

    return omega, R


#V     = h/eta                   # = m /rho
#print(f"V                                       : {(V)}")

V    = h/delta
print(f"V                                       : {(V)}")

# electron

omega, R = getOmegaR(m_e)

print(f"omega electron                           : {(omega)}")
print(f"R electron                               : {(R)}")
print(f"R electron / r_e                         : {(R/r_e)}")

exit()





# rho * V = rho * h/eta = h/k = m

#sphere
R     = (3*V/4*pi)**(1/3)       #  V = 4/3 pi R^3
RR    = R/(4*pi)

print(f"R sphere                                : {(R)}")
print(f"R sphere / 4 pi                         : {(RR)}")

print(f"R sphere / r_e                          : {(R/r_e)}")
print(f"R sphere / (4 pi r_e)                   : {(RR/r_e)}")


# dual torus within a sphere
#R = \left(\frac{V_{\text{dual}}}{2 \pi^2 (\sqrt{2} - 1)}\right)^{1/3}.

R     = (V / (2 * pi**2 * (sqrt(2) - 1)) )**(1/3)
RR    = R/(4*pi)

print(f"R dual torus                            : {(R)}")
print(f"R dual torus / 4 pi                     : {(RR)}")

print(f"R dual torus / r_e                      : {(R/r_e)}")
print(f"R dual torus / (4 pi r_e)               : {(RR/r_e)}")


# single torus r = R/2
# for r = R/2:   V = 2 pi^2 R^3 / 4 = pi/2 R^3
# R = (V / (pi/2))**(1/3)

R     = (V / (pi/2))**(1/3)
RR    = R/(4*pi)

print(f"R single torus r = R/2                  : {(R)}")
print(f"R single torus r = R/2                  : {(RR)}")

print(f"R single torus r = R/2/r_e              : {(R/r_e)}")
print(f"R single torus r = R/2/(4 pi r_e)       : {(RR/r_e)}")



# single torus r = RR/sqrt(2)
# for R = RR (sqrt(2) -1)/sqrt(2):  
# and r = RR / sqrt(2)
# V = 2 pi^2 RR (sqrt(2) -1)/sqrt(2) RR^2 / 2 = pi^2 (sqrt(2) -1) /sqrt(2) RR^3

R    = (V / (pi**2 * (sqrt(2) - 1) / sqrt(2)))**(1/3)
RR   = R/(4*pi)

print(f"R single torus r = RR/sqrt(2)           : {(R)}")
print(f"R single torus r = RR/sqrt(2)           : {(RR)}")

print(f"R single torus r = RR/sqrt(2)/r_e       : {(R/r_e)}")
print(f"R single torus r = RR/sqrt(2)/(4 pi r_e): {(RR/r_e)}")


m_ee = m * N_a

print(f"m_ee                      : {(m_ee)}")
print(f"m_ee / m_e                : {(m_ee / m_e)}")
print(f"m_ee / m_u                : {(m_ee / m_u)}")
print(f"m_ee / m_p                : {(m_ee / m_p)}")
print(f"m_ee / m_n                : {(m_ee / m_n)}")
print(f"m_ee / m_he4              : {(m_ee / m_he4)}")

# elementary particle
h_e   = h / N_a
omega = 2 * pi * N_a
r_m   = sqrt(h_e/( (4/6)*pi*omega ) )

print(f"r_m                       : {(r_m)}")

# electron
"""
https://en.wikipedia.org/wiki/Electron

The issue of the radius of the electron is a challenging problem of modern
theoretical physics. The admission of the hypothesis of a finite radius of the
electron is incompatible to the premises of the theory of relativity. On the
other hand, a point-like electron (zero radius) generates serious mathematical
difficulties due to the self-energy of the electron tending to infinity.[92]
Observation of a single electron in a Penning trap suggests the upper limit of
the particle's radius to be 10−22 meters.[93] The upper bound of the electron
radius of 10−18 meters[94] can be derived using the uncertainty relation in
energy. There is also a physical constant called the "classical electron
radius", with the much larger value of 2.8179×10−15 m, greater than the radius
of the proton. However, the terminology comes from a simplistic calculation that
ignores the effects of quantum mechanics; in reality, the so-called classical
electron radius has little to do with the true fundamental structure of the
electron.[95][96][e]

"""

f_e = m_e * c**2 / h
l_e = h / (m_e * c)
omega = 2 * pi * f_e
r_e2   = sqrt(h/( (4/6)*pi*omega ) )

print(f"r_e2                      : {(r_e2)}")
print(f"l_e                       : {(l_e)}")

# proton
f_p = m_p * c**2 / h
l_p = h / (m_p * c)
omega = 2 * pi * f_p
r_p2   = sqrt(h/( (4/6)*pi*omega ) )

print(f"r_p2                      : {(r_p2)}")
print(f"r_p2/r_e2                 : {(r_p2/r_e2)}")



exit()



# S =   4 pi R^2
# V = 4/3 pi R^3
# S/V = 3/R

#sphere
#div_rho_V = rho * 3/R

#dual torus div_rho_V = div( rho * V ) = div( rho * h/eta ) = div( h/k ) = div( m ) 
div_rho_V = rho * (2 * sqrt(2)) / R

print(f"div_rho_V                 : {(div_rho_V)}")
print(f"div_rho_V / k             : {(div_rho_V / k)}")
print(f"div_rho_V / k**2          : {(div_rho_V / k**2)}")
print(f"div_rho_V / c             : {(div_rho_V / c)}")

grad_div_rho_V = 3 / R**2

print(f"grad_div_rho_V            : {(grad_div_rho_V)}")
print(f"grad_div_rho_V / k        : {(grad_div_rho_V / k)}")
print(f"grad_div_rho_V / k**2     : {(grad_div_rho_V / k**2)}")
print(f"grad_div_rho_V / c        : {(grad_div_rho_V / c)}")





exit()

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

#vcmb = 620*1000
vcmb = 611.111111*1000
#vcmb = 100*1000
initial_beta = vcmb/c

print(f"Initial beta            : {(initial_beta)}")

#he galaxy group that includes our own Milky Way galaxy — appears to be moving at 620±15 km/s
#beta_upper = sqrt(((620+15)*1000)**2/c**2)
#beta_lower = sqrt(((620-15)*1000)**2/c**2)

#beta_lower = (605*1000)/c
#beta_upper = (635*1000)/c

beta_lower = (100*1000)/c
#beta_lower = 1/c
beta_upper = (900*1000)/c


#print(f"beta_lower              : {(beta_lower)}")
#print(f"beta_upper              : {(beta_upper)}")




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

# 1/sqrt(1 - beta^2)  =  sqrt(1 - beta)/(sqrt(1 + beta) * (1 - beta))
# 

# 1/sqrt(1 - beta^2)  =  sqrt(1 - beta)/(1 - beta)       * 1/(sqrt(1 + beta) 

# sqrt(1 + beta)/sqrt(1 - beta^2)  =  sqrt(1 - beta) * (1 - beta))

# sqrt(1 + beta) =  sqrt(1 - beta) * (1 - beta)) * sqrt(1 - beta^2)



def gamma_lin(beta):
    #return 1/sqrt(1 - beta)
    return 1/sqrt(1 + beta)  # less than 1

def gamma_ang(beta):
    #return 1/sqrt(1 + beta)
    return 1/(sqrt(1 - beta)) # goes to infinity when beta approaches 1

def gamma(beta):
    return 1/sqrt(1 - beta**2)



def f(alpha,beta, R, return_all=False):

    ee = ( (2 * alpha * rho * h * c ) )       * gamma(beta)

    e1 = (     rho * k           *  pi * R  ) * gamma_lin(beta)
    e2 = ((2 * m   * c * alpha ) / (pi * R) ) * gamma_ang(beta)

    eet = e**2 * gamma(beta) 

    ret = abs(1 - eet/(e1 * e2)) + abs(1 - (alpha/alpha_codata))
    
    #print(f"ee: {(ee)}, sqrt(ee): {(sqrt(ee))}, e1: {(e1)}, e2: {(e2)}")
    #print(f"alpha: {(alpha)}, beta: {(beta)}, R: {(R)}, k_air: {(k_air)}, ret: {(ret)}")

    if return_all:
        return ee, e1, e2
    return ret

def target(x):
    return f(alpha=x[0],beta=x[1], R=x[2])

R_initial = 6.405e-26

first_guess = [alpha_codata, initial_beta, R_initial]

#print(f"First guess             : {(first_guess)}")

#fac = 1+1.6e-10
fac = 1+1.8e-11

bounds = Bounds(
    [alpha_codata/fac, beta_lower, 6.3e-26],
    [alpha_codata*fac, beta_upper, 6.5e-26]
)


res = optimize.minimize( target, x0=first_guess, bounds=bounds, method='nelder-mead', tol=1e-15,
                         options={'disp': False, 'maxiter': 100000, 'adaptive': True })

optimized_alpha = res.x[0]
optimized_beta  = res.x[1]
optimized_R     = res.x[2]



# Print results
print(f"Optimized beta          : {(optimized_beta)}")

ee, e1, e2 = f(alpha=optimized_alpha,beta=optimized_beta, R=optimized_R, return_all=True)

print(f"q                       : {sqrt(ee)}")
print(f"e/q                     : {e/sqrt(ee)}")
print(f"q/e                     : {sqrt(ee)/e}")
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






print(f"-------------------------------------------------")
print(f"-   Second step: factor out alpha              -")
print(f"-       Optimizing beta and R                   -")
print(f"-------------------------------------------------")


# e*e = rho * k_air * k * pi * R* 1/sqrt(1 - beta**2)  * (m * c * alpha * sqrt(1 + beta**2)) / (pi * R )

# (1 - beta) / sqrt(1 - beta^2) =  sqrt(1 - beta)  / sqrt(1 + beta) = sqrt((1-beta)/(1+beta))

# (1 - beta)   sqrt(1 + beta)   =  sqrt(1 - beta^2)  sqrt(1 - beta)

# 1 /( 1 + beta) = sqrt(1 - beta) / sqrt(1 - beta^2)  = sqrt((1-beta)/(1+beta))

# alpha = 1 / (8 * pi**2 * sqrt(3/k_air) * (1 + beta))

# base1: return rho * k_air * k * pi * 1/sqrt( 1 - beta**2 )
# base2: return (m * c) / (4 * pi**3 * pi/2 * 1/sqrt(1 + beta**2) )

def f2(alpha, beta, R, return_all=False):
    d_alpha = alpha * gamma(beta)

    ee = ( (2 * d_alpha * rho * h * c ) )

    e1 = (     rho * k             *  pi * R  )
    e2 = ((2 * m   * c * d_alpha ) / (pi * R) )

    eet = e**2 * gamma(beta) 

    ret = abs(1 - eet/(e1 * e2))  + abs(1 - (alpha/alpha_codata))

    if return_all:
        return ee, e1, e2
    return ret


def target2(x):
    return f2(alpha=x[0],beta=x[1], R=x[2])

R_initial = 6.405e-26

first_guess = [alpha_codata, initial_beta, R_initial]

#print(f"First guess             : {(first_guess)}")

#fac = 1+1.6e-10
fac = 1+1.8e-11

bounds = Bounds(
    [alpha_codata/fac, beta_lower, 6.3e-26],
    [alpha_codata*fac, beta_upper, 6.5e-26]
)


res = optimize.minimize( target2, x0=first_guess, bounds=bounds, method='nelder-mead', tol=1e-15,
                         options={'disp': False, 'maxiter': 100000, 'adaptive': True })

optimized_alpha = res.x[0]
optimized_beta  = res.x[1]
optimized_R     = res.x[2]


# Print results
print(f"Optimized beta          : {(optimized_beta)}")

ee, e1, e2 = f2(alpha=optimized_alpha,beta=optimized_beta, R=optimized_R, return_all=True)

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
print(f"alpha_codata / alpha    : {(alpha_codata / optimized_alpha)}")




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
print(f"-  Step 3. Determine geometric factor           -")
print(f"-------------------------------------------------")


def Alpha(beta,geom_fac):
    return 1 / (2 * geom_fac * gamma(beta)) 

def base1(beta):
    return   rho * k  * pi              * gamma_lin(beta)

def base2(beta,geom_fac):
    return ( (m * c) / (pi * geom_fac)) * gamma_ang(beta)

def R_1(beta):
    return e / base1(beta)

def R_2(beta, geom_fac):
    return base2(beta, geom_fac) / e

def Q1(beta, R):
    return base1(beta) * R

def Q2(beta, R, geom_fac):
    return base2(beta, geom_fac) / R

def QQ(beta, geom_fac):
    return base1(beta) * base2(beta, geom_fac)

def q_term_difference(beta,R,geom_fac):

    eet = e**2 * gamma(beta) 
    ret = abs(1 - eet/(Q1(beta,R) * Q2(beta,R,geom_fac))) + abs(1 - (alpha_codata / Alpha(beta,geom_fac)))

    #print(f"beta: {(beta)}, R: {(R)}, ret: {(ret)}")

    return ret

initial_geom_fac = 4 * pi**2 * sqrt(3)

print(f"Initial geom_fac        : {(initial_geom_fac)}")

alpha = Alpha(optimized_beta,initial_geom_fac)

def geom_objective(x):
    #return radii_difference(x[0])
    return q_term_difference(beta=optimized_beta,R=optimized_R,geom_fac=x[0])

first_guess  = [initial_geom_fac]

bounds = Bounds(
    [ 60 ],
    [ 90 ]
)

#geom_fac = 4 * pi**2 * sqrt(3)
def optimize_geom():
    #print(f"First guess             : {(first_guess)}")

    res = optimize.minimize( geom_objective, x0=first_guess, bounds=bounds, method='nelder-mead', tol=1e-20,
                             options={'disp': False, 'maxiter': 100000, 'adaptive': True })

    optimized_geom_fac = res.x[0]


    # Print results
    print(f"Optimized beta          : {(optimized_beta)}")
    print(f"Initial beta / Optimized: {(initial_beta / optimized_beta)}")
    print(f"1 / sqrt(1 - beta)      : {(1 / sqrt(1 - optimized_beta))}")
    print(f"1/sqrt(1 + beta)        : {(1 / sqrt(1 + optimized_beta))}")
    print(f"sqrt((1 + beta**2)/(1 - beta**2)): {(sqrt((1 + optimized_beta**2)/(1 - optimized_beta**2)))}")

    print(f"Optimized R             : {(optimized_R)}")

    q1 = Q1(optimized_beta, optimized_R)
    q2 = Q2(optimized_beta, optimized_R, optimized_geom_fac)
    qq = QQ(optimized_beta , optimized_geom_fac)

    print(f"q                       : {sqrt(qq)}")
    print(f"q/e                     : {e/sqrt(qq)}")
    print(f"e/q                     : {sqrt(qq)/e}")
    print(f"q_linear                : {q1}")
    print(f"q_angular               : {q2}")
    print(f"Optimized R             : {(optimized_R)}")

    print(f"Optimized geom_fac      : {(optimized_geom_fac)}")
    print(f"Initial geom_fac        : {(initial_geom_fac)}")
    print(f"Initial geom_fac / Optimized geom_fac: {(initial_geom_fac / optimized_geom_fac)}")
    print(f"Optimized geom_fac / Initial geom_fac: {(optimized_geom_fac / initial_geom_fac)}")

    term = optimized_geom_fac / (4 * pi**2)
    print(f"term                    : {(term)}")
    print(f"term / sqrt(3)          : {(term / sqrt(3))}")

    vcmb = sqrt(optimized_beta**2 * c**2)
    print(f"vcmb                    : {(vcmb / 1000)} km/s")

    gamma = 1 / sqrt(1 - optimized_beta**2)
    alpha = Alpha(optimized_beta,optimized_geom_fac)

    return alpha

alpha = optimize_geom()

#exit()

# Optimize
print(f"-------------------------------------------------")
print(f"Optimizing beta...")
print(f"-------------------------------------------------")


def objective(x):
    #return radii_difference(x[0])
    return q_term_difference(beta=x[0],R=x[1],geom_fac=x[2])

#geom_fac = 4 * pi**2 * sqrt(3)
def optimize_beta(initial_R):

    beta_lower = (300*1000)/c
    beta_upper = (900*1000)/c

    print(f"beta_lower              : {(beta_lower)}")
    print(f"beta_upper              : {(beta_upper)}")

    #initial_beta = optimized_beta
    #initial_R    = optimized_R

    first_guess  = [initial_beta, initial_R, initial_geom_fac]

    bounds = Bounds(
        [beta_lower, 6e-26, 60 ],
        [beta_upper, 8e-26, 90 ]
    )

    #print(f"First guess             : {(first_guess)}")

    res = optimize.minimize( objective, x0=first_guess, bounds=bounds, method='nelder-mead', tol=1e-15,
                             options={'disp': True, 'maxiter': 100000, 'adaptive': True })

    optimized_beta     = res.x[0]
    optimized_R        = res.x[1]
    optimized_geom_fac = res.x[2]


    # Print results
    print(f"Optimized beta          : {(optimized_beta)}")
    print(f"Initial beta / Optimized: {(initial_beta / optimized_beta)}")
    print(f"1 / sqrt(1 - beta)      : {(1 / sqrt(1 - optimized_beta))}")
    print(f"1/sqrt(1 + beta)        : {(1 / sqrt(1 + optimized_beta))}")
    print(f"sqrt((1 + beta**2)/(1 - beta**2)): {(sqrt((1 + optimized_beta**2)/(1 - optimized_beta**2)))}")

    print(f"Optimized R             : {(optimized_R)}")

    q1 = Q1(optimized_beta, optimized_R)
    q2 = Q2(optimized_beta, optimized_R, optimized_geom_fac)
    qq = QQ(optimized_beta , optimized_geom_fac)

    print(f"q                       : {sqrt(qq)}")
    print(f"q/e                     : {e/sqrt(qq)}")
    print(f"e/q                     : {sqrt(qq)/e}")
    print(f"q_linear                : {q1}")
    print(f"q_angular               : {q2}")
    print(f"Optimized R             : {(optimized_R)}")

    print(f"Optimized geom_fac      : {(optimized_geom_fac)}")

    vcmb = sqrt(optimized_beta**2 * c**2)
    print(f"vcmb                    : {(vcmb / 1000)} km/s")

    gamma = 1 / sqrt(1 - optimized_beta**2)
    alpha = Alpha(optimized_beta,optimized_geom_fac)

    print(f"alpha                   : {(alpha)}")
    print(f"alpha / alpha_codata    : {(alpha / alpha_codata)}")
    print(f"alpha_codata / alpha    : {(alpha_codata / alpha)}")


    return alpha

alpha = optimize_beta(optimized_R)





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

"""