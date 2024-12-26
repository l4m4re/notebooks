from mpmath import mp, mpf, sqrt, pi

# Set precision (e.g., 100 decimal places)
mp.dps = 50

# Set tighter tolerance
tolerance = mpf('1e-35')

# Constants
c = mpf('299792458.0')          # [m/s] - speed of light
k = c**2                        # dynamic viscosity [m^2/s]
h = mpf('6.62607015e-34')       # Planck's constant [J-s], [kg-m^2/s]
m = h / k                       # elemental mass
e = mpf('1.602176634e-19')      # Elementary charge (C)
eta = mpf('1') / (4 * pi * mpf('1e-7'))  # viscosity [Pa-s], [kg/m-s], [N-s/m^2], [J-s/m^3]
alpha_codata = mpf('0.007297352569311') # fine structure constant

# Function definitions (unchanged)
def base1(beta):
    #return eta * pi * sqrt(1 - beta)
    #return eta * pi * ( 1 - sqrt(beta) )
    #return eta * pi * ( 1 - beta )
    return eta * pi * sqrt( 1 - beta )

def base2(beta):
    #return (m * c) / (4 * pi**3 * sqrt(3) * sqrt(1 + beta))
    #return (m * c) / (4 * pi**3 * sqrt(3) * (1 + sqrt(beta)) )
    #return (m * c) / (4 * pi**3 * sqrt(3) * (1 + beta) )
    return (m * c) / (4 * pi**3 * sqrt(3) * sqrt(1 + beta) )

def R_1(beta):
    return e / base1(beta)

def R_2(beta):
    return base2(beta)/ e

def Q1(beta, R):
    return base1(beta) * R

def Q2(beta, R):
    return base2(beta) / R

def Alpha(beta):
    #return sqrt(1-beta**2) / (8 * pi**2 * sqrt(3))
    #return 1 / (8 * pi**2 * sqrt(3) * sqrt(1-beta**2) )
    #return 1 / (8 * pi**2 * sqrt(3) )
    #return 1 / (8 * pi**2 * sqrt(3) * (beta + 1))

    # with this, the initial value for beta results in alpha being very close to the CODATA value
    #return 1 / (8 * pi**2 * sqrt(3) * (1 + beta))

    # with this, the optimized value for beta results in alpha being very close to the CODATA value
    return 1 / (8 * pi**2 * sqrt(3) * (1 + beta)/sqrt(1-beta**2) )

def radii_difference(beta):
    return (R_1(beta) - R_2(beta))

def find_bounds(func, initial, step_factor=mpf('1.001'), max_iter=10000):
    lower_bound = initial
    upper_bound = initial
    initial_val = radii_difference(initial)
    lower_val = radii_difference(lower_bound)
    upper_val = radii_difference(upper_bound)

    for iter in range(max_iter):
        if lower_val * upper_val < mpf('0.0'):
            return lower_bound, upper_bound
        if initial_val * lower_val < mpf('0.0'):
            return initial, lower_bound
        if initial_val * upper_val < mpf('0.0'):
            return initial, upper_bound

        lower_bound /= step_factor
        upper_bound *= step_factor

        lower_val = radii_difference(lower_bound)
        upper_val = radii_difference(upper_bound)

    raise ValueError("Failed to find bounds where the function changes sign.")

def interval_halving(func, lower, upper, tol, max_iter=100000):

    for iter in range(max_iter):
        mid = (upper + lower) / 2
        
        fl = func(lower)
        fm = func(mid)
        fu = func(upper)

        if fm > 0:
            if fu > 0:
                if fu >= fm:
                    upper = mid
            elif fl > 0:
                if fl >= fm:
                    lower = mid        
            else:
                print(f"iter: {iter}")
                print(f"lower: {lower}, mid: {mid}, upper: {upper}")
                print(f"fl: {fl}, fm: {fm}, fu: {fu}")
                raise ValueError("Initial conditions are not correct 1.")
        elif fm < 0:
            if fu < 0:
                if fu <= fm:
                    upper = mid
            elif fl < 0:
                if fl <= fm:
                    lower = mid
            else:
                print(f"iter: {iter}")
                print(f"lower: {lower}, mid: {mid}, upper: {upper}")
                print(f"fl: {fl}, fm: {fm}, fu: {fu}")
                raise ValueError("Initial conditions are not correct 2.")

        if abs(upper - lower) < tol:
            beta_opt = (upper + lower) / 2
            return beta_opt, func(beta_opt)

    raise ValueError("Interval halving did not converge within the maximum iterations.")

# Define initial beta using high precision
initial_beta = (m * c * eta) / (4 * pi**2 * sqrt(3) * e**2) - 1

# Print the initial beta with proper conversion to float for formatting
print(f"Initial beta            : {float(initial_beta)}")

R1 = R_1(initial_beta)
R2 = R_2(initial_beta)

# Print R1, R2, and their ratio with proper conversion to float
print(f"R1                      : {float(R1)}")
print(f"R2                      : {float(R2)}")
print(f"R1/R2                   : {float(R1/R2)}")

q1 = Q1(initial_beta, R1)
q2 = Q2(initial_beta, R2)

print(f"q1                      : {float(q1)}")
print(f"q2                      : {float(q2)}")
qratio = q1 / q2
print(f"q1/q2                   : {float(qratio)}")
print(f"q1/e                    : {float(q1 / e)}")

alpha = Alpha(initial_beta)

print(f"alpha                   : {float(alpha)}")
print(f"alpha / alpha_codata    : {float(alpha / alpha_codata)}")
print(f"alpha_codata / alpha    : {float(alpha_codata / alpha)}")



# Find the bounds
lower_bound, upper_bound = find_bounds(radii_difference, initial_beta)

# Run interval halving
optimized_beta, min_difference = interval_halving(radii_difference, lower_bound, upper_bound, tol=tolerance)

# Compute final R1 and R2 with high precision
R1_optimized = R_1(optimized_beta)
R2_optimized = R_2(optimized_beta)

# Print results
print(f"Optimized beta          : {float(optimized_beta)}")
print(f"Initial beta / Optimized: {float(initial_beta / optimized_beta)}")
print(f"1 / sqrt(1 - beta)      : {float(1 / sqrt(1 - optimized_beta))}")
print(f"1/sqrt(1 + beta)        : {float(1 / sqrt(1 + optimized_beta))}")

print(f"R1 (optimized)          : {float(R1_optimized)}")
print(f"R2 (optimized)          : {float(R2_optimized)}")
print(f"R2/R1                   : {float(R2_optimized / R1_optimized)}")

q1 = Q1(optimized_beta, R1_optimized)
q2 = Q2(optimized_beta, R2_optimized)

print(f"q1                      : {float(q1)}")
print(f"q2                      : {float(q2)}")
qratio = q1 / q2
print(f"q1/q2                   : {float(qratio)}")
print(f"q1/e                    : {float(q1 / e)}")

mma = mpf('1.001165923')
mmma2 = mma**2

print(f"mma                     : {float(mma)}")
print(f"mma2                    : {float(mmma2)}")
print(f"mma2 / mma              : {float(mmma2 / mma)}")

vcmb = sqrt(optimized_beta**2 * c**2)
print(f"vcmb                    : {float(vcmb / 1000)} km/s")

gamma = 1 / sqrt(1 - optimized_beta**2)
alpha = Alpha(optimized_beta)

print(f"alpha                   : {float(alpha)}")
print(f"alpha / alpha_codata    : {float(alpha / alpha_codata)}")
print(f"alpha_codata / alpha    : {float(alpha_codata / alpha)}")


# Experimental value
a_e = 0.00115965218059

print(f"mma-1                   : {float(mma - 1)}")
print(f"a_e                     : {float(a_e)}")
print(f"mma-1 / a_e             : {float((mma - 1) / a_e)}")

print(f"alpha / 2 pi            : {float(alpha / (2 * pi))}")
print(f"a_e /(alpha / 2 pi)     : {float(a_e / (alpha / (2 * pi)))}")
