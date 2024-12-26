from mpmath import mp, mpf, sqrt, pi

from nelder_mead import NelderMead

# Set precision (e.g., 100 decimal places)
mp.dps = 500

# Set tighter tolerance
tolerance = mpf('1e-35')

# Constants
c = mpf('299792458.0')          # [m/s] - speed of light
k = c**2                        # dynamic viscosity [m^2/s]
h = mpf('6.62607015e-34')       # Planck's constant [J-s], [kg-m^2/s]
m = h / k                       # elemental mass
q = mpf('1.602176634e-19')      # Elementary charge (C)
k_air = mpf('1.00058986')
e = q  # * sqrt(k_air)                  # Elementary charge (C)

print(f"e: {float(e)}")


eta = mpf('1') / (4 * pi * mpf('1e-7'))  # viscosity [Pa-s], [kg/m-s], [N-s/m^2], [J-s/m^3]
alpha_codata = mpf('0.007297352569311') # fine structure constant

# Experimental values https://en.wikipedia.org/wiki/Anomalous_magnetic_dipole_moment
a_e  = mpf('0.00115965218059')    # electron magnetic moment anomaly
a_mu = mpf('0.00116592059')       # muon magnetic moment anomaly

# \[
#   1 + \beta = \sqrt{1 - \beta} \cdot \sqrt{1 + \beta}
# \]

#  1 + beta = sqrt(1 - beta) * sqrt(1 + beta)

# alpha_stowe = 1/(2*sqrt(3/k_air)*(2*pi*mma)**2)

def Alpha(beta):
    #return sqrt(1-beta**2) / (8 * pi**2 * sqrt(3))
    #return 1 / (8 * pi**2 * sqrt(3) * sqrt(1-beta**2) )
    #return 1 / (8 * pi**2 * sqrt(3) )
    #return 1 / (8 * pi**2 * sqrt(3) * (beta + 1))

    # with this, the initial value for beta results in alpha being very close to the CODATA value
    #return 1 / (8 * pi**2 * sqrt(3) * (1 + beta))
    #return 1 / (8 * pi**2 * sqrt(3/k_air) * (1 + beta))

    # with this, the optimized value for beta results in alpha being very close to the CODATA value
    return 1 / (8 * pi**2 * sqrt(3) * (1 + beta)/sqrt(1-beta**2) )

    #return 1 / (8 * pi**2 * sqrt(3/k_air) * (1 + beta)/sqrt(1-beta**2) )


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
    #return (m * c) / (4 * pi**3 * sqrt(3/k_air) * sqrt(1 + beta) )

def R_1(beta):
    return e / base1(beta)

def R_2(beta):
    return base2(beta)/ e

def Q1(beta, R):
    return base1(beta) * R

def Q2(beta, R):
    return base2(beta) / R


def radii_difference(beta):
    #print(f"beta: {(beta)}", end="")
    #print(f"beta: {float(beta)}", end="")
    #ret = R_1(beta) / R_2(beta) - 1
    #print(f", ret: {float(ret)}") 
    #return ret
    return (R_1(beta) - R_2(beta))

def q_term_difference(beta,R):
    return Q1(beta,R) - Q2(beta,R)


def nelder_mead_with_bounds(f, simplex, bounds, tol=mpf('1e-50'), max_iter=10000):
    """
    Nelder-Mead optimization with boundary constraints.
    
    f: Objective function to minimize.
    simplex: List of initial vertices (list of mpf arrays).
    bounds: List of tuples [(lower_bound, upper_bound)] for each parameter.
    tol: Convergence tolerance.
    max_iter: Maximum iterations.
    """
    n = len(simplex) - 1  # Dimensionality
    alpha, gamma, rho, sigma = mpf('1'), mpf('2'), mpf('0.5'), mpf('0.5')

    def project_to_bounds(point):
        """
        Projects a point to the feasible region defined by bounds.
        """
        return [
            max(bounds[j][0], min(bounds[j][1], point[j]))
            for j in range(len(point))
        ]

    for i in range(max_iter):
        # Sort simplex by function values
        simplex = sorted(simplex, key=f)

        # Centroid of all but the worst point
        centroid = [
            sum(vertex[j] for vertex in simplex[:-1]) / n
            for j in range(len(simplex[0]))
        ]
        worst = simplex[-1]

        # Reflection
        reflected = [
            centroid[j] + alpha * (centroid[j] - worst[j])
            for j in range(len(centroid))
        ]
        reflected = project_to_bounds(reflected)
        if f(simplex[0]) <= f(reflected) < f(simplex[-2]):
            simplex[-1] = reflected
            continue

        # Expansion
        if f(reflected) < f(simplex[0]):
            expanded = [
                centroid[j] + gamma * (reflected[j] - centroid[j])
                for j in range(len(centroid))
            ]
            expanded = project_to_bounds(expanded)
            simplex[-1] = expanded if f(expanded) < f(reflected) else reflected
            continue

        # Contraction
        contracted = [
            centroid[j] + rho * (worst[j] - centroid[j])
            for j in range(len(centroid))
        ]
        contracted = project_to_bounds(contracted)
        if f(contracted) < f(worst):
            simplex[-1] = contracted
        else:
            # Shrink: Move all points towards the best
            best = simplex[0]
            simplex = [best] + [
                project_to_bounds([
                    best[j] + sigma * (vertex[j] - best[j])
                    for j in range(len(best))
                ])
                for vertex in simplex[1:]
            ]

        # Convergence check: Objective function values
        f_values = [f(vertex) for vertex in simplex]
        if max(abs(f_val) for f_val in f_values) < tol:
            print(f"Converged after {i} iterations on f_values with tol={tol}.")
            print(f"Function values: {f_values}")
            break

        # Optional: Convergence check based on simplex geometry
        #max_distance = max(
        #    sum((vertex1[j] - vertex2[j]) ** 2 for j in range(len(vertex1))) ** 0.5
        #    for vertex1 in simplex
        #    for vertex2 in simplex
        #)
        #if max_distance < tol:
        #    print(f"Converged after {i} iterations on max_distance.")
        #    break

    return simplex[0]  # Return the best point


def find_bounds(func, initial, step_factor=mpf('1.001'), max_iter=10000):
    lower_bound = initial
    upper_bound = initial
    initial_val = func(initial)
    lower_val = func(lower_bound)
    upper_val = func(upper_bound)

    for iter in range(max_iter):
        if lower_val * upper_val < mpf('0.0'):
            return lower_bound, upper_bound
        if initial_val * lower_val < mpf('0.0'):
            return initial, lower_bound
        if initial_val * upper_val < mpf('0.0'):
            return initial, upper_bound

        lower_bound /= step_factor
        upper_bound *= step_factor

        lower_val = func(lower_bound)
        upper_val = func(upper_bound)

        #print(f"iter: {iter}, lower: {float(lower_bound)}, upper: {float(upper_bound)}, initial: {float(initial)}") 

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
#lower_bound, upper_bound = find_bounds(q_term_difference, initial_beta)

# Run interval halving
optimized_beta, min_difference = interval_halving(radii_difference, lower_bound, upper_bound, tol=tolerance)
#optimized_beta, min_difference = interval_halving(q_term_difference, lower_bound, upper_bound, tol=tolerance)


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


vcmb = sqrt(optimized_beta**2 * c**2)
print(f"vcmb                    : {float(vcmb / 1000)} km/s")

gamma = 1 / sqrt(1 - optimized_beta**2)
alpha = Alpha(optimized_beta)

print(f"alpha                   : {float(alpha)}")
print(f"alpha / alpha_codata    : {float(alpha / alpha_codata)}")
print(f"alpha_codata / alpha    : {float(alpha_codata / alpha)}")


mma = mpf('1.001165923')

print(f"mma                     : {float(mma)}")
print(f"a_e + 1                 : {float(a_e + 1)}")
print(f"a_mu + 1                : {float(a_mu + 1)}")

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

print(f"alpha_stowe             : {float(alpha_stowe)}")
print(f"alpha_stowe / alpha_codata: {float(alpha_stowe / alpha_codata)}")
print(f"alpha_codata / alpha_stowe: {float(alpha_codata / alpha_stowe)}")

# Example usage
def objective(x):
    #return x[0]**2 + x[1]**2
    return radii_difference(x[0])

"""

# Define bounds
#factor = mpf('4')
#bounds = [(initial_beta / factor, initial_beta * factor)]
bounds = [(lower_bound, upper_bound)]

factor = mpf('1.01')
# Define the initial simplex
#simplex = [
#    [initial_beta],
#    [initial_beta * factor],
#    [initial_beta / factor],
#]

simplex = [
    [optimized_beta],
    [optimized_beta * factor],
    [optimized_beta / factor],
]


# Run the optimization with bounds
result = nelder_mead_with_bounds(objective, simplex, bounds)

#print simplex
print("Initial simplex:", simplex)


print("Optimized point:", result)
print("Function value at optimized point:", objective(result))

print("Optimized beta / result:", optimized_beta / result[0])

func = sphere
params = {
    "x1": ["real", (-512, 512)],
    "x2": ["real", (-512, 512)],
}

"""

func = objective
params = {
    "beta": ["mpf", (lower_bound, upper_bound)],
}

nm = NelderMead(func, params)
nm.minimize(n_iter=3000)

optimized_beta = nm.simplex[0].x[0]



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


vcmb = sqrt(optimized_beta**2 * c**2)
print(f"vcmb                    : {float(vcmb / 1000)} km/s")
