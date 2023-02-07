from math import *

# define the fundamental constants

c     = 299792458.0     # speed of light     [m/s]
eta   = 1/(4*pi*1e-7)   # viscosity          [kg/m-s],   [Pa-s] 
h     = 6.62607015e-34  # Planck's constant  [kg/m^2-s], [J-s]  

e     = 1.602176634e-19 # elementary charge  [kg/s]

# define some more

k     = c*c     # quantum circulation constant 
                # 8.987551787368176e+16                 [m^2/s] 
    
m     = h/k     # elementary mass 7.372497323812708e-51 [kg]

rho   = eta/k   # mass density   8.85418781762039e-12   [kg/m^3]


print("Calculate R")

R = e/(rho*k*pi)     # Approximate radius of elemental vortex ring 
                     # 6.408706536e-26 [m]

# Print R and confirm angular momentum is equal to linear momentum
# for a superfluid vortex ring with radius R
print("R:                   ", R)
print("angular momentum:    ", rho*k*pi*R**2)
print("Linear momentum      ", R*e)

print()
print("Calculate the two terms in the definition of e*e")

angular = rho*(h/m)*pi*R
linear  = (m*c)/(4*pi**3*sqrt(3)*R)

print("angular or magnetic: ", angular)
print("linear or dielectric:", linear)

print()
print("Calculate epsilon and a")

X   = (h*c)/(4*pi*pi*sqrt(3)*rho*k*k*pi*pi*R*R)
eps = sqrt( (4/3)* (sqrt(X)- 1) )
a   = eps*R    # radius of hollow core 2.361813747378764e-27 [m]

print("eps:                 ", eps)
print("a:                   ", a)


print()
print("Calculate corrected definition of elemental charge")


a_e = 1 + (3/4) * eps *eps

linear  = (m*c)/(4*pi**3*sqrt(3)*R*a_e*a_e)
angular = rho*(h/m)*pi*R
q       = sqrt(linear*angular)

print("angular or magnetic charge: ", angular)
print("linear or dielectric charge:", linear)
print("elemental charge:           ", q)










