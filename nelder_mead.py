import numpy as np
from mpmath import mp, mpf


# based on: https://github.com/owruby/nelder_mead

class NelderMead(object):

    def __init__(self, func, params, tol=mpf('1e-500'),*args, **kwargs):
        """ the Nelder-Mead method

        :param func: objective function object
        :param params: dict, tuning parameter, name: (min, max)
        """
        self.func = func
        self.dim = len(params)
        self.n_eval = 0
        self.names = []
        self.p_types = []
        self.p_min = []
        self.p_max = []
        self.simplex = []
        self.initialized = False
        self.tol = tol
        self._parse_minmax(params)

    def initialize(self, init_params):
        """ Initialize first simplex point
        :param init_params(list):
        """
        assert len(init_params) == (self.dim + 1), "Invalid length of init_params"
        for param in init_params:
            p = Point(self.dim)
            p.x = [mpf(val) for val in param]  # Use mpf for arbitrary precision
            self.simplex.append(p)
        self.initialized = True

    def maximize(self, n_iter=20, delta_r=1, delta_e=2, delta_ic=-0.5, delta_oc=0.5, gamma_s=0.5):
        """ Maximize the objective function. """
        self._coef = -1
        variables = locals()
        for k, v in variables.items():
            setattr(self, k, v)
        self._opt(n_iter)

    def minimize(self, n_iter=20, delta_r=1, delta_e=2, delta_ic=-0.5, delta_oc=0.5, gamma_s=0.5):
        """ Minimize the objective function. """
        self._coef = 1
        variables = locals()
        for k, v in variables.items():
            setattr(self, k, v)
        self._opt(n_iter)

    def func_impl(self, x):
        objval, invalid = None, False
        for i, t in enumerate(x):
            if t < self.p_min[i] or t > self.p_max[i]:
                objval = float("inf")
                invalid = True
        if not invalid:
            x = [int(round(x_t)) if p_t == "integer" else x_t for p_t, x_t in zip(self.p_types, x)]
            objval = self._coef * self.func(x)
        print("{:5d} | {} | {:>15.5f}".format(
            self.n_eval,
            " | ".join([f"{float(t):15.5f}" for t in x]),
            self._coef * float(objval)
        ))
        self.n_eval += 1
        return objval

    def _opt(self, n_iter):
        # Print Header
        print("{:>5} | {} | {:>15}".format(
            "Eval",
            " | ".join([f"{name:>15}" for name in self.names]),
            "ObjVal"
        ))
        print("-" * (20 + self.dim * 20))

        if not self.initialized:
            self._initialize()

        stable_count = 0
        prev_best_objval = None

        for p in self.simplex:
            p.f = self.func_impl(p.x)

        for i in range(n_iter):
            self.simplex = sorted(self.simplex, key=lambda p: abs(p.f))  # Sort by absolute value of f

            best_objval = self.simplex[0].f
            if prev_best_objval is not None and abs(prev_best_objval - best_objval) < self.tol:
                stable_count += 1
                if stable_count >= 50 and abs:
                    print(f"prev_best_objval: {float(prev_best_objval)}")
                    print(f"best_objval: {float(best_objval)}")
                    print("tol: ", float(self.tol))
                    print("Converged!")
                    break
            else:
                stable_count = 0

            if prev_best_objval is None or abs(best_objval) < abs(prev_best_objval):
                prev_best_objval = best_objval


            # Keep the best point, re-randomize others if stuck
            if stable_count >= 10:
                print("Re-randomizing simplex points (except the best one)...")
                self._re_randomize_simplex()

            # Continue with the usual Nelder-Mead steps
            p_c = self._centroid()
            p_r = self._reflect(p_c)

            if p_r < self.simplex[0]:
                p_e = self._expand(p_c)
                if p_e < p_r:
                    self.simplex[-1] = p_e
                else:
                    self.simplex[-1] = p_r
                continue
            elif p_r > self.simplex[-2]:
                if p_r <= self.simplex[-1]:
                    p_cont = self._outside(p_c)
                    if p_cont < p_r:
                        self.simplex[-1] = p_cont
                        continue
                    self.simplex[-1] = p_r
                elif p_r > self.simplex[-1]:
                    p_cont = self._inside(p_c)
                    if p_cont < self.simplex[-1]:
                        self.simplex[-1] = p_cont
                        continue

                # Shrink
                for j in range(len(self.simplex) - 1):
                    p = Point(self.dim)
                    p.x = [p_c.x[i] + self.gamma_s * (self.simplex[j + 1].x[i] - p_c.x[i]) for i in range(self.dim)]
                    p.f = self.func_impl(p.x)
                    self.simplex[j + 1] = p
            else:
                self.simplex[-1] = p_r

        self.simplex = sorted(self.simplex, key=lambda p: abs(p.f))
        print("\nBest Point: {}".format(self.simplex[0]))

    def _centroid(self):
        p_c = Point(self.dim)
        x_sum = [p.x for p in self.simplex[:-1]]
        p_c.x = [sum(x[i] for x in x_sum) / len(x_sum) for i in range(self.dim)]  # Use arbitrary precision
        return p_c

    def _reflect(self, p_c):
        return self._generate_point(p_c, self.delta_r)

    def _expand(self, p_c):
        return self._generate_point(p_c, self.delta_e)

    def _inside(self, p_c):
        return self._generate_point(p_c, self.delta_ic)

    def _outside(self, p_c):
        return self._generate_point(p_c, self.delta_oc)

    def _generate_point(self, p_c, x_coef):
        p = Point(self.dim)
        p.x = [p_c.x[i] + x_coef * (p_c.x[i] - self.simplex[-1].x[i]) for i in range(self.dim)]
        p.f = self.func_impl(p.x)
        return p

    def _parse_minmax(self, params):
        types = ["real", "integer", "mpf"]
        for name, values in params.items():
            assert values[0] in types, "Invalid param type, Please check it."

            self.names.append(name)
            self.p_types.append(values[0])
            self.p_min.append(values[1][0])
            self.p_max.append(values[1][1])

    def _initialize(self):
        for i in range(self.dim + 1):
            p = Point(self.dim)
            init_val = [mpf((m2 - m1) * np.random.random() + m1) for m1, m2 in zip(self.p_min, self.p_max)]
            p.x = init_val
            self.simplex.append(p)

    def _re_randomize_simplex(self):
        # Keep the best point, and re-randomize others
        best_point = self.simplex[0]
        self.simplex = [best_point]  # Start with the best point
        for i in range(self.dim):
            p = Point(self.dim)
            init_val = [mpf((m2 - m1) * np.random.random() + m1) for m1, m2 in zip(self.p_min, self.p_max)]
            p.x = init_val
            self.simplex.append(p)


class Point(object):

    def __init__(self, dim):
        self.x = [mpf(0) for _ in range(dim)]  # Use mpf for arbitrary precision
        self.f = mpf(0)

    def __str__(self):
        return "Params: {}, ObjValue: {}".format(
            ", ".join([f"{float(x):10.5f}" for x in self.x]),
            float(self.f))

    def __eq__(self, rhs):
        return self.f == rhs.f

    def __lt__(self, rhs):
        return self.f < rhs.f

    def __le__(self, rhs):
        return self.f <= rhs.f

    def __gt__(self, rhs):
        return self.f > rhs.f

    def __ge__(self, rhs):
        return self.f >= rhs.f
