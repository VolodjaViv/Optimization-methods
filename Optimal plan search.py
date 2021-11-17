import numpy as np
import copy


def main():

    # From teacher
    c = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    a = [0, 0, 0]
    b = [0, 0, 0]
    m, n = c.shape

    if sum(a) != sum(b):
        print("No solutions")
        return

    print("a =", a)
    print("b =", b)
    print("c =\n", c, end="\n\n")
    m, n = c.shape

    temp_a = copy.deepcopy(a)
    temp_b = copy.deepcopy(b)

    plan = np.zeros((m, n))
    print("plan =\n", plan, end="\n\n")

    j_b = set()
    i, j = 0, 0
    while i < len(a) and j < len(b):
        if temp_a[i] < temp_b[j]:
            temp_b[j] -= temp_a[i]
            plan[i, j] = temp_a[i]
            j_b.add((i, j))

            i += 1
        else:
            temp_a[i] -= temp_b[j]
            plan[i, j] = temp_b[j]
            j_b.add((i, j))

            j += 1

    for x in range(len(a)):
        for y in range(len(b)):
            if  len(j_b) < m + n - 1 and (x, y) not in j_b:
                j_b.add((x, y))

                cycle = np.zeros((len(a), len(b)))
                for i, j in j_b:
                    cycle[i, j] = 1
                print("cycle =\n", cycle)

                cont = True
                while cont:
                    cont = False

                    for i, row in enumerate(cycle):
                        if sum(row) != 1:
                            continue

                        cycle[i] = np.zeros(len(b))
                        print("cycle (del row) =\n", cycle)

                        cont = True

                    for j in range(len(b)):
                        if sum(cycle[:, j]) != 1:
                            continue

                        cycle[:, j] = np.zeros(m)
                        print("cycle (del col) =\n", cycle)

                        cont = True
                
                print("cycle =\n", cycle)


                row_sum = []
                for row in cycle:
                    row_sum.append(sum(row))
                qqq = sum(row_sum) 
                if qqq != 0:
                    j_b.remove((x, y))
    print("plan =\n", plan, end="\n\n")

    j_b = list(j_b)

    iter = 1
    while True:
        print("iter =", iter)
        print("plan =\n", plan)
        print("j_b =", j_b)

        u, v = [None] * len(a), [None] * len(b)
        u[0] = 0

        for _ in range(m + n):
            for x, y in j_b:
                if u[x] is not None and v[y] is None:
                    v[y] = c[x, y] - u[x]
                    break
                elif u[x] is None and v[y] is not None:
                    u[x] = c[x, y] - v[y]
                    break

        print("u =", u)
        print("v =", v)

        delta = np.zeros((len(a), len(b)))
        for i in range(len(a)):
            for j in range(len(b)):
                delta[i, j] = c[i, j] - u[i] - v[j]
        print("delta =\n", delta)

        if delta.min() >= 0:
            print("\noptimal plan =\n", plan)
            print("j_b =", j_b)

            expenses = 0
            for i in range(len(a)):
                for j in range(len(b)):
                    expenses += plan[i, j] * c[i, j]
            print("expenses = ", expenses)

            return

        min_delta = 0
        i0 = 0
        j0 = 0
        for i in range(len(a)):
            for j in range(len(b)):
                if delta[i, j] < min_delta and (i, j) not in j_b:
                    min_delta = delta[i, j]
                    i0, j0 = i, j

        print("i min (i0) =", i0)
        print("j min (j0) =", j0)

        j_b.append((i0, j0))
        print("j_b =", j_b)




        # cycle
        cycle = np.zeros((len(a), len(b)))
        for i, j in j_b:
            cycle[i, j] = 1
        print("cycle =\n", cycle)

        cont = True
        while cont:
            cont = False

            for i, row in enumerate(cycle):
                if sum(row) != 1:
                    continue

                cycle[i] = np.zeros(len(b))
                print("cycle (del row) =\n", cycle)

                cont = True

            for j in range(len(b)):
                if sum(cycle[:, j]) != 1:
                    continue

                cycle[:, j] = np.zeros(m)
                print("cycle (del col) =\n", cycle)

                cont = True
        
        print("cycle =\n", cycle)




        # update plan
        count_basis_position = 0
        for i in range(len(a)):
            for j in range(len(b)):
                count_basis_position += int(cycle[i, j])
        print("count_basis_position =", count_basis_position)

        cycle_znak = copy.deepcopy(cycle)
        znak = 1

        i, j = i0, j0
        for _ in range(count_basis_position):
            cycle_znak[i, j] = znak
            cycle[i, j] = 0
            znak = -znak

            for j_new in range(0, len(b)):
                if cycle[i, j_new] == 1:
                    j = j_new
                    break
            else:
                for i_new in range(0, len(a)):
                    if cycle[i_new, j] == 1:
                        i = i_new
                        break
        
        print("cycle_znak =\n", cycle_znak)




        # find theta
        min_theta = np.inf
        i_theta, j_theta = 0, 0
        for i in range(len(a)):
            for j in range(len(b)):
                if cycle_znak[i, j] == -1 and plan[i, j] < min_theta:
                    min_theta = plan[i, j]
                    i_theta = i
                    j_theta = j

        print("min_theta =", min_theta, "(i_theta, j_theta) =", (i_theta, j_theta))

        for i in range(len(a)):
            for j in range(len(b)):
                if cycle_znak[i, j] == 1:
                    plan[i, j] += min_theta
                elif cycle_znak[i, j] == -1:
                    plan[i, j] -= min_theta

        j_b.remove((i_theta, j_theta))





        print("plan =\n", plan)
        print("j_b =", j_b)

        print()
        print("-" * 50)
        print()

        iter += 1




if __name__ == "__main__":
    main()
