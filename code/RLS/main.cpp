#include "solver.h"
#include "util.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

int main()
{
    for (int i = 0; i < 20; i++) {
        cout << i << " ";
        Solver solver;
        solver.solve();
    }
}
