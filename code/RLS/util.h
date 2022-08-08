#pragma once
#include"route.h"
#include <math.h>
#include <vector>
#include <list>
#include <iostream>

using namespace std;

class Util
{
  public:
    static void infoCopy(Customer *target, Customer *source);
    static double standardDeviation(vector<double> sample);
    static double calcTravelTime(Position a, Position b);
    static double minTravelTimeCalc(list<pair<double, Customer *> > data);
};