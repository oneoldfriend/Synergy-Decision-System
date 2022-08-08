#pragma once
#include <string>

using namespace std;

class Position
{
public:
  double x, y;
  Position()
  {
    x = 0;
    y = 0;
  };
};

class Customer
{
public:
  Position origin, dest;
  double startTime, endTime, priority, weight;
  string id;
  Customer();
};