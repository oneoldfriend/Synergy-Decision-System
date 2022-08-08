#pragma once
#include "Eigen/Dense"
#include <vector>

using namespace std;

class Action;
class Route;
class Order;

class Solution
{
public:
  vector<Route> routes;
  Eigen::Vector4d info;
  double cost, penalty, waitTime, travelTime;
  bool greedyInsertion(vector<pair<Order *, Order *> > orderWaitToBeInserted);
  void solutionCopy(Solution *source);
  void solutionDelete();
  void solutionUpdate();
  double calcCost();
  void calcInfo();
  Solution();
};