#pragma once
#include "customer.h"
#include "util.h"
#include "avi.h"
#include <iostream>

using namespace std;

typedef class Order *PointOrder;

typedef class Route *PointRoute;

class Order
{
public:
  double arrivalTime, waitTime, departureTime, currentWeight;
  bool isOrigin;
  Customer *customer;
  Position position;
  PointOrder prior;
  PointOrder next;
  void infoCopy(PointOrder source);
  Order(Customer *customer, bool isOrigin);
};
class Route
{
public:
  Customer *depot;
  PointOrder head;
  PointOrder tail;
  double cost, waitTime, penalty, travelTime;
  PointOrder currentPos;
  void routeUpdate();
  void insertOrder(PointOrder p);
  void removeOrder(PointOrder p);
  bool findBestPosition(PointOrder origin, PointOrder dest, double *bestCost);
  void routeCopy(Route source);
  bool checkFeasibility();
  void deleteRoute();
  double calcCost();
  Route();
};
