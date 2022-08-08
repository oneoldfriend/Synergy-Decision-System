#pragma once
#include "mdp.h"
#include "vfa.h"
#include <math.h>
#include <fstream>

#define REJECTION_COST 60.0
#define MAX_WORK_TIME 720.0
#define CAPACITY 140
#define PENALTY_FACTOR 2
#define DEGREE_OF_DYNAMISM 1
#define MAX_COST 999999.0
#define MAX_EDGE 100.0
#define UNIT_TIME 5.0
#define SPEED 20
#define CUSTOMER_NUMBER 30
#define MAX_VEHICLE 1
#define MAX_SIMULATION 10000
#define MAX_TEST_INSTANCE 5000
#define LAG_APPROXIMATE 0
#define REVIEW_MAX 100
#define GENERATOR 0
#define ASSIGNMENT_MYOPIC 0
#define ROUTING_MYOPIC 0
#define IS_SYNERGY false

class AVI
{
public:
	void approximation(ValueFunction* valueFunction, vector<pair<double, double>> pos_matrix);
};
