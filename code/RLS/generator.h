#pragma once
#include "avi.h"
#include <random>

class Generator
{
public:
    static void instanceGenenrator(bool testInstanceGenerate, list<pair<double, Customer*>>* sequenceData, string fileName, vector<pair<double, double>> pos_matrix, bool is_synergy);
    static void instanceGenenrator(bool testInstanceGenerate, list<pair<double, Customer*>>* sequenceData, string fileName);
    static bool sortAscend(const pair<double, Customer *> a, const pair<double, Customer *> b);
    static int restaurants_selector(vector<double> users_cvr, double random_num);
};