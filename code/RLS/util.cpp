#include "util.h"

void Util::infoCopy(Customer *target, Customer *source)
{
    target->dest.x = source->dest.x;
    target->dest.y = source->dest.y;
    target->origin.x = source->origin.x;
    target->origin.y = source->origin.y;
    target->startTime = source->startTime;
    target->endTime = source->endTime;
    target->weight = source->weight;
    target->priority = source->priority;
    target->id = source->id;
}

double Util::standardDeviation(vector<double> sample)
{
    double sum = 0, mean = 0, variance = 0;
    for (auto iter = sample.begin(); iter != sample.end(); ++iter)
    {
        sum += *iter;
    }
    mean = sum / sample.size();
    for (auto iter = sample.begin(); iter != sample.end(); ++iter)
    {
        variance += pow(*iter - mean, 2);
    }
    variance = variance / (sample.size() - 1);
    return sqrt(variance);
}

double Util::calcTravelTime(Position a, Position b)
{
    double time = sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2)) / SPEED * 60.0;
    time = ceil(time * 100.0) / 100.0;
    return time;
}

double Util::minTravelTimeCalc(list<pair<double, Customer *>> data)
{
    double minTravelTime = MAX_WORK_TIME;
    Position depot;
    depot.x = 0.0;
    depot.y = 0.0;
    vector<Position> positions;
    positions.push_back(depot);
    for (auto iter1 = data.begin(); iter1 != data.end(); ++iter1)
    {
        positions.push_back(iter1->second->origin);
        positions.push_back(iter1->second->dest);
    }
    for (auto iter1 = positions.begin(); iter1 != positions.end(); ++iter1)
    {
        for (auto iter2 = positions.begin(); iter2 != positions.end(); ++iter2)
        {
            double tempTime = Util::calcTravelTime(*iter1, *iter2);
            if (tempTime != 0.0 && tempTime < minTravelTime)
            {
                minTravelTime = tempTime;
            }
        }
    }
    return minTravelTime;
}