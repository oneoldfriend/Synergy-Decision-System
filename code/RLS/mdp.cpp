#include "mdp.h"
#include "route.h"
#include "generator.h"

using namespace std;

double Action::rejectionReward()
{
    double count = 0;
    for (auto iter = this->customerConfirmation.begin(); iter != this->customerConfirmation.end(); ++iter)
    {
        if (iter->second == false)
        {
            count++;
        }
    }
    return count * REJECTION_COST;
}

State::State()
{
    this->cumOutsourcedCost = 0.0;
    this->currentTime = 0;
    this->currentRoute = nullptr;
    this->attributes = Eigen::VectorXd(ATTRIBUTES_NUMBER);
}

void State::calcAttribute(Action a, bool assignment)
{
    if (assignment)
    {
        int acceptNum = 0;
        for (auto iter = a.customerConfirmation.begin(); iter != a.customerConfirmation.end(); iter++)
        {
            if (iter->second)
            {
                acceptNum++;
            }
        }
        this->pointSolution->calcInfo();
        this->attributes[0] = 100;
        this->attributes[1] = this->currentRoute->currentPos->departureTime;
        this->attributes[2] = this->pointSolution->info[2];
        this->attributes[3] = this->notServicedCustomer.size() + acceptNum;
        this->attributes[4] = this->cumOutsourcedCost / (double)REJECTION_COST + (double)a.customerConfirmation.size() - (double)acceptNum;
    }
    else
    {
        this->pointSolution->calcInfo();
        this->attributes[0] = 100;
        this->attributes[1] = this->currentRoute->currentPos->departureTime;
        this->attributes[2] = this->pointSolution->info[2];
        if (a.positionToVisit != nullptr && a.positionToVisit->isOrigin)
        {
            this->attributes[3] = this->notServicedCustomer.size() + 1;
        }
        else
        {
            this->attributes[3] = this->notServicedCustomer.size();
            if (a.positionToVisit != nullptr)
            {
                this->attributes[2] = this->pointSolution->info[2] + 1;
            }
        }
        this->attributes[4] = this->pointSolution->info[1];
    }
}

void MDP::executeAction(Action a)
{
    if (a.positionToVisit != nullptr)
    {
        if (a.positionToVisit->isOrigin)
        {
            PointOrder dest = this->currentState.notServicedCustomer[a.positionToVisit->customer->id].second;
            this->currentState.currentRoute->removeOrder(a.positionToVisit);
            this->currentState.currentRoute->removeOrder(dest);
            a.positionToVisit->prior = this->currentState.currentRoute->currentPos;
            a.positionToVisit->next = this->currentState.currentRoute->currentPos->next;
            dest->prior = a.positionToVisit;
            dest->next = a.positionToVisit->next;
            this->currentState.currentRoute->insertOrder(a.positionToVisit);
            this->currentState.currentRoute->insertOrder(dest);
        }
        else
        {
            this->currentState.currentRoute->removeOrder(a.positionToVisit);
            a.positionToVisit->prior = this->currentState.currentRoute->currentPos;
            a.positionToVisit->next = this->currentState.currentRoute->currentPos->next;
            this->currentState.currentRoute->insertOrder(a.positionToVisit);
        }
    }
    else
    {
        this->currentState.currentRoute->currentPos->departureTime += UNIT_TIME;
        this->currentState.currentRoute->currentPos->waitTime += UNIT_TIME;
    }
    this->currentState.pointSolution->solutionUpdate();
}

void MDP::undoAction(Action a)
{
    if (a.positionToVisit != nullptr)
    {
        if (a.positionToVisit->isOrigin)
        {
            PointOrder dest = a.positionToVisit->next;
            this->currentState.currentRoute->removeOrder(a.positionToVisit);
            this->currentState.currentRoute->removeOrder(dest);
            a.positionToVisit->next->prior = a.destPosBeforeExecution.first;
            a.positionToVisit->next->next = a.destPosBeforeExecution.second;
            a.positionToVisit->prior = a.originPosBeforeExecution.first;
            a.positionToVisit->next = a.originPosBeforeExecution.second;
            this->currentState.currentRoute->insertOrder(dest);
            this->currentState.currentRoute->insertOrder(a.positionToVisit);
        }
        else
        {
            this->currentState.currentRoute->removeOrder(a.positionToVisit);
            a.positionToVisit->prior = a.destPosBeforeExecution.first;
            a.positionToVisit->next = a.destPosBeforeExecution.second;
            this->currentState.currentRoute->insertOrder(a.positionToVisit);
        }
    }
    else
    {
        this->currentState.currentRoute->currentPos->departureTime -= UNIT_TIME;
        this->currentState.currentRoute->currentPos->waitTime -= UNIT_TIME;
    }
    this->currentState.pointSolution->solutionUpdate();
}

void MDP::integerToRoutingAction(int actionNum, State S, Action *a)
{
    //根据当前动作号进行二进制转换为具体动作
    if (actionNum >= 0)
    {
        a->positionToVisit = S.reachableCustomer[actionNum];
        if (!a->positionToVisit->isOrigin)
        {
            a->destPosBeforeExecution.first = S.reachableCustomer[actionNum]->prior;
            a->destPosBeforeExecution.second = S.reachableCustomer[actionNum]->next;
        }
        else
        {
            a->originPosBeforeExecution.first = a->positionToVisit->prior;
            a->originPosBeforeExecution.second = a->positionToVisit->next;
            a->destPosBeforeExecution.first = S.notServicedCustomer[a->positionToVisit->customer->id].second->prior;
            a->destPosBeforeExecution.second = S.notServicedCustomer[a->positionToVisit->customer->id].second->next;
        }
    }
    else
    {
        a->positionToVisit = nullptr;
    }
}

void MDP::integerToAssignmentAction(int actionNum, State S, Action *a)
{
    //根据当前动作号进行二进制转换为具体动作
    int leftOver = actionNum;
    for (auto customerIter = S.newCustomers.begin(); customerIter != S.newCustomers.end(); ++customerIter)
    {
        if (leftOver % 2 == 1)
        {
            a->customerConfirmation[*customerIter] = true;
        }
        else
        {
            a->customerConfirmation[*customerIter] = false;
        }
        leftOver = leftOver / 2;
    }
}

void MDP::findBestAssignmentAction(Action *a, ValueFunction valueFunction, double *reward, bool myopic)
{
    int actionNum = 0, maxActionNum = pow(2, this->currentState.newCustomers.size()), bestActionNum = -1;
    double bestActionValue = MAX_COST;
    if (this->currentState.newCustomers.size() != 0)
    {
        //若有新顾客被观察到
        while (actionNum < maxActionNum)
        {
            //检查每个可能动作的可行性并对可行动作进行评估
            Action tempAction;
            double actionValue = 0;
            this->integerToAssignmentAction(actionNum, this->currentState, &tempAction);
            double immediateReward = 0;
            if (this->checkAssignmentActionFeasibility(tempAction, &immediateReward))
            {
                actionValue = immediateReward + valueFunction.getValue(this->currentState, tempAction, true, myopic);
                if (actionValue < bestActionValue)
                {
                    //记录更优的动作
                    *reward = immediateReward;
                    bestActionValue = actionValue;
                    bestActionNum = actionNum;
                }
            }
            actionNum++;
        }
    }
    this->integerToAssignmentAction(bestActionNum, this->currentState, a);
}

void MDP::findBestRoutingAction(Action *a, ValueFunction valueFunction, double *reward, bool myopic)
{
    int actionNum = 0, maxActionNum = this->currentState.reachableCustomer.size(), bestActionNum = -1;
    double bestActionValue = MAX_COST, totalValue = 0.0;
    map<int, double> actionWheel;
    while (actionNum < maxActionNum)
    {
        //检查每个可能动作的可行性并对可行动作进行评估
        Action tempAction;
        double actionValue = 0;
        this->integerToRoutingAction(actionNum, this->currentState, &tempAction);
        double immediateReward = 0;
        if (this->checkRoutingActionFeasibility(tempAction, &immediateReward))
        {
            //若动作可行，则进行评估
            actionValue = immediateReward + valueFunction.getValue(this->currentState, tempAction, false, myopic);
            if (actionValue < bestActionValue)
            {
                //记录更优的动作
                *reward = immediateReward;
                bestActionValue = actionValue;
                bestActionNum = actionNum;
            }
            totalValue += actionValue;
            actionWheel[actionNum] = actionValue;
        }
        //回撤动作继续下一个评估
        this->undoAction(tempAction);
        actionNum++;
    }
    if (false)
    {
        double prob = rand() / double(RAND_MAX), cumProb = 0.0;
        for (auto iter = actionWheel.begin(); iter != actionWheel.end(); ++iter)
        {
            cumProb += iter->second / totalValue;
            if (prob <= cumProb)
            {
                bestActionNum = iter->first;
                break;
            }
        }
    }
    this->integerToRoutingAction(bestActionNum, this->currentState, a);
}

bool MDP::checkAssignmentActionFeasibility(Action a, double *reward)
{
    double currentCost = this->currentState.pointSolution->cost, acceptNum = 0.0;
    vector<pair<PointOrder, PointOrder>> orderWaitToBeInserted;
    for (auto iter = a.customerConfirmation.begin(); iter != a.customerConfirmation.end(); ++iter)
    {
        if (iter->second)
        {
            acceptNum++;
            orderWaitToBeInserted.push_back(make_pair(new Order(iter->first, true), new Order(iter->first, false)));
        }
    }
    bool feasibility = this->currentState.pointSolution->greedyInsertion(orderWaitToBeInserted);
    for (int i = 0; i < (int)orderWaitToBeInserted.size(); i++)
    {
        this->currentState.pointSolution->routes[0].removeOrder(orderWaitToBeInserted[i].first);
        this->currentState.pointSolution->routes[0].removeOrder(orderWaitToBeInserted[i].second);
        delete orderWaitToBeInserted[i].first;
        delete orderWaitToBeInserted[i].second;
    }
    double newCost = this->currentState.pointSolution->cost;
    this->currentState.pointSolution->solutionUpdate();
    newCost += a.rejectionReward();
    *reward = newCost - currentCost;
    return feasibility;
}

bool MDP::checkRoutingActionFeasibility(Action a, double *reward)
{
    double currentCost = this->currentState.currentRoute->cost;
    this->executeAction(a);
    bool feasibility = this->currentState.currentRoute->checkFeasibility();
    double newCost = this->currentState.currentRoute->cost;
    *reward = newCost - currentCost;
    return feasibility;
}

void MDP::assignmentConfirmed(Action a)
{
    vector<pair<PointOrder, PointOrder>> orderWaitToBeInserted;
    for (auto iter = a.customerConfirmation.begin(); iter != a.customerConfirmation.end(); ++iter)
    {
        if (iter->second)
        {
            PointOrder origin = new Order(iter->first, true);
            this->currentState.reachableCustomer.push_back(origin);
            this->currentState.notServicedCustomer[iter->first->id] = make_pair(origin, new Order(iter->first, false));
            orderWaitToBeInserted.push_back(this->currentState.notServicedCustomer[iter->first->id]);
        }
        else
        {
            this->currentState.cumOutsourcedCost += REJECTION_COST;
        }
    }
    this->currentState.pointSolution->greedyInsertion(orderWaitToBeInserted);
}

MDP::MDP(bool approx, string fileName, list<pair<double, Customer *>> *data)
{
    if (approx)
    {
        this->sequenceData = *data;
    }
    else
    {
        ifstream trainFile(fileName, ios::in);
        double appearTime = 0;
        while (!trainFile.eof())
        {
            //读取instance数据
            trainFile >> appearTime;
            Customer *customer = new Customer();
            trainFile >> customer->id;
            trainFile >> customer->origin.x;
            trainFile >> customer->origin.y;
            trainFile >> customer->dest.x;
            trainFile >> customer->dest.y;
            trainFile >> customer->startTime;
            trainFile >> customer->endTime;
            trainFile >> customer->weight;
            trainFile >> customer->priority;
            this->sequenceData.push_back(make_pair(appearTime, customer));
        }
        auto last = this->sequenceData.rbegin();
        delete last->second;
        this->sequenceData.pop_back();
        trainFile.close();
    }
    this->solution = Solution();
    this->currentState = State();
    for (auto iter = this->sequenceData.begin(); iter != this->sequenceData.end(); ++iter)
    {
        if (iter->first == 0.0)
        {
            //加入提前已知的顾客
            this->currentState.newCustomers.push_back(iter->second);
        }
        this->customers[iter->second->id] = iter->second;
    }
    //状态当前车辆初始化为第一辆车
    this->currentState.currentRoute = &this->solution.routes[0];
    this->currentState.pointSolution = &this->solution;
    //this->minTravelTime = Util::minTravelTimeCalc(this->sequenceData);
    //cout << this->minTravelTime << endl;
}

MDP::~MDP()
{
    for (auto iter = this->solution.routes.begin(); iter != this->solution.routes.end(); iter++)
    {
        PointOrder p = iter->head;
        while (p != nullptr)
        {
            PointOrder pNext = p->next;
            delete p;
            p = pNext;
        }
    }
}

double MDP::reward(State S, Action a)
{
    //复制当前解并对副本执行动作计算立即反馈
    double currentCost = S.currentRoute->cost;
    this->executeAction(a);
    double newCost = S.currentRoute->cost;
    this->undoAction(a);
    return newCost - currentCost;
}

void MDP::transition(Action a)
{
    //执行动作
    double lastDecisionTime = this->currentState.currentTime;
    this->executeAction(a);
    //更新当前状态
    if (a.positionToVisit != nullptr)
    {
        //在未服务位置集中删除该位置
        for (auto iter = this->currentState.notServicedCustomer.begin(); iter != this->currentState.notServicedCustomer.end(); ++iter)
        {
            if (iter->second.second == a.positionToVisit)
            {
                this->currentState.notServicedCustomer.erase(iter);
                break;
            }
            else if (iter->second.first == a.positionToVisit)
            {
                iter->second.first = nullptr;
            }
        }
        //更新当前路径的available position为下一个点
        this->currentState.currentRoute->currentPos = this->currentState.currentRoute->currentPos->next;
    }
    else
    {
        //若选择原地待命则检查车辆是否能原地待命或者返回仓库结束配送
        if (this->currentState.currentRoute->tail->departureTime + UNIT_TIME > MAX_WORK_TIME)
        {
            //若车辆不能原地等待则直接返回仓库结束配送
            this->undoAction(a);
            this->currentState.currentRoute->currentPos = this->currentState.currentRoute->tail;
        }
    }
    //this->checkIgnorance(a);
    this->currentState.currentTime = MAX_WORK_TIME;
    this->currentState.currentRoute = nullptr;
    //找到下一辆空闲车辆,更新状态相关信息
    for (auto iter = this->solution.routes.begin(); iter != this->solution.routes.end(); ++iter)
    {
        //检查车辆是否已经结束配送
        if (iter->currentPos == iter->tail)
        {
            continue;
        }
        else if (this->currentState.currentTime > iter->currentPos->departureTime)
        {
            this->currentState.currentTime = iter->currentPos->departureTime;
            this->currentState.currentRoute = &*iter;
        }
    }
    //观察新顾客信息
    this->observation(lastDecisionTime);
}

/*void MDP::checkIgnorance(Action a)
{
    for (auto iter = this->currentState.notServicedCustomer.begin(); iter != this->currentState.notServicedCustomer.end();)
    {
        if (iter->second.first != nullptr)
        {
            if (a.positionToVisit != iter->second.first &&
                this->currentState.currentTime - iter->second.first->customer->startTime > IGNORANCE_TOLERANCE)
            {
                this->cumOutsourcedCost += MAX_WORK_TIME;
                delete iter->second.first;
                delete iter->second.second;
                this->currentState.notServicedCustomer.erase(iter++);
            }
            else
            {
                ++iter;
            }
        }
        else
        {
            ++iter;
        }
    }
}*/

void MDP::observation(double lastDecisionTime)
{
    this->currentState.newCustomers.clear();
    for (auto sequenceIter = this->sequenceData.begin(); sequenceIter != this->sequenceData.end();)
    {
        if (sequenceIter->first > this->currentState.currentTime)
        {
            break;
        }
        //找到上次决策时间到当前决策时间中产生的新顾客信息
        if (sequenceIter->first > lastDecisionTime && sequenceIter->first <= this->currentState.currentTime)
        {
            if (sequenceIter->second->priority == 1)
            {
                //若有新顾客产生，则将其加入到待插入顾客集中
                this->customers[sequenceIter->second->id] = sequenceIter->second;
                this->currentState.newCustomers.push_back(sequenceIter->second);
            }
            else
            {
                //若观察到顾客退单或催单，则直接对原有顾客信息进行更新
                for (auto notSrvCstm = this->currentState.notServicedCustomer.begin(); notSrvCstm != this->currentState.notServicedCustomer.end();)
                {
                    if (notSrvCstm->second.second->customer->id == sequenceIter->second->id)
                    {
                        notSrvCstm->second.second->customer->priority = sequenceIter->second->priority;
                        if (sequenceIter->second->priority == 0)
                        {
                            this->currentState.notServicedCustomer.erase(notSrvCstm++);
                        }
                        else
                        {
                            ++notSrvCstm;
                        }
                    }
                    else
                    {
                        ++notSrvCstm;
                    }
                }
            }
        }
        this->sequenceData.erase(sequenceIter++);
    }
    //更新当前车辆可以合法访问的点
    this->currentState.reachableCustomer.clear();
    if (this->currentState.currentRoute != nullptr)
    {
        for (auto iter = this->currentState.notServicedCustomer.begin(); iter != this->currentState.notServicedCustomer.end(); ++iter)
        {
            if (iter->second.first == nullptr)
            {
                PointOrder p = this->currentState.currentRoute->head->next;
                while (p != this->currentState.currentRoute->tail)
                {
                    if (p->customer->id == iter->second.second->customer->id)
                    {
                        this->currentState.reachableCustomer.push_back(iter->second.second);
                        break;
                    }
                    else
                    {
                        p = p->next;
                    }
                }
            }
            else
            {
                this->currentState.reachableCustomer.push_back(iter->second.first);
            }
        }
    }
    //update the solution(delete the customers with cancellation)
    for (auto routeIter = this->solution.routes.begin(); routeIter != this->solution.routes.end(); ++routeIter)
    {
        PointOrder p = routeIter->currentPos;
        bool cancellation = false;
        while (p != routeIter->tail)
        {
            //检查若有顾客退单，则将其从解中删除
            if (p->customer->priority == 0)
            {
                cancellation = true;
                PointOrder tempPtr = p->next;
                if (p == routeIter->currentPos)
                {
                    p->currentWeight -= p->customer->weight;
                }
                else
                {
                    routeIter->removeOrder(p);
                    delete p;
                }
                p = tempPtr;
            }
            else
            {
                p = p->next;
            }
        }
        if (cancellation)
        {
            //若有顾客退单导致解的变化，则更新对应路径的信息
            routeIter->routeUpdate();
        }
    }
}
