#include "generator.h"
#include "util.h"
#include <ctime>
#include <algorithm>
#include <numeric>
#include "NumCpp.hpp"

bool Generator::sortAscend(const pair<double, Customer*> a, const pair<double, Customer*> b)
{
	return a.first < b.first;
}

int Generator::restaurants_selector(vector<double> users_cvr, double random_num)
{
	double sum = accumulate(users_cvr.begin(), users_cvr.end(), 0.0);
	double sample = 0;
	int restaurant_id = 0;
	for (auto iter = users_cvr.begin(); iter != users_cvr.end(); ++iter) {
		sample += *iter / sum;
		if (sample >= random_num) {
			return restaurant_id + 50;
		}
		restaurant_id++;
	}
}

void Generator::instanceGenenrator(bool testInstanceGenerate, list<pair<double, Customer*>>* sequenceData, string fileName, vector<pair<double, double>> pos_matrix, bool is_synergy)
{
	if (!testInstanceGenerate)
	{
		sequenceData->clear();
	}
	list<pair<double, Customer*>> generatedCustomers;
	vector<vector<double>> cvr_matrix;
	string file_name;
	if (testInstanceGenerate) {
		file_name = "./dataset/user_cvr_matrix.csv";
	}
	else {
		//file_name = "./dataset/cvr_prediction_matrix.csv";
		file_name = "./dataset/user_cvr_matrix.csv";
	}
	ifstream matrix_file(file_name, ios::in);
	while (!matrix_file.eof()) {
		int count = 0;
		vector<double> cvr_for_one_user;
		while (count < 100) {
			double cvr;
			matrix_file >> cvr;
			cvr_for_one_user.push_back(cvr);
			count++;
		}
		cvr_matrix.push_back(cvr_for_one_user);
		count = 0;
	}
	double timeWindowLength = 60.0, blankLength = 10.0;
	int customer_count = 0, maxDemand = 5;
	while (customer_count < CUSTOMER_NUMBER) {
		Customer* customer = new Customer();
		int user_id = rand() % 50;
		int restaurant_id;
		if (is_synergy) {
			restaurant_id = Generator::restaurants_selector(cvr_matrix[user_id], rand() / double(RAND_MAX));
		}
		else {
			restaurant_id = rand() % 100 + 50;
		}
		customer->origin.x = pos_matrix[restaurant_id].first;
		customer->origin.y = pos_matrix[restaurant_id].second;
		customer->dest.x = pos_matrix[user_id].first;
		customer->dest.y = pos_matrix[user_id].second;
		double appearTime = (double)int(rand() / double(RAND_MAX) * (MAX_WORK_TIME - timeWindowLength - blankLength));
		customer->startTime = appearTime + blankLength;
		customer->endTime = customer->startTime + timeWindowLength;
		customer->weight = rand() % maxDemand;
		char idString[] = { char(customer_count / 1000 + 48), char(customer_count % 1000 / 100 + 48),
						   char(customer_count % 100 / 10 + 48), char(customer_count % 10 + 48), '\0' };
		customer->id = idString;
		if (!testInstanceGenerate)
		{
			sequenceData->push_back(make_pair(appearTime, customer));
		}
		else
		{
			generatedCustomers.push_back(make_pair(appearTime, customer));
		}
		customer_count++;
	}
	if (!testInstanceGenerate)
	{
		sequenceData->sort(sortAscend);
	}
	else
	{
		generatedCustomers.sort(sortAscend);
	}
	if (testInstanceGenerate)
	{
		ofstream outFile(fileName, ios::out);
		for (auto iter = generatedCustomers.begin(); iter != generatedCustomers.end(); ++iter)
		{
			outFile << iter->first << " ";
			outFile << iter->second->id << " ";
			outFile << iter->second->origin.x << " ";
			outFile << iter->second->origin.y << " ";
			outFile << iter->second->dest.x << " ";
			outFile << iter->second->dest.y << " ";
			outFile << iter->second->startTime << " ";
			outFile << iter->second->endTime << " ";
			outFile << iter->second->weight << " ";
			outFile << iter->second->priority << endl;
			delete iter->second;
		}
		outFile.close();
	}
}

void Generator::instanceGenenrator(bool testInstanceGenerate, list<pair<double, Customer*>>* sequenceData, string fileName)
{
	if (!testInstanceGenerate)
	{
		sequenceData->clear();
	}
	list<pair<double, Customer*>> generatedCustomers;
	random_device rd;
	default_random_engine e(rd());
	double shopLocation = 10.0, serviceRange = 5.0;
	uniform_real_distribution<double> ratio(0.0, 1.0);
	double cancellationRatio = 0, hurryRatio = 0, timeWindowLength = 60.0, blankLength = 10.0, maxDemand = 5.0;
	double staticCustomer = double(CUSTOMER_NUMBER) * (1 - DEGREE_OF_DYNAMISM);
	int customerCount = 0;
	double staticCustomerCount = 0.0;
	double cluster[3][4] = { {-10.0, 0.0, 0.0, 10.0}, {0.0, 10.0, -10.0, 0.0}, {0.0, 10.0, 0.0, 10.0} };
	int clusterNum = 3;
	int clusterIndex = 0, clusterCount = 0, clusterLimit = CUSTOMER_NUMBER - (int)staticCustomer / clusterNum;
	double lambda = (double)CUSTOMER_NUMBER / ((double)MAX_WORK_TIME - timeWindowLength - blankLength), i = 0.0;
	vector<double> apStore;
	while (i < (MAX_WORK_TIME - timeWindowLength - blankLength))
	{
		i += -(1 / lambda) * log(ratio(e));
		apStore.push_back(i);
	}
	while (customerCount++ < CUSTOMER_NUMBER)
	{
		/*if (customerCount > apStore.size())
		{
			break;
		}*/
		Customer* customer = new Customer();
		//normal_distribution<double> ap(360, 120);
		//double appearTime = max(0.0,min(650.0,ap(e)));//(MAX_WORK_TIME - timeWindowLength - blankLength) * ap(e);
		//double appearTime = apStore[customerCount - 1];
		double appearTime = ratio(e) * (MAX_WORK_TIME - timeWindowLength - blankLength);
		if (staticCustomerCount++ < staticCustomer)
		{
			customer->origin.x = 0.0;
			customer->origin.y = 0.0;
			uniform_real_distribution<double> customerPosX(-shopLocation, shopLocation);
			uniform_real_distribution<double> customerPosy(-shopLocation, shopLocation);
			customer->dest.x = customerPosX(e);
			customer->dest.y = customerPosy(e);
			customer->startTime = 0;
			customer->endTime = MAX_WORK_TIME;
			customer->weight = ratio(e) * maxDemand * 10.0;
			char idString[] = { char(customerCount / 1000 + 48), char(customerCount % 1000 / 100 + 48),
							   char(customerCount % 100 / 10 + 48), char(customerCount % 10 + 48), '\0' };
			customer->id = idString;
			if (!testInstanceGenerate)
			{
				sequenceData->push_back(make_pair(0, customer));
			}
			else
			{
				generatedCustomers.push_back(make_pair(0, customer));
			}
			continue;
		}
		if (clusterIndex + 1 < clusterNum && clusterCount > clusterLimit)
		{
			clusterIndex++;
		}
		clusterCount++;
		uniform_real_distribution<double> shopPosX(-shopLocation, shopLocation);
		uniform_real_distribution<double> shopPosY(-shopLocation, shopLocation);
		customer->origin.x = shopPosX(e);
		customer->origin.y = shopPosY(e);
		/*uniform_real_distribution<double> customerPosX(max(-shopLocation, customer->origin.x - serviceRange),
													   min(shopLocation, customer->origin.x + serviceRange));
		uniform_real_distribution<double> customerPosy(max(-shopLocation, customer->origin.y - serviceRange),
													   min(shopLocation, customer->origin.y + serviceRange));*/
		uniform_real_distribution<double> customerPosX(-shopLocation, shopLocation);
		uniform_real_distribution<double> customerPosy(-shopLocation, shopLocation);
		customer->dest.x = customerPosX(e);
		customer->dest.y = customerPosy(e);
		customer->startTime = appearTime + blankLength;
		customer->endTime = customer->startTime + timeWindowLength;
		customer->weight = ratio(e) * maxDemand;
		char idString[] = { char(customerCount / 1000 + 48), char(customerCount % 1000 / 100 + 48),
						   char(customerCount % 100 / 10 + 48), char(customerCount % 10 + 48), '\0' };
		customer->id = idString;
		if (!testInstanceGenerate)
		{
			sequenceData->push_back(make_pair(appearTime, customer));
		}
		else
		{
			generatedCustomers.push_back(make_pair(appearTime, customer));
		}
		double isCanceled = ratio(e), isHurry = ratio(e);
		if (isCanceled <= cancellationRatio)
		{
			Customer* cancel = new Customer();
			Util::infoCopy(cancel, customer);
			cancel->priority = 0;
			double cancelTime = customer->startTime - blankLength + ratio(e) * blankLength;
			if (!testInstanceGenerate)
			{
				sequenceData->push_back(make_pair(cancelTime, cancel));
			}
			else
			{
				generatedCustomers.push_back(make_pair(cancelTime, cancel));
			}
		}
		if (isHurry <= hurryRatio)
		{
			Customer* hurry = new Customer();
			Util::infoCopy(hurry, customer);
			hurry->priority = 2;
			double hurryTime = customer->startTime + timeWindowLength / 2.0 + ratio(e) * timeWindowLength / 2.0;
			if (!testInstanceGenerate)
			{
				sequenceData->push_back(make_pair(hurryTime, hurry));
			}
			else
			{
				generatedCustomers.push_back(make_pair(hurryTime, hurry));
			}
		}
	}
	if (!testInstanceGenerate)
	{
		sequenceData->sort(sortAscend);
	}
	else
	{
		generatedCustomers.sort(sortAscend);
	}
	if (testInstanceGenerate)
	{
		ofstream outFile(fileName, ios::out);
		for (auto iter = generatedCustomers.begin(); iter != generatedCustomers.end(); ++iter)
		{
			outFile << iter->first << " ";
			outFile << iter->second->id << " ";
			outFile << iter->second->origin.x << " ";
			outFile << iter->second->origin.y << " ";
			outFile << iter->second->dest.x << " ";
			outFile << iter->second->dest.y << " ";
			outFile << iter->second->startTime << " ";
			outFile << iter->second->endTime << " ";
			outFile << iter->second->weight << " ";
			outFile << iter->second->priority << endl;
			delete iter->second;
		}
		outFile.close();
	}
}
