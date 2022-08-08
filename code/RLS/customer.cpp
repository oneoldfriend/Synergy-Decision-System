#include"customer.h"
#include"avi.h"


Customer::Customer(){
    this->priority = 1;
    this->startTime = 0;
    this->endTime = MAX_WORK_TIME;
    this->weight = 0;
    this->origin.x = 0;
    this->origin.y = 0;
    this->dest.x = 0;
    this->dest.y = 0;
    this->id = "0000";
}