#pragma once
#include "stdafx.h"
#include <vector>
#include <stdio.h>
#include <tobii/tobii.h>
#include <tobii/tobii_streams.h>
#include "state_controller.hpp"
#include<math.h>

double u_centroid_x = -1;
double u_centroid_y = -1;

std::vector<unsigned int> u_ti;


double distanceCalculate(double x1, double y1, double x2, double y2)
{
	double x = x1 - x2; //calculating number to square in next step
	double y = y1 - y2;
	double dist;

	dist = pow(x, 2) + pow(y, 2);       //calculating Euclidean distance
	dist = sqrt(dist);

	return dist;
}

void untrained_classify(tobii_gaze_point_t const* gaze_point, broadcast_server* server_instance) {

	if (u_centroid_x < 0 || u_centroid_y < 0 || get_reset_timer()) {
		set_reset_timer(false);
		u_ti.clear();
		u_centroid_x = gaze_point->position_xy[0];
		u_centroid_y = gaze_point->position_xy[1];
		u_ti.push_back(gaze_point->timestamp_us);
	}

	else {
		double dist = distanceCalculate(u_centroid_x, u_centroid_y, gaze_point->position_xy[0], gaze_point->position_xy[1]);
		
		if (dist <= get_untrained_fixation_radius()) {
			u_ti.push_back(gaze_point->timestamp_us);
			
			if (u_ti.back() - u_ti.front() >= get_untrained_fixation_time()) {

				//send fixation
				server_instance->sendall("{\"type\":\"fixation\",\"data\":[" + std::to_string(u_centroid_x) + "," + std::to_string(u_centroid_y) + "]}");
				u_centroid_x = -1;
				u_centroid_y = -1;
			}
		}

		else {
			u_ti.clear();
			u_centroid_x = gaze_point->position_xy[0];
			u_centroid_y = gaze_point->position_xy[1];
			u_ti.push_back(gaze_point->timestamp_us);
		}
	}
}