#pragma once
#include "stdafx.h"
#include <armadillo>
#include "nlohmann/json.hpp"

using json = nlohmann::json;

/********************
STATE VALUES
0 - trained classification
1 - collect data for training
2 - training
3 - standard classification
********************/

std::atomic<int> state = 3;
//std::atomic<int> err = TOBII_ERROR_NO_ERROR;
//std::atomic<int> prev_err = TOBII_ERROR_NO_ERROR;
std::mutex mtx;
arma::mat transitions_model("0,0;"); 
std::atomic<bool> custom_transition = false;
std::atomic<bool> reset_timer = false;
std::atomic<bool> pro_driver = false;
std::atomic<bool> training_video_end = false;

std::atomic<int> untrained_fixation_time = 600;
std::atomic<double> untrained_fixation_radius = 0.05;
std::atomic<double> trained_fixation_time = 0.5;
//std::atomic<std::string> session = "";
 

std::string serialize_mat(arma::mat& X)
{
	std::stringstream output;

	for (arma::uword row = 0; row < X.n_rows; row++)
	{
		for (arma::uword col = 0; col < X.n_cols; col++) {
			output << X(row, col);
			if (col != X.n_cols - 1) output << ' ';
		}

		output << ";";
	}
	return output.str();
}

void set_transitions_model(arma::mat mat) {
	mtx.lock();
	transitions_model = mat;
	custom_transition = true;
	mtx.unlock();
}

void set_transitions_model(std::string smat) {
	mtx.lock();
	transitions_model = arma::mat(smat);
	custom_transition = true;
	mtx.unlock();
}

arma::mat get_transitions_model() {
	return transitions_model;
}

void set_untrained_fixation_time(int t) {
	untrained_fixation_time = t;
}

int get_untrained_fixation_time() {
	return untrained_fixation_time * 1000;
}

void set_reset_timer(bool val) {
	reset_timer = val;
}
bool get_reset_timer() {
	return reset_timer;
}

void set_trained_fixation_time(double t) {
	trained_fixation_time = t;
}

double get_trained_fixation_time() {
	return trained_fixation_time;
}

void set_untrained_fixation_radius(double r) {
	untrained_fixation_radius = r;
}

double get_untrained_fixation_radius() {
	return untrained_fixation_radius;
}


void write_state() {

	json jsonDump = {
	{ "state",state.load() },
	{ "transitions_model",serialize_mat(get_transitions_model()) },
	{ "custom_transition",custom_transition.load() },
	{ "untrained_fixation_time",untrained_fixation_time.load() },
	{ "untrained_fixation_radius",untrained_fixation_radius.load() },
	{ "trained_fixation_time",trained_fixation_time.load() }
	};

	// write prettified JSON to another file
	std::ofstream o("lastState.json");
	o << std::setw(4) << jsonDump << std::endl;

}

bool recover_state() {
	std::ifstream ifs("lastState.json");
	if (ifs.fail()) {
		std::cout << "\nfile not found, skipping\n";
		return false;
	}

	bool should_reset = false;
	
	json j = json::parse(ifs);
	
	state = j["state"].get<int>();
	
	if (state == 1 || state == 2) {
		state = 0;
		should_reset = true;
	}

	set_transitions_model(j["transitions_model"].get<std::string>());
	custom_transition = j["custom_transition"].get<bool>();
	trained_fixation_time = j["trained_fixation_time"].get<int>();
	untrained_fixation_radius = j["untrained_fixation_radius"].get<int>();
	untrained_fixation_time = j["untrained_fixation_time"].get<int>();

	std::cout << "recovered state is " << state;

	return should_reset;
}

void set_state(int i) {
	state = i;
	write_state();
}

int get_state() {
	return state;
}


