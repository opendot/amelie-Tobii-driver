#pragma once
#include "stdafx.h"
#include <tobii/tobii.h>
#include <tobii/tobii_streams.h>
#include "nslr_hmm.hpp"
#include "state_controller.hpp"
#include <Eigen/StdVector>


std::vector<std::vector<double>> times_arr;
std::vector<std::vector<double>> x_arr;
std::vector<std::vector<double>> y_arr;
int times_count = 0;

std::vector<double> times;
std::vector<double> x;
std::vector<double> y;
std::vector<int> chunks;
double dist = 40;
double w = 0;
double h = 0;
const unsigned int min_chunk_size = 800;
const unsigned int n_training_samples = 3000;
const unsigned int max_offscreen_time = 100000;
const unsigned int hz = 90;
const double min_fixation_time = 0.6;


void reset_data() {
	chunks.clear();
	times.clear();
	x.clear();
	y.clear();
}


int get_longest_chunk(std::vector<int>& vec, bool index) {
	
	int max = -1;
	int ind = -1;

	for (int i = 0; i < vec.size()-1; i++) {
		if (vec[i + 1] - vec[i] > max) { 
			max = vec[i + 1] - vec[i];
			ind = i;
		}
	}
	if (index) return ind;
	return max;
}

std::string mat_to_string (arma::mat& X)
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


double min_max_fixation_time(std::tuple<Segmentation<Segment<Nslr2d::Vector>>, std::vector<unsigned int>, std::vector<unsigned int>> result) {

	double min = 1e6;
	double max = 0;
	double cumulative = 0;
	std::vector<double> fixations;
	int n = 0;
	std::vector<unsigned int> seg_classes = std::get<1>(result);
	auto segmentation = std::get<0>(result);

	for (int i = 0; i < seg_classes.size(); i++) {
		if (seg_classes[i] == 0) {
			n++;
			double duration = std::get<1>(segmentation.segments[i].t) - std::get<0>(segmentation.segments[i].t);
			fixations.push_back(duration);
			cumulative += duration;
		}
	}

	std::sort(fixations.begin(), fixations.end());
	int quartile_ind = (fixations.size() + 1) / 4 * 3;

	std::cout << "\nn. of fixations: " << fixations.size()  << "\nmin fixation time: " << fixations[0] << "\n max fixation time: " << fixations[fixations.size() - 1] << "\n avg fixation time: " << cumulative / fixations.size() << "\n third quartile: " << fixations[quartile_ind] << std::endl;
	//return fixations[quartile_ind] < 0.6 ? 0.6 : fixations[quartile_ind];
	return 0.6;
}

//prepare data for feature extraction - pixels to angles
Array<double, Dynamic, 2> pixels_to_angles(Ref<Array<double, Dynamic, 2>> xy, double dist, double w, double h) {

	Array<double, Dynamic, 2> coords(xy);
	coords -= 0.5;
	coords.col(0) = coords.col(0) * w;
	coords.col(1) = coords.col(1) * h;
	coords = coords.atan() / dist;
	Array<double, Dynamic, 2> angles(coords.rows(), 2);
	angles = (coords * 180) / M_PI;

	return angles;
}

Array<double, Dynamic, 1> fix_time(Ref<Array<double, Dynamic, 1>> ts) {
	ts -= ts[0];
	ts /= 1e6;
	return ts;
}


//check for fixations
void analyze_fixations(std::tuple<Segmentation<Segment<Nslr2d::Vector>>, std::vector<unsigned int>, std::vector<unsigned int>> result, Ref<Array<double, Dynamic, 2>> xy, broadcast_server* server_instance) {

	std::vector<unsigned int> seg_classes = std::get<1>(result);
	if (std::find(seg_classes.begin(), seg_classes.end(), 0) != seg_classes.end()) {

		auto segmentation = std::get<0>(result);

		for (int i = 0; i < seg_classes.size(); i++) {
			if (seg_classes[i] == 0) {

				double duration = std::get<1>(segmentation.segments[i].t) - std::get<0>(segmentation.segments[i].t);
				

				if (duration > trained_fixation_time) {
					//std::cout << "\nfixation detected";
					size_t start = std::get<0>(segmentation.segments[i].i);
					size_t end = std::get<1>(segmentation.segments[i].i);

					Array<double, Dynamic, 2> fix_coords = xy.middleRows(start, end - start);
					double coord_x = fix_coords.col(0).mean();
					double coord_y = fix_coords.col(1).mean();

					server_instance->sendall("{\"type\":\"fixation\",\"data\":[" + std::to_string(coord_x) + "," + std::to_string(coord_y) + "]}");

				}
			}
		}
	}
	return;
}

void prepare_data(Ref<ArrayX2d> eye, Ref<ArrayXd> t, Ref<ArrayX2d> xy)
{
	//prepare matrices: pointer to std::vectors
	double* ptr = &times[0];
	double* ptrx = &x[0];
	double* ptry = &y[0];

	//copy data from std::vector to eigen::array
	Eigen::Map<Array<double, Dynamic, 1>> ts(ptr, times.size());
	Eigen::Map<Array<double, Dynamic, 1>> xs(ptrx, x.size());
	Eigen::Map<Array<double, Dynamic, 1>> ys(ptry, y.size());

	//build the xy matrix
	//xy.resize(x.size(), 2);
	xy.col(0) = xs;
	xy.col(1) = ys;
	eye = pixels_to_angles(xy, dist, w, h);
	t = fix_time(ts);
}

void prepare_data(Ref<ArrayX2d> eye, Ref<ArrayXd> t, Ref<ArrayX2d> xy, std::vector<double> &times,  std::vector<double> &x, std::vector<double> &y)
{
	//prepare matrices: pointer to std::vectors
	double* ptr = &times[0];
	double* ptrx = &x[0];
	double* ptry = &y[0];

	//copy data from std::vector to eigen::array
	Eigen::Map<Array<double, Dynamic, 1>> ts(ptr, times.size());
	Eigen::Map<Array<double, Dynamic, 1>> xs(ptrx, x.size());
	Eigen::Map<Array<double, Dynamic, 1>> ys(ptry, y.size());

	//build the xy matrix
	//xy.resize(x.size(), 2);
	xy.col(0) = xs;
	xy.col(1) = ys;
	eye = pixels_to_angles(xy, dist, w, h);
	t = fix_time(ts);
}

void perform_training(broadcast_server* server_instance) {
	ArrayX2d eye(times.size(), 2);
	ArrayXd t(times.size());
	ArrayX2d xy(times.size(), 2);

	//estimate new transition model
	prepare_data(eye, t, xy, times, x, y);


	if (chunks.size() > 2) set_transitions_model(cumulative_dataset_features(t, eye, chunks, 0.14, false));
	else set_transitions_model(dataset_features(t, eye, 0.14, false));
	custom_transition = true;


	//find longest chunk
	int ind = get_longest_chunk(chunks, true);
	int chunkLength = chunks[ind + 1] - chunks[ind];
	//check minimum and maximum fixation times
	std::tuple<Segmentation<Segment<Nslr2d::Vector>>, std::vector<unsigned int>, std::vector<unsigned int>> result = classify_gaze(t.block(chunks[ind], 0, chunkLength, 1), eye.block(chunks[ind], 0, chunkLength, 2), 0.14, false, get_transitions_model());
	double val = min_max_fixation_time(result);
	

	std::string output_mat = mat_to_string(get_transitions_model());

	trained_fixation_time = val < min_fixation_time ? min_fixation_time : val;
	reset_data();

	server_instance->sendall("{\"type\":\"TRAINING_END\",\"data\":{\"transition_matrix\":\"" + output_mat + "\",\"trained_fixation_time\":" + std::to_string(val) + "}}");

	set_state(0);
}


void handle_new_training_data(tobii_gaze_point_t const* gaze_point, broadcast_server* server_instance) {

	if (training_video_end) {
		server_instance->sendall("{\"type\":\"TRAINING_FAILED\"}");
		reset_data();
		set_state(0);
		std::cout << "\ntraining failed - video ended" << std::endl;
		training_video_end = false;
		return;
	}

	if ((!times.empty() && gaze_point->timestamp_us - times[times.size() - 1] > max_offscreen_time)) {
		
		chunks.push_back(times.size() - 1);
		std::cout << "\nNew chunk during recording, samples are now " << times.size() << std::endl;
	}

	//push values to corresponding vectors
	times.push_back(gaze_point->timestamp_us);
	x.push_back(gaze_point->position_xy[0]);
	y.push_back(gaze_point->position_xy[1]);

	if (times.size() >= n_training_samples) {

		chunks.push_back(times.size() - 1);
		chunks.insert(chunks.begin(), 0);

		if (get_longest_chunk(chunks,false) < min_chunk_size) {
			server_instance->sendall("{\"type\":\"TRAINING_FAILED\"}");
			reset_data();
			set_state(0);
			std::cout << "training failed" << std::endl;
			return;
		}

		set_state(2);
		perform_training(server_instance);
	}
}

void handle_new_classification_data(tobii_gaze_point_t const* gaze_point, broadcast_server* server_instance) {

	if (reset_timer) {
		reset_data();
		reset_timer = false;
	}

	//push values to corresponding vectors
	times.push_back(gaze_point->timestamp_us);
	x.push_back(gaze_point->position_xy[0]);
	y.push_back(gaze_point->position_xy[1]);

	if (times.size() >= hz * trained_fixation_time + 5) {

		ArrayX2d eye(times.size(), 2);
		ArrayXd t(times.size());
		ArrayX2d xy(times.size(), 2);

		prepare_data(eye, t, xy);

		if (custom_transition) {
			std::tuple<Segmentation<Segment<Nslr2d::Vector>>, std::vector<unsigned int>, std::vector<unsigned int>> result = classify_gaze(t, eye, 0.14, false, get_transitions_model());
			analyze_fixations(result, xy, server_instance);
		}
		else {
			std::tuple<Segmentation<Segment<Nslr2d::Vector>>, std::vector<unsigned int>, std::vector<unsigned int>> result = classify_gaze(t, eye, 0.14, false);
			analyze_fixations(result, xy, server_instance);
		}
		reset_data();
	}
}

//process data from device, divide in windows of N samples
void process_data(tobii_gaze_point_t const* gaze_point, broadcast_server* server_instance) {

	if (get_state() == 0) {
		handle_new_classification_data(gaze_point, server_instance);
	}
	else if (get_state() == 1) {
		handle_new_training_data(gaze_point, server_instance);
	}

}