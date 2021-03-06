#include "stdafx.h"
#include <tobii/tobii.h>
#include <tobii/tobii_streams.h>
#include <stdio.h>
#include <assert.h>
#include <string>
#include <chrono>
#include <iostream>
#include "screensize.hpp"
#include "websocket.hpp"
#include "state_controller.hpp"
#include "trained_classification.hpp"
#include "untrained_classification.hpp"
#include "curl_post.hpp"

using namespace std::chrono;


clock_t gaze_timer;
clock_t position_timer;
clock_t gaze_start;

const float gaze_ms = 0.08;
const float pos_ms = 0.3;

broadcast_server server_instance;


//get gaze points from device
void gaze_point_callback(tobii_gaze_point_t const* gaze_point, void* user_data)
{
	gaze_start = clock();

	if (gaze_point->validity == TOBII_VALIDITY_VALID) {

		if ((clock() - gaze_timer) / (double)CLOCKS_PER_SEC >= gaze_ms) {
			gaze_timer = clock();
			server_instance.sendall("{\"type\":\"cursor\",\"data\":[" + std::to_string(gaze_point->position_xy[0]) + "," + std::to_string(gaze_point->position_xy[1]) + "]}");
			//std::cout << "sending gaze: "<< gaze_point->position_xy[0] << " - " << gaze_point->position_xy[1] << " - "<< gaze_point->timestamp_us;
		}

		if (state < 3) process_data(gaze_point, &server_instance);
		else untrained_classify(gaze_point, &server_instance);
	}

	//std::cout << "\nelapsed: \n" << (clock() - gaze_start) / (double)CLOCKS_PER_SEC;

}

//update distance from screen, send y-axis of eyes position
void gaze_origin_callback(tobii_gaze_origin_t const* origin_point, void* user_data)
{
	if (origin_point->left_validity == TOBII_VALIDITY_VALID || origin_point->right_validity == TOBII_VALIDITY_VALID) {

		float ypos = -2;

		if ((clock() - position_timer) / (double)CLOCKS_PER_SEC >= pos_ms) {
			ypos = -1;
		}

		if (origin_point->left_validity == TOBII_VALIDITY_VALID && origin_point->right_validity != TOBII_VALIDITY_VALID) {

			dist = origin_point->left_xyz[2] / 10;
			if (ypos == -1) {
				ypos = (origin_point->left_xyz[1] / 10);
				ypos = (ypos + h / 2) / h;
			}
		}
		else if (origin_point->left_validity != TOBII_VALIDITY_VALID && origin_point->right_validity == TOBII_VALIDITY_VALID) {

			dist = origin_point->right_xyz[2] / 10;

			if (ypos == -1) {
				ypos = (origin_point->left_xyz[1] / 10);
				ypos = (ypos + h / 2) / h;
			}
			else if (ypos > 0) {
				float ypos_r = (origin_point->right_xyz[1] / 10);
				ypos_r = (ypos_r + h / 2) / h;
				ypos = (ypos + ypos_r) / 2;
			}
		}
		else {

			dist = ((origin_point->left_xyz[2] + origin_point->right_xyz[2]) / 2) / 10;

			if (ypos == -1) {

				float ypos_l = (origin_point->left_xyz[1] / 10);
				ypos_l = (ypos_l + h / 2) / h;

				float ypos_r = (origin_point->right_xyz[1] / 10);
				ypos_r = (ypos_r + h / 2) / h;

				ypos = (ypos_l + ypos_r) / 2;
			}
		}

		if (ypos > -2) {
			ypos = 1 - ypos;
			server_instance.sendall("{\"type\":\"position\",\"data\":{\"y\":" + std::to_string(ypos) + "}}");
			//std::cout << "sending ypos: " << ypos;
			position_timer = clock();
		}
	}
}


static void url_receiver(char const* url, void* user_data)
{
	char* buffer = (char*)user_data;
	if (*buffer != '\0') return; // only keep first value

	if (strlen(url) < 256)
		strcpy(buffer, url);
}


static auto reconnect(tobii_device_t* device)
{
	auto error = TOBII_ERROR_CONNECTION_FAILED;
	std::cout << "\nChecking for device reconnection" << std::endl;

	// Try reconnecting for 10 seconds before giving up
	while(error == TOBII_ERROR_CONNECTION_FAILED)
	{
		
		error = tobii_device_reconnect(device);
		if (error != TOBII_ERROR_CONNECTION_FAILED)
			{
				std::cout << "\nDevice reconnected" << std::endl;
				return error; 
			}
		std::this_thread::sleep_for(std::chrono::milliseconds(250));
	}

	return TOBII_ERROR_CONNECTION_FAILED;
}

int main(int argc, char* argv[])
{

	if (argc >= 2) {
		
		/*if (argv[1] == std::string("--recover")) {
			std::cout << "\nrecovering past state\n" << std::endl;
			bool shoud_reset = recover_state();
			if (shoud_reset) {
				server_instance.sendall("{\"type\":\"TRAINING_FAILED\"}");
			}
		}*/

		for (int i = 1; i < argc; ++i) {
			if (argv[i] == std::string("--recover")) {
				std::cout << "\nrecovering past state\n" << std::endl;
				bool shoud_reset = recover_state();
				if (shoud_reset) {
					server_instance.sendall("{\"type\":\"TRAINING_FAILED\"}");
				}
			}
			else if (argv[i] == std::string("--pro")) {
				pro_driver = true;
			}
		}
	}

	//postCall();
	std::pair <short, short> dims = getScreenPhysicalSize();
	w = dims.first / 10.0;
	h = dims.second / 10.0;

	tobii_api_t* api;
	tobii_error_t error = tobii_api_create(&api, NULL, NULL);
	assert(error == TOBII_ERROR_NO_ERROR);

	char url[256] = { 0 };
	error = tobii_enumerate_local_device_urls(api, url_receiver, url);
	assert(error == TOBII_ERROR_NO_ERROR && *url != '\0');

	tobii_device_t* device;
	error = tobii_device_create(api, url, &device);
	assert(error == TOBII_ERROR_NO_ERROR);

	tobii_supported_t supported;
	error = tobii_capability_supported(device,TOBII_CAPABILITY_CALIBRATION_3D, &supported);
	if (supported == TOBII_SUPPORTED) {
		std::cout << "can send data!" << std::endl;
	}

	error = tobii_gaze_point_subscribe(device, gaze_point_callback, 0);
	assert(error == TOBII_ERROR_NO_ERROR);

	error = tobii_gaze_origin_subscribe(device, gaze_origin_callback, 0);
	assert(error == TOBII_ERROR_NO_ERROR);
	

	gaze_timer = clock();
	position_timer = clock();

	// Create atomic used for inter thread communication
	std::atomic<bool> exit_thread(false);
	// Start the background processing thread before subscribing to data
	std::thread thread(
		[&exit_thread, device]()
	{
		while (!exit_thread)
		{
			// Do a timed blocking wait for new gaze data, will time out after some hundred milliseconds
			auto error = tobii_wait_for_callbacks(NULL, 1, &device);

			if (error == TOBII_ERROR_TIMED_OUT) continue; // If timed out, redo the wait for callbacks call

			if (error == TOBII_ERROR_CONNECTION_FAILED)
			{
				// Block here while attempting reconnect, if it fails, exit the thread
				error = reconnect(device);
				if (error != TOBII_ERROR_NO_ERROR)
				{
					std::cerr << "Connection was lost and reconnection failed." << std::endl;
					return;
				}
				continue;
			}
			else if (error != TOBII_ERROR_NO_ERROR)
			{
				std::cerr << "tobii_wait_for_callbacks failed: " << tobii_error_message(error) << "." << std::endl;
				return;
			}
			// Calling this function will execute the subscription callback functions
			error = tobii_device_process_callbacks(device);

			if (error == TOBII_ERROR_CONNECTION_FAILED)
			{
				// Block here while attempting reconnect, if it fails, exit the thread
				error = reconnect(device);
				if (error != TOBII_ERROR_NO_ERROR)
				{
					std::cerr << "Connection was lost and reconnection failed." << std::endl;
					return;
				}
				continue;
			}
			else if (error != TOBII_ERROR_NO_ERROR)
			{
				std::cerr << "tobii_device_process_callbacks failed: " << tobii_error_message(error) << "." << std::endl;
				return;
			}
		}
	});


	// Start a thread to run the processing loop
	std::thread t(bind(&broadcast_server::process_messages, &server_instance));

	// Run the asio loop with the main thread
	server_instance.run(4000);

	t.join();
	thread.join();

	error = tobii_gaze_point_unsubscribe(device);
	assert(error == TOBII_ERROR_NO_ERROR);

	error = tobii_device_destroy(device);
	assert(error == TOBII_ERROR_NO_ERROR);

	error = tobii_api_destroy(api);
	assert(error == TOBII_ERROR_NO_ERROR);

	return 0;
}

