#include "stdafx.h"
#include <iostream>
#include <set>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>
#include "websocketpp/config/asio_no_tls.hpp"
#include "websocketpp/server.hpp"
#include "state_controller.hpp"
//#include "nlohmann/json.hpp"

//#include "websocketpp/common/thread.hpp"

typedef websocketpp::server<websocketpp::config::asio> server;

using websocketpp::connection_hdl;
using websocketpp::lib::placeholders::_1;
using websocketpp::lib::placeholders::_2;
using websocketpp::lib::bind;

using websocketpp::lib::thread;
using websocketpp::lib::mutex;
using websocketpp::lib::lock_guard;
using websocketpp::lib::unique_lock;
using websocketpp::lib::condition_variable;

//using json = nlohmann::json;

/* on_open insert connection_hdl into channel
* on_close remove connection_hdl from channel
* on_message queue send to all channels
*/

enum action_type {
	SUBSCRIBE,
	UNSUBSCRIBE,
	MESSAGE
};

struct action {
	action(action_type t, connection_hdl h) : type(t), hdl(h) {}
	action(action_type t, connection_hdl h, server::message_ptr m)
		: type(t), hdl(h), msg(m) {}

	action_type type;
	websocketpp::connection_hdl hdl;
	server::message_ptr msg;
};

class broadcast_server {
public:
	broadcast_server() {
		// Initialize Asio Transport
		m_server.init_asio();

		// Register handler callbacks
		m_server.set_open_handler(bind(&broadcast_server::on_open, this, ::_1));
		m_server.set_close_handler(bind(&broadcast_server::on_close, this, ::_1));
		m_server.set_message_handler(bind(&broadcast_server::on_message, this, ::_1, ::_2));
		m_server.set_access_channels(websocketpp::log::alevel::none);
	}

	void run(uint16_t port) {
		// listen on specified port
		m_server.listen(port);

		// Start the server accept loop
		m_server.start_accept();

		// Start the ASIO io_service run loop
		try {
			m_server.run();
		}
		catch (const std::exception & e) {
			std::cout << e.what() << std::endl;
		}
	}

	void on_open(connection_hdl hdl) {
		{
			lock_guard<mutex> guard(m_action_lock);
			//std::cout << "on_open" << std::endl;
			m_actions.push(action(SUBSCRIBE, hdl));
		}
		m_action_cond.notify_one();
	}

	void on_close(connection_hdl hdl) {
		{
			lock_guard<mutex> guard(m_action_lock);
			std::cout << "\nconnection closed" << std::endl;
			m_actions.push(action(UNSUBSCRIBE, hdl));
		}
		m_action_cond.notify_one();
	}

	void on_message(connection_hdl hdl, server::message_ptr msg) {
		// queue message up for sending by processing thread
		std::string ev = msg->get_payload();
		std::cout << "\n" + ev << std::endl;
		auto j = json::parse(ev);

		if (j["type"] == "START_TRAINING") {
			set_state(1);
			sendall("{\"type\":\"TRAINING_STARTED\"}");
			std::cout << "\nstart training" << std::endl;
		}

		if (j["type"] == "GET_PRO") {
			sendall("{\"type\":\"pro_driver\",\"data\":" + std::to_string(pro_driver) + "}");
		}

		else if (j["type"] == "MANUAL_CLASSIFIER") {
			set_state(3);
			std::cout << "\nuntrained classifier" << std::endl;
		}

		else if (j["type"] == "TRAINED_CLASSIFIER") {
			set_state(0);
			std::cout << "\ntrained classifier" << std::endl;
		}

		else if (j["type"] == "INTERRUPT_TRAINING") {
			set_state(0);
			std::cout << "\ntrained classifier" << std::endl;
		}

		else if (j["type"] == "TRAINING_VIDEO_END") {
			if(get_state() == 1) training_video_end = true;
			std::cout << "\ntraining video end" << std::endl;
		}

		else if (j["type"] == "FIXATION_TIME") {
			untrained_fixation_time = j["data"].get<int>();
			std::cout << "\nnew fixation time: " << untrained_fixation_time << std::endl;
		}

		else if (j["type"] == "FIXATION_RADIUS") {
			untrained_fixation_radius = j["data"].get<double>();
			std::cout << "\nnew fixation radius: " << untrained_fixation_radius << std::endl;
		}

		else if (j["type"] == "RESET_TIMER") {
			set_reset_timer(true);
			std::cout << "\nreset timer" << std::endl;
		}

		else if (j["type"] == "SET_TRAINED_PARAMS") {
			if (!j["data"]["transition_matrix"].is_null()) {
				set_transitions_model(j["data"]["transition_matrix"].get<std::string>());
				custom_transition = true;
			}
			if (!j["data"]["trained_fixation_time"].is_null()) {
				set_trained_fixation_time(j["data"]["trained_fixation_time"].get<double>());
			}
			set_state(0);
			std::cout << "new trained params: " << j["data"];
		}

		else if (j["type"] == "SET_UNTRAINED_PARAMS") {
			set_untrained_fixation_radius(j["data"]["fixation_radius"].get<double>());
			set_untrained_fixation_time(j["data"]["fixation_time"].get<double>());
			set_state(3);
			std::cout << "new untrained params: " << j["data"];
		}
	}


	void sendall(std::string msg) {
		con_list::iterator it;
		try
		{
			for (it = m_connections.begin(); it != m_connections.end(); ++it) {

				m_server.send(*it, msg, websocketpp::frame::opcode::text);

			}
		}
		catch (std::exception& e) {
			std::cout << "exception: " << e.what() << std::endl;
		}
	}

	void process_messages() {
		while (1) {
			unique_lock<mutex> lock(m_action_lock);

			while (m_actions.empty()) {
				m_action_cond.wait(lock);
			}

			action a = m_actions.front();
			m_actions.pop();

			lock.unlock();

			if (a.type == SUBSCRIBE) {
				lock_guard<mutex> guard(m_connection_lock);
				m_connections.insert(a.hdl);
			}
			else if (a.type == UNSUBSCRIBE) {
				lock_guard<mutex> guard(m_connection_lock);
				m_connections.erase(a.hdl);
			}
			else if (a.type == MESSAGE) {
				lock_guard<mutex> guard(m_connection_lock);

				con_list::iterator it;
				for (it = m_connections.begin(); it != m_connections.end(); ++it) {
					m_server.send(*it, a.msg);
				}
			}
			else {
				// undefined.
			}
		}
	}
private:
	typedef std::set<connection_hdl, std::owner_less<connection_hdl> > con_list;

	server m_server;
	con_list m_connections;
	std::queue<action> m_actions;

	mutex m_action_lock;
	mutex m_connection_lock;
	condition_variable m_action_cond;
};
