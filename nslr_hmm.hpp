#pragma once
#include "stdafx.h"
#include <iostream>
#include <vector>
#include <math.h>  
#include <Eigen\Dense>
#include <mlpack/core.hpp>
#include <mlpack/methods/hmm/hmm.hpp>
#include <mlpack/methods/kmeans/kmeans.hpp>
#include "segmented_regression.hpp"

using namespace mlpack;
using namespace mlpack::hmm;
using namespace mlpack::kmeans;
using namespace mlpack::distribution;
using namespace Eigen;

//given gaussian distributions for observations model
GaussianDistribution g1("0.6039844795867605 -0.7788440631929878", "0.1651734722683456 0.0; 0.0 1.5875256060544993");
GaussianDistribution g2("2.3259276064858194 1.1333265634427712", "0.080879690559802 0.0; 0.0 2.0718979621084372");
GaussianDistribution g3("1.7511546389160744, -1.817487032170937", "0.0752678429860497 0.0; 0.0 1.356411391040218");
GaussianDistribution g4("0.8175021916433242, 0.3047120126632254", "0.13334607025750783 0.0; 0.0 2.5328705587328173");

//emissions distributions
GaussianDistribution e1("0.0063521 0.00520559 0.01847933 0.00456646", "0.00036145 0.0 0.0 0.0; 0.0 0.00044437 0.0 0.0; 0.0 0.0 0.00167195 0.0; 0.0 0.0 0.0 0.00029474");
GaussianDistribution e2("0.02982293 0.00233648 0.21245763 0.02946372", "0.00106009 0.0 0.0 0.0; 0.0 0.00052601 0.0 0.0; 0.0 0.0 0.01685202 0.0; 0.0 0.0 0.0 0.00103402");
GaussianDistribution e3("1.22658702e-01 3.91570498e-05 6.23102917e-03 1.11351014e-01", "0.00510384 0.0 0.0 0.0; 0.0 0.00018311 0.0 0.0; 0.0 0.0 0.00030294 0.0; 0.0 0.0 0.0 0.00527668");
GaussianDistribution e4("1.48159130e-04 1.66214541e-01 9.29178408e-03 1.44831436e-04", "0.00112389 0.0 0.0 0.0; 0.0 0.0113309 0.0 0.0; 0.0 0.0 0.00134187 0.0; 0.0 0.0 0.0 0.00112096");



HMM<GaussianDistribution> hmminst;
bool initialized = false;

auto segment_features(Segmentation<Segment<Nslr2d::Vector>> segmentation) {

	Vector2d prev_direction(0.0, 0.0);
	Segment<Nslr2d::Vector> last = segmentation.segments.back();
	ArrayXf outliers = ArrayXf::Zero(std::get<1>(last.i));
	int len = segmentation.segments.size();
	arma::mat feature(2, len);
	int nth = 0;

	for (auto & seg : segmentation.segments) {

		//duration
		double duration = std::get<1>(seg.t) - std::get<0>(seg.t);
		auto x1 = std::get<0>(seg.x);
		auto x2 = std::get<1>(seg.x);

		//speed
		Vector2d speed((x2[0] - x1[0]), (x2[1] - x1[1]));
		speed = speed / duration;

		//velocity
		double velocity = speed.norm();

		//direction
		Vector2d direction = speed / velocity;

		//cosangle
		double cosangle = direction.dot(prev_direction);
		cosangle *= (1 - 1e-6);
		cosangle = atanh(cosangle);

		//velocity = log10(velocity)
		velocity = log10(velocity);
		if (velocity < 1e-6) velocity = 1e-6;

		//add to feature vector and prepare for next segment
		feature.col(nth) = arma::vec({ velocity,cosangle });
		prev_direction = direction;
		nth++;
	}
	//std::cout << "\nfeatures: " << feature;
	return feature;
}

arma::mat transition_model() {

	//build initial transition matrix
	arma::mat transitions(4, 4);
	transitions.ones();
	transitions(2, 0) = 0;
	transitions(1, 2) = 0;
	transitions(2, 3) = 0;
	transitions(3, 0) = 0.5;
	transitions(0, 3) = 0.5;
	for (int i = 0; i < transitions.n_cols; i++) {
		transitions.col(i) = transitions.col(i) / sum(transitions.col(i));
	}

	return transitions;
}


void init(arma::mat samples, arma::mat transitions = transition_model()) {

	//initial probabilities
	arma::vec init = arma::ones(4);
	const arma::vec init2 = init / arma::sum(init);

	//kmeans for initial emission probability estimation
	size_t clusters = 4;

	// The assignments will be stored in this vector.
	arma::Row<size_t> assignments;

	// Initialize with the default arguments.
	KMeans<> k(1000);


	k.Cluster(samples, clusters, assignments);


	arma::mat samp1(4, samples.n_cols);
	arma::mat samp2(4, samples.n_cols);
	arma::mat samp3(4, samples.n_cols);
	arma::mat samp4(4, samples.n_cols);

	int sizes[4] = { 0,0,0,0 };

	for (int i = 0; i < assignments.n_cols; i++) {
		int j = assignments[i];
		switch (j) {
		case 0:
			samp1.col(sizes[j]) = samples.col(i);
			break;
		case 1:
			samp2.col(sizes[j]) = samples.col(i);
			break;
		case 2:
			samp3.col(sizes[j]) = samples.col(i);
			break;
		case 3:
			samp4.col(sizes[j]) = samples.col(i);
			break;
		}

		sizes[j]++;
	}

	std::cout << sizes[0] << " - " << sizes[1] << " - " << sizes[2] << " - " << sizes[3];

	samp1.resize(4, sizes[0]);
	samp2.resize(4, sizes[1]);
	samp3.resize(4, sizes[2]);
	samp4.resize(4, sizes[3]);

	e1.Train(samp1);
	e2.Train(samp2);
	e3.Train(samp3);
	e4.Train(samp4);

	//vector of emissions distributions
	std::vector<GaussianDistribution> emission;
	emission.push_back(e1);
	emission.push_back(e2);
	emission.push_back(e3);
	emission.push_back(e4);

	//initialize HMM
	hmminst = HMM<GaussianDistribution>(init2, transitions, emission, 1e-6);
	initialized = true;
}

const arma::mat & dataset_features(Timestamps ts, Points2d xs, double structural_error, bool optimize_noise) {


	//extract features
	Segmentation<Segment<Nslr2d::Vector>> res = fit_gaze(ts, xs, structural_error, optimize_noise);
	arma::mat feature = segment_features(res);


	//compute likelihoods
	arma::mat liks(4, feature.n_cols);

	for (int i = 0; i < feature.n_cols; i++) {
		liks(0, i) = g1.Probability(feature.col(i));
		liks(1, i) = g2.Probability(feature.col(i));
		liks(2, i) = g3.Probability(feature.col(i));
		liks(3, i) = g4.Probability(feature.col(i));
	}
	std::vector<arma::mat> liksvec = { liks };

	//init the HMM
	init(liks);

	//train the hmm - NEEDS A VECTOR WITH A MATRIX OF OBSERVATIONS!
	hmminst.Train(liksvec);

	//return the new transition matrix
	return hmminst.Transition();
}


const arma::mat & cumulative_dataset_features(Timestamps ts_arr, Points2d xs_arr, std::vector<int> chunks, double structural_error, bool optimize_noise) {


	std::vector<arma::mat> liksvec;
	int maxind = -1;
	int maxlen = -1;
	
	for (int i = 0; i < chunks.size()-1; i++) {

		int nextChunk = chunks[i + 1] - chunks[i];

		//check if current chunk is the longest
		if (nextChunk >= maxlen) {
			maxlen = nextChunk;
			maxind = i;
		}
		
		//extract features
		Segmentation<Segment<Nslr2d::Vector>> res = fit_gaze(ts_arr.block(chunks[i], 0, nextChunk, 1), xs_arr.block(chunks[i], 0, nextChunk, 2), structural_error, optimize_noise);
		arma::mat feature = segment_features(res);

		//compute likelihoods
		arma::mat liks(4, feature.n_cols);

		for (int i = 0; i < feature.n_cols; i++) {
			liks(0, i) = g1.Probability(feature.col(i));
			liks(1, i) = g2.Probability(feature.col(i));
			liks(2, i) = g3.Probability(feature.col(i));
			liks(3, i) = g4.Probability(feature.col(i));
		}

		liksvec.push_back(liks);
	}

	//init the HMM
	init(liksvec[maxind]);

	//train the hmm - NEEDS A VECTOR WITH A MATRIX OF OBSERVATIONS!
	hmminst.Train(liksvec);

	//return the new transition matrix
	return hmminst.Transition();
}

std::vector<unsigned int> viterbi(arma::mat transition_probs, arma::mat emission) {

	//initial probabilities
	arma::vec init = arma::ones(4);
	arma::vec initial_probs = init / arma::sum(init);

	transition_probs = arma::clamp(transition_probs, 1e-6, 1);
	transition_probs = log10(transition_probs);

	initial_probs = arma::clamp(initial_probs, 1e-6, 1);
	initial_probs = log10(initial_probs);

	arma::mat probs = arma::clamp(emission.col(0), 1e-6, 1);
	probs = log10(probs);

	probs = probs + initial_probs;

	std::vector <arma::ucolvec> states_stack;
	std::vector <unsigned int> states_seq;

	for (int i = 1; i < emission.n_cols; i++) {
		emission.col(i) = arma::normalise(emission.col(i));
		arma::mat trans_prob(4, 4);
		for (int j = 0; j < 4; j++) {
			trans_prob.row(j) = transition_probs.row(j) + probs.t();
		}

		arma::ucolvec most_likely = arma::index_max(trans_prob, 1);

		arma::mat new_probs = arma::clamp(emission.col(i), 1e-6, 1);
		new_probs = log10(new_probs);


		arma::uvec lin = arma::linspace<arma::uvec>(0, 3, 4);
		arma::vec sel_tran(lin.n_rows);

		for (int it = 0; it < lin.n_rows; it++) {
			sel_tran[it] = trans_prob(lin[it], most_likely[it]);
		}

		probs = new_probs + sel_tran;
		states_stack.push_back(most_likely);

	}

	arma::uvec last_prob = arma::index_max(probs);
	states_seq.push_back(last_prob[0]);

	while (states_stack.size() > 0) {
		arma::ucolvec most_likely = states_stack.back();
		states_stack.pop_back();
		states_seq.push_back(most_likely[states_seq.back()]);
	}

	std::reverse(states_seq.begin(), states_seq.end());


	for (unsigned int a : states_seq) {
		a = a + 1;
	}
	return states_seq;
}


std::vector<unsigned int> classify_segments(Segmentation<Segment<Nslr2d::Vector>> res, arma::mat transition) {

	arma::mat feature = segment_features(res);

	//compute likelihoods
	arma::mat liks(4, feature.n_cols);

	for (int i = 0; i < feature.n_cols; i++) {

		liks(0, i) = g1.Probability(feature.col(i));
		liks(1, i) = g2.Probability(feature.col(i));
		liks(2, i) = g3.Probability(feature.col(i));
		liks(3, i) = g4.Probability(feature.col(i));
	}

	std::vector<unsigned int> predictedClasses = viterbi(transition, liks);

	return predictedClasses;
}

std::tuple<Segmentation<Segment<Nslr2d::Vector>>, std::vector<unsigned int>, std::vector<unsigned int>> classify_gaze(Timestamps ts, Points2d xs, double structural_error, bool optimize_noise, arma::mat transition = transition_model()) {

	Segmentation<Segment<Nslr2d::Vector>> res = fit_gaze(ts, xs, structural_error, optimize_noise);

	std::vector<unsigned int> seg_classes = classify_segments(res, transition);
	std::vector<unsigned int> sample_classes(ts.rows(), -1);

	//arma::uvec sample_classes(ts.rows());

	for (int j = 0; j < ts.rows(); j++) {
		auto seg_i = res.segments[j].i;
		int start = std::get<0>(seg_i);
		int end = std::get<1>(seg_i);
		std::fill(sample_classes.begin() + start, sample_classes.begin() + start, seg_classes[j]);
	}

	return std::make_tuple(res, seg_classes, sample_classes);

}