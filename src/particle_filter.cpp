/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 25;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	default_random_engine gen;

	// create particles
	for (int i = 0; i < num_particles; ++i) {
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;
		particles.push_back(p);
		weights.push_back(p.weight);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// sensor noise generators
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);
	default_random_engine gen;

	for (int i = 0; i < num_particles; ++i) {

		if (fabs(yaw_rate) < 0.0001) {
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}
		else {
			particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
			particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}

		// add sensor noise
		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	// loop over observation
	for (int i = 0; i < observations.size(); ++i) {

		LandmarkObs obs = observations[i];

		// initialize minimum distance to max double
		double min_dist = numeric_limits<double>::max();

		// loop over landmarks, find one closes to current observation
		for (int j = 0; j < predicted.size(); ++j) {

			LandmarkObs landmark = predicted[j];

			double distance = dist(obs.x, obs.y, landmark.x, landmark.y);

			if (distance < min_dist) {
				// save this landmark as "closest".
				min_dist = distance;
				//observations[i].id = landmark.id;
				observations[i].id = j; // using index instead of id to avoid lookup loop.
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// loop over particles
	for (int i = 0; i < num_particles; ++i) {

		// get the particle x, y coordinates
		double particle_x = particles[i].x;
		double particle_y = particles[i].y;
		double particle_theta = particles[i].theta;

		vector<LandmarkObs> in_range_landmarks;

		// find landmarks within sensor range of current particle
		for (int j = 0; j < map_landmarks.landmark_list.size(); ++j) {

			float landmark_x = map_landmarks.landmark_list[j].x_f;
			float landmark_y = map_landmarks.landmark_list[j].y_f;
			int landmark_id  = map_landmarks.landmark_list[j].id_i;

			// using axis distance to do initial filtering of landmarks for performance reasons.
			if (fabs(landmark_x - particle_x) <= sensor_range || fabs(landmark_y - particle_y) <= sensor_range) {
				in_range_landmarks.push_back(LandmarkObs{ landmark_id, landmark_x, landmark_y });
			}
		}

		// transform observations to map coordinates
		vector<LandmarkObs> transformed_observations;
		for (int j = 0; j < observations.size(); ++j) {
			double obs_x = observations[j].x;
			double obs_y = observations[j].y;

			double trans_x = cos(particle_theta) * obs_x - sin(particle_theta) * obs_y + particle_x;
			double trans_y = sin(particle_theta) * obs_x + cos(particle_theta) * obs_y + particle_y;
			transformed_observations.push_back(LandmarkObs{ observations[j].id, trans_x, trans_y });
		}

		dataAssociation(in_range_landmarks, transformed_observations);


		// update weights
		particles[i].weight = 1.0;
		for (int j = 0; j < transformed_observations.size(); ++j) {

			double obs_x = transformed_observations[j].x;
			double obs_y = transformed_observations[j].y;

			int landmark_id = transformed_observations[j].id;

			// lookup the observation's landmark
			double landmark_x, landmark_y;
//			for (int k = 0; k < in_range_landmarks.size(); ++k) {
//				if (in_range_landmarks[k].id == landmark_id) {
//					landmark_x = in_range_landmarks[k].x;
//					landmark_y = in_range_landmarks[k].y;
//					break;
//				}
//			}
			// returning index instead of id to get rid of lookup loop.
			landmark_x = in_range_landmarks[landmark_id].x;
			landmark_y = in_range_landmarks[landmark_id].y;

			// calculate weight for this observation with multivariate Gaussian
			double std_x = std_landmark[0];
			double std_y = std_landmark[1];
			double obs_w = ( 1/(2*M_PI*std_x*std_y)) * exp( -( pow(landmark_x-obs_x,2)/(2*pow(std_x, 2)) + (pow(landmark_y-obs_y,2)/(2*pow(std_y, 2))) ) );

			// product of this obersvation weight with total observations weight
			particles[i].weight *= obs_w;
		}
		weights[i] = particles[i].weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	vector<Particle> new_particles;

	// get max weight
	double max_weight = *max_element(weights.begin(), weights.end());

	discrete_distribution<int> dist_ind(weights.begin(), weights.end());
	uniform_real_distribution<double> dist_beta(0.0, 2 * max_weight);
	default_random_engine gen;

	double beta = 0.0;
	int index = dist_ind(gen);
	for (int i = 0; i < num_particles; ++i) {
		beta += dist_beta(gen);
		while (weights[index] < beta) {
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
		new_particles.push_back(particles[index]);
	}

	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
