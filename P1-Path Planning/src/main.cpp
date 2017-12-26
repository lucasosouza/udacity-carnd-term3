#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "json.hpp"
#include "spline.h"

using namespace std;

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("}");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

double distance(double x1, double y1, double x2, double y2)
{
	return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}
int ClosestWaypoint(double x, double y, vector<double> maps_x, vector<double> maps_y)
{

	double closestLen = 100000; //large number
	int closestWaypoint = 0;

	for(int i = 0; i < maps_x.size(); i++)
	{
		double map_x = maps_x[i];
		double map_y = maps_y[i];
		double dist = distance(x,y,map_x,map_y);
		if(dist < closestLen)
		{
			closestLen = dist;
			closestWaypoint = i;
		}

	}

	return closestWaypoint;

}

int NextWaypoint(double x, double y, double theta, vector<double> maps_x, vector<double> maps_y)
{

	int closestWaypoint = ClosestWaypoint(x,y,maps_x,maps_y);

	double map_x = maps_x[closestWaypoint];
	double map_y = maps_y[closestWaypoint];

	double heading = atan2( (map_y-y),(map_x-x) );

	double angle = abs(theta-heading);

	if(angle > pi()/4)
	{
		closestWaypoint++;
	}

	return closestWaypoint;

}

// Transform from Cartesian x,y coordinates to Frenet s,d coordinates
vector<double> getFrenet(double x, double y, double theta, vector<double> maps_x, vector<double> maps_y)
{
	int next_wp = NextWaypoint(x,y, theta, maps_x,maps_y);

	int prev_wp;
	prev_wp = next_wp-1;
	if(next_wp == 0)
	{
		prev_wp  = maps_x.size()-1;
	}

	double n_x = maps_x[next_wp]-maps_x[prev_wp];
	double n_y = maps_y[next_wp]-maps_y[prev_wp];
	double x_x = x - maps_x[prev_wp];
	double x_y = y - maps_y[prev_wp];

	// find the projection of x onto n
	double proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y);
	double proj_x = proj_norm*n_x;
	double proj_y = proj_norm*n_y;

	double frenet_d = distance(x_x,x_y,proj_x,proj_y);

	//see if d value is positive or negative by comparing it to a center point

	double center_x = 1000-maps_x[prev_wp];
	double center_y = 2000-maps_y[prev_wp];
	double centerToPos = distance(center_x,center_y,x_x,x_y);
	double centerToRef = distance(center_x,center_y,proj_x,proj_y);

	if(centerToPos <= centerToRef)
	{
		frenet_d *= -1;
	}

	// calculate s value
	double frenet_s = 0;
	for(int i = 0; i < prev_wp; i++)
	{
		frenet_s += distance(maps_x[i],maps_y[i],maps_x[i+1],maps_y[i+1]);
	}

	frenet_s += distance(0,0,proj_x,proj_y);

	return {frenet_s,frenet_d};

}

// Transform from Frenet s,d coordinates to Cartesian x,y
vector<double> getXY(double s, double d, vector<double> maps_s, vector<double> maps_x, vector<double> maps_y)
{
	int prev_wp = -1;

	while(s > maps_s[prev_wp+1] && (prev_wp < (int)(maps_s.size()-1) ))
	{
		prev_wp++;
	}

	int wp2 = (prev_wp+1)%maps_x.size();

	double heading = atan2((maps_y[wp2]-maps_y[prev_wp]),(maps_x[wp2]-maps_x[prev_wp]));
	// the x,y,s along the segment
	double seg_s = (s-maps_s[prev_wp]);

	double seg_x = maps_x[prev_wp]+seg_s*cos(heading);
	double seg_y = maps_y[prev_wp]+seg_s*sin(heading);

	double perp_heading = heading-pi()/2;

	double x = seg_x + d*cos(perp_heading);
	double y = seg_y + d*sin(perp_heading);

	return {x,y};

}

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // define max, reference velocity, and increase in velocity per step (5m/s)
  double max_speed = 49.5;
  double ref_speed = 0;
  double inc_speed = .224;  
  double avc_speed = max_speed; // avoid collision speed

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  ifstream in_map_(map_file_.c_str(), ifstream::in);

  string line;
  while (getline(in_map_, line)) {
  	istringstream iss(line);
  	double x;
  	double y;
  	float s;
  	float d_x;
  	float d_y;
  	iss >> x;
  	iss >> y;
  	iss >> s;
  	iss >> d_x;
  	iss >> d_y;
  	map_waypoints_x.push_back(x);
  	map_waypoints_y.push_back(y);
  	map_waypoints_s.push_back(s);
  	map_waypoints_dx.push_back(d_x);
  	map_waypoints_dy.push_back(d_y);
  }

  h.onMessage([&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,&map_waypoints_dx,&map_waypoints_dy,&ref_speed,&max_speed,&inc_speed, &avc_speed](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    //auto sdata = string(data).substr(0, length);
    //cout << sdata << endl;
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);
        
        string event = j[0].get<string>();
        
        if (event == "telemetry") {
          // j[1] is the data JSON object
          
        	// Main car's localization Data
          	double car_x = j[1]["x"];
          	double car_y = j[1]["y"];
          	double car_s = j[1]["s"];
          	double car_d = j[1]["d"];
          	double car_yaw = j[1]["yaw"];
          	double car_speed = j[1]["speed"];

          	// Previous path data given to the Planner
          	auto previous_path_x = j[1]["previous_path_x"];
          	auto previous_path_y = j[1]["previous_path_y"];
          	// Previous path's end s and d values 
          	double end_path_s = j[1]["end_path_s"];
          	double end_path_d = j[1]["end_path_d"];

          	// Sensor Fusion Data, a list of all other cars on the same side of the road.
          	auto sensor_fusion = j[1]["sensor_fusion"];

          	json msgJson;

            //////////////////////   BEHAVIOR PLANNING   ///////////////////////

            // change s to previous path s, smooth
            int previous_path_size = previous_path_x.size();
            if (previous_path_size>0) {
              car_s = end_path_s;
            }

            // calculate d. check if route of collision with another car, and decide if stay on lane or change
            // define vector to keep which lanes are available
            vector<double> lane_speed; 
            vector<double> lane_pos; 
            for (int i=0;i<3;i++) { 
              lane_speed.push_back(0);
              lane_pos.push_back(0); 
            }

            // define default lane
            int car_lane = floor(car_d/4);

            // iterate through cars identified in sensor fusion and mark if lane is occupied
            int other_car_lane;
            double other_car_d, other_car_s, other_car_vx, other_car_vy, other_car_speed;
            for (int i=0; i < sensor_fusion.size(); i++) {
              other_car_s = sensor_fusion[i][5];
              other_car_d = sensor_fusion[i][6];
              other_car_vx = sensor_fusion[i][3];
              other_car_vy = sensor_fusion[i][4];
              other_car_speed = sqrt(other_car_vx*other_car_vx+other_car_vy*other_car_vy);
              other_car_lane = floor(other_car_d/4);

              // if using previous points, project car s based on its speed
              other_car_s += double(previous_path_size) * 0.02 * other_car_speed;

              // check if on a route of collision
              // should be between 0 and 30 meters in front 
              // speed should be greater than current speed of ego car
              if ((other_car_s > (car_s-10)) and ((other_car_s-car_s) < 20)) //and (other_car_speed < ref_speed))
              {
                  //std::cout << "lane occupied" << std::endl;
                  // get the closest car in other lane
                  if ((other_car_s < lane_pos[other_car_lane]) or (lane_pos[other_car_lane]==0)) {
                    lane_speed[other_car_lane] = other_car_speed;
                    lane_pos[other_car_lane] = other_car_s;
                  }
              }
            }

            bool prep_change_left = false;
            bool prep_change_right = false;
            bool keep_lane = false;
            int left_most_lane = 0;
            int right_most_lane = 2;

            // behavior rule to change lane if required
            if((lane_speed[car_lane] > 0) and (lane_speed[car_lane] < car_speed)) {
              if (car_lane==left_most_lane) {
                prep_change_right = true;
              } else if (car_lane==right_most_lane) {
                prep_change_left = true;
              } else {
                // if not on edge lanes, could move either left or right
                prep_change_left = true;
                prep_change_right = true;
              }
            }

            double left_lane_car_s, right_lane_car_s;
            double safety_margin = 20;
            max_speed = 49.5;
            // apply change lane left and change lane right behaviors
            // first check left, priority, then check right
            if (prep_change_left) {
              //std::cout << "prep change left" << std::endl;
              // check where car in left lane is
              left_lane_car_s = lane_pos[car_lane-1];
              // if it is safe to change lane
              if (left_lane_car_s < (car_s-safety_margin) or left_lane_car_s > (car_s+safety_margin)) {
                car_lane -= 1;
                // if already prepping to change lane left, set change right to false
                prep_change_right = false;
              } else {
                //std::cout << "prep change left rejected" << std::endl;
                // if can't change lane, keep on it
                keep_lane = true;
              }
            } 
            if (prep_change_right) {
              //std::cout << "prep change right" << std::endl;
              // check where car in right lane is
              right_lane_car_s = lane_pos[car_lane+1];
              // if it is safe to change lane
              if (right_lane_car_s < (car_s-safety_margin) or right_lane_car_s > (car_s+safety_margin)) {
                car_lane += 1;
              } else {
                //std::cout << "prep change right rejected" << std::endl;
                keep_lane = true;
              }              
            }
            
            // apply keep lane behavior
            if (keep_lane) {
              // set avc speed to the car in front of you (if it is greater than current avc speed)
              if (lane_speed[car_lane] < avc_speed) {
                avc_speed = lane_speed[car_lane];
              }
            } else {
              // if not keep lane, reset to max speed
              avc_speed = max_speed;
            }

            // incremental change in speed
            if (ref_speed > avc_speed) {
              ref_speed -= inc_speed;
            } else if (ref_speed < max_speed) {
              ref_speed += inc_speed;
            }

            //////////////////////   TRAJECTORY GENERATION - BASE POINTS  ///////////////////////

            // initialize vectors with baseline points, to be used in interpolatin
            vector<double> ptsx;
            vector<double> ptsy;

            // define ref points
            double ref_x = car_x;
            double ref_y = car_y;
            double ref_yaw = car_yaw;
            double prev_car_x, prev_car_y;

            // add previous path last point to guarantee smooth interpolation
            if (previous_path_size <  2) {
              // calculate previous points
              prev_car_x = ref_x - cos(ref_yaw);
              prev_car_y = ref_y - sin(ref_yaw);
            } else {
              // redefine reference state to t-1
              ref_x = previous_path_x[previous_path_size -1];
              ref_y = previous_path_y[previous_path_size -1];
              // t-2
              prev_car_x = previous_path_x[previous_path_size - 2];
              prev_car_y = previous_path_y[previous_path_size - 2];
              // calculate new ref yaw, needed later when transforming coordinates
              ref_yaw = atan2(ref_y - prev_car_y, ref_x - prev_car_x);
            }

            // push t-2
            ptsx.push_back(prev_car_x);
            ptsy.push_back(prev_car_y);

            // push t-1
            ptsx.push_back(ref_x);
            ptsy.push_back(ref_y);

            // push 3 more points
            double next_d = 2 + 4*car_lane;
            double next_s;
            vector<double>map_coords(2);

            for (int i=1; i<4; i++) {
              // calculate s
              next_s = car_s + 30*i;
              // get map coordinates
              map_coords = getXY(next_s, next_d, map_waypoints_s, map_waypoints_x, map_waypoints_y);
              // push points to vector
              ptsx.push_back(map_coords[0]);
              ptsy.push_back(map_coords[1]);
            }

            // transform point from map coordinates to car referenced coordinates
            double shift_x, shift_y;
            for (int i=0; i < ptsx.size(); i++){
              // shift reference
              shift_x = ptsx[i]-ref_x;
              shift_y = ptsy[i]-ref_y;
              // update in pts vectors
              ptsx[i] = shift_x * cos(0-ref_yaw) - shift_y * sin(0-ref_yaw);
              ptsy[i] = shift_x * sin(0-ref_yaw) + shift_y * cos(0-ref_yaw);
            }

            // initialize spline
            tk::spline s;
            s.set_points(ptsx, ptsy);

            //////////////////////   TRAJECTORY GENERATION - INTERPOLATION   ///////////////////////

            // initialize vectors that will store final x and y vals to be sent to simulator
            vector<double> next_x_vals;
            vector<double> next_y_vals;

            // add all previous points to path - helps in the transition
            for (int i=0; i < previous_path_size; i++){
              next_x_vals.push_back(previous_path_x[i]);
              next_y_vals.push_back(previous_path_y[i]);
            }

            // fix a target x, and calculate target y and dist based on it
            double target_x = 30.0;
            double target_y = s(target_x);
            double target_dist = sqrt(target_x*target_x+target_y*target_y);
            // define x add ons, start at the origin
            double x_add_on = 0;
            double x_ref, y_ref, x_point, y_point, N;

            // interpolate using spline and generate additional points
            for(int i = 1; i <= 50-previous_path_size; i++) {

              // decide number of points based on speed and acceleratiom (defined by dist between points)
              N = target_dist/(.02*ref_speed/2.24);
              x_point = x_add_on + (target_x)/N;
              y_point = s(x_point);

              // update addon
              x_add_on = x_point;

              // define reference point (save x and y points prior to transformation)
              x_ref = x_point;
              y_ref = y_point;

              // rotate back to map coordinates
              x_point = x_ref * cos(ref_yaw) - y_ref*sin(ref_yaw);
              y_point = x_ref * sin(ref_yaw) + y_ref*cos(ref_yaw);

              // add reference
              x_point += ref_x;
              y_point += ref_y;

              // push to x and y vals vectors
              next_x_vals.push_back(x_point);
              next_y_vals.push_back(y_point);
            }


            //////////////////////   SIMULATOR COMM   ///////////////////////

          	// TODO: define a path made up of (x,y) points that the car will visit sequentially every .02 seconds
          	msgJson["next_x"] = next_x_vals;
          	msgJson["next_y"] = next_y_vals;

          	auto msg = "42[\"control\","+ msgJson.dump()+"]";

          	//this_thread::sleep_for(chrono::milliseconds(1000));
          	ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
          
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }

    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
















































































