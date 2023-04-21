#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <memory>
#include <limits>
#include "point.h"
#include "kdtree.h"

class Point;

size_t Point::dimensionality = 0;

std::vector<Point> normalize(const std::vector<Point>& points) {
    std::vector<Point> result(points.size());
    if (points.empty()) {
        return result;
    }

    const int num_dimensions = points[0].size();

    // Find the maximum and minimum values for each dimension
    std::vector<double> minvd(num_dimensions, std::numeric_limits<double>::min());
    std::vector<double> maxvd(num_dimensions, std::numeric_limits<double>::max());
    Point max_values(-1, minvd);
    Point min_values(-1, maxvd);
    for (const auto& point : points) {
        for (int i = 0; i < num_dimensions; i++) {
            max_values.coords[i] = std::max(max_values.coords[i], point.coords[i]);
            min_values.coords[i] = std::min(min_values.coords[i], point.coords[i]);
        }
    }

    // Normalize the data points
    for (int i = 0; i < points.size(); i++) {
        result[i].coords.resize(num_dimensions);
        for (int j = 0; j < num_dimensions; j++) {
            result[i].coords[j] = (points[i].coords[j] - min_values.coords[j]) / (max_values.coords[j] - min_values.coords[j]);
        }
    }

    return result;
}

double euclideanDistance(Point p, Point q) {
    double distance = 0.0;
    for (int i = 0; i < p.coords.size(); i++) {
        distance += pow(p.coords[i] - q.coords[i], 2);
    }
    return sqrt(distance);
}

std::vector<int> rangeQuery(std::vector<Point>& points, int p, double eps) {
    std::vector<int> neighbors;
    for (int i = 0; i < points.size(); i++) {
        if (euclideanDistance(points[p], points[i]) <= eps) {
            neighbors.push_back(i);
        }
    }
    return neighbors;
}


std::vector<int> naive_dbscan(std::vector<Point>& points, double eps, int minPts) {
    std::vector<int> visited(points.size(), 0);
    std::vector<int> cluster(points.size(), -1);
    int clusterIdx = 0;

    for (int i = 0; i < points.size(); i++) {
        if (visited[i] == 1) continue;

        visited[i] = 1;
        std::vector<int> neighbors = rangeQuery(points, i, eps);
        if (neighbors.size() < minPts) {
            cluster[i] = -1;
        } else {
            cluster[i] = clusterIdx;
            for (int j = 0; j < neighbors.size(); j++) {
                int idx = neighbors[j];
                if (visited[idx] == 0) {
                    visited[idx] = 1;
                    std::vector<int> subNeighbors = rangeQuery(points, idx, eps);
                    if (subNeighbors.size() >= minPts) {
                        neighbors.insert(neighbors.end(), subNeighbors.begin(), subNeighbors.end());
                    }
                }
                if (cluster[idx] == -1) {
                    cluster[idx] = clusterIdx;
                }
            }
            clusterIdx++;
        }
    }

    std::cout << "Clusters: " << clusterIdx << std::endl;
    for (int i = 0; i < points.size(); i++) {
        std::cout << "(";
        for (int j = 0; j < points[i].coords.size(); j++) {
            std::cout << points[i].coords[j];
            if (j < points[i].coords.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "): " << cluster[i] << std::endl;
    }

    return cluster;
}

std::vector<Point> parse(std::string filename) 
{
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        throw std::runtime_error("Error opening file");
    }

    std::string line{};
    std::vector<Point> vupp{};
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::vector<double> values{};
        std::string token{};
        while (std::getline(iss, token, ',')) {
            double value{0};
            try {
                value = std::stod(token);
            } catch (const std::exception& e) {
                std::cerr << "Error parsing token " << token << ": " << e.what() << "\n";
                continue;
            }
            values.push_back(value);
        }
        int label = static_cast<int>(values.back());
        values.pop_back();
        Point p = Point(label, values);
        vupp.push_back(p);
    }

    infile.close();
    return vupp;
}

int main(int argc, char* argv[]) {

    double eps{0.5};
    int minPts{5};
    if (argc == 3)
    {
        eps = std::stof(argv[1]);
        minPts = std::stoi(argv[2]);
    }
    std::cout << "eps: " << eps << "  minPts: " << minPts << std::endl;


    auto points(normalize(parse("iris_dataset.csv")));
    auto clusters(naive_dbscan(points, eps, minPts));
    return 0;
}
