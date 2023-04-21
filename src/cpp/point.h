#ifndef POINT_H
#define POINT_H

#include <vector>
#include <stdexcept>

class Point {
public:
    Point(const int _label, const std::vector<double> vd, const int _id = -1): label{_label}, id{_id}, coords{vd}, cluster{-1} {
        if (dimensionality == 0) {
            dimensionality = coords.size();
        } else if (dimensionality != coords.size()) {
            throw std::invalid_argument("Invalid dimensionality.");
        }
    };
    Point() {};
    size_t size() const {return coords.size();}
    bool empty() const {return coords.empty();}
    int label;
    int id;
    int cluster;
    std::vector<double> coords;
    static size_t dimensionality;

    bool operator<(const Point& other) const {
        size_t dim = std::min(coords.size(), other.coords.size());
        for (size_t i = 0; i < dim; i++) {
            if (coords[i] < other.coords[i]) {
                return true;
            } else if (coords[i] > other.coords[i]) {
                return false;
            }
        }
        return coords.size() < other.coords.size();
    }
};
#endif