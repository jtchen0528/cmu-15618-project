#ifndef POINT_H
#define POINT_H

#include <vector>
#include <stdexcept>

class Point {
public:
    Point(const int _label, const std::vector<double> vd): label{_label}, coords{vd} {
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
    std::vector<double> coords;
    static size_t dimensionality;
};
#endif