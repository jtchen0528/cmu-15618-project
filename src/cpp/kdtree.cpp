#include "kdtree.h"
#include "point.h"
#include <algorithm>
#include <cmath>

KDtree::KDtree() : root(nullptr), leafsize(30) {}

void KDtree::buildTree(std::vector<Point> points) {
    buildTreeHelper(points, root, 0);
}

void KDtree::buildTreeHelper(std::vector<Point> points, std::unique_ptr<Node>& curr_node, size_t depth) {
    if (points.empty()) {
        return;
    }

    size_t num_points = points.size();
    size_t axis = depth % Point::dimensionality;
    size_t median_index = num_points / 2;
    std::nth_element(points.begin(), points.begin() + median_index, points.end(), [&](const Point& a, const Point& b) { return a.coords[axis] < b.coords[axis]; });

    curr_node = std::make_unique<Node>(Node{ points[median_index], std::vector<Point>(), nullptr, nullptr });
    curr_node->points = points;
    buildTreeHelper(std::vector<Point>(points.begin(), points.begin() + median_index), curr_node->left, depth + 1);
    buildTreeHelper(std::vector<Point>(points.begin() + median_index + 1, points.end()), curr_node->right, depth + 1);
}

std::vector<Point> KDtree::search(Point& target_point, double eps) {
    std::vector<Point> points;
    searchHelper(root.get(), target_point, eps, points);
    return points;
}

void KDtree::searchHelper(Node* curr_node, Point& target_point, double eps, std::vector<Point>& points) {
    if (curr_node == nullptr) {
        return;
    }

    double dist = distance(curr_node->point, target_point);
    if (dist <= eps) {
        points.insert(points.end(), curr_node->points.begin(), curr_node->points.end());
    }

    size_t axis = points.size() % Point::dimensionality;
    if (curr_node->left != nullptr && target_point.coords[axis] - eps <= curr_node->point.coords[axis]) {
        searchHelper(curr_node->left.get(), target_point, eps, points);
    }
    if (curr_node->right != nullptr && target_point.coords[axis] + eps >= curr_node->point.coords[axis]) {
        searchHelper(curr_node->right.get(), target_point, eps, points);
    }
}

double KDtree::distance(Point& p1, Point& p2) {
    double sum = 0;
    for (size_t i = 0; i < Point::dimensionality; i++) {
        sum += std::pow(p1.coords[i] - p2.coords[i], 2);
    }
    return std::sqrt(sum);
}

