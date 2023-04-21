#ifndef KDTREE_H
#define KDTREE_H

#include "point.h"
#include <vector>
#include <memory>

class KDtree {
public:
    KDtree();
    void buildTree(std::vector<Point> points);
    std::vector<Point> search(const Point& target_point, const double distance);
private:
    struct Node {
        Point point;
        std::vector<Point> points;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
    };
    std::unique_ptr<Node> root;
    size_t leafsize;
    void buildTreeHelper(std::vector<Point> points, std::unique_ptr<Node>& curr_node, const size_t depth);
    void searchHelper(const Node* curr_node, const Point& target_point, const double distance, std::vector<Point>& points);
    double distance(const Point& p1, const Point& p2);
};

#endif // KDTREE_H