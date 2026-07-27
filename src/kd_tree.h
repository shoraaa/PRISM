#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <queue>
#include <utility>
#include <vector>

namespace prism {

class KDTree2D {
public:
  explicit KDTree2D(const std::vector<float> &coordinates)
      : KDTree2D(coordinates, {}) {}

  KDTree2D(const std::vector<float> &coordinates,
           const std::vector<int32_t> &indices) {
    const int32_t count = static_cast<int32_t>(coordinates.size() / 2);
    query_points_.reserve(count);
    for (int32_t index = 0; index < count; ++index) {
      const Point point{
          coordinates[2 * index], coordinates[2 * index + 1], index};
      query_points_.push_back(point);
    }
    if (indices.empty()) {
      points_ = query_points_;
    } else {
      points_.reserve(indices.size());
      for (int32_t index : indices) {
        if (index >= 0 && index < count)
          points_.push_back(query_points_[index]);
      }
    }
    root_ = build(0, static_cast<int32_t>(points_.size()), 0);
  }

  std::vector<int32_t> nearest(int32_t query_index, int32_t count) const {
    if (count <= 0 || query_index < 0 ||
        query_index >= static_cast<int32_t>(points_.size())) {
      return {};
    }
    const Point &query = query_points_[query_index];
    Heap heap;
    search(root_.get(), query, query_index, count, heap);
    std::vector<std::pair<float, int32_t>> ordered;
    ordered.reserve(heap.size());
    while (!heap.empty()) {
      ordered.emplace_back(heap.top().distance2, heap.top().index);
      heap.pop();
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const auto &lhs, const auto &rhs) {
                return lhs.first == rhs.first ? lhs.second < rhs.second
                                              : lhs.first < rhs.first;
              });
    std::vector<int32_t> result;
    result.reserve(ordered.size());
    for (const auto &[distance2, index] : ordered) {
      (void)distance2;
      result.push_back(index);
    }
    return result;
  }

private:
  struct Point {
    float x;
    float y;
    int32_t index;
  };

  struct Node {
    Point point;
    int axis;
    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;
  };

  struct Neighbor {
    float distance2;
    int32_t index;
    bool operator<(const Neighbor &other) const {
      return distance2 == other.distance2 ? index < other.index
                                          : distance2 < other.distance2;
    }
  };
  using Heap = std::priority_queue<Neighbor>;

  std::vector<Point> points_;
  std::vector<Point> query_points_;
  std::unique_ptr<Node> root_;

  std::unique_ptr<Node> build(int32_t begin, int32_t end, int axis) {
    if (begin >= end) {
      return nullptr;
    }
    const int32_t middle = begin + (end - begin) / 2;
    const auto compare = [axis](const Point &lhs, const Point &rhs) {
      const float lhs_value = axis == 0 ? lhs.x : lhs.y;
      const float rhs_value = axis == 0 ? rhs.x : rhs.y;
      return lhs_value == rhs_value ? lhs.index < rhs.index
                                    : lhs_value < rhs_value;
    };
    std::nth_element(points_.begin() + begin, points_.begin() + middle,
                     points_.begin() + end, compare);
    auto node = std::make_unique<Node>();
    node->point = points_[middle];
    node->axis = axis;
    node->left = build(begin, middle, 1 - axis);
    node->right = build(middle + 1, end, 1 - axis);
    return node;
  }

  static float squared_distance(const Point &lhs, const Point &rhs) {
    const float dx = lhs.x - rhs.x;
    const float dy = lhs.y - rhs.y;
    return dx * dx + dy * dy;
  }

  static void consider(const Point &point, const Point &query,
                       int32_t excluded, int32_t count, Heap &heap) {
    if (point.index == excluded) {
      return;
    }
    const Neighbor candidate{squared_distance(point, query), point.index};
    if (static_cast<int32_t>(heap.size()) < count) {
      heap.push(candidate);
    } else if (candidate.distance2 < heap.top().distance2 ||
               (candidate.distance2 == heap.top().distance2 &&
                candidate.index < heap.top().index)) {
      heap.pop();
      heap.push(candidate);
    }
  }

  static void search(const Node *node, const Point &query, int32_t excluded,
                     int32_t count, Heap &heap) {
    if (node == nullptr) {
      return;
    }
    consider(node->point, query, excluded, count, heap);
    const float query_value = node->axis == 0 ? query.x : query.y;
    const float split_value = node->axis == 0 ? node->point.x : node->point.y;
    const Node *near = query_value < split_value ? node->left.get()
                                                 : node->right.get();
    const Node *far = query_value < split_value ? node->right.get()
                                                : node->left.get();
    search(near, query, excluded, count, heap);
    const float delta = query_value - split_value;
    if (static_cast<int32_t>(heap.size()) < count ||
        delta * delta <= heap.top().distance2) {
      search(far, query, excluded, count, heap);
    }
  }
};

} // namespace prism
