#include "decoder.h"
#include "kd_tree.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <memory>
#include <optional>
#include <queue>
#include <stdexcept>

#include <omp.h>

namespace prism {
namespace {

constexpr float EPS = 1.0e-6f;
constexpr float FEASIBILITY_EPS = 1.0e-5f;
constexpr int32_t SRR_STRING_CANDIDATES = 8;
constexpr int32_t SRR_DIRECTED_CANDIDATES = 16;
constexpr int32_t SEQUENCE_DISJOINT_LIMIT = 256;
constexpr double SEQUENCE_INFINITY = 1.0e30;

struct SequenceSummary {
  bool empty = true;
  int32_t first = -1;
  int32_t last = -1;
  int32_t count = 0;
  double distance = 0.0;
  double load_delta = 0.0;
  double min_load_delta = 0.0;
  double max_load_delta = 0.0;
  double duration = 0.0;
  double time_warp = 0.0;
  double earliest = 0.0;
  double latest = 0.0;
  double prize = 0.0;
  double penalty = 0.0;
  bool has_linehaul = false;
  bool has_backhaul = false;
  bool backhaul_violation = false;
};

SequenceSummary node_summary(const Problem &problem, int32_t node) {
  SequenceSummary result;
  result.empty = false;
  result.first = node;
  result.last = node;
  result.count = 1;
  result.load_delta = -problem.demand[node];
  result.min_load_delta = std::min(0.0, result.load_delta);
  result.max_load_delta = std::max(0.0, result.load_delta);
  result.duration = problem.service_time[node];
  result.earliest = std::max(static_cast<double>(problem.tw_start[node]),
                             -SEQUENCE_INFINITY);
  result.latest = std::min(static_cast<double>(problem.tw_end[node]),
                           SEQUENCE_INFINITY);
  result.prize = problem.prize[node];
  result.penalty = problem.penalty[node];
  result.has_linehaul = problem.demand[node] > FEASIBILITY_EPS;
  result.has_backhaul = problem.demand[node] < -FEASIBILITY_EPS;
  return result;
}

SequenceSummary concatenate(const Problem &problem,
                            const SequenceSummary &lhs,
                            const SequenceSummary &rhs) {
  if (lhs.empty)
    return rhs;
  if (rhs.empty)
    return lhs;
  SequenceSummary result;
  result.empty = false;
  result.first = lhs.first;
  result.last = rhs.last;
  result.count = lhs.count + rhs.count;
  const double travel = problem.dist(lhs.last, rhs.first);
  result.distance = lhs.distance + travel + rhs.distance;
  result.load_delta = lhs.load_delta + rhs.load_delta;
  result.min_load_delta =
      std::min(lhs.min_load_delta, lhs.load_delta + rhs.min_load_delta);
  result.max_load_delta =
      std::max(lhs.max_load_delta, lhs.load_delta + rhs.max_load_delta);

  const double delta = lhs.duration - lhs.time_warp + travel;
  const double wait = std::max(rhs.earliest - delta - lhs.latest, 0.0);
  const double warp = std::max(lhs.earliest + delta - rhs.latest, 0.0);
  result.duration = lhs.duration + rhs.duration + travel + wait;
  result.time_warp = lhs.time_warp + rhs.time_warp + warp;
  result.earliest = std::max(rhs.earliest - delta, lhs.earliest) - wait;
  result.latest = std::min(rhs.latest - delta, lhs.latest) + warp;
  result.prize = lhs.prize + rhs.prize;
  result.penalty = lhs.penalty + rhs.penalty;
  result.has_linehaul = lhs.has_linehaul || rhs.has_linehaul;
  result.has_backhaul = lhs.has_backhaul || rhs.has_backhaul;
  result.backhaul_violation =
      lhs.backhaul_violation || rhs.backhaul_violation ||
      (lhs.has_backhaul && rhs.has_linehaul);
  return result;
}

struct SequenceTable {
  std::vector<int32_t> nodes;
  std::vector<SequenceSummary> singleton;
  std::vector<SequenceSummary> reverse_singleton;
  std::vector<std::vector<SequenceSummary>> forward;
  std::vector<std::vector<SequenceSummary>> backward;
  int32_t tree_base = 0;
  std::vector<SequenceSummary> forward_tree;
  std::vector<SequenceSummary> backward_tree;
};

std::vector<std::vector<SequenceSummary>>
build_disjoint_table(const Problem &problem,
                     const std::vector<SequenceSummary> &values) {
  const int32_t size = static_cast<int32_t>(values.size());
  int32_t levels = 0;
  while ((int64_t{1} << levels) < std::max(size, 1))
    ++levels;
  std::vector<std::vector<SequenceSummary>> table(
      std::max(levels, 1), std::vector<SequenceSummary>(size));
  for (int32_t level = 0; level < levels; ++level) {
    const int32_t half = 1 << level;
    const int32_t block = half << 1;
    for (int32_t start = 0; start < size; start += block) {
      const int32_t middle = std::min(start + half, size);
      const int32_t end = std::min(start + block, size);
      if (middle > start) {
        table[level][middle - 1] = values[middle - 1];
        for (int32_t index = middle - 2; index >= start; --index) {
          table[level][index] =
              concatenate(problem, values[index], table[level][index + 1]);
        }
      }
      if (middle < end) {
        table[level][middle] = values[middle];
        for (int32_t index = middle + 1; index < end; ++index) {
          table[level][index] =
              concatenate(problem, table[level][index - 1], values[index]);
        }
      }
    }
  }
  return table;
}

std::pair<int32_t, std::vector<SequenceSummary>>
build_segment_tree(const Problem &problem,
                   const std::vector<SequenceSummary> &values) {
  int32_t base = 1;
  while (base < static_cast<int32_t>(values.size()))
    base <<= 1;
  std::vector<SequenceSummary> tree(2 * base);
  for (int32_t index = 0; index < static_cast<int32_t>(values.size()); ++index)
    tree[base + index] = values[index];
  for (int32_t index = base - 1; index > 0; --index) {
    tree[index] = concatenate(problem, tree[2 * index], tree[2 * index + 1]);
  }
  return {base, std::move(tree)};
}

SequenceTable build_sequence_table(const Problem &problem,
                                   std::vector<int32_t> nodes) {
  SequenceTable result;
  result.nodes = std::move(nodes);
  result.singleton.reserve(result.nodes.size());
  for (int32_t node : result.nodes)
    result.singleton.push_back(node_summary(problem, node));
  result.reverse_singleton.assign(result.singleton.rbegin(),
                                  result.singleton.rend());
  if (result.nodes.size() <= SEQUENCE_DISJOINT_LIMIT) {
    result.forward = build_disjoint_table(problem, result.singleton);
    result.backward = build_disjoint_table(problem, result.reverse_singleton);
  } else {
    auto forward = build_segment_tree(problem, result.singleton);
    result.tree_base = forward.first;
    result.forward_tree = std::move(forward.second);
    auto backward = build_segment_tree(problem, result.reverse_singleton);
    result.backward_tree = std::move(backward.second);
  }
  return result;
}

SequenceSummary query_disjoint(
    const Problem &problem,
    const std::vector<std::vector<SequenceSummary>> &table,
    const std::vector<SequenceSummary> &singletons, int32_t begin,
    int32_t end) {
  if (begin >= end)
    return {};
  if (end == begin + 1)
    return singletons[begin];
  uint32_t difference = static_cast<uint32_t>(begin ^ (end - 1));
  int32_t level = 0;
  while (difference >>= 1)
    ++level;
  return concatenate(problem, table[level][begin], table[level][end - 1]);
}

SequenceSummary query_segment_tree(const Problem &problem,
                                   const std::vector<SequenceSummary> &tree,
                                   int32_t base, int32_t begin, int32_t end) {
  SequenceSummary lhs;
  SequenceSummary rhs;
  begin += base;
  end += base;
  while (begin < end) {
    if (begin & 1)
      lhs = concatenate(problem, lhs, tree[begin++]);
    if (end & 1)
      rhs = concatenate(problem, tree[--end], rhs);
    begin >>= 1;
    end >>= 1;
  }
  return concatenate(problem, lhs, rhs);
}

SequenceSummary query_sequence(const Problem &problem,
                               const SequenceTable &table, int32_t begin,
                               int32_t end, bool reverse = false) {
  if (!reverse) {
    if (table.tree_base > 0)
      return query_segment_tree(problem, table.forward_tree, table.tree_base,
                                begin, end);
    return query_disjoint(problem, table.forward, table.singleton, begin, end);
  }
  const int32_t size = static_cast<int32_t>(table.nodes.size());
  if (table.tree_base > 0)
    return query_segment_tree(problem, table.backward_tree, table.tree_base,
                              size - end, size - begin);
  return query_disjoint(problem, table.backward, table.reverse_singleton,
                        size - end, size - begin);
}

enum Metric : int {
  METRIC_TIME_WINDOW,
  METRIC_CAPACITY,
  METRIC_BACKHAUL,
  METRIC_PICKUP_DELIVERY,
  METRIC_ROUTE_LIMIT,
  METRIC_PRIZE,
};

uint64_t splitmix64(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

bool finite_nonnegative(float value) {
  return std::isfinite(value) && value >= 0.0f;
}

} // namespace

bool Problem::has(Constraint constraint) const {
  return (constraints & static_cast<uint32_t>(constraint)) != 0;
}

float Problem::dist(int32_t from, int32_t to) const {
  if (distance.empty()) {
    const float dx = coordinates[2 * from] - coordinates[2 * to];
    const float dy = coordinates[2 * from + 1] - coordinates[2 * to + 1];
    return std::hypot(dx, dy);
  }
  return distance[static_cast<size_t>(from) * node_count + to];
}

int32_t Problem::customer_count() const { return node_count - depot_count; }

void Problem::validate() const {
  if (name.empty()) {
    throw std::invalid_argument("problem name must not be empty");
  }
  if (node_count < 2) {
    throw std::invalid_argument("a problem must contain at least two nodes");
  }
  if (depot_count < 0 || depot_count >= node_count) {
    throw std::invalid_argument("depot_count must be in [0, node_count)");
  }
  if (multi_route && depot_count == 0) {
    throw std::invalid_argument("multi-route problems require a depot");
  }
  const size_t n = static_cast<size_t>(node_count);
  if (!distance.empty() && distance.size() != n * n) {
    throw std::invalid_argument(
        "distance must have shape (node_count, node_count)");
  }
  if (distance.empty() && coordinates.empty()) {
    throw std::invalid_argument(
        "either distance or coordinates must be provided");
  }
  if (!coordinates.empty()) {
    if (coordinates.size() != 2 * n) {
      throw std::invalid_argument(
          "coordinates must have shape (node_count, 2)");
    }
    for (float value : coordinates) {
      if (!std::isfinite(value)) {
        throw std::invalid_argument("coordinates must be finite");
      }
    }
  }
  for (float value : distance) {
    if (!finite_nonnegative(value)) {
      throw std::invalid_argument(
          "distance must contain finite non-negative values");
    }
  }

  const auto require_node_vector = [n](const std::vector<float> &values,
                                       const char *field) {
    if (values.size() != n) {
      throw std::invalid_argument(std::string(field) +
                                  " must contain node_count values");
    }
  };
  require_node_vector(demand, "demand");
  require_node_vector(prize, "prize");
  require_node_vector(penalty, "penalty");
  require_node_vector(tw_start, "tw_start");
  require_node_vector(tw_end, "tw_end");
  require_node_vector(service_time, "service_time");

  if (delivery_of_pickup.size() != n || pickup_of_delivery.size() != n) {
    throw std::invalid_argument(
        "pickup-delivery relation arrays must contain node_count values");
  }
  if (has(CAPACITY) && (!std::isfinite(capacity) || capacity <= 0.0f)) {
    throw std::invalid_argument("capacity must be finite and positive");
  }
  if (has(ROUTE_LIMIT) &&
      (!std::isfinite(route_limit) || route_limit <= 0.0f)) {
    throw std::invalid_argument("route_limit must be finite and positive");
  }
  if (has(TOUR_LIMIT) && (!std::isfinite(tour_limit) || tour_limit <= 0.0f)) {
    throw std::invalid_argument("tour_limit must be finite and positive");
  }
  if (has(PRIZE_QUOTA) && (!std::isfinite(prize_quota) || prize_quota < 0.0f)) {
    throw std::invalid_argument("prize_quota must be finite and non-negative");
  }
  if (has(PICKUP_DELIVERY)) {
    for (int32_t node = depot_count; node < node_count; ++node) {
      const int32_t delivery = delivery_of_pickup[node];
      const int32_t pickup = pickup_of_delivery[node];
      if (delivery >= 0 && (delivery < depot_count || delivery >= node_count ||
                            pickup_of_delivery[delivery] != node)) {
        throw std::invalid_argument("inconsistent pickup-delivery relation");
      }
      if (pickup >= 0 && (pickup < depot_count || pickup >= node_count ||
                          delivery_of_pickup[pickup] != node)) {
        throw std::invalid_argument("inconsistent pickup-delivery relation");
      }
    }
  }
  for (const ResourceSpec &resource : resources) {
    if (resource.name.empty())
      throw std::invalid_argument("resource name must not be empty");
    if (resource.state_dim != 1) {
      throw std::invalid_argument(
          "resource algebra v1 currently requires state_dim == 1");
    }
    if (!std::isfinite(resource.initial) || !std::isfinite(resource.scale) ||
        resource.scale <= 0.0f || std::isnan(resource.lower) ||
        std::isnan(resource.upper) || resource.lower > resource.upper ||
        !std::isfinite(resource.edge_coefficient) ||
        !std::isfinite(resource.node_coefficient) ||
        !std::isfinite(resource.reset_value)) {
      throw std::invalid_argument("invalid resource algebra scalar");
    }
    if (!resource.edge_values.empty() && resource.edge_values.size() != n * n)
      throw std::invalid_argument(
          "resource edge values must have shape (node_count, node_count)");
    if (!resource.node_values.empty() && resource.node_values.size() != n)
      throw std::invalid_argument(
          "resource node values must have shape (node_count,)");
    if (!resource.reset_nodes.empty() && resource.reset_nodes.size() != n)
      throw std::invalid_argument(
          "resource reset flags must have shape (node_count,)");
    for (float value : resource.edge_values) {
      if (!std::isfinite(value))
        throw std::invalid_argument("resource edge values must be finite");
    }
    for (float value : resource.node_values) {
      if (!std::isfinite(value))
        throw std::invalid_argument("resource node values must be finite");
    }
  }
  for (int32_t node = 0; node < node_count; ++node) {
    if (!std::isfinite(demand[node]) || !finite_nonnegative(prize[node]) ||
        !finite_nonnegative(penalty[node]) ||
        !finite_nonnegative(service_time[node]) || std::isnan(tw_start[node]) ||
        std::isnan(tw_end[node]) || tw_start[node] > tw_end[node]) {
      throw std::invalid_argument("invalid node resource value");
    }
  }
}

void CandidateConfig::validate() const {
  if (max_candidates <= 0 || max_candidates > 64) {
    throw std::invalid_argument("max_candidates must be in [1, 64]");
  }
  const float weights[] = {
      gamma_unit,     gamma_wait,       gamma_time_warp, gamma_load_fit,
      gamma_ordering, gamma_precedence, gamma_route,     gamma_prize,
  };
  for (float weight : weights) {
    if (!std::isfinite(weight) || weight < 0.0f) {
      throw std::invalid_argument(
          "candidate proximity weights must be finite and non-negative");
    }
  }
}

void SearchConfig::validate() const {
  if (min_changed_edges <= 0) {
    throw std::invalid_argument("min_changed_edges must be positive");
  }
  if (max_perturb_attempts <= 0) {
    throw std::invalid_argument("max_perturb_attempts must be positive");
  }
  if (or_opt_max_segment <= 0 || or_opt_max_segment > 3) {
    throw std::invalid_argument("or_opt_max_segment must be in [1, 3]");
  }
  if (feasibility_lookahead_depth < 0 ||
      feasibility_lookahead_depth > 4) {
    throw std::invalid_argument(
        "feasibility_lookahead_depth must be in [0, 4]");
  }
  if (srr_candidate_limit <= 0 || srr_candidate_limit > 64) {
    throw std::invalid_argument("srr_candidate_limit must be in [1, 64]");
  }
}

const char *objective_name(Objective objective) {
  switch (objective) {
  case Objective::MIN_DISTANCE:
    return "distance";
  case Objective::MAX_PRIZE:
    return "prize";
  case Objective::MIN_DISTANCE_PLUS_PENALTY:
    return "distance_plus_penalty";
  }
  return "unknown";
}

const char *objective_direction(Objective objective) {
  return objective == Objective::MAX_PRIZE ? "maximize" : "minimize";
}

std::vector<std::string> constraint_names(uint32_t constraints) {
  const std::pair<Constraint, const char *> known[] = {
      {VISIT_ALL, "visit_all"},           {CAPACITY, "capacity"},
      {BACKHAUL_ORDER, "backhaul_order"}, {PICKUP_DELIVERY, "pickup_delivery"},
      {ROUTE_LIMIT, "route_limit"},       {TIME_WINDOWS, "time_windows"},
      {TOUR_LIMIT, "tour_limit"},         {PRIZE_QUOTA, "prize_quota"},
  };
  std::vector<std::string> result;
  for (const auto &[flag, name] : known) {
    if ((constraints & static_cast<uint32_t>(flag)) != 0) {
      result.emplace_back(name);
    }
  }
  return result;
}

std::vector<std::string> candidate_feature_names() {
  return {"distance",          "capacity",       "time_window",
          "route_limit",       "tour_limit",     "backhaul_order",
          "pickup_delivery",   "prize_quota",    "incumbent_forward",
          "incumbent_backward", "open_return"};
}

std::vector<std::string> node_feature_names() {
  return {"x",
          "y",
          "is_depot",
          "linehaul_demand",
          "backhaul_demand",
          "prize",
          "penalty",
          "window_start",
          "window_end",
          "service_time",
          "is_pickup",
          "is_delivery",
          "incumbent_served",
          "route_position",
          "forward_load",
          "forward_time",
          "forward_slack",
          "open_pickups",
          "forward_distance",
          "backward_distance",
          "backward_load",
          "backward_time",
          "backward_slack",
          "backward_open_pickups"};
}

std::vector<std::string> field_channel_names() {
  return {"capacity",       "time_window",     "route_limit",
          "tour_limit",     "backhaul_order",  "pickup_delivery",
          "prize_quota"};
}

RoutingDecoder::RoutingDecoder(Problem problem, CandidateConfig candidate_config,
                       SearchConfig search_config, int32_t n_rollouts, float beta)
    : problem_(std::move(problem)),
      candidate_config_(std::move(candidate_config)),
      search_config_(std::move(search_config)), n_rollouts_(n_rollouts), beta_(beta) {
  problem_.validate();
  candidate_config_.validate();
  search_config_.validate();
  if (n_rollouts_ <= 0) {
    throw std::invalid_argument("n_rollouts must be positive");
  }
  if (beta_ < 0.0f) {
    throw std::invalid_argument("beta must be non-negative");
  }
  for (float value : problem_.distance) {
    if (std::isfinite(value))
      distance_scale_ = std::max(distance_scale_, value);
  }
  if (problem_.distance.empty()) {
    float min_x = problem_.coordinates[0];
    float max_x = min_x;
    float min_y = problem_.coordinates[1];
    float max_y = min_y;
    for (int32_t node = 1; node < problem_.node_count; ++node) {
      min_x = std::min(min_x, problem_.coordinates[2 * node]);
      max_x = std::max(max_x, problem_.coordinates[2 * node]);
      min_y = std::min(min_y, problem_.coordinates[2 * node + 1]);
      max_y = std::max(max_y, problem_.coordinates[2 * node + 1]);
    }
    distance_scale_ =
        std::max(distance_scale_, std::hypot(max_x - min_x, max_y - min_y));
  }
  for (int32_t node = 0; node < problem_.node_count; ++node) {
    if (std::isfinite(problem_.tw_end[node]))
      time_scale_ = std::max(time_scale_, problem_.tw_end[node]);
    time_scale_ = std::max(time_scale_, problem_.service_time[node]);
    prize_scale_ = std::max(prize_scale_, problem_.prize[node]);
    penalty_scale_ = std::max(penalty_scale_, problem_.penalty[node]);
    pair_count_ += problem_.delivery_of_pickup[node] >= 0 ? 1 : 0;
  }
  time_scale_ = std::max(time_scale_, distance_scale_);
  reversal_safe_ = !problem_.has(TIME_WINDOWS) &&
                   !problem_.has(BACKHAUL_ORDER) &&
                   !problem_.has(PICKUP_DELIVERY);
  for (int32_t from = 0;
       !problem_.distance.empty() && reversal_safe_ &&
       from < problem_.node_count;
       ++from) {
    for (int32_t to = from + 1; to < problem_.node_count; ++to) {
      const float scale =
          std::max({1.0f, problem_.dist(from, to), problem_.dist(to, from)});
      if (std::abs(problem_.dist(from, to) - problem_.dist(to, from)) >
          1.0e-5f * scale) {
        reversal_safe_ = false;
        break;
      }
    }
  }
  build_resource_registry();
  build_resource_descriptors();
  build_candidate_graph({});
}

void RoutingDecoder::build_resource_registry() {
  resources_.clear();
  legacy_resource_index_.fill(-1);
  const auto add_legacy = [&](FieldChannel channel, ResourceOperator op,
                              const char *name, bool active) {
    ResourceSpec spec;
    spec.name = name;
    spec.active = active;
    spec.op = op;
    spec.scale = resource_scale(static_cast<int32_t>(channel));
    const int32_t index = static_cast<int32_t>(resources_.size());
    resources_.push_back(std::move(spec));
    legacy_resource_index_[static_cast<int32_t>(channel)] = index;
  };
  add_legacy(FieldChannel::CAPACITY, ResourceOperator::LEGACY_CAPACITY,
             "capacity", problem_.has(CAPACITY));
  add_legacy(FieldChannel::TIME_WINDOW, ResourceOperator::LEGACY_TIME_WINDOW,
             "time_window", problem_.has(TIME_WINDOWS));
  add_legacy(FieldChannel::ROUTE_LIMIT, ResourceOperator::LEGACY_ROUTE_LIMIT,
             "route_limit", problem_.has(ROUTE_LIMIT));
  add_legacy(FieldChannel::TOUR_LIMIT, ResourceOperator::LEGACY_TOUR_LIMIT,
             "tour_limit", problem_.has(TOUR_LIMIT));
  add_legacy(FieldChannel::BACKHAUL_ORDER,
             ResourceOperator::LEGACY_BACKHAUL_ORDER, "backhaul_order",
             problem_.has(BACKHAUL_ORDER));
  add_legacy(FieldChannel::PICKUP_DELIVERY,
             ResourceOperator::LEGACY_PICKUP_DELIVERY, "pickup_delivery",
             problem_.has(PICKUP_DELIVERY));
  add_legacy(FieldChannel::PRIZE_QUOTA, ResourceOperator::LEGACY_PRIZE_QUOTA,
             "prize_quota", problem_.has(PRIZE_QUOTA));

  for (const ResourceSpec &spec : problem_.resources) {
    if (std::any_of(resources_.begin(), resources_.end(),
                    [&](const ResourceSpec &existing) {
                      return existing.name == spec.name;
                    })) {
      throw std::invalid_argument("duplicate resource name: " + spec.name);
    }
    resources_.push_back(spec);
  }
}

int32_t RoutingDecoder::legacy_resource_index(FieldChannel channel) const {
  return legacy_resource_index_[static_cast<int32_t>(channel)];
}

const ResourceSpec &RoutingDecoder::resource(int32_t index) const {
  if (index < 0 || index >= resource_count())
    throw std::out_of_range("resource index is out of range");
  return resources_[index];
}

void RoutingDecoder::build_resource_descriptors() {
  resource_descriptors_.assign(
      static_cast<size_t>(resource_count()) * RESOURCE_DESCRIPTOR_DIM, 0.0f);
  const auto squash = [](double value) {
    value = std::max(value, 0.0);
    return static_cast<float>(value / (1.0 + value));
  };
  for (int32_t index = 0; index < resource_count(); ++index) {
    const ResourceSpec &spec = resources_[index];
    float *descriptor = resource_descriptors_.data() +
                        static_cast<size_t>(index) * RESOURCE_DESCRIPTOR_DIM;
    switch (spec.op) {
    case ResourceOperator::AFFINE_ACCUMULATOR:
    case ResourceOperator::LEGACY_CAPACITY:
    case ResourceOperator::LEGACY_ROUTE_LIMIT:
    case ResourceOperator::LEGACY_TOUR_LIMIT:
    case ResourceOperator::LEGACY_PRIZE_QUOTA:
      descriptor[0] = 1.0f;
      descriptor[20] = 1.0f;
      break;
    case ResourceOperator::LEGACY_TIME_WINDOW:
      descriptor[1] = 1.0f;
      descriptor[21] = 1.0f;
      break;
    case ResourceOperator::LEGACY_BACKHAUL_ORDER:
      descriptor[2] = 1.0f;
      descriptor[22] = 1.0f;
      break;
    case ResourceOperator::LEGACY_PICKUP_DELIVERY:
      descriptor[3] = 1.0f;
      descriptor[22] = 1.0f;
      break;
    }
    const bool has_lower = std::isfinite(spec.lower) ||
                           spec.op == ResourceOperator::LEGACY_PRIZE_QUOTA;
    const bool has_upper = std::isfinite(spec.upper) ||
                           spec.op == ResourceOperator::LEGACY_CAPACITY ||
                           spec.op == ResourceOperator::LEGACY_TIME_WINDOW ||
                           spec.op == ResourceOperator::LEGACY_ROUTE_LIMIT ||
                           spec.op == ResourceOperator::LEGACY_TOUR_LIMIT;
    descriptor[has_lower && has_upper ? 7 : has_lower ? 5 : has_upper ? 6 : 4] =
        1.0f;
    descriptor[8 + static_cast<int32_t>(spec.bound_check)] = 1.0f;
    descriptor[11 + static_cast<int32_t>(spec.direction)] = 1.0f;
    descriptor[14 + static_cast<int32_t>(spec.scope)] = 1.0f;
    const bool event_reset = !spec.reset_nodes.empty();
    descriptor[event_reset ? 19 : spec.reset_at_depot ? 18 : 17] = 1.0f;
    descriptor[23] = !spec.node_values.empty() ? 1.0f : 0.0f;
    descriptor[24] =
        (spec.edge_uses_distance || !spec.edge_values.empty()) ? 1.0f : 0.0f;
    descriptor[25] = spec.op == ResourceOperator::LEGACY_PICKUP_DELIVERY ? 1.0f
                                                                         : 0.0f;
    descriptor[26] = spec.edge_coefficient >= 0.0f &&
                             spec.node_coefficient >= 0.0f
                         ? 1.0f
                         : 0.0f;
    descriptor[27] = spec.edge_coefficient <= 0.0f &&
                             spec.node_coefficient <= 0.0f
                         ? 1.0f
                         : 0.0f;
    descriptor[28] = squash(spec.state_dim);
    descriptor[29] = squash(std::log1p(
        spec.scale / std::max<double>(distance_scale_, EPS)));
    double magnitude = 0.0;
    if (spec.edge_uses_distance)
      magnitude += std::abs(spec.edge_coefficient) * distance_scale_;
    for (float value : spec.edge_values)
      magnitude += std::abs(value * spec.edge_coefficient);
    for (float value : spec.node_values)
      magnitude += std::abs(value * spec.node_coefficient);
    const size_t count = spec.edge_values.size() + spec.node_values.size() +
                         (spec.edge_uses_distance ? 1 : 0);
    descriptor[30] = squash(count ? magnitude / count / spec.scale : 0.0);
    const double events =
        std::count(spec.reset_nodes.begin(), spec.reset_nodes.end(), uint8_t{1});
    descriptor[31] = problem_.node_count > 0
                         ? static_cast<float>(events / problem_.node_count)
                         : 0.0f;
  }
}

void RoutingDecoder::seed(uint64_t value) {
  seed_ = value;
  generation_ = 0;
}

std::vector<int32_t> RoutingDecoder::rank_by_distance(int32_t from,
                                                      int32_t limit) const {
  std::vector<int32_t> nodes;
  nodes.reserve(problem_.node_count - 1);
  for (int32_t to = 0; to < problem_.node_count; ++to) {
    if (to != from) {
      nodes.push_back(to);
    }
  }
  const auto compare = [&](int32_t lhs, int32_t rhs) {
    const float lhs_score = problem_.dist(from, lhs);
    const float rhs_score = problem_.dist(from, rhs);
    return lhs_score == rhs_score ? lhs < rhs : lhs_score < rhs_score;
  };
  limit = std::min<int32_t>(limit, nodes.size());
  if (limit < static_cast<int32_t>(nodes.size())) {
    std::nth_element(nodes.begin(), nodes.begin() + limit, nodes.end(),
                     compare);
    nodes.resize(limit);
  }
  std::sort(nodes.begin(), nodes.end(), compare);
  return nodes;
}

float RoutingDecoder::resource_proximity(int32_t from, int32_t to,
                                     int metric) const {
  const CandidateConfig &config = candidate_config_;
  const float travel = problem_.dist(from, to);
  const float base = config.gamma_unit * travel;
  switch (metric) {
  case METRIC_TIME_WINDOW: {
    // Vidal-lineage directed proximity: latest departure from i determines
    // unavoidable waiting, while earliest departure determines time warp.
    const float wait =
        std::isfinite(problem_.tw_end[from])
            ? std::max(problem_.tw_start[to] - problem_.service_time[from] -
                           travel - problem_.tw_end[from],
                       0.0f)
            : 0.0f;
    const float warp =
        std::isfinite(problem_.tw_end[to])
            ? std::max(problem_.tw_start[from] + problem_.service_time[from] +
                           travel - problem_.tw_end[to],
                       0.0f)
            : 0.0f;
    return base + config.gamma_wait * wait + config.gamma_time_warp * warp;
  }
  case METRIC_CAPACITY: {
    const float from_demand = std::abs(problem_.demand[from]);
    const float to_demand = std::abs(problem_.demand[to]);
    const bool same_sign = problem_.demand[from] * problem_.demand[to] >= 0.0f;
    const float combined = same_sign ? from_demand + to_demand : to_demand;
    const float overflow = std::max(combined - problem_.capacity, 0.0f);
    const float unused = std::max(
        problem_.capacity - std::min(combined, problem_.capacity), 0.0f);
    return base + config.gamma_load_fit * (unused + 10.0f * overflow);
  }
  case METRIC_BACKHAUL: {
    const bool illegal_order = problem_.demand[from] < -FEASIBILITY_EPS &&
                               problem_.demand[to] > FEASIBILITY_EPS;
    return base + config.gamma_ordering * (illegal_order ? 1.0f : 0.0f);
  }
  case METRIC_PICKUP_DELIVERY: {
    const int32_t required_pickup = problem_.pickup_of_delivery[to];
    const bool direct_pair = problem_.delivery_of_pickup[from] == to;
    const bool precedence_unknown = required_pickup >= 0 && !direct_pair;
    return base + config.gamma_precedence * (precedence_unknown ? 1.0f : 0.0f);
  }
  case METRIC_ROUTE_LIMIT: {
    float return_distance = 0.0f;
    if (!problem_.open_route && problem_.depot_count > 0) {
      return_distance = std::numeric_limits<float>::infinity();
      for (int32_t depot = 0; depot < problem_.depot_count; ++depot) {
        return_distance = std::min(return_distance, problem_.dist(to, depot));
      }
    }
    return base + config.gamma_route * return_distance;
  }
  case METRIC_PRIZE: {
    const float value = problem_.prize[to] + problem_.penalty[to];
    return base + config.gamma_prize / std::max(value, EPS);
  }
  default:
    return base;
  }
}

float RoutingDecoder::classical_proximity(int32_t from, int32_t to) const {
  const float base = candidate_config_.gamma_unit * problem_.dist(from, to);
  float result = base;
  const auto add_resource = [&](int metric) {
    result += resource_proximity(from, to, metric) - base;
  };
  if (problem_.has(TIME_WINDOWS))
    add_resource(METRIC_TIME_WINDOW);
  if (problem_.has(CAPACITY))
    add_resource(METRIC_CAPACITY);
  if (problem_.has(BACKHAUL_ORDER))
    add_resource(METRIC_BACKHAUL);
  if (problem_.has(PICKUP_DELIVERY))
    add_resource(METRIC_PICKUP_DELIVERY);
  if (problem_.has(ROUTE_LIMIT) || problem_.has(TOUR_LIMIT))
    add_resource(METRIC_ROUTE_LIMIT);
  if (problem_.objective != Objective::MIN_DISTANCE)
    add_resource(METRIC_PRIZE);
  return result;
}

bool RoutingDecoder::field_channel_active(int32_t channel) const {
  switch (static_cast<FieldChannel>(channel)) {
  case FieldChannel::CAPACITY:
    return problem_.has(CAPACITY);
  case FieldChannel::TIME_WINDOW:
    return problem_.has(TIME_WINDOWS);
  case FieldChannel::ROUTE_LIMIT:
    return problem_.has(ROUTE_LIMIT);
  case FieldChannel::TOUR_LIMIT:
    return problem_.has(TOUR_LIMIT);
  case FieldChannel::BACKHAUL_ORDER:
    return problem_.has(BACKHAUL_ORDER);
  case FieldChannel::PICKUP_DELIVERY:
    return problem_.has(PICKUP_DELIVERY);
  case FieldChannel::PRIZE_QUOTA:
    return problem_.has(PRIZE_QUOTA);
  }
  return false;
}

float RoutingDecoder::objective_edge_cost(int32_t from, int32_t to) const {
  const float travel = problem_.dist(from, to);
  if (to < problem_.depot_count)
    // The return-to-depot leg is charged its real travel for closed routes and
    // is genuinely free for open routes -- exactly matching the true objective
    // accumulated in transition()/finish(). Charging it for open routes used to
    // hide a phantom cost in the ranking energy that a non-negative field could
    // not discount, making the learned field net-harmful on open variants. The
    // fragmentation this previously guarded against (free returns making the
    // depot the cheapest move at every step) is now handled structurally in
    // select_next(), which drops depot options while a customer can still
    // legally extend the open route.
    return problem_.open_route ? 0.0f : travel;
  switch (problem_.objective) {
  case Objective::MIN_DISTANCE:
    return travel;
  case Objective::MAX_PRIZE:
    return -problem_.prize[to] + 1.0e-3f * travel / distance_scale_;
  case Objective::MIN_DISTANCE_PLUS_PENALTY:
    return travel - problem_.penalty[to];
  }
  return travel;
}

float RoutingDecoder::resource_scale(int32_t channel) const {
  switch (static_cast<FieldChannel>(channel)) {
  case FieldChannel::CAPACITY:
    return std::max(problem_.capacity, EPS);
  case FieldChannel::TIME_WINDOW:
    return time_scale_;
  case FieldChannel::ROUTE_LIMIT:
    return std::isfinite(problem_.route_limit)
               ? std::max(problem_.route_limit, EPS)
               : distance_scale_;
  case FieldChannel::TOUR_LIMIT:
    return std::isfinite(problem_.tour_limit)
               ? std::max(problem_.tour_limit, EPS)
               : distance_scale_;
  case FieldChannel::BACKHAUL_ORDER:
  case FieldChannel::PICKUP_DELIVERY:
    return 1.0f;
  case FieldChannel::PRIZE_QUOTA:
    return std::max(problem_.prize_quota, EPS);
  }
  return 1.0f;
}

std::vector<float> RoutingDecoder::resource_scales() const {
  std::vector<float> scales(resource_count());
  for (int32_t index = 0; index < resource_count(); ++index)
    scales[index] = runtime_resource_scale(index);
  return scales;
}

float RoutingDecoder::runtime_resource_scale(int32_t resource_index) const {
  const ResourceSpec &spec = resource(resource_index);
  switch (spec.op) {
  case ResourceOperator::LEGACY_CAPACITY:
    return resource_scale(static_cast<int32_t>(FieldChannel::CAPACITY));
  case ResourceOperator::LEGACY_TIME_WINDOW:
    return resource_scale(static_cast<int32_t>(FieldChannel::TIME_WINDOW));
  case ResourceOperator::LEGACY_ROUTE_LIMIT:
    return resource_scale(static_cast<int32_t>(FieldChannel::ROUTE_LIMIT));
  case ResourceOperator::LEGACY_TOUR_LIMIT:
    return resource_scale(static_cast<int32_t>(FieldChannel::TOUR_LIMIT));
  case ResourceOperator::LEGACY_BACKHAUL_ORDER:
    return resource_scale(static_cast<int32_t>(FieldChannel::BACKHAUL_ORDER));
  case ResourceOperator::LEGACY_PICKUP_DELIVERY:
    return resource_scale(static_cast<int32_t>(FieldChannel::PICKUP_DELIVERY));
  case ResourceOperator::LEGACY_PRIZE_QUOTA:
    return resource_scale(static_cast<int32_t>(FieldChannel::PRIZE_QUOTA));
  case ResourceOperator::AFFINE_ACCUMULATOR:
    return std::max(spec.scale, EPS);
  }
  return 1.0f;
}

float RoutingDecoder::objective_scale() const {
  // Average the objective magnitude over customer-arrival edges; depot legs are
  // mode-specific (open/closed) and would bias the scale, so they are skipped.
  double total = 0.0;
  int64_t count = 0;
  for (int32_t from = 0; from < problem_.node_count; ++from) {
    for (int32_t edge = edge_offsets_[from]; edge < edge_offsets_[from + 1];
         ++edge) {
      if (edge_to_[edge] < problem_.depot_count)
        continue;
      total += std::abs(objective_edge_costs_[edge]);
      ++count;
    }
  }
  if (count == 0)
    return 0.0f;
  const double ratio =
      (total / static_cast<double>(count)) / std::max(distance_scale_, EPS);
  // ratio / (1 + ratio) squashes [0, inf) into [0, 1) without a hard clamp.
  return static_cast<float>(ratio / (1.0 + ratio));
}

float RoutingDecoder::analytic_resource_pressure(int32_t from, int32_t to,
                                             int32_t channel) const {
  const float travel = problem_.dist(from, to);
  switch (static_cast<FieldChannel>(channel)) {
  case FieldChannel::CAPACITY:
    return std::abs(problem_.demand[to]);
  case FieldChannel::TIME_WINDOW: {
    const float wait =
        std::isfinite(problem_.tw_end[from])
            ? std::max(problem_.tw_start[to] - problem_.service_time[from] -
                           travel - problem_.tw_end[from],
                       0.0f)
            : 0.0f;
    const float warp =
        std::isfinite(problem_.tw_end[to])
            ? std::max(problem_.tw_start[from] + problem_.service_time[from] +
                           travel - problem_.tw_end[to],
                       0.0f)
            : 0.0f;
    return wait + warp;
  }
  case FieldChannel::ROUTE_LIMIT:
  case FieldChannel::TOUR_LIMIT: {
    float return_distance = 0.0f;
    if (!problem_.open_route && problem_.depot_count > 0) {
      return_distance = std::numeric_limits<float>::infinity();
      for (int32_t depot = 0; depot < problem_.depot_count; ++depot) {
        return_distance =
            std::min(return_distance, problem_.dist(to, depot));
      }
    }
    return travel + return_distance;
  }
  case FieldChannel::BACKHAUL_ORDER:
    return problem_.demand[from] < -FEASIBILITY_EPS &&
                   problem_.demand[to] > FEASIBILITY_EPS
               ? 1.0f / std::max(problem_.customer_count(), 1)
               : 0.0f;
  case FieldChannel::PICKUP_DELIVERY: {
    const int32_t pickup = problem_.pickup_of_delivery[to];
    return pickup >= 0 && pickup != from
               ? 1.0f / std::max(pair_count_, 1)
               : 0.0f;
  }
  case FieldChannel::PRIZE_QUOTA:
    return to < problem_.depot_count
               ? resource_scale(channel)
               : std::max(resource_scale(channel) - problem_.prize[to], 0.0f);
  }
  return 0.0f;
}

float RoutingDecoder::runtime_resource_pressure(int32_t from, int32_t to,
                                                int32_t resource_index) const {
  const ResourceSpec &spec = resource(resource_index);
  if (!spec.active)
    return 0.0f;
  switch (spec.op) {
  case ResourceOperator::LEGACY_CAPACITY:
    return analytic_resource_pressure(
        from, to, static_cast<int32_t>(FieldChannel::CAPACITY));
  case ResourceOperator::LEGACY_TIME_WINDOW:
    return analytic_resource_pressure(
        from, to, static_cast<int32_t>(FieldChannel::TIME_WINDOW));
  case ResourceOperator::LEGACY_ROUTE_LIMIT:
    return analytic_resource_pressure(
        from, to, static_cast<int32_t>(FieldChannel::ROUTE_LIMIT));
  case ResourceOperator::LEGACY_TOUR_LIMIT:
    return analytic_resource_pressure(
        from, to, static_cast<int32_t>(FieldChannel::TOUR_LIMIT));
  case ResourceOperator::LEGACY_BACKHAUL_ORDER:
    return analytic_resource_pressure(
        from, to, static_cast<int32_t>(FieldChannel::BACKHAUL_ORDER));
  case ResourceOperator::LEGACY_PICKUP_DELIVERY:
    return analytic_resource_pressure(
        from, to, static_cast<int32_t>(FieldChannel::PICKUP_DELIVERY));
  case ResourceOperator::LEGACY_PRIZE_QUOTA:
    return analytic_resource_pressure(
        from, to, static_cast<int32_t>(FieldChannel::PRIZE_QUOTA));
  case ResourceOperator::AFFINE_ACCUMULATOR: {
    double delta = 0.0;
    if (spec.edge_uses_distance)
      delta += spec.edge_coefficient * problem_.dist(from, to);
    if (!spec.edge_values.empty())
      delta += spec.edge_coefficient *
               spec.edge_values[static_cast<size_t>(from) *
                                    problem_.node_count +
                                to];
    if (!spec.node_values.empty())
      delta += spec.node_coefficient * spec.node_values[to];
    // Pressure is the bound-worsening magnitude in physical units. Lower-bound
    // resources (battery remaining) worsen on negative extension; upper-bound
    // accumulators worsen on positive extension.
    if (std::isfinite(spec.lower) && !std::isfinite(spec.upper))
      return static_cast<float>(std::max(-delta, 0.0));
    return static_cast<float>(std::max(delta, 0.0));
  }
  }
  return 0.0f;
}

float RoutingDecoder::candidate_resource_relevance(
    int32_t from, int32_t to, int32_t resource_index) const {
  const ResourceSpec &spec = resource(resource_index);
  if (!spec.active)
    return 0.0f;
  float relevance = runtime_resource_pressure(from, to, resource_index) /
                    std::max(runtime_resource_scale(resource_index), EPS);
  const bool resets = (to < problem_.depot_count && spec.reset_at_depot) ||
                      (!spec.reset_nodes.empty() && spec.reset_nodes[to]);
  if (resets)
    relevance = std::max(relevance, 1.0f);
  return relevance;
}

void RoutingDecoder::validate_guidance(const float *edge_field,
                                       const float *edge_additive,
                                       const float *multipliers,
                                       const float *coupler_weights,
                                       const float *coupler_bias,
                                       const float *edge_risk,
                                       float risk_penalty) const {
  if (search_config_.classical_behavior) {
    if (edge_field != nullptr || edge_additive != nullptr ||
        multipliers != nullptr || coupler_weights != nullptr ||
        coupler_bias != nullptr || edge_risk != nullptr ||
        risk_penalty != 0.0f) {
      throw std::invalid_argument(
          "classical_behavior does not accept a learned field");
    }
    return;
  }
  if (edge_field == nullptr) {
    throw std::invalid_argument(
        "edge_field is required when classical_behavior is false");
  }
  const size_t value_count =
      static_cast<size_t>(edge_count()) * resource_count();
  for (size_t index = 0; index < value_count; ++index) {
    if (!std::isfinite(edge_field[index]) || edge_field[index] < 0.0f) {
      throw std::invalid_argument(
          "edge_field residuals must be finite and non-negative");
    }
  }
  if (edge_additive != nullptr) {
    for (size_t index = 0; index < value_count; ++index) {
      if (!std::isfinite(edge_additive[index])) {
        throw std::invalid_argument(
            "edge additive corrections must be finite");
      }
    }
  }
  if (multipliers != nullptr) {
    for (int32_t channel = 0; channel < multiplier_count(); ++channel) {
      if (!std::isfinite(multipliers[channel]) || multipliers[channel] < 0.0f) {
        throw std::invalid_argument(
            "field multipliers must be finite and non-negative");
      }
    }
  }
  if (coupler_weights != nullptr) {
    const size_t count =
        static_cast<size_t>(multiplier_count()) * live_state_feature_count();
    for (size_t index = 0; index < count; ++index) {
      if (!std::isfinite(coupler_weights[index])) {
        throw std::invalid_argument("coupler weights must be finite");
      }
    }
  }
  if (coupler_bias != nullptr) {
    for (int32_t channel = 0; channel < multiplier_count(); ++channel) {
      if (!std::isfinite(coupler_bias[channel])) {
        throw std::invalid_argument("coupler bias must be finite");
      }
    }
  }
  if (!std::isfinite(risk_penalty) || risk_penalty < 0.0f) {
    throw std::invalid_argument(
        "feasibility risk penalty must be finite and non-negative");
  }
  if (edge_risk != nullptr) {
    for (int32_t edge = 0; edge < edge_count(); ++edge) {
      if (!std::isfinite(edge_risk[edge]) || edge_risk[edge] < 0.0f ||
          edge_risk[edge] > 1.0f) {
        throw std::invalid_argument(
            "edge feasibility risk must be normalized to [0, 1]");
      }
    }
  }
}

std::vector<float> RoutingDecoder::live_state_features(const State &state) const {
  const auto unit = [](double value) {
    return static_cast<float>(std::clamp(value, 0.0, 1.0));
  };
  const double route_scale =
      std::isfinite(problem_.route_limit)
          ? std::max<double>(problem_.route_limit, EPS)
          : distance_scale_;
  const double tour_scale =
      std::isfinite(problem_.tour_limit)
          ? std::max<double>(problem_.tour_limit, EPS)
          : time_scale_;
  std::vector<float> result(resource_count(), 0.0f);
  for (int32_t index = 0; index < resource_count(); ++index) {
    if (!resource(index).active)
      continue;
    switch (resource(index).op) {
    case ResourceOperator::LEGACY_CAPACITY:
      result[index] = unit(1.0 - state.load / std::max(problem_.capacity, EPS));
      break;
    case ResourceOperator::LEGACY_TIME_WINDOW:
      result[index] = unit(state.current_time / time_scale_);
      break;
    case ResourceOperator::LEGACY_ROUTE_LIMIT:
      result[index] = unit(state.route_distance / route_scale);
      break;
    case ResourceOperator::LEGACY_TOUR_LIMIT:
      result[index] = unit(state.route_distance / tour_scale);
      break;
    case ResourceOperator::LEGACY_BACKHAUL_ORDER:
      result[index] = state.route_has_backhaul ? 1.0f : 0.0f;
      break;
    case ResourceOperator::LEGACY_PICKUP_DELIVERY:
      result[index] = unit(static_cast<double>(state.open_pickups) /
                           std::max(pair_count_, 1));
      break;
    case ResourceOperator::LEGACY_PRIZE_QUOTA:
      result[index] = unit(1.0 - state.collected_prize /
                                     std::max(problem_.prize_quota, EPS));
      break;
    case ResourceOperator::AFFINE_ACCUMULATOR:
      result[index] = resource_state_feature(state, index);
      break;
    }
  }
  return result;
}

float RoutingDecoder::resource_state_feature(const State &state,
                                             int32_t resource_index) const {
  const ResourceSpec &spec = resource(resource_index);
  const float value = state.algebra_state[resource_index];
  if (std::isfinite(spec.lower) && std::isfinite(spec.upper))
    return std::clamp((value - spec.lower) /
                          std::max(spec.upper - spec.lower, EPS),
                      0.0f, 1.0f);
  if (std::isfinite(spec.lower))
    return std::clamp(1.0f - (value - spec.lower) / runtime_resource_scale(resource_index),
                      0.0f, 1.0f);
  if (std::isfinite(spec.upper))
    return std::clamp(value / runtime_resource_scale(resource_index), 0.0f, 1.0f);
  return std::clamp(std::abs(value) / runtime_resource_scale(resource_index),
                    0.0f, 1.0f);
}

std::vector<float> RoutingDecoder::incumbent_state_features(int32_t current) const {
  std::vector<float> result(resource_count(), 0.0f);
  if (current < 0 || current >= problem_.node_count ||
      incumbent_live_state_.size() !=
          static_cast<size_t>(problem_.node_count) * resource_count())
    return result;
  std::copy_n(incumbent_live_state_.data() +
                  static_cast<size_t>(current) * resource_count(),
              resource_count(), result.begin());
  return result;
}

bool RoutingDecoder::incumbent_prefix_state(int32_t current,
                                            State &state) const {
  if (incumbent_route_.empty())
    return false;
  state = initial_state(incumbent_route_.front());
  if (state.current == current)
    return true;
  for (size_t index = 1; index < incumbent_route_.size(); ++index) {
    std::string error;
    if (!transition(state, incumbent_route_[index], error))
      return false;
    if (state.current == current)
      return true;
  }
  return false;
}

double RoutingDecoder::coupled_multiplier(
    int32_t channel, const float *multipliers, const float *coupler_weights,
    const float *coupler_bias, const float *live_state) const {
  const double base = multipliers == nullptr ? 1.0 : multipliers[channel];
  if (live_state == nullptr ||
      (coupler_weights == nullptr && coupler_bias == nullptr)) {
    return base;
  }
  double logit = coupler_bias == nullptr ? 0.0 : coupler_bias[channel];
  if (coupler_weights != nullptr) {
    const float *weights = coupler_weights +
                           static_cast<size_t>(channel) *
                               live_state_feature_count();
    for (int32_t feature = 0; feature < live_state_feature_count(); ++feature)
      logit += weights[feature] * live_state[feature];
  }
  const double modulation =
      logit >= 0.0 ? 2.0 / (1.0 + std::exp(-logit))
                   : 2.0 * std::exp(logit) / (1.0 + std::exp(logit));
  return base * modulation;
}

void RoutingDecoder::record_decision(RolloutTrace *trace, int32_t current,
                                     const std::vector<int32_t> &valid_indices,
                                     int32_t chosen_index, bool stochastic,
                                     float log_probability,
                                     const float *live_state) const {
  if (trace == nullptr || chosen_index < 0 || live_state == nullptr)
    return;
  trace->current_nodes.push_back(current);
  trace->valid_indices.insert(trace->valid_indices.end(), valid_indices.begin(),
                              valid_indices.end());
  trace->valid_offsets.push_back(
      static_cast<int32_t>(trace->valid_indices.size()));
  trace->chosen_indices.push_back(chosen_index);
  trace->stochastic.push_back(stochastic ? 1 : 0);
  trace->log_probabilities.push_back(log_probability);
  trace->live_state.insert(trace->live_state.end(), live_state,
                           live_state + live_state_feature_count());
}

void RoutingDecoder::record_feasibility_labels(RolloutTrace *trace,
                                               State &state) const {
  if (trace == nullptr)
    return;
  for (int32_t edge = edge_offsets_[state.current];
       edge < edge_offsets_[state.current + 1]; ++edge) {
    const int32_t node = edge_to_[edge];
    trace->feasibility_edges.push_back(edge);
    trace->feasibility_risk_labels.push_back(
        legal_node(state, node) ? feasibility_risk_label(state, node) : 1.0f);
  }
}

double RoutingDecoder::field_score(int32_t from, int32_t to, int32_t edge,
                                   const float *edge_field,
                                   const float *edge_additive,
                                   const float *multipliers,
                                   const float *coupler_weights,
                                   const float *coupler_bias,
                                   const float *live_state) const {
  if (search_config_.classical_behavior) {
    return 0.0;
  }
  double result = 0.0;
  for (int32_t channel = 0; channel < resource_count(); ++channel) {
    if (!resource(channel).active)
      continue;
    const double multiplier = coupled_multiplier(
        channel, multipliers, coupler_weights, coupler_bias, live_state);
    result += multiplier * resource_field_value(
                               from, to, edge, channel, edge_field,
                               edge_additive);
  }
  return result;
}

double RoutingDecoder::resource_field_value(
    int32_t from, int32_t to, int32_t edge, int32_t channel,
    const float *edge_field, const float *edge_additive) const {
  // edge_field is the learned per-edge resource field (already scaled to the
  // resource's units by the model). Analytic pressure is no longer a
  // multiplicative gate -- it is exposed to the GNN as an input feature -- so
  // it only serves as the fallback field on off-graph reachability edges.
  const double field = edge >= 0 && edge_field != nullptr
                           ? edge_field[static_cast<size_t>(edge) *
                                            resource_count() +
                                        channel]
                           : runtime_resource_pressure(from, to, channel);
  const double additive =
      edge >= 0 && edge_additive != nullptr
          ? edge_additive[static_cast<size_t>(edge) * resource_count() +
                          channel]
          : 0.0;
  return std::max(field + additive, 0.0);
}

double RoutingDecoder::edge_energy(int32_t from, int32_t to, int32_t edge,
                                   const float *edge_field,
                                   const float *edge_additive,
                                   const float *multipliers,
                                   const float *coupler_weights,
                                   const float *coupler_bias,
                                   const float *live_state,
                                   const float *edge_risk,
                                   float risk_penalty) const {
  const double risk = edge >= 0 && edge_risk != nullptr ? edge_risk[edge] : 0.0;
  // The objective enters through a learned, state-conditioned weight w_obj
  // (multiplier slot OBJECTIVE_MULTIPLIER) rather than a hard unit coefficient.
  // With no guidance the coupler returns 1.0, recovering the plain objective.
  const double objective_weight = coupled_multiplier(
      objective_multiplier(), multipliers, coupler_weights, coupler_bias,
      live_state);
  return objective_weight * objective_edge_cost(from, to) +
         field_score(from, to, edge, edge_field, edge_additive, multipliers,
                     coupler_weights, coupler_bias, live_state) +
         risk_penalty * risk;
}

void RoutingDecoder::build_candidate_graph(const std::vector<int32_t> &incumbent,
                                           std::vector<float> *edge_field,
                                           std::vector<float> *edge_additive,
                                           std::vector<float> *edge_risk) {
  const int32_t n = problem_.node_count;
  const int32_t k = std::min(candidate_config_.max_candidates, n - 1);

  const std::vector<int32_t> old_offsets = std::move(edge_offsets_);
  const std::vector<int32_t> old_to = std::move(edge_to_);
  std::vector<float> old_field;
  std::vector<float> old_additive;
  std::vector<float> old_risk;
  if (edge_field != nullptr)
    old_field.swap(*edge_field);
  if (edge_additive != nullptr)
    old_additive.swap(*edge_additive);
  if (edge_risk != nullptr)
    old_risk.swap(*edge_risk);

  // The graph topology is deliberately geometric only. Depot connectivity is
  // the sole overlay because a depot may be required to close/reset a route
  // even when it is not among a customer's nearest spatial neighbours.
  std::vector<std::vector<int32_t>> depot_edges(n);
  for (int32_t customer = problem_.depot_count; customer < n; ++customer) {
    for (int32_t depot = 0; depot < problem_.depot_count; ++depot) {
      depot_edges[customer].push_back(depot);
      depot_edges[depot].push_back(customer);
    }
  }

  std::unique_ptr<KDTree2D> kd_tree;
  if (!problem_.coordinates.empty()) {
    kd_tree = std::make_unique<KDTree2D>(problem_.coordinates);
  }

  // Effective per-resource candidate allocation, constant across source nodes.
  // The schema-derived candidate_resource_relevance is the deterministic,
  // variant-agnostic admission rule: when the learned quota policy has installed
  // fractions we honor them, otherwise we synthesize a uniform equal-share over
  // active resources (plus an implicit geometric slot, mirroring the learned
  // head's softmax structure). This keeps a newly declared resource covered with
  // no per-variant tuning. GEOMETRIC mode drops resource channels entirely.
  std::vector<float> effective_quotas;
  if (candidate_config_.candidate_mode == CandidateMode::SCHEMA) {
    if (!candidate_resource_quotas_.empty()) {
      effective_quotas = candidate_resource_quotas_;
    } else {
      int32_t active = 0;
      for (const ResourceSpec &spec : resources_)
        active += spec.active ? 1 : 0;
      if (active > 0) {
        effective_quotas.assign(resource_count(), 0.0f);
        const float share = 1.0f / static_cast<float>(active + 1);
        for (int32_t index = 0; index < resource_count(); ++index)
          if (resources_[index].active)
            effective_quotas[index] = share;
      }
    }
  }

  std::vector<std::vector<int32_t>> rows(n);
  std::vector<int32_t> included_at(n, -1);
  for (int32_t from = 0; from < n; ++from) {
    const auto add = [&](int32_t to) {
      if (to != from && to >= 0 && to < n && included_at[to] != from) {
        included_at[to] = from;
        rows[from].push_back(to);
        return true;
      }
      return false;
    };
    for (int32_t to : depot_edges[from]) {
      add(to);
    }

    // A depot must be able to start/reset a route at any customer. These rows
    // are the required depot overlay; customer rows are bounded by K plus any
    // depots that must be retained when the depot count itself exceeds K.
    if (from < problem_.depot_count) {
      std::sort(rows[from].begin(), rows[from].end());
      continue;
    }

    const int32_t target = std::max(k, static_cast<int32_t>(rows[from].size()));
    const int32_t allocatable = target - static_cast<int32_t>(rows[from].size());
    if (!effective_quotas.empty() && allocatable > 0) {
      for (int32_t resource_index = 0; resource_index < resource_count();
           ++resource_index) {
        const int32_t quota = std::min(
            allocatable,
            static_cast<int32_t>(std::floor(
                effective_quotas[resource_index] * allocatable)));
        if (quota <= 0 || !resource(resource_index).active)
          continue;
        std::vector<int32_t> ranked;
        ranked.reserve(n - 1);
        for (int32_t to = 0; to < n; ++to) {
          if (to != from && included_at[to] != from)
            ranked.push_back(to);
        }
        std::sort(ranked.begin(), ranked.end(), [&](int32_t lhs, int32_t rhs) {
          const float lhs_score =
              candidate_resource_relevance(from, lhs, resource_index);
          const float rhs_score =
              candidate_resource_relevance(from, rhs, resource_index);
          if (lhs_score != rhs_score)
            return lhs_score > rhs_score;
          const float lhs_distance = problem_.dist(from, lhs);
          const float rhs_distance = problem_.dist(from, rhs);
          return lhs_distance == rhs_distance ? lhs < rhs
                                              : lhs_distance < rhs_distance;
        });
        int32_t admitted = 0;
        for (int32_t to : ranked) {
          if (candidate_resource_relevance(from, to, resource_index) <= 0.0f)
            break;
          admitted += add(to) ? 1 : 0;
          if (admitted >= quota || static_cast<int32_t>(rows[from].size()) >= target)
            break;
        }
        if (static_cast<int32_t>(rows[from].size()) >= target)
          break;
      }
    }
    const int32_t query_count =
        std::min(n - 1, target + problem_.depot_count);
    const std::vector<int32_t> nearest =
        kd_tree ? kd_tree->nearest(from, query_count)
                : rank_by_distance(from, query_count);
    for (int32_t to : nearest) {
      if (static_cast<int32_t>(rows[from].size()) >= target)
        break;
      add(to);
    }
    std::sort(rows[from].begin(), rows[from].end());
  }

  edge_offsets_.assign(n + 1, 0);
  for (int32_t from = 0; from < n; ++from) {
    edge_offsets_[from + 1] =
        edge_offsets_[from] + static_cast<int32_t>(rows[from].size());
  }
  edge_to_.clear();
  edge_to_.reserve(edge_offsets_.back());
  for (const auto &row : rows) {
    edge_to_.insert(edge_to_.end(), row.begin(), row.end());
  }
  proximity_.resize(edge_to_.size());
  heuristic_.resize(edge_to_.size());
  resource_pressure_.assign(edge_to_.size() * resource_count(), 0.0f);
  resource_events_.assign(edge_to_.size() * resource_count(), 0.0f);
  objective_edge_costs_.assign(edge_to_.size(), 0.0f);
  if (edge_field != nullptr) {
    edge_field->assign(static_cast<size_t>(edge_to_.size()) * resource_count(),
                       1.0f);
  }
  if (edge_additive != nullptr) {
    edge_additive->assign(static_cast<size_t>(edge_to_.size()) *
                              resource_count(),
                          0.0f);
  }
  if (edge_risk != nullptr)
    edge_risk->assign(edge_to_.size(), 0.0f);
  for (int32_t from = 0; from < n; ++from) {
    int32_t old_edge =
        old_offsets.size() == static_cast<size_t>(n + 1)
            ? old_offsets[from]
            : 0;
    const int32_t old_end =
        old_offsets.size() == static_cast<size_t>(n + 1)
            ? old_offsets[from + 1]
            : 0;
    for (int32_t edge = edge_offsets_[from]; edge < edge_offsets_[from + 1];
         ++edge) {
      const int32_t to = edge_to_[edge];
      while (old_edge < old_end && old_to[old_edge] < to)
        ++old_edge;
      const bool preserved = old_edge < old_end && old_to[old_edge] == to;
      proximity_[edge] = classical_proximity(from, edge_to_[edge]);
      heuristic_[edge] = 1.0f / std::max(proximity_[edge], EPS);
      objective_edge_costs_[edge] = objective_edge_cost(from, to);
      for (int32_t channel = 0; channel < resource_count(); ++channel) {
        resource_pressure_[static_cast<size_t>(edge) * resource_count() +
                           channel] =
            runtime_resource_pressure(from, to, channel);
        const ResourceSpec &spec = resource(channel);
        const bool reset =
            (to < problem_.depot_count && spec.reset_at_depot) ||
            (!spec.reset_nodes.empty() && spec.reset_nodes[to]);
        resource_events_[static_cast<size_t>(edge) * resource_count() +
                         channel] = reset ? 1.0f : 0.0f;
      }
      if (edge_field != nullptr && preserved &&
          static_cast<size_t>(old_edge + 1) * resource_count() <=
              old_field.size()) {
        std::copy_n(old_field.data() +
                        static_cast<size_t>(old_edge) * resource_count(),
                    resource_count(),
                    edge_field->data() +
                        static_cast<size_t>(edge) * resource_count());
      }
      if (edge_additive != nullptr && preserved &&
          static_cast<size_t>(old_edge + 1) * resource_count() <=
              old_additive.size()) {
        std::copy_n(old_additive.data() +
                        static_cast<size_t>(old_edge) * resource_count(),
                    resource_count(),
                    edge_additive->data() +
                        static_cast<size_t>(edge) * resource_count());
      }
      if (edge_risk != nullptr && preserved &&
          static_cast<size_t>(old_edge) < old_risk.size())
        (*edge_risk)[edge] = old_risk[old_edge];
    }
  }
  incumbent_route_ = incumbent;
  build_model_features();
  ++graph_version_;
}

void RoutingDecoder::build_model_features() {
  const int32_t n = problem_.node_count;
  const auto unit = [](double value) {
    return static_cast<float>(std::clamp(value, 0.0, 1.0));
  };
  node_features_.assign(static_cast<size_t>(n) * NODE_FEATURE_COUNT, 0.0f);
  incumbent_live_state_.assign(
      static_cast<size_t>(n) * resource_count(), 0.0f);

  float min_x = 0.0f;
  float min_y = 0.0f;
  float coordinate_scale = 1.0f;
  if (!problem_.coordinates.empty()) {
    float max_x = problem_.coordinates[0];
    float max_y = problem_.coordinates[1];
    min_x = max_x;
    min_y = max_y;
    for (int32_t node = 0; node < n; ++node) {
      min_x = std::min(min_x, problem_.coordinates[2 * node]);
      max_x = std::max(max_x, problem_.coordinates[2 * node]);
      min_y = std::min(min_y, problem_.coordinates[2 * node + 1]);
      max_y = std::max(max_y, problem_.coordinates[2 * node + 1]);
    }
    coordinate_scale = std::max({max_x - min_x, max_y - min_y, EPS});
  }
  const float capacity_scale = std::max(problem_.capacity, EPS);
  const float route_scale =
      resource_scale(static_cast<int32_t>(FieldChannel::ROUTE_LIMIT));
  const float tour_scale =
      resource_scale(static_cast<int32_t>(FieldChannel::TOUR_LIMIT));
  int32_t pair_count = 0;
  for (int32_t node = 0; node < n; ++node) {
    pair_count += problem_.delivery_of_pickup[node] >= 0 ? 1 : 0;
    float *features = node_features_.data() +
                      static_cast<size_t>(node) * NODE_FEATURE_COUNT;
    if (!problem_.coordinates.empty()) {
      features[0] =
          unit((problem_.coordinates[2 * node] - min_x) / coordinate_scale);
      features[1] = unit((problem_.coordinates[2 * node + 1] - min_y) /
                         coordinate_scale);
    }
    features[2] = node < problem_.depot_count ? 1.0f : 0.0f;
    features[3] = unit(std::max(problem_.demand[node], 0.0f) / capacity_scale);
    features[4] = unit(std::max(-problem_.demand[node], 0.0f) / capacity_scale);
    features[5] = unit(problem_.prize[node] / prize_scale_);
    features[6] = unit(problem_.penalty[node] / penalty_scale_);
    features[7] = unit(problem_.tw_start[node] / time_scale_);
    features[8] = std::isfinite(problem_.tw_end[node])
                      ? unit(problem_.tw_end[node] / time_scale_)
                      : 1.0f;
    features[9] = unit(problem_.service_time[node] / time_scale_);
    features[10] = problem_.delivery_of_pickup[node] >= 0 ? 1.0f : 0.0f;
    features[11] = problem_.pickup_of_delivery[node] >= 0 ? 1.0f : 0.0f;
  }

  double cumulative_prize = 0.0;
  const auto scan_route = [&](const std::vector<int32_t> &nodes,
                              int32_t depot) {
    if (nodes.empty())
      return;
    bool has_linehaul = false;
    for (int32_t node : nodes)
      has_linehaul |= problem_.demand[node] > FEASIBILITY_EPS;
    float load = has_linehaul ? problem_.capacity : 0.0f;
    float time = 0.0f;
    float distance = 0.0f;
    int32_t open = 0;
    bool backhaul = false;
    int32_t previous = depot >= 0 ? depot : nodes.front();
    for (size_t index = 0; index < nodes.size(); ++index) {
      const int32_t node = nodes[index];
      if (index > 0 || depot >= 0) {
        const float travel = problem_.dist(previous, node);
        distance += travel;
        time = std::max(time + travel, problem_.tw_start[node]);
      }
      load -= problem_.demand[node];
      backhaul |= problem_.demand[node] < -FEASIBILITY_EPS;
      cumulative_prize += problem_.prize[node];
      if (problem_.delivery_of_pickup[node] >= 0)
        ++open;
      if (problem_.pickup_of_delivery[node] >= 0)
        --open;
      float *features = node_features_.data() +
                        static_cast<size_t>(node) * NODE_FEATURE_COUNT;
      features[12] = 1.0f;
      features[13] = nodes.size() == 1
                         ? 0.0f
                         : unit(static_cast<double>(index) /
                                static_cast<double>(nodes.size() - 1));
      features[14] = unit(load / capacity_scale);
      features[15] = unit(time / time_scale_);
      features[16] = std::isfinite(problem_.tw_end[node])
                         ? unit((problem_.tw_end[node] - time) / time_scale_)
                         : 1.0f;
      features[17] = unit(static_cast<double>(std::max(open, 0)) /
                          std::max(pair_count, 1));
      features[18] = unit(distance / distance_scale_);
      const float state_time = time + problem_.service_time[node];
      float *live = incumbent_live_state_.data() +
                    static_cast<size_t>(node) * resource_count();
      const auto set_legacy = [&](FieldChannel channel, float value) {
        const int32_t slot = legacy_resource_index(channel);
        if (slot >= 0)
          live[slot] = value;
      };
      set_legacy(FieldChannel::CAPACITY, 1.0f - features[14]);
      set_legacy(FieldChannel::TIME_WINDOW, unit(state_time / time_scale_));
      set_legacy(FieldChannel::ROUTE_LIMIT, unit(distance / route_scale));
      set_legacy(FieldChannel::TOUR_LIMIT, unit(distance / tour_scale));
      set_legacy(FieldChannel::BACKHAUL_ORDER, backhaul ? 1.0f : 0.0f);
      set_legacy(FieldChannel::PICKUP_DELIVERY, features[17]);
      set_legacy(FieldChannel::PRIZE_QUOTA,
                 unit(1.0 - cumulative_prize /
                                std::max<double>(problem_.prize_quota, EPS)));
      time = state_time;
      previous = node;
    }
    float backward = depot >= 0 && !problem_.open_route
                         ? problem_.dist(nodes.back(), depot)
                         : 0.0f;
    float suffix_load = 0.0f;
    float suffix_time = backward;
    float suffix_slack = 1.0f;
    int32_t suffix_open = 0;
    for (int32_t index = static_cast<int32_t>(nodes.size()) - 1; index >= 0;
         --index) {
      const int32_t node = nodes[index];
      float *features = node_features_.data() +
                        static_cast<size_t>(node) * NODE_FEATURE_COUNT;
      features[19] = unit(backward / distance_scale_);
      suffix_load += std::abs(problem_.demand[node]);
      suffix_time += problem_.service_time[node];
      suffix_slack = std::min(suffix_slack, features[16]);
      if (problem_.pickup_of_delivery[node] >= 0)
        ++suffix_open;
      if (problem_.delivery_of_pickup[node] >= 0)
        suffix_open = std::max(suffix_open - 1, 0);
      features[20] = unit(suffix_load / capacity_scale);
      features[21] = unit(suffix_time / time_scale_);
      features[22] = unit(suffix_slack);
      features[23] = unit(static_cast<double>(suffix_open) /
                          std::max(pair_count, 1));
      if (index > 0) {
        const float travel = problem_.dist(nodes[index - 1], node);
        backward += travel;
        suffix_time += travel;
      }
    }
  };

  if (!incumbent_route_.empty()) {
    if (problem_.depot_count == 0) {
      scan_route(incumbent_route_, -1);
    } else {
      size_t token = 0;
      while (token < incumbent_route_.size()) {
        if (incumbent_route_[token] >= problem_.depot_count) {
          ++token;
          continue;
        }
        const int32_t depot = incumbent_route_[token++];
        std::vector<int32_t> nodes;
        while (token < incumbent_route_.size() &&
               incumbent_route_[token] >= problem_.depot_count) {
          nodes.push_back(incumbent_route_[token++]);
        }
        scan_route(nodes, depot);
      }
    }
    State replay = initial_state(incumbent_route_.front());
    const auto store_replay = [&](int32_t node) {
      const std::vector<float> live = live_state_features(replay);
      if (node >= 0 && node < n && !live.empty())
        std::copy(live.begin(), live.end(),
                  incumbent_live_state_.begin() +
                      static_cast<size_t>(node) * resource_count());
    };
    store_replay(replay.current);
    for (size_t index = 1; index < incumbent_route_.size(); ++index) {
      std::string error;
      if (!transition(replay, incumbent_route_[index], error))
        break;
      store_replay(replay.current);
    }
  }

  // Reference counts make incremental removal robust when a depot edge occurs
  // in more than one route. Screening only needs the zero/nonzero predicate.
  std::vector<int32_t> incumbent_edges(edge_count(), 0);
  std::vector<uint8_t> reverse_incumbent_edges(edge_count(), 0);
  for (size_t index = 1; index < incumbent_route_.size(); ++index) {
    const int32_t from = incumbent_route_[index - 1];
    const int32_t to = incumbent_route_[index];
    const int32_t edge = find_edge(from, to);
    if (edge >= 0)
      incumbent_edges[edge] = 1;
    const int32_t reverse = find_edge(to, from);
    if (reverse >= 0)
      reverse_incumbent_edges[reverse] = 1;
  }

  resource_features_.assign(static_cast<size_t>(edge_count()) *
                                resource_count(),
                            0.0f);
  edge_features_.assign(static_cast<size_t>(edge_count()) * EDGE_FEATURE_COUNT,
                        0.0f);
  for (int32_t from = 0; from < n; ++from) {
    for (int32_t edge = edge_offsets_[from]; edge < edge_offsets_[from + 1];
         ++edge) {
      const int32_t to = edge_to_[edge];
      float *resources = resource_features_.data() +
                         static_cast<size_t>(edge) * resource_count();
      for (int32_t channel = 0; channel < resource_count(); ++channel) {
        resources[channel] =
            unit(resource_pressure_[static_cast<size_t>(edge) *
                                        resource_count() +
                                    channel] /
                 runtime_resource_scale(channel));
      }
      float *features = edge_features_.data() +
                        static_cast<size_t>(edge) * EDGE_FEATURE_COUNT;
      features[0] = unit(problem_.dist(from, to) / distance_scale_);
      for (int32_t legacy = 0; legacy < FIELD_CHANNEL_COUNT; ++legacy) {
        const int32_t slot = legacy_resource_index(
            static_cast<FieldChannel>(legacy));
        features[1 + legacy] = slot >= 0 ? resources[slot] : 0.0f;
      }
      features[8] = incumbent_edges[edge] ? 1.0f : 0.0f;
      features[9] = reverse_incumbent_edges[edge] ? 1.0f : 0.0f;
      // Structural openness lever: on an open route the arrival-at-depot leg is
      // genuinely free in the true objective, yet objective_edge_cost still
      // charges its travel (to avoid greedy fragmenting the route into one
      // customer per trip). Expose exactly that waived return distance on the
      // depot-incident edges so the field can learn to discount them. This is
      // problem-geometric (independent of any incumbent) and stays zero for
      // closed routes, where the return leg is legitimately charged.
      features[10] = problem_.open_route && to < problem_.depot_count
                         ? features[0]
                         : 0.0f;
    }
  }
}

int32_t RoutingDecoder::find_edge(int32_t from, int32_t to) const {
  const auto begin = edge_to_.begin() + edge_offsets_[from];
  const auto end = edge_to_.begin() + edge_offsets_[from + 1];
  const auto found = std::lower_bound(begin, end, to);
  if (found == end || *found != to) {
    return -1;
  }
  return static_cast<int32_t>(found - edge_to_.begin());
}

RoutingDecoder::State RoutingDecoder::initial_state(int32_t start_node) const {
  State state;
  state.visited.assign(problem_.node_count, 0);
  for (int32_t node = problem_.depot_count; node < problem_.node_count;
       ++node) {
    state.unvisited_linehauls +=
        problem_.demand[node] > FEASIBILITY_EPS ? 1 : 0;
    state.unvisited_backhauls +=
        problem_.demand[node] < -FEASIBILITY_EPS ? 1 : 0;
  }
  state.current = start_node;
  state.start_node = start_node;
  state.route.push_back(start_node);
  state.load = problem_.capacity;
  state.algebra_state.resize(resource_count(), 0.0f);
  for (int32_t index = 0; index < resource_count(); ++index) {
    if (!resource(index).is_legacy())
      state.algebra_state[index] = resource(index).initial;
  }

  if (problem_.depot_count == 0) {
    if (start_node < 0 || start_node >= problem_.node_count) {
      throw std::invalid_argument("invalid start node");
    }
    state.visited[start_node] = 1;
    state.visited_customers = 1;
    state.unvisited_linehauls -=
        problem_.demand[start_node] > FEASIBILITY_EPS ? 1 : 0;
    state.unvisited_backhauls -=
        problem_.demand[start_node] < -FEASIBILITY_EPS ? 1 : 0;
    state.collected_prize = problem_.prize[start_node];
  } else {
    if (start_node < 0 || start_node >= problem_.depot_count) {
      throw std::invalid_argument("a depot problem must start at a depot");
    }
    state.route_depot = start_node;
    state.at_depot = true;
    state.load = depot_reload(state);
  }
  return state;
}

bool RoutingDecoder::algebra_transition_feasible(const State &state,
                                                  int32_t next,
                                                  int32_t resource_index,
                                                  float *next_value) const {
  const ResourceSpec &spec = resource(resource_index);
  if (spec.is_legacy())
    return true;
  float value = state.algebra_state[resource_index];
  const bool depot = next < problem_.depot_count;
  const bool event_reset = !spec.reset_nodes.empty() && spec.reset_nodes[next];
  if (spec.edge_uses_distance)
    value += spec.edge_coefficient * problem_.dist(state.current, next);
  if (!spec.edge_values.empty()) {
    value += spec.edge_coefficient *
             spec.edge_values[static_cast<size_t>(state.current) *
                                  problem_.node_count +
                              next];
  }
  if (!spec.node_values.empty())
    value += spec.node_coefficient * spec.node_values[next];
  const bool check = spec.bound_check == BoundCheck::TRANSITION ||
                     (depot && spec.bound_check == BoundCheck::ROUTE_END);
  const bool feasible = !check || (value >= spec.lower - FEASIBILITY_EPS &&
                                   value <= spec.upper + FEASIBILITY_EPS);
  if ((depot && spec.reset_at_depot) || event_reset)
    value = spec.reset_value;
  else if (depot && spec.scope == ResourceScope::ROUTE)
    value = spec.initial;
  if (next_value != nullptr)
    *next_value = value;
  return feasible;
}

float RoutingDecoder::depot_reload(const State &state) const {
  if (!problem_.has(CAPACITY)) {
    return problem_.capacity;
  }
  if (state.unvisited_linehauls > 0)
    return problem_.capacity;
  return state.unvisited_backhauls > 0 ? 0.0f : problem_.capacity;
}

bool RoutingDecoder::legal_node(const State &state, int32_t node) const {
  const int32_t depots = problem_.depot_count;
  if (node < 0 || node >= problem_.node_count)
    return false;
  if (node >= depots) {
    if (state.visited[node])
      return false;
    const int32_t pickup = problem_.pickup_of_delivery[node];
    if (problem_.has(PICKUP_DELIVERY) && pickup >= 0 &&
        !state.visited[pickup])
      return false;

    const float node_demand = problem_.demand[node];
    if (problem_.has(CAPACITY)) {
      const float next_load = state.load - node_demand;
      if (next_load < -FEASIBILITY_EPS ||
          next_load > problem_.capacity + FEASIBILITY_EPS)
        return false;
    }
    if (problem_.has(BACKHAUL_ORDER) && state.route_has_backhaul &&
        node_demand > FEASIBILITY_EPS)
      return false;

    const float edge = problem_.dist(state.current, node);
    const float next_route_distance = state.route_distance + edge;
    if (problem_.has(ROUTE_LIMIT)) {
      float required = next_route_distance;
      if (!problem_.open_route) {
        required += problem_.dist(node, state.route_depot);
      }
      if (required > problem_.route_limit + FEASIBILITY_EPS)
        return false;
    }
    if (problem_.has(TOUR_LIMIT)) {
      const float required =
          next_route_distance + problem_.dist(node, state.route_depot);
      if (required > problem_.tour_limit + FEASIBILITY_EPS)
        return false;
    }
    if (problem_.has(TIME_WINDOWS)) {
      const float arrival =
          std::max(state.current_time + edge, problem_.tw_start[node]);
      if (arrival > problem_.tw_end[node] + FEASIBILITY_EPS)
        return false;
      if (!problem_.open_route) {
        const float return_time = arrival + problem_.service_time[node] +
                                  problem_.dist(node, state.route_depot);
        if (return_time >
            problem_.tw_end[state.route_depot] + FEASIBILITY_EPS)
          return false;
      }
    }
    for (int32_t index = 0; index < resource_count(); ++index) {
      if (!algebra_transition_feasible(state, node, index))
        return false;
    }
    return true;
  }

  if (depots == 0 || state.at_depot)
    return false;

  bool depot_allowed = problem_.multi_route || !problem_.has(VISIT_ALL);
  if (problem_.has(PICKUP_DELIVERY) && state.open_pickups != 0)
    depot_allowed = false;
  if (problem_.has(PRIZE_QUOTA) &&
      state.collected_prize + FEASIBILITY_EPS < problem_.prize_quota &&
      state.visited_customers < problem_.customer_count())
    depot_allowed = false;
  for (int32_t index = 0; depot_allowed && index < resource_count(); ++index) {
    if (!algebra_transition_feasible(state, node, index))
      depot_allowed = false;
  }
  return depot_allowed;
}

std::vector<uint8_t> RoutingDecoder::legal_mask(const State &state) const {
  std::vector<uint8_t> legal(problem_.node_count, 0);
  for (int32_t node = 0; node < problem_.node_count; ++node)
    legal[node] = legal_node(state, node) ? 1 : 0;
  return legal;
}

bool RoutingDecoder::transition(State &state, int32_t next,
                            std::string &error) const {
  if (next < 0 || next >= problem_.node_count) {
    error = "node index is out of range";
    return false;
  }
  if (!legal_node(state, next)) {
    error = "route contains an infeasible transition to node " +
            std::to_string(next);
    return false;
  }
  if (find_edge(state.current, next) < 0) {
    ++state.off_graph_edges;
  }
  std::vector<float> next_algebra = state.algebra_state;
  for (int32_t index = 0; index < resource_count(); ++index) {
    if (!resource(index).is_legacy()) {
      (void)algebra_transition_feasible(state, next, index,
                                        &next_algebra[index]);
    }
  }

  if (next < problem_.depot_count) {
    if (!problem_.open_route) {
      state.distance += problem_.dist(state.current, state.route_depot);
    }
    state.route.push_back(next);
    state.current = next;
    state.route_depot = next;
    state.at_depot = true;
    state.route_has_backhaul = false;
    state.route_distance = 0.0f;
    state.current_time = 0.0f;
    state.load = depot_reload(state);
    state.algebra_state = std::move(next_algebra);
    return true;
  }

  const float edge = problem_.dist(state.current, next);
  state.distance += edge;
  state.route_distance += edge;
  state.current_time =
      std::max(state.current_time + edge, problem_.tw_start[next]) +
      problem_.service_time[next];
  state.load -= problem_.demand[next];
  if (problem_.demand[next] < -FEASIBILITY_EPS) {
    state.route_has_backhaul = true;
  }
  if (problem_.delivery_of_pickup[next] >= 0) {
    ++state.open_pickups;
  }
  if (problem_.pickup_of_delivery[next] >= 0) {
    --state.open_pickups;
  }
  state.current = next;
  state.at_depot = false;
  state.visited[next] = 1;
  state.unvisited_linehauls -=
      problem_.demand[next] > FEASIBILITY_EPS ? 1 : 0;
  state.unvisited_backhauls -=
      problem_.demand[next] < -FEASIBILITY_EPS ? 1 : 0;
  ++state.visited_customers;
  state.collected_prize += problem_.prize[next];
  state.algebra_state = std::move(next_algebra);
  state.route.push_back(next);
  return true;
}

bool RoutingDecoder::has_feasible_lookahead(State &state,
                                            int32_t depth) const {
  if (complete(state) || depth <= 0)
    return true;
  std::vector<int32_t> candidates;
  candidates.reserve(edge_offsets_[state.current + 1] -
                     edge_offsets_[state.current]);
  for (int32_t edge = edge_offsets_[state.current];
       edge < edge_offsets_[state.current + 1]; ++edge) {
    if (legal_node(state, edge_to_[edge]))
      candidates.push_back(edge_to_[edge]);
  }
  if (candidates.empty()) {
    for (int32_t node = 0; node < problem_.node_count; ++node) {
      if (legal_node(state, node))
        candidates.push_back(node);
    }
  }
  for (int32_t node : candidates) {
    if (feasible_after_lookahead_transition(state, node, depth - 1)) {
      return true;
    }
  }
  return false;
}

bool RoutingDecoder::feasible_after_lookahead_transition(
    State &state, int32_t next, int32_t depth) const {
  const size_t route_size = state.route.size();
  const uint8_t was_visited = state.visited[next];
  const int32_t current = state.current;
  const int32_t route_depot = state.route_depot;
  const int32_t visited_customers = state.visited_customers;
  const int32_t open_pickups = state.open_pickups;
  const int32_t unvisited_linehauls = state.unvisited_linehauls;
  const int32_t unvisited_backhauls = state.unvisited_backhauls;
  const bool at_depot = state.at_depot;
  const bool route_has_backhaul = state.route_has_backhaul;
  const float load = state.load;
  const float route_distance = state.route_distance;
  const float current_time = state.current_time;
  const float distance = state.distance;
  const float collected_prize = state.collected_prize;
  const int32_t off_graph_edges = state.off_graph_edges;
  const std::vector<float> algebra_state = state.algebra_state;

  std::string error;
  const bool transitioned = transition(state, next, error);
  const bool feasible = transitioned && has_feasible_lookahead(state, depth);

  state.route.resize(route_size);
  state.visited[next] = was_visited;
  state.current = current;
  state.route_depot = route_depot;
  state.visited_customers = visited_customers;
  state.open_pickups = open_pickups;
  state.unvisited_linehauls = unvisited_linehauls;
  state.unvisited_backhauls = unvisited_backhauls;
  state.at_depot = at_depot;
  state.route_has_backhaul = route_has_backhaul;
  state.load = load;
  state.route_distance = route_distance;
  state.current_time = current_time;
  state.distance = distance;
  state.collected_prize = collected_prize;
  state.off_graph_edges = off_graph_edges;
  state.algebra_state = algebra_state;
  return feasible;
}

float RoutingDecoder::feasibility_risk_label(State &state,
                                             int32_t next) const {
  return feasible_after_lookahead_transition(
             state, next, search_config_.feasibility_lookahead_depth)
             ? 0.0f
             : 1.0f;
}

bool RoutingDecoder::complete(const State &state) const {
  if (problem_.depot_count == 0) {
    return state.visited_customers == problem_.customer_count();
  }
  if (problem_.has(VISIT_ALL)) {
    if (state.visited_customers != problem_.customer_count()) {
      return false;
    }
    return problem_.multi_route ? state.at_depot : true;
  }
  return state.route.size() > 1 && state.at_depot;
}

Solution RoutingDecoder::finish(State state) const {
  Solution solution;
  solution.route = state.route;
  if (!complete(state)) {
    solution.error = "route ended before satisfying the completion condition";
    return solution;
  }
  if (problem_.has(PICKUP_DELIVERY) && state.open_pickups != 0) {
    solution.error = "route ended with an undelivered pickup";
    return solution;
  }
  if (!problem_.open_route && !state.at_depot) {
    const int32_t end =
        problem_.depot_count == 0 ? state.start_node : state.route_depot;
    for (int32_t index = 0; index < resource_count(); ++index) {
      if (resource(index).is_legacy())
        continue;
      float value = state.algebra_state[index];
      if (!algebra_transition_feasible(state, end, index, &value)) {
        solution.error = "closing resource bound failed: " +
                         resource(index).name;
        return solution;
      }
      state.algebra_state[index] = value;
    }
  }
  for (int32_t index = 0; index < resource_count(); ++index) {
    const ResourceSpec &spec = resource(index);
    if (!spec.is_legacy() &&
        (spec.bound_check == BoundCheck::SOLUTION_END ||
         spec.bound_check == BoundCheck::ROUTE_END)) {
      const float value = state.algebra_state[index];
      if (value < spec.lower - FEASIBILITY_EPS ||
          value > spec.upper + FEASIBILITY_EPS) {
        solution.error = "terminal resource bound failed: " + spec.name;
        return solution;
      }
    }
  }

  if (problem_.has(VISIT_ALL) && !problem_.multi_route && !state.at_depot) {
    const int32_t end =
        problem_.depot_count == 0 ? state.start_node : state.route_depot;
    if (!problem_.open_route) {
      state.distance += problem_.dist(state.current, end);
      if (find_edge(state.current, end) < 0) {
        ++state.off_graph_edges;
      }
    }
  }

  for (int32_t node = problem_.depot_count; node < problem_.node_count;
       ++node) {
    if (!state.visited[node]) {
      solution.missed_penalty += problem_.penalty[node];
    }
  }
  solution.distance = state.distance;
  solution.collected_prize = state.collected_prize;
  solution.off_graph_edges = state.off_graph_edges;
  switch (problem_.objective) {
  case Objective::MIN_DISTANCE:
    solution.objective = solution.distance;
    break;
  case Objective::MAX_PRIZE:
    solution.objective = solution.collected_prize;
    break;
  case Objective::MIN_DISTANCE_PLUS_PENALTY:
    solution.objective = solution.distance + solution.missed_penalty;
    break;
  }
  solution.feasible = std::isfinite(solution.objective);
  if (!solution.feasible) {
    solution.error = "route objective is not finite";
  } else {
    solution.raw_objective = solution.objective;
  }
  return solution;
}

int32_t RoutingDecoder::select_next(State &state,
                                    std::mt19937_64 &rng,
                                    const float *edge_field,
                                    const float *edge_additive,
                                    const float *multipliers,
                                    const float *coupler_weights,
                                    const float *coupler_bias,
                                    const float *edge_risk,
                                    float risk_penalty, RolloutTrace *trace,
                                    bool greedy) const {
  struct Choice {
    int32_t node;
    int32_t edge;
    int32_t local_index;
  };
  std::vector<Choice> pool;
  pool.reserve(edge_offsets_[state.current + 1] - edge_offsets_[state.current]);
  for (int32_t edge = edge_offsets_[state.current];
       edge < edge_offsets_[state.current + 1]; ++edge) {
    const int32_t node = edge_to_[edge];
    if (legal_node(state, node)) {
      pool.push_back({node, edge, edge - edge_offsets_[state.current]});
    }
  }
  // Sparse reachability repair. It is deliberately used only when every
  // stored candidate is masked, and is reported in Solution::off_graph_edges.
  if (pool.empty()) {
    for (int32_t node = 0; node < problem_.node_count; ++node) {
      if (legal_node(state, node)) {
        pool.push_back({node, -1, -1});
      }
    }
  }
  if (pool.empty()) {
    return -1;
  }
  // Anti-fragmentation guard for open routes. objective_edge_cost() charges the
  // return leg 0 for open routes (matching the true objective), which would
  // otherwise make closing the route the cheapest move at every step and
  // shatter open routes into one customer each. While any customer can still
  // legally extend the current route, drop the depot options so routes fill up;
  // a depot return stays available only when no customer continuation is legal
  // (a forced close), and the local search reshapes routes afterwards. This
  // replaces the old phantom return cost with a structural rule, so the
  // ranking energy no longer contains a distortion the non-negative field must
  // fight. Only the field/neutral energy path zeroes the return; the classical
  // proximity heuristic never did, so it neither fragments nor needs the guard.
  if (problem_.open_route && !search_config_.classical_behavior) {
    bool has_customer = false;
    for (const Choice &choice : pool) {
      if (choice.node >= problem_.depot_count) {
        has_customer = true;
        break;
      }
    }
    if (has_customer) {
      pool.erase(
          std::remove_if(pool.begin(), pool.end(),
                         [&](const Choice &choice) {
                           return choice.node < problem_.depot_count;
                         }),
          pool.end());
    }
  }
  const std::vector<float> live_state =
      live_state_features(state);
  record_feasibility_labels(trace, state);
  std::vector<int32_t> valid_indices;
  valid_indices.reserve(pool.size());
  for (const Choice &choice : pool) {
    if (choice.local_index >= 0) {
      valid_indices.push_back(choice.local_index);
    }
  }
  const auto selected = [&](size_t index, bool stochastic,
                            double log_probability) {
    record_decision(trace, state.current, valid_indices,
                    pool[index].local_index, stochastic,
                    static_cast<float>(log_probability), live_state.data());
    return pool[index].node;
  };
  if (pool.size() == 1) {
    return selected(0, false, 0.0);
  }

  std::vector<double> log_weights(pool.size());
  double maximum = -std::numeric_limits<double>::infinity();
  const uint32_t savings_flags =
      static_cast<uint32_t>(VISIT_ALL) | static_cast<uint32_t>(CAPACITY);
  const bool use_savings = problem_.multi_route && !problem_.open_route &&
                           problem_.has(CAPACITY) &&
                           (problem_.constraints & ~savings_flags) == 0;
  for (size_t index = 0; index < pool.size(); ++index) {
    const int32_t edge = pool[index].edge;
    double value = 0.0;
    if (search_config_.classical_behavior) {
      float eta = edge >= 0
                      ? heuristic_[edge]
                      : 1.0f / std::max(classical_proximity(state.current,
                                                            pool[index].node),
                                        EPS);
      if (use_savings) {
        const int32_t node = pool[index].node;
        if (node < problem_.depot_count) {
          eta = EPS;
        } else if (greedy || state.at_depot) {
          eta = 1.0f / std::max(problem_.dist(state.current, node), EPS);
        } else {
          eta = std::max(problem_.dist(state.current, state.route_depot) +
                             problem_.dist(state.route_depot, node) -
                             problem_.dist(state.current, node),
                         EPS);
        }
      }
      value += beta_ * std::log(std::max(eta, EPS));
    } else {
      const double energy = edge_energy(state.current, pool[index].node, edge,
                                        edge_field, edge_additive, multipliers,
                                        coupler_weights, coupler_bias,
                                        live_state.data(), edge_risk,
                                        risk_penalty);
      value -= beta_ * energy;
    }
    log_weights[index] = value;
    maximum = std::max(maximum, value);
  }

  double total = 0.0;
  size_t best_index = 0;
  for (size_t index = 1; index < log_weights.size(); ++index) {
    if (log_weights[index] > log_weights[best_index])
      best_index = index;
  }
  if (greedy) {
    return selected(best_index, false, 0.0);
  }
  for (double &value : log_weights) {
    value = std::exp(value - maximum);
    total += value;
  }
  if (!(total > 0.0) || !std::isfinite(total)) {
    std::uniform_int_distribution<size_t> choose(0, pool.size() - 1);
    return selected(choose(rng), true, -std::log(pool.size()));
  }
  std::uniform_real_distribution<double> choose(0.0, total);
  double threshold = choose(rng);
  for (size_t index = 0; index < pool.size(); ++index) {
    threshold -= log_weights[index];
    if (threshold <= 0.0) {
      return selected(index, true, std::log(log_weights[index] / total));
    }
  }
  return selected(pool.size() - 1, true,
                  std::log(log_weights.back() / total));
}

Solution RoutingDecoder::construct(uint64_t rollout_seed, const float *edge_field,
                                   const float *edge_additive,
                                   const float *multipliers,
                                   const float *coupler_weights,
                                   const float *coupler_bias,
                                   const float *edge_risk,
                                   float risk_penalty, RolloutTrace *trace,
                                   bool greedy) const {
  std::mt19937_64 rng(rollout_seed);
  int32_t start = 0;
  if (problem_.depot_count > 0) {
    start = static_cast<int32_t>(rollout_seed % problem_.depot_count);
  } else {
    start = static_cast<int32_t>(rollout_seed % problem_.node_count);
  }
  State state = initial_state(start);
  const int32_t max_steps = 3 * problem_.node_count + 8;

  for (int32_t step = 0; step < max_steps && !complete(state); ++step) {
    const int32_t next = select_next(
        state, rng, edge_field, edge_additive, multipliers,
        coupler_weights, coupler_bias, edge_risk, risk_penalty, trace, greedy);
    if (next < 0) {
      Solution failed;
      failed.route = state.route;
      failed.error = "no feasible node remains during construction";
      return failed;
    }
    std::string error;
    if (!transition(state, next, error)) {
      Solution failed;
      failed.route = state.route;
      failed.error = error;
      return failed;
    }
  }
  if (!complete(state)) {
    Solution failed;
    failed.route = state.route;
    failed.error = "construction exceeded its step bound";
    return failed;
  }
  return finish(std::move(state));
}

std::vector<RoutingDecoder::OrderedChoice>
RoutingDecoder::perturbation_order(int32_t current,
                                   const std::vector<uint8_t> &used,
                                   std::mt19937_64 &rng,
                                   const float *edge_field,
                                   const float *edge_additive,
                                   const float *multipliers,
                                   const float *coupler_weights,
                                   const float *coupler_bias,
                                   const float *edge_risk,
                                   float risk_penalty,
                                   bool greedy) const {
  struct RankedChoice {
    int32_t node;
    int32_t local_index;
    double log_weight;
    double score;
  };
  std::vector<RankedChoice> ranked;
  ranked.reserve(edge_offsets_[current + 1] - edge_offsets_[current]);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  const uint32_t savings_flags =
      static_cast<uint32_t>(VISIT_ALL) | static_cast<uint32_t>(CAPACITY);
  const bool use_savings = problem_.multi_route && !problem_.open_route &&
                           problem_.has(CAPACITY) &&
                           (problem_.constraints & ~savings_flags) == 0;
  int32_t route_depot = 0;
  if (use_savings) {
    for (int32_t depot = 1; depot < problem_.depot_count; ++depot) {
      if (problem_.dist(current, depot) < problem_.dist(current, route_depot))
        route_depot = depot;
    }
  }
  const std::vector<float> live_state =
      incumbent_state_features(current);
  for (int32_t edge = edge_offsets_[current]; edge < edge_offsets_[current + 1];
       ++edge) {
    const int32_t node = edge_to_[edge];
    if (node == current || (node >= problem_.depot_count && used[node]) ||
        (node < problem_.depot_count && !problem_.multi_route)) {
      continue;
    }
    double log_weight = 0.0;
    if (search_config_.classical_behavior) {
      float eta = heuristic_[edge];
      if (use_savings) {
        eta = node < problem_.depot_count
                  ? EPS
                  : std::max(problem_.dist(current, route_depot) +
                                 problem_.dist(route_depot, node) -
                                 problem_.dist(current, node),
                             EPS);
      }
      log_weight += beta_ * std::log(std::max(eta, EPS));
    } else {
      log_weight -= beta_ *
                    edge_energy(current, node, edge, edge_field,
                                edge_additive, multipliers, coupler_weights,
                                coupler_bias,
                                live_state.data(), edge_risk, risk_penalty);
    }
    // Gumbel-top-k gives a weighted order without replacement.
    const double draw = greedy
                            ? std::exp(-1.0)
                            : std::clamp(uniform(rng), 1.0e-12, 1.0 - 1.0e-12);
    ranked.push_back({node, edge - edge_offsets_[current], log_weight,
                      greedy ? log_weight
                             : log_weight - std::log(-std::log(draw))});
  }
  std::sort(ranked.begin(), ranked.end(),
            [](const RankedChoice &lhs, const RankedChoice &rhs) {
              return lhs.score > rhs.score;
            });
  std::vector<OrderedChoice> result;
  result.reserve(ranked.size());
  for (const RankedChoice &choice : ranked) {
    result.push_back({choice.node, choice.local_index, choice.log_weight});
  }
  return result;
}

std::vector<int32_t>
RoutingDecoder::changed_scope(const std::vector<int32_t> &source,
                          const std::vector<int32_t> &candidate,
                          int32_t *new_edge_count) const {
  const int32_t n = problem_.node_count;
  std::vector<int32_t> source_successor(n, -1);
  std::vector<int32_t> candidate_successor(n, -1);
  std::vector<int32_t> source_depot_predecessor(n, -1);
  std::vector<int32_t> candidate_depot_predecessor(n, -1);
  const auto collect = [&](const std::vector<int32_t> &route,
                           std::vector<int32_t> &successor,
                           std::vector<int32_t> &depot_predecessor) {
    const auto add = [&](int32_t from, int32_t to) {
      if (from < 0 || from >= n || to < 0 || to >= n)
        return;
      if (from < problem_.depot_count) {
        if (to >= problem_.depot_count)
          depot_predecessor[to] = from;
      } else {
        successor[from] = to;
      }
    };
    for (size_t index = 1; index < route.size(); ++index)
      add(route[index - 1], route[index]);
    if (!route.empty() && problem_.has(VISIT_ALL) && !problem_.multi_route &&
        !problem_.open_route) {
      add(route.back(), route.front());
    }
  };
  collect(source, source_successor, source_depot_predecessor);
  collect(candidate, candidate_successor, candidate_depot_predecessor);
  std::vector<uint8_t> touched(problem_.node_count, 0);
  int32_t added = 0;
  const auto mark = [&](int32_t node) {
    if (node >= 0 && node < problem_.node_count)
      touched[node] = 1;
  };
  for (int32_t node = 0; node < n; ++node) {
    if (source_successor[node] != candidate_successor[node]) {
      mark(node);
      mark(source_successor[node]);
      mark(candidate_successor[node]);
      added += candidate_successor[node] >= 0 ? 1 : 0;
    }
    if (source_depot_predecessor[node] !=
        candidate_depot_predecessor[node]) {
      mark(node);
      mark(source_depot_predecessor[node]);
      mark(candidate_depot_predecessor[node]);
      added += candidate_depot_predecessor[node] >= 0 ? 1 : 0;
    }
  }
  if (new_edge_count != nullptr) {
    *new_edge_count = added;
  }
  std::vector<int32_t> result;
  for (int32_t node = 0; node < problem_.node_count; ++node) {
    if (touched[node])
      result.push_back(node);
  }
  return result;
}

bool RoutingDecoder::reversal_safe() const { return reversal_safe_; }

Solution RoutingDecoder::scope_restricted_refine(
    Solution solution, const std::vector<int32_t> &initial_scope,
    const float *edge_field, const float *edge_additive,
    const float *multipliers,
    const float *coupler_weights, const float *coupler_bias,
    const float *edge_risk, float risk_penalty,
    RolloutTrace *trace) const {
  if (!search_config_.use_srr || !solution.feasible) {
    return solution;
  }
  const auto position_of = [](const std::vector<int32_t> &route,
                              int32_t node) -> int32_t {
    const auto found = std::find(route.begin(), route.end(), node);
    return found == route.end() ? -1
                                : static_cast<int32_t>(found - route.begin());
  };
  std::deque<int32_t> checklist;
  std::vector<uint8_t> in_queue(problem_.node_count, 0);
  std::vector<uint8_t> dont_look(problem_.node_count, 0);
  std::vector<int32_t> visits(problem_.node_count, 0);
  const auto enqueue = [&](int32_t node) {
    if (node >= problem_.depot_count && node < problem_.node_count &&
        !in_queue[node]) {
      checklist.push_back(node);
      in_queue[node] = 1;
    }
  };
  for (int32_t node : initial_scope)
    enqueue(node);

  const auto relocate =
      [&](const std::vector<int32_t> &route, int32_t segment_start,
          int32_t after, int32_t length, std::vector<int32_t> &trial) -> bool {
    const int32_t start = position_of(route, segment_start);
    const int32_t after_position = position_of(route, after);
    if (start < 0 || after_position < 0 || length <= 0)
      return false;
    int32_t actual = 0;
    while (actual < length &&
           start + actual < static_cast<int32_t>(route.size()) &&
           route[start + actual] >= problem_.depot_count) {
      ++actual;
    }
    if (actual == 0 ||
        (after_position >= start && after_position < start + actual)) {
      return false;
    }
    const std::vector<int32_t> segment(route.begin() + start,
                                       route.begin() + start + actual);
    trial = route;
    trial.erase(trial.begin() + start, trial.begin() + start + actual);
    const int32_t new_after = position_of(trial, after);
    if (new_after < 0)
      return false;
    trial.insert(trial.begin() + new_after + 1, segment.begin(), segment.end());
    return trial != route;
  };
  const auto relocate_before =
      [&](const std::vector<int32_t> &route, int32_t segment_start,
          int32_t before, int32_t length, std::vector<int32_t> &trial) -> bool {
    const int32_t start = position_of(route, segment_start);
    const int32_t before_position = position_of(route, before);
    if (start < 0 || before_position < 0 || length <= 0)
      return false;
    int32_t actual = 0;
    while (actual < length &&
           start + actual < static_cast<int32_t>(route.size()) &&
           route[start + actual] >= problem_.depot_count) {
      ++actual;
    }
    if (actual == 0 ||
        (before_position >= start && before_position < start + actual)) {
      return false;
    }
    const std::vector<int32_t> segment(route.begin() + start,
                                       route.begin() + start + actual);
    trial = route;
    trial.erase(trial.begin() + start, trial.begin() + start + actual);
    const int32_t new_before = position_of(trial, before);
    if (new_before < 0)
      return false;
    trial.insert(trial.begin() + new_before, segment.begin(), segment.end());
    return trial != route;
  };
  const auto same_route = [&](const std::vector<int32_t> &route, int32_t lhs,
                              int32_t rhs) {
    int32_t first = position_of(route, lhs);
    int32_t last = position_of(route, rhs);
    if (first < 0 || last < 0)
      return false;
    if (first > last)
      std::swap(first, last);
    for (int32_t index = first + 1; index < last; ++index) {
      if (route[index] < problem_.depot_count)
        return false;
    }
    return true;
  };
  const auto two_opt = [&](const std::vector<int32_t> &route, int32_t lhs,
                           int32_t rhs, std::vector<int32_t> &trial) -> bool {
    if (!reversal_safe() || !same_route(route, lhs, rhs))
      return false;
    int32_t first = position_of(route, lhs);
    int32_t last = position_of(route, rhs);
    if (first > last)
      std::swap(first, last);
    if (last <= first + 1)
      return false;
    trial = route;
    std::reverse(trial.begin() + first + 1, trial.begin() + last + 1);
    return trial != route;
  };
  const auto route_end = [&](const std::vector<int32_t> &route,
                             int32_t position) {
    int32_t end = position + 1;
    while (end < static_cast<int32_t>(route.size()) &&
           route[end] >= problem_.depot_count) {
      ++end;
    }
    return end;
  };
  const auto two_opt_star = [&](const std::vector<int32_t> &route, int32_t lhs,
                                int32_t rhs,
                                std::vector<int32_t> &trial) -> bool {
    if (!problem_.multi_route || same_route(route, lhs, rhs))
      return false;
    int32_t first = position_of(route, lhs);
    int32_t second = position_of(route, rhs);
    if (first < 0 || second < 0)
      return false;
    if (first > second)
      std::swap(first, second);
    const int32_t first_end = route_end(route, first);
    const int32_t second_end = route_end(route, second);
    if (first_end >= second || second_end > static_cast<int32_t>(route.size()))
      return false;
    const std::vector<int32_t> first_tail(route.begin() + first + 1,
                                          route.begin() + first_end);
    const std::vector<int32_t> second_tail(route.begin() + second + 1,
                                           route.begin() + second_end);
    trial = route;
    trial.erase(trial.begin() + second + 1, trial.begin() + second_end);
    trial.insert(trial.begin() + second + 1, first_tail.begin(),
                 first_tail.end());
    trial.erase(trial.begin() + first + 1, trial.begin() + first_end);
    trial.insert(trial.begin() + first + 1, second_tail.begin(),
                 second_tail.end());
    return trial != route;
  };
  const auto insert_after = [&](const std::vector<int32_t> &route, int32_t node,
                                int32_t after,
                                std::vector<int32_t> &trial) -> bool {
    if (node < problem_.depot_count || position_of(route, node) >= 0)
      return false;
    const int32_t after_position = position_of(route, after);
    if (after_position < 0)
      return false;
    trial = route;
    trial.insert(trial.begin() + after_position + 1, node);
    return true;
  };
  const auto insert_before = [&](const std::vector<int32_t> &route,
                                 int32_t node, int32_t before,
                                 std::vector<int32_t> &trial) -> bool {
    if (node < problem_.depot_count || position_of(route, node) >= 0)
      return false;
    const int32_t before_position = position_of(route, before);
    if (before_position < 0)
      return false;
    trial = route;
    trial.insert(trial.begin() + before_position, node);
    return true;
  };
  const auto exchange_nodes = [&](const std::vector<int32_t> &route,
                                  int32_t served, int32_t replacement,
                                  std::vector<int32_t> &trial) -> bool {
    const int32_t served_position = position_of(route, served);
    if (served_position < 0 || replacement < problem_.depot_count ||
        position_of(route, replacement) >= 0)
      return false;
    trial = route;
    trial[served_position] = replacement;
    return true;
  };
  const auto swap_nodes = [&](const std::vector<int32_t> &route, int32_t lhs,
                              int32_t rhs,
                              std::vector<int32_t> &trial) -> bool {
    const int32_t lhs_position = position_of(route, lhs);
    const int32_t rhs_position = position_of(route, rhs);
    if (lhs_position < 0 || rhs_position < 0 || lhs_position == rhs_position)
      return false;
    trial = route;
    std::swap(trial[lhs_position], trial[rhs_position]);
    return true;
  };
  const auto relocate_pair = [&](const std::vector<int32_t> &route,
                                 int32_t pair_node, int32_t after,
                                 std::vector<int32_t> &trial) -> bool {
    int32_t pickup = pair_node;
    int32_t delivery = problem_.delivery_of_pickup[pair_node];
    if (delivery < 0) {
      pickup = problem_.pickup_of_delivery[pair_node];
      delivery = pair_node;
    }
    if (pickup < problem_.depot_count || delivery < problem_.depot_count ||
        after == pickup || after == delivery)
      return false;
    const int32_t pickup_position = position_of(route, pickup);
    const int32_t delivery_position = position_of(route, delivery);
    if (pickup_position < 0 || delivery_position < 0 ||
        position_of(route, after) < 0)
      return false;
    trial = route;
    const int32_t later = std::max(pickup_position, delivery_position);
    const int32_t earlier = std::min(pickup_position, delivery_position);
    trial.erase(trial.begin() + later);
    trial.erase(trial.begin() + earlier);
    const int32_t after_position = position_of(trial, after);
    if (after_position < 0)
      return false;
    trial.insert(trial.begin() + after_position + 1, pickup);
    trial.insert(trial.begin() + after_position + 2, delivery);
    return trial != route;
  };
  const auto exchange_segments =
      [&](const std::vector<int32_t> &route, int32_t lhs, int32_t lhs_length,
          int32_t rhs, int32_t rhs_length,
          std::vector<int32_t> &trial) -> bool {
    int32_t lhs_position = position_of(route, lhs);
    int32_t rhs_position = position_of(route, rhs);
    if (lhs_position < 0 || rhs_position < 0 || lhs_position == rhs_position ||
        lhs_length <= 0 || rhs_length <= 0) {
      return false;
    }
    int32_t lhs_actual = 0;
    while (lhs_actual < lhs_length &&
           lhs_position + lhs_actual < static_cast<int32_t>(route.size()) &&
           route[lhs_position + lhs_actual] >= problem_.depot_count) {
      ++lhs_actual;
    }
    int32_t rhs_actual = 0;
    while (rhs_actual < rhs_length &&
           rhs_position + rhs_actual < static_cast<int32_t>(route.size()) &&
           route[rhs_position + rhs_actual] >= problem_.depot_count) {
      ++rhs_actual;
    }
    if (lhs_actual == 0 || rhs_actual == 0)
      return false;
    if (lhs_position > rhs_position) {
      std::swap(lhs_position, rhs_position);
      std::swap(lhs_actual, rhs_actual);
    }
    if (lhs_position + lhs_actual > rhs_position)
      return false;
    const std::vector<int32_t> lhs_segment(
        route.begin() + lhs_position, route.begin() + lhs_position + lhs_actual);
    const std::vector<int32_t> rhs_segment(
        route.begin() + rhs_position, route.begin() + rhs_position + rhs_actual);
    trial.clear();
    trial.reserve(route.size());
    trial.insert(trial.end(), route.begin(), route.begin() + lhs_position);
    trial.insert(trial.end(), rhs_segment.begin(), rhs_segment.end());
    trial.insert(trial.end(), route.begin() + lhs_position + lhs_actual,
                 route.begin() + rhs_position);
    trial.insert(trial.end(), lhs_segment.begin(), lhs_segment.end());
    trial.insert(trial.end(), route.begin() + rhs_position + rhs_actual,
                 route.end());
    return trial != route;
  };

  struct GuidanceValue {
    double objective = 0.0;
    double feasibility_risk = 0.0;
    std::vector<double> resource;
    explicit GuidanceValue(int32_t resource_count = 0)
        : resource(resource_count, 0.0) {}
  };
  struct GuidedSequence : GuidanceValue {
    bool empty = true;
    int32_t first = -1;
    int32_t last = -1;
  };
  const auto add_guidance = [&](GuidanceValue lhs, const GuidanceValue &rhs) {
    lhs.objective += rhs.objective;
    lhs.feasibility_risk += rhs.feasibility_risk;
    lhs.resource.resize(resource_count(), 0.0);
    for (int32_t channel = 0; channel < resource_count(); ++channel)
      lhs.resource[channel] += channel < static_cast<int32_t>(rhs.resource.size())
                                   ? rhs.resource[channel]
                                   : 0.0;
    return lhs;
  };
  const auto subtract_guidance = [&](GuidanceValue lhs,
                                    const GuidanceValue &rhs) {
    lhs.objective -= rhs.objective;
    lhs.feasibility_risk -= rhs.feasibility_risk;
    lhs.resource.resize(resource_count(), 0.0);
    for (int32_t channel = 0; channel < resource_count(); ++channel)
      lhs.resource[channel] -= channel < static_cast<int32_t>(rhs.resource.size())
                                   ? rhs.resource[channel]
                                   : 0.0;
    return lhs;
  };
  const auto edge_guidance = [&](int32_t from, int32_t to) {
    GuidanceValue value(resource_count());
    value.objective = objective_edge_cost(from, to);
    const int32_t edge = find_edge(from, to);
    value.feasibility_risk =
        edge >= 0 && edge_risk != nullptr ? edge_risk[edge] : 0.0;
    for (int32_t channel = 0; channel < resource_count(); ++channel) {
      if (!resource(channel).active)
        continue;
      value.resource[channel] = resource_field_value(
          from, to, edge, channel, edge_field, edge_additive);
    }
    return value;
  };
  const auto guidance_energy = [&](const GuidanceValue &value,
                                   const float *live_state) {
    double result = coupled_multiplier(
                        objective_multiplier(), multipliers, coupler_weights,
                        coupler_bias, live_state) *
                    value.objective;
    result += risk_penalty * value.feasibility_risk;
    for (int32_t channel = 0; channel < resource_count(); ++channel) {
      if (resource(channel).active) {
        result += coupled_multiplier(channel, multipliers, coupler_weights,
                                     coupler_bias, live_state) *
                  value.resource[channel];
      }
    }
    return result;
  };

  struct RouteResourceMetrics {
    bool exact = true;
    double capacity_excess = 0.0;
    double capacity_binding = 0.0;
    double time_warp = 0.0;
    double time_binding = 0.0;
    double route_excess = 0.0;
    double tour_excess = 0.0;
    double route_ratio = 0.0;
    double tour_ratio = 0.0;
    double prize = 0.0;
    int32_t backhaul_count = 0;
  };
  struct CachedRoute {
    bool active = true;
    int32_t previous = -1;
    int32_t next = -1;
    int32_t depot = -1;
    int32_t token_position = -1;
    int32_t closing_depot = -1;
    SequenceTable sequence;
    SequenceSummary summary;
    RouteResourceMetrics resources;
    double distance = 0.0;
    std::vector<int32_t> open_pickups;
    std::vector<GuidanceValue> forward_guidance;
    std::vector<GuidanceValue> reverse_guidance;
    GuidanceValue guidance;
  };
  struct SequencePiece {
    int32_t route = -1;
    int32_t begin = 0;
    int32_t end = 0;
    bool reverse = false;
    int32_t singleton = -1;
  };
  struct PlannedRoute {
    int32_t depot = -1;
    int32_t slot = -1;
    std::vector<SequencePiece> pieces;
  };

  std::vector<CachedRoute> cached_routes;
  int32_t route_head = -1;
  std::vector<int32_t> node_route(problem_.node_count, -1);
  std::vector<int32_t> node_local(problem_.node_count, -1);
  const auto route_distance = [&](int32_t depot,
                                  const SequenceSummary &sequence,
                                  bool force_return = false) {
    if (sequence.empty)
      return std::numeric_limits<double>::infinity();
    if (depot < 0) {
      return sequence.distance +
             (problem_.open_route
                  ? 0.0
                  : static_cast<double>(
                        problem_.dist(sequence.last, sequence.first)));
    }
    double result = problem_.dist(depot, sequence.first) + sequence.distance;
    if (force_return || !problem_.open_route)
      result += problem_.dist(sequence.last, depot);
    return result;
  };
  const double capacity_scale = resource_scale(
      static_cast<int32_t>(FieldChannel::CAPACITY));
  const double time_scale = resource_scale(
      static_cast<int32_t>(FieldChannel::TIME_WINDOW));
  const double route_scale = resource_scale(
      static_cast<int32_t>(FieldChannel::ROUTE_LIMIT));
  const double tour_scale = resource_scale(
      static_cast<int32_t>(FieldChannel::TOUR_LIMIT));
  const double quota_scale = resource_scale(
      static_cast<int32_t>(FieldChannel::PRIZE_QUOTA));
  struct TimeWindowMetrics {
    double time_warp = 0.0;
    double time_binding = 0.0;
  };
  // Preserve evaluate_resources() semantics exactly: start every depot route
  // at time zero, wait until each customer's opening time, sum lateness at
  // every visit, and exclude the depot return from the minimum-slack binding.
  // The visitor form lets planned routes be scanned directly from their pieces
  // without materializing the full candidate solution (or allocating a route).
  const auto time_window_metrics = [&]
      (int32_t depot, const auto &visit_nodes) {
    TimeWindowMetrics result;
    if (!problem_.has(TIME_WINDOWS))
      return result;
    float route_time = 0.0f;
    float time_warp = 0.0f;
    float min_time_slack = static_cast<float>(time_scale);
    int32_t current = depot;
    bool has_current = depot >= 0;
    bool visited_customer = false;
    visit_nodes([&](int32_t next) {
      // Depot-free tours use their first node as evaluate_resources()'s start
      // token; that token is not itself evaluated as a time-window arrival.
      if (!has_current) {
        current = next;
        has_current = true;
        return;
      }
      const float travel = problem_.dist(current, next);
      const float raw_arrival = route_time + travel;
      const float arrival = std::max(raw_arrival, problem_.tw_start[next]);
      time_warp += std::max(arrival - problem_.tw_end[next], 0.0f);
      min_time_slack = std::min(
          min_time_slack,
          std::max(problem_.tw_end[next] - arrival, 0.0f));
      route_time = arrival + problem_.service_time[next];
      current = next;
      visited_customer = true;
    });
    if (depot >= 0 && visited_customer && !problem_.open_route) {
      const float return_time = route_time + problem_.dist(current, depot);
      time_warp += std::max(return_time - problem_.tw_end[depot], 0.0f);
    }
    result.time_warp = time_warp;
    result.time_binding = std::clamp(
        1.0 - static_cast<double>(min_time_slack) / time_scale, 0.0, 1.0);
    return result;
  };
  const auto cached_time_window_metrics = [&]
      (int32_t depot, const std::vector<int32_t> &nodes) {
    return time_window_metrics(depot, [&](const auto &visit) {
      for (int32_t node : nodes)
        visit(node);
    });
  };
  const auto route_resource_metrics = [&]
      (int32_t depot, const SequenceSummary &sequence) {
    RouteResourceMetrics result;
    if (sequence.empty) {
      result.exact = false;
      return result;
    }

    const double initial_load =
        sequence.has_linehaul || !sequence.has_backhaul
            ? static_cast<double>(problem_.capacity)
            : 0.0;
    result.capacity_excess =
        std::max({-(initial_load + sequence.min_load_delta),
                  initial_load + sequence.max_load_delta - problem_.capacity,
                  0.0});
    const double positive_load = std::max(-sequence.min_load_delta, 0.0);
    const double negative_load = std::max(
        sequence.load_delta - sequence.min_load_delta, 0.0);
    result.capacity_binding =
        std::max(positive_load, negative_load) / capacity_scale;

    const double distance = route_distance(depot, sequence);
    const double tour = route_distance(depot, sequence, true);
    result.route_excess = problem_.has(ROUTE_LIMIT)
                              ? std::max(distance - problem_.route_limit, 0.0)
                              : 0.0;
    result.tour_excess = problem_.has(TOUR_LIMIT)
                             ? std::max(tour - problem_.tour_limit, 0.0)
                             : 0.0;
    result.route_ratio =
        problem_.has(ROUTE_LIMIT) ? distance / route_scale : 0.0;
    result.tour_ratio =
        problem_.has(TOUR_LIMIT) ? tour / tour_scale : 0.0;
    result.prize = sequence.prize;
    result.backhaul_count = sequence.has_backhaul ? 1 : 0;
    result.exact = !problem_.has(PICKUP_DELIVERY) &&
                   !sequence.backhaul_violation;
    return result;
  };

  enum RouteRank : int32_t {
    CAPACITY_EXCESS_RANK = 0,
    CAPACITY_BINDING_RANK,
    TIME_BINDING_RANK,
    ROUTE_RATIO_RANK,
    TOUR_RATIO_RANK,
    ROUTE_RANK_COUNT,
  };
  struct RankedRoute {
    double value = 0.0;
    int32_t route = -1;
    uint64_t version = 0;
  };
  struct RankedRouteLess {
    bool operator()(const RankedRoute &lhs, const RankedRoute &rhs) const {
      if (lhs.value != rhs.value)
        return lhs.value < rhs.value;
      return lhs.route > rhs.route;
    }
  };
  std::array<
      std::priority_queue<RankedRoute, std::vector<RankedRoute>,
                          RankedRouteLess>,
      ROUTE_RANK_COUNT>
      ranked_routes;
  std::vector<uint64_t> route_rank_versions;
  double total_route_excess = 0.0;
  double total_tour_excess = 0.0;
  double total_time_warp = 0.0;
  double total_prize = 0.0;
  int32_t total_backhauls = 0;
  GuidanceValue total_guidance(resource_count());
  // Directed edges can occur more than once in a multi-route incumbent.  Keep
  // reference counts so replacing one route cannot clear an edge still used by
  // another route.
  std::vector<int32_t> incumbent_edges(edge_count(), 0);
  const auto build_route_guidance = [&](CachedRoute &route) {
    const std::vector<int32_t> &nodes = route.sequence.nodes;
    route.forward_guidance.assign(nodes.size(), GuidanceValue(resource_count()));
    route.reverse_guidance.assign(nodes.size(), GuidanceValue(resource_count()));
    for (size_t index = 1; index < nodes.size(); ++index) {
      route.forward_guidance[index] = add_guidance(
          route.forward_guidance[index - 1],
          edge_guidance(nodes[index - 1], nodes[index]));
      route.reverse_guidance[index] = add_guidance(
          route.reverse_guidance[index - 1],
          edge_guidance(nodes[index], nodes[index - 1]));
    }
    if (nodes.empty())
      return;
    route.guidance = route.forward_guidance.back();
    if (route.depot >= 0) {
      route.guidance = add_guidance(
          edge_guidance(route.depot, nodes.front()), route.guidance);
      if (!problem_.open_route) {
        route.guidance = add_guidance(
            route.guidance, edge_guidance(nodes.back(), route.depot));
      }
    } else if (!problem_.open_route) {
      route.guidance = add_guidance(
          route.guidance, edge_guidance(nodes.back(), nodes.front()));
    }
  };
  const auto rebuild_cache = [&]() {
    cached_routes.clear();
    route_head = -1;
    total_guidance = GuidanceValue(resource_count());
    std::fill(node_route.begin(), node_route.end(), -1);
    std::fill(node_local.begin(), node_local.end(), -1);
    if (problem_.depot_count == 0) {
      CachedRoute route;
      route.sequence = build_sequence_table(problem_, solution.route);
      route.summary = query_sequence(
          problem_, route.sequence, 0,
          static_cast<int32_t>(route.sequence.nodes.size()));
      route.distance = route_distance(-1, route.summary);
      build_route_guidance(route);
      total_guidance = route.guidance;
      route.open_pickups.push_back(0);
      int32_t open = 0;
      const int32_t route_id = static_cast<int32_t>(cached_routes.size());
      for (int32_t local = 0;
           local < static_cast<int32_t>(route.sequence.nodes.size()); ++local) {
        const int32_t node = route.sequence.nodes[local];
        node_route[node] = route_id;
        node_local[node] = local;
        if (problem_.delivery_of_pickup[node] >= 0)
          ++open;
        if (problem_.pickup_of_delivery[node] >= 0)
          --open;
        route.open_pickups.push_back(open);
      }
      cached_routes.push_back(std::move(route));
      route_head = route_id;
      return;
    }

    int32_t token = 0;
    int32_t previous_route = -1;
    const int32_t size = static_cast<int32_t>(solution.route.size());
    while (token < size) {
      if (solution.route[token] >= problem_.depot_count)
        break;
      int32_t end = token + 1;
      while (end < size && solution.route[end] >= problem_.depot_count)
        ++end;
      if (end > token + 1) {
        CachedRoute route;
        route.depot = solution.route[token];
        route.token_position = token;
        route.closing_depot = end < size ? solution.route[end] : -1;
        route.sequence = build_sequence_table(
            problem_, std::vector<int32_t>(solution.route.begin() + token + 1,
                                           solution.route.begin() + end));
        route.summary = query_sequence(
            problem_, route.sequence, 0,
            static_cast<int32_t>(route.sequence.nodes.size()));
        route.distance = route_distance(route.depot, route.summary);
        build_route_guidance(route);
        total_guidance = add_guidance(total_guidance, route.guidance);
        route.open_pickups.push_back(0);
        int32_t open = 0;
        const int32_t route_id = static_cast<int32_t>(cached_routes.size());
        route.previous = previous_route;
        for (int32_t local = 0;
             local < static_cast<int32_t>(route.sequence.nodes.size());
             ++local) {
          const int32_t node = route.sequence.nodes[local];
          node_route[node] = route_id;
          node_local[node] = local;
          if (problem_.delivery_of_pickup[node] >= 0)
            ++open;
          if (problem_.pickup_of_delivery[node] >= 0)
            --open;
          route.open_pickups.push_back(open);
        }
        cached_routes.push_back(std::move(route));
        if (previous_route >= 0)
          cached_routes[previous_route].next = route_id;
        else
          route_head = route_id;
        previous_route = route_id;
      }
      token = end;
    }
  };
  const auto rebuild_resource_cache = [&]() {
    for (auto &ranking : ranked_routes)
      ranking = {};
    route_rank_versions.assign(cached_routes.size(), 1);
    total_route_excess = 0.0;
    total_tour_excess = 0.0;
    total_time_warp = 0.0;
    total_prize = 0.0;
    total_backhauls = 0;
    std::fill(incumbent_edges.begin(), incumbent_edges.end(), 0);
    for (size_t index = 1; index < solution.route.size(); ++index) {
      const int32_t edge =
          find_edge(solution.route[index - 1], solution.route[index]);
      if (edge >= 0)
        ++incumbent_edges[edge];
    }

    for (int32_t route_id = 0;
         route_id < static_cast<int32_t>(cached_routes.size()); ++route_id) {
      CachedRoute &route = cached_routes[route_id];
      if (!route.active)
        continue;
      route.resources = route_resource_metrics(route.depot, route.summary);
      const TimeWindowMetrics time = cached_time_window_metrics(
          route.depot, route.sequence.nodes);
      route.resources.time_warp = time.time_warp;
      route.resources.time_binding = time.time_binding;
      double positive_load = 0.0;
      double negative_load = 0.0;
      for (int32_t node : route.sequence.nodes) {
        positive_load += std::max(static_cast<double>(problem_.demand[node]),
                                  0.0);
        negative_load += std::max(-static_cast<double>(problem_.demand[node]),
                                  0.0);
      }
      route.resources.capacity_binding =
          std::max(positive_load, negative_load) / capacity_scale;
      const RouteResourceMetrics &resource = route.resources;
      const uint64_t version = route_rank_versions[route_id];
      ranked_routes[CAPACITY_EXCESS_RANK].push(
          {resource.capacity_excess, route_id, version});
      ranked_routes[CAPACITY_BINDING_RANK].push(
          {resource.capacity_binding, route_id, version});
      ranked_routes[TIME_BINDING_RANK].push(
          {resource.time_binding, route_id, version});
      ranked_routes[ROUTE_RATIO_RANK].push(
          {resource.route_ratio, route_id, version});
      ranked_routes[TOUR_RATIO_RANK].push(
          {resource.tour_ratio, route_id, version});
      total_route_excess += resource.route_excess;
      total_tour_excess += resource.tour_excess;
      total_time_warp += resource.time_warp;
      total_prize += resource.prize;
      total_backhauls += resource.backhaul_count;
    }
  };
  const auto summarize_piece = [&](const SequencePiece &piece) {
    if (piece.singleton >= 0)
      return node_summary(problem_, piece.singleton);
    return query_sequence(problem_, cached_routes[piece.route].sequence,
                          piece.begin, piece.end, piece.reverse);
  };
  const auto summarize_plan = [&](const PlannedRoute &plan) {
    SequenceSummary result;
    for (const SequencePiece &piece : plan.pieces)
      result = concatenate(problem_, result, summarize_piece(piece));
    return result;
  };
  const auto materialize_plan_nodes = [&](const PlannedRoute &plan) {
    std::vector<int32_t> nodes;
    for (const SequencePiece &piece : plan.pieces) {
      if (piece.singleton >= 0) {
        nodes.push_back(piece.singleton);
        continue;
      }
      const std::vector<int32_t> &source =
          cached_routes[piece.route].sequence.nodes;
      if (!piece.reverse) {
        nodes.insert(nodes.end(), source.begin() + piece.begin,
                     source.begin() + piece.end);
      } else {
        for (int32_t local = piece.end - 1; local >= piece.begin; --local)
          nodes.push_back(source[local]);
      }
    }
    return nodes;
  };
  const auto planned_time_window_metrics = [&](const PlannedRoute &plan) {
    return time_window_metrics(plan.depot, [&](const auto &visit) {
      for (const SequencePiece &piece : plan.pieces) {
        if (piece.singleton >= 0) {
          visit(piece.singleton);
          continue;
        }
        const std::vector<int32_t> &source =
            cached_routes[piece.route].sequence.nodes;
        if (!piece.reverse) {
          for (int32_t local = piece.begin; local < piece.end; ++local)
            visit(source[local]);
        } else {
          for (int32_t local = piece.end - 1; local >= piece.begin; --local)
            visit(source[local]);
        }
      }
    });
  };
  const auto guided_piece = [&](const SequencePiece &piece) {
    GuidedSequence result;
    result.empty = false;
    if (piece.singleton >= 0) {
      result.first = piece.singleton;
      result.last = piece.singleton;
      return result;
    }
    const CachedRoute &route = cached_routes[piece.route];
    if (piece.begin >= piece.end) {
      result.empty = true;
      return result;
    }
    const std::vector<int32_t> &nodes = route.sequence.nodes;
    if (!piece.reverse) {
      result.first = nodes[piece.begin];
      result.last = nodes[piece.end - 1];
      if (piece.end > piece.begin + 1) {
        static_cast<GuidanceValue &>(result) = subtract_guidance(
            route.forward_guidance[piece.end - 1],
            route.forward_guidance[piece.begin]);
      }
    } else {
      result.first = nodes[piece.end - 1];
      result.last = nodes[piece.begin];
      if (piece.end > piece.begin + 1) {
        static_cast<GuidanceValue &>(result) = subtract_guidance(
            route.reverse_guidance[piece.end - 1],
            route.reverse_guidance[piece.begin]);
      }
    }
    return result;
  };
  const auto append_guided = [&](GuidedSequence &lhs,
                                 const GuidedSequence &rhs) {
    if (rhs.empty)
      return;
    if (lhs.empty) {
      lhs = rhs;
      return;
    }
    static_cast<GuidanceValue &>(lhs) = add_guidance(
        static_cast<const GuidanceValue &>(lhs),
        edge_guidance(lhs.last, rhs.first));
    static_cast<GuidanceValue &>(lhs) = add_guidance(
        static_cast<const GuidanceValue &>(lhs),
        static_cast<const GuidanceValue &>(rhs));
    lhs.last = rhs.last;
  };
  const auto guided_plan = [&](const PlannedRoute &plan) {
    GuidedSequence sequence;
    for (const SequencePiece &piece : plan.pieces)
      append_guided(sequence, guided_piece(piece));
    GuidanceValue result = sequence;
    if (sequence.empty)
      return result;
    if (plan.depot >= 0) {
      result = add_guidance(edge_guidance(plan.depot, sequence.first), result);
      if (!problem_.open_route)
        result = add_guidance(
            result, edge_guidance(sequence.last, plan.depot));
    } else if (!problem_.open_route) {
      result = add_guidance(
          result, edge_guidance(sequence.last, sequence.first));
    }
    return result;
  };
  const auto time_feasible = [&](int32_t depot,
                                 const SequenceSummary &sequence) {
    if (!problem_.has(TIME_WINDOWS))
      return true;
    SequenceSummary start;
    start.empty = false;
    start.first = depot;
    start.last = depot;
    start.earliest = 0.0;
    start.latest = 0.0;
    SequenceSummary full = concatenate(problem_, start, sequence);
    if (!problem_.open_route) {
      SequenceSummary end;
      end.empty = false;
      end.first = depot;
      end.last = depot;
      end.earliest = -SEQUENCE_INFINITY;
      end.latest = std::min(static_cast<double>(problem_.tw_end[depot]),
                            SEQUENCE_INFINITY);
      full = concatenate(problem_, full, end);
    }
    return full.time_warp <= FEASIBILITY_EPS;
  };
  const auto pickup_closed = [&](const SequencePiece &piece) {
    if (!problem_.has(PICKUP_DELIVERY) || piece.singleton >= 0)
      return true;
    const CachedRoute &route = cached_routes[piece.route];
    return route.open_pickups[piece.begin] == 0 &&
           route.open_pickups[piece.end] == 0;
  };
  rebuild_cache();
  rebuild_resource_cache();
  static constexpr size_t MAX_SCREENING_LABELS = 512;
  ResourceEvaluation current_resource;
  if (trace != nullptr)
    current_resource = evaluate_resources(solution.route);
  const auto screening_delta = [&](const ResourceEvaluation &candidate) {
    std::array<float, FIELD_CHANNEL_COUNT> delta{};
    for (int32_t channel = 0; channel < FIELD_CHANNEL_COUNT; ++channel) {
      delta[channel] = std::clamp(
          candidate.binding[channel] - current_resource.binding[channel] +
              candidate.violation[channel],
          0.0f, 1.0f);
    }
    return delta;
  };
  const auto append_screening_edge = [&]
      (int32_t from, int32_t to,
       const std::array<float, FIELD_CHANNEL_COUNT> &delta) {
    if (trace == nullptr ||
        trace->screened_edges.size() >= MAX_SCREENING_LABELS) {
      return;
    }
    const int32_t edge = find_edge(from, to);
    if (edge < 0 || incumbent_edges[edge])
      return;
    trace->screened_edges.push_back(edge);
    trace->screened_resource_delta.insert(
        trace->screened_resource_delta.end(), delta.begin(), delta.end());
  };
  const auto record_screening = [&](const Solution &candidate) {
    if (trace == nullptr || candidate.route.empty() ||
        trace->screened_edges.size() >= MAX_SCREENING_LABELS)
      return;
    ++trace->screening_fallback_evaluations;
    const ResourceEvaluation candidate_resource =
        evaluate_resources(candidate.route);
    if (!candidate_resource.structurally_valid)
      return;
    const auto delta = screening_delta(candidate_resource);
    for (size_t index = 1;
         index < candidate.route.size() &&
         trace->screened_edges.size() < MAX_SCREENING_LABELS;
         ++index) {
      append_screening_edge(candidate.route[index - 1],
                            candidate.route[index], delta);
    }
  };
  const auto unaffected_rank = [&]
      (RouteRank rank, const std::array<int32_t, 2> &affected,
       int32_t affected_count, double fallback) {
    auto &ranking = ranked_routes[rank];
    std::array<RankedRoute, 2> skipped;
    int32_t skipped_count = 0;
    double result = fallback;
    while (!ranking.empty()) {
      const RankedRoute entry = ranking.top();
      if (entry.route < 0 ||
          entry.route >= static_cast<int32_t>(route_rank_versions.size()) ||
          entry.version != route_rank_versions[entry.route]) {
        ranking.pop();
        continue;
      }
      bool is_affected = false;
      for (int32_t index = 0; index < affected_count; ++index)
        is_affected |= affected[index] == entry.route;
      if (!is_affected) {
        result = entry.value;
        break;
      }
      ranking.pop();
      skipped[skipped_count++] = entry;
    }
    for (int32_t index = 0; index < skipped_count; ++index)
      ranking.push(skipped[index]);
    return result;
  };
  const auto evaluate_planned_resources = [&]
      (const std::vector<PlannedRoute> &plans,
       const std::vector<SequenceSummary> &sequences,
       const std::vector<int32_t> &affected_routes)
      -> std::optional<ResourceEvaluation> {
    if (plans.size() != sequences.size())
      return std::nullopt;
    // Pickup-delivery binding tracks maximum open pairs and does not yet have
    // an exact fixed-size concatenation summary. Time-window labels are handled
    // below by scanning only the affected route pieces, preserving the legacy
    // summed-lateness and minimum-slack definitions exactly.
    if (problem_.has(PICKUP_DELIVERY))
      return std::nullopt;
    std::array<int32_t, 2> affected{-1, -1};
    int32_t affected_count = 0;
    for (int32_t route : affected_routes) {
      if (route >= 0 &&
          route < static_cast<int32_t>(cached_routes.size())) {
        bool duplicate = false;
        for (int32_t index = 0; index < affected_count; ++index)
          duplicate |= affected[index] == route;
        if (!duplicate) {
          if (affected_count == static_cast<int32_t>(affected.size()))
            return std::nullopt;
          affected[affected_count++] = route;
        }
      }
    }

    double route_excess = total_route_excess;
    double tour_excess = total_tour_excess;
    double time_warp = total_time_warp;
    double prize = total_prize;
    int32_t backhauls = total_backhauls;
    for (int32_t index = 0; index < affected_count; ++index) {
      const int32_t route = affected[index];
      const RouteResourceMetrics &old = cached_routes[route].resources;
      route_excess -= old.route_excess;
      tour_excess -= old.tour_excess;
      time_warp -= old.time_warp;
      prize -= old.prize;
      backhauls -= old.backhaul_count;
    }

    double capacity_excess = unaffected_rank(
        CAPACITY_EXCESS_RANK, affected, affected_count, 0.0);
    double capacity_binding = unaffected_rank(
        CAPACITY_BINDING_RANK, affected, affected_count, 0.0);
    double time_binding = unaffected_rank(
        TIME_BINDING_RANK, affected, affected_count, 0.0);
    double max_route_ratio = unaffected_rank(
        ROUTE_RATIO_RANK, affected, affected_count, 0.0);
    double max_tour_ratio = unaffected_rank(
        TOUR_RATIO_RANK, affected, affected_count, 0.0);

    for (size_t index = 0; index < plans.size(); ++index) {
      const PlannedRoute &plan = plans[index];
      const SequenceSummary &sequence = sequences[index];
      if (sequence.empty)
        return std::nullopt;
      RouteResourceMetrics resource = route_resource_metrics(
          plan.depot, sequence);
      if (!resource.exact)
        return std::nullopt;
      const TimeWindowMetrics time = planned_time_window_metrics(plan);
      resource.time_warp = time.time_warp;
      resource.time_binding = time.time_binding;
      capacity_excess =
          std::max(capacity_excess, resource.capacity_excess);
      capacity_binding =
          std::max(capacity_binding, resource.capacity_binding);
      time_binding = std::max(time_binding, resource.time_binding);
      max_route_ratio = std::max(max_route_ratio, resource.route_ratio);
      max_tour_ratio = std::max(max_tour_ratio, resource.tour_ratio);
      route_excess += resource.route_excess;
      tour_excess += resource.tour_excess;
      time_warp += resource.time_warp;
      prize += resource.prize;
      backhauls += resource.backhaul_count;
    }

    ResourceEvaluation result;
    result.violation.assign(FIELD_CHANNEL_COUNT, 0.0f);
    result.binding.assign(FIELD_CHANNEL_COUNT, 0.0f);
    result.violation[static_cast<int32_t>(FieldChannel::CAPACITY)] =
        static_cast<float>(capacity_excess / capacity_scale);
    result.violation[static_cast<int32_t>(FieldChannel::TIME_WINDOW)] =
        static_cast<float>(time_warp / time_scale);
    result.violation[static_cast<int32_t>(FieldChannel::ROUTE_LIMIT)] =
        static_cast<float>(route_excess / route_scale);
    result.violation[static_cast<int32_t>(FieldChannel::TOUR_LIMIT)] =
        static_cast<float>(tour_excess / tour_scale);
    result.violation[static_cast<int32_t>(FieldChannel::BACKHAUL_ORDER)] = 0.0f;
    result.violation[static_cast<int32_t>(FieldChannel::PICKUP_DELIVERY)] = 0.0f;
    result.violation[static_cast<int32_t>(FieldChannel::PRIZE_QUOTA)] =
        static_cast<float>(
            std::max(problem_.prize_quota - prize, 0.0) / quota_scale);

    result.binding[static_cast<int32_t>(FieldChannel::CAPACITY)] =
        static_cast<float>(std::clamp(capacity_binding, 0.0, 1.0));
    result.binding[static_cast<int32_t>(FieldChannel::TIME_WINDOW)] =
        static_cast<float>(std::clamp(time_binding, 0.0, 1.0));
    result.binding[static_cast<int32_t>(FieldChannel::ROUTE_LIMIT)] =
        static_cast<float>(std::clamp(max_route_ratio, 0.0, 1.0));
    result.binding[static_cast<int32_t>(FieldChannel::TOUR_LIMIT)] =
        static_cast<float>(std::clamp(max_tour_ratio, 0.0, 1.0));
    result.binding[static_cast<int32_t>(FieldChannel::BACKHAUL_ORDER)] =
        backhauls > 0 ? 1.0f : 0.0f;
    result.binding[static_cast<int32_t>(FieldChannel::PICKUP_DELIVERY)] = 0.0f;
    result.binding[static_cast<int32_t>(FieldChannel::PRIZE_QUOTA)] =
        problem_.has(PRIZE_QUOTA)
            ? static_cast<float>(std::clamp(prize / quota_scale, 0.0, 1.0))
            : 0.0f;
    for (int32_t channel = 0; channel < FIELD_CHANNEL_COUNT; ++channel) {
      if (!field_channel_active(channel)) {
        result.violation[channel] = 0.0f;
        result.binding[channel] = 0.0f;
      } else if (result.violation[channel] > FEASIBILITY_EPS) {
        result.binding[channel] = 1.0f;
      }
    }
    result.structurally_valid = true;
    return result;
  };
  const auto record_planned_screening = [&]
      (const std::vector<PlannedRoute> &plans,
       const ResourceEvaluation &candidate_resource) {
    if (trace == nullptr ||
        trace->screened_edges.size() >= MAX_SCREENING_LABELS) {
      return;
    }
    ++trace->screening_fast_evaluations;
    const auto delta = screening_delta(candidate_resource);
    std::vector<const PlannedRoute *> ordered;
    ordered.reserve(plans.size());
    for (const PlannedRoute &plan : plans)
      ordered.push_back(&plan);
    std::sort(ordered.begin(), ordered.end(),
              [](const PlannedRoute *lhs, const PlannedRoute *rhs) {
                return lhs->slot < rhs->slot;
              });
    for (const PlannedRoute *plan : ordered) {
      bool has_previous = plan->depot >= 0;
      int32_t previous = plan->depot;
      for (const SequencePiece &piece : plan->pieces) {
        if (piece.singleton >= 0) {
          if (has_previous)
            append_screening_edge(previous, piece.singleton, delta);
          previous = piece.singleton;
          has_previous = true;
          continue;
        }
        const std::vector<int32_t> &nodes =
            cached_routes[piece.route].sequence.nodes;
        if (!piece.reverse) {
          const int32_t first = nodes[piece.begin];
          if (has_previous)
            append_screening_edge(previous, first, delta);
          previous = nodes[piece.end - 1];
          has_previous = true;
          continue;
        }
        for (int32_t local = piece.end - 1; local >= piece.begin; --local) {
          const int32_t node = nodes[local];
          if (has_previous)
            append_screening_edge(previous, node, delta);
          previous = node;
          has_previous = true;
        }
      }
      if (has_previous && plan->slot >= 0) {
        const int32_t closing = cached_routes[plan->slot].closing_depot;
        if (closing >= 0)
          append_screening_edge(previous, closing, delta);
      }
    }
  };

  const auto build_cached_route = [&](CachedRoute route,
                                      std::vector<int32_t> nodes) {
    route.sequence = {};
    route.summary = {};
    route.resources = {};
    route.distance = 0.0;
    route.open_pickups.clear();
    route.forward_guidance.clear();
    route.reverse_guidance.clear();
    route.guidance = GuidanceValue(resource_count());
    route.sequence = build_sequence_table(problem_, nodes);
    route.summary = query_sequence(
        problem_, route.sequence, 0,
        static_cast<int32_t>(route.sequence.nodes.size()));
    route.distance = route_distance(route.depot, route.summary);
    build_route_guidance(route);
    route.open_pickups.reserve(route.sequence.nodes.size() + 1);
    route.open_pickups.push_back(0);
    int32_t open = 0;
    double positive_load = 0.0;
    double negative_load = 0.0;
    for (int32_t node : route.sequence.nodes) {
      if (problem_.delivery_of_pickup[node] >= 0)
        ++open;
      if (problem_.pickup_of_delivery[node] >= 0)
        --open;
      route.open_pickups.push_back(open);
      positive_load +=
          std::max(static_cast<double>(problem_.demand[node]), 0.0);
      negative_load +=
          std::max(-static_cast<double>(problem_.demand[node]), 0.0);
    }
    route.resources = route_resource_metrics(route.depot, route.summary);
    const TimeWindowMetrics time = cached_time_window_metrics(
        route.depot, route.sequence.nodes);
    route.resources.time_warp = time.time_warp;
    route.resources.time_binding = time.time_binding;
    route.resources.capacity_binding =
        std::max(positive_load, negative_load) / capacity_scale;
    return route;
  };
  const auto build_incremental_route = [&](int32_t slot,
                                           std::vector<int32_t> nodes) {
    return build_cached_route(cached_routes[slot], std::move(nodes));
  };
  const auto adjust_route_edges = [&](const CachedRoute &route, int32_t delta) {
    const std::vector<int32_t> &nodes = route.sequence.nodes;
    if (nodes.empty())
      return;
    const auto adjust = [&](int32_t from, int32_t to) {
      const int32_t edge = find_edge(from, to);
      if (edge < 0)
        return;
      incumbent_edges[edge] += delta;
      if (incumbent_edges[edge] < 0)
        throw std::runtime_error("negative incumbent edge reference count");
    };
    if (route.depot >= 0)
      adjust(route.depot, nodes.front());
    for (size_t index = 1; index < nodes.size(); ++index)
      adjust(nodes[index - 1], nodes[index]);
    if (route.depot >= 0 && route.closing_depot >= 0)
      adjust(nodes.back(), route.closing_depot);
  };
  const auto rank_value = [](const RouteResourceMetrics &resource,
                             RouteRank rank) {
    switch (rank) {
    case CAPACITY_EXCESS_RANK:
      return resource.capacity_excess;
    case CAPACITY_BINDING_RANK:
      return resource.capacity_binding;
    case TIME_BINDING_RANK:
      return resource.time_binding;
    case ROUTE_RATIO_RANK:
      return resource.route_ratio;
    case TOUR_RATIO_RANK:
      return resource.tour_ratio;
    case ROUTE_RANK_COUNT:
      break;
    }
    return 0.0;
  };
  std::vector<int32_t> old_successor(problem_.node_count, -1);
  std::vector<int32_t> new_successor(problem_.node_count, -1);
  std::vector<int32_t> old_depot_predecessor(problem_.node_count, -1);
  std::vector<int32_t> new_depot_predecessor(problem_.node_count, -1);
  std::vector<int32_t> scope_stamp(problem_.node_count, 0);
  std::vector<int32_t> touched_stamp(problem_.node_count, 0);
  int32_t scope_generation = 0;
  int32_t touched_generation = 0;
  const auto replace_planned_routes = [&]
      (const std::vector<PlannedRoute> &plans,
       std::vector<int32_t> &touched, int64_t &rebuilt_nodes) {
    if (plans.empty() || plans.size() > 2)
      return false;
    std::array<int32_t, 2> slots{-1, -1};
    std::array<CachedRoute, 2> replacements;
    int32_t count = 0;
    for (const PlannedRoute &plan : plans) {
      if (plan.slot < 0 ||
          plan.slot >= static_cast<int32_t>(cached_routes.size()) ||
          !cached_routes[plan.slot].active) {
        return false;
      }
      for (int32_t index = 0; index < count; ++index) {
        if (slots[index] == plan.slot)
          return false;
      }
      std::vector<int32_t> nodes = materialize_plan_nodes(plan);
      if (nodes.empty())
        return false;
      slots[count] = plan.slot;
      rebuilt_nodes += static_cast<int64_t>(nodes.size());
      replacements[count] =
          build_incremental_route(plan.slot, std::move(nodes));
      ++count;
    }

    if (++scope_generation == std::numeric_limits<int32_t>::max()) {
      std::fill(scope_stamp.begin(), scope_stamp.end(), 0);
      scope_generation = 1;
    }
    std::vector<int32_t> union_nodes;
    const auto add_union = [&](int32_t node) {
      if (node >= problem_.depot_count && node < problem_.node_count &&
          scope_stamp[node] != scope_generation) {
        scope_stamp[node] = scope_generation;
        union_nodes.push_back(node);
      }
    };
    for (int32_t index = 0; index < count; ++index) {
      for (int32_t node : cached_routes[slots[index]].sequence.nodes)
        add_union(node);
      for (int32_t node : replacements[index].sequence.nodes)
        add_union(node);
    }
    for (int32_t node : union_nodes) {
      old_successor[node] = -1;
      new_successor[node] = -1;
      old_depot_predecessor[node] = -1;
      new_depot_predecessor[node] = -1;
    }
    const auto collect_links = [&](const CachedRoute &route,
                                   std::vector<int32_t> &successor,
                                   std::vector<int32_t> &depot_predecessor) {
      const std::vector<int32_t> &nodes = route.sequence.nodes;
      if (nodes.empty())
        return;
      if (route.depot >= 0)
        depot_predecessor[nodes.front()] = route.depot;
      for (size_t local = 1; local < nodes.size(); ++local)
        successor[nodes[local - 1]] = nodes[local];
      if (route.depot >= 0) {
        successor[nodes.back()] = route.closing_depot;
      } else if (!problem_.open_route) {
        successor[nodes.back()] = nodes.front();
      }
    };
    for (int32_t index = 0; index < count; ++index) {
      collect_links(cached_routes[slots[index]], old_successor,
                    old_depot_predecessor);
      collect_links(replacements[index], new_successor,
                    new_depot_predecessor);
    }
    touched.clear();
    if (++touched_generation == std::numeric_limits<int32_t>::max()) {
      std::fill(touched_stamp.begin(), touched_stamp.end(), 0);
      touched_generation = 1;
    }
    const auto mark_touched = [&](int32_t node) {
      if (node >= problem_.depot_count && node < problem_.node_count &&
          touched_stamp[node] != touched_generation) {
        touched_stamp[node] = touched_generation;
        touched.push_back(node);
      }
    };
    for (int32_t node : union_nodes) {
      if (old_successor[node] != new_successor[node]) {
        mark_touched(node);
        mark_touched(old_successor[node]);
        mark_touched(new_successor[node]);
      }
      if (old_depot_predecessor[node] != new_depot_predecessor[node])
        mark_touched(node);
    }

    for (int32_t index = 0; index < count; ++index) {
      const CachedRoute &old = cached_routes[slots[index]];
      adjust_route_edges(old, -1);
      total_guidance = subtract_guidance(total_guidance, old.guidance);
      total_route_excess -= old.resources.route_excess;
      total_tour_excess -= old.resources.tour_excess;
      total_time_warp -= old.resources.time_warp;
      total_prize -= old.resources.prize;
      total_backhauls -= old.resources.backhaul_count;
      for (int32_t node : old.sequence.nodes) {
        node_route[node] = -1;
        node_local[node] = -1;
      }
    }
    for (int32_t index = 0; index < count; ++index) {
      const int32_t slot = slots[index];
      cached_routes[slot] = std::move(replacements[index]);
      CachedRoute &route = cached_routes[slot];
      for (int32_t local = 0;
           local < static_cast<int32_t>(route.sequence.nodes.size()); ++local) {
        const int32_t node = route.sequence.nodes[local];
        node_route[node] = slot;
        node_local[node] = local;
      }
      adjust_route_edges(route, 1);
      total_guidance = add_guidance(total_guidance, route.guidance);
      total_route_excess += route.resources.route_excess;
      total_tour_excess += route.resources.tour_excess;
      total_time_warp += route.resources.time_warp;
      total_prize += route.resources.prize;
      total_backhauls += route.resources.backhaul_count;
      const uint64_t version = ++route_rank_versions[slot];
      for (int32_t rank = 0; rank < ROUTE_RANK_COUNT; ++rank) {
        ranked_routes[rank].push(
            {rank_value(route.resources, static_cast<RouteRank>(rank)), slot,
             version});
      }
    }
    return true;
  };
  struct RouteReplacement {
    int32_t slot = -1;
    CachedRoute route;
  };
  const auto replace_structural_routes = [&]
      (const std::vector<int32_t> &old_slots,
       std::vector<RouteReplacement> replacements,
       std::vector<int32_t> &touched, int64_t &rebuilt_nodes) {
    if (old_slots.empty() || old_slots.size() > 2 || replacements.empty() ||
        replacements.size() > 2)
      return false;
    std::vector<CachedRoute> old_routes;
    old_routes.reserve(old_slots.size());
    for (int32_t slot : old_slots) {
      if (slot < 0 || slot >= static_cast<int32_t>(cached_routes.size()) ||
          !cached_routes[slot].active)
        return false;
      old_routes.push_back(cached_routes[slot]);
    }
    for (const RouteReplacement &replacement : replacements) {
      if (replacement.slot < 0 || !replacement.route.active ||
          replacement.route.sequence.nodes.empty())
        return false;
      rebuilt_nodes +=
          static_cast<int64_t>(replacement.route.sequence.nodes.size());
    }

    if (++scope_generation == std::numeric_limits<int32_t>::max()) {
      std::fill(scope_stamp.begin(), scope_stamp.end(), 0);
      scope_generation = 1;
    }
    std::vector<int32_t> union_nodes;
    const auto add_union = [&](int32_t node) {
      if (node >= problem_.depot_count && node < problem_.node_count &&
          scope_stamp[node] != scope_generation) {
        scope_stamp[node] = scope_generation;
        union_nodes.push_back(node);
      }
    };
    for (const CachedRoute &route : old_routes)
      for (int32_t node : route.sequence.nodes)
        add_union(node);
    for (const RouteReplacement &replacement : replacements)
      for (int32_t node : replacement.route.sequence.nodes)
        add_union(node);
    for (int32_t node : union_nodes) {
      old_successor[node] = -1;
      new_successor[node] = -1;
      old_depot_predecessor[node] = -1;
      new_depot_predecessor[node] = -1;
    }
    const auto collect_links = [&](const CachedRoute &route,
                                   std::vector<int32_t> &successor,
                                   std::vector<int32_t> &depot_predecessor) {
      const std::vector<int32_t> &nodes = route.sequence.nodes;
      if (nodes.empty())
        return;
      if (route.depot >= 0)
        depot_predecessor[nodes.front()] = route.depot;
      for (size_t local = 1; local < nodes.size(); ++local)
        successor[nodes[local - 1]] = nodes[local];
      if (route.depot >= 0)
        successor[nodes.back()] = route.closing_depot;
      else if (!problem_.open_route)
        successor[nodes.back()] = nodes.front();
    };
    for (const CachedRoute &route : old_routes)
      collect_links(route, old_successor, old_depot_predecessor);
    for (const RouteReplacement &replacement : replacements)
      collect_links(replacement.route, new_successor, new_depot_predecessor);
    touched.clear();
    if (++touched_generation == std::numeric_limits<int32_t>::max()) {
      std::fill(touched_stamp.begin(), touched_stamp.end(), 0);
      touched_generation = 1;
    }
    const auto mark_touched = [&](int32_t node) {
      if (node >= problem_.depot_count && node < problem_.node_count &&
          touched_stamp[node] != touched_generation) {
        touched_stamp[node] = touched_generation;
        touched.push_back(node);
      }
    };
    for (int32_t node : union_nodes) {
      if (old_successor[node] != new_successor[node]) {
        mark_touched(node);
        mark_touched(old_successor[node]);
        mark_touched(new_successor[node]);
      }
      if (old_depot_predecessor[node] != new_depot_predecessor[node])
        mark_touched(node);
    }

    std::vector<uint8_t> version_updated(
        std::max(cached_routes.size(), route_rank_versions.size()), 0);
    for (size_t index = 0; index < old_slots.size(); ++index) {
      const int32_t slot = old_slots[index];
      const CachedRoute &old = old_routes[index];
      adjust_route_edges(old, -1);
      total_guidance = subtract_guidance(total_guidance, old.guidance);
      total_route_excess -= old.resources.route_excess;
      total_tour_excess -= old.resources.tour_excess;
      total_time_warp -= old.resources.time_warp;
      total_prize -= old.resources.prize;
      total_backhauls -= old.resources.backhaul_count;
      for (int32_t node : old.sequence.nodes) {
        node_route[node] = -1;
        node_local[node] = -1;
      }
      ++route_rank_versions[slot];
      version_updated[slot] = 1;
      cached_routes[slot].active = false;
    }
    int32_t largest_slot = -1;
    for (const RouteReplacement &replacement : replacements)
      largest_slot = std::max(largest_slot, replacement.slot);
    if (largest_slot >= static_cast<int32_t>(cached_routes.size())) {
      const size_t previous_size = cached_routes.size();
      cached_routes.resize(static_cast<size_t>(largest_slot) + 1);
      for (size_t slot = previous_size; slot < cached_routes.size(); ++slot)
        cached_routes[slot].active = false;
    }
    if (route_rank_versions.size() < cached_routes.size())
      route_rank_versions.resize(cached_routes.size(), 0);
    if (version_updated.size() < cached_routes.size())
      version_updated.resize(cached_routes.size(), 0);

    for (RouteReplacement &replacement : replacements) {
      const int32_t slot = replacement.slot;
      if (!version_updated[slot])
        ++route_rank_versions[slot];
      cached_routes[slot] = std::move(replacement.route);
      CachedRoute &route = cached_routes[slot];
      for (int32_t local = 0;
           local < static_cast<int32_t>(route.sequence.nodes.size()); ++local) {
        const int32_t node = route.sequence.nodes[local];
        node_route[node] = slot;
        node_local[node] = local;
      }
      adjust_route_edges(route, 1);
      total_guidance = add_guidance(total_guidance, route.guidance);
      total_route_excess += route.resources.route_excess;
      total_tour_excess += route.resources.tour_excess;
      total_time_warp += route.resources.time_warp;
      total_prize += route.resources.prize;
      total_backhauls += route.resources.backhaul_count;
      const uint64_t version = route_rank_versions[slot];
      for (int32_t rank = 0; rank < ROUTE_RANK_COUNT; ++rank) {
        ranked_routes[rank].push(
            {rank_value(route.resources, static_cast<RouteRank>(rank)), slot,
             version});
      }
    }
    return true;
  };

  enum class StructuralMoveKind {
    NONE,
    MERGE_ROUTES,
    SPLIT_ROUTE,
    REASSIGN_DEPOT,
  };
  struct StructuralMove {
    StructuralMoveKind kind = StructuralMoveKind::NONE;
    int32_t first = -1;
    int32_t second = -1;
    int32_t position = -1;
    int32_t depot = -1;

    bool valid() const { return kind != StructuralMoveKind::NONE; }
  };
  const auto inactive_route_slot = [&]() {
    for (int32_t slot = 0;
         slot < static_cast<int32_t>(cached_routes.size()); ++slot) {
      if (!cached_routes[slot].active)
        return slot;
    }
    return static_cast<int32_t>(cached_routes.size());
  };
  const auto apply_structural_move = [&]
      (const StructuralMove &move, std::vector<int32_t> &touched,
       int64_t &rebuilt_nodes) {
    if (!move.valid())
      return false;
    if (move.kind == StructuralMoveKind::MERGE_ROUTES) {
      if (move.first < 0 || move.second < 0 ||
          move.first >= static_cast<int32_t>(cached_routes.size()) ||
          move.second >= static_cast<int32_t>(cached_routes.size()))
        return false;
      const CachedRoute left = cached_routes[move.first];
      const CachedRoute right = cached_routes[move.second];
      if (!left.active || !right.active || left.next != move.second ||
          right.previous != move.first)
        return false;
      std::vector<int32_t> nodes = left.sequence.nodes;
      nodes.insert(nodes.end(), right.sequence.nodes.begin(),
                   right.sequence.nodes.end());
      CachedRoute merged = left;
      merged.next = right.next;
      merged.closing_depot = right.closing_depot;
      merged = build_cached_route(std::move(merged), std::move(nodes));
      const int32_t following = right.next;
      if (!replace_structural_routes(
              {move.first, move.second},
              {{move.first, std::move(merged)}}, touched, rebuilt_nodes))
        return false;
      if (following >= 0)
        cached_routes[following].previous = move.first;
      return true;
    }
    if (move.kind == StructuralMoveKind::SPLIT_ROUTE) {
      if (move.first < 0 ||
          move.first >= static_cast<int32_t>(cached_routes.size()) ||
          !cached_routes[move.first].active)
        return false;
      const CachedRoute old = cached_routes[move.first];
      const int32_t size =
          static_cast<int32_t>(old.sequence.nodes.size());
      if (move.position < 0 || move.position + 1 >= size || move.depot < 0 ||
          move.depot >= problem_.depot_count)
        return false;
      const int32_t right_slot = inactive_route_slot();
      CachedRoute left = old;
      left.next = right_slot;
      left.closing_depot = move.depot;
      left = build_cached_route(
          std::move(left),
          std::vector<int32_t>(old.sequence.nodes.begin(),
                               old.sequence.nodes.begin() + move.position + 1));
      CachedRoute right;
      right.active = true;
      right.previous = move.first;
      right.next = old.next;
      right.depot = move.depot;
      right.closing_depot = old.closing_depot;
      right = build_cached_route(
          std::move(right),
          std::vector<int32_t>(old.sequence.nodes.begin() + move.position + 1,
                               old.sequence.nodes.end()));
      const int32_t following = old.next;
      if (!replace_structural_routes(
              {move.first},
              {{move.first, std::move(left)}, {right_slot, std::move(right)}},
              touched, rebuilt_nodes))
        return false;
      if (following >= 0)
        cached_routes[following].previous = right_slot;
      return true;
    }
    if (move.kind == StructuralMoveKind::REASSIGN_DEPOT) {
      if (move.first < 0 ||
          move.first >= static_cast<int32_t>(cached_routes.size()) ||
          !cached_routes[move.first].active || move.depot < 0 ||
          move.depot >= problem_.depot_count)
        return false;
      const CachedRoute current_old = cached_routes[move.first];
      CachedRoute current = current_old;
      current.depot = move.depot;
      current = build_cached_route(std::move(current),
                                   current_old.sequence.nodes);
      if (current_old.previous < 0) {
        return replace_structural_routes(
            {move.first}, {{move.first, std::move(current)}}, touched,
            rebuilt_nodes);
      }
      const int32_t previous_slot = current_old.previous;
      const CachedRoute previous_old = cached_routes[previous_slot];
      CachedRoute previous = previous_old;
      previous.closing_depot = move.depot;
      previous = build_cached_route(std::move(previous),
                                    previous_old.sequence.nodes);
      return replace_structural_routes(
          {previous_slot, move.first},
          {{previous_slot, std::move(previous)},
           {move.first, std::move(current)}},
          touched, rebuilt_nodes);
    }
    return false;
  };
  const auto verify_incremental_cache = [&]() {
    if (!search_config_.verify_incremental_srr)
      return;
    std::vector<CachedRoute> incremental_routes;
    std::vector<int32_t> incremental_node_route(problem_.node_count, -1);
    int32_t route_slot = route_head;
    while (route_slot >= 0) {
      if (route_slot >= static_cast<int32_t>(cached_routes.size()) ||
          !cached_routes[route_slot].active)
        throw std::runtime_error("incremental SRR cache has invalid route links");
      const int32_t ordinal =
          static_cast<int32_t>(incremental_routes.size());
      incremental_routes.push_back(cached_routes[route_slot]);
      for (int32_t node : cached_routes[route_slot].sequence.nodes)
        incremental_node_route[node] = ordinal;
      route_slot = cached_routes[route_slot].next;
      if (incremental_routes.size() > cached_routes.size())
        throw std::runtime_error("incremental SRR cache has a route cycle");
    }
    const std::vector<int32_t> incremental_node_local = node_local;
    const std::vector<int32_t> incremental_edges = incumbent_edges;
    std::array<double, ROUTE_RANK_COUNT> incremental_rank_max{};
    const std::array<int32_t, 2> no_affected{-1, -1};
    for (int32_t rank = 0; rank < ROUTE_RANK_COUNT; ++rank) {
      incremental_rank_max[rank] = unaffected_rank(
          static_cast<RouteRank>(rank), no_affected, 0, 0.0);
    }
    const double incremental_route_excess = total_route_excess;
    const double incremental_tour_excess = total_tour_excess;
    const double incremental_time_warp = total_time_warp;
    const double incremental_prize = total_prize;
    const int32_t incremental_backhauls = total_backhauls;
    const GuidanceValue incremental_guidance = total_guidance;

    rebuild_cache();
    rebuild_resource_cache();

    const auto close = [](double lhs, double rhs) {
      const double scale = std::max({1.0, std::abs(lhs), std::abs(rhs)});
      return std::abs(lhs - rhs) <= 1.0e-8 * scale;
    };
    const auto same_guidance = [&](const GuidanceValue &lhs,
                                   const GuidanceValue &rhs) {
      if (!close(lhs.objective, rhs.objective) ||
          !close(lhs.feasibility_risk, rhs.feasibility_risk)) {
        return false;
      }
      for (int32_t channel = 0; channel < resource_count(); ++channel) {
        if (!close(lhs.resource[channel], rhs.resource[channel]))
          return false;
      }
      return true;
    };
    const auto same_resource = [&](const RouteResourceMetrics &lhs,
                                   const RouteResourceMetrics &rhs) {
      return lhs.exact == rhs.exact &&
             close(lhs.capacity_excess, rhs.capacity_excess) &&
             close(lhs.capacity_binding, rhs.capacity_binding) &&
             close(lhs.time_warp, rhs.time_warp) &&
             close(lhs.time_binding, rhs.time_binding) &&
             close(lhs.route_excess, rhs.route_excess) &&
             close(lhs.tour_excess, rhs.tour_excess) &&
             close(lhs.route_ratio, rhs.route_ratio) &&
             close(lhs.tour_ratio, rhs.tour_ratio) &&
             close(lhs.prize, rhs.prize) &&
             lhs.backhaul_count == rhs.backhaul_count;
    };
    const auto fail = [](const char *component) {
      throw std::runtime_error(std::string("incremental SRR cache mismatch: ") +
                               component);
    };
    if (incremental_routes.size() != cached_routes.size())
      fail("route count");
    for (size_t index = 0; index < cached_routes.size(); ++index) {
      const CachedRoute &expected = incremental_routes[index];
      const CachedRoute &actual = cached_routes[index];
      if (expected.depot != actual.depot ||
          expected.closing_depot != actual.closing_depot ||
          expected.sequence.nodes != actual.sequence.nodes ||
          expected.open_pickups != actual.open_pickups ||
          !close(expected.distance, actual.distance) ||
          !same_resource(expected.resources, actual.resources) ||
          !same_guidance(expected.guidance, actual.guidance)) {
        fail("route");
      }
    }
    if (incremental_node_route != node_route ||
        incremental_node_local != node_local)
      fail("node ownership");
    if (incremental_edges != incumbent_edges)
      fail("incumbent edges");
    for (size_t rank = 0; rank < ranked_routes.size(); ++rank) {
      if (!close(incremental_rank_max[rank],
                 unaffected_rank(static_cast<RouteRank>(rank), no_affected, 0,
                                 0.0)))
        fail("ranking values");
    }
    if (!close(incremental_route_excess, total_route_excess) ||
        !close(incremental_tour_excess, total_tour_excess) ||
        !close(incremental_time_warp, total_time_warp) ||
        !close(incremental_prize, total_prize) ||
        incremental_backhauls != total_backhauls ||
        !same_guidance(incremental_guidance, total_guidance)) {
      fail("aggregate resources");
    }
  };

  struct AcceptedPlan {
    bool valid = false;
    std::vector<PlannedRoute> plans;
    std::optional<ResourceEvaluation> resources;
  };

  std::vector<int32_t> ranked_local_edges(edge_count());
  std::vector<double> ranking_energy(
      search_config_.classical_behavior ? 0 : edge_count());
  for (int32_t node = 0; node < problem_.node_count; ++node) {
    const auto state = incumbent_state_features(node);
    for (int32_t edge = edge_offsets_[node];
         edge < edge_offsets_[node + 1]; ++edge) {
      ranked_local_edges[edge] = edge;
      if (!search_config_.classical_behavior) {
        ranking_energy[edge] = edge_energy(
            node, edge_to_[edge], edge, edge_field, edge_additive,
            multipliers, coupler_weights, coupler_bias, state.data(),
            edge_risk, risk_penalty);
      }
    }
    std::stable_sort(
        ranked_local_edges.begin() + edge_offsets_[node],
        ranked_local_edges.begin() + edge_offsets_[node + 1],
        [&](int32_t lhs, int32_t rhs) {
          if (search_config_.classical_behavior)
            return proximity_[lhs] < proximity_[rhs];
          return ranking_energy[lhs] == ranking_energy[rhs]
                     ? proximity_[lhs] < proximity_[rhs]
                     : ranking_energy[lhs] < ranking_energy[rhs];
        });
  }

  int32_t moves = 0;
  int32_t full_evaluations = 0;
  int32_t incremental_rebuilds = 0;
  int32_t full_rebuilds = 0;
  int64_t rebuilt_nodes = 0;
  while (!checklist.empty()) {
    const int32_t anchor = checklist.front();
    checklist.pop_front();
    in_queue[anchor] = 0;
    ++visits[anchor];
    if (anchor < problem_.depot_count || node_route[anchor] < 0) {
      continue;
    }
    if (search_config_.srr_dont_look && dont_look[anchor])
      continue;
    Solution best_move;
    AcceptedPlan best_plan;
    StructuralMove best_structural;
    const GuidanceValue current_guidance = total_guidance;
    const std::vector<float> anchor_state =
        incumbent_state_features(anchor);
    double best_guided_energy = std::numeric_limits<double>::infinity();
    const auto found_first = [&]() {
      return search_config_.srr_first_improvement && best_move.feasible;
    };
    const auto consider = [&](const std::vector<int32_t> &trial,
                              StructuralMove structural) {
      if (found_first())
        return;
      ++full_evaluations;
      const Solution candidate = evaluate(trial);
      record_screening(candidate);
      if (better(candidate, solution) && better(candidate, best_move)) {
        best_move = candidate;
        best_plan = {};
        best_structural = structural;
      }
    };
    const auto piece = [](int32_t route, int32_t begin, int32_t end,
                          bool reverse = false) {
      return SequencePiece{route, begin, end, reverse, -1};
    };
    const auto singleton_piece = [](int32_t node) {
      return SequencePiece{-1, 0, 0, false, node};
    };
    const auto append_piece = [](PlannedRoute &plan,
                                 const SequencePiece &part) {
      if (part.singleton >= 0 || part.begin < part.end)
        plan.pieces.push_back(part);
    };
    const auto consider_plans =
        [&](const std::vector<PlannedRoute> &plans,
            const std::vector<int32_t> &affected_routes,
            const auto &materialize) {
          if (plans.empty() || found_first())
            return;
          std::array<int32_t, 2> affected{-1, -1};
          int32_t affected_count = 0;
          double distance = solution.distance;
          double prize = solution.collected_prize;
          double missed_penalty = solution.missed_penalty;
          GuidanceValue guided = current_guidance;
          for (int32_t route : affected_routes) {
            if (route < 0 ||
                route >= static_cast<int32_t>(cached_routes.size())) {
              continue;
            }
            bool duplicate = false;
            for (int32_t index = 0; index < affected_count; ++index)
              duplicate |= affected[index] == route;
            if (duplicate)
              continue;
            if (affected_count == static_cast<int32_t>(affected.size()))
              return;
            affected[affected_count++] = route;
            const CachedRoute &old = cached_routes[route];
            distance -= old.distance;
            prize -= old.summary.prize;
            missed_penalty += old.summary.penalty;
            guided = subtract_guidance(guided, old.guidance);
          }

          bool feasible = true;
          bool structurally_invalid_plan = false;
          std::vector<SequenceSummary> planned_sequences;
          planned_sequences.reserve(plans.size());
          for (const PlannedRoute &plan : plans) {
            const SequenceSummary sequence = summarize_plan(plan);
            planned_sequences.push_back(sequence);
            if (sequence.empty) {
              structurally_invalid_plan = true;
              feasible = false;
              continue;
            }
            if (problem_.has(BACKHAUL_ORDER) &&
                sequence.backhaul_violation) {
              feasible = false;
              continue;
            }
            const double cost = route_distance(plan.depot, sequence);
            distance += cost;
            prize += sequence.prize;
            missed_penalty -= sequence.penalty;
            guided = add_guidance(guided, guided_plan(plan));
            if (problem_.has(ROUTE_LIMIT) &&
                cost > problem_.route_limit + FEASIBILITY_EPS) {
              feasible = false;
            }
            if (problem_.has(TOUR_LIMIT) &&
                route_distance(plan.depot, sequence, true) >
                    problem_.tour_limit + FEASIBILITY_EPS) {
              feasible = false;
            }
            if (plan.depot >= 0 && !time_feasible(plan.depot, sequence)) {
              feasible = false;
            }
            // Mixed pickup/backhaul loads use the exact linear guard below;
            // nonnegative deliveries have an exact scalar prefix test here.
            if (problem_.has(CAPACITY) && !sequence.has_backhaul &&
                (problem_.capacity + sequence.min_load_delta <
                     -FEASIBILITY_EPS ||
                 problem_.capacity + sequence.max_load_delta >
                     problem_.capacity + FEASIBILITY_EPS)) {
              feasible = false;
            }
          }
          const bool needs_planned_resource =
              trace != nullptr &&
              (trace->screened_edges.size() < MAX_SCREENING_LABELS ||
               search_config_.verify_screening_resources);
          const std::optional<ResourceEvaluation> planned_resource =
              needs_planned_resource
                  ? evaluate_planned_resources(
                        plans, planned_sequences, affected_routes)
                  : std::nullopt;
          if (search_config_.verify_screening_resources &&
              planned_resource.has_value()) {
            std::vector<int32_t> verification_route;
            if (!materialize(verification_route)) {
              // A no-op plan never contributes a changed edge label.
            } else {
              const ResourceEvaluation expected =
                  evaluate_resources(verification_route);
              if (!expected.structurally_valid) {
                ++trace->screening_verification_failures;
              } else {
                for (int32_t channel = 0; channel < FIELD_CHANNEL_COUNT;
                     ++channel) {
                  const float binding_error = std::abs(
                      expected.binding[channel] -
                      planned_resource->binding[channel]);
                  const float violation_error = std::abs(
                      expected.violation[channel] -
                      planned_resource->violation[channel]);
                  if (binding_error > 2.0e-5f ||
                      violation_error > 2.0e-5f) {
                    ++trace->screening_verification_failures;
                    ++trace->screening_verification_failures_by_channel[
                        channel];
                  }
                }
              }
            }
          }
          if (!feasible) {
            if (trace != nullptr &&
                trace->screened_edges.size() < MAX_SCREENING_LABELS &&
                !structurally_invalid_plan) {
              if (planned_resource.has_value()) {
                record_planned_screening(plans, *planned_resource);
              } else {
                std::vector<int32_t> trial;
                if (materialize(trial)) {
                  Solution candidate;
                  candidate.route = std::move(trial);
                  record_screening(candidate);
                }
              }
            }
            return;
          }

          Solution scored;
          scored.feasible = true;
          scored.distance = static_cast<float>(distance);
          scored.collected_prize = static_cast<float>(prize);
          scored.missed_penalty =
              static_cast<float>(std::max(missed_penalty, 0.0));
          switch (problem_.objective) {
          case Objective::MIN_DISTANCE:
            scored.objective = scored.distance;
            break;
          case Objective::MAX_PRIZE:
            scored.objective = scored.collected_prize;
            break;
          case Objective::MIN_DISTANCE_PLUS_PENALTY:
            scored.objective = scored.distance + scored.missed_penalty;
            break;
          }
          const double planned_energy =
              guidance_energy(guided, anchor_state.data());
          if (search_config_.classical_behavior) {
            if (!better(scored, solution) || !better(scored, best_move))
              return;
          } else {
            // The guided energy is evaluated with anchor-specific coupled
            // multipliers, so it is not a global potential. Using it as the
            // acceptance test lets moves anchored at different nodes lower each
            // other's local energy while raising the objective, oscillating
            // forever (SRR only ever accepts feasible moves, so the penalty
            // terms price nothing here). Gate acceptance on the global
            // objective -- a monotone, bounded-below potential -- exactly as
            // the classical path does, and use the guided energy only to pick
            // among strictly-improving moves. This keeps the field's influence
            // on move selection while guaranteeing SRR terminates.
            if (!better(scored, solution) ||
                planned_energy >= best_guided_energy - 1.0e-12)
              return;
          }
          std::vector<int32_t> trial;
          if (!materialize(trial))
            return;
          ++full_evaluations;
          Solution candidate = evaluate(trial);
          if (planned_resource.has_value()) {
            record_planned_screening(plans, *planned_resource);
          } else {
            record_screening(candidate);
          }
          if (!candidate.feasible)
            return;
          if (search_config_.classical_behavior) {
            if (better(candidate, solution) && better(candidate, best_move)) {
              best_move = std::move(candidate);
              best_plan.valid = true;
              best_plan.plans = plans;
              best_plan.resources = planned_resource;
              best_structural = {};
            }
          } else {
            if (!better(candidate, solution))
              return;
            best_move = std::move(candidate);
            best_guided_energy = planned_energy;
            best_plan.valid = true;
            best_plan.plans = plans;
            best_plan.resources = planned_resource;
            best_structural = {};
          }
        };
    const auto new_plan = [&](int32_t route) {
      PlannedRoute plan;
      plan.depot = cached_routes[route].depot;
      plan.slot = route;
      return plan;
    };
    const auto consider_relocate_plan =
        [&](int32_t segment_start, int32_t target, int32_t length,
            bool before) {
          const int32_t source_route = node_route[segment_start];
          const int32_t target_route = node_route[target];
          if (source_route < 0 || target_route < 0 || length <= 0)
            return;
          const int32_t source_position = node_local[segment_start];
          const int32_t target_position = node_local[target];
          const int32_t source_size = static_cast<int32_t>(
              cached_routes[source_route].sequence.nodes.size());
          const int32_t target_size = static_cast<int32_t>(
              cached_routes[target_route].sequence.nodes.size());
          const int32_t actual = std::min(length, source_size - source_position);
          if (actual <= 0 ||
              (source_route == target_route &&
               target_position >= source_position &&
               target_position < source_position + actual)) {
            return;
          }
          const SequencePiece moved = piece(
              source_route, source_position, source_position + actual);
          if (!pickup_closed(moved))
            return;
          std::vector<PlannedRoute> plans;
          if (source_route == target_route) {
            PlannedRoute plan = new_plan(source_route);
            if (source_position > target_position) {
              const int32_t insertion =
                  before ? target_position : target_position + 1;
              append_piece(plan, piece(source_route, 0, insertion));
              append_piece(plan, moved);
              append_piece(plan,
                           piece(source_route, insertion, source_position));
              append_piece(plan, piece(source_route,
                                       source_position + actual, source_size));
            } else {
              const int32_t insertion =
                  before ? target_position : target_position + 1;
              append_piece(plan, piece(source_route, 0, source_position));
              append_piece(plan, piece(source_route, source_position + actual,
                                       insertion));
              append_piece(plan, moved);
              append_piece(plan,
                           piece(source_route, insertion, source_size));
            }
            plans.push_back(std::move(plan));
          } else {
            PlannedRoute source = new_plan(source_route);
            append_piece(source, piece(source_route, 0, source_position));
            append_piece(source, piece(source_route, source_position + actual,
                                       source_size));
            PlannedRoute destination = new_plan(target_route);
            const int32_t insertion =
                before ? target_position : target_position + 1;
            append_piece(destination, piece(target_route, 0, insertion));
            append_piece(destination, moved);
            append_piece(destination,
                         piece(target_route, insertion, target_size));
            plans.push_back(std::move(source));
            plans.push_back(std::move(destination));
          }
          consider_plans(
              plans, {source_route, target_route},
              [&](std::vector<int32_t> &trial) {
                return before
                           ? relocate_before(solution.route, segment_start,
                                             target, length, trial)
                           : relocate(solution.route, segment_start, target,
                                      length, trial);
              });
        };
    const auto consider_swap_plan = [&](int32_t lhs, int32_t rhs) {
      const int32_t lhs_route = node_route[lhs];
      const int32_t rhs_route = node_route[rhs];
      if (lhs_route < 0 || rhs_route < 0 || lhs == rhs)
        return;
      if (problem_.has(PICKUP_DELIVERY) &&
          (problem_.delivery_of_pickup[lhs] >= 0 ||
           problem_.pickup_of_delivery[lhs] >= 0 ||
           problem_.delivery_of_pickup[rhs] >= 0 ||
           problem_.pickup_of_delivery[rhs] >= 0)) {
        return;
      }
      const int32_t lhs_position = node_local[lhs];
      const int32_t rhs_position = node_local[rhs];
      std::vector<PlannedRoute> plans;
      if (lhs_route == rhs_route) {
        int32_t first = lhs_position;
        int32_t second = rhs_position;
        int32_t first_node = lhs;
        int32_t second_node = rhs;
        if (first > second) {
          std::swap(first, second);
          std::swap(first_node, second_node);
        }
        PlannedRoute plan = new_plan(lhs_route);
        append_piece(plan, piece(lhs_route, 0, first));
        append_piece(plan, singleton_piece(second_node));
        append_piece(plan, piece(lhs_route, first + 1, second));
        append_piece(plan, singleton_piece(first_node));
        append_piece(plan, piece(lhs_route, second + 1,
                                 static_cast<int32_t>(
                                     cached_routes[lhs_route]
                                         .sequence.nodes.size())));
        plans.push_back(std::move(plan));
      } else {
        PlannedRoute first = new_plan(lhs_route);
        append_piece(first, piece(lhs_route, 0, lhs_position));
        append_piece(first, singleton_piece(rhs));
        append_piece(first, piece(lhs_route, lhs_position + 1,
                                  static_cast<int32_t>(
                                      cached_routes[lhs_route]
                                          .sequence.nodes.size())));
        PlannedRoute second = new_plan(rhs_route);
        append_piece(second, piece(rhs_route, 0, rhs_position));
        append_piece(second, singleton_piece(lhs));
        append_piece(second, piece(rhs_route, rhs_position + 1,
                                   static_cast<int32_t>(
                                       cached_routes[rhs_route]
                                           .sequence.nodes.size())));
        plans.push_back(std::move(first));
        plans.push_back(std::move(second));
      }
      consider_plans(plans, {lhs_route, rhs_route},
                     [&](std::vector<int32_t> &trial) {
                       return swap_nodes(solution.route, lhs, rhs, trial);
                     });
    };
    const auto consider_two_opt_plan = [&](int32_t lhs, int32_t rhs) {
      if (!reversal_safe())
        return;
      const int32_t route = node_route[lhs];
      if (route < 0 || route != node_route[rhs])
        return;
      int32_t first = node_local[lhs];
      int32_t last = node_local[rhs];
      if (first > last)
        std::swap(first, last);
      if (last <= first + 1)
        return;
      PlannedRoute plan = new_plan(route);
      append_piece(plan, piece(route, 0, first + 1));
      append_piece(plan, piece(route, first + 1, last + 1, true));
      append_piece(plan,
                   piece(route, last + 1,
                         static_cast<int32_t>(
                             cached_routes[route].sequence.nodes.size())));
      consider_plans({plan}, {route}, [&](std::vector<int32_t> &trial) {
        return two_opt(solution.route, lhs, rhs, trial);
      });
    };
    const auto consider_two_opt_star_plan = [&](int32_t lhs, int32_t rhs) {
      if (!problem_.multi_route)
        return;
      const int32_t lhs_route = node_route[lhs];
      const int32_t rhs_route = node_route[rhs];
      if (lhs_route < 0 || rhs_route < 0 || lhs_route == rhs_route)
        return;
      const int32_t lhs_cut = node_local[lhs] + 1;
      const int32_t rhs_cut = node_local[rhs] + 1;
      if (problem_.has(PICKUP_DELIVERY) &&
          (cached_routes[lhs_route].open_pickups[lhs_cut] != 0 ||
           cached_routes[rhs_route].open_pickups[rhs_cut] != 0)) {
        return;
      }
      PlannedRoute first = new_plan(lhs_route);
      append_piece(first, piece(lhs_route, 0, lhs_cut));
      append_piece(first, piece(rhs_route, rhs_cut,
                                static_cast<int32_t>(
                                    cached_routes[rhs_route]
                                        .sequence.nodes.size())));
      PlannedRoute second = new_plan(rhs_route);
      append_piece(second, piece(rhs_route, 0, rhs_cut));
      append_piece(second, piece(lhs_route, lhs_cut,
                                 static_cast<int32_t>(
                                     cached_routes[lhs_route]
                                         .sequence.nodes.size())));
      consider_plans({first, second}, {lhs_route, rhs_route},
                     [&](std::vector<int32_t> &trial) {
                       return two_opt_star(solution.route, lhs, rhs, trial);
                     });
    };
    const auto consider_exchange_plan =
        [&](int32_t lhs, int32_t lhs_length, int32_t rhs,
            int32_t rhs_length) {
          const int32_t lhs_route = node_route[lhs];
          const int32_t rhs_route = node_route[rhs];
          if (lhs_route < 0 || rhs_route < 0 || lhs_route == rhs_route ||
              lhs_length <= 0 || rhs_length <= 0) {
            return;
          }
          const int32_t lhs_position = node_local[lhs];
          const int32_t rhs_position = node_local[rhs];
          const int32_t lhs_size = static_cast<int32_t>(
              cached_routes[lhs_route].sequence.nodes.size());
          const int32_t rhs_size = static_cast<int32_t>(
              cached_routes[rhs_route].sequence.nodes.size());
          const int32_t lhs_actual =
              std::min(lhs_length, lhs_size - lhs_position);
          const int32_t rhs_actual =
              std::min(rhs_length, rhs_size - rhs_position);
          if (lhs_actual <= 0 || rhs_actual <= 0)
            return;
          const SequencePiece lhs_segment =
              piece(lhs_route, lhs_position, lhs_position + lhs_actual);
          const SequencePiece rhs_segment =
              piece(rhs_route, rhs_position, rhs_position + rhs_actual);
          if (!pickup_closed(lhs_segment) || !pickup_closed(rhs_segment))
            return;
          PlannedRoute first = new_plan(lhs_route);
          append_piece(first, piece(lhs_route, 0, lhs_position));
          append_piece(first, rhs_segment);
          append_piece(first,
                       piece(lhs_route, lhs_position + lhs_actual, lhs_size));
          PlannedRoute second = new_plan(rhs_route);
          append_piece(second, piece(rhs_route, 0, rhs_position));
          append_piece(second, lhs_segment);
          append_piece(second,
                       piece(rhs_route, rhs_position + rhs_actual, rhs_size));
          consider_plans(
              {first, second}, {lhs_route, rhs_route},
              [&](std::vector<int32_t> &trial) {
                return exchange_segments(solution.route, lhs, lhs_length, rhs,
                                         rhs_length, trial);
              });
        };
    const auto consider_delete_plan = [&](int32_t node) {
      const int32_t route = node_route[node];
      if (route < 0)
        return;
      if (problem_.has(PICKUP_DELIVERY) &&
          (problem_.delivery_of_pickup[node] >= 0 ||
           problem_.pickup_of_delivery[node] >= 0)) {
        return;
      }
      const int32_t position = node_local[node];
      PlannedRoute plan = new_plan(route);
      append_piece(plan, piece(route, 0, position));
      append_piece(plan,
                   piece(route, position + 1,
                         static_cast<int32_t>(
                             cached_routes[route].sequence.nodes.size())));
      consider_plans({plan}, {route}, [&](std::vector<int32_t> &trial) {
        trial = solution.route;
        const auto position =
            std::find(trial.begin(), trial.end(), node);
        if (position == trial.end())
          return false;
        trial.erase(position);
        return true;
      });
    };
    const auto consider_insert_plan = [&](int32_t node, int32_t target,
                                          bool before) {
      const int32_t route = node_route[target];
      if (route < 0 || node_route[node] >= 0)
        return;
      const int32_t position = node_local[target] + (before ? 0 : 1);
      PlannedRoute plan = new_plan(route);
      append_piece(plan, piece(route, 0, position));
      append_piece(plan, singleton_piece(node));
      append_piece(plan,
                   piece(route, position,
                         static_cast<int32_t>(
                             cached_routes[route].sequence.nodes.size())));
      consider_plans({plan}, {route}, [&](std::vector<int32_t> &trial) {
        return before ? insert_before(solution.route, node, target, trial)
                      : insert_after(solution.route, node, target, trial);
      });
    };
    const auto consider_replace_plan = [&](int32_t served,
                                           int32_t replacement) {
      const int32_t route = node_route[served];
      if (route < 0 || node_route[replacement] >= 0)
        return;
      const int32_t position = node_local[served];
      PlannedRoute plan = new_plan(route);
      append_piece(plan, piece(route, 0, position));
      append_piece(plan, singleton_piece(replacement));
      append_piece(plan,
                   piece(route, position + 1,
                         static_cast<int32_t>(
                             cached_routes[route].sequence.nodes.size())));
      consider_plans({plan}, {route}, [&](std::vector<int32_t> &trial) {
        return exchange_nodes(solution.route, served, replacement, trial);
      });
    };
    const auto consider_relocate_pair_plan = [&](int32_t pair_node,
                                                  int32_t after) {
      int32_t pickup = pair_node;
      int32_t delivery = problem_.delivery_of_pickup[pair_node];
      if (delivery < 0) {
        pickup = problem_.pickup_of_delivery[pair_node];
        delivery = pair_node;
      }
      if (pickup < problem_.depot_count || delivery < problem_.depot_count ||
          after == pickup || after == delivery)
        return;
      const int32_t source_route = node_route[pickup];
      const int32_t target_route = node_route[after];
      if (source_route < 0 || target_route < 0 ||
          node_route[delivery] != source_route)
        return;
      const int32_t pickup_position = node_local[pickup];
      const int32_t delivery_position = node_local[delivery];
      const int32_t target_position = node_local[after];
      const int32_t source_size = static_cast<int32_t>(
          cached_routes[source_route].sequence.nodes.size());
      const int32_t target_size = static_cast<int32_t>(
          cached_routes[target_route].sequence.nodes.size());
      std::vector<PlannedRoute> plans;
      if (source_route != target_route) {
        PlannedRoute source = new_plan(source_route);
        const int32_t first = std::min(pickup_position, delivery_position);
        const int32_t second = std::max(pickup_position, delivery_position);
        append_piece(source, piece(source_route, 0, first));
        append_piece(source, piece(source_route, first + 1, second));
        append_piece(source, piece(source_route, second + 1, source_size));

        PlannedRoute target = new_plan(target_route);
        append_piece(target, piece(target_route, 0, target_position + 1));
        append_piece(target, singleton_piece(pickup));
        append_piece(target, singleton_piece(delivery));
        append_piece(target,
                     piece(target_route, target_position + 1, target_size));
        plans.push_back(std::move(source));
        plans.push_back(std::move(target));
      } else {
        PlannedRoute route = new_plan(source_route);
        std::array<int32_t, 3> special{
            pickup_position, delivery_position, target_position};
        std::sort(special.begin(), special.end());
        int32_t begin = 0;
        for (int32_t position : special) {
          append_piece(route, piece(source_route, begin, position));
          if (position == target_position) {
            append_piece(route, singleton_piece(after));
            append_piece(route, singleton_piece(pickup));
            append_piece(route, singleton_piece(delivery));
          }
          begin = position + 1;
        }
        append_piece(route, piece(source_route, begin, source_size));
        plans.push_back(std::move(route));
      }
      consider_plans(plans, {source_route, target_route},
                     [&](std::vector<int32_t> &trial) {
                       return relocate_pair(solution.route, pair_node, after,
                                            trial);
                     });
    };

      if (!problem_.has(VISIT_ALL)) {
        consider_delete_plan(anchor);
      }

      int32_t anchor_position = -1;
      const auto find_anchor_position = [&]() {
        if (anchor_position >= 0)
          return anchor_position;
        const auto found =
            std::find(solution.route.begin(), solution.route.end(), anchor);
        if (found != solution.route.end())
          anchor_position =
              static_cast<int32_t>(found - solution.route.begin());
        return anchor_position;
      };
      if (problem_.multi_route && !found_first()) {
        // A boundary token closes the preceding route and starts the next.
        // Removing it merges routes; changing it reassigns the next route.
        const int32_t route_slot = node_route[anchor];
        const int32_t route_local = node_local[anchor];
        const int32_t route_size = static_cast<int32_t>(
            cached_routes[route_slot].sequence.nodes.size());
        const int32_t previous_slot = cached_routes[route_slot].previous;
        const int32_t next_slot = cached_routes[route_slot].next;
        if (route_local == 0 && previous_slot >= 0 &&
            find_anchor_position() > 1) {
          std::vector<int32_t> trial = solution.route;
          trial.erase(trial.begin() + anchor_position - 1);
          consider(trial,
                   {StructuralMoveKind::MERGE_ROUTES, previous_slot,
                    route_slot, -1, -1});
        }
        if (!found_first() && route_local + 1 == route_size &&
            next_slot >= 0 &&
            find_anchor_position() >= 0 &&
            anchor_position + 2 < static_cast<int32_t>(solution.route.size())) {
          std::vector<int32_t> trial = solution.route;
          trial.erase(trial.begin() + anchor_position + 1);
          consider(trial,
                   {StructuralMoveKind::MERGE_ROUTES, route_slot, next_slot,
                    -1, -1});
        }
      }

      int32_t candidate_rank = 0;
      int32_t served_candidate_rank = 0;
      int32_t examined_candidates = 0;
      for (int32_t rank = edge_offsets_[anchor];
           rank < edge_offsets_[anchor + 1]; ++rank) {
        if (found_first() ||
            examined_candidates >= search_config_.srr_candidate_limit)
          break;
        ++examined_candidates;
        const int32_t edge = ranked_local_edges[rank];
        const int32_t candidate_node = edge_to_[edge];
        if (candidate_node == anchor)
          continue;
        std::vector<int32_t> trial;
        if (candidate_node < problem_.depot_count) {
          if (!problem_.multi_route)
            continue;
          if (find_anchor_position() < 0)
            continue;
          if (anchor_position + 1 <
                  static_cast<int32_t>(solution.route.size()) &&
              solution.route[anchor_position + 1] >= problem_.depot_count) {
            trial = solution.route;
            trial.insert(trial.begin() + anchor_position + 1, candidate_node);
            consider(trial,
                     {StructuralMoveKind::SPLIT_ROUTE, node_route[anchor], -1,
                      node_local[anchor], candidate_node});
          }
          if (found_first())
            break;
          int32_t route_start = anchor_position - 1;
          while (route_start >= 0 &&
                 solution.route[route_start] >= problem_.depot_count) {
            --route_start;
          }
          if (route_start >= 0 &&
              solution.route[route_start] != candidate_node) {
            trial = solution.route;
            trial[route_start] = candidate_node;
            consider(trial,
                     {StructuralMoveKind::REASSIGN_DEPOT,
                      node_route[anchor], -1, -1, candidate_node});
          }
          continue;
        }
        ++candidate_rank;

        const bool candidate_served = node_route[candidate_node] >= 0;
        if (!candidate_served) {
          if (!problem_.has(VISIT_ALL)) {
            consider_insert_plan(candidate_node, anchor, false);
            if (!found_first())
              consider_insert_plan(candidate_node, anchor, true);
            if (!found_first())
              consider_replace_plan(anchor, candidate_node);
          }
          continue;
        }
        ++served_candidate_rank;
        // DyNACO's hot loop is deliberately small: relocate, swap, 2-opt*,
        // then intra-route 2-opt. The generic plan evaluator below supplies
        // the schema-specific legality that its CVRP-only O(1) deltas lack.
        consider_relocate_plan(anchor, candidate_node, 1, false);
        if (!found_first())
          consider_relocate_plan(anchor, candidate_node, 1, true);
        if (!found_first())
          consider_swap_plan(anchor, candidate_node);
        if (!found_first())
          consider_two_opt_star_plan(anchor, candidate_node);
        if (!found_first())
          consider_two_opt_plan(anchor, candidate_node);

        if (!found_first() && search_config_.srr_extended_operators) {
          for (int32_t length = 1;
               length <= search_config_.or_opt_max_segment; ++length) {
            consider_relocate_plan(candidate_node, anchor, length, false);
            if (!found_first() && candidate_rank <= SRR_DIRECTED_CANDIDATES)
              consider_relocate_plan(candidate_node, anchor, length, true);
            if (found_first())
              break;
          }
        }
        if (!found_first() && search_config_.srr_extended_operators &&
            problem_.multi_route &&
            node_route[anchor] != node_route[candidate_node] &&
            served_candidate_rank <= SRR_STRING_CANDIDATES) {
          for (int32_t length = 2;
               length <= search_config_.or_opt_max_segment; ++length) {
            consider_exchange_plan(anchor, 1, candidate_node, length);
            if (!found_first())
              consider_exchange_plan(anchor, length, candidate_node, 1);
            if (!found_first())
              consider_exchange_plan(anchor, length, candidate_node, length);
            if (found_first())
              break;
          }
        }
        if (!found_first() && problem_.has(PICKUP_DELIVERY)) {
          consider_relocate_pair_plan(anchor, candidate_node);
          if (!found_first())
            consider_relocate_pair_plan(candidate_node, anchor);
        }
      }
    if (best_move.feasible) {
      std::vector<int32_t> old_route;
      if (!best_plan.valid && !best_structural.valid())
        old_route = solution.route;
      solution = std::move(best_move);
      ++moves;
      std::vector<int32_t> touched;
      if (best_plan.valid) {
        if (!replace_planned_routes(best_plan.plans, touched, rebuilt_nodes)) {
          throw std::runtime_error(
              "accepted planned SRR move could not update its route cache");
        }
        ++incremental_rebuilds;
        verify_incremental_cache();
      } else if (best_structural.valid()) {
        if (!apply_structural_move(best_structural, touched, rebuilt_nodes)) {
          throw std::runtime_error(
              "accepted structural SRR move could not update its route cache");
        }
        ++incremental_rebuilds;
        verify_incremental_cache();
      } else {
        rebuild_cache();
        rebuild_resource_cache();
        touched = changed_scope(old_route, solution.route);
        ++full_rebuilds;
        rebuilt_nodes += static_cast<int64_t>(solution.route.size());
      }
      if (trace != nullptr &&
          trace->screened_edges.size() < MAX_SCREENING_LABELS) {
        current_resource = best_plan.resources.has_value()
                               ? *best_plan.resources
                               : evaluate_resources(solution.route);
      }
      for (int32_t node : touched) {
        dont_look[node] = 0;
        enqueue(node);
        const int32_t route_slot = node_route[node];
        const int32_t local = node_local[node];
        if (route_slot < 0 || local < 0)
          continue;
        const std::vector<int32_t> &nodes =
            cached_routes[route_slot].sequence.nodes;
        if (local > 0) {
          dont_look[nodes[local - 1]] = 0;
          enqueue(nodes[local - 1]);
        }
        if (local + 1 < static_cast<int32_t>(nodes.size())) {
          dont_look[nodes[local + 1]] = 0;
          enqueue(nodes[local + 1]);
        }
        if (problem_.depot_count == 0 && !problem_.open_route &&
            !nodes.empty()) {
          const int32_t previous =
              nodes[(local + nodes.size() - 1) % nodes.size()];
          const int32_t next = nodes[(local + 1) % nodes.size()];
          dont_look[previous] = 0;
          dont_look[next] = 0;
          enqueue(previous);
          enqueue(next);
        }
      }
    } else if (search_config_.srr_dont_look) {
      dont_look[anchor] = 1;
    }
  }
  solution.srr_moves = moves;
  solution.srr_evaluations = full_evaluations;
  solution.srr_incremental_rebuilds = incremental_rebuilds;
  solution.srr_full_rebuilds = full_rebuilds;
  solution.srr_rebuilt_nodes = rebuilt_nodes;
  for (int32_t count : visits) {
    if (count > 0)
      ++solution.srr_scope_nodes;
    if (count > 1)
      solution.srr_revisits += count - 1;
  }
  return solution;
}

Solution RoutingDecoder::perturb(uint64_t rollout_seed, const float *edge_field,
                                 const float *edge_additive,
                                 const float *multipliers,
                                 const float *coupler_weights,
                                 const float *coupler_bias,
                                 const float *edge_risk,
                                 float risk_penalty, RolloutTrace *trace,
                                 bool greedy) const {
  const Solution source = evaluate(incumbent_route_);
  if (!source.feasible) {
    return construct(rollout_seed, edge_field, edge_additive, multipliers,
                     coupler_weights, coupler_bias, edge_risk, risk_penalty,
                     trace, greedy);
  }
  std::mt19937_64 rng(rollout_seed);
  Solution raw = source;
  std::vector<uint8_t> used(problem_.node_count, 0);
  std::vector<int32_t> active;
  for (int32_t node : raw.route) {
    if (node >= problem_.depot_count)
      active.push_back(node);
  }
  if (active.empty())
    return raw;
  std::uniform_int_distribution<size_t> choose_start(0, active.size() - 1);
  int32_t current = active[greedy ? 0 : choose_start(rng)];
  used[current] = 1;
  std::vector<int32_t> initial_scope = {current};
  int32_t changed = 0;
  std::vector<int32_t> position(problem_.node_count, -1);
  const auto rebuild_positions = [&]() {
    std::fill(position.begin(), position.end(), -1);
    for (int32_t index = 0; index < static_cast<int32_t>(raw.route.size());
         ++index) {
      const int32_t node = raw.route[index];
      if (node >= problem_.depot_count)
        position[node] = index;
    }
  };
  rebuild_positions();
  for (int32_t attempt = 0; attempt < search_config_.max_perturb_attempts &&
                            changed < search_config_.min_changed_edges;
       ++attempt) {
    bool accepted = false;
    const std::vector<OrderedChoice> order = perturbation_order(
        current, used, rng, edge_field, edge_additive, multipliers,
        coupler_weights, coupler_bias, edge_risk, risk_penalty, greedy);
    std::vector<int32_t> valid_indices;
    valid_indices.reserve(order.size());
    State prefix_state;
    const bool has_prefix =
        trace != nullptr && incumbent_prefix_state(current, prefix_state);
    if (has_prefix)
      record_feasibility_labels(trace, prefix_state);
    for (const OrderedChoice &choice : order) {
      valid_indices.push_back(choice.local_index);
    }
    const std::vector<float> live_state =
        incumbent_state_features(current);
    for (size_t order_index = 0; order_index < order.size(); ++order_index) {
      const OrderedChoice &choice = order[order_index];
      const int32_t chosen = choice.node;
      double maximum = -std::numeric_limits<double>::infinity();
      for (size_t index = order_index; index < order.size(); ++index)
        maximum = std::max(maximum, order[index].log_weight);
      double total = 0.0;
      for (size_t index = order_index; index < order.size(); ++index)
        total += std::exp(order[index].log_weight - maximum);
      const double log_probability =
          choice.log_weight - maximum -
          std::log(std::max(total, static_cast<double>(EPS)));
      record_decision(trace, current, valid_indices, choice.local_index,
                      valid_indices.size() > 1,
                      static_cast<float>(log_probability),
                      live_state.data());
      valid_indices.erase(valid_indices.begin());
      std::vector<int32_t> lengths;
      for (int32_t length = 1; length <= search_config_.or_opt_max_segment;
           ++length) {
        lengths.push_back(length);
      }
      if (!greedy)
        std::shuffle(lengths.begin(), lengths.end(), rng);
      for (int32_t length : lengths) {
        std::vector<int32_t> trial = raw.route;
        const int32_t after = position[current];
        if (after < 0)
          break;

        if (chosen < problem_.depot_count) {
          if (!problem_.multi_route ||
              (after + 1 < static_cast<int32_t>(trial.size()) &&
               trial[after + 1] < problem_.depot_count)) {
            continue;
          }
          trial.insert(trial.begin() + after + 1, chosen);
        } else {
          const int32_t start = position[chosen];
          if (start < 0) {
            if (problem_.has(VISIT_ALL) || length != 1)
              continue;
            trial.insert(trial.begin() + after + 1, chosen);
          } else {
            int32_t actual = 0;
            while (actual < length &&
                   start + actual < static_cast<int32_t>(trial.size()) &&
                   trial[start + actual] >= problem_.depot_count) {
              ++actual;
            }
            if (actual == 0 || (after >= start && after < start + actual))
              continue;
            const std::vector<int32_t> segment(trial.begin() + start,
                                               trial.begin() + start + actual);
            trial.erase(trial.begin() + start, trial.begin() + start + actual);
            const int32_t shifted_after =
                start < after ? after - actual : after;
            trial.insert(trial.begin() + shifted_after + 1, segment.begin(),
                         segment.end());
          }
        }
        if (trial == raw.route)
          continue;
        Solution candidate = evaluate(trial);
        if (!candidate.feasible)
          continue;
        raw = std::move(candidate);
        rebuild_positions();
        initial_scope.push_back(current);
        initial_scope.push_back(chosen);
        if (chosen >= problem_.depot_count) {
          used[chosen] = 1;
          current = chosen;
        } else {
          active.clear();
          for (int32_t node : raw.route) {
            if (node >= problem_.depot_count && !used[node])
              active.push_back(node);
          }
          if (!active.empty()) {
            std::uniform_int_distribution<size_t> choose(0, active.size() - 1);
            current = active[greedy ? 0 : choose(rng)];
            used[current] = 1;
          }
        }
        initial_scope = changed_scope(incumbent_route_, raw.route, &changed);
        accepted = true;
        break;
      }
      if (accepted)
        break;
    }
    if (!accepted) {
      active.clear();
      for (int32_t node : raw.route) {
        if (node >= problem_.depot_count && !used[node])
          active.push_back(node);
      }
      if (active.empty())
        break;
      std::uniform_int_distribution<size_t> choose(0, active.size() - 1);
      current = active[greedy ? 0 : choose(rng)];
      used[current] = 1;
      initial_scope.push_back(current);
    }
  }

  raw.raw_objective = raw.objective;
  raw.changed_edges = changed;
  Solution refined =
      scope_restricted_refine(raw, initial_scope, edge_field, edge_additive,
                              multipliers, coupler_weights, coupler_bias,
                              edge_risk, risk_penalty, trace);
  refined.raw_objective = raw.raw_objective;
  refined.changed_edges = raw.changed_edges;
  return refined;
}

std::vector<Solution> RoutingDecoder::sample(const float *edge_field,
                                             const float *edge_additive,
                                             const float *multipliers,
                                             const float *coupler_weights,
                                             const float *coupler_bias,
                                             const float *edge_risk,
                                             float risk_penalty,
                                             DecisionTrace *trace) {
  validate_guidance(edge_field, edge_additive, multipliers, coupler_weights,
                    coupler_bias, edge_risk, risk_penalty);
  std::vector<Solution> solutions(n_rollouts_);
  std::vector<RolloutTrace> rollout_traces(trace == nullptr ? 0 : n_rollouts_);
  const uint64_t generation_seed = splitmix64(seed_ ^ generation_++);
  const int32_t thread_count = std::min(n_rollouts_, omp_get_max_threads());
#pragma omp parallel for schedule(static) num_threads(thread_count)
  for (int32_t rollout = 0; rollout < n_rollouts_; ++rollout) {
    const uint64_t rollout_seed = splitmix64(generation_seed + rollout);
    RolloutTrace *rollout_trace = trace == nullptr ? nullptr : &rollout_traces[rollout];
    solutions[rollout] =
        incumbent_route_.empty()
            ? construct(rollout_seed, edge_field, edge_additive, multipliers,
                        coupler_weights, coupler_bias, edge_risk, risk_penalty,
                        rollout_trace,
                        rollout < std::max(1, n_rollouts_ / 2))
            : perturb(rollout_seed, edge_field, edge_additive, multipliers,
                      coupler_weights, coupler_bias, edge_risk, risk_penalty,
                      rollout_trace);
  }
  if (trace != nullptr) {
    *trace = DecisionTrace{};
    trace->starts.reserve(n_rollouts_ + 1);
    trace->starts.push_back(0);
    trace->valid_offsets.push_back(0);
    for (RolloutTrace &rollout : rollout_traces) {
      trace->current_nodes.insert(trace->current_nodes.end(),
                                  rollout.current_nodes.begin(),
                                  rollout.current_nodes.end());
      const int32_t valid_base =
          static_cast<int32_t>(trace->valid_indices.size());
      trace->valid_indices.insert(trace->valid_indices.end(),
                                  rollout.valid_indices.begin(),
                                  rollout.valid_indices.end());
      for (size_t index = 1; index < rollout.valid_offsets.size(); ++index)
        trace->valid_offsets.push_back(valid_base + rollout.valid_offsets[index]);
      trace->chosen_indices.insert(trace->chosen_indices.end(),
                                   rollout.chosen_indices.begin(),
                                   rollout.chosen_indices.end());
      trace->stochastic.insert(trace->stochastic.end(), rollout.stochastic.begin(),
                               rollout.stochastic.end());
      trace->log_probabilities.insert(trace->log_probabilities.end(),
                                      rollout.log_probabilities.begin(),
                                      rollout.log_probabilities.end());
      trace->live_state.insert(trace->live_state.end(), rollout.live_state.begin(),
                               rollout.live_state.end());
      trace->feasibility_edges.insert(trace->feasibility_edges.end(),
                                      rollout.feasibility_edges.begin(),
                                      rollout.feasibility_edges.end());
      trace->feasibility_risk_labels.insert(
          trace->feasibility_risk_labels.end(),
          rollout.feasibility_risk_labels.begin(),
          rollout.feasibility_risk_labels.end());
      trace->screened_edges.insert(trace->screened_edges.end(),
                                   rollout.screened_edges.begin(),
                                   rollout.screened_edges.end());
      trace->screened_resource_delta.insert(
          trace->screened_resource_delta.end(),
          rollout.screened_resource_delta.begin(),
          rollout.screened_resource_delta.end());
      trace->screening_fast_evaluations += rollout.screening_fast_evaluations;
      trace->screening_fallback_evaluations +=
          rollout.screening_fallback_evaluations;
      trace->screening_verification_failures +=
          rollout.screening_verification_failures;
      for (int32_t channel = 0; channel < FIELD_CHANNEL_COUNT; ++channel) {
        trace->screening_verification_failures_by_channel[channel] +=
            rollout.screening_verification_failures_by_channel[channel];
      }
      trace->starts.push_back(
          static_cast<int32_t>(trace->current_nodes.size()));
    }
  }
  return solutions;
}

Solution RoutingDecoder::sample_greedy(
    const float *edge_field, const float *edge_additive,
    const float *multipliers, const float *coupler_weights,
    const float *coupler_bias, const float *edge_risk,
    float risk_penalty) const {
  validate_guidance(edge_field, edge_additive, multipliers, coupler_weights,
                    coupler_bias, edge_risk, risk_penalty);
  const uint64_t deterministic_seed = splitmix64(seed_);
  return incumbent_route_.empty()
             ? construct(deterministic_seed, edge_field, edge_additive,
                         multipliers, coupler_weights, coupler_bias, edge_risk,
                         risk_penalty, nullptr, true)
             : perturb(deterministic_seed, edge_field, edge_additive,
                       multipliers, coupler_weights, coupler_bias, edge_risk,
                       risk_penalty, nullptr, true);
}

bool RoutingDecoder::better(const Solution &lhs, const Solution &rhs) const {
  if (!lhs.feasible) {
    return false;
  }
  if (!rhs.feasible) {
    return true;
  }
  if (problem_.objective == Objective::MAX_PRIZE) {
    if (std::abs(lhs.objective - rhs.objective) > FEASIBILITY_EPS) {
      return lhs.objective > rhs.objective;
    }
    return lhs.distance < rhs.distance;
  }
  return lhs.objective < rhs.objective;
}

Solution RoutingDecoder::solve(int32_t iterations, const float *edge_field,
                               const float *edge_additive,
                               const float *multipliers,
                               const float *coupler_weights,
                               const float *coupler_bias,
                               const float *edge_risk,
                               float risk_penalty) {
  if (iterations <= 0) {
    throw std::invalid_argument("iterations must be positive");
  }
  validate_guidance(edge_field, edge_additive, multipliers, coupler_weights,
                    coupler_bias, edge_risk, risk_penalty);
  std::vector<float> working_field;
  std::vector<float> working_additive;
  std::vector<float> working_risk;
  if (edge_field != nullptr) {
    working_field.assign(
        edge_field,
        edge_field + static_cast<size_t>(edge_to_.size()) *
                         resource_count());
  }
  if (edge_additive != nullptr) {
    working_additive.assign(
        edge_additive,
        edge_additive + static_cast<size_t>(edge_to_.size()) *
                            resource_count());
  }
  if (edge_risk != nullptr)
    working_risk.assign(edge_risk, edge_risk + edge_to_.size());
  for (int32_t iteration = 0; iteration < iterations; ++iteration) {
    std::vector<Solution> solutions = sample(
        working_field.empty() ? nullptr : working_field.data(),
        working_additive.empty() ? nullptr : working_additive.data(),
        multipliers, coupler_weights, coupler_bias,
        working_risk.empty() ? nullptr : working_risk.data(), risk_penalty);
    Solution iteration_best;
    for (const Solution &solution : solutions) {
      if (better(solution, iteration_best)) {
        iteration_best = solution;
      }
    }
    if (!iteration_best.feasible) {
      continue;
    }
    const bool improved = better(iteration_best, best_solution_);
    if (improved) {
      best_solution_ = iteration_best;
    }
    if (improved && best_solution_.route != incumbent_route_) {
      build_candidate_graph(best_solution_.route,
                            working_field.empty() ? nullptr : &working_field,
                            working_additive.empty() ? nullptr
                                                     : &working_additive,
                            working_risk.empty() ? nullptr : &working_risk);
    }
  }
  if (!best_solution_.feasible) {
    best_solution_.error = "decoder did not construct a feasible solution";
  }
  return best_solution_;
}

void RoutingDecoder::set_incumbent(const std::vector<int32_t> &route) {
  const Solution solution = evaluate(route);
  if (!solution.feasible) {
    throw std::invalid_argument("incumbent is infeasible: " + solution.error);
  }
  build_candidate_graph(route);
  if (better(solution, best_solution_)) {
    best_solution_ = solution;
  }
}

void RoutingDecoder::set_candidate_resource_quotas(
    const std::vector<float> &quotas) {
  if (quotas.size() != static_cast<size_t>(resource_count()))
    throw std::invalid_argument(
        "candidate resource quotas must match resource_count");
  double total = 0.0;
  for (float quota : quotas) {
    if (!std::isfinite(quota) || quota < 0.0f || quota > 1.0f)
      throw std::invalid_argument(
          "candidate resource quotas must be finite values in [0, 1]");
    total += quota;
  }
  if (total > 1.0 + FEASIBILITY_EPS)
    throw std::invalid_argument("candidate resource quotas must sum to at most 1");
  candidate_resource_quotas_ = quotas;
}

Solution RoutingDecoder::evaluate(const std::vector<int32_t> &route) const {
  Solution failed;
  failed.route = route;
  if (route.empty()) {
    failed.error = "route must not be empty";
    return failed;
  }
  if (problem_.depot_count > 0 &&
      (route.front() < 0 || route.front() >= problem_.depot_count)) {
    failed.error = "route must start at a depot";
    return failed;
  }
  if (problem_.depot_count == 0 &&
      (route.front() < 0 || route.front() >= problem_.node_count)) {
    failed.error = "route starts with an invalid node";
    return failed;
  }

  std::vector<uint8_t> visited(problem_.node_count, 0);
  int32_t current = route.front();
  int32_t route_depot = problem_.depot_count > 0 ? current : -1;
  const int32_t start_node = current;
  int32_t visited_customers = 0;
  int32_t open_pickups = 0;
  int32_t remaining_positive = 0;
  int32_t remaining_negative = 0;
  bool at_depot = problem_.depot_count > 0;
  bool route_has_backhaul = false;
  float load = problem_.capacity;
  float route_distance = 0.0f;
  float current_time = 0.0f;
  float distance = 0.0f;
  float collected_prize = 0.0f;
  float served_penalty = 0.0f;
  float total_penalty = 0.0f;
  int32_t off_graph_edges = 0;

  for (int32_t node = problem_.depot_count; node < problem_.node_count;
       ++node) {
    total_penalty += problem_.penalty[node];
    if (problem_.demand[node] > FEASIBILITY_EPS)
      ++remaining_positive;
    else if (problem_.demand[node] < -FEASIBILITY_EPS)
      ++remaining_negative;
  }
  const auto reload = [&]() {
    if (!problem_.has(CAPACITY) || remaining_positive > 0)
      return problem_.capacity;
    return remaining_negative > 0 ? 0.0f : problem_.capacity;
  };
  const auto is_complete = [&]() {
    if (problem_.depot_count == 0)
      return visited_customers == problem_.customer_count();
    if (problem_.has(VISIT_ALL)) {
      return visited_customers == problem_.customer_count() &&
             (!problem_.multi_route || at_depot);
    }
    return visited_customers > 0 && at_depot;
  };

  if (problem_.depot_count == 0) {
    visited[current] = 1;
    visited_customers = 1;
    collected_prize = problem_.prize[current];
    served_penalty = problem_.penalty[current];
    if (problem_.demand[current] > FEASIBILITY_EPS)
      --remaining_positive;
    else if (problem_.demand[current] < -FEASIBILITY_EPS)
      --remaining_negative;
  } else {
    load = reload();
  }

  for (size_t index = 1; index < route.size(); ++index) {
    if (is_complete()) {
      failed.error = "route continues after the problem is complete";
      return failed;
    }
    const int32_t next = route[index];
    if (next < 0 || next >= problem_.node_count) {
      failed.error = "node index is out of range";
      return failed;
    }
    if (find_edge(current, next) < 0)
      ++off_graph_edges;

    if (next < problem_.depot_count) {
      bool depot_allowed = problem_.depot_count > 0 && !at_depot &&
                           (problem_.multi_route ||
                            !problem_.has(VISIT_ALL));
      if (problem_.has(PICKUP_DELIVERY) && open_pickups != 0)
        depot_allowed = false;
      if (problem_.has(PRIZE_QUOTA) &&
          collected_prize + FEASIBILITY_EPS < problem_.prize_quota &&
          visited_customers < problem_.customer_count()) {
        depot_allowed = false;
      }
      if (!depot_allowed) {
        failed.error = "route contains an infeasible transition to node " +
                       std::to_string(next);
        return failed;
      }
      if (!problem_.open_route)
        distance += problem_.dist(current, route_depot);
      current = next;
      route_depot = next;
      at_depot = true;
      route_has_backhaul = false;
      route_distance = 0.0f;
      current_time = 0.0f;
      load = reload();
      continue;
    }

    if (visited[next]) {
      failed.error = "route contains an infeasible transition to node " +
                     std::to_string(next);
      return failed;
    }
    const int32_t pickup = problem_.pickup_of_delivery[next];
    if (problem_.has(PICKUP_DELIVERY) && pickup >= 0 && !visited[pickup]) {
      failed.error = "route contains an infeasible transition to node " +
                     std::to_string(next);
      return failed;
    }
    const float next_load = load - problem_.demand[next];
    if (problem_.has(CAPACITY) &&
        (next_load < -FEASIBILITY_EPS ||
         next_load > problem_.capacity + FEASIBILITY_EPS)) {
      failed.error = "route contains an infeasible transition to node " +
                     std::to_string(next);
      return failed;
    }
    if (problem_.has(BACKHAUL_ORDER) && route_has_backhaul &&
        problem_.demand[next] > FEASIBILITY_EPS) {
      failed.error = "route contains an infeasible transition to node " +
                     std::to_string(next);
      return failed;
    }

    const float edge = problem_.dist(current, next);
    const float next_route_distance = route_distance + edge;
    if (problem_.has(ROUTE_LIMIT)) {
      float required = next_route_distance;
      if (!problem_.open_route)
        required += problem_.dist(next, route_depot);
      if (required > problem_.route_limit + FEASIBILITY_EPS) {
        failed.error = "route contains an infeasible transition to node " +
                       std::to_string(next);
        return failed;
      }
    }
    if (problem_.has(TOUR_LIMIT) &&
        next_route_distance + problem_.dist(next, route_depot) >
            problem_.tour_limit + FEASIBILITY_EPS) {
      failed.error = "route contains an infeasible transition to node " +
                     std::to_string(next);
      return failed;
    }
    const float arrival = std::max(current_time + edge,
                                   problem_.tw_start[next]);
    if (problem_.has(TIME_WINDOWS) &&
        (arrival > problem_.tw_end[next] + FEASIBILITY_EPS ||
         (!problem_.open_route &&
          arrival + problem_.service_time[next] +
                  problem_.dist(next, route_depot) >
              problem_.tw_end[route_depot] + FEASIBILITY_EPS))) {
      failed.error = "route contains an infeasible transition to node " +
                     std::to_string(next);
      return failed;
    }

    distance += edge;
    route_distance = next_route_distance;
    current_time = arrival + problem_.service_time[next];
    load = next_load;
    route_has_backhaul |= problem_.demand[next] < -FEASIBILITY_EPS;
    if (problem_.delivery_of_pickup[next] >= 0)
      ++open_pickups;
    if (pickup >= 0)
      --open_pickups;
    current = next;
    at_depot = false;
    visited[next] = 1;
    ++visited_customers;
    collected_prize += problem_.prize[next];
    served_penalty += problem_.penalty[next];
    if (problem_.demand[next] > FEASIBILITY_EPS)
      --remaining_positive;
    else if (problem_.demand[next] < -FEASIBILITY_EPS)
      --remaining_negative;
  }

  if (!is_complete()) {
    failed.error = "route ended before satisfying the completion condition";
    return failed;
  }
  if (problem_.has(PICKUP_DELIVERY) && open_pickups != 0) {
    failed.error = "route ended with an undelivered pickup";
    return failed;
  }
  if (problem_.has(VISIT_ALL) && !problem_.multi_route && !at_depot &&
      !problem_.open_route) {
    const int32_t end = problem_.depot_count == 0 ? start_node : route_depot;
    distance += problem_.dist(current, end);
    if (find_edge(current, end) < 0)
      ++off_graph_edges;
  }

  // The legacy evaluator above remains the frozen parity oracle. Explicit
  // algebra rows are replayed alongside it from the same route so schema-only
  // resources participate in incumbent validation and SRR fallback checks.
  std::vector<float> algebra(resource_count(), 0.0f);
  for (int32_t resource_index = 0; resource_index < resource_count();
       ++resource_index) {
    if (!resource(resource_index).is_legacy())
      algebra[resource_index] = resource(resource_index).initial;
  }
  int32_t algebra_current = route.front();
  const auto extend_algebra = [&](int32_t next, bool force_route_end) {
    const bool depot = next < problem_.depot_count;
    for (int32_t resource_index = 0; resource_index < resource_count();
         ++resource_index) {
      const ResourceSpec &spec = resource(resource_index);
      if (spec.is_legacy())
        continue;
      float value = algebra[resource_index];
      const bool event_reset =
          !spec.reset_nodes.empty() && spec.reset_nodes[next];
      if (spec.edge_uses_distance)
        value +=
            spec.edge_coefficient * problem_.dist(algebra_current, next);
      if (!spec.edge_values.empty())
        value += spec.edge_coefficient *
                 spec.edge_values[static_cast<size_t>(algebra_current) *
                                      problem_.node_count +
                                  next];
      if (!spec.node_values.empty())
        value += spec.node_coefficient * spec.node_values[next];
      const bool check = spec.bound_check == BoundCheck::TRANSITION ||
                         ((depot || force_route_end) &&
                          spec.bound_check == BoundCheck::ROUTE_END);
      if (check && (value < spec.lower - FEASIBILITY_EPS ||
                    value > spec.upper + FEASIBILITY_EPS)) {
        failed.error = "resource bound failed: " + spec.name;
        return false;
      }
      if ((depot && spec.reset_at_depot) || event_reset)
        value = spec.reset_value;
      else if (depot && spec.scope == ResourceScope::ROUTE)
        value = spec.initial;
      algebra[resource_index] = value;
    }
    algebra_current = next;
    return true;
  };
  for (size_t route_index = 1; route_index < route.size(); ++route_index) {
    if (!extend_algebra(route[route_index], false))
      return failed;
  }
  if (problem_.has(VISIT_ALL) && !problem_.multi_route && !at_depot &&
      !problem_.open_route) {
    const int32_t end = problem_.depot_count == 0 ? start_node : route_depot;
    if (!extend_algebra(end, true))
      return failed;
  }
  for (int32_t resource_index = 0; resource_index < resource_count();
       ++resource_index) {
    const ResourceSpec &spec = resource(resource_index);
    if (!spec.is_legacy() && spec.bound_check == BoundCheck::SOLUTION_END &&
        (algebra[resource_index] < spec.lower - FEASIBILITY_EPS ||
         algebra[resource_index] > spec.upper + FEASIBILITY_EPS)) {
      failed.error = "terminal resource bound failed: " + spec.name;
      return failed;
    }
  }

  Solution solution;
  solution.route = route;
  solution.distance = distance;
  solution.collected_prize = collected_prize;
  solution.missed_penalty = total_penalty - served_penalty;
  solution.off_graph_edges = off_graph_edges;
  switch (problem_.objective) {
  case Objective::MIN_DISTANCE:
    solution.objective = solution.distance;
    break;
  case Objective::MAX_PRIZE:
    solution.objective = solution.collected_prize;
    break;
  case Objective::MIN_DISTANCE_PLUS_PENALTY:
    solution.objective = solution.distance + solution.missed_penalty;
    break;
  }
  solution.feasible = std::isfinite(solution.objective);
  if (!solution.feasible)
    solution.error = "route objective is not finite";
  else
    solution.raw_objective = solution.objective;
  return solution;
}

ResourceEvaluation RoutingDecoder::evaluate_resources(
    const std::vector<int32_t> &route) const {
  ResourceEvaluation result;
  result.violation.assign(resource_count(), 0.0f);
  result.binding.assign(resource_count(), 0.0f);
  if (route.empty()) {
    result.error = "route must not be empty";
    return result;
  }
  if (route.front() < 0 || route.front() >= problem_.node_count ||
      (problem_.depot_count > 0 && route.front() >= problem_.depot_count)) {
    result.error = "route has an invalid start node";
    return result;
  }

  const float capacity_scale =
      resource_scale(static_cast<int32_t>(FieldChannel::CAPACITY));
  const float time_scale =
      resource_scale(static_cast<int32_t>(FieldChannel::TIME_WINDOW));
  const float route_scale =
      resource_scale(static_cast<int32_t>(FieldChannel::ROUTE_LIMIT));
  const float tour_scale =
      resource_scale(static_cast<int32_t>(FieldChannel::TOUR_LIMIT));
  const float quota_scale =
      resource_scale(static_cast<int32_t>(FieldChannel::PRIZE_QUOTA));
  std::vector<uint8_t> visited(problem_.node_count, 0);
  int32_t current = route.front();
  int32_t route_depot = problem_.depot_count > 0 ? current : -1;
  bool at_depot = problem_.depot_count > 0;
  bool route_has_backhaul = false;
  bool any_backhaul = false;
  int32_t open_pickups = 0;
  int32_t max_open_pickups = 0;
  int32_t pair_count = 0;
  int32_t precedence_violations = 0;
  int32_t backhaul_violations = 0;
  int32_t visited_customers = 0;
  const auto initial_load = [&](size_t begin) {
    bool has_linehaul = false;
    bool has_backhaul = false;
    for (size_t index = begin; index < route.size(); ++index) {
      const int32_t node = route[index];
      if (node < problem_.depot_count)
        break;
      if (node >= problem_.node_count)
        continue;
      has_linehaul |= problem_.demand[node] > FEASIBILITY_EPS;
      has_backhaul |= problem_.demand[node] < -FEASIBILITY_EPS;
    }
    return has_linehaul || !has_backhaul ? problem_.capacity : 0.0f;
  };
  float load = initial_load(problem_.depot_count > 0 ? 1 : 0);
  float route_positive = 0.0f;
  float route_negative = 0.0f;
  float route_distance = 0.0f;
  float route_time = 0.0f;
  float collected_prize = 0.0f;
  float capacity_excess = 0.0f;
  float time_warp = 0.0f;
  float route_excess = 0.0f;
  float tour_excess = 0.0f;
  float max_route_ratio = 0.0f;
  float max_tour_ratio = 0.0f;
  float min_time_slack = time_scale_;

  for (int32_t node = problem_.depot_count; node < problem_.node_count;
       ++node) {
    pair_count += problem_.delivery_of_pickup[node] >= 0 ? 1 : 0;
  }
  if (problem_.depot_count == 0) {
    visited[current] = 1;
    ++visited_customers;
    collected_prize += problem_.prize[current];
  }

  const auto close_route = [&]() {
    float closed_distance = route_distance;
    if (!problem_.open_route && route_depot >= 0 && !at_depot)
      closed_distance += problem_.dist(current, route_depot);
    const float closed_tour =
        route_depot >= 0 && !at_depot
            ? route_distance + problem_.dist(current, route_depot)
            : route_distance;
    if (problem_.has(ROUTE_LIMIT)) {
      route_excess += std::max(closed_distance - problem_.route_limit, 0.0f);
      max_route_ratio =
          std::max(max_route_ratio, closed_distance / route_scale);
    }
    if (problem_.has(TOUR_LIMIT)) {
      tour_excess += std::max(closed_tour - problem_.tour_limit, 0.0f);
      max_tour_ratio = std::max(max_tour_ratio, closed_tour / tour_scale);
    }
    if (problem_.has(TIME_WINDOWS) && route_depot >= 0 && !at_depot &&
        !problem_.open_route) {
      const float return_time = route_time + problem_.dist(current, route_depot);
      time_warp +=
          std::max(return_time - problem_.tw_end[route_depot], 0.0f);
    }
    result.binding[static_cast<int32_t>(FieldChannel::CAPACITY)] =
        std::max(result.binding[static_cast<int32_t>(FieldChannel::CAPACITY)],
                 std::max(route_positive, route_negative) / capacity_scale);
    route_positive = 0.0f;
    route_negative = 0.0f;
  };

  for (size_t index = 1; index < route.size(); ++index) {
    const int32_t next = route[index];
    if (next < 0 || next >= problem_.node_count) {
      result.error = "route contains an out-of-range node";
      return result;
    }
    if (next < problem_.depot_count) {
      if (problem_.depot_count == 0 || at_depot) {
        result.error = "route contains an invalid depot transition";
        return result;
      }
      if (open_pickups > 0)
        precedence_violations += open_pickups;
      close_route();
      current = next;
      route_depot = next;
      at_depot = true;
      route_has_backhaul = false;
      route_distance = 0.0f;
      route_time = 0.0f;
      load = initial_load(index + 1);
      open_pickups = 0;
      continue;
    }
    if (visited[next]) {
      result.error = "route visits a customer more than once";
      return result;
    }

    const float travel = problem_.dist(current, next);
    route_distance += travel;
    const float raw_arrival = route_time + travel;
    const float arrival = std::max(raw_arrival, problem_.tw_start[next]);
    if (problem_.has(TIME_WINDOWS)) {
      time_warp += std::max(arrival - problem_.tw_end[next], 0.0f);
      min_time_slack = std::min(
          min_time_slack, std::max(problem_.tw_end[next] - arrival, 0.0f));
    }
    route_time = arrival + problem_.service_time[next];

    const float demand = problem_.demand[next];
    load -= demand;
    route_positive += std::max(demand, 0.0f);
    route_negative += std::max(-demand, 0.0f);
    if (problem_.has(CAPACITY)) {
      capacity_excess =
          std::max({capacity_excess, -load, load - problem_.capacity, 0.0f});
    }
    if (problem_.has(BACKHAUL_ORDER) && route_has_backhaul &&
        demand > FEASIBILITY_EPS) {
      ++backhaul_violations;
    }
    route_has_backhaul |= demand < -FEASIBILITY_EPS;
    any_backhaul |= demand < -FEASIBILITY_EPS;

    const int32_t pickup = problem_.pickup_of_delivery[next];
    if (problem_.has(PICKUP_DELIVERY) && pickup >= 0 && !visited[pickup])
      ++precedence_violations;
    if (problem_.delivery_of_pickup[next] >= 0)
      ++open_pickups;
    if (pickup >= 0 && open_pickups > 0)
      --open_pickups;
    max_open_pickups = std::max(max_open_pickups, open_pickups);

    visited[next] = 1;
    ++visited_customers;
    collected_prize += problem_.prize[next];
    current = next;
    at_depot = false;
  }

  if (!at_depot || problem_.depot_count == 0)
    close_route();
  if (open_pickups > 0)
    precedence_violations += open_pickups;
  if (problem_.has(VISIT_ALL) &&
      visited_customers != problem_.customer_count()) {
    result.error = "route omits required customers";
    return result;
  }

  result.violation[static_cast<int32_t>(FieldChannel::CAPACITY)] =
      capacity_excess / capacity_scale;
  result.violation[static_cast<int32_t>(FieldChannel::TIME_WINDOW)] =
      time_warp / time_scale;
  result.violation[static_cast<int32_t>(FieldChannel::ROUTE_LIMIT)] =
      route_excess / route_scale;
  result.violation[static_cast<int32_t>(FieldChannel::TOUR_LIMIT)] =
      tour_excess / tour_scale;
  result.violation[static_cast<int32_t>(FieldChannel::BACKHAUL_ORDER)] =
      static_cast<float>(backhaul_violations) /
      std::max(problem_.customer_count(), 1);
  result.violation[static_cast<int32_t>(FieldChannel::PICKUP_DELIVERY)] =
      static_cast<float>(precedence_violations) / std::max(pair_count, 1);
  result.violation[static_cast<int32_t>(FieldChannel::PRIZE_QUOTA)] =
      std::max(problem_.prize_quota - collected_prize, 0.0f) / quota_scale;

  result.binding[static_cast<int32_t>(FieldChannel::CAPACITY)] =
      std::clamp(result.binding[static_cast<int32_t>(FieldChannel::CAPACITY)],
                 0.0f, 1.0f);
  result.binding[static_cast<int32_t>(FieldChannel::TIME_WINDOW)] =
      problem_.has(TIME_WINDOWS)
          ? std::clamp(1.0f - min_time_slack / time_scale, 0.0f, 1.0f)
          : 0.0f;
  result.binding[static_cast<int32_t>(FieldChannel::ROUTE_LIMIT)] =
      std::clamp(max_route_ratio, 0.0f, 1.0f);
  result.binding[static_cast<int32_t>(FieldChannel::TOUR_LIMIT)] =
      std::clamp(max_tour_ratio, 0.0f, 1.0f);
  result.binding[static_cast<int32_t>(FieldChannel::BACKHAUL_ORDER)] =
      any_backhaul ? 1.0f : 0.0f;
  result.binding[static_cast<int32_t>(FieldChannel::PICKUP_DELIVERY)] =
      std::clamp(static_cast<float>(max_open_pickups) /
                     std::max(pair_count, 1),
                 0.0f, 1.0f);
  result.binding[static_cast<int32_t>(FieldChannel::PRIZE_QUOTA)] =
      problem_.has(PRIZE_QUOTA)
          ? std::clamp(collected_prize / quota_scale, 0.0f, 1.0f)
          : 0.0f;
  for (int32_t channel = 0; channel < FIELD_CHANNEL_COUNT; ++channel) {
    if (!field_channel_active(channel)) {
      result.violation[channel] = 0.0f;
      result.binding[channel] = 0.0f;
    } else if (result.violation[channel] > FEASIBILITY_EPS) {
      result.binding[channel] = 1.0f;
    }
  }
  std::vector<float> algebra(resource_count(), 0.0f);
  for (int32_t resource_index = FIELD_CHANNEL_COUNT;
       resource_index < resource_count(); ++resource_index) {
    algebra[resource_index] = resource(resource_index).initial;
  }
  int32_t algebra_current = route.front();
  const auto record_custom = [&](int32_t resource_index, float value,
                                 bool check_bound) {
    const ResourceSpec &spec = resource(resource_index);
    const float scale = runtime_resource_scale(resource_index);
    float violation = 0.0f;
    if (check_bound) {
      if (std::isfinite(spec.lower))
        violation = std::max(violation, spec.lower - value);
      if (std::isfinite(spec.upper))
        violation = std::max(violation, value - spec.upper);
    }
    result.violation[resource_index] = std::max(
        result.violation[resource_index], violation / std::max(scale, EPS));
    float binding = 0.0f;
    if (std::isfinite(spec.lower) && std::isfinite(spec.upper)) {
      const float slack = std::min(value - spec.lower, spec.upper - value);
      binding = 1.0f - slack / std::max(0.5f * (spec.upper - spec.lower), EPS);
    } else if (std::isfinite(spec.lower)) {
      binding = 1.0f - (value - spec.lower) / std::max(scale, EPS);
    } else if (std::isfinite(spec.upper)) {
      binding = 1.0f - (spec.upper - value) / std::max(scale, EPS);
    }
    result.binding[resource_index] =
        std::max(result.binding[resource_index],
                 std::clamp(binding, 0.0f, 1.0f));
  };
  const auto extend_custom = [&](int32_t next, bool force_route_end) {
    const bool depot = next < problem_.depot_count;
    for (int32_t resource_index = FIELD_CHANNEL_COUNT;
         resource_index < resource_count(); ++resource_index) {
      const ResourceSpec &spec = resource(resource_index);
      float value = algebra[resource_index];
      const bool event_reset =
          !spec.reset_nodes.empty() && spec.reset_nodes[next];
      if (spec.edge_uses_distance)
        value +=
            spec.edge_coefficient * problem_.dist(algebra_current, next);
      if (!spec.edge_values.empty())
        value += spec.edge_coefficient *
                 spec.edge_values[static_cast<size_t>(algebra_current) *
                                      problem_.node_count +
                                  next];
      if (!spec.node_values.empty())
        value += spec.node_coefficient * spec.node_values[next];
      const bool check = spec.bound_check == BoundCheck::TRANSITION ||
                         ((depot || force_route_end) &&
                          spec.bound_check == BoundCheck::ROUTE_END);
      record_custom(resource_index, value, check);
      if ((depot && spec.reset_at_depot) || event_reset)
        value = spec.reset_value;
      else if (depot && spec.scope == ResourceScope::ROUTE)
        value = spec.initial;
      algebra[resource_index] = value;
    }
    algebra_current = next;
  };
  for (size_t route_index = 1; route_index < route.size(); ++route_index) {
    extend_custom(route[route_index], false);
  }
  if (problem_.has(VISIT_ALL) && !problem_.multi_route && !at_depot &&
      !problem_.open_route) {
    const int32_t end = problem_.depot_count == 0 ? route.front() : route_depot;
    extend_custom(end, true);
  }
  for (int32_t resource_index = FIELD_CHANNEL_COUNT;
       resource_index < resource_count(); ++resource_index) {
    if (resource(resource_index).bound_check == BoundCheck::SOLUTION_END)
      record_custom(resource_index, algebra[resource_index], true);
    if (result.violation[resource_index] > FEASIBILITY_EPS)
      result.binding[resource_index] = 1.0f;
  }
  result.structurally_valid = true;
  return result;
}

std::vector<uint8_t>
RoutingDecoder::mask(const std::vector<int32_t> &prefix) const {
  if (prefix.empty()) {
    std::vector<uint8_t> starts(problem_.node_count, 0);
    const int32_t count =
        problem_.depot_count > 0 ? problem_.depot_count : problem_.node_count;
    std::fill(starts.begin(), starts.begin() + count, 1);
    return starts;
  }
  if (problem_.depot_count > 0 &&
      (prefix.front() < 0 || prefix.front() >= problem_.depot_count)) {
    return std::vector<uint8_t>(problem_.node_count, 0);
  }
  if (problem_.depot_count == 0 &&
      (prefix.front() < 0 || prefix.front() >= problem_.node_count)) {
    return std::vector<uint8_t>(problem_.node_count, 0);
  }
  State state = initial_state(prefix.front());
  for (size_t index = 1; index < prefix.size(); ++index) {
    if (complete(state)) {
      return std::vector<uint8_t>(problem_.node_count, 0);
    }
    std::string error;
    if (!transition(state, prefix[index], error)) {
      return std::vector<uint8_t>(problem_.node_count, 0);
    }
  }
  if (complete(state)) {
    return std::vector<uint8_t>(problem_.node_count, 0);
  }
  return legal_mask(state);
}

} // namespace prism
