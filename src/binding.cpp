#include "decoder.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using prism::BACKHAUL_ORDER;
using prism::CandidateConfig;
using prism::CAPACITY;
using prism::Constraint;
using prism::DecisionTrace;
using prism::Objective;
using prism::PICKUP_DELIVERY;
using prism::PRIZE_QUOTA;
using prism::Problem;
using prism::ROUTE_LIMIT;
using prism::ResourceEvaluation;
using prism::ResourceSpec;
using prism::ResourceOperator;
using prism::ResourceDirection;
using prism::ResourceScope;
using prism::BoundCheck;
using prism::SearchConfig;
using prism::Solution;
using prism::TIME_WINDOWS;
using prism::TOUR_LIMIT;
using prism::RoutingDecoder;
using prism::VISIT_ALL;

namespace {

template <typename T>
py::array_t<T> vector_copy(const std::vector<T> &values,
                           const std::vector<py::ssize_t> &shape) {
  py::array_t<T> result(shape);
  std::copy(values.begin(), values.end(), result.mutable_data());
  return result;
}

std::vector<float> required_matrix(const py::dict &data, const char *key,
                                   int32_t &node_count) {
  if (!data.contains(key)) {
    throw std::invalid_argument(std::string("missing required field '") + key +
                                "'");
  }
  py::array_t<float, py::array::c_style | py::array::forcecast> array =
      data[key]
          .cast<
              py::array_t<float, py::array::c_style | py::array::forcecast>>();
  const py::buffer_info buffer = array.request();
  if (buffer.ndim != 2 || buffer.shape[0] != buffer.shape[1]) {
    throw std::invalid_argument(std::string(key) +
                                " must be a square 2D array");
  }
  node_count = static_cast<int32_t>(buffer.shape[0]);
  const float *values = static_cast<const float *>(buffer.ptr);
  return std::vector<float>(values, values + node_count * node_count);
}

std::vector<float> optional_vector(const py::dict &data, const char *key,
                                   int32_t node_count, float default_value) {
  if (!data.contains(key) || data[key].is_none()) {
    return std::vector<float>(node_count, default_value);
  }
  py::array_t<float, py::array::c_style | py::array::forcecast> array =
      data[key]
          .cast<
              py::array_t<float, py::array::c_style | py::array::forcecast>>();
  const py::buffer_info buffer = array.request();
  if (buffer.ndim != 1 || buffer.shape[0] != node_count) {
    throw std::invalid_argument(std::string(key) +
                                " must have shape (node_count,)");
  }
  const float *values = static_cast<const float *>(buffer.ptr);
  return std::vector<float>(values, values + node_count);
}

std::vector<float> optional_coordinates(const py::dict &data,
                                        int32_t &node_count) {
  if (!data.contains("coordinates") || data["coordinates"].is_none()) {
    return {};
  }
  py::array_t<float, py::array::c_style | py::array::forcecast> array =
      data["coordinates"]
          .cast<
              py::array_t<float, py::array::c_style | py::array::forcecast>>();
  const py::buffer_info buffer = array.request();
  if (buffer.ndim != 2 || buffer.shape[1] != 2 ||
      (node_count != 0 && buffer.shape[0] != node_count)) {
    throw std::invalid_argument("coordinates must have shape (node_count, 2)");
  }
  node_count = static_cast<int32_t>(buffer.shape[0]);
  const float *values = static_cast<const float *>(buffer.ptr);
  return std::vector<float>(values, values + 2 * node_count);
}

bool contains(const std::string &value, const std::string &part) {
  return value.find(part) != std::string::npos;
}

bool is_pctsp(const std::string &name) { return contains(name, "pctsp"); }

bool is_orienteering(const std::string &name) {
  return name == "op" || name == "aop";
}

bool is_vrp(const std::string &name) { return contains(name, "cvrp"); }

bool is_open_vrp(const std::string &name) {
  return contains(name, "ocvrp") || contains(name, "opdcvrp");
}

uint32_t urs_constraints(const std::string &name) {
  uint32_t flags = 0;
  if (!is_orienteering(name) && !is_pctsp(name)) {
    flags |= VISIT_ALL;
  }
  if (is_vrp(name)) {
    flags |= CAPACITY;
  }
  if (contains(name, "bp")) {
    flags |= BACKHAUL_ORDER;
  }
  if (contains(name, "pd")) {
    flags |= PICKUP_DELIVERY;
  }
  if (contains(name, "tw")) {
    flags |= TIME_WINDOWS;
  }
  if (contains(name, "l")) {
    flags |= ROUTE_LIMIT;
  }
  if (is_orienteering(name)) {
    flags |= TOUR_LIMIT;
  }
  if (is_pctsp(name)) {
    flags |= PRIZE_QUOTA;
  }
  return flags;
}

constexpr std::pair<const char *, Constraint> CONSTRAINT_BY_NAME[] = {
    {"visit_all", VISIT_ALL},           {"capacity", CAPACITY},
    {"backhaul_order", BACKHAUL_ORDER}, {"pickup_delivery", PICKUP_DELIVERY},
    {"route_limit", ROUTE_LIMIT},       {"time_windows", TIME_WINDOWS},
    {"tour_limit", TOUR_LIMIT},         {"prize_quota", PRIZE_QUOTA},
};

uint32_t parse_constraints(const py::dict &data, const std::string &name) {
  if (!data.contains("constraints")) {
    return urs_constraints(name);
  }
  uint32_t flags = 0;
  for (const std::string &constraint :
       data["constraints"].cast<std::vector<std::string>>()) {
    bool found = false;
    for (const auto &[candidate, value] : CONSTRAINT_BY_NAME) {
      if (constraint == candidate) {
        flags |= value;
        found = true;
        break;
      }
    }
    if (!found) {
      throw std::invalid_argument("unknown constraint: " + constraint);
    }
  }
  return flags;
}

Objective parse_objective(const py::dict &data, const std::string &name) {
  std::string objective;
  if (data.contains("objective")) {
    objective = data["objective"].cast<std::string>();
  } else if (is_orienteering(name)) {
    objective = "prize";
  } else if (is_pctsp(name)) {
    objective = "distance_plus_penalty";
  } else {
    objective = "distance";
  }
  if (objective == "distance") {
    return Objective::MIN_DISTANCE;
  }
  if (objective == "prize") {
    return Objective::MAX_PRIZE;
  }
  if (objective == "distance_plus_penalty") {
    return Objective::MIN_DISTANCE_PLUS_PENALTY;
  }
  throw std::invalid_argument("unknown objective: " + objective);
}

template <typename T>
T value_or(const py::dict &data, const char *key, T default_value) {
  return data.contains(key) ? data[key].cast<T>() : default_value;
}

float algebra_scalar(const py::dict &problem, py::handle expression,
                     const char *field) {
  if (py::isinstance<py::float_>(expression) ||
      py::isinstance<py::int_>(expression))
    return py::cast<float>(expression);
  if (!py::isinstance<py::dict>(expression))
    throw std::invalid_argument(std::string(field) +
                                " must be a number or scalar reference");
  const py::dict reference = py::reinterpret_borrow<py::dict>(expression);
  if (!reference.contains("scalar"))
    throw std::invalid_argument(std::string(field) +
                                " scalar reference is missing 'scalar'");
  const std::string name = reference["scalar"].cast<std::string>();
  if (!problem.contains(name.c_str()))
    throw std::invalid_argument("missing resource scalar: " + name);
  return problem[name.c_str()].cast<float>();
}

std::vector<float> algebra_node_attribute(const py::dict &problem,
                                          const std::string &name,
                                          int32_t node_count) {
  if (!problem.contains("node_attributes"))
    throw std::invalid_argument("problem is missing node_attributes");
  const py::dict attributes = problem["node_attributes"].cast<py::dict>();
  if (!attributes.contains(name.c_str()))
    throw std::invalid_argument("missing node attribute: " + name);
  py::dict wrapper;
  wrapper["value"] = attributes[name.c_str()];
  return optional_vector(wrapper, "value", node_count, 0.0f);
}

std::vector<float> algebra_edge_attribute(const py::dict &problem,
                                          const std::string &name,
                                          int32_t node_count) {
  if (name == "distance") {
    if (!problem.contains("distance") || problem["distance"].is_none())
      throw std::invalid_argument(
          "distance resource input requires an explicit distance matrix");
    int32_t parsed_count = node_count;
    return required_matrix(problem, "distance", parsed_count);
  }
  if (!problem.contains("edge_attributes"))
    throw std::invalid_argument("problem is missing edge_attributes");
  const py::dict attributes = problem["edge_attributes"].cast<py::dict>();
  if (!attributes.contains(name.c_str()))
    throw std::invalid_argument("missing edge attribute: " + name);
  py::dict wrapper;
  wrapper["value"] = attributes[name.c_str()];
  int32_t parsed_count = node_count;
  return required_matrix(wrapper, "value", parsed_count);
}

std::vector<ResourceSpec> parse_resource_algebra(const py::dict &data,
                                                 int32_t node_count) {
  std::vector<ResourceSpec> result;
  if (!data.contains("resources"))
    return result;
  const py::list rows = data["resources"].cast<py::list>();
  result.reserve(rows.size());
  for (py::handle item : rows) {
    const py::dict row = py::reinterpret_borrow<py::dict>(item);
    ResourceSpec spec;
    if (!row.contains("name") || !row.contains("operator"))
      throw std::invalid_argument(
          "resource row requires 'name' and 'operator'");
    spec.name = row["name"].cast<std::string>();
    const std::string op = row["operator"].cast<std::string>();
    if (op != "affine_accumulator")
      throw std::invalid_argument("unknown resource operator: " + op);
    spec.op = ResourceOperator::AFFINE_ACCUMULATOR;
    spec.state_dim = value_or<int32_t>(row, "state_dim", 1);
    const std::string direction =
        value_or<std::string>(row, "direction", "forward");
    spec.direction = direction == "forward"
                         ? ResourceDirection::FORWARD
                         : direction == "backward"
                               ? ResourceDirection::BACKWARD
                               : direction == "bidirectional"
                                     ? ResourceDirection::BIDIRECTIONAL
                                     : throw std::invalid_argument(
                                           "unknown resource direction: " +
                                           direction);
    const std::string scope = value_or<std::string>(row, "scope", "route");
    spec.scope = scope == "route"
                     ? ResourceScope::ROUTE
                     : scope == "tour"
                           ? ResourceScope::TOUR
                           : scope == "solution"
                                 ? ResourceScope::SOLUTION
                                 : throw std::invalid_argument(
                                       "unknown resource scope: " + scope);
    if (row.contains("initial"))
      spec.initial = algebra_scalar(data, row["initial"], "initial");
    if (row.contains("scale"))
      spec.scale = algebra_scalar(data, row["scale"], "scale");
    if (row.contains("increment")) {
      const py::dict increment = row["increment"].cast<py::dict>();
      const float coefficient = value_or<float>(increment, "coefficient", 1.0f);
      if (increment.contains("edge_attribute")) {
        spec.edge_coefficient = coefficient;
        const std::string attribute =
            increment["edge_attribute"].cast<std::string>();
        if (attribute == "distance")
          spec.edge_uses_distance = true;
        else
          spec.edge_values =
              algebra_edge_attribute(data, attribute, node_count);
      }
      if (increment.contains("node_attribute")) {
        spec.node_coefficient = coefficient;
        spec.node_values = algebra_node_attribute(
            data, increment["node_attribute"].cast<std::string>(), node_count);
      }
    }
    if (row.contains("reset")) {
      const py::dict reset = row["reset"].cast<py::dict>();
      spec.reset_value = reset.contains("value")
                             ? algebra_scalar(data, reset["value"], "reset.value")
                             : spec.initial;
      spec.reset_at_depot = value_or<bool>(reset, "at_depot", false);
      if (reset.contains("node_attribute")) {
        const std::vector<float> flags = algebra_node_attribute(
            data, reset["node_attribute"].cast<std::string>(), node_count);
        spec.reset_nodes.resize(node_count);
        std::transform(flags.begin(), flags.end(), spec.reset_nodes.begin(),
                       [](float value) { return value > 0.5f ? 1 : 0; });
      }
    }
    if (row.contains("bounds")) {
      const py::list bounds = row["bounds"].cast<py::list>();
      if (bounds.size() != 1)
        throw std::invalid_argument(
            "resource algebra v1 requires exactly one bound row");
      const py::dict bound = py::reinterpret_borrow<py::dict>(bounds[0]);
      if (bound.contains("lower"))
        spec.lower = algebra_scalar(data, bound["lower"], "bound.lower");
      if (bound.contains("upper"))
        spec.upper = algebra_scalar(data, bound["upper"], "bound.upper");
      const std::string check =
          value_or<std::string>(bound, "check", "transition");
      spec.bound_check = check == "transition"
                             ? BoundCheck::TRANSITION
                             : check == "route_end"
                                   ? BoundCheck::ROUTE_END
                                   : check == "solution_end"
                                         ? BoundCheck::SOLUTION_END
                                         : throw std::invalid_argument(
                                               "unknown resource bound phase: " +
                                               check);
    }
    result.push_back(std::move(spec));
  }
  return result;
}

void set_pickup_delivery_relations(Problem &problem, const py::dict &data) {
  problem.delivery_of_pickup.assign(problem.node_count, -1);
  problem.pickup_of_delivery.assign(problem.node_count, -1);
  if (!problem.has(PICKUP_DELIVERY)) {
    return;
  }

  if (data.contains("pickup_delivery_pairs")) {
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> pairs =
        data["pickup_delivery_pairs"]
            .cast<py::array_t<int32_t,
                              py::array::c_style | py::array::forcecast>>();
    const py::buffer_info buffer = pairs.request();
    if (buffer.ndim != 2 || buffer.shape[1] != 2) {
      throw std::invalid_argument(
          "pickup_delivery_pairs must have shape (pair_count, 2)");
    }
    const int32_t *values = static_cast<const int32_t *>(buffer.ptr);
    for (py::ssize_t index = 0; index < buffer.shape[0]; ++index) {
      const int32_t pickup = values[2 * index];
      const int32_t delivery = values[2 * index + 1];
      if (pickup < problem.depot_count || pickup >= problem.node_count ||
          delivery < problem.depot_count || delivery >= problem.node_count) {
        throw std::invalid_argument("pickup-delivery node is out of range");
      }
      problem.delivery_of_pickup[pickup] = delivery;
      problem.pickup_of_delivery[delivery] = pickup;
    }
    return;
  }

  const int32_t customers = problem.customer_count();
  if (customers % 2 != 0) {
    throw std::invalid_argument(
        "URS pickup-delivery variants require an even customer count");
  }
  const int32_t pair_count = customers / 2;
  for (int32_t index = 0; index < pair_count; ++index) {
    const int32_t pickup = problem.depot_count + index;
    const int32_t delivery = pickup + pair_count;
    problem.delivery_of_pickup[pickup] = delivery;
    problem.pickup_of_delivery[delivery] = pickup;
  }
}

Problem parse_problem(const py::dict &data) {
  if (!data.contains("name")) {
    throw std::invalid_argument("missing required field 'name'");
  }
  Problem problem;
  problem.name = data["name"].cast<std::string>();
  std::transform(
      problem.name.begin(), problem.name.end(), problem.name.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (data.contains("distance") && !data["distance"].is_none())
    problem.distance = required_matrix(data, "distance", problem.node_count);
  problem.coordinates = optional_coordinates(data, problem.node_count);
  if (problem.node_count == 0) {
    throw std::invalid_argument(
        "one of 'distance' or 'coordinates' must be provided");
  }

  const bool no_depot = problem.name == "tsp" || problem.name == "atsp";
  const int32_t default_depots =
      no_depot ? 0 : (contains(problem.name, "md") ? 3 : 1);
  problem.depot_count = value_or<int32_t>(data, "depot_count", default_depots);
  problem.constraints = parse_constraints(data, problem.name);
  problem.objective = parse_objective(data, problem.name);
  problem.multi_route =
      value_or<bool>(data, "multi_route", is_vrp(problem.name));
  problem.open_route =
      value_or<bool>(data, "open_route", is_open_vrp(problem.name));

  problem.capacity = value_or<float>(data, "capacity", 1.0f);
  problem.route_limit = value_or<float>(data, "route_limit",
                                        std::numeric_limits<float>::infinity());
  const float default_tour_limit = problem.name == "aop" ? 1.0f : 4.0f;
  problem.tour_limit = value_or<float>(
      data, "tour_limit",
      problem.has(TOUR_LIMIT) ? default_tour_limit
                              : std::numeric_limits<float>::infinity());
  problem.prize_quota = value_or<float>(data, "prize_quota", 1.0f);

  problem.demand = optional_vector(data, "demand", problem.node_count, 0.0f);
  problem.prize = optional_vector(data, "prize", problem.node_count, 0.0f);
  problem.penalty = optional_vector(data, "penalty", problem.node_count, 0.0f);
  problem.tw_start =
      optional_vector(data, "tw_start", problem.node_count, 0.0f);
  problem.tw_end = optional_vector(data, "tw_end", problem.node_count,
                                   std::numeric_limits<float>::infinity());
  problem.service_time =
      optional_vector(data, "service_time", problem.node_count, 0.0f);
  set_pickup_delivery_relations(problem, data);
  problem.resources = parse_resource_algebra(data, problem.node_count);
  return problem;
}

CandidateConfig parse_candidate_config(const py::dict &data) {
  CandidateConfig config;
  if (data.empty()) {
    return config;
  }
#define CONFIG_INT(field)                                                      \
  config.field = value_or<int32_t>(data, #field, config.field)
#define CONFIG_FLOAT(field)                                                    \
  config.field = value_or<float>(data, #field, config.field)
  CONFIG_INT(max_candidates);
  if (data.contains("candidate_mode")) {
    const std::string mode = data["candidate_mode"].cast<std::string>();
    if (mode == "schema")
      config.candidate_mode = prism::CandidateMode::SCHEMA;
    else if (mode == "geometric")
      config.candidate_mode = prism::CandidateMode::GEOMETRIC;
    else
      throw std::invalid_argument("unknown candidate_mode: " + mode);
  }
  CONFIG_FLOAT(gamma_unit);
  CONFIG_FLOAT(gamma_wait);
  CONFIG_FLOAT(gamma_time_warp);
  CONFIG_FLOAT(gamma_load_fit);
  CONFIG_FLOAT(gamma_ordering);
  CONFIG_FLOAT(gamma_precedence);
  CONFIG_FLOAT(gamma_route);
  CONFIG_FLOAT(gamma_prize);
#undef CONFIG_FLOAT
#undef CONFIG_INT
  return config;
}

SearchConfig parse_search_config(const py::dict &data) {
  SearchConfig config;
  if (data.empty()) {
    return config;
  }
  config.min_changed_edges =
      value_or<int32_t>(data, "min_changed_edges", config.min_changed_edges);
  config.max_perturb_attempts = value_or<int32_t>(data, "max_perturb_attempts",
                                                  config.max_perturb_attempts);
  config.or_opt_max_segment =
      value_or<int32_t>(data, "or_opt_max_segment", config.or_opt_max_segment);
  config.feasibility_lookahead_depth = value_or<int32_t>(
      data, "feasibility_lookahead_depth",
      config.feasibility_lookahead_depth);
  config.srr_candidate_limit = value_or<int32_t>(
      data, "srr_candidate_limit", config.srr_candidate_limit);
  config.use_srr = value_or<bool>(data, "use_srr", config.use_srr);
  config.classical_behavior =
      value_or<bool>(data, "classical_behavior", config.classical_behavior);
  config.srr_first_improvement = value_or<bool>(
      data, "srr_first_improvement", config.srr_first_improvement);
  config.srr_dont_look =
      value_or<bool>(data, "srr_dont_look", config.srr_dont_look);
  config.srr_extended_operators = value_or<bool>(
      data, "srr_extended_operators", config.srr_extended_operators);
  config.verify_screening_resources = value_or<bool>(
      data, "verify_screening_resources",
      config.verify_screening_resources);
  config.verify_incremental_srr = value_or<bool>(
      data, "verify_incremental_srr", config.verify_incremental_srr);
  return config;
}

py::dict solution_to_dict(const Solution &solution, Objective objective) {
  py::dict result;
  result["route"] = vector_copy<int32_t>(
      solution.route, {static_cast<py::ssize_t>(solution.route.size())});
  result["feasible"] = solution.feasible;
  result["objective"] = solution.objective;
  result["objective_name"] = prism::objective_name(objective);
  result["direction"] = prism::objective_direction(objective);
  result["distance"] = solution.distance;
  result["collected_prize"] = solution.collected_prize;
  result["missed_penalty"] = solution.missed_penalty;
  result["raw_objective"] = solution.raw_objective;
  result["changed_edges"] = solution.changed_edges;
  result["srr_moves"] = solution.srr_moves;
  result["srr_scope_nodes"] = solution.srr_scope_nodes;
  result["srr_revisits"] = solution.srr_revisits;
  result["srr_evaluations"] = solution.srr_evaluations;
  result["srr_incremental_rebuilds"] = solution.srr_incremental_rebuilds;
  result["srr_full_rebuilds"] = solution.srr_full_rebuilds;
  result["srr_rebuilt_nodes"] = solution.srr_rebuilt_nodes;
  result["off_graph_edges"] = solution.off_graph_edges;
  result["error"] = solution.error;
  return result;
}

void parse_guidance(py::object edge_field, py::object edge_additive,
                    py::object multipliers,
                    py::object coupler_weights, py::object coupler_bias,
                    py::object edge_risk,
                    int32_t edge_count, int32_t resource_count,
                    int32_t multiplier_count, int32_t live_state_count,
                    bool classical_behavior,
                    py::array_t<float> &field_storage,
                    py::array_t<float> &additive_storage,
                    py::array_t<float> &multiplier_storage,
                    py::array_t<float> &coupler_weight_storage,
                    py::array_t<float> &coupler_bias_storage,
                    py::array_t<float> &risk_storage,
                    const float *&field_values,
                    const float *&additive_values,
                    const float *&multiplier_values,
                    const float *&coupler_weight_values,
                    const float *&coupler_bias_values,
                    const float *&risk_values) {
  field_values = nullptr;
  additive_values = nullptr;
  multiplier_values = nullptr;
  coupler_weight_values = nullptr;
  coupler_bias_values = nullptr;
  risk_values = nullptr;
  if (classical_behavior) {
    if (!edge_field.is_none() || !edge_additive.is_none() ||
        !multipliers.is_none() ||
        !coupler_weights.is_none() || !coupler_bias.is_none() ||
        !edge_risk.is_none()) {
      throw std::invalid_argument(
          "classical_behavior does not accept edge_field or multipliers");
    }
    return;
  }
  if (edge_field.is_none()) {
    throw std::invalid_argument(
        "edge_field is required when classical_behavior is false");
  }
  field_storage = edge_field.cast<
      py::array_t<float, py::array::c_style | py::array::forcecast>>();
  const py::buffer_info field_buffer = field_storage.request();
  if (field_buffer.ndim != 2 || field_buffer.shape[0] != edge_count ||
      field_buffer.shape[1] != resource_count) {
    throw std::invalid_argument(
        "edge_field must have shape (edge_count, resource_count)");
  }
  field_values = static_cast<const float *>(field_buffer.ptr);
  if (!edge_additive.is_none()) {
    additive_storage = edge_additive.cast<
        py::array_t<float, py::array::c_style | py::array::forcecast>>();
    const py::buffer_info additive_buffer = additive_storage.request();
    if (additive_buffer.ndim != 2 ||
        additive_buffer.shape[0] != edge_count ||
        additive_buffer.shape[1] != resource_count) {
      throw std::invalid_argument(
          "edge_additive must have shape (edge_count, resource_count)");
    }
    additive_values = static_cast<const float *>(additive_buffer.ptr);
  }
  if (!multipliers.is_none()) {
    multiplier_storage = multipliers.cast<
        py::array_t<float, py::array::c_style | py::array::forcecast>>();
    const py::buffer_info multiplier_buffer = multiplier_storage.request();
    if (multiplier_buffer.ndim != 1 ||
        multiplier_buffer.shape[0] != multiplier_count) {
      throw std::invalid_argument(
          "multipliers must have shape (multiplier_count,)");
    }
    multiplier_values = static_cast<const float *>(multiplier_buffer.ptr);
  }
  if (!coupler_weights.is_none()) {
    coupler_weight_storage = coupler_weights.cast<
        py::array_t<float, py::array::c_style | py::array::forcecast>>();
    const py::buffer_info buffer = coupler_weight_storage.request();
    if (buffer.ndim != 2 || buffer.shape[0] != multiplier_count ||
        buffer.shape[1] != live_state_count) {
      throw std::invalid_argument(
          "coupler_weights must have shape (multiplier_count, "
          "live_state_count)");
    }
    coupler_weight_values = static_cast<const float *>(buffer.ptr);
  }
  if (!coupler_bias.is_none()) {
    coupler_bias_storage = coupler_bias.cast<
        py::array_t<float, py::array::c_style | py::array::forcecast>>();
    const py::buffer_info buffer = coupler_bias_storage.request();
    if (buffer.ndim != 1 || buffer.shape[0] != multiplier_count) {
      throw std::invalid_argument(
          "coupler_bias must have shape (multiplier_count,)");
    }
    coupler_bias_values = static_cast<const float *>(buffer.ptr);
  }
  if (!edge_risk.is_none()) {
    risk_storage = edge_risk.cast<
        py::array_t<float, py::array::c_style | py::array::forcecast>>();
    const py::buffer_info buffer = risk_storage.request();
    if (buffer.ndim != 1 || buffer.shape[0] != edge_count) {
      throw std::invalid_argument(
          "edge_risk must have shape (edge_count,)");
    }
    risk_values = static_cast<const float *>(buffer.ptr);
  }
}

py::dict trace_to_dict(const DecisionTrace &trace, int32_t resource_count) {
  py::dict result;
  result["starts"] = vector_copy<int32_t>(
      trace.starts, {static_cast<py::ssize_t>(trace.starts.size())});
  result["current_nodes"] = vector_copy<int32_t>(
      trace.current_nodes,
      {static_cast<py::ssize_t>(trace.current_nodes.size())});
  result["valid_offsets"] = vector_copy<int32_t>(
      trace.valid_offsets,
      {static_cast<py::ssize_t>(trace.valid_offsets.size())});
  result["valid_indices"] = vector_copy<int32_t>(
      trace.valid_indices,
      {static_cast<py::ssize_t>(trace.valid_indices.size())});
  result["chosen_indices"] = vector_copy<int32_t>(
      trace.chosen_indices,
      {static_cast<py::ssize_t>(trace.chosen_indices.size())});
  result["stochastic"] = vector_copy<uint8_t>(
      trace.stochastic, {static_cast<py::ssize_t>(trace.stochastic.size())});
  result["log_probabilities"] = vector_copy<float>(
      trace.log_probabilities,
      {static_cast<py::ssize_t>(trace.log_probabilities.size())});
  result["live_state"] = vector_copy<float>(
      trace.live_state,
      {static_cast<py::ssize_t>(trace.current_nodes.size()),
       resource_count});
  result["feasibility_edges"] = vector_copy<int32_t>(
      trace.feasibility_edges,
      {static_cast<py::ssize_t>(trace.feasibility_edges.size())});
  result["feasibility_risk_labels"] = vector_copy<float>(
      trace.feasibility_risk_labels,
      {static_cast<py::ssize_t>(trace.feasibility_edges.size())});
  result["screened_edges"] = vector_copy<int32_t>(
      trace.screened_edges,
      {static_cast<py::ssize_t>(trace.screened_edges.size())});
  result["screened_resource_delta"] = vector_copy<float>(
      trace.screened_resource_delta,
      {static_cast<py::ssize_t>(trace.screened_edges.size()),
       prism::FIELD_CHANNEL_COUNT});
  result["screening_fast_evaluations"] = trace.screening_fast_evaluations;
  result["screening_fallback_evaluations"] =
      trace.screening_fallback_evaluations;
  result["screening_verification_failures"] =
      trace.screening_verification_failures;
  result["screening_verification_failures_by_channel"] =
      vector_copy<int64_t>(
          std::vector<int64_t>(
              trace.screening_verification_failures_by_channel.begin(),
              trace.screening_verification_failures_by_channel.end()),
          {prism::FIELD_CHANNEL_COUNT});
  return result;
}

class PyDecoder {
public:
  PyDecoder(py::dict problem, py::dict candidate_config,
               py::dict search_config, int32_t n_rollouts, float beta)
      : solver_(parse_problem(problem),
                parse_candidate_config(candidate_config),
                parse_search_config(search_config), n_rollouts, beta) {}

  py::list sample(py::object edge_field, py::object edge_additive,
                  py::object multipliers,
                  py::object coupler_weights, py::object coupler_bias,
                  py::object edge_risk, float risk_penalty) {
    py::array_t<float> field_storage;
    py::array_t<float> additive_storage;
    py::array_t<float> multiplier_storage;
    py::array_t<float> coupler_weight_storage;
    py::array_t<float> coupler_bias_storage;
    py::array_t<float> risk_storage;
    const float *field_values;
    const float *additive_values;
    const float *multiplier_values;
    const float *coupler_weight_values;
    const float *coupler_bias_values;
    const float *risk_values;
    parse_guidance(edge_field, edge_additive, multipliers, coupler_weights,
                   coupler_bias, edge_risk, solver_.edge_count(),
                   solver_.resource_count(), solver_.multiplier_count(),
                   solver_.live_state_feature_count(),
                   solver_.search_config().classical_behavior, field_storage,
                   additive_storage, multiplier_storage,
                   coupler_weight_storage, coupler_bias_storage, risk_storage,
                   field_values,
                   additive_values, multiplier_values, coupler_weight_values,
                   coupler_bias_values, risk_values);
    std::vector<Solution> solutions;
    {
      py::gil_scoped_release release;
      solutions = solver_.sample(field_values, additive_values,
                                 multiplier_values,
                                 coupler_weight_values, coupler_bias_values,
                                 risk_values, risk_penalty);
    }
    py::list result;
    for (const Solution &solution : solutions) {
      result.append(solution_to_dict(solution, solver_.problem().objective));
    }
    return result;
  }

  py::dict sample_traced(py::object edge_field, py::object edge_additive,
                         py::object multipliers,
                         py::object coupler_weights,
                         py::object coupler_bias, py::object edge_risk,
                         float risk_penalty) {
    py::array_t<float> field_storage;
    py::array_t<float> additive_storage;
    py::array_t<float> multiplier_storage;
    py::array_t<float> coupler_weight_storage;
    py::array_t<float> coupler_bias_storage;
    py::array_t<float> risk_storage;
    const float *field_values;
    const float *additive_values;
    const float *multiplier_values;
    const float *coupler_weight_values;
    const float *coupler_bias_values;
    const float *risk_values;
    parse_guidance(edge_field, edge_additive, multipliers, coupler_weights,
                   coupler_bias, edge_risk, solver_.edge_count(),
                   solver_.resource_count(), solver_.multiplier_count(),
                   solver_.live_state_feature_count(),
                   solver_.search_config().classical_behavior, field_storage,
                   additive_storage, multiplier_storage,
                   coupler_weight_storage, coupler_bias_storage, risk_storage,
                   field_values,
                   additive_values, multiplier_values, coupler_weight_values,
                   coupler_bias_values, risk_values);
    std::vector<Solution> solutions;
    DecisionTrace trace;
    {
      py::gil_scoped_release release;
      solutions = solver_.sample(field_values, additive_values,
                                 multiplier_values,
                                 coupler_weight_values, coupler_bias_values,
                                 risk_values, risk_penalty,
                                 &trace);
    }
    py::list serialized;
    for (const Solution &solution : solutions)
      serialized.append(
          solution_to_dict(solution, solver_.problem().objective));
    py::dict result;
    result["solutions"] = std::move(serialized);
    result["trace"] = trace_to_dict(trace, solver_.resource_count());
    result["graph_version"] = solver_.graph_version();
    return result;
  }

  py::dict sample_greedy(py::object edge_field, py::object edge_additive,
                         py::object multipliers, py::object coupler_weights,
                         py::object coupler_bias, py::object edge_risk,
                         float risk_penalty) {
    py::array_t<float> field_storage;
    py::array_t<float> additive_storage;
    py::array_t<float> multiplier_storage;
    py::array_t<float> coupler_weight_storage;
    py::array_t<float> coupler_bias_storage;
    py::array_t<float> risk_storage;
    const float *field_values;
    const float *additive_values;
    const float *multiplier_values;
    const float *coupler_weight_values;
    const float *coupler_bias_values;
    const float *risk_values;
    parse_guidance(edge_field, edge_additive, multipliers, coupler_weights,
                   coupler_bias, edge_risk, solver_.edge_count(),
                   solver_.resource_count(), solver_.multiplier_count(),
                   solver_.live_state_feature_count(),
                   solver_.search_config().classical_behavior, field_storage,
                   additive_storage, multiplier_storage,
                   coupler_weight_storage, coupler_bias_storage, risk_storage,
                   field_values,
                   additive_values, multiplier_values, coupler_weight_values,
                   coupler_bias_values, risk_values);
    Solution solution;
    {
      py::gil_scoped_release release;
      solution = solver_.sample_greedy(
          field_values, additive_values, multiplier_values,
          coupler_weight_values, coupler_bias_values, risk_values,
          risk_penalty);
    }
    return solution_to_dict(solution, solver_.problem().objective);
  }

  py::dict solve(int32_t iterations, py::object edge_field,
                 py::object edge_additive, py::object multipliers,
                 py::object coupler_weights,
                 py::object coupler_bias, py::object edge_risk,
                 float risk_penalty) {
    py::array_t<float> field_storage;
    py::array_t<float> additive_storage;
    py::array_t<float> multiplier_storage;
    py::array_t<float> coupler_weight_storage;
    py::array_t<float> coupler_bias_storage;
    py::array_t<float> risk_storage;
    const float *field_values;
    const float *additive_values;
    const float *multiplier_values;
    const float *coupler_weight_values;
    const float *coupler_bias_values;
    const float *risk_values;
    parse_guidance(edge_field, edge_additive, multipliers, coupler_weights,
                   coupler_bias, edge_risk, solver_.edge_count(),
                   solver_.resource_count(), solver_.multiplier_count(),
                   solver_.live_state_feature_count(),
                   solver_.search_config().classical_behavior, field_storage,
                   additive_storage, multiplier_storage,
                   coupler_weight_storage, coupler_bias_storage, risk_storage,
                   field_values,
                   additive_values, multiplier_values, coupler_weight_values,
                   coupler_bias_values, risk_values);
    Solution solution;
    {
      py::gil_scoped_release release;
      solution = solver_.solve(
          iterations, field_values, additive_values, multiplier_values,
          coupler_weight_values, coupler_bias_values, risk_values,
          risk_penalty);
    }
    return solution_to_dict(solution, solver_.problem().objective);
  }

  py::dict evaluate(
      py::array_t<int32_t, py::array::c_style | py::array::forcecast> route)
      const {
    const py::buffer_info buffer = route.request();
    if (buffer.ndim != 1) {
      throw std::invalid_argument("route must be one-dimensional");
    }
    const int32_t *values = static_cast<const int32_t *>(buffer.ptr);
    const std::vector<int32_t> route_values(values, values + buffer.shape[0]);
    return solution_to_dict(solver_.evaluate(route_values),
                            solver_.problem().objective);
  }

  py::dict evaluate_resources(
      py::array_t<int32_t, py::array::c_style | py::array::forcecast> route)
      const {
    const py::buffer_info buffer = route.request();
    if (buffer.ndim != 1) {
      throw std::invalid_argument("route must be one-dimensional");
    }
    const int32_t *values = static_cast<const int32_t *>(buffer.ptr);
    const ResourceEvaluation evaluation = solver_.evaluate_resources(
        std::vector<int32_t>(values, values + buffer.shape[0]));
    py::dict result;
    result["violation"] = vector_copy<float>(
        evaluation.violation, {solver_.resource_count()});
    result["binding"] =
        vector_copy<float>(evaluation.binding, {solver_.resource_count()});
    result["structurally_valid"] = evaluation.structurally_valid;
    result["error"] = evaluation.error;
    return result;
  }

  py::array_t<uint8_t>
  mask(py::array_t<int32_t, py::array::c_style | py::array::forcecast> prefix)
      const {
    const py::buffer_info buffer = prefix.request();
    if (buffer.ndim != 1) {
      throw std::invalid_argument("prefix must be one-dimensional");
    }
    const int32_t *values = static_cast<const int32_t *>(buffer.ptr);
    const std::vector<int32_t> prefix_values(values, values + buffer.shape[0]);
    const std::vector<uint8_t> legal = solver_.mask(prefix_values);
    return vector_copy<uint8_t>(legal,
                                {static_cast<py::ssize_t>(legal.size())});
  }

  py::dict metadata() const {
    const Problem &problem = solver_.problem();
    py::dict result;
    result["name"] = problem.name;
    result["node_count"] = problem.node_count;
    result["customer_count"] = problem.customer_count();
    result["depot_count"] = problem.depot_count;
    result["constraints"] = prism::constraint_names(problem.constraints);
    result["objective"] = prism::objective_name(problem.objective);
    result["objective_scale"] = solver_.objective_scale();
    result["direction"] = prism::objective_direction(problem.objective);
    result["multi_route"] = problem.multi_route;
    result["open_route"] = problem.open_route;
    result["edge_count"] = solver_.edge_count();
    result["graph_version"] = solver_.graph_version();
    result["guidance_mode"] =
        solver_.search_config().classical_behavior ? "classical" : "field";
    result["max_candidates"] = solver_.candidate_config().max_candidates;
    int32_t maximum_degree = 0;
    const auto &offsets = solver_.edge_offsets();
    for (size_t node = 1; node < offsets.size(); ++node) {
      maximum_degree =
          std::max(maximum_degree, offsets[node] - offsets[node - 1]);
    }
    result["maximum_degree"] = maximum_degree;
    const CandidateConfig &config = solver_.candidate_config();
    bool any_active_resource = false;
    for (const auto &spec : solver_.resources())
      any_active_resource |= spec.active;
    result["candidate_strategy"] =
        config.candidate_mode == prism::CandidateMode::GEOMETRIC ||
                !any_active_resource
            ? "distance"
            : solver_.candidate_resource_quotas().empty()
                  ? "uniform_schema"
                  : "typed_resource_quota";
    result["candidate_resource_quotas"] = vector_copy<float>(
        solver_.candidate_resource_quotas(),
        {static_cast<py::ssize_t>(solver_.candidate_resource_quotas().size())});
    py::dict weights;
    weights["unit"] = config.gamma_unit;
    weights["wait"] = config.gamma_wait;
    weights["time_warp"] = config.gamma_time_warp;
    weights["load_fit"] = config.gamma_load_fit;
    weights["ordering"] = config.gamma_ordering;
    weights["precedence"] = config.gamma_precedence;
    weights["route"] = config.gamma_route;
    weights["prize"] = config.gamma_prize;
    result["proximity_weights"] = weights;
    result["candidate_feature_names"] = prism::candidate_feature_names();
    result["field_channel_names"] = prism::field_channel_names();
    py::list resource_rows;
    for (const ResourceSpec &resource : solver_.resources()) {
      py::dict row;
      row["name"] = resource.name;
      row["active"] = resource.active;
      row["state_dim"] = resource.state_dim;
      row["legacy"] = resource.is_legacy();
      resource_rows.append(std::move(row));
    }
    result["resources"] = std::move(resource_rows);
    result["resource_count"] = solver_.resource_count();
    result["multiplier_count"] = solver_.multiplier_count();
    result["resource_descriptor_version"] = "resource_descriptor_v1";
    result["node_feature_names"] = prism::node_feature_names();
    std::vector<uint8_t> active_channels;
    active_channels.reserve(solver_.resource_count());
    for (const ResourceSpec &resource : solver_.resources())
      active_channels.push_back(static_cast<uint8_t>(resource.active));
    result["field_channel_mask"] = vector_copy<uint8_t>(
        active_channels, {solver_.resource_count()});
    const SearchConfig &search = solver_.search_config();
    py::dict search_values;
    search_values["min_changed_edges"] = search.min_changed_edges;
    search_values["max_perturb_attempts"] = search.max_perturb_attempts;
    search_values["or_opt_max_segment"] = search.or_opt_max_segment;
    search_values["feasibility_lookahead_depth"] =
        search.feasibility_lookahead_depth;
    search_values["srr_candidate_limit"] = search.srr_candidate_limit;
    search_values["use_srr"] = search.use_srr;
    search_values["classical_behavior"] = search.classical_behavior;
    search_values["srr_first_improvement"] = search.srr_first_improvement;
    search_values["srr_dont_look"] = search.srr_dont_look;
    search_values["srr_extended_operators"] =
        search.srr_extended_operators;
    search_values["verify_screening_resources"] =
        search.verify_screening_resources;
    search_values["verify_incremental_srr"] =
        search.verify_incremental_srr;
    result["search"] = search_values;
    return result;
  }

  py::array_t<float> heuristic() const {
    return vector_copy<float>(solver_.heuristic(), {solver_.edge_count()});
  }

  py::array_t<float> proximity() const {
    return vector_copy<float>(solver_.proximity(), {solver_.edge_count()});
  }

  py::array_t<float> edge_features() const {
    return vector_copy<float>(
        solver_.edge_features(),
        {solver_.edge_count(), prism::EDGE_FEATURE_COUNT});
  }

  py::array_t<float> node_features() const {
    return vector_copy<float>(
        solver_.node_features(),
        {solver_.problem().node_count, prism::NODE_FEATURE_COUNT});
  }

  py::array_t<float> incumbent_live_state() const {
    return vector_copy<float>(
        solver_.incumbent_live_state(),
        {solver_.problem().node_count, solver_.live_state_feature_count()});
  }

  py::array_t<float> resource_features() const {
    return vector_copy<float>(
        solver_.resource_features(),
        {solver_.edge_count(), solver_.resource_count()});
  }

  py::array_t<float> resource_pressure() const {
    return vector_copy<float>(
        solver_.resource_pressure(),
        {solver_.edge_count(), solver_.resource_count()});
  }

  py::array_t<float> resource_events() const {
    return vector_copy<float>(
        solver_.resource_events(),
        {solver_.edge_count(), solver_.resource_count()});
  }

  py::array_t<float> resource_scales() const {
    return vector_copy<float>(solver_.resource_scales(),
                              {solver_.resource_count()});
  }

  py::array_t<float> resource_descriptors() const {
    return vector_copy<float>(
        solver_.resource_descriptors(),
        {solver_.resource_count(), prism::RESOURCE_DESCRIPTOR_DIM});
  }

  py::array_t<float> objective_edge_costs() const {
    return vector_copy<float>(solver_.objective_edge_costs(),
                              {solver_.edge_count()});
  }

  py::array_t<int32_t> edge_offsets() const {
    return vector_copy<int32_t>(
        solver_.edge_offsets(),
        {static_cast<py::ssize_t>(solver_.edge_offsets().size())});
  }

  py::array_t<int32_t> edge_index() const {
    std::vector<int32_t> index(2 * solver_.edge_count());
    for (int32_t from = 0; from < solver_.problem().node_count; ++from) {
      for (int32_t edge = solver_.edge_offsets()[from];
           edge < solver_.edge_offsets()[from + 1]; ++edge) {
        index[edge] = from;
        index[solver_.edge_count() + edge] = solver_.edge_to()[edge];
      }
    }
    return vector_copy<int32_t>(index, {2, solver_.edge_count()});
  }

  py::dict best_solution() const {
    return solution_to_dict(solver_.best_solution(),
                            solver_.problem().objective);
  }

  uint64_t graph_version() const { return solver_.graph_version(); }

  void seed(uint64_t value) { solver_.seed(value); }
  void set_incumbent(
      py::array_t<int32_t, py::array::c_style | py::array::forcecast> route) {
    const py::buffer_info buffer = route.request();
    if (buffer.ndim != 1) {
      throw std::invalid_argument("route must be one-dimensional");
    }
    const int32_t *values = static_cast<const int32_t *>(buffer.ptr);
    solver_.set_incumbent(
        std::vector<int32_t>(values, values + buffer.shape[0]));
  }

  void set_candidate_resource_quotas(
      py::array_t<float, py::array::c_style | py::array::forcecast> quotas) {
    const py::buffer_info buffer = quotas.request();
    if (buffer.ndim != 1)
      throw std::invalid_argument("candidate resource quotas must be one-dimensional");
    const float *values = static_cast<const float *>(buffer.ptr);
    solver_.set_candidate_resource_quotas(
        std::vector<float>(values, values + buffer.shape[0]));
  }

private:
  RoutingDecoder solver_;
};

} // namespace

PYBIND11_MODULE(prism_decoder, module) {
  module.doc() =
      "Variant-general decoder for compositional constraint interaction fields";

  module.def("set_num_threads", [](int32_t count) {
    if (count <= 0) {
      throw std::invalid_argument("thread count must be positive");
    }
    omp_set_dynamic(0);
    omp_set_num_threads(count);
  });
  module.def("get_max_threads", []() { return omp_get_max_threads(); });
  module.def("get_available_threads", []() { return omp_get_num_procs(); });

  py::class_<PyDecoder>(module, "Decoder")
      .def(py::init<py::dict, py::dict, py::dict, int32_t, float>(),
           py::arg("problem"), py::arg("candidate_config") = py::dict(),
           py::arg("search_config") = py::dict(), py::arg("n_rollouts") = 20,
           py::arg("beta") = 2.0f)
      .def("seed", &PyDecoder::seed, py::arg("value"))
      .def("sample", &PyDecoder::sample,
           py::arg("edge_field") = py::none(),
           py::arg("edge_additive") = py::none(),
           py::arg("multipliers") = py::none(),
           py::arg("coupler_weights") = py::none(),
           py::arg("coupler_bias") = py::none(),
           py::arg("edge_risk") = py::none(),
           py::arg("risk_penalty") = 0.0f)
      .def("sample_traced", &PyDecoder::sample_traced,
           py::arg("edge_field") = py::none(),
           py::arg("edge_additive") = py::none(),
           py::arg("multipliers") = py::none(),
           py::arg("coupler_weights") = py::none(),
           py::arg("coupler_bias") = py::none(),
           py::arg("edge_risk") = py::none(),
           py::arg("risk_penalty") = 0.0f)
      .def("sample_greedy", &PyDecoder::sample_greedy,
           py::arg("edge_field") = py::none(),
           py::arg("edge_additive") = py::none(),
           py::arg("multipliers") = py::none(),
           py::arg("coupler_weights") = py::none(),
           py::arg("coupler_bias") = py::none(),
           py::arg("edge_risk") = py::none(),
           py::arg("risk_penalty") = 0.0f)
      .def("solve", &PyDecoder::solve, py::arg("iterations"),
           py::arg("edge_field") = py::none(),
           py::arg("edge_additive") = py::none(),
           py::arg("multipliers") = py::none(),
           py::arg("coupler_weights") = py::none(),
           py::arg("coupler_bias") = py::none(),
           py::arg("edge_risk") = py::none(),
           py::arg("risk_penalty") = 0.0f)
      .def("evaluate", &PyDecoder::evaluate, py::arg("route"))
      .def("evaluate_resources", &PyDecoder::evaluate_resources,
           py::arg("route"))
      .def("set_incumbent", &PyDecoder::set_incumbent, py::arg("route"))
      .def("set_candidate_resource_quotas",
           &PyDecoder::set_candidate_resource_quotas, py::arg("quotas"))
      .def("mask", &PyDecoder::mask, py::arg("prefix"))
      .def_property_readonly("metadata", &PyDecoder::metadata)
      .def_property_readonly("heuristic", &PyDecoder::heuristic)
      .def_property_readonly("proximity", &PyDecoder::proximity)
      .def_property_readonly("edge_features", &PyDecoder::edge_features)
      .def_property_readonly("node_features", &PyDecoder::node_features)
      .def_property_readonly("incumbent_live_state",
                             &PyDecoder::incumbent_live_state)
      .def_property_readonly("resource_features",
                             &PyDecoder::resource_features)
      .def_property_readonly("resource_pressure",
                             &PyDecoder::resource_pressure)
      .def_property_readonly("resource_events", &PyDecoder::resource_events)
      .def_property_readonly("resource_scales", &PyDecoder::resource_scales)
      .def_property_readonly("resource_descriptors",
                             &PyDecoder::resource_descriptors)
      .def_property_readonly("objective_edge_costs",
                             &PyDecoder::objective_edge_costs)
      .def_property_readonly("edge_offsets", &PyDecoder::edge_offsets)
      .def_property_readonly("edge_index", &PyDecoder::edge_index)
      .def_property_readonly("graph_version", &PyDecoder::graph_version)
      .def_property_readonly("best_solution", &PyDecoder::best_solution);
  module.attr("CANDIDATE_FEATURE_NAMES") = prism::candidate_feature_names();
  module.attr("NODE_FEATURE_NAMES") = prism::node_feature_names();
  module.attr("FIELD_CHANNEL_NAMES") = prism::field_channel_names();
  module.attr("NODE_FEATURE_COUNT") = prism::NODE_FEATURE_COUNT;
  module.attr("EDGE_FEATURE_COUNT") = prism::EDGE_FEATURE_COUNT;
  module.attr("FIELD_CHANNEL_COUNT") = prism::FIELD_CHANNEL_COUNT;
  module.attr("LIVE_STATE_FEATURE_COUNT") = prism::LIVE_STATE_FEATURE_COUNT;
  module.attr("MULTIPLIER_COUNT") = prism::MULTIPLIER_COUNT;
  module.attr("RESOURCE_DESCRIPTOR_DIM") = prism::RESOURCE_DESCRIPTOR_DIM;
}
