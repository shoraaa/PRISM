#include "decoder.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using prism::BoundHorizon;
using prism::PrecedenceRelation;
using prism::BACKHAUL_ORDER;
using prism::CandidateConfig;
using prism::CAPACITY;
using prism::Constraint;
using prism::DecisionTrace;
using prism::ObjectiveSpec;
using prism::PICKUP_DELIVERY;
using prism::PRIZE_QUOTA;
using prism::Problem;
using prism::ROUTE_LIMIT;
using prism::ResourceEvaluation;
using prism::ResourceSpec;
using prism::ResourceOperator;
using prism::ResourceDirection;
using prism::ResourceScope;
using prism::ResourceSemiring;

// Spelling of a row's semiring in the published declaration. Kept next to the
// parse above so the two directions cannot drift.
static const char *semiring_name(ResourceSemiring semiring) {
  switch (semiring) {
  case ResourceSemiring::MAX_PLUS:
    return "max_plus";
  case ResourceSemiring::MIN_PLUS:
    return "min_plus";
  case ResourceSemiring::ARITHMETIC:
    break;
  }
  return "arithmetic";
}
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

uint32_t parse_constraints(const py::dict &data) {
  if (!data.contains("constraints"))
    throw std::invalid_argument("normalized schema is missing 'constraints'");
  uint32_t flags = 0;
  for (const std::string &constraint :
       data["constraints"].cast<std::vector<std::string>>()) {
    const prism::ConstraintKernelSpec *kernel =
        prism::constraint_kernel(constraint);
    if (kernel == nullptr)
      throw std::invalid_argument("unknown constraint: " + constraint);
    flags |= kernel->constraint;
  }
  return flags;
}

template <typename T>
T value_or(const py::dict &data, const char *key, T default_value);

ObjectiveSpec parse_objective(const py::dict &data) {
  if (!data.contains("objective"))
    throw std::invalid_argument("normalized schema is missing 'objective'");
  const py::object spec = data["objective"];
  // A dict declares the coefficient algebra directly, so a brand-new objective
  // is expressible from the schema with no C++ change.
  if (py::isinstance<py::dict>(spec)) {
    const py::dict fields = spec.cast<py::dict>();
    ObjectiveSpec objective;
    objective.name = value_or<std::string>(fields, "name", "custom");
    objective.distance_coeff = value_or<float>(fields, "distance_coeff", 1.0f);
    objective.visit_coeff = value_or<float>(fields, "visit_coeff", 0.0f);
    objective.miss_coeff = value_or<float>(fields, "miss_coeff", 0.0f);
    objective.distance_regularizer =
        value_or<float>(fields, "distance_regularizer", 0.0f);
    objective.sense = value_or<float>(fields, "sense", 1.0f);
    return objective;
  }
  // A string names a pre-declared objective resolved from a data table (not a
  // control-flow switch): {distance, visit(prize), miss(penalty), reg, sense}.
  static const std::unordered_map<std::string, ObjectiveSpec> table = {
      {"distance", {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, "distance"}},
      {"prize", {0.0f, 1.0f, 0.0f, 1.0e-3f, -1.0f, "prize"}},
      {"distance_plus_penalty",
       {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, "distance_plus_penalty"}},
  };
  const auto found = table.find(spec.cast<std::string>());
  if (found == table.end())
    throw std::invalid_argument("unknown objective: " +
                                spec.cast<std::string>());
  return found->second;
}

py::dict normalize_problem_schema(const py::dict &source) {
  py::dict result;
  for (const auto &item : source)
    result[item.first] = item.second;

  if (source.contains("name")) {
    std::string name = source["name"].cast<std::string>();
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    result["name"] = std::move(name);
  } else {
    result["name"] = "schema";
  }

  for (const char *key : {"constraints", "objective", "depot_count",
                          "multi_route", "open_route"}) {
    if (!source.contains(key))
      throw std::invalid_argument(std::string("explicit schema is missing '") +
                                  key + "'");
  }

  if (!source.contains("capacity"))
    result["capacity"] = 1.0f;
  if (!source.contains("route_limit"))
    result["route_limit"] = std::numeric_limits<float>::infinity();
  if (!source.contains("prize_quota"))
    result["prize_quota"] = 1.0f;
  if (!source.contains("tour_limit")) {
    const uint32_t constraints = parse_constraints(result);
    if ((constraints & static_cast<uint32_t>(TOUR_LIMIT)) != 0)
      throw std::invalid_argument(
          "explicit tour_limit constraint is missing 'tour_limit'");
    result["tour_limit"] = std::numeric_limits<float>::infinity();
  }
  return result;
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

std::vector<int32_t> algebra_index_attribute(const py::dict &problem,
                                             const std::string &name,
                                             int32_t node_count) {
  const std::vector<float> values =
      algebra_node_attribute(problem, name, node_count);
  std::vector<int32_t> indices(values.size());
  for (size_t i = 0; i < values.size(); ++i)
    indices[i] = static_cast<int32_t>(std::lround(values[i]));
  return indices;
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
    if (op == "affine_accumulator") {
      spec.op = ResourceOperator::AFFINE_ACCUMULATOR;
    } else if (op == "affine_max") {
      // Retained spelling. `affine_max` was an operator family; it is now the
      // affine family in the max-plus semiring, which is the same execution it
      // always had. Declaring `semiring` explicitly is equivalent.
      spec.op = ResourceOperator::AFFINE_ACCUMULATOR;
      spec.semiring = ResourceSemiring::MAX_PLUS;
    } else if (op == "precedence") {
      spec.op = ResourceOperator::PRECEDENCE;
    } else {
      throw std::invalid_argument("unknown resource operator: " + op);
    }
    if (row.contains("semiring")) {
      const std::string semiring = row["semiring"].cast<std::string>();
      if (semiring == "arithmetic")
        spec.semiring = ResourceSemiring::ARITHMETIC;
      else if (semiring == "max_plus")
        spec.semiring = ResourceSemiring::MAX_PLUS;
      else if (semiring == "min_plus")
        spec.semiring = ResourceSemiring::MIN_PLUS;
      else
        throw std::invalid_argument("unknown resource semiring: " + semiring);
    }
    if (spec.op == ResourceOperator::PRECEDENCE) {
      const std::string relation =
          value_or<std::string>(row, "relation", "pairwise");
      if (relation == "pairwise") {
        spec.relation = PrecedenceRelation::PAIRWISE;
        if (!row.contains("predecessor"))
          throw std::invalid_argument(
              "a pairwise precedence row requires 'predecessor'");
        const py::dict reference = row["predecessor"].cast<py::dict>();
        spec.predecessor = algebra_index_attribute(
            data, reference["node_attribute"].cast<std::string>(), node_count);
        // The inverse is derived, never declared twice: an obligation is opened
        // at the predecessor and closed at the node that requires it.
        spec.successor.assign(node_count, -1);
        for (int32_t node = 0; node < node_count; ++node) {
          const int32_t required = spec.predecessor[node];
          if (required < -1 || required >= node_count)
            throw std::invalid_argument("precedence predecessor out of range");
          if (required >= 0) {
            if (required == node)
              throw std::invalid_argument("a node cannot precede itself");
            if (spec.successor[required] >= 0)
              throw std::invalid_argument(
                  "a precedence predecessor may be required by one node only");
            spec.successor[required] = node;
          }
        }
      } else if (relation == "dag") {
        spec.relation = PrecedenceRelation::DAG;
        if (!row.contains("predecessors"))
          throw std::invalid_argument(
              "a dag precedence row requires 'predecessors'");
        // One list of required predecessors per node, in node order. A CSR is
        // built once here because admission reads it on every candidate edge.
        const py::sequence rows = row["predecessors"].cast<py::sequence>();
        if (static_cast<int32_t>(py::len(rows)) != node_count)
          throw std::invalid_argument(
              "dag 'predecessors' needs one entry per node");
        spec.predecessor_offsets.assign(node_count + 1, 0);
        spec.successor_count.assign(node_count, 0);
        spec.predecessor_list.clear();
        for (int32_t node = 0; node < node_count; ++node) {
          const py::sequence required = rows[node].cast<py::sequence>();
          for (const auto &item : required) {
            const int32_t from = item.cast<int32_t>();
            if (from < 0 || from >= node_count)
              throw std::invalid_argument("precedence predecessor out of range");
            if (from == node)
              throw std::invalid_argument("a node cannot precede itself");
            spec.predecessor_list.push_back(from);
            spec.successor_count[from] += 1;
          }
          spec.predecessor_offsets[node + 1] =
              static_cast<int32_t>(spec.predecessor_list.size());
        }
        // The counter starts at every relation outstanding and is discharged
        // one in-edge at a time, so terminal feasibility is the same "nothing
        // unresolved" test the matching relation uses. The scale that turns
        // that counter into a fraction is pinned in the decoder, next to the
        // pin it replaces, so there is one place a relation's unit is decided.
        spec.initial = static_cast<float>(spec.predecessor_list.size());
      } else if (relation == "class_order") {
        spec.relation = PrecedenceRelation::CLASS_ORDER;
        if (!row.contains("class"))
          throw std::invalid_argument(
              "a class_order precedence row requires 'class'");
        const py::dict reference = row["class"].cast<py::dict>();
        spec.node_class = algebra_index_attribute(
            data, reference["node_attribute"].cast<std::string>(), node_count);
        for (int32_t value : spec.node_class) {
          if (value < 0)
            throw std::invalid_argument("precedence class must be non-negative");
        }
      } else {
        throw std::invalid_argument("unknown precedence relation: " + relation);
      }
    }
    spec.state_dim = value_or<int32_t>(row, "state_dim", 1);
    const std::string direction =
        value_or<std::string>(row, "direction", "forward");
    // Execution extends a route forward only. `backward` and `bidirectional`
    // name real REF directions and keep their descriptor slots, but accepting
    // them here would silently run forward semantics under a backward label, so
    // they are rejected until the extension exists.
    if (direction == "backward" || direction == "bidirectional")
      throw std::invalid_argument(
          "resource direction '" + direction +
          "' is declared but not executed: extension is forward-only");
    if (direction != "forward")
      throw std::invalid_argument("unknown resource direction: " + direction);
    spec.direction = ResourceDirection::FORWARD;
    const std::string scope = value_or<std::string>(row, "scope", "route");
    // `tour` is indistinguishable from `solution` in execution -- only `route`
    // resets at the depot -- so accepting it would silently run solution
    // semantics under a different name.
    if (scope == "tour")
      throw std::invalid_argument(
          "resource scope 'tour' is declared but not executed: it is "
          "indistinguishable from 'solution'");
    if (scope == "route")
      spec.scope = ResourceScope::ROUTE;
    else if (scope == "solution")
      spec.scope = ResourceScope::SOLUTION;
    else
      throw std::invalid_argument("unknown resource scope: " + scope);
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
    // `clamp` is the older spelling of the same operand, from when the join was
    // hardcoded to max. Both name one field; declaring both is a contradiction
    // rather than a merge, so it is rejected.
    if (row.contains("join") && row.contains("clamp"))
      throw std::invalid_argument(
          "declare either 'join' or its older spelling 'clamp', not both");
    const char *operand_key = row.contains("join") ? "join" : "clamp";
    if (row.contains(operand_key)) {
      const py::dict operand = row[operand_key].cast<py::dict>();
      if (operand.contains("node_attribute"))
        spec.join_values = algebra_node_attribute(
            data, operand["node_attribute"].cast<std::string>(), node_count);
      else
        spec.join_values.assign(
            static_cast<size_t>(node_count),
            algebra_scalar(data, operand["value"],
                           row.contains("join") ? "join.value"
                                                : "clamp.value"));
    }
    if (row.contains("departure")) {
      const py::dict departure = row["departure"].cast<py::dict>();
      spec.departure_coefficient =
          value_or<float>(departure, "coefficient", 1.0f);
      spec.departure_values = algebra_node_attribute(
          data, departure["node_attribute"].cast<std::string>(), node_count);
    }
    if (row.contains("reset")) {
      const py::dict reset = row["reset"].cast<py::dict>();
      spec.reset_value = reset.contains("value")
                             ? algebra_scalar(data, reset["value"], "reset.value")
                             : spec.initial;
      spec.reset_at_depot = value_or<bool>(reset, "at_depot", false);
      // A reset value conditional on the work the route has left. `guard`
      // names a node attribute and a sign; while some unserved node still
      // matches, the reset installs `value`, and once none does it installs
      // `otherwise`.
      if (reset.contains("guard")) {
        const py::dict guard = reset["guard"].cast<py::dict>();
        if (!guard.contains("node_attribute"))
          throw std::invalid_argument(
              "reset.guard requires 'node_attribute'");
        spec.reset_guard_values = algebra_node_attribute(
            data, guard["node_attribute"].cast<std::string>(), node_count);
        const std::string sign =
            value_or<std::string>(guard, "sign", "positive");
        if (sign == "positive")
          spec.reset_guard_sign = 1.0f;
        else if (sign == "negative")
          spec.reset_guard_sign = -1.0f;
        else
          throw std::invalid_argument("unknown reset.guard sign: " + sign);
        spec.reset_otherwise =
            reset.contains("otherwise")
                ? algebra_scalar(data, reset["otherwise"], "reset.otherwise")
                : spec.initial;
      } else if (reset.contains("otherwise")) {
        throw std::invalid_argument(
            "reset.otherwise has no meaning without reset.guard");
      }
      const bool optional_before_transition =
          value_or<bool>(reset, "optional_before_transition", false);
      spec.optional_reset_duration =
          value_or<float>(reset, "duration", 0.0f);
      if (reset.contains("node_attribute")) {
        const std::vector<float> flags = algebra_node_attribute(
            data, reset["node_attribute"].cast<std::string>(), node_count);
        std::vector<uint8_t> &nodes = optional_before_transition
                                          ? spec.optional_reset_nodes
                                          : spec.reset_nodes;
        nodes.resize(node_count);
        std::transform(flags.begin(), flags.end(), nodes.begin(),
                       [](float value) { return value > 0.5f ? 1 : 0; });
      } else if (optional_before_transition) {
        throw std::invalid_argument(
            "optional pre-transition reset requires node_attribute");
      }
    }
    // Generic term list: the row states its own extension in the language's own
    // coordinates rather than through named fields. `increment`, `departure`
    // and `join` remain as the shorthand the instance generators emit; a row
    // may use one form or the other, never both, because they would silently
    // sum rather than conflict.
    if (row.contains("terms")) {
      // A row has one execution form.  All named forms lower to terms, so
      // mixing them with an explicit term set would silently duplicate effects.
      for (const char *named : {"increment", "departure", "join", "clamp",
                                "reset"}) {
        if (row.contains(named))
          throw std::invalid_argument(
              std::string("declare either 'terms' or the shorthand '") + named +
              "', not both");
      }
      for (const py::handle handle : row["terms"].cast<py::list>()) {
        const py::dict entry = py::reinterpret_borrow<py::dict>(handle);
        prism::ResourceTerm term;
        term.coefficient = value_or<float>(entry, "coefficient", 1.0f);

        const std::string operation =
            value_or<std::string>(entry, "op", "add");
        if (operation == "add")
          term.operation = prism::TermOperation::ADD;
        else if (operation == "join")
          term.operation = prism::TermOperation::JOIN;
        else if (operation == "assign")
          term.operation = prism::TermOperation::ASSIGN;
        else if (operation == "checkpoint")
          term.operation = prism::TermOperation::CHECKPOINT;
        else if (operation == "restore")
          term.operation = prism::TermOperation::RESTORE;
        else
          throw std::invalid_argument("unknown term operation: " + operation);

        const std::string phase =
            value_or<std::string>(entry, "phase", "before_bound");
        if (phase == "before_bound")
          term.phase = prism::TermPhase::BEFORE_BOUND;
        else if (phase == "after_bound")
          term.phase = prism::TermPhase::AFTER_BOUND;
        else
          throw std::invalid_argument("unknown term phase: " + phase);

        const std::string when =
            value_or<std::string>(entry, "when", "always");
        if (when == "always")
          term.trigger = prism::TermTrigger::ALWAYS;
        else if (when == "reset_departure")
          term.trigger = prism::TermTrigger::RESET_DEPARTURE;
        else if (when == "reset_arrival")
          term.trigger = prism::TermTrigger::RESET_ARRIVAL;
        else if (when == "bound_failure")
          term.trigger = prism::TermTrigger::BOUND_FAILURE;
        else if (when == "checkpoint_arrival")
          term.trigger = prism::TermTrigger::CHECKPOINT_ARRIVAL;
        else
          throw std::invalid_argument("unknown term trigger: " + when);

        term.trigger_at_depot = value_or<bool>(entry, "at_depot", false);
        if (entry.contains("at_nodes")) {
          const py::dict selector = entry["at_nodes"].cast<py::dict>();
          if (!selector.contains("node_attribute"))
            throw std::invalid_argument(
                "term.at_nodes requires node_attribute");
          const std::vector<float> flags = algebra_node_attribute(
              data, selector["node_attribute"].cast<std::string>(),
              node_count);
          term.trigger_nodes.resize(node_count);
          std::transform(flags.begin(), flags.end(), term.trigger_nodes.begin(),
                         [](float value) { return value > 0.5f ? 1 : 0; });
        }

        if (entry.contains("gate")) {
          const py::dict gate = entry["gate"].cast<py::dict>();
          if (!gate.contains("node_attribute"))
            throw std::invalid_argument(
                "term.gate requires node_attribute");
          term.gate_values = algebra_node_attribute(
              data, gate["node_attribute"].cast<std::string>(), node_count);
          const std::string sign =
              value_or<std::string>(gate, "sign", "positive");
          if (sign == "positive")
            term.gate_sign = 1.0f;
          else if (sign == "negative")
            term.gate_sign = -1.0f;
          else
            throw std::invalid_argument("unknown term gate sign: " + sign);
          const std::string branch =
              value_or<std::string>(gate, "branch", "default");
          if (branch == "default")
            term.gate = prism::TermGate::REMAINDER_DEFAULT;
          else if (branch == "alternative")
            term.gate = prism::TermGate::REMAINDER_ALTERNATIVE;
          else
            throw std::invalid_argument("unknown term gate branch: " + branch);
        }

        const std::string at = value_or<std::string>(entry, "at", "to");
        if (at == "to")
          term.point = prism::TermPoint::TO;
        else if (at == "from")
          term.point = prism::TermPoint::FROM;
        else
          throw std::invalid_argument("unknown term read point: " + at);

        if (entry.contains("node_attribute")) {
          term.source = prism::TermSource::NODE_ATTRIBUTE;
          term.values = algebra_node_attribute(
              data, entry["node_attribute"].cast<std::string>(), node_count);
        } else if (entry.contains("edge_attribute")) {
          const std::string attribute =
              entry["edge_attribute"].cast<std::string>();
          if (attribute == "distance") {
            term.source = prism::TermSource::DISTANCE;
          } else {
            term.source = prism::TermSource::EDGE_ATTRIBUTE;
            term.values = algebra_edge_attribute(data, attribute, node_count);
          }
        } else if (entry.contains("value")) {
          term.source = prism::TermSource::CONSTANT;
          term.constant = algebra_scalar(data, entry["value"], "term.value");
        } else if (term.operation != prism::TermOperation::RESTORE) {
          throw std::invalid_argument(
              "a term needs one of 'node_attribute', 'edge_attribute' or "
              "'value'");
        }
        spec.terms.push_back(std::move(term));
      }
    }
    if (row.contains("bounds")) {
      const py::list bounds = row["bounds"].cast<py::list>();
      if (bounds.size() != 1)
        throw std::invalid_argument(
            "resource algebra v1 requires exactly one bound row");
      const py::dict bound = py::reinterpret_borrow<py::dict>(bounds[0]);
      // A bound side may be a scalar or, for a window that varies node by
      // node, a named node attribute.
      const auto read_side = [&](const char *side, float &scalar,
                                 std::vector<float> &values) {
        if (!bound.contains(side))
          return;
        const py::handle value = bound[side];
        if (py::isinstance<py::dict>(value)) {
          const py::dict reference = py::reinterpret_borrow<py::dict>(value);
          if (reference.contains("node_attribute")) {
            values = algebra_node_attribute(
                data, reference["node_attribute"].cast<std::string>(),
                node_count);
            return;
          }
        }
        scalar = algebra_scalar(data, value, side);
      };
      read_side("lower", spec.lower, spec.lower_values);
      read_side("upper", spec.upper, spec.upper_values);
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
      const std::string horizon =
          value_or<std::string>(bound, "horizon", "transition");
      spec.horizon =
          horizon == "transition"
              ? BoundHorizon::TRANSITION
              : horizon == "return"
                    ? BoundHorizon::RETURN
                    : horizon == "return_construction"
                          ? BoundHorizon::RETURN_CONSTRUCTION
                          : throw std::invalid_argument(
                                "unknown resource bound horizon: " + horizon);
    }
    static const char *kExtensionKeys[] = {
        "increment", "reset",     "bounds",    "join",      "clamp",
        "departure", "initial",   "state_dim", "semiring",  "terms"};
    static const char *kRelationKeys[] = {"relation", "predecessor",
                                         "predecessors", "class"};
    // A key this operator does not read would be silently ignored, which is how
    // a declaration ends up meaning something other than it says.
    if (spec.op == ResourceOperator::PRECEDENCE) {
      for (const char *key : kExtensionKeys) {
        if (row.contains(key))
          throw std::invalid_argument(
              std::string("a precedence row has no '") + key +
              "': it accumulates nothing");
      }
    } else {
      for (const char *key : kRelationKeys) {
        if (row.contains(key))
          throw std::invalid_argument(
              std::string("only a precedence row declares '") + key + "'");
      }
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
  const py::dict schema = normalize_problem_schema(data);
  Problem problem;
  problem.name = schema["name"].cast<std::string>();
  if (schema.contains("distance") && !schema["distance"].is_none())
    problem.distance = required_matrix(schema, "distance", problem.node_count);
  problem.coordinates = optional_coordinates(schema, problem.node_count);
  if (problem.node_count == 0) {
    throw std::invalid_argument(
        "one of 'distance' or 'coordinates' must be provided");
  }

  problem.depot_count = schema["depot_count"].cast<int32_t>();
  problem.constraints = parse_constraints(schema);
  problem.objective = parse_objective(schema);
  problem.multi_route = schema["multi_route"].cast<bool>();
  problem.open_route = schema["open_route"].cast<bool>();

  problem.capacity = schema["capacity"].cast<float>();
  problem.route_limit = schema["route_limit"].cast<float>();
  problem.tour_limit = schema["tour_limit"].cast<float>();
  problem.prize_quota = schema["prize_quota"].cast<float>();

  problem.demand =
      optional_vector(schema, "demand", problem.node_count, 0.0f);
  problem.prize =
      optional_vector(schema, "prize", problem.node_count, 0.0f);
  problem.penalty =
      optional_vector(schema, "penalty", problem.node_count, 0.0f);
  problem.tw_start =
      optional_vector(schema, "tw_start", problem.node_count, 0.0f);
  problem.tw_end = optional_vector(schema, "tw_end", problem.node_count,
                                   std::numeric_limits<float>::infinity());
  problem.service_time =
      optional_vector(schema, "service_time", problem.node_count, 0.0f);
  set_pickup_delivery_relations(problem, schema);
  problem.resources = parse_resource_algebra(schema, problem.node_count);
  return problem;
}

CandidateConfig parse_candidate_config(const py::dict &data) {
  CandidateConfig config;
  if (data.empty()) {
    return config;
  }
#define CONFIG_INT(field)                                                      \
  config.field = value_or<int32_t>(data, #field, config.field)
  CONFIG_INT(max_candidates);
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
  config.use_srr = value_or<bool>(data, "use_srr", config.use_srr);
  config.verify_screening_resources = value_or<bool>(
      data, "verify_screening_resources",
      config.verify_screening_resources);
  config.verify_incremental_srr = value_or<bool>(
      data, "verify_incremental_srr", config.verify_incremental_srr);
  config.srr_exploration_budget = value_or<int32_t>(
      data, "srr_exploration_budget", config.srr_exploration_budget);
  config.random_escape =
      value_or<bool>(data, "random_escape", config.random_escape);
  config.srr_exploration_margin = value_or<float>(
      data, "srr_exploration_margin", config.srr_exploration_margin);
  return config;
}

py::dict solution_to_dict(const Solution &solution,
                          const ObjectiveSpec &objective) {
  py::dict result;
  result["route"] = vector_copy<int32_t>(
      solution.route, {static_cast<py::ssize_t>(solution.route.size())});
  result["feasible"] = solution.feasible;
  result["objective"] = solution.objective;
  result["objective_name"] = objective.name;
  result["direction"] = objective.direction();
  result["distance"] = solution.distance;
  result["collected_prize"] = solution.collected_prize;
  result["missed_penalty"] = solution.missed_penalty;
  result["raw_objective"] = solution.raw_objective;
  result["changed_edges"] = solution.changed_edges;
  result["srr_moves"] = solution.srr_moves;
  result["srr_scope_nodes"] = solution.srr_scope_nodes;
  result["srr_revisits"] = solution.srr_revisits;
  result["srr_evaluations"] = solution.srr_evaluations;
  result["srr_certified_evaluations"] =
      solution.srr_certified_evaluations;
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
                    py::object objective_residual,
                    int32_t edge_count, int32_t resource_count,
                    int32_t multiplier_count, int32_t live_state_count,
                    py::array_t<float> &field_storage,
                    py::array_t<float> &additive_storage,
                    py::array_t<float> &multiplier_storage,
                    py::array_t<float> &coupler_weight_storage,
                    py::array_t<float> &coupler_bias_storage,
                    py::array_t<float> &residual_storage,
                    const float *&field_values,
                    const float *&additive_values,
                    const float *&multiplier_values,
                    const float *&coupler_weight_values,
                    const float *&coupler_bias_values,
                    const float *&residual_values) {
  field_values = nullptr;
  additive_values = nullptr;
  multiplier_values = nullptr;
  coupler_weight_values = nullptr;
  coupler_bias_values = nullptr;
  residual_values = nullptr;
  if (!edge_field.is_none()) {
    field_storage = edge_field.cast<
        py::array_t<float, py::array::c_style | py::array::forcecast>>();
    const py::buffer_info field_buffer = field_storage.request();
    if (field_buffer.ndim != 2 || field_buffer.shape[0] != edge_count ||
        field_buffer.shape[1] != resource_count) {
      throw std::invalid_argument(
          "edge_field must have shape (edge_count, resource_count)");
    }
    field_values = static_cast<const float *>(field_buffer.ptr);
  }
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
    if (buffer.ndim != 3 || buffer.shape[0] != edge_count ||
        buffer.shape[1] != multiplier_count ||
        buffer.shape[2] != live_state_count) {
      throw std::invalid_argument(
          "coupler_weights must have shape (edge_count, multiplier_count, "
          "live_state_count)");
    }
    coupler_weight_values = static_cast<const float *>(buffer.ptr);
  }
  if (!coupler_bias.is_none()) {
    coupler_bias_storage = coupler_bias.cast<
        py::array_t<float, py::array::c_style | py::array::forcecast>>();
    const py::buffer_info buffer = coupler_bias_storage.request();
    if (buffer.ndim != 2 || buffer.shape[0] != edge_count ||
        buffer.shape[1] != multiplier_count) {
      throw std::invalid_argument(
          "coupler_bias must have shape (edge_count, multiplier_count)");
    }
    coupler_bias_values = static_cast<const float *>(buffer.ptr);
  }
  if (!objective_residual.is_none()) {
    residual_storage = objective_residual.cast<
        py::array_t<float, py::array::c_style | py::array::forcecast>>();
    const py::buffer_info buffer = residual_storage.request();
    if (buffer.ndim != 1 || buffer.shape[0] != edge_count) {
      throw std::invalid_argument(
          "objective_residual must have shape (edge_count,)");
    }
    residual_values = static_cast<const float *>(buffer.ptr);
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
  result["screened_edges"] = vector_copy<int32_t>(
      trace.screened_edges,
      {static_cast<py::ssize_t>(trace.screened_edges.size())});
  const py::ssize_t screened_rows =
      static_cast<py::ssize_t>(trace.screened_edges.size());
  result["screened_resource_delta"] = vector_copy<float>(
      trace.screened_resource_delta,
      {screened_rows,
       screened_rows > 0
           ? static_cast<py::ssize_t>(trace.screened_resource_delta.size()) /
                 screened_rows
           : 0});
  result["screening_fast_evaluations"] = trace.screening_fast_evaluations;
  result["screening_fallback_evaluations"] =
      trace.screening_fallback_evaluations;
  result["screening_verification_failures"] =
      trace.screening_verification_failures;
  result["screening_verification_failures_by_channel"] =
      vector_copy<int64_t>(
          trace.screening_verification_failures_by_channel,
          {static_cast<py::ssize_t>(
              trace.screening_verification_failures_by_channel.size())});
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
                  py::object objective_residual) {
    py::array_t<float> field_storage;
    py::array_t<float> additive_storage;
    py::array_t<float> multiplier_storage;
    py::array_t<float> coupler_weight_storage;
    py::array_t<float> coupler_bias_storage;
    py::array_t<float> residual_storage;
    const float *field_values;
    const float *additive_values;
    const float *multiplier_values;
    const float *coupler_weight_values;
    const float *coupler_bias_values;
    const float *residual_values;
    parse_guidance(edge_field, edge_additive, multipliers, coupler_weights,
                   coupler_bias, objective_residual, solver_.edge_count(),
                   solver_.resource_count(), solver_.multiplier_count(),
                   solver_.live_state_feature_count(),
                   field_storage,
                   additive_storage, multiplier_storage,
                   coupler_weight_storage, coupler_bias_storage, residual_storage,
                   field_values,
                   additive_values, multiplier_values, coupler_weight_values,
                   coupler_bias_values, residual_values);
    std::vector<Solution> solutions;
    {
      py::gil_scoped_release release;
      solutions = solver_.sample(field_values, additive_values,
                                 multiplier_values,
                                 coupler_weight_values, coupler_bias_values,
                                 residual_values);
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
                         py::object coupler_bias,
                         py::object objective_residual) {
    py::array_t<float> field_storage;
    py::array_t<float> additive_storage;
    py::array_t<float> multiplier_storage;
    py::array_t<float> coupler_weight_storage;
    py::array_t<float> coupler_bias_storage;
    py::array_t<float> residual_storage;
    const float *field_values;
    const float *additive_values;
    const float *multiplier_values;
    const float *coupler_weight_values;
    const float *coupler_bias_values;
    const float *residual_values;
    parse_guidance(edge_field, edge_additive, multipliers, coupler_weights,
                   coupler_bias, objective_residual, solver_.edge_count(),
                   solver_.resource_count(), solver_.multiplier_count(),
                   solver_.live_state_feature_count(),
                   field_storage,
                   additive_storage, multiplier_storage,
                   coupler_weight_storage, coupler_bias_storage, residual_storage,
                   field_values,
                   additive_values, multiplier_values, coupler_weight_values,
                   coupler_bias_values, residual_values);
    std::vector<Solution> solutions;
    DecisionTrace trace;
    {
      py::gil_scoped_release release;
      solutions = solver_.sample(field_values, additive_values,
                                 multiplier_values,
                                 coupler_weight_values, coupler_bias_values,
                                 residual_values,
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
                         py::object coupler_bias,
                         py::object objective_residual) {
    py::array_t<float> field_storage;
    py::array_t<float> additive_storage;
    py::array_t<float> multiplier_storage;
    py::array_t<float> coupler_weight_storage;
    py::array_t<float> coupler_bias_storage;
    py::array_t<float> residual_storage;
    const float *field_values;
    const float *additive_values;
    const float *multiplier_values;
    const float *coupler_weight_values;
    const float *coupler_bias_values;
    const float *residual_values;
    parse_guidance(edge_field, edge_additive, multipliers, coupler_weights,
                   coupler_bias, objective_residual, solver_.edge_count(),
                   solver_.resource_count(), solver_.multiplier_count(),
                   solver_.live_state_feature_count(),
                   field_storage,
                   additive_storage, multiplier_storage,
                   coupler_weight_storage, coupler_bias_storage, residual_storage,
                   field_values,
                   additive_values, multiplier_values, coupler_weight_values,
                   coupler_bias_values, residual_values);
    Solution solution;
    {
      py::gil_scoped_release release;
      solution = solver_.sample_greedy(
          field_values, additive_values, multiplier_values,
          coupler_weight_values, coupler_bias_values, residual_values);
    }
    return solution_to_dict(solution, solver_.problem().objective);
  }

  py::dict solve(int32_t iterations, py::object edge_field,
                 py::object edge_additive, py::object multipliers,
                 py::object coupler_weights,
                 py::object coupler_bias, py::object objective_residual) {
    py::array_t<float> field_storage;
    py::array_t<float> additive_storage;
    py::array_t<float> multiplier_storage;
    py::array_t<float> coupler_weight_storage;
    py::array_t<float> coupler_bias_storage;
    py::array_t<float> residual_storage;
    const float *field_values;
    const float *additive_values;
    const float *multiplier_values;
    const float *coupler_weight_values;
    const float *coupler_bias_values;
    const float *residual_values;
    parse_guidance(edge_field, edge_additive, multipliers, coupler_weights,
                   coupler_bias, objective_residual, solver_.edge_count(),
                   solver_.resource_count(), solver_.multiplier_count(),
                   solver_.live_state_feature_count(),
                   field_storage,
                   additive_storage, multiplier_storage,
                   coupler_weight_storage, coupler_bias_storage, residual_storage,
                   field_values,
                   additive_values, multiplier_values, coupler_weight_values,
                   coupler_bias_values, residual_values);
    Solution solution;
    {
      py::gil_scoped_release release;
      solution = solver_.solve(
          iterations, field_values, additive_values, multiplier_values,
          coupler_weight_values, coupler_bias_values, residual_values);
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
    py::list kernel_rows;
    for (const prism::ConstraintKernelSpec *kernel :
         solver_.active_constraint_kernels()) {
      py::dict row;
      row["name"] = kernel->schema_name;
      row["field_channel"] = kernel->field_channel;
      row["resource"] =
          kernel->field_channel < 0
              ? py::none()
              : py::cast(
                    prism::resource_kernel(kernel->resource_operator).name);
      py::list capabilities;
      if ((kernel->capabilities & prism::KERNEL_ROUTE_STATE) != 0)
        capabilities.append("route_state");
      if ((kernel->capabilities & prism::KERNEL_SOLUTION_STATE) != 0)
        capabilities.append("solution_state");
      if ((kernel->capabilities & prism::KERNEL_ORDER_SENSITIVE) != 0)
        capabilities.append("order_sensitive");
      if ((kernel->capabilities & prism::KERNEL_REVERSAL_SENSITIVE) != 0)
        capabilities.append("reversal_sensitive");
      if ((kernel->capabilities & prism::KERNEL_RELATIONAL) != 0)
        capabilities.append("relational");
      row["capabilities"] = std::move(capabilities);
      kernel_rows.append(std::move(row));
    }
    result["constraint_kernels"] = std::move(kernel_rows);
    result["objective"] = problem.objective.name;
    result["objective_scale"] = solver_.objective_scale();
    result["objective_energy_scale"] = solver_.objective_energy_scale();
    result["direction"] = problem.objective.direction();
    // Expose the declared objective algebra so the network conditions on the
    // coefficient vector (shared head) instead of a categorical one-hot.
    py::dict objective_coeffs;
    objective_coeffs["distance_coeff"] = problem.objective.distance_coeff;
    objective_coeffs["visit_coeff"] = problem.objective.visit_coeff;
    objective_coeffs["miss_coeff"] = problem.objective.miss_coeff;
    objective_coeffs["distance_regularizer"] =
        problem.objective.distance_regularizer;
    objective_coeffs["sense"] = problem.objective.sense;
    result["objective_coeffs"] = std::move(objective_coeffs);
    result["multi_route"] = problem.multi_route;
    result["open_route"] = problem.open_route;
    result["metric_symmetric"] = solver_.metric_symmetric();
    result["metric_skew"] = solver_.metric_skew();
    result["edge_count"] = solver_.edge_count();
    result["graph_version"] = solver_.graph_version();
    result["guidance_mode"] = "energy";
    result["max_candidates"] = solver_.candidate_config().max_candidates;
    int32_t maximum_degree = 0;
    const auto &offsets = solver_.edge_offsets();
    for (size_t node = 1; node < offsets.size(); ++node) {
      maximum_degree =
          std::max(maximum_degree, offsets[node] - offsets[node - 1]);
    }
    result["maximum_degree"] = maximum_degree;
    result["candidate_strategy"] = "distance";
    result["candidate_feature_names"] = prism::candidate_feature_names();
    result["field_channel_names"] = prism::field_channel_names();
    py::list resource_rows;
    for (const ResourceSpec &resource : solver_.resources()) {
      py::dict row;
      row["name"] = resource.name;
      row["active"] = resource.active;
      row["state_dim"] = resource.state_dim;
      row["operator"] = prism::resource_kernel(resource.op).name;
      row["semiring"] = semiring_name(resource.semiring);
      resource_rows.append(std::move(row));
    }
    result["resources"] = std::move(resource_rows);
    result["resource_count"] = solver_.resource_count();
    result["multiplier_count"] = solver_.multiplier_count();
    result["resource_program_version"] = "resource_terms_v1";
    result["node_feature_names"] = prism::node_feature_names();
    std::vector<uint8_t> active_channels;
    active_channels.reserve(solver_.resource_count());
    std::vector<std::string> resource_names;
    resource_names.reserve(solver_.resource_count());
    for (const ResourceSpec &resource : solver_.resources()) {
      active_channels.push_back(static_cast<uint8_t>(resource.active));
      resource_names.push_back(resource.name);
    }
    // Registry order for THIS problem. A row's position depends on which
    // constraints the instance declares, so it cannot be read off a global
    // channel list.
    result["resource_names"] = resource_names;
    result["field_channel_mask"] = vector_copy<uint8_t>(
        active_channels, {solver_.resource_count()});
    const SearchConfig &search = solver_.search_config();
    py::dict search_values;
    search_values["min_changed_edges"] = search.min_changed_edges;
    search_values["max_perturb_attempts"] = search.max_perturb_attempts;
    search_values["or_opt_max_segment"] = search.or_opt_max_segment;
    search_values["feasibility_lookahead_depth"] =
        search.feasibility_lookahead_depth;
    search_values["use_srr"] = search.use_srr;
    search_values["verify_screening_resources"] =
        search.verify_screening_resources;
    search_values["verify_incremental_srr"] =
        search.verify_incremental_srr;
    search_values["srr_exploration_budget"] = search.srr_exploration_budget;
    search_values["random_escape"] = search.random_escape;
    search_values["srr_exploration_margin"] = search.srr_exploration_margin;
    result["search"] = search_values;
    return result;
  }

  py::array_t<float> edge_features() const {
    return vector_copy<float>(
        solver_.edge_features(),
        {solver_.edge_count(), prism::EDGE_FEATURE_COUNT});
  }

  py::array_t<float> node_resource_features() const {
    return vector_copy<float>(
        solver_.node_resource_features(),
        {solver_.problem().node_count, solver_.resource_count(),
         prism::NODE_RESOURCE_FEATURE_COUNT});
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

  py::array_t<float> node_objective_features() const {
    return vector_copy<float>(solver_.node_objective_features(),
                              {solver_.problem().node_count,
                               prism::OBJECTIVE_NODE_TERM_COUNT});
  }

  py::array_t<float> incumbent_suffix_state() const {
    return vector_copy<float>(
        solver_.incumbent_suffix_state(),
        {solver_.problem().node_count, solver_.live_state_feature_count()});
  }

  py::array_t<float> incumbent_suffix_features() const {
    return vector_copy<float>(
        solver_.incumbent_suffix_features(),
        {solver_.problem().node_count, solver_.resource_count(),
         prism::RESOURCE_SUFFIX_FEATURE_COUNT});
  }

  py::array_t<float> incumbent_transition_features() const {
    return vector_copy<float>(
        solver_.incumbent_transition_features(),
        {solver_.edge_count(), solver_.resource_count(),
         prism::RESOURCE_TRANSITION_FEATURE_COUNT});
  }

  py::array_t<uint8_t> incumbent_transition_feature_mask() const {
    return vector_copy<uint8_t>(
        solver_.incumbent_transition_feature_mask(),
        {solver_.edge_count(), solver_.resource_count()});
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

  py::array_t<float> compiled_resource_pressure() const {
    return vector_copy<float>(
        solver_.compiled_resource_pressures(),
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

  // The declarative resource row each registry entry implements. A compiled
  // kernel is an execution fast path for the row it publishes here, so the row
  // -- not the kernel enum -- defines the semantics, drives the properties, and
  // is what tests replay to prove the two agree. `declared` is false for the
  // kernels whose semantics the language cannot yet express.
  py::list resource_declarations() const {
    py::list rows;
    for (int32_t index = 0; index < solver_.resource_count(); ++index) {
      bool declared = false;
      const prism::ResourceSpec spec = solver_.declared_algebra(index, &declared);
      py::dict row;
      row["name"] = spec.name;
      row["active"] = spec.active;
      row["declared"] = declared;
      // Which compiled kernel executes this row, or "" when the declarative
      // interpreter does. This used to read the row's operator, which named the
      // kernel only while a row's operator doubled as a constraint identity.
      row["kernel"] = prism::fast_path_name(
          solver_.resources()[index].fast_path);
      row["operator"] = prism::resource_kernel(spec.op).name;
      row["semiring"] = semiring_name(spec.semiring);
      if (!declared) {
        rows.append(std::move(row));
        continue;
      }
      if (spec.op == prism::ResourceOperator::PRECEDENCE) {
        row["relation"] = spec.relation == PrecedenceRelation::PAIRWISE
                              ? "pairwise"
                              : "class_order";
        row["predecessor"] = vector_copy<int32_t>(
            spec.predecessor,
            {static_cast<py::ssize_t>(spec.predecessor.size())});
        row["class"] = vector_copy<int32_t>(
            spec.node_class,
            {static_cast<py::ssize_t>(spec.node_class.size())});
        row["scope"] = spec.scope == prism::ResourceScope::ROUTE
                           ? "route"
                           : spec.scope == prism::ResourceScope::TOUR
                                 ? "tour"
                                 : "solution";
        rows.append(std::move(row));
        continue;
      }
      row["state_dim"] = spec.state_dim;
      row["direction"] = spec.direction == prism::ResourceDirection::FORWARD
                             ? "forward"
                             : spec.direction ==
                                       prism::ResourceDirection::BACKWARD
                                   ? "backward"
                                   : "bidirectional";
      row["scope"] = spec.scope == prism::ResourceScope::ROUTE
                         ? "route"
                         : spec.scope == prism::ResourceScope::TOUR ? "tour"
                                                                    : "solution";
      row["initial"] = spec.initial;
      row["scale"] = spec.scale;
      // Canonical execution form. Reset, join, opening and optional checkpoint
      // restore are terms here; no named execution block survives publication.
      py::list terms;
      for (const prism::ResourceTerm &term : spec.terms) {
        py::dict entry;
        switch (term.operation) {
        case prism::TermOperation::ADD:
          entry["op"] = "add";
          break;
        case prism::TermOperation::JOIN:
          entry["op"] = "join";
          break;
        case prism::TermOperation::ASSIGN:
          entry["op"] = "assign";
          break;
        case prism::TermOperation::CHECKPOINT:
          entry["op"] = "checkpoint";
          break;
        case prism::TermOperation::RESTORE:
          entry["op"] = "restore";
          break;
        }
        entry["phase"] = term.phase == prism::TermPhase::BEFORE_BOUND
                             ? "before_bound"
                             : "after_bound";
        switch (term.trigger) {
        case prism::TermTrigger::ALWAYS:
          entry["when"] = "always";
          break;
        case prism::TermTrigger::RESET_DEPARTURE:
          entry["when"] = "reset_departure";
          break;
        case prism::TermTrigger::RESET_ARRIVAL:
          entry["when"] = "reset_arrival";
          break;
        case prism::TermTrigger::BOUND_FAILURE:
          entry["when"] = "bound_failure";
          break;
        case prism::TermTrigger::CHECKPOINT_ARRIVAL:
          entry["when"] = "checkpoint_arrival";
          break;
        }
        entry["at"] = term.point == prism::TermPoint::FROM ? "from" : "to";
        entry["coefficient"] = term.coefficient;
        entry["at_depot"] = term.trigger_at_depot;
        entry["trigger_nodes"] = vector_copy<uint8_t>(
            term.trigger_nodes,
            {static_cast<py::ssize_t>(term.trigger_nodes.size())});
        entry["gate"] = term.gate == prism::TermGate::REMAINDER_DEFAULT
                            ? "remainder_default"
                        : term.gate ==
                                  prism::TermGate::REMAINDER_ALTERNATIVE
                            ? "remainder_alternative"
                            : "always";
        entry["gate_sign"] = term.gate_sign;
        entry["gate_values"] = vector_copy<float>(
            term.gate_values,
            {static_cast<py::ssize_t>(term.gate_values.size())});
        switch (term.source) {
        case prism::TermSource::DISTANCE:
          entry["source"] = "distance";
          break;
        case prism::TermSource::EDGE_ATTRIBUTE:
          entry["source"] = "edge_attribute";
          break;
        case prism::TermSource::NODE_ATTRIBUTE:
          entry["source"] = "node_attribute";
          break;
        case prism::TermSource::CONSTANT:
          entry["source"] = "value";
          entry["value"] = term.constant;
          break;
        }
        entry["values"] = vector_copy<float>(
            term.values, {static_cast<py::ssize_t>(term.values.size())});
        terms.append(std::move(entry));
      }
      row["terms"] = std::move(terms);
      py::dict bound;
      bound["lower"] = spec.lower;
      bound["upper"] = spec.upper;
      bound["lower_values"] = vector_copy<float>(
          spec.lower_values,
          {static_cast<py::ssize_t>(spec.lower_values.size())});
      bound["upper_values"] = vector_copy<float>(
          spec.upper_values,
          {static_cast<py::ssize_t>(spec.upper_values.size())});
      bound["check"] = spec.bound_check == prism::BoundCheck::TRANSITION
                           ? "transition"
                           : spec.bound_check == prism::BoundCheck::ROUTE_END
                                 ? "route_end"
                                 : "solution_end";
      bound["horizon"] =
          spec.horizon == BoundHorizon::TRANSITION
              ? "transition"
              : spec.horizon == BoundHorizon::RETURN ? "return"
                                                     : "return_construction";
      row["bounds"] = py::make_tuple(std::move(bound));
      rows.append(std::move(row));
    }
    return rows;
  }

  py::array_t<float> resource_row_properties() const {
    return vector_copy<float>(
        solver_.resource_row_properties(),
        {solver_.resource_count(), prism::RESOURCE_ROW_PROPERTY_DIM});
  }

  py::array_t<float> resource_term_properties() const {
    const py::ssize_t terms = static_cast<py::ssize_t>(
        solver_.resource_term_properties().size() /
        prism::RESOURCE_TERM_PROPERTY_DIM);
    return vector_copy<float>(solver_.resource_term_properties(),
                              {terms, prism::RESOURCE_TERM_PROPERTY_DIM});
  }

  py::array_t<int32_t> resource_term_counts() const {
    return vector_copy<int32_t>(solver_.resource_term_counts(),
                                {solver_.resource_count()});
  }

  py::array_t<float> objective_edge_costs() const {
    return vector_copy<float>(solver_.objective_edge_costs(),
                              {solver_.edge_count()});
  }

  float objective_energy_scale() const {
    return solver_.objective_energy_scale();
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
  module.def("normalize_problem_schema", [](const py::dict &problem) {
    return normalize_problem_schema(problem);
  });

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
           py::arg("objective_residual") = py::none())
      .def("sample_traced", &PyDecoder::sample_traced,
           py::arg("edge_field") = py::none(),
           py::arg("edge_additive") = py::none(),
           py::arg("multipliers") = py::none(),
           py::arg("coupler_weights") = py::none(),
           py::arg("coupler_bias") = py::none(),
           py::arg("objective_residual") = py::none())
      .def("sample_greedy", &PyDecoder::sample_greedy,
           py::arg("edge_field") = py::none(),
           py::arg("edge_additive") = py::none(),
           py::arg("multipliers") = py::none(),
           py::arg("coupler_weights") = py::none(),
           py::arg("coupler_bias") = py::none(),
           py::arg("objective_residual") = py::none())
      .def("solve", &PyDecoder::solve, py::arg("iterations"),
           py::arg("edge_field") = py::none(),
           py::arg("edge_additive") = py::none(),
           py::arg("multipliers") = py::none(),
           py::arg("coupler_weights") = py::none(),
           py::arg("coupler_bias") = py::none(),
           py::arg("objective_residual") = py::none())
      .def("evaluate", &PyDecoder::evaluate, py::arg("route"))
      .def("evaluate_resources", &PyDecoder::evaluate_resources,
           py::arg("route"))
      .def("set_incumbent", &PyDecoder::set_incumbent, py::arg("route"))
      .def("mask", &PyDecoder::mask, py::arg("prefix"))
      .def_property_readonly("metadata", &PyDecoder::metadata)
      .def_property_readonly("edge_features", &PyDecoder::edge_features)
      .def_property_readonly("node_features", &PyDecoder::node_features)
      .def_property_readonly("node_resource_features",
                             &PyDecoder::node_resource_features)
      .def_property_readonly("incumbent_live_state",
                             &PyDecoder::incumbent_live_state)
      .def_property_readonly("incumbent_suffix_state",
                             &PyDecoder::incumbent_suffix_state)
      .def_property_readonly("incumbent_suffix_features",
                             &PyDecoder::incumbent_suffix_features)
      .def_property_readonly("incumbent_transition_features",
                             &PyDecoder::incumbent_transition_features)
      .def_property_readonly("incumbent_transition_feature_mask",
                             &PyDecoder::incumbent_transition_feature_mask)
      .def_property_readonly("node_objective_features",
                             &PyDecoder::node_objective_features)
      .def_property_readonly("resource_features",
                             &PyDecoder::resource_features)
      .def_property_readonly("resource_pressure",
                             &PyDecoder::resource_pressure)
      // Diagnostic reference: each registry row priced by its own kernel rather
      // than by the declaration it publishes. Equal to resource_pressure
      // wherever a published row is faithful.
      .def_property_readonly("compiled_resource_pressure",
                             &PyDecoder::compiled_resource_pressure)
      .def_property_readonly("resource_events", &PyDecoder::resource_events)
      .def_property_readonly("resource_scales", &PyDecoder::resource_scales)
      .def_property_readonly("resource_row_properties",
                             &PyDecoder::resource_row_properties)
      .def_property_readonly("resource_term_properties",
                             &PyDecoder::resource_term_properties)
      .def_property_readonly("resource_term_counts",
                             &PyDecoder::resource_term_counts)
      .def_property_readonly("resource_declarations",
                             &PyDecoder::resource_declarations)
      .def_property_readonly("objective_edge_costs",
                             &PyDecoder::objective_edge_costs)
      .def_property_readonly("objective_energy_scale",
                             &PyDecoder::objective_energy_scale)
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
  module.attr("OBJECTIVE_NODE_TERM_COUNT") = prism::OBJECTIVE_NODE_TERM_COUNT;


  module.attr("RESOURCE_ROW_PROPERTY_DIM") =
      prism::RESOURCE_ROW_PROPERTY_DIM;
  module.attr("RESOURCE_TERM_PROPERTY_DIM") =
      prism::RESOURCE_TERM_PROPERTY_DIM;
  module.attr("NODE_RESOURCE_FEATURE_COUNT") =
      prism::NODE_RESOURCE_FEATURE_COUNT;
  module.attr("RESOURCE_SUFFIX_FEATURE_COUNT") =
      prism::RESOURCE_SUFFIX_FEATURE_COUNT;
  module.attr("RESOURCE_TRANSITION_FEATURE_COUNT") =
      prism::RESOURCE_TRANSITION_FEATURE_COUNT;
}
