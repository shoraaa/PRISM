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

// "Has everything `next` must wait for been served?", for either precedence
// relation that has predecessors. `seen(node)` is supplied by the caller so
// each site can answer from whatever record it keeps -- a visited array during
// construction, an epoch stamp inside SRR -- exactly as precedence_admits
// already expects. PAIRWISE reads its single matched predecessor; DAG reads
// its whole predecessor list, which is the only structural difference between
// them at admission time.
template <typename Seen>
bool precedence_prerequisites_met(const ResourceSpec &spec, int32_t next,
                                  Seen &&seen) {
  if (spec.relation == PrecedenceRelation::DAG) {
    if (spec.predecessor_offsets.empty())
      return true;
    const size_t head = static_cast<size_t>(next);
    for (int32_t at = spec.predecessor_offsets[head];
         at < spec.predecessor_offsets[head + 1]; ++at) {
      if (!seen(spec.predecessor_list[static_cast<size_t>(at)]))
        return false;
    }
    return true;
  }
  const int32_t required =
      spec.predecessor.empty() ? -1 : spec.predecessor[static_cast<size_t>(next)];
  return required < 0 || seen(required);
}
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
  double positive_load = 0.0;
  double negative_load = 0.0;
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
  result.positive_load = std::max(static_cast<double>(problem.demand[node]),
                                  0.0);
  result.negative_load =
      std::max(-static_cast<double>(problem.demand[node]), 0.0);
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
  result.positive_load = lhs.positive_load + rhs.positive_load;
  result.negative_load = lhs.negative_load + rhs.negative_load;

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

const std::array<ConstraintKernelSpec, CONSTRAINT_KERNEL_COUNT> &
constraint_kernel_registry() {
  static const std::array<ConstraintKernelSpec,
                          CONSTRAINT_KERNEL_COUNT>
      kernels = {{
          {VISIT_ALL, "visit_all", -1, ResourceOperator::AFFINE_ACCUMULATOR,
           KERNEL_SOLUTION_STATE},
          {CAPACITY, "capacity", static_cast<int32_t>(FieldChannel::CAPACITY),
           ResourceOperator::CAPACITY, KERNEL_ROUTE_STATE},
          {BACKHAUL_ORDER, "backhaul_order",
           static_cast<int32_t>(FieldChannel::BACKHAUL_ORDER),
           ResourceOperator::BACKHAUL_ORDER,
           KERNEL_ROUTE_STATE | KERNEL_ORDER_SENSITIVE |
               KERNEL_REVERSAL_SENSITIVE},
          {PICKUP_DELIVERY, "pickup_delivery",
           static_cast<int32_t>(FieldChannel::PICKUP_DELIVERY),
           ResourceOperator::PICKUP_DELIVERY,
           KERNEL_ROUTE_STATE | KERNEL_ORDER_SENSITIVE |
               KERNEL_REVERSAL_SENSITIVE | KERNEL_RELATIONAL},
          {ROUTE_LIMIT, "route_limit",
           static_cast<int32_t>(FieldChannel::ROUTE_LIMIT),
           ResourceOperator::ROUTE_LIMIT, KERNEL_ROUTE_STATE},
          {TIME_WINDOWS, "time_windows",
           static_cast<int32_t>(FieldChannel::TIME_WINDOW),
           ResourceOperator::TIME_WINDOW,
           KERNEL_ROUTE_STATE | KERNEL_ORDER_SENSITIVE |
               KERNEL_REVERSAL_SENSITIVE},
          {TOUR_LIMIT, "tour_limit",
           static_cast<int32_t>(FieldChannel::TOUR_LIMIT),
           ResourceOperator::TOUR_LIMIT, KERNEL_SOLUTION_STATE},
          {PRIZE_QUOTA, "prize_quota",
           static_cast<int32_t>(FieldChannel::PRIZE_QUOTA),
           ResourceOperator::PRIZE_QUOTA, KERNEL_SOLUTION_STATE},
      }};
  return kernels;
}

const std::array<ResourceKernelSpec, RESOURCE_KERNEL_COUNT> &
resource_kernel_registry() {
  static const std::array<ResourceKernelSpec, RESOURCE_KERNEL_COUNT> kernels = {{
      {ResourceOperator::CAPACITY, "capacity",
       static_cast<int32_t>(FieldChannel::CAPACITY)},
      {ResourceOperator::TIME_WINDOW, "time_window",
       static_cast<int32_t>(FieldChannel::TIME_WINDOW)},
      {ResourceOperator::ROUTE_LIMIT, "route_limit",
       static_cast<int32_t>(FieldChannel::ROUTE_LIMIT)},
      {ResourceOperator::TOUR_LIMIT, "tour_limit",
       static_cast<int32_t>(FieldChannel::TOUR_LIMIT)},
      {ResourceOperator::BACKHAUL_ORDER, "backhaul_order",
       static_cast<int32_t>(FieldChannel::BACKHAUL_ORDER)},
      {ResourceOperator::PICKUP_DELIVERY, "pickup_delivery",
       static_cast<int32_t>(FieldChannel::PICKUP_DELIVERY)},
      {ResourceOperator::PRIZE_QUOTA, "prize_quota",
       static_cast<int32_t>(FieldChannel::PRIZE_QUOTA)},
      {ResourceOperator::AFFINE_ACCUMULATOR, "affine_accumulator", -1},
      {ResourceOperator::PRECEDENCE, "precedence", -1},
  }};
  return kernels;
}

const ResourceKernelSpec &resource_kernel(ResourceOperator op) {
  const size_t index = static_cast<size_t>(op);
  if (index >= resource_kernel_registry().size() ||
      resource_kernel_registry()[index].op != op)
    throw std::logic_error("unregistered resource operator");
  return resource_kernel_registry()[index];
}

const ConstraintKernelSpec *constraint_kernel(Constraint constraint) {
  for (const ConstraintKernelSpec &kernel : constraint_kernel_registry()) {
    if (kernel.constraint == constraint)
      return &kernel;
  }
  return nullptr;
}

const ConstraintKernelSpec *field_channel_kernel(int32_t channel) {
  for (const ConstraintKernelSpec &kernel : constraint_kernel_registry()) {
    if (kernel.field_channel == channel)
      return &kernel;
  }
  return nullptr;
}

const ConstraintKernelSpec *constraint_kernel(const std::string &schema_name) {
  for (const ConstraintKernelSpec &kernel : constraint_kernel_registry()) {
    if (schema_name == kernel.schema_name)
      return &kernel;
  }
  return nullptr;
}

bool Problem::has(Constraint constraint) const {
  return (constraints & static_cast<uint32_t>(constraint)) != 0;
}

bool Problem::has_capability(ConstraintCapability capability) const {
  for (const ConstraintKernelSpec &kernel : constraint_kernel_registry()) {
    if (has(kernel.constraint) &&
        (kernel.capabilities & static_cast<uint32_t>(capability)) != 0)
      return true;
  }
  return false;
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
  // Generic operators move a related pair together and refuse to cut between
  // them, which assumes each node has at most one partner. Two rows claiming the
  // same node would silently leave one of them unenforced by those operators.
  std::vector<uint8_t> claimed(n, 0);
  if (has(PICKUP_DELIVERY)) {
    for (int32_t node = 0; node < n; ++node)
      claimed[node] = delivery_of_pickup[node] >= 0 ||
                              pickup_of_delivery[node] >= 0
                          ? 1
                          : 0;
  }
  for (const ResourceSpec &row : resources) {
    if (row.op != ResourceOperator::PRECEDENCE ||
        row.relation != PrecedenceRelation::PAIRWISE)
      continue;
    for (int32_t node = 0; node < n; ++node) {
      const bool involved =
          (!row.predecessor.empty() && row.predecessor[node] >= 0) ||
          (!row.successor.empty() && row.successor[node] >= 0);
      if (!involved)
        continue;
      if (claimed[node])
        throw std::invalid_argument(
            "a node may take part in one pairwise precedence relation only");
      claimed[node] = 1;
    }
  }
  for (const ResourceSpec &resource : resources) {
    if (resource.name.empty())
      throw std::invalid_argument("resource name must not be empty");
    if (resource.state_dim != 1) {
      throw std::invalid_argument(
          "resource algebra v1 currently requires state_dim == 1");
    }
    if (!RoutingDecoder::tropical(resource) &&
        !resource.join_values.empty()) {
      throw std::invalid_argument(
          "only a tropical semiring may declare a join operand; an arithmetic "
          "row has no join to apply it to");
    }
    if (RoutingDecoder::tropical(resource) &&
        resource.op != ResourceOperator::AFFINE_ACCUMULATOR) {
      throw std::invalid_argument(
          "only an affine row runs in a semiring; a precedence relation has no "
          "accumulation to join against");
    }
    if (RoutingDecoder::tropical(resource) &&
        !resource.optional_reset_nodes.empty()) {
      throw std::invalid_argument(
          "a tropical row cannot declare optional resets: a break's duration is "
          "charged to another row, which the language cannot express yet");
    }
    for (const std::vector<float> *values :
         {&resource.join_values, &resource.lower_values,
          &resource.upper_values, &resource.departure_values}) {
      if (!values->empty() && values->size() != n)
        throw std::invalid_argument(
            "resource join, bound, and departure arrays must have shape "
            "(node_count,)");
      for (float value : *values) {
        if (std::isnan(value))
          throw std::invalid_argument("resource algebra array must not be NaN");
      }
    }
    if (!std::isfinite(resource.initial) || !std::isfinite(resource.scale) ||
        resource.scale <= 0.0f || std::isnan(resource.lower) ||
        std::isnan(resource.upper) || resource.lower > resource.upper ||
        !std::isfinite(resource.edge_coefficient) ||
        !std::isfinite(resource.node_coefficient) ||
        !std::isfinite(resource.reset_value) ||
        !finite_nonnegative(resource.optional_reset_duration)) {
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
    if (!resource.optional_reset_nodes.empty() &&
        resource.optional_reset_nodes.size() != n)
      throw std::invalid_argument(
          "optional resource reset flags must have shape (node_count,)");
    for (float value : resource.edge_values) {
      if (!std::isfinite(value))
        throw std::invalid_argument("resource edge values must be finite");
    }
    for (float value : resource.node_values) {
      if (!std::isfinite(value))
        throw std::invalid_argument("resource node values must be finite");
    }
    const ResourceTerm *gate_reference = nullptr;
    for (const ResourceTerm &term : resource.terms) {
      if (!std::isfinite(term.coefficient) || !std::isfinite(term.constant) ||
          !std::isfinite(term.gate_sign))
        throw std::invalid_argument("resource term scalar must be finite");
      const size_t expected = term.source == TermSource::EDGE_ATTRIBUTE
                                  ? n * n
                                  : term.source == TermSource::NODE_ATTRIBUTE
                                        ? n
                                        : 0;
      if (expected != 0 && term.values.size() != expected)
        throw std::invalid_argument("resource term values have invalid shape");
      if (!term.trigger_nodes.empty() && term.trigger_nodes.size() != n)
        throw std::invalid_argument(
            "resource term trigger flags must have shape (node_count,)");
      if (!term.gate_values.empty() && term.gate_values.size() != n)
        throw std::invalid_argument(
            "resource term gate values must have shape (node_count,)");
      if (term.operation == TermOperation::JOIN &&
          resource.semiring == ResourceSemiring::ARITHMETIC)
        throw std::invalid_argument(
            "a join term requires a tropical semiring");
      if (term.operation == TermOperation::JOIN &&
          term.phase != TermPhase::BEFORE_BOUND)
        throw std::invalid_argument(
            "a join term must run before the bound");
      if (term.operation == TermOperation::ASSIGN &&
          term.phase != TermPhase::AFTER_BOUND)
        throw std::invalid_argument(
            "an assign term must run after the bound");
      if (term.operation == TermOperation::CHECKPOINT &&
          term.phase != TermPhase::AFTER_BOUND)
        throw std::invalid_argument(
            "a checkpoint term must run after the bound");
      if (term.operation == TermOperation::RESTORE &&
          (term.phase != TermPhase::BEFORE_BOUND ||
           term.trigger != TermTrigger::BOUND_FAILURE ||
           term.gate != TermGate::ALWAYS))
        throw std::invalid_argument(
            "a restore term must run before the bound, trigger on "
            "bound_failure, and have no gate");
      if (term.trigger == TermTrigger::BOUND_FAILURE &&
          term.operation != TermOperation::RESTORE)
        throw std::invalid_argument(
            "bound_failure currently triggers only restore terms");
      if (term.operation == TermOperation::ADD &&
          term.phase == TermPhase::AFTER_BOUND &&
          term.trigger != TermTrigger::ALWAYS)
        throw std::invalid_argument(
            "a triggered add term must run before the bound");
      const bool selected_event =
          term.trigger == TermTrigger::RESET_DEPARTURE ||
          term.trigger == TermTrigger::RESET_ARRIVAL ||
          term.trigger == TermTrigger::CHECKPOINT_ARRIVAL;
      if (selected_event && !term.trigger_at_depot &&
          term.trigger_nodes.empty())
        throw std::invalid_argument(
            "a reset/checkpoint trigger requires at_depot or at_nodes");
      if (term.gate != TermGate::ALWAYS && term.gate_values.empty())
        throw std::invalid_argument(
            "a remainder-gated term requires gate values");
      if (term.gate != TermGate::ALWAYS) {
        if (gate_reference == nullptr) {
          gate_reference = &term;
        } else if (term.gate_sign != gate_reference->gate_sign ||
                   term.gate_values != gate_reference->gate_values) {
          throw std::invalid_argument(
              "one resource may use one remainder gate attribute and sign");
        }
      }
      for (float value : term.values) {
        if (!std::isfinite(value))
          throw std::invalid_argument("resource term values must be finite");
      }
    }
    const auto selectors_overlap = [n, this](const ResourceTerm &left,
                                             const ResourceTerm &right) {
      if (left.trigger == TermTrigger::ALWAYS ||
          right.trigger == TermTrigger::ALWAYS)
        return true;
      // Different event kinds can occur on the same transition (for example,
      // depot departure and customer arrival), so conservatively treat them as
      // overlapping state writes.
      if (left.trigger != right.trigger)
        return true;
      if (left.trigger_at_depot && right.trigger_at_depot)
        return true;
      for (int32_t depot = 0; depot < depot_count; ++depot) {
        if ((left.trigger_at_depot && !right.trigger_nodes.empty() &&
             right.trigger_nodes[depot]) ||
            (right.trigger_at_depot && !left.trigger_nodes.empty() &&
             left.trigger_nodes[depot]))
          return true;
      }
      for (size_t node = 0; node < n; ++node) {
        if (!left.trigger_nodes.empty() && !right.trigger_nodes.empty() &&
            left.trigger_nodes[node] && right.trigger_nodes[node])
          return true;
      }
      return false;
    };
    for (size_t i = 0; i < resource.terms.size(); ++i) {
      const ResourceTerm &left = resource.terms[i];
      if (left.operation != TermOperation::ASSIGN &&
          left.operation != TermOperation::CHECKPOINT)
        continue;
      for (size_t j = i + 1; j < resource.terms.size(); ++j) {
        const ResourceTerm &right = resource.terms[j];
        if (right.operation != left.operation ||
            !selectors_overlap(left, right))
          continue;
        const bool complementary =
            left.operation == TermOperation::ASSIGN &&
            left.gate != TermGate::ALWAYS &&
            right.gate != TermGate::ALWAYS && left.gate != right.gate &&
            left.gate_sign == right.gate_sign &&
            left.gate_values == right.gate_values;
        if (!complementary)
          throw std::invalid_argument(
              "overlapping state-writing terms would depend on term order");
      }
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
  // 64 is the geometric working point: on Euclidean instances the K nearest
  // neighbours carry the structure, and holding K fixed is what lets one model
  // transfer across instance sizes. Non-geometric classes (bin packing,
  // multidimensional knapsack) have no meaningful nearest neighbour -- every
  // pairwise distance is equal -- so a truncated neighbourhood would keep an
  // arbitrary index-ordered subset and strand the rest. Those run complete
  // graphs instead, which is why the bound is no longer the geometric K.
  if (max_candidates <= 0 || max_candidates > 1024) {
    throw std::invalid_argument("max_candidates must be in [1, 1024]");
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
  if (srr_exploration_budget < 0) {
    throw std::invalid_argument("srr_exploration_budget must be non-negative");
  }
}


std::vector<std::string> constraint_names(uint32_t constraints) {
  std::vector<std::string> result;
  for (const ConstraintKernelSpec &kernel : constraint_kernel_registry()) {
    if ((constraints & static_cast<uint32_t>(kernel.constraint)) != 0) {
      result.emplace_back(kernel.schema_name);
    }
  }
  return result;
}

std::vector<std::string> candidate_feature_names() {
  // The seven per-channel pressure slots that used to sit after `distance` are
  // gone; they duplicated resource_features exactly and existed only for the
  // compiled kernels. Every row is now read from resource_features.
  return {"distance", "incumbent_forward", "incumbent_backward",
          "objective_edge_term", "reverse_distance"};
}

std::vector<std::string> node_feature_names() {
  // Demand, window bounds, and service time used to have slots here. They are
  // per-node attributes of a resource, so they now live in
  // node_resource_features indexed by row, where a declared resource's
  // attributes sit alongside them under the same shared weights.
  // prize and penalty used to sit right after is_depot. They are
  // node terms of the DECLARED objective, so they now live in
  // node_objective_features, read through the coefficient vector that says what
  // each term weighs.
  return {"x",
          "y",
          "is_depot",
          "incumbent_served",
          "route_position",
          "forward_distance",
          "backward_distance",
          "mean_out_distance",
          "mean_in_distance"};
}

const char *fast_path_name(FastPath fast_path) {
  // Static storage on purpose: field_channel_names() returns by value, so
  // returning .c_str() into it would dangle the moment the vector died.
  switch (fast_path) {
  case FastPath::NONE:
    return "";
  case FastPath::CAPACITY:
    return "capacity";
  case FastPath::TIME_WINDOW:
    return "time_window";
  case FastPath::ROUTE_LIMIT:
    return "route_limit";
  case FastPath::TOUR_LIMIT:
    return "tour_limit";
  case FastPath::BACKHAUL_ORDER:
    return "backhaul_order";
  case FastPath::PICKUP_DELIVERY:
    return "pickup_delivery";
  case FastPath::PRIZE_QUOTA:
    return "prize_quota";
  }
  return "";
}

std::vector<std::string> field_channel_names() {
  std::vector<std::string> result(FIELD_CHANNEL_COUNT);
  for (const ConstraintKernelSpec &kernel : constraint_kernel_registry()) {
    if (kernel.field_channel >= 0)
      result[kernel.field_channel] =
          resource_kernel(kernel.resource_operator).name;
  }
  return result;
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
  build_constraint_kernel_set();
  // Every scale is the instance's own magnitude guarded by EPS, never floored
  // at one. A floor of one is only a no-op for instances that happen to measure
  // more than a unit across: below it the division stops normalizing and the
  // features carry raw magnitude instead, which made the whole asymmetric
  // family (metric-closure distances shrink toward zero as node count grows)
  // arrive at the model between one and four percent of the range its Euclidean
  // counterpart uses, and made the same instance in different units encode
  // differently. capacity_scale never had the floor; these now match it.
  float distance_magnitude = 0.0f;
  for (float value : problem_.distance) {
    if (std::isfinite(value))
      distance_magnitude = std::max(distance_magnitude, value);
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
    distance_magnitude =
        std::max(distance_magnitude, std::hypot(max_x - min_x, max_y - min_y));
  }
  distance_scale_ = std::max(distance_magnitude, EPS);
  float time_magnitude = 0.0f;
  float prize_magnitude = 0.0f;
  float penalty_magnitude = 0.0f;
  for (int32_t node = 0; node < problem_.node_count; ++node) {
    if (std::isfinite(problem_.tw_end[node]))
      time_magnitude = std::max(time_magnitude, problem_.tw_end[node]);
    time_magnitude = std::max(time_magnitude, problem_.service_time[node]);
    prize_magnitude = std::max(prize_magnitude, problem_.prize[node]);
    penalty_magnitude = std::max(penalty_magnitude, problem_.penalty[node]);
    pair_count_ += problem_.delivery_of_pickup[node] >= 0 ? 1 : 0;
  }
  // Travel is charged in time units, so the time scale can never be smaller
  // than the distance scale.
  time_scale_ = std::max({time_magnitude, distance_scale_, EPS});
  prize_scale_ = std::max(prize_magnitude, EPS);
  penalty_scale_ = std::max(penalty_magnitude, EPS);
  build_resource_registry();
  // The metric scan runs to completion rather than breaking at the first
  // asymmetric pair: the mean skew it accumulates is the graph-level regime
  // signal the model conditions on, and a bit alone cannot convey degree. A
  // coordinate-only instance is Euclidean, hence symmetric with zero skew, so
  // the scan is skipped entirely there.
  metric_symmetric_ = true;
  metric_skew_ = 0.0f;
  if (!problem_.distance.empty() && problem_.node_count > 1) {
    double skew_total = 0.0;
    int64_t pairs = 0;
    for (int32_t from = 0; from < problem_.node_count; ++from) {
      for (int32_t to = from + 1; to < problem_.node_count; ++to) {
        const float forward = problem_.dist(from, to);
        const float backward = problem_.dist(to, from);
        const float scale = std::max({1.0f, forward, backward});
        const float gap = std::abs(forward - backward);
        if (gap > 1.0e-5f * scale)
          metric_symmetric_ = false;
        skew_total += gap;
        ++pairs;
      }
    }
    if (pairs > 0)
      metric_skew_ = static_cast<float>(
          skew_total / static_cast<double>(pairs) /
          std::max(distance_scale_, EPS));
  }
  reversal_safe_ =
      (active_kernel_capabilities_ & KERNEL_REVERSAL_SENSITIVE) == 0 &&
      metric_symmetric_;
  build_resource_properties();
  build_candidate_graph({});
}

void RoutingDecoder::build_constraint_kernel_set() {
  active_constraint_kernels_.clear();
  active_kernel_capabilities_ = 0;
  active_field_channels_.fill(0);
  for (const ConstraintKernelSpec &kernel : constraint_kernel_registry()) {
    if (!problem_.has(kernel.constraint))
      continue;
    active_constraint_kernels_.push_back(&kernel);
    active_kernel_capabilities_ |= kernel.capabilities;
    if (kernel.field_channel >= 0)
      active_field_channels_[kernel.field_channel] = 1;
  }
}

void RoutingDecoder::build_resource_registry() {
  resources_.clear();
  active_resource_indices_.clear();
  scalar_resource_indices_.clear();
  precedence_resource_indices_.clear();
  declared_specs_valid_.clear();

  field_resource_index_.fill(-1);
  static constexpr FastPath CHANNEL_FAST_PATH[FIELD_CHANNEL_COUNT] = {
      FastPath::CAPACITY,       FastPath::TIME_WINDOW,
      FastPath::ROUTE_LIMIT,    FastPath::TOUR_LIMIT,
      FastPath::BACKHAUL_ORDER, FastPath::PICKUP_DELIVERY,
      FastPath::PRIZE_QUOTA,
  };
  const auto add_resource_row = [&](FieldChannel channel, ResourceOperator op,
                                    const char *name, bool active) {
    ResourceSpec row;
    row.name = name;
    row.active = active;
    row.op = op;
    row.scale = resource_scale(static_cast<int32_t>(channel));
    // Store the DECLARATION, not the kernel identity. The row used to keep its
    // compiled operator and every consumer that wanted the algebra asked for a
    // rewritten copy; the rewrite happens once, here, so a compiled row and an
    // equivalent declared row are the same object apart from the tag below.
    ResourceSpec spec = publish(row);
    spec.terms = build_terms(spec);
    spec.fast_path = CHANNEL_FAST_PATH[static_cast<int32_t>(channel)];
    const int32_t index = static_cast<int32_t>(resources_.size());
    resources_.push_back(std::move(spec));
    field_resource_index_[static_cast<int32_t>(channel)] = index;
  };
  // One row per constraint the problem actually declares. Every field channel
  // used to get a row here whether or not the problem used it, so the registry
  // opened with the same seven rows in the same order for every instance and a
  // row's position encoded which constraint it was. A TSP now has an empty
  // registry rather than seven inactive rows, and an appended row's position
  // depends on what else is present rather than on a fixed prefix.
  for (int32_t channel = 0; channel < FIELD_CHANNEL_COUNT; ++channel) {
    if (active_field_channels_[channel] == 0)
      continue;
    const ConstraintKernelSpec *kernel = field_channel_kernel(channel);
    if (kernel == nullptr)
      throw std::logic_error("missing field-channel kernel");
    const ResourceKernelSpec &resource_spec =
        resource_kernel(kernel->resource_operator);
    if (resource_spec.field_channel != channel)
      throw std::logic_error("resource and field-channel registries disagree");
    add_resource_row(static_cast<FieldChannel>(channel), resource_spec.op,
                     resource_spec.name, true);
  }

  for (const ResourceSpec &spec : problem_.resources) {
    if (std::any_of(resources_.begin(), resources_.end(),
                    [&](const ResourceSpec &existing) {
                      return existing.name == spec.name;
                    })) {
      throw std::invalid_argument("duplicate resource name: " + spec.name);
    }
    // A declared row is INTERPRETED (fast_path stays NONE), and that is a
    // property of the kernels, not of the row. Each compiled kernel reads the
    // problem fields it was written against -- problem_.capacity and
    // problem_.demand, problem_.tw_end, problem_.pickup_of_delivery -- rather
    // than the arrays on the row it executes. A declared row states the same
    // algebra through its own arrays, so handing it to a kernel would silently
    // enforce whatever those problem fields happen to hold (for a problem that
    // does not declare the constraint, their defaults). Matching a row onto a
    // kernel by its algebra is sound only once every kernel reads its own row;
    // until then the tag belongs to the frontend that populated those fields.
    ResourceSpec published = publish(spec);
    published.terms = build_terms(published);
    resources_.push_back(std::move(published));
  }
  // runtime_resource_scale reads the row, so every row must leave this function
  // with the normalization it will be priced by. A precedence row counts
  // unresolved obligations, which are integers in no declared unit; the scale
  // switch used to return 1 for it whatever the row said, so pinning it here
  // keeps that answer while making the row -- not the operator -- authoritative.
  for (ResourceSpec &spec : resources_) {
    if (spec.op == ResourceOperator::PRECEDENCE) {
      // A matching or class order counts at most one obligation per node, so 1
      // is the unit and the state is already on the scale trained rows sit on.
      // A DAG counts up to one per ordered pair, so the same pin would publish
      // a live state two orders of magnitude outside that range; its relation
      // count is the unit, which makes the state the fraction outstanding.
      spec.scale =
          spec.relation == PrecedenceRelation::DAG
              ? std::max(1.0f,
                         static_cast<float>(spec.predecessor_list.size()))
              : 1.0f;
    }
    if (!(spec.scale > 0.0f) || !std::isfinite(spec.scale))
      throw std::invalid_argument("resource scale must be finite and positive: " +
                                  spec.name);
  }
  for (int32_t index = 0; index < resource_count(); ++index) {
    if (resource(index).active) {
      active_resource_indices_.push_back(index);
      // Interpreted rows only: a row its kernel executes has its running
      // scalar maintained by that kernel, and advancing it here as well would
      // apply the increment twice.
      const bool interpreted = fast_path(index) == FastPath::NONE;
      if (interpreted &&
          resource(index).op == ResourceOperator::AFFINE_ACCUMULATOR)
        scalar_resource_indices_.push_back(index);
      if (resource(index).op == ResourceOperator::PRECEDENCE) {
        if (interpreted)
          precedence_resource_indices_.push_back(index);
        ResourceSpec &spec = resources_[index];
        spec.relation_count =
            spec.relation == PrecedenceRelation::PAIRWISE
                ? static_cast<int32_t>(std::count_if(
                      spec.successor.begin(), spec.successor.end(),
                      [](int32_t value) { return value >= 0; }))
            : spec.relation == PrecedenceRelation::DAG
                ? static_cast<int32_t>(spec.predecessor_list.size())
                : static_cast<int32_t>(std::count_if(
                      spec.node_class.begin(), spec.node_class.end(),
                      [](int32_t value) { return value > 0; }));
      }
      // A declared row publishes no hand-written capability entry, so its
      // search capabilities come from its algebra. Compiled kernels keep their
      // registry entry; test_resource_algebra_equivalence asserts the derived
      // capabilities reproduce it for every declarable kernel, so the two
      // sources agree wherever both exist.
      active_kernel_capabilities_ |= derived_capabilities(resource(index));
    }
  }
  // Split the active registry by who executes it. The cached-route-summary
  // screening path describes exactly the compiled field channels; everything
  // else runs through the declarative interpreter and has to be certified
  // separately.
  declared_resource_indices_.clear();
  for (int32_t index : active_resource_indices_) {
    bool backed_by_kernel = false;
    for (int32_t channel = 0; channel < FIELD_CHANNEL_COUNT; ++channel)
      backed_by_kernel |= field_resource_index_[channel] == index;
    if (!backed_by_kernel)
      declared_resource_indices_.push_back(index);
  }
  planned_screening_covers_registry_ = declared_resource_indices_.empty();
  declared_specs_valid_.assign(resource_count(), 0);
  for (int32_t index = 0; index < resource_count(); ++index) {
    // Every consumer reads the *published* algebra, abstaining rows included.
    // What escapes the language is always the extension rule -- a
    // state-dependent reload, a cross-row duration, an exempt class -- and none
    // of it is what these consume: the descriptor encodes the reset *form*, not
    // its value; the pressure and the live-state feature are positions within
    // the declared bounds. Gating them on exactness instead made the same
    // channel mean opposite things on neighbouring variants (capacity read
    // `remaining/cap` where it was declared and `served/cap` where it was not).
    // `declared_specs_valid_` remains the exactness flag, and it is what
    // `resource_declarations` publishes and the equivalence tests key on.
    declared_specs_valid_[index] = algebra_is_exact(index) ? 1 : 0;
  }
  build_relation_index();
}

void RoutingDecoder::build_relation_index() {
  // Every pairwise relation, read through the declaration its row publishes.
  // The compiled pickup-delivery kernel publishes exactly the PRECEDENCE spec
  // its hand-written branch used to supply, so folding it in here removes the
  // last place a consumer asked which path executes a constraint.
  const int32_t n = problem_.node_count;
  relation_predecessor_.assign(n, -1);
  relation_successor_.assign(n, -1);
  for (int32_t index = 0; index < resource_count(); ++index) {
    if (!resource(index).active)
      continue;
    const ResourceSpec &spec = resource(index);
    if (spec.op != ResourceOperator::PRECEDENCE ||
        spec.relation != PrecedenceRelation::PAIRWISE)
      continue;
    for (int32_t node = 0; node < n; ++node) {
      const int32_t successor =
          spec.successor.empty() ? -1 : spec.successor[node];
      const int32_t predecessor =
          spec.predecessor.empty() ? -1 : spec.predecessor[node];
      // A solution-scoped relation lets its two nodes sit in different routes,
      // so it must not restrict where a route may be cut; only route-scoped
      // rows enter the partner/delta index the SRR cutter reads.
      if (spec.scope != ResourceScope::ROUTE)
        continue;
      if (successor >= 0 && relation_successor_[node] < 0)
        relation_successor_[node] = successor;
      if (predecessor >= 0 && relation_predecessor_[node] < 0)
        relation_predecessor_[node] = predecessor;
    }
  }
}

int32_t RoutingDecoder::field_resource_index(FieldChannel channel) const {
  return field_resource_index_[static_cast<int32_t>(channel)];
}

int32_t RoutingDecoder::channel_slot(FieldChannel channel) const {
  const int32_t index = field_resource_index_[static_cast<int32_t>(channel)];
  return index >= 0 ? index
                    : resource_count() + static_cast<int32_t>(channel);
}

FastPath RoutingDecoder::fast_path(const ResourceSpec &spec) const {
  return spec.fast_path;
}

FastPath RoutingDecoder::fast_path(int32_t resource_index) const {
  return fast_path(resource(resource_index));
}


float &RoutingDecoder::slot(State &state, FieldChannel channel) const {
  return state.resource_state[static_cast<size_t>(channel_slot(channel))];
}

float RoutingDecoder::slot(const State &state, FieldChannel channel) const {
  return state.resource_state[static_cast<size_t>(channel_slot(channel))];
}

const ResourceSpec &RoutingDecoder::resource(int32_t index) const {
  if (index < 0 || index >= resource_count())
    throw std::out_of_range("resource index is out of range");
  return resources_[index];
}

float RoutingDecoder::spec_lower(const ResourceSpec &spec, int32_t node) {
  return spec.lower_values.empty()
             ? spec.lower
             : spec.lower_values[static_cast<size_t>(node)];
}

float RoutingDecoder::spec_upper(const ResourceSpec &spec, int32_t node) {
  return spec.upper_values.empty()
             ? spec.upper
             : spec.upper_values[static_cast<size_t>(node)];
}

std::vector<ResourceTerm> RoutingDecoder::build_terms(
    const ResourceSpec &spec) const {
  // Named fields are only lowering shorthand.  Every executable effect leaves
  // this function as an ordinary term with independent operation, phase,
  // trigger and gate coordinates.
  std::vector<ResourceTerm> terms = spec.terms;
  const auto add = [&terms](ResourceTerm term) {
    terms.push_back(std::move(term));
  };
  if (spec.edge_uses_distance) {
    ResourceTerm term;
    term.source = TermSource::DISTANCE;
    term.operation = TermOperation::ADD;
    term.phase = TermPhase::BEFORE_BOUND;
    term.coefficient = spec.edge_coefficient;
    add(std::move(term));
  }
  if (!spec.edge_values.empty()) {
    ResourceTerm term;
    term.source = TermSource::EDGE_ATTRIBUTE;
    term.operation = TermOperation::ADD;
    term.phase = TermPhase::BEFORE_BOUND;
    term.coefficient = spec.edge_coefficient;
    term.values = spec.edge_values;
    add(std::move(term));
  }
  if (!spec.node_values.empty()) {
    ResourceTerm term;
    term.source = TermSource::NODE_ATTRIBUTE;
    term.point = TermPoint::TO;
    term.operation = TermOperation::ADD;
    term.phase = TermPhase::BEFORE_BOUND;
    term.coefficient = spec.node_coefficient;
    term.values = spec.node_values;
    add(std::move(term));
  }
  if (!spec.departure_values.empty()) {
    ResourceTerm term;
    term.source = TermSource::NODE_ATTRIBUTE;
    term.point = TermPoint::TO;
    term.operation = TermOperation::ADD;
    term.phase = TermPhase::AFTER_BOUND;
    term.coefficient = spec.departure_coefficient;
    term.values = spec.departure_values;
    add(std::move(term));
  }
  if (!spec.opening_values.empty()) {
    // This is additive in the frozen behavior: it corrects a value installed
    // on reset rather than replacing it.
    ResourceTerm term;
    term.source = TermSource::NODE_ATTRIBUTE;
    term.point = TermPoint::TO;
    term.operation = TermOperation::ADD;
    term.phase = TermPhase::BEFORE_BOUND;
    term.trigger = TermTrigger::RESET_DEPARTURE;
    term.coefficient = spec.opening_coefficient;
    term.values = spec.opening_values;
    term.trigger_at_depot = spec.reset_at_depot;
    term.trigger_nodes = spec.reset_nodes;
    add(std::move(term));
  }
  if (!spec.join_values.empty()) {
    ResourceTerm term;
    term.source = TermSource::NODE_ATTRIBUTE;
    term.point = TermPoint::TO;
    term.operation = TermOperation::JOIN;
    term.phase = TermPhase::BEFORE_BOUND;
    term.values = spec.join_values;
    add(std::move(term));
  }
  if (spec.reset_at_depot || !spec.reset_nodes.empty()) {
    const auto reset_term = [&](float installed, TermGate gate) {
      ResourceTerm term;
      term.source = TermSource::CONSTANT;
      term.operation = TermOperation::ASSIGN;
      term.phase = TermPhase::AFTER_BOUND;
      term.trigger = TermTrigger::RESET_ARRIVAL;
      term.constant = installed;
      term.trigger_at_depot = spec.reset_at_depot;
      term.trigger_nodes = spec.reset_nodes;
      term.gate = gate;
      term.gate_values = spec.reset_guard_values;
      term.gate_sign = spec.reset_guard_sign;
      return term;
    };
    if (spec.reset_guard_values.empty()) {
      add(reset_term(spec.reset_value, TermGate::ALWAYS));
    } else {
      add(reset_term(spec.reset_value, TermGate::REMAINDER_DEFAULT));
      add(reset_term(spec.reset_otherwise,
                     TermGate::REMAINDER_ALTERNATIVE));
    }
  }
  // Route scope itself implies the depot initialization that used to live in
  // an else branch of extend_declared.  An explicit depot reset supersedes it.
  const bool has_depot_assign = std::any_of(
      terms.begin(), terms.end(), [](const ResourceTerm &term) {
        return term.operation == TermOperation::ASSIGN &&
               term.trigger == TermTrigger::RESET_ARRIVAL &&
               term.trigger_at_depot;
      });
  if (spec.scope == ResourceScope::ROUTE && !has_depot_assign) {
    ResourceTerm term;
    term.source = TermSource::CONSTANT;
    term.operation = TermOperation::ASSIGN;
    term.phase = TermPhase::AFTER_BOUND;
    term.trigger = TermTrigger::RESET_ARRIVAL;
    term.constant = spec.initial;
    term.trigger_at_depot = true;
    add(std::move(term));
  }
  if (!spec.optional_reset_nodes.empty()) {
    ResourceTerm checkpoint;
    checkpoint.source = TermSource::CONSTANT;
    checkpoint.operation = TermOperation::CHECKPOINT;
    checkpoint.phase = TermPhase::AFTER_BOUND;
    checkpoint.trigger = TermTrigger::CHECKPOINT_ARRIVAL;
    checkpoint.constant = spec.reset_value;
    checkpoint.trigger_nodes = spec.optional_reset_nodes;
    add(std::move(checkpoint));

    ResourceTerm restore;
    restore.operation = TermOperation::RESTORE;
    restore.phase = TermPhase::BEFORE_BOUND;
    restore.trigger = TermTrigger::BOUND_FAILURE;
    add(std::move(restore));
  }
  return terms;
}

float RoutingDecoder::term_value(const ResourceTerm &term, int32_t from,
                                 int32_t to) const {
  float raw = 0.0f;
  switch (term.source) {
  case TermSource::CONSTANT:
    raw = term.constant;
    break;
  case TermSource::DISTANCE:
    raw = problem_.dist(from, to);
    break;
  case TermSource::EDGE_ATTRIBUTE:
    raw = term.values[static_cast<size_t>(from) * problem_.node_count + to];
    break;
  case TermSource::NODE_ATTRIBUTE:
    raw = term.values[static_cast<size_t>(
        term.point == TermPoint::TO ? to : from)];
    break;
  }
  return term.coefficient * raw;
}

double RoutingDecoder::node_term_total(const ResourceSpec &spec,
                                       int32_t node) const {
  // Node-sourced accumulation charged at one node, independent of the edge.
  double total = 0.0;
  for (const ResourceTerm &term : spec.terms) {
    if (term.operation == TermOperation::ADD &&
        term.phase == TermPhase::BEFORE_BOUND &&
        term.trigger == TermTrigger::ALWAYS &&
        term.source == TermSource::NODE_ATTRIBUTE)
      total += term.coefficient * term.values[static_cast<size_t>(node)];
  }
  return total;
}

double RoutingDecoder::phase_total(const ResourceSpec &spec, TermPhase phase,
                                   int32_t from, int32_t to) const {
  double total = 0.0;
  for (const ResourceTerm &term : spec.terms) {
    if (term.operation == TermOperation::ADD && term.phase == phase &&
        term.trigger == TermTrigger::ALWAYS)
      total += term_value(term, from, to);
  }
  return total;
}

void RoutingDecoder::apply_add_terms(const ResourceSpec &spec, TermPhase phase,
                                     int32_t from, int32_t to,
                                     float &value) const {
  value += static_cast<float>(phase_total(spec, phase, from, to));
}

bool RoutingDecoder::has_operation(const ResourceSpec &spec,
                                   TermOperation operation) {
  for (const ResourceTerm &term : spec.terms) {
    if (term.operation == operation)
      return true;
  }
  return false;
}

bool RoutingDecoder::term_event_matches(const ResourceTerm &term, int32_t from,
                                        int32_t to, bool depot,
                                        bool bound_failed) const {
  switch (term.trigger) {
  case TermTrigger::ALWAYS:
    return true;
  case TermTrigger::BOUND_FAILURE:
    return bound_failed;
  case TermTrigger::RESET_DEPARTURE:
    return from >= 0 &&
           ((from < problem_.depot_count && term.trigger_at_depot) ||
            (!term.trigger_nodes.empty() && term.trigger_nodes[from]));
  case TermTrigger::RESET_ARRIVAL:
  case TermTrigger::CHECKPOINT_ARRIVAL:
    return (depot && term.trigger_at_depot) ||
           (to >= 0 && !term.trigger_nodes.empty() &&
            term.trigger_nodes[to]);
  }
  return false;
}

bool RoutingDecoder::guarded_node(const ResourceSpec &spec, int32_t node) {
  for (const ResourceTerm &term : spec.terms) {
    if (term.gate == TermGate::ALWAYS || term.gate_values.empty())
      continue;
    const float value = term.gate_values[static_cast<size_t>(node)];
    return term.gate_sign >= 0.0f ? value > FEASIBILITY_EPS
                                  : value < -FEASIBILITY_EPS;
  }
  return false;
}

bool RoutingDecoder::opposing_node(const ResourceSpec &spec, int32_t node) {
  for (const ResourceTerm &term : spec.terms) {
    if (term.gate == TermGate::ALWAYS || term.gate_values.empty())
      continue;
    const float value = term.gate_values[static_cast<size_t>(node)];
    // Zero matches neither side, preserving the compiled capacity convention.
    return term.gate_sign >= 0.0f ? value < -FEASIBILITY_EPS
                                  : value > FEASIBILITY_EPS;
  }
  return false;
}

std::vector<int32_t> RoutingDecoder::initial_reset_guards() const {
  std::vector<int32_t> remaining(2 * resource_count(), 0);
  for (int32_t index = 0; index < resource_count(); ++index) {
    const ResourceSpec &spec = resource(index);
    if (std::none_of(spec.terms.begin(), spec.terms.end(),
                     [](const ResourceTerm &term) {
                       return term.gate != TermGate::ALWAYS;
                     }))
      continue;
    for (int32_t node = problem_.depot_count; node < problem_.node_count;
         ++node) {
      remaining[2 * index] += guarded_node(spec, node) ? 1 : 0;
      remaining[2 * index + 1] += opposing_node(spec, node) ? 1 : 0;
    }
  }
  return remaining;
}

void RoutingDecoder::consume_reset_guards(std::vector<int32_t> &remaining,
                                          int32_t node) const {
  if (remaining.empty())
    return;
  for (int32_t index = 0; index < resource_count(); ++index) {
    const ResourceSpec &spec = resource(index);
    if (guarded_node(spec, node))
      --remaining[2 * index];
    else if (opposing_node(spec, node))
      --remaining[2 * index + 1];
  }
}

bool RoutingDecoder::tropical(const ResourceSpec &spec) {
  return spec.semiring != ResourceSemiring::ARITHMETIC;
}

float RoutingDecoder::join_identity(const ResourceSpec &spec) {
  // The value that leaves the accumulation untouched under this row's join.
  switch (spec.semiring) {
  case ResourceSemiring::MAX_PLUS:
    return -std::numeric_limits<float>::infinity();
  case ResourceSemiring::MIN_PLUS:
    return std::numeric_limits<float>::infinity();
  case ResourceSemiring::ARITHMETIC:
    break;
  }
  // Arithmetic has no join, so no operand can perturb it. Returning the max
  // identity keeps any isfinite() test on the result reading "absent".
  return -std::numeric_limits<float>::infinity();
}

float RoutingDecoder::spec_join_operand(const ResourceSpec &spec,
                                        int32_t node) {
  for (const ResourceTerm &term : spec.terms) {
    if (term.operation != TermOperation::JOIN)
      continue;
    if (term.source == TermSource::CONSTANT)
      return term.coefficient * term.constant;
    if (term.source == TermSource::NODE_ATTRIBUTE)
      return term.coefficient * term.values[static_cast<size_t>(node)];
  }
  return join_identity(spec);
}

float RoutingDecoder::join(const ResourceSpec &spec, float accumulated,
                           float operand) {
  switch (spec.semiring) {
  case ResourceSemiring::MAX_PLUS:
    return std::max(accumulated, operand);
  case ResourceSemiring::MIN_PLUS:
    return std::min(accumulated, operand);
  case ResourceSemiring::ARITHMETIC:
    break;
  }
  return accumulated;
}

// Search capabilities a row's algebra implies.
//
// These used to be written by hand next to each compiled kernel. Every property
// they encode is readable off the declared row instead: where the state lives
// (scope), and whether the extension is order-invariant. An extension is
// order-invariant when it is a monotone affine sum -- reversing or reordering a
// segment then moves the same total and the binding value is still at the end.
// A tropical clamp destroys that (arrival relative to a window depends on when
// you arrive), and so does a reset at an interior node (the reset point moves),
// and so does a mixed-sign increment (the running extremum moves).
uint32_t RoutingDecoder::derived_capabilities(const ResourceSpec &spec) {
  uint32_t capabilities = spec.scope == ResourceScope::ROUTE
                              ? KERNEL_ROUTE_STATE
                              : KERNEL_SOLUTION_STATE;
  const bool interior_reset = std::any_of(
      spec.terms.begin(), spec.terms.end(), [](const ResourceTerm &term) {
        return !term.trigger_nodes.empty() &&
               (term.operation == TermOperation::ASSIGN ||
                term.operation == TermOperation::CHECKPOINT);
      });
  bool any_positive = false;
  bool any_negative = false;
  for (const ResourceTerm &term : spec.terms) {
    if (term.operation != TermOperation::ADD ||
        term.phase != TermPhase::BEFORE_BOUND ||
        term.trigger != TermTrigger::ALWAYS)
      continue;
    any_positive = any_positive || term.coefficient > 0.0f;
    any_negative = any_negative || term.coefficient < 0.0f;
  }
  const bool mixed_sign = any_positive && any_negative;
  if (tropical(spec) || interior_reset || mixed_sign)
    capabilities |= KERNEL_ORDER_SENSITIVE | KERNEL_REVERSAL_SENSITIVE;
  if (spec.op == ResourceOperator::PRECEDENCE) {
    // A precedence relation is order-sensitive by definition, and a pairwise
    // relation additionally ties two nodes together, which generic operators
    // must respect when they cut a route.
    capabilities |= KERNEL_ORDER_SENSITIVE | KERNEL_REVERSAL_SENSITIVE;
    if (spec.relation == PrecedenceRelation::PAIRWISE)
      capabilities |= KERNEL_RELATIONAL;
  }
  return capabilities;
}

// Declarative algebra published by a compiled kernel.
//
// A compiled kernel is an execution fast path, not a private semantics. Each
// one that lies inside the declarative language publishes the resource row it
// implements, and that row -- not a switch over the kernel enum -- is what the
// descriptor is derived from and what tests replay to prove the fast path
// equivalent. Kernels whose semantics fall outside the current language return
// `declared = false` rather than a plausible-looking approximation:
//   * time_windows    needs a tropical (affine + max) operator and per-node
//                     bounds; the language has neither.
//   * backhaul_order  and pickup_delivery are precedence relations over nodes,
//                     not resource extension functions at all.
// Whether the published algebra reproduces the compiled kernel exactly.
//
// A kernel that fails this abstains: `declared_algebra` returns the bare
// registry row and `declared` is false, so the descriptor, the pressure, and
// the live-state feature all fall back to the kernel's own path rather than
// describe it with a row that is only nearly right. The node attributes are a
// deliberate exception -- see node_resource_features_ -- because in every case
// below it is the *extension rule* that escapes the language, never the
// per-node quantities.
bool RoutingDecoder::algebra_is_exact(int32_t resource_index) const {
  // Keyed on the compiled kernel, not on the operator: a row's operator is now
  // always a language operator, so it can no longer name the kernel whose
  // domain is in question.
  switch (resource(resource_index).fast_path) {
  case FastPath::CAPACITY: {
    // Both halves of the kernel are declared now: the state-dependent reload
    // as a guarded reset, and the class-ordered opening load as an opening
    // term (see publish()).
    //
    // What remains uncovered is a *declared* class-order row without the
    // problem flag. publish() runs during the registry build, before precedence
    // rows are indexed, so it cannot see one -- the opening term would be
    // missing while opening_load applied the rule at runtime.
    const bool signed_demand =
        std::any_of(problem_.demand.begin(), problem_.demand.end(),
                    [](float value) { return value < -FEASIBILITY_EPS; });
    return !signed_demand || problem_.has(BACKHAUL_ORDER) || !class_ordered();
  }
  case FastPath::TIME_WINDOW:
    // A driver break costs wall time, so a break taken by the driving-hours row
    // delays every arrival. No operator expresses one row's reset incrementing
    // another row's state.
    return std::none_of(resources_.begin(), resources_.end(),
                        [](const ResourceSpec &other) {
                          return other.active &&
                                 !other.optional_reset_nodes.empty();
                        });
  case FastPath::BACKHAUL_ORDER: {
    // The kernel latches on demand < 0 but only blocks demand > 0, so a
    // zero-demand customer is exempt from both. One non-decreasing class order
    // cannot exempt a class: it would sit below the latch and above it at once.
    for (int32_t node = problem_.depot_count; node < problem_.node_count;
         ++node) {
      if (std::abs(problem_.demand[node]) <= FEASIBILITY_EPS)
        return false;
    }
    return true;
  }
  default:
    return true;
  }
}

ResourceSpec RoutingDecoder::published_algebra(int32_t resource_index) const {
  // The row IS its declaration now: build_resource_registry rewrites a compiled
  // kernel into the language once, at construction, instead of every consumer
  // asking for a rewritten copy. Kept as an accessor because the exactness
  // question below still has two different answers to report.
  return resource(resource_index);
}

ResourceSpec RoutingDecoder::declared_algebra(int32_t resource_index,
                                              bool *declared) const {
  if (declared != nullptr)
    *declared = algebra_is_exact(resource_index);
  return resource(resource_index);
}

// The per-node term of a pairwise relation: +1 where a node opens an obligation
// and -1 where it requires one. This is the same quantity open_relation_delta
// reports, read off the row's own arrays rather than the global index, so it is
// available while the registry is still being built. Publishing it as an
// ordinary node term is what lets the relation reach the model through the
// shared per-resource weights instead of two node columns named after it.
static void publish_relation_node_term(prism::ResourceSpec &spec,
                                       int32_t node_count) {
  if (spec.relation == prism::PrecedenceRelation::DAG) {
    // Same quantity as the pairwise case, counted with multiplicity: serving a
    // node opens one obligation per successor and closes one per predecessor.
    // The matching case is this with every degree restricted to one, so the
    // descriptor reads one node term across both relations rather than two.
    if (spec.predecessor_offsets.empty())
      return;
    spec.node_values.assign(static_cast<size_t>(node_count), 0.0f);
    for (int32_t node = 0; node < node_count; ++node) {
      const size_t index = static_cast<size_t>(node);
      const float indegree =
          static_cast<float>(spec.predecessor_offsets[index + 1] -
                             spec.predecessor_offsets[index]);
      const float outdegree =
          spec.successor_count.empty()
              ? 0.0f
              : static_cast<float>(spec.successor_count[index]);
      spec.node_values[node] = outdegree - indegree;
    }
    spec.node_coefficient = 1.0f;
    return;
  }
  if (spec.relation != prism::PrecedenceRelation::PAIRWISE)
    return;
  if (spec.successor.empty() && spec.predecessor.empty())
    return;
  spec.node_values.assign(static_cast<size_t>(node_count), 0.0f);
  for (int32_t node = 0; node < node_count; ++node) {
    float delta = 0.0f;
    if (!spec.successor.empty() && spec.successor[node] >= 0)
      delta += 1.0f;
    if (!spec.predecessor.empty() && spec.predecessor[node] >= 0)
      delta -= 1.0f;
    spec.node_values[node] = delta;
  }
  spec.node_coefficient = 1.0f;
}

ResourceSpec RoutingDecoder::publish(const ResourceSpec &row) const {
  ResourceSpec spec = row;
  switch (row.op) {
  case ResourceOperator::AFFINE_ACCUMULATOR:
    return spec;
  case ResourceOperator::PRECEDENCE:
    publish_relation_node_term(spec, problem_.node_count);
    return spec;
  case ResourceOperator::CAPACITY:
    // Remaining load: what the vehicle can still take. This is the quantity the
    // kernel itself carries, so the declaration and the fast path describe one
    // number in one convention rather than two complementary ones.
    spec.op = ResourceOperator::AFFINE_ACCUMULATOR;
    spec.initial = problem_.capacity;
    spec.node_values = problem_.demand;
    spec.node_coefficient = -1.0f;
    spec.lower = 0.0f;
    spec.upper = problem_.capacity;
    spec.reset_at_depot = true;
    spec.reset_value = problem_.capacity;
    // The kernel's reload is not a constant: a vehicle leaves the depot full
    // while deliveries remain and empty once only pickups do (depot_reload).
    // Declaring that as a guard is what lets a signed-demand capacity row
    // reproduce the kernel instead of abstaining.
    spec.reset_guard_values = problem_.demand;
    spec.reset_guard_sign = 1.0f;
    spec.reset_otherwise = 0.0f;
    spec.scope = ResourceScope::ROUTE;
    spec.bound_check = BoundCheck::TRANSITION;
    // Class-ordered opening load: with a backhaul order active, a route that
    // opens on a pickup starts empty rather than carrying the reload
    // (opening_load). In this row's remaining-load convention that is the
    // reload negated on exactly the pickups, applied as the route leaves the
    // depot -- an ADD term before the bound on RESET_DEPARTURE. It deliberately
    // is not
    // an increment: the correction belongs to the reset, and pricing it as an
    // edge cost would tell the model that reaching a pickup from a depot
    // consumes a full vehicle.
    //
    // Keyed on the problem flag rather than class_ordered(), whose second
    // branch reads precedence_resource_indices_ -- not yet populated when
    // publish() runs during the registry build. algebra_is_exact keeps
    // abstaining for the case that leaves uncovered.
    if (problem_.has(BACKHAUL_ORDER)) {
      spec.opening_coefficient = 1.0f;
      spec.opening_values.assign(problem_.node_count, 0.0f);
      for (int32_t node = problem_.depot_count; node < problem_.node_count;
           ++node) {
        if (problem_.demand[node] < -FEASIBILITY_EPS)
          spec.opening_values[node] = -problem_.capacity;
      }
    }
    break;
  case ResourceOperator::ROUTE_LIMIT:
    spec.op = ResourceOperator::AFFINE_ACCUMULATOR;
    spec.initial = 0.0f;
    spec.edge_uses_distance = true;
    spec.edge_coefficient = 1.0f;
    spec.upper = problem_.route_limit;
    spec.reset_at_depot = true;
    spec.reset_value = 0.0f;
    spec.scope = ResourceScope::ROUTE;
    spec.bound_check = BoundCheck::TRANSITION;
    spec.horizon = BoundHorizon::RETURN;
    break;
  case ResourceOperator::TOUR_LIMIT:
    spec.op = ResourceOperator::AFFINE_ACCUMULATOR;
    spec.initial = 0.0f;
    spec.edge_uses_distance = true;
    spec.edge_coefficient = 1.0f;
    spec.upper = problem_.tour_limit;
    spec.reset_at_depot = true;
    spec.reset_value = 0.0f;
    // A tour budget is solution-scoped; the depot reset is what makes it
    // coincide with the route budget on the single-route variants that use it.
    spec.scope = ResourceScope::SOLUTION;
    spec.bound_check = BoundCheck::TRANSITION;
    spec.horizon = BoundHorizon::RETURN;
    break;
  case ResourceOperator::PRIZE_QUOTA:
    // Prize accumulates across the whole solution and gates route closure.
    // The compiled kernel additionally lets a route close once every customer
    // has been visited, so an instance whose total prize cannot reach the quota
    // stays feasible; the declarative bound has no way to say "and there is no
    // more prize to collect", so the two differ only on such instances.
    spec.op = ResourceOperator::AFFINE_ACCUMULATOR;
    spec.initial = 0.0f;
    spec.node_values = problem_.prize;
    spec.node_coefficient = 1.0f;
    spec.lower = problem_.prize_quota;
    spec.scope = ResourceScope::SOLUTION;
    spec.bound_check = BoundCheck::ROUTE_END;
    break;
  case ResourceOperator::TIME_WINDOW:
    // Elapsed time: travel accumulates, arrival waits for the window to open,
    // the window close is a per-node upper bound, and service delays departure
    // without being tested against the arriving node's own window. The return
    // leg must land inside the depot's window, hence the return horizon.
    spec.op = ResourceOperator::AFFINE_ACCUMULATOR;
    spec.semiring = ResourceSemiring::MAX_PLUS;
    spec.initial = 0.0f;
    spec.edge_uses_distance = true;
    spec.edge_coefficient = 1.0f;
    spec.join_values = problem_.tw_start;
    spec.upper_values = problem_.tw_end;
    spec.departure_values = problem_.service_time;
    spec.departure_coefficient = 1.0f;
    spec.reset_at_depot = true;
    spec.reset_value = 0.0f;
    spec.scope = ResourceScope::ROUTE;
    spec.bound_check = BoundCheck::TRANSITION;
    spec.horizon = BoundHorizon::RETURN;
    break;
  case ResourceOperator::BACKHAUL_ORDER:
    // Linehaul before backhaul within a route is the two-class case of a
    // non-decreasing class order. The compiled kernel uses two thresholds
    // though; see algebra_is_exact for the zero-demand case it cannot cover.
    spec.op = ResourceOperator::PRECEDENCE;
    spec.relation = PrecedenceRelation::CLASS_ORDER;
    spec.node_class.assign(problem_.node_count, 0);
    for (int32_t node = problem_.depot_count; node < problem_.node_count; ++node)
      spec.node_class[node] =
          problem_.demand[node] > FEASIBILITY_EPS ? 0 : 1;
    spec.relation_count = static_cast<int32_t>(
        std::count_if(spec.node_class.begin(), spec.node_class.end(),
                      [](int32_t value) { return value > 0; }));
    spec.scope = ResourceScope::ROUTE;
    break;
  case ResourceOperator::PICKUP_DELIVERY:
    // Each delivery requires its pickup first, and a route may not close while
    // a pickup it served is undelivered.
    spec.op = ResourceOperator::PRECEDENCE;
    spec.relation = PrecedenceRelation::PAIRWISE;
    spec.predecessor = problem_.pickup_of_delivery;
    spec.successor = problem_.delivery_of_pickup;
    spec.relation_count = static_cast<int32_t>(
        std::count_if(spec.successor.begin(), spec.successor.end(),
                      [](int32_t value) { return value >= 0; }));
    spec.scope = ResourceScope::ROUTE;
    publish_relation_node_term(spec, problem_.node_count);
    break;
  }
  return spec;
}

void RoutingDecoder::build_resource_properties() {
  resource_row_properties_.assign(
      static_cast<size_t>(resource_count()) * RESOURCE_ROW_PROPERTY_DIM, 0.0f);
  resource_term_properties_.clear();
  resource_term_counts_.assign(resource_count(), 0);
  const auto squash = [](double value) {
    value = std::max(value, 0.0);
    return static_cast<float>(value / (1.0 + value));
  };
  const auto sign_bit = [](double value) {
    return value > 0.0 ? 1.0f : value < 0.0 ? 0.0f : 0.5f;
  };
  for (int32_t index = 0; index < resource_count(); ++index) {
    const ResourceSpec &spec = resource(index);
    float *row = resource_row_properties_.data() +
                 static_cast<size_t>(index) * RESOURCE_ROW_PROPERTY_DIM;
    // Independent properties of the row, not membership tests over an enum.
    // Every complementary pair, every constant, and every property of a
    // declaration the parser rejects has been removed: a dimension earns its
    // place by being separately observable, so the encoding's width is a claim
    // about the language rather than about the implementation.
    const bool scalar = spec.op == ResourceOperator::AFFINE_ACCUMULATOR;
    const bool relation = spec.op == ResourceOperator::PRECEDENCE;
    // Carries state through an extension function, versus constraining the
    // order nodes may be served in. A relation is the complement, so it needs
    // no slot of its own; the relation form below is read only when this is 0.
    row[0] = scalar;
    // Relation form. Pairwise ties two nodes; the class order is its
    // complement within a relation, so one bit separates the two.
    row[1] = relation && spec.relation == PrecedenceRelation::PAIRWISE;
    // Which sides of the bound are finite, and whether either varies by node.
    row[2] = std::isfinite(spec.lower);
    row[3] = std::isfinite(spec.upper);
    row[4] = !spec.lower_values.empty() || !spec.upper_values.empty();
    // How early admissibility is decided: an ordinal, not a one-hot.
    // transition (1,1) < route end (0,1) < solution end (0,0), so the pair
    // orders the check phases and a new phase falls between existing points.
    row[5] = spec.bound_check == BoundCheck::TRANSITION;
    row[6] = spec.bound_check != BoundCheck::SOLUTION_END;
    // Whether the state resets at the depot. Solution scope is the complement.
    row[7] = spec.scope == ResourceScope::ROUTE;
    // How far past the transition the bound is projected: the return leg, and
    // whether that projection is construction-only.
    row[8] = spec.horizon == BoundHorizon::RETURN;
    row[9] = spec.horizon == BoundHorizon::RETURN_CONSTRUCTION;
    // Quantitative: state width, and the initial value split into magnitude
    // and sign so a sign flip is not a large step in the same coordinate.
    row[10] = squash(spec.state_dim);
    row[11] = squash(std::abs(spec.initial) / spec.scale);
    row[12] = sign_bit(spec.initial);
    // Density of the declared relation, the order-constraint analogue of an
    // increment magnitude: how much of the instance this relation constrains.
    //
    // A matching and a class order have at most one relation per node, so their
    // relation count *is* that quantity. A DAG does not: it holds up to one
    // relation per ordered pair, and dividing an O(n^2) count by n leaves the
    // coordinate outside [0, 1] and off the scale every trained row sits on.
    // Counting order-constrained nodes instead restores both, and reads the
    // same way across all three relation forms. relation_count keeps the edge
    // count, which is the right denominator for pressure and nothing else.
    int32_t constrained = spec.relation_count;
    if (spec.relation == PrecedenceRelation::DAG) {
      constrained = 0;
      for (size_t node = 0; node + 1 < spec.predecessor_offsets.size(); ++node) {
        if (spec.predecessor_offsets[node + 1] > spec.predecessor_offsets[node])
          ++constrained;
      }
    }
    row[13] = relation && problem_.node_count > 0
                  ? static_cast<float>(static_cast<double>(constrained) /
                                       problem_.node_count)
                  : 0.0f;
    // Direction had two slots. Execution is forward-only and the parser
    // rejects the others, so both were a constant: a property of a
    // declaration that cannot exist is not a property.

    resource_term_counts_[index] = static_cast<int32_t>(spec.terms.size());
    for (const ResourceTerm &term : spec.terms) {
      const size_t offset = resource_term_properties_.size();
      resource_term_properties_.resize(offset + RESOURCE_TERM_PROPERTY_DIM,
                                       0.0f);
      float *property = resource_term_properties_.data() + offset;
      // WHERE THE VALUE COMES FROM. Three independent facts, not a four-way
      // membership test: a constant reads neither endpoint nor the pair, so it
      // is the origin of this subspace rather than a slot.
      property[0] = term.source == TermSource::NODE_ATTRIBUTE;
      property[1] = term.source == TermSource::EDGE_ATTRIBUTE ||
                    term.source == TermSource::DISTANCE;
      // The metric is the one source that is intrinsic to the instance rather
      // than a declared array, which is a real distinction: it is the only
      // quantity a new problem always has.
      property[2] = term.source == TermSource::DISTANCE;
      property[3] = term.point == TermPoint::FROM;

      // WHAT THE OPERATION DOES TO THE STATE. Algebraic characterization, so
      // a future operation is a combination of these rather than a new slot.
      const bool add = term.operation == TermOperation::ADD;
      const bool join = term.operation == TermOperation::JOIN;
      const bool assign = term.operation == TermOperation::ASSIGN;
      const bool checkpoint = term.operation == TermOperation::CHECKPOINT;
      const bool restore = term.operation == TermOperation::RESTORE;
      // Linear in the incoming state (x -> x + c).
      property[4] = add;
      // Discards the incoming state rather than combining with it.
      property[5] = assign || restore;
      // Applying it twice is applying it once.
      property[6] = join || assign || checkpoint || restore;
      // Reads or writes the shadow copy rather than the live value; the only
      // axis on which a checkpoint differs from a join, and a restore from an
      // assignment.
      property[7] = checkpoint || restore;
      // Which way a join moves the value. Ordinal rather than a pair of flags:
      // min is 0, max is 1, and no join sits between them, so a further
      // semiring is a value on this axis and not a new one.
      property[8] = !join ? 0.5f
                    : spec.semiring == ResourceSemiring::MAX_PLUS ? 1.0f
                    : spec.semiring == ResourceSemiring::MIN_PLUS ? 0.0f
                                                                  : 0.5f;

      // WHEN IT ACTS. Phase relative to the admissibility test, then three
      // independent facts about the trigger. `RESET_ARRIVAL` and
      // `CHECKPOINT_ARRIVAL` need no slot: they are conditional, not
      // failure-driven, not departure-side, and the operation axis above
      // already separates a checkpoint from a reset.
      property[9] = term.phase == TermPhase::AFTER_BOUND;
      property[10] = term.trigger != TermTrigger::ALWAYS;
      property[11] = term.trigger == TermTrigger::BOUND_FAILURE;
      property[12] = term.trigger == TermTrigger::RESET_DEPARTURE;
      // Where it fires: at any depot, or at declared nodes.
      property[13] = term.trigger_at_depot;
      property[14] = !term.trigger_nodes.empty();

      // WHETHER A PREDICATE OVER THE UNSERVED REMAINDER SUPPRESSES IT.
      property[15] = term.gate != TermGate::ALWAYS;
      // Signed polarity in one coordinate: which side of the remainder the
      // gate selects, zero when ungated.
      property[16] = term.gate == TermGate::ALWAYS ? 0.5f
                     : term.gate == TermGate::REMAINDER_ALTERNATIVE
                         ? 1.0f
                         : 0.0f;
      property[17] = sign_bit(term.gate_sign);
      double magnitude = 0.0;
      size_t count = 0;
      if (term.source == TermSource::DISTANCE) {
        magnitude = std::abs(term.coefficient) * distance_scale_;
        count = 1;
      } else if (term.source == TermSource::CONSTANT) {
        magnitude = std::abs(term.coefficient * term.constant);
        count = 1;
      } else {
        for (float value : term.values)
          magnitude += std::abs(term.coefficient * value);
        count = term.values.size();
      }
      // HOW LARGE IT IS, split from its direction so a sign flip is not a
      // large step along the magnitude axis.
      property[18] = squash(count ? magnitude / count / spec.scale : 0.0);
      // The coefficient's direction. Kept apart from the gate sign above:
      // folding them would conflate two unrelated polarities.
      property[19] = sign_bit(term.coefficient);
    }
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

bool RoutingDecoder::field_channel_active(int32_t channel) const {
  return channel >= 0 && channel < FIELD_CHANNEL_COUNT &&
         active_field_channels_[channel] != 0;
}

float RoutingDecoder::objective_edge_cost(int32_t from, int32_t to) const {
  const float travel = problem_.dist(from, to);
  const ObjectiveSpec &obj = problem_.objective;
  if (to < problem_.depot_count)
    // The return-to-depot leg is charged through the declared travel term for
    // closed routes and is genuinely free for open routes -- exactly matching
    // the true objective accumulated in transition()/finish(). Charging it for
    // open routes used to hide a phantom cost in the ranking energy that the
    // learned field had to counteract, making guidance net-harmful on open
    // variants. The fragmentation this previously guarded against (free returns
    // making the depot the cheapest move at every step) is now handled
    // structurally in select_next(), which drops depot options while a customer
    // can still legally extend the open route.
    //
    // It used to charge raw travel here, which is the declared term only when
    // the objective happens to be plain minimized distance. A prize objective
    // declares no travel term at all (distance_coeff == 0, sense == -1), so raw
    // travel put a cost on the depot leg that the objective does not contain and
    // that no positive rescale of the coefficients could move.
    return problem_.open_route
               ? 0.0f
               : obj.sense * obj.distance_coeff * travel +
                     obj.distance_regularizer * travel / distance_scale_;
  // Marginal change to the (minimization-normalized) objective from traversing
  // into `to`: it adds `travel`, collects `prize[to]`, and removes `penalty[to]`
  // from the unvisited set. The regularizer is a scale-relative travel tie-break
  // for node-only objectives that would otherwise leave many edges tied.
  return obj.sense * (obj.distance_coeff * travel +
                      obj.visit_coeff * problem_.prize[to] -
                      obj.miss_coeff * problem_.penalty[to]) +
         obj.distance_regularizer * travel / distance_scale_;
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
  // Every row carries its own normalization. An op->channel switch used to sit
  // here, so a compiled kernel's scale came from its channel while a declared
  // row's came from the row; the two answers had to be kept in agreement by
  // hand, and a row whose operator matched a compiled kernel could not choose
  // its own units. build_resource_registry now resolves the scale once, into
  // the row, whichever path produced it.
  return std::max(resource(resource_index).scale, EPS);
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
  // Normalize by the objective's own energy scale, not by a distance scale:
  // objective_edge_cost mixes travel with prize and penalty terms, so dividing
  // it by a length made the ratio depend on the unit the instance happened to
  // be written in and on a positive rescale of the coefficients, neither of
  // which changes the problem. Against the row-centered RMS of the same
  // quantity the ratio is dimensionless -- it reports how offset-dominated the
  // objective is, which is what the field can actually use.
  const double ratio = (total / static_cast<double>(count)) /
                       std::max<double>(objective_energy_scale_, EPS);
  // ratio / (1 + ratio) squashes [0, inf) into [0, 1) without a hard clamp.
  return static_cast<float>(ratio / (1.0 + ratio));
}

void RoutingDecoder::refresh_objective_energy_scale() {
  // Center each outgoing candidate row before taking an RMS. This measures the
  // objective differences that can actually change a decision, rather than an
  // arbitrary absolute offset. Under c -> k*c (k > 0), the scale also changes
  // by k, so the complete stochastic policy is invariant at fixed beta.
  double squared_difference = 0.0;
  int64_t difference_count = 0;
  for (int32_t from = 0; from < problem_.node_count; ++from) {
    const int32_t begin = edge_offsets_[from];
    const int32_t end = edge_offsets_[from + 1];
    if (begin == end)
      continue;
    double mean = 0.0;
    for (int32_t edge = begin; edge < end; ++edge)
      mean += objective_edge_costs_[edge];
    mean /= static_cast<double>(end - begin);
    for (int32_t edge = begin; edge < end; ++edge) {
      const double difference = objective_edge_costs_[edge] - mean;
      squared_difference += difference * difference;
      ++difference_count;
    }
  }
  const double row_rms =
      difference_count > 0
          ? std::sqrt(squared_difference /
                      static_cast<double>(difference_count))
          : 0.0;
  if (std::isfinite(row_rms) && row_rms > EPS) {
    objective_energy_scale_ = static_cast<float>(row_rms);
    return;
  }

  // Degenerate rows still need a valid scale. Mean absolute magnitude retains
  // positive scale equivariance when all candidates in a row have equal cost.
  double absolute_total = 0.0;
  for (float cost : objective_edge_costs_)
    absolute_total += std::abs(cost);
  const double mean_absolute = objective_edge_costs_.empty()
                                   ? 0.0
                                   : absolute_total / objective_edge_costs_.size();
  objective_energy_scale_ =
      std::isfinite(mean_absolute) && mean_absolute > EPS
          ? static_cast<float>(mean_absolute)
          : 1.0f;
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
  // A kernel that publishes a declaration is priced from that declaration, so
  // the pressure the model sees does not depend on which execution path a
  // constraint happens to take. Only a kernel outside the language keeps its
  // hand-written per-channel formula.
  return resource_pressure_of(resource(resource_index), from, to);
}

float RoutingDecoder::compiled_pressure_of(const ResourceSpec &spec,
                                           int32_t from, int32_t to) const {
  // The kernel's own hand-written per-channel formula, selected by which kernel
  // runs the row. resource_pressure_of reads the row's algebra instead; the two
  // must agree, which is what test_published_pressure_matches_the_compiled_kernel
  // checks.
  if (!spec.active)
    return 0.0f;
  switch (spec.fast_path) {
  case FastPath::NONE:
    return resource_pressure_of(spec, from, to);
  case FastPath::CAPACITY:
  case FastPath::TIME_WINDOW:
  case FastPath::ROUTE_LIMIT:
  case FastPath::TOUR_LIMIT:
  case FastPath::BACKHAUL_ORDER:
  case FastPath::PICKUP_DELIVERY:
  case FastPath::PRIZE_QUOTA:
    return analytic_resource_pressure(
        from, to, static_cast<int32_t>(spec.fast_path) - 1);
  }
  return 0.0f;
}

float RoutingDecoder::compiled_resource_pressure(int32_t from, int32_t to,
                                                 int32_t resource_index) const {
  // The registry row's own pricing, bypassing its published declaration. This
  // is the reference the declaration has to reproduce: a published row that
  // prices differently from the kernel it replaces is a silent feature-level
  // divergence that solve-level equivalence cannot see, because both sides of
  // that comparison read the same published algebra.
  return compiled_pressure_of(resource(resource_index), from, to);
}

std::vector<float> RoutingDecoder::compiled_resource_pressures() const {
  std::vector<float> result(static_cast<size_t>(edge_count()) *
                                resource_count(),
                            0.0f);
  for (int32_t from = 0; from < problem_.node_count; ++from) {
    for (int32_t edge = edge_offsets_[from]; edge < edge_offsets_[from + 1];
         ++edge) {
      for (int32_t index = 0; index < resource_count(); ++index) {
        result[static_cast<size_t>(edge) * resource_count() + index] =
            compiled_resource_pressure(from, edge_to_[edge], index);
      }
    }
  }
  return result;
}

float RoutingDecoder::resource_pressure_of(const ResourceSpec &spec,
                                           int32_t from, int32_t to) const {
  if (!spec.active)
    return 0.0f;
  switch (spec.op) {
  case ResourceOperator::PRECEDENCE: {
    // Precedence pressure is the share of the row's relations an edge puts in
    // the wrong order: entering a successor whose predecessor is not the node
    // we came from, or descending to an earlier class. Both are read off the
    // row's own arrays rather than from a per-channel formula. What counts as
    // one relation differs by relation kind, so the denominator does too.
    if (spec.relation == PrecedenceRelation::PAIRWISE) {
      // One relation per pair, which is what relation_count counts here.
      const float share = 1.0f / std::max(spec.relation_count, 1);
      const int32_t required =
          spec.predecessor.empty() ? -1
                                   : spec.predecessor[static_cast<size_t>(to)];
      return required >= 0 && required != from ? share : 0.0f;
    }
    if (spec.relation == PrecedenceRelation::DAG) {
      // Same share per relation, counted with multiplicity: an arc into a node
      // with many outstanding predecessors is under more pressure than one into
      // a node with a single predecessor, which the matching case cannot show.
      if (spec.predecessor_offsets.empty())
        return 0.0f;
      const float share = 1.0f / std::max(spec.relation_count, 1);
      const size_t head = static_cast<size_t>(to);
      float pending = 0.0f;
      for (int32_t at = spec.predecessor_offsets[head];
           at < spec.predecessor_offsets[head + 1]; ++at) {
        if (spec.predecessor_list[static_cast<size_t>(at)] != from)
          pending += 1.0f;
      }
      return pending * share;
    }
    if (spec.node_class.empty())
      return 0.0f;
    // A route-scoped class order resets at the depot, so entering one ends the
    // ordering rather than taking part in it. Without this the depot's class
    // (zero, the lowest, because published_algebra only classifies customers)
    // made every backhaul->depot arc read as a descent -- pressure on the
    // ordinary way a backhaul route closes, and on nothing the kernel prices.
    if (spec.scope == ResourceScope::ROUTE && to < problem_.depot_count)
      return 0.0f;
    // Every classified node takes part in a class order: a linehaul is as much
    // a party to "linehaul before backhaul" as a backhaul is. relation_count
    // counts only the nonzero classes, which is the right quantity for the
    // descriptor (it reports how much of the instance is constrained) but the
    // wrong one here -- it would scale this pressure with the instance's
    // composition, reading one misordered arc twenty times larger at a five
    // percent backhaul share than at sixty, for the identical violation.
    const float share = 1.0f / std::max(problem_.customer_count(), 1);
    return spec.node_class[static_cast<size_t>(from)] >
                   spec.node_class[static_cast<size_t>(to)]
               ? share
               : 0.0f;
  }
  case ResourceOperator::AFFINE_ACCUMULATOR: {
    if (tropical(spec)) {
    // Tropical pressure: how far the transition pushes outside the row's window
    // in either direction -- arriving before the target opens (wait) or after it
    // closes (warp). Every term is read off this row's own clamp, bound, and
    // departure fields, so it is the same quantity the hand-written time-window
    // formula computed, without a per-channel case.
    // Edge-sourced accumulation only: a node term is the arrival's own charge,
    // not the travel between the pair.
    double travel = 0.0;
    for (const ResourceTerm &term : spec.terms) {
      if (term.operation == TermOperation::ADD &&
          term.phase == TermPhase::BEFORE_BOUND &&
          term.trigger == TermTrigger::ALWAYS &&
          term.source != TermSource::NODE_ATTRIBUTE)
        travel += term_value(term, from, to);
    }
    // The departure charge is levied at the node being left.
    const double departure = phase_total(spec, TermPhase::AFTER_BOUND, from, from);
    const auto floor_at = [&](int32_t node) {
      const float value = spec_join_operand(spec, node);
      return std::isfinite(value) ? static_cast<double>(value) : 0.0;
    };
    const double wait =
        std::isfinite(spec_upper(spec, from))
            ? std::max(floor_at(to) - spec_upper(spec, from) - departure -
                           travel,
                       0.0)
            : 0.0;
    const double warp =
        std::isfinite(spec_upper(spec, to))
            ? std::max(floor_at(from) + departure + travel -
                           spec_upper(spec, to),
                       0.0)
            : 0.0;
    return static_cast<float>(wait + warp);
    }
    // Arithmetic semiring: the pressure is the plain signed increment measured
    // against whichever bounds the row declares.
    // Only unconditional ADD terms before the bound price an edge. An opening correction or a
    // departure term is not what traversing this edge costs, and reading them
    // here is exactly the mistake that made a depot-to-pickup edge look like it
    // consumed a whole vehicle.
    const double delta = phase_total(spec, TermPhase::BEFORE_BOUND, from, to);
    // A declared horizon commits the depot return leg as well, so the pressure
    // an edge exerts includes it. Without this a row declaring
    // `horizon: return` reports half the pressure of the compiled route- and
    // tour-limit kernels it is otherwise equivalent to.
    double horizon_leg = 0.0;
    if (spec.horizon != BoundHorizon::TRANSITION && !problem_.open_route &&
        problem_.depot_count > 0) {
      horizon_leg = std::numeric_limits<double>::infinity();
      for (int32_t depot = 0; depot < problem_.depot_count; ++depot) {
        double leg = 0.0;
        for (const ResourceTerm &term : spec.terms) {
          if (term.operation == TermOperation::ADD &&
              term.phase == TermPhase::BEFORE_BOUND &&
              term.trigger == TermTrigger::ALWAYS &&
              term.source != TermSource::NODE_ATTRIBUTE)
            leg += term_value(term, to, depot);
        }
        horizon_leg = std::min(horizon_leg, std::abs(leg));
      }
    }
    // Pressure is the bound-worsening magnitude in physical units, summed over
    // whichever bounds the row actually declares. A two-sided row (capacity:
    // 0 <= load <= cap) worsens in BOTH directions -- a linehaul consumes
    // toward the floor, a backhaul fills toward the ceiling -- so selecting a
    // single branch by which bound happens to be finite silently zeroed the
    // linehaul side on every pure-capacity variant. The floor term depends on
    // what the bound means: a terminal requirement (a prize quota, tested when
    // the route ends) is priced by how much of it this edge leaves unmet, while
    // an invariant floor (a battery that must never run out) is priced by how
    // far this edge descends. The two coincide wherever the floor is zero.
    double pressure = 0.0;
    if (std::isfinite(spec_upper(spec, to)))
      pressure += std::max(delta, 0.0);
    const double floor_bound = spec_lower(spec, to);
    if (std::isfinite(floor_bound))
      pressure += spec.bound_check == BoundCheck::ROUTE_END
                      ? std::max(floor_bound - delta, 0.0)
                      : std::max(-delta, 0.0);
    return static_cast<float>(pressure + horizon_leg);
  }
  }
  return 0.0f;
}

void RoutingDecoder::validate_guidance(const float *edge_field,
                                       const float *edge_additive,
                                       const float *edge_state_field,
                                       const float *multipliers,
                                       const float *coupler_weights,
                                       const float *coupler_bias,
                                       const float *objective_residual) const {
  const size_t value_count =
      static_cast<size_t>(edge_count()) * resource_count();
  if (edge_field != nullptr) {
    for (size_t index = 0; index < value_count; ++index) {
      if (!std::isfinite(edge_field[index])) {
        throw std::invalid_argument(
            "edge_field residuals must be finite");
      }
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
  if (edge_state_field != nullptr) {
    const size_t count = static_cast<size_t>(edge_count()) * resource_count() *
                         live_state_feature_count();
    for (size_t index = 0; index < count; ++index) {
      if (!std::isfinite(edge_state_field[index])) {
        throw std::invalid_argument(
            "state-conditioned field values must be finite");
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
  if (objective_residual != nullptr) {
    for (int32_t edge = 0; edge < edge_count(); ++edge) {
      if (!std::isfinite(objective_residual[edge])) {
        throw std::invalid_argument(
            "objective energy residuals must be finite");
      }
    }
  }
}

std::vector<float> RoutingDecoder::live_state_features(const State &state) const {
  const auto unit = [](double value) {
    return static_cast<float>(std::clamp(value, 0.0, 1.0));
  };
  std::vector<float> result(resource_count(), 0.0f);
  for (int32_t index = 0; index < resource_count(); ++index) {
    if (!resource(index).active)
      continue;
    // A row that publishes a declaration is read through that declaration: its
    // state is mirrored into the generic vector, so the feature the model sees
    // does not depend on which path executes the constraint.
    // One path: the row's own declaration. A per-channel switch used to sit
    // here as a fallback for compiled kernels, reachable only before the
    // registry was built; a row now arrives already stating its algebra.
    const ResourceSpec &spec = resource(index);
    result[index] =
        spec.op == ResourceOperator::PRECEDENCE
            ? unit(static_cast<double>(state.resource_state[index]) /
                   std::max(spec.relation_count, 1))
            : resource_state_feature(state, index);
  }
  return result;
}

float RoutingDecoder::resource_state_feature(const State &state,
                                             int32_t resource_index) const {
  const ResourceSpec &spec = resource(resource_index);
  const float value = state.resource_state[resource_index];
  const float lower = spec_lower(spec, state.current);
  const float upper = spec_upper(spec, state.current);
  if (std::isfinite(lower) && std::isfinite(upper))
    return std::clamp((value - lower) / std::max(upper - lower, EPS), 0.0f,
                      1.0f);
  if (std::isfinite(lower))
    return std::clamp(1.0f - (value - lower) / runtime_resource_scale(resource_index),
                      0.0f, 1.0f);
  if (std::isfinite(upper))
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

// The per-edge half of the learned live-state response. It is deliberately
// LINEAR in the live state: the SRR aggregate is a set of prefix sums built
// once and queried at many anchors, and only a term of this shape lets the
// anchor's state factor out of a sum over edges. A per-edge sigmoid gain would
// force the state to be frozen at aggregate-build time, which is exactly the
// anchor-varying behaviour we want to keep.
double RoutingDecoder::resource_state_field_value(
    int32_t edge, int32_t channel, const float *edge_state_field,
    const float *live_state) const {
  if (edge < 0 || edge_state_field == nullptr || live_state == nullptr)
    return 0.0;
  const float *row =
      edge_state_field +
      (static_cast<size_t>(edge) * resource_count() + channel) *
          live_state_feature_count();
  double total = 0.0;
  for (int32_t feature = 0; feature < live_state_feature_count(); ++feature)
    total += row[feature] * live_state[feature];
  return total;
}

void RoutingDecoder::record_decision(RolloutTrace *trace, int32_t current,
                                     const std::vector<int32_t> &valid_indices,
                                     int32_t chosen_index, bool stochastic,
                                     float log_probability,
                                     const std::vector<float> &live_state) const {
  if (trace == nullptr || chosen_index < 0)
    return;
  if (static_cast<int32_t>(live_state.size()) != live_state_feature_count())
    throw std::logic_error("live state must have one value per registry row");
  trace->current_nodes.push_back(current);
  trace->valid_indices.insert(trace->valid_indices.end(), valid_indices.begin(),
                              valid_indices.end());
  trace->valid_offsets.push_back(
      static_cast<int32_t>(trace->valid_indices.size()));
  trace->chosen_indices.push_back(chosen_index);
  trace->stochastic.push_back(stochastic ? 1 : 0);
  trace->log_probabilities.push_back(log_probability);
  trace->live_state.insert(trace->live_state.end(), live_state.begin(),
                           live_state.end());
}

double RoutingDecoder::field_score(int32_t from, int32_t to, int32_t edge,
                                   const float *edge_field,
                                   const float *edge_additive,
                                   const float *edge_state_field,
                                   const float *multipliers,
                                   const float *coupler_weights,
                                   const float *coupler_bias,
                                   const float *live_state) const {
  double result = 0.0;
  for (int32_t channel = 0; channel < resource_count(); ++channel) {
    if (!resource(channel).active)
      continue;
    const double multiplier = coupled_multiplier(
        channel, multipliers, coupler_weights, coupler_bias, live_state);
    result += multiplier * resource_field_value(
                               from, to, edge, channel, edge_field,
                               edge_additive);
    const double base = multipliers == nullptr ? 1.0 : multipliers[channel];
    result += base * resource_state_field_value(edge, channel, edge_state_field,
                                                live_state);
  }
  return result;
}

double RoutingDecoder::resource_field_value(
    int32_t from, int32_t to, int32_t edge, int32_t channel,
    const float *edge_field, const float *edge_additive) const {
  // V4 resource guidance is a direct learned signed field. Analytic pressure is
  // available to the GNN as an input feature but is never injected into energy
  // by the decoder. Off-graph edges have no aligned model output and therefore
  // receive a neutral zero field; exact feasibility and objective energy remain.
  const double field = edge >= 0 && edge_field != nullptr
                           ? edge_field[static_cast<size_t>(edge) *
                                            resource_count() +
                                        channel]
                           : 0.0;
  const double additive =
      edge >= 0 && edge_additive != nullptr
          ? edge_additive[static_cast<size_t>(edge) * resource_count() +
                          channel]
          : 0.0;
  return field + additive;
}

double RoutingDecoder::edge_energy(int32_t from, int32_t to, int32_t edge,
                                   const float *edge_field,
                                   const float *edge_additive,
                                   const float *edge_state_field,
                                   const float *multipliers,
                                   const float *coupler_weights,
                                   const float *coupler_bias,
                                   const float *live_state,
                                   const float *objective_residual) const {
  const double learned_objective =
      edge >= 0 && objective_residual != nullptr ? objective_residual[edge]
                                                 : 0.0;
  const double objective_weight = coupled_multiplier(
      objective_multiplier(), multipliers, coupler_weights, coupler_bias,
      live_state);
  return objective_weight *
             (objective_edge_cost(from, to) / objective_energy_scale_ +
              learned_objective) +
         field_score(from, to, edge, edge_field, edge_additive, edge_state_field, multipliers,
                     coupler_weights, coupler_bias, live_state);
}

void RoutingDecoder::build_candidate_graph(const std::vector<int32_t> &incumbent,
                                           std::vector<float> *edge_field,
                                           std::vector<float> *edge_additive,
                                           std::vector<float> *edge_state_field,
                                           std::vector<float>
                                               *objective_residual,
                                           std::vector<float> *coupler_weights,
                                           std::vector<float> *coupler_bias) {
  const int32_t n = problem_.node_count;
  const int32_t k = std::min(candidate_config_.max_candidates, n - 1);

  const std::vector<int32_t> old_offsets = std::move(edge_offsets_);
  const std::vector<int32_t> old_to = std::move(edge_to_);
  std::vector<float> old_field;
  std::vector<float> old_additive;
  std::vector<float> old_residual;
  std::vector<float> old_state_field;
  if (edge_field != nullptr)
    old_field.swap(*edge_field);
  if (edge_additive != nullptr)
    old_additive.swap(*edge_additive);
  if (objective_residual != nullptr)
    old_residual.swap(*objective_residual);
  if (edge_state_field != nullptr)
    old_state_field.swap(*edge_state_field);

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
  if (objective_residual != nullptr)
    objective_residual->assign(edge_to_.size(), 0.0f);
  // A rebuilt edge with no predecessor contributes no state-conditioned energy,
  // the same "no learned opinion yet" default the additive field and the
  // objective residual use.
  if (edge_state_field != nullptr) {
    edge_state_field->assign(static_cast<size_t>(edge_to_.size()) *
                                 resource_count() *
                                 live_state_feature_count(),
                             0.0f);
  }
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
      objective_edge_costs_[edge] = objective_edge_cost(from, to);
      for (int32_t channel = 0; channel < resource_count(); ++channel) {
        resource_pressure_[static_cast<size_t>(edge) * resource_count() +
                           channel] =
            runtime_resource_pressure(from, to, channel);
        const ResourceSpec &spec = resource(channel);
        const bool reset = std::any_of(
            spec.terms.begin(), spec.terms.end(), [&](const ResourceTerm &term) {
              if (term.trigger == TermTrigger::RESET_DEPARTURE)
                return term_event_matches(term, from, to,
                                          to < problem_.depot_count, false);
              if (term.trigger == TermTrigger::RESET_ARRIVAL)
                return term_event_matches(term, from, to,
                                          to < problem_.depot_count, false);
              return term.operation == TermOperation::CHECKPOINT &&
                     from >= 0 && !term.trigger_nodes.empty() &&
                     term.trigger_nodes[from];
            });
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
      if (objective_residual != nullptr && preserved &&
          static_cast<size_t>(old_edge) < old_residual.size())
        (*objective_residual)[edge] = old_residual[old_edge];
      const size_t state_field_stride =
          static_cast<size_t>(resource_count()) * live_state_feature_count();
      if (edge_state_field != nullptr && preserved &&
          static_cast<size_t>(old_edge + 1) * state_field_stride <=
              old_state_field.size()) {
        std::copy_n(old_state_field.data() +
                        static_cast<size_t>(old_edge) * state_field_stride,
                    state_field_stride,
                    edge_state_field->data() +
                        static_cast<size_t>(edge) * state_field_stride);
      }
    }
  }
  refresh_objective_energy_scale();
  incumbent_route_ = incumbent;
  build_model_features();
  ++graph_version_;
}

void RoutingDecoder::build_incumbent_suffix_state() {
  // Reverse counterpart of the forward incumbent replay: for every resource at
  // once, how much of that row the REMAINING route still spends. Walking the
  // incumbent backwards accumulates each row's own declared increment -- node
  // term plus outgoing edge term, or the open-relation delta for a precedence
  // row -- so a declared row gets the quantity the hand-written backward_load /
  // backward_time / backward_open_pickups slots only ever gave three compiled
  // kernels. A route-scoped row resets at its depot, so each node reports its
  // own route's remainder rather than the whole tour's.
  //
  // Replaying transition() in reverse would not do: resources are not
  // symmetric (a time window read backwards is not a time window), whereas the
  // accumulated increment is well defined in either direction.
  const int32_t n = problem_.node_count;
  incumbent_suffix_state_.assign(static_cast<size_t>(n) * resource_count(),
                                 0.0f);
  incumbent_suffix_features_.assign(
      static_cast<size_t>(n) * resource_count() *
          RESOURCE_SUFFIX_FEATURE_COUNT,
      0.0f);
  if (incumbent_route_.empty())
    return;
  const auto unit = [](double value) {
    return static_cast<float>(std::clamp(value, 0.0, 1.0));
  };
  // Resolved once per row, not once per (position, row): published_algebra
  // returns by value and reassigns several node-sized vectors, so calling it
  // inside the walk deep-copied the whole spec route_length times over. It
  // depends only on the registry row and the problem, neither of which moves
  // during the walk, so hoisting it is an exact no-op.
  std::vector<ResourceSpec> specs(resource_count());
  for (int32_t index = 0; index < resource_count(); ++index) {
    if (resource(index).active)
      specs[index] = resource(index);
  }
  std::vector<double> running(resource_count(), 0.0);
  std::vector<std::array<double, RESOURCE_SUFFIX_FEATURE_COUNT>>
      running_features(resource_count());
  for (auto &parts : running_features)
    parts.fill(0.0);
  for (int32_t position = static_cast<int32_t>(incumbent_route_.size()) - 1;
       position >= 0; --position) {
    const int32_t node = incumbent_route_[position];
    const int32_t next =
        position + 1 < static_cast<int32_t>(incumbent_route_.size())
            ? incumbent_route_[position + 1]
            : -1;
    const bool at_depot = node < problem_.depot_count;
    for (int32_t index = 0; index < resource_count(); ++index) {
      if (!resource(index).active)
        continue;
      const ResourceSpec &spec = specs[index];
      const double scale = std::max<double>(spec.scale, EPS);
      double increment = 0.0;
      double departure = 0.0;
      if (spec.op == ResourceOperator::PRECEDENCE) {
        increment = static_cast<double>(open_relation_delta(node));
      } else {
        increment += node_term_total(spec, node);
        if (next >= 0) {
          for (const ResourceTerm &term : spec.terms) {
            if (term.operation == TermOperation::ADD &&
                term.phase == TermPhase::BEFORE_BOUND &&
                term.trigger == TermTrigger::ALWAYS &&
                term.source != TermSource::NODE_ATTRIBUTE)
              increment += term_value(term, node, next);
          }
        }
        // A departure term is charged by transition() after serving this node.
        // Omitting it made a time-window suffix contain travel but none of the
        // remaining service time.  Keep it separate as well as in the total so
        // a shared encoder can recover either statistic without knowing that the
        // row happens to be a time window.
        departure = phase_total(spec, TermPhase::AFTER_BOUND, node, node);
        increment += departure;
      }
      const double total = running[index] + increment;
      incumbent_suffix_state_[static_cast<size_t>(node) * resource_count() +
                              index] = unit(std::abs(total) / scale);
      auto &parts = running_features[index];
      parts[static_cast<int32_t>(ResourceSuffixFeature::POSITIVE_TOTAL)] +=
          std::max(increment, 0.0);
      parts[static_cast<int32_t>(ResourceSuffixFeature::NEGATIVE_TOTAL)] +=
          std::max(-increment, 0.0);
      parts[static_cast<int32_t>(
          ResourceSuffixFeature::POSITIVE_DEPARTURE)] +=
          std::max(departure, 0.0);
      parts[static_cast<int32_t>(
          ResourceSuffixFeature::NEGATIVE_DEPARTURE)] +=
          std::max(-departure, 0.0);
      float *published = incumbent_suffix_features_.data() +
                         (static_cast<size_t>(node) * resource_count() + index) *
                             RESOURCE_SUFFIX_FEATURE_COUNT;
      for (int32_t feature = 0; feature < RESOURCE_SUFFIX_FEATURE_COUNT;
           ++feature)
        published[feature] = unit(parts[feature] / scale);
      // Going backwards, a depot is the START of the route just accumulated, so
      // the reset applies after it is reported.
      const bool reset = at_depot && std::any_of(
          spec.terms.begin(), spec.terms.end(), [](const ResourceTerm &term) {
            return term.operation == TermOperation::ASSIGN &&
                   term.trigger == TermTrigger::RESET_ARRIVAL &&
                   term.trigger_at_depot;
          });
      running[index] = reset ? 0.0 : total;
      if (reset)
        parts.fill(0.0);
    }
  }
}

void RoutingDecoder::build_model_features() {
  const int32_t n = problem_.node_count;
  const auto unit = [](double value) {
    return static_cast<float>(std::clamp(value, 0.0, 1.0));
  };
  node_features_.assign(static_cast<size_t>(n) * NODE_FEATURE_COUNT, 0.0f);
  // Per-node, per-resource attributes from each row's published algebra. This
  // reads published_algebra rather than declared_algebra on purpose: where a
  // kernel abstains it is the extension rule that escapes the language (a
  // state-dependent reload, a cross-row duration, an exempt class), never the
  // per-node quantities, so the attributes here stay exact even when the row
  // as a whole is not publishable.
  node_resource_features_.assign(
      static_cast<size_t>(n) * resource_count() * NODE_RESOURCE_FEATURE_COUNT,
      0.0f);
  for (int32_t index = 0; index < resource_count(); ++index) {
    if (!resource(index).active)
      continue;
    const ResourceSpec &spec = resource(index);
    const double scale = std::max<double>(spec.scale, EPS);
    // Highest declared class, so the rank below is a position in [0, 1] rather
    // than a raw label whose meaning would change with the number of classes.
    int32_t top_class = 0;
    for (int32_t value : spec.node_class)
      top_class = std::max(top_class, value);
    for (int32_t node = 0; node < n; ++node) {
      float *slot = node_resource_features_.data() +
                    (static_cast<size_t>(node) * resource_count() + index) *
                        NODE_RESOURCE_FEATURE_COUNT;
      const double increment = node_term_total(spec, node);
      slot[0] = unit(std::max(increment, 0.0) / scale);
      slot[1] = unit(std::max(-increment, 0.0) / scale);
      slot[2] = 0.0f;
      for (const ResourceTerm &term : spec.terms) {
        if (term.operation == TermOperation::JOIN) {
          slot[2] = unit(term_value(term, node, node) / scale);
          break;
        }
      }
      // The binding side of the bound, normalized. An unbounded row reports 1,
      // matching how an infinite window end has always been encoded.
      const float upper = spec_upper(spec, node);
      const float lower = spec_lower(spec, node);
      slot[3] = std::isfinite(upper)  ? unit(upper / scale)
                : std::isfinite(lower) ? unit(lower / scale)
                                       : 1.0f;
      slot[4] = !std::any_of(spec.terms.begin(), spec.terms.end(),
                             [](const ResourceTerm &term) {
                               return term.operation == TermOperation::ADD &&
                                      term.phase == TermPhase::AFTER_BOUND;
                             })
                    ? 0.0f
                    : unit(phase_total(spec, TermPhase::AFTER_BOUND, node, node) /
                           scale);
      // Rank within whatever order this row declares. A class order reports the
      // node's class; a pairwise relation reports which side of the pair the
      // node sits on, which is the two-element case of the same thing. Zero
      // means "this row declares no order here", so a node that takes part in
      // no relation stays distinguishable from the lowest class.
      float rank = 0.0f;
      if (!spec.node_class.empty()) {
        rank = static_cast<float>(spec.node_class[node] + 1) /
               static_cast<float>(top_class + 1);
      } else if (!spec.successor.empty() || !spec.predecessor.empty()) {
        const bool opens = !spec.successor.empty() && spec.successor[node] >= 0;
        const bool requires_partner =
            !spec.predecessor.empty() && spec.predecessor[node] >= 0;
        // Three positions rather than two, because a chain's middle node both
        // opens an obligation and requires one. Its NET increment is zero --
        // correctly, it adds one and closes one -- so slots 0 and 1 alone
        // cannot tell it from a node in no relation at all. Ordering the
        // positions head < middle < tail keeps this a rank rather than a
        // category, and makes (opens, requires) recoverable from it.
        const int32_t position =
            (opens && requires_partner) ? 2 : (opens ? 1 : 3);
        if (opens || requires_partner)
          rank = static_cast<float>(position) / 3.0f;
      }
      slot[5] = unit(rank);
    }
  }
  incumbent_live_state_.assign(
      static_cast<size_t>(n) * resource_count(), 0.0f);
  incumbent_suffix_state_.assign(
      static_cast<size_t>(n) * resource_count(), 0.0f);
  incumbent_transition_features_.assign(
      static_cast<size_t>(edge_count()) * resource_count() *
          RESOURCE_TRANSITION_FEATURE_COUNT,
      0.0f);
  incumbent_transition_feature_mask_.assign(
      static_cast<size_t>(edge_count()) * resource_count(), 0);
  node_objective_features_.assign(
      static_cast<size_t>(n) * OBJECTIVE_NODE_TERM_COUNT, 0.0f);

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
  for (int32_t node = 0; node < n; ++node) {
    float *features = node_features_.data() +
                      static_cast<size_t>(node) * NODE_FEATURE_COUNT;
    if (!problem_.coordinates.empty()) {
      features[0] =
          unit((problem_.coordinates[2 * node] - min_x) / coordinate_scale);
      features[1] = unit((problem_.coordinates[2 * node + 1] - min_y) /
                         coordinate_scale);
    }
    features[2] = node < problem_.depot_count ? 1.0f : 0.0f;
    // opens_relation / requires_relation used to sit here. A pairwise row now
    // publishes the same information as its node term, so it arrives through
    // node_resource_features under the shared per-resource weights -- and a
    // class-ordered row, which these two columns never described at all, is
    // covered by the same slot.
    float *objective = node_objective_features_.data() +
                       static_cast<size_t>(node) * OBJECTIVE_NODE_TERM_COUNT;
    objective[0] = unit(problem_.prize[node] / prize_scale_);
    objective[1] = unit(problem_.penalty[node] / penalty_scale_);
  }

  const auto scan_route = [&](const std::vector<int32_t> &nodes,
                              int32_t depot) {
    if (nodes.empty())
      return;
    // Only the clock and the running distance are still needed here: the load
    // and open-pair counters they used to feed are written per resource by the
    // incumbent replay and build_incumbent_suffix_state.
    float time = 0.0f;
    float distance = 0.0f;
    int32_t previous = depot >= 0 ? depot : nodes.front();
    for (size_t index = 0; index < nodes.size(); ++index) {
      const int32_t node = nodes[index];
      if (index > 0 || depot >= 0) {
        const float travel = problem_.dist(previous, node);
        distance += travel;
        time = std::max(time + travel, problem_.tw_start[node]);
      }
      float *features = node_features_.data() +
                        static_cast<size_t>(node) * NODE_FEATURE_COUNT;
      features[3] = 1.0f;
      features[4] = nodes.size() == 1
                         ? 0.0f
                         : unit(static_cast<double>(index) /
                                static_cast<double>(nodes.size() - 1));
      features[5] = unit(distance / distance_scale_);
      // Load, clock, slack and open-pair counters used to be written here, one
      // slot per constraint. They are written per resource instead, by the
      // incumbent replay (forward) and build_incumbent_suffix_state (reverse),
      // so a declared row is described the same way a compiled one is.
      const float state_time = time + problem_.service_time[node];
      time = state_time;
      previous = node;
    }
    float backward = depot >= 0 && !problem_.open_route
                         ? problem_.dist(nodes.back(), depot)
                         : 0.0f;
    for (int32_t index = static_cast<int32_t>(nodes.size()) - 1; index >= 0;
         --index) {
      const int32_t node = nodes[index];
      float *features = node_features_.data() +
                        static_cast<size_t>(node) * NODE_FEATURE_COUNT;
      features[6] = unit(backward / distance_scale_);
      // The per-constraint suffix counters that used to be written here are now
      // build_incumbent_suffix_state's job, for every resource at once.
      if (index > 0)
        backward += problem_.dist(nodes[index - 1], node);
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
    const auto store_transition_features = [&](const State &prefix) {
      const int32_t from = prefix.current;
      if (from < 0 || from >= n)
        return;
      for (int32_t edge = edge_offsets_[from]; edge < edge_offsets_[from + 1];
           ++edge) {
        const int32_t next = edge_to_[edge];
        const bool depot = next < problem_.depot_count;
        for (int32_t resource_index = 0; resource_index < resource_count();
             ++resource_index) {
          const ResourceSpec &spec = resource(resource_index);
          if (!spec.active)
            continue;
          const size_t row = static_cast<size_t>(edge) * resource_count() +
                             resource_index;
          incumbent_transition_feature_mask_[row] = 1;
          float *feature = incumbent_transition_features_.data() +
                           row * RESOURCE_TRANSITION_FEATURE_COUNT;

          bool feasible = true;
          float next_value = prefix.resource_state[resource_index];
          float signed_margin = 1.0f;
          if (spec.op == ResourceOperator::PRECEDENCE) {
            const bool predecessor_served = precedence_prerequisites_met(
                spec, next, [&](int32_t node) {
                  return prefix.visited[static_cast<size_t>(node)] != 0;
                });
            feasible = precedence_admits(spec, next, depot, next_value,
                                          predecessor_served);
            next_value = precedence_next_state(spec, next, depot, next_value);
            // A precedence row's bound is zero outstanding obligations at the
            // end of the route -- terminal feasibility is exactly `state <=
            // eps`. So it has a distance to its bound like every other row,
            // and the fraction already discharged is that distance on the unit
            // its relation count defines. Publishing `feasible ? 1 : -1` here
            // instead threw that away and left precedence the only row whose
            // margin is a sign bit, unnormalized at the clamp extremes while
            // every other row sits at a graded slack. NEXT_STATE below already
            // normalizes by the same relation count.
            signed_margin =
                feasible
                    ? 1.0f - std::clamp(next_value /
                                            static_cast<float>(std::max(
                                                spec.relation_count, 1)),
                                        0.0f, 1.0f)
                    : -1.0f;
          } else {
            float since_rest = has_operation(spec, TermOperation::CHECKPOINT)
                                   ? prefix.resource_since_rest[resource_index]
                                   : next_value;
            feasible = extend_declared(
                spec, from, next, prefix.route_depot, false,
                prefix.reset_guard_remaining, resource_index,
                next_value, since_rest, nullptr, nullptr, &signed_margin);
            signed_margin = std::clamp(
                signed_margin /
                    std::max(runtime_resource_scale(resource_index), EPS),
                -1.0f, 1.0f);
          }

          // The declaration supplies the transferable margin, but a compiled
          // executor remains authoritative for its post-state and legality.
          // Usually the two are identical. The override matters precisely on
          // the few semantics outside the declaration language (for example a
          // backhaul capacity reload or a driver break affecting the clock),
          // and preserves the direct signal the v6 named columns carried.
          const FastPath executor = fast_path(spec);
          if (executor != FastPath::NONE) {
            feasible = resource_transition_feasible(
                prefix, next, resource_index, nullptr, false);
            const float travel = problem_.dist(from, next);
            switch (executor) {
            case FastPath::CAPACITY:
              next_value = depot ? depot_reload(prefix)
                                 : opening_load(prefix, next) -
                                       problem_.demand[next];
              break;
            case FastPath::TIME_WINDOW:
              next_value = depot
                               ? 0.0f
                               : std::max(slot(prefix, FieldChannel::TIME_WINDOW) +
                                              transition_break_duration(prefix,
                                                                        next) +
                                              travel,
                                          problem_.tw_start[next]) +
                                     problem_.service_time[next];
              break;
            case FastPath::ROUTE_LIMIT:
              next_value = depot ? 0.0f
                                 : slot(prefix, FieldChannel::ROUTE_LIMIT) +
                                       travel;
              break;
            case FastPath::TOUR_LIMIT:
              next_value = depot ? 0.0f
                                 : slot(prefix, FieldChannel::TOUR_LIMIT) +
                                       travel;
              break;
            case FastPath::BACKHAUL_ORDER:
              next_value = depot
                               ? 0.0f
                               : (slot(prefix, FieldChannel::BACKHAUL_ORDER) >
                                          FEASIBILITY_EPS ||
                                      problem_.demand[next] < -FEASIBILITY_EPS
                                      ? 1.0f
                                      : 0.0f);
              break;
            case FastPath::PICKUP_DELIVERY:
              next_value = slot(prefix, FieldChannel::PICKUP_DELIVERY);
              if (!depot) {
                if (problem_.delivery_of_pickup[next] >= 0)
                  next_value += 1.0f;
                if (problem_.pickup_of_delivery[next] >= 0)
                  next_value -= 1.0f;
              }
              break;
            case FastPath::PRIZE_QUOTA:
              next_value = slot(prefix, FieldChannel::PRIZE_QUOTA) +
                           (depot ? 0.0f : problem_.prize[next]);
              break;
            case FastPath::NONE:
              break;
            }
          }

          float next_state = 0.0f;
          if (spec.op == ResourceOperator::PRECEDENCE) {
            next_state = std::clamp(
                next_value / static_cast<float>(std::max(spec.relation_count, 1)),
                0.0f, 1.0f);
          } else {
            State projected = prefix;
            projected.current = next;
            projected.resource_state[resource_index] = next_value;
            next_state = resource_state_feature(projected, resource_index);
          }
          // Keep the sign tied to the authoritative admissibility result even
          // at the numerical boundary where a tiny tolerance decides legality.
          if (feasible && signed_margin < 0.0f)
            signed_margin = 0.0f;
          if (!feasible && signed_margin >= 0.0f)
            signed_margin = -std::max(FEASIBILITY_EPS, signed_margin);
          feature[static_cast<int32_t>(
              ResourceTransitionFeature::NEXT_STATE)] = next_state;
          feature[static_cast<int32_t>(
              ResourceTransitionFeature::SIGNED_MARGIN)] = signed_margin;
        }
      }
    };
    const auto store_replay = [&](int32_t node) {
      const std::vector<float> live = live_state_features(replay);
      if (node >= 0 && node < n && !live.empty())
        std::copy(live.begin(), live.end(),
                  incumbent_live_state_.begin() +
                      static_cast<size_t>(node) * resource_count());
    };
    store_replay(replay.current);
    store_transition_features(replay);
    for (size_t index = 1; index < incumbent_route_.size(); ++index) {
      std::string error;
      if (!transition(replay, incumbent_route_[index], error))
        break;
      store_replay(replay.current);
      store_transition_features(replay);
    }
  }
  build_incumbent_suffix_state();

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
  // Per-node in/out distance profiles over the candidate rows. On an
  // asymmetric instance the x/y slots are empty (there are no coordinates), so
  // these two carry the only node-level positional signal the model gets, and
  // their difference is the node's own asymmetry. On a Euclidean instance they
  // coincide and vary monotonically with remoteness from the candidate
  // neighbourhood's centre, so the slots stay meaningful rather than dead.
  std::vector<double> out_distance_total(n, 0.0);
  std::vector<double> in_distance_total(n, 0.0);
  std::vector<int32_t> out_degree(n, 0);
  std::vector<int32_t> in_degree(n, 0);
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
      // The seven per-channel pressure slots that sat here were a verbatim copy
      // of the first seven columns of `resources`, which the model already
      // reads per resource. Only the compiled kernels ever had one.
      features[1] = incumbent_edges[edge] ? 1.0f : 0.0f;
      features[2] = reverse_incumbent_edges[edge] ? 1.0f : 0.0f;
      // The reverse leg's cost, which slot 0 cannot carry and which the reverse
      // arc may never deliver: candidate rows are ranked per source, so (j, i)
      // is often absent while (i, j) is present. Always defined -- the metric
      // is queryable in both directions regardless of the candidate set --
      // and it collapses onto slot 0 on symmetric instances.
      features[4] = unit(problem_.dist(to, from) / distance_scale_);
      const double travel = problem_.dist(from, to);
      out_distance_total[from] += travel;
      ++out_degree[from];
      in_distance_total[to] += travel;
      ++in_degree[to];
      // The DECLARED objective's edge term, the counterpart of the node terms
      // in node_objective_features. A waived open-route return leg used to be
      // spelled out here as its own column, which was a hard-coded special case
      // for one route structure; the objective already says the leg is free, so
      // the model reads that rather than a flag. Squashed about 0.5 because the
      // term is signed (a prize objective's edge cost can be negative) and
      // unbounded, and divided by the energy scale so a positive rescale of the
      // coefficients leaves it unchanged -- the same equivariance the token's
      // coefficient encoding keeps.
      const double objective_term =
          objective_edge_cost(from, to) /
          std::max<double>(objective_energy_scale_, EPS);
      features[3] = unit(0.5 + 0.5 * objective_term /
                                   (1.0 + std::abs(objective_term)));
    }
  }
  for (int32_t node = 0; node < n; ++node) {
    float *features = node_features_.data() +
                      static_cast<size_t>(node) * NODE_FEATURE_COUNT;
    features[7] = out_degree[node] > 0
                       ? unit(out_distance_total[node] /
                              static_cast<double>(out_degree[node]) /
                              distance_scale_)
                       : 0.0f;
    features[8] = in_degree[node] > 0
                       ? unit(in_distance_total[node] /
                              static_cast<double>(in_degree[node]) /
                              distance_scale_)
                       : 0.0f;
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
  // The slot accessors index resource_state, so it has to exist before any of
  // them is touched -- including the scratch tail an undeclared compiled
  // channel writes to.
  state.resource_state.assign(state_slot_count(), 0.0f);
  state.resource_since_rest.assign(state_slot_count(), 0.0f);
  state.reset_guard_remaining = initial_reset_guards();
  for (int32_t index = 0; index < resource_count(); ++index) {
    const float initial = resource(index).initial;
    state.resource_state[index] = initial;
    state.resource_since_rest[index] = initial;
  }
  slot(state, FieldChannel::CAPACITY) = problem_.capacity;

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
    consume_reset_guards(state.reset_guard_remaining, start_node);
    slot(state, FieldChannel::PRIZE_QUOTA) = problem_.prize[start_node];
  } else {
    if (start_node < 0 || start_node >= problem_.depot_count) {
      throw std::invalid_argument("a depot problem must start at a depot");
    }
    state.route_depot = start_node;
    state.at_depot = true;
    slot(state, FieldChannel::CAPACITY) = depot_reload(state);
  }
  return state;
}

bool RoutingDecoder::resource_transition_feasible(const State &state,
                                                   int32_t next,
                                                   int32_t resource_index,
                                                   float *next_value,
                                                   bool force_route_end,
                                                   bool *optional_reset_taken,
                                                   float *next_since_rest) const {
  const ResourceSpec &spec = resource(resource_index);
  if (optional_reset_taken != nullptr)
    *optional_reset_taken = false;
  if (!spec.active)
    return true;
  const bool depot = next < problem_.depot_count;
  const bool customer = !depot;
  const float travel = problem_.dist(state.current, next);
  // Dispatch on WHO EXECUTES the row, not on what it is. These branches used to
  // be selected by spec.op, which meant a declared row stating the same algebra
  // could never reach them and a compiled row could never be interpreted.
  switch (fast_path(spec)) {
  case FastPath::CAPACITY: {
    if (!customer)
      return true;
    const float load = opening_load(state, next);
    return load - problem_.demand[next] >= -FEASIBILITY_EPS &&
           load - problem_.demand[next] <= problem_.capacity + FEASIBILITY_EPS;
  }
  case FastPath::TIME_WINDOW: {
    const float break_duration = transition_break_duration(state, next);
    if (force_route_end)
      return slot(state, FieldChannel::TIME_WINDOW) + break_duration + travel <=
             problem_.tw_end[next] + FEASIBILITY_EPS;
    if (!customer)
      return true;
    const float arrival =
        std::max(slot(state, FieldChannel::TIME_WINDOW) + break_duration + travel,
                 problem_.tw_start[next]);
    if (arrival > problem_.tw_end[next] + FEASIBILITY_EPS)
      return false;
    if (problem_.open_route)
      return true;
    State projected = state;
    projected.current = next;
    slot(projected, FieldChannel::TIME_WINDOW) = arrival + problem_.service_time[next];
    for (int32_t index : scalar_resource_indices_) {
      float value = projected.resource_state[index];
      float rest = projected.resource_since_rest[index];
      if (!resource_transition_feasible(state, next, index, &value, false,
                                        nullptr, &rest))
        return false;
      projected.resource_state[index] = value;
      projected.resource_since_rest[index] = rest;
    }
    const float return_break =
        transition_break_duration(projected, state.route_depot);
    return slot(projected, FieldChannel::TIME_WINDOW) + return_break +
               problem_.dist(next, state.route_depot) <=
           problem_.tw_end[state.route_depot] + FEASIBILITY_EPS;
  }
  case FastPath::ROUTE_LIMIT: {
    if (!customer && !force_route_end)
      return true;
    float required = slot(state, FieldChannel::ROUTE_LIMIT) + travel;
    if (customer && !problem_.open_route)
      required += problem_.dist(next, state.route_depot);
    return required <= problem_.route_limit + FEASIBILITY_EPS;
  }
  case FastPath::TOUR_LIMIT:
    if (!customer && !force_route_end)
      return true;
    return slot(state, FieldChannel::TOUR_LIMIT) + travel +
               (customer ? problem_.dist(next, state.route_depot) : 0.0f) <=
           problem_.tour_limit + FEASIBILITY_EPS;
  case FastPath::BACKHAUL_ORDER:
    return !customer || !slot(state, FieldChannel::BACKHAUL_ORDER) ||
           problem_.demand[next] <= FEASIBILITY_EPS;
  case FastPath::PICKUP_DELIVERY:
    if (depot)
      return slot(state, FieldChannel::PICKUP_DELIVERY) == 0;
    return problem_.pickup_of_delivery[next] < 0 ||
           state.visited[problem_.pickup_of_delivery[next]];
  case FastPath::PRIZE_QUOTA:
    return customer ||
           slot(state, FieldChannel::PRIZE_QUOTA) + FEASIBILITY_EPS >= problem_.prize_quota ||
           state.visited_customers >= problem_.customer_count();
  case FastPath::NONE:
    break;
  }

  // Interpreted: read the row's own algebra.
  if (spec.op == ResourceOperator::PRECEDENCE) {
    return precedence_admits(
        spec, next, depot, state.resource_state[resource_index],
        precedence_prerequisites_met(spec, next, [&](int32_t node) {
          return state.visited[static_cast<size_t>(node)] != 0;
        }));
  }

  float value = state.resource_state[resource_index];
  float rest = has_operation(spec, TermOperation::CHECKPOINT)
                   ? state.resource_since_rest[resource_index]
                   : value;
  const bool feasible =
      extend_declared(spec, state.current, next, state.route_depot,
                      force_route_end,
                      state.reset_guard_remaining, resource_index,
                      value, rest, optional_reset_taken);
  if (next_value != nullptr)
    *next_value = value;
  if (next_since_rest != nullptr)
    *next_since_rest = rest;
  return feasible;
}

float RoutingDecoder::transition_break_duration(const State &state,
                                                 int32_t next) const {
  float duration = 0.0f;
  for (int32_t index : scalar_resource_indices_) {
    bool reset_taken = false;
    if (resource_transition_feasible(state, next, index, nullptr, false,
                                     &reset_taken) &&
        reset_taken) {
      duration += resource(index).optional_reset_duration;
    }
  }
  return duration;
}

bool RoutingDecoder::resource_terminal_feasible(const State &state,
                                                 int32_t resource_index) const {
  const ResourceSpec &spec = resource(resource_index);
  if (!spec.active)
    return true;
  if (fast_path(spec) == FastPath::PICKUP_DELIVERY)
    return slot(state, FieldChannel::PICKUP_DELIVERY) == 0;
  if (fast_path(spec) != FastPath::NONE)
    return true;
  if (spec.op == ResourceOperator::PRECEDENCE)
    return spec.relation == PrecedenceRelation::CLASS_ORDER ||
           state.resource_state[resource_index] <= FEASIBILITY_EPS;
  if (spec.bound_check == BoundCheck::SOLUTION_END ||
      spec.bound_check == BoundCheck::ROUTE_END) {
    const float value = state.resource_state[resource_index];
    return value >= spec_lower(spec, state.current) - FEASIBILITY_EPS &&
           value <= spec_upper(spec, state.current) + FEASIBILITY_EPS;
  }
  return true;
}

bool RoutingDecoder::class_ordered() const {
  if (problem_.has(BACKHAUL_ORDER))
    return true;
  for (int32_t index : precedence_resource_indices_) {
    if (resource(index).relation == PrecedenceRelation::CLASS_ORDER)
      return true;
  }
  return false;
}

float RoutingDecoder::opening_load(const State &state, int32_t next) const {
  if (state.at_depot && class_ordered() &&
      next >= problem_.depot_count &&
      problem_.demand[next] < -FEASIBILITY_EPS) {
    return 0.0f;
  }
  return slot(state, FieldChannel::CAPACITY);
}

float RoutingDecoder::depot_reload(const State &state) const {
  if (!problem_.has(CAPACITY)) {
    return problem_.capacity;
  }
  if (state.unvisited_linehauls > 0)
    return problem_.capacity;
  return state.unvisited_backhauls > 0 ? 0.0f : problem_.capacity;
}

float RoutingDecoder::return_horizon_slack(const ResourceSpec &spec,
                                           int32_t next, int32_t depot,
                                           float arrival_value,
                                           const std::vector<int32_t> &remaining,
                                           int32_t resource_index) const {
  // Slack of `spec`'s bound after the depot return leg is appended to the
  // arrival state at `next`. Positive means the bound still holds once the
  // vehicle drives home. Both bounds are projected; an infinite bound
  // contributes infinite slack, so the finite side decides.
  //
  // If `next` itself resets the accumulator on departure (a charger or a
  // rest-eligible node), the return leg begins from the reset value. Arrival
  // resets are already reflected in `arrival_value`; the optional departure
  // reset of a break node is not, so account for it here.
  const auto gate_matches = [&](const ResourceTerm &term) {
    if (term.gate == TermGate::ALWAYS)
      return true;
    const size_t slot = static_cast<size_t>(2 * resource_index);
    const bool alternative = slot + 1 < remaining.size() &&
                             remaining[slot] == 0 &&
                             remaining[slot + 1] > 0;
    return term.gate == TermGate::REMAINDER_ALTERNATIVE ? alternative
                                                        : !alternative;
  };
  float leg = arrival_value;
  for (const ResourceTerm &term : spec.terms) {
    const bool starts_new_leg =
        term.operation == TermOperation::CHECKPOINT ||
        term.operation == TermOperation::ASSIGN;
    if (!starts_new_leg || !gate_matches(term) ||
        !term_event_matches(term, next, next, false, false))
      continue;
    leg = term_value(term, next, next);
  }
  // The whole accumulation over the return edge: edge-sourced terms priced on
  // (next, depot), node-sourced terms charged at the depot, exactly as an
  // ordinary transition would.
  apply_add_terms(spec, TermPhase::BEFORE_BOUND, next, depot, leg);
  for (const ResourceTerm &term : spec.terms) {
    if (term.operation == TermOperation::JOIN &&
        term.phase == TermPhase::BEFORE_BOUND &&
        term.trigger == TermTrigger::ALWAYS)
      leg = join(spec, leg, term_value(term, next, depot));
  }
  return std::min(leg - spec_lower(spec, depot), spec_upper(spec, depot) - leg);
}

int32_t RoutingDecoder::relation_partner(int32_t node,
                                         bool *node_is_predecessor) const {
  const auto answer = [&](int32_t partner, bool predecessor) {
    if (node_is_predecessor != nullptr)
      *node_is_predecessor = predecessor;
    return partner;
  };
  if (relation_successor_.empty())
    return answer(-1, false);
  if (relation_successor_[node] >= 0)
    return answer(relation_successor_[node], true);
  if (relation_predecessor_[node] >= 0)
    return answer(relation_predecessor_[node], false);
  return answer(-1, false);
}

int32_t RoutingDecoder::open_relation_delta(int32_t node) const {
  // Net change in unresolved pairwise obligations when `node` is served. The
  // SRR piece cutter may only split a route where this prefix is zero, so a
  // declared pairwise relation constrains the move set exactly as the compiled
  // pickup-delivery kernel always has.
  int32_t delta = 0;
  if (relation_successor_.empty())
    return delta;
  // The index already excludes solution-scoped relations, which permit their
  // two nodes to sit in different routes and so must not restrict where a route
  // may be cut.
  if (relation_successor_[node] >= 0)
    ++delta;
  if (relation_predecessor_[node] >= 0)
    --delta;
  return delta;
}

int32_t RoutingDecoder::precedence_required(const ResourceSpec &spec,
                                            int32_t next) {
  if (spec.relation != PrecedenceRelation::PAIRWISE || spec.predecessor.empty())
    return -1;
  return spec.predecessor[static_cast<size_t>(next)];
}

bool RoutingDecoder::precedence_admits(const ResourceSpec &spec, int32_t next,
                                       bool depot, float state_value,
                                       bool predecessor_served) {
  const bool route_scoped = spec.scope == ResourceScope::ROUTE;
  switch (spec.relation) {
  case PrecedenceRelation::PAIRWISE:
    // A route-scoped relation must be resolved before the route closes, which
    // is what confines a pair to one route. A solution-scoped one only requires
    // the predecessor to come first somewhere, so the depot always admits and
    // the obligation survives to the terminal check.
    if (depot)
      return !route_scoped || state_value <= FEASIBILITY_EPS;
    return predecessor_served;
  case PrecedenceRelation::CLASS_ORDER:
    // Classes are served in non-decreasing order; a route-scoped row restarts
    // the order at each depot, a solution-scoped one carries it across routes.
    if (depot || spec.node_class.empty())
      return true;
    return static_cast<float>(spec.node_class[static_cast<size_t>(next)]) >=
           state_value - FEASIBILITY_EPS;
  case PrecedenceRelation::DAG:
    // The whole relation is enforced here: a node is admitted only once every
    // predecessor is served, so a route that reaches the end has satisfied it
    // edge by edge. The depot opens and closes routes and takes part in no
    // ordering, so it always admits.
    return depot || predecessor_served;
  }
  return true;
}

float RoutingDecoder::precedence_next_state(const ResourceSpec &spec,
                                            int32_t next, bool depot,
                                            float state_value) {
  const bool route_scoped = spec.scope == ResourceScope::ROUTE;
  switch (spec.relation) {
  case PrecedenceRelation::PAIRWISE: {
    if (depot)
      return route_scoped ? 0.0f : state_value;
    float open = state_value;
    if (!spec.successor.empty() &&
        spec.successor[static_cast<size_t>(next)] >= 0)
      open += 1.0f;
    if (!spec.predecessor.empty() &&
        spec.predecessor[static_cast<size_t>(next)] >= 0)
      open -= 1.0f;
    return open;
  }
  case PrecedenceRelation::CLASS_ORDER:
    if (depot)
      return route_scoped ? 0.0f : state_value;
    return spec.node_class.empty()
               ? state_value
               : std::max(state_value,
                          static_cast<float>(
                              spec.node_class[static_cast<size_t>(next)]));
  case PrecedenceRelation::DAG: {
    // Unresolved obligations remaining: every relation whose head is still
    // unserved. Serving a node discharges exactly its in-degree, so the counter
    // reaches zero once every node has been served -- the same "no obligation
    // outstanding" reading the pairwise counter has, and the same terminal test.
    if (depot || spec.predecessor_offsets.empty())
      return state_value;
    const size_t head = static_cast<size_t>(next);
    const float indegree =
        static_cast<float>(spec.predecessor_offsets[head + 1] -
                           spec.predecessor_offsets[head]);
    return state_value - indegree;
  }
  }
  return state_value;
}

bool RoutingDecoder::extend_declared(const ResourceSpec &spec, int32_t from,
                                     int32_t next, int32_t route_depot,
                                     bool force_route_end,
                                     const std::vector<int32_t> &remaining,
                                     int32_t resource_index,
                                     float &value, float &rest,
                                     bool *break_taken, float *bounded_value,
                                     float *admissibility_margin) const {
  const bool depot = next < problem_.depot_count;
  const auto gate_matches = [&](const ResourceTerm &term) {
    if (term.gate == TermGate::ALWAYS)
      return true;
    const size_t slot = static_cast<size_t>(2 * resource_index);
    const bool alternative = slot + 1 < remaining.size() &&
                             remaining[slot] == 0 &&
                             remaining[slot + 1] > 0;
    return term.gate == TermGate::REMAINDER_ALTERNATIVE ? alternative
                                                        : !alternative;
  };
  double triggered_add = 0.0;
  for (const ResourceTerm &term : spec.terms) {
    if (term.operation != TermOperation::ADD ||
        term.phase != TermPhase::BEFORE_BOUND ||
        term.trigger == TermTrigger::ALWAYS ||
        !term_event_matches(term, from, next, depot, false) ||
        !gate_matches(term))
      continue;
    triggered_add += term_value(term, from, next);
  }
  value += static_cast<float>(triggered_add);
  rest += static_cast<float>(triggered_add);
  const float previous_rest = rest;
  apply_add_terms(spec, TermPhase::BEFORE_BOUND, from, next, value);
  if (!has_operation(spec, TermOperation::CHECKPOINT)) {
    rest = value;
  } else {
    rest = previous_rest;
    apply_add_terms(spec, TermPhase::BEFORE_BOUND, from, next, rest);
  }
  // The row's join against this node's operand -- under max_plus, arrival may
  // not precede the node's floor, i.e. the vehicle waits. Applied before the
  // bound so the bound tests the true arrival, and before the departure term so
  // a service time is charged on top of the wait. Inert for an arithmetic row
  // and for any row that declares no operand.
  for (const ResourceTerm &term : spec.terms) {
    if (term.operation != TermOperation::JOIN ||
        term.phase != TermPhase::BEFORE_BOUND ||
        !term_event_matches(term, from, next, depot, false) ||
        !gate_matches(term))
      continue;
    const float operand = term_value(term, from, next);
    value = join(spec, value, operand);
    rest = join(spec, rest, operand);
  }
  const bool check = spec.bound_check == BoundCheck::TRANSITION ||
                     ((depot || force_route_end) &&
                      spec.bound_check == BoundCheck::ROUTE_END);
  const float lower = spec_lower(spec, next);
  const float upper = spec_upper(spec, next);
  bool feasible = !check || (value >= lower - FEASIBILITY_EPS &&
                             value <= upper + FEASIBILITY_EPS);
  bool taken = false;
  if (!feasible && std::isfinite(upper) && value > upper + FEASIBILITY_EPS) {
    for (const ResourceTerm &term : spec.terms) {
      if (term.operation != TermOperation::RESTORE ||
          !term_event_matches(term, from, next, depot, true) ||
          rest > upper + FEASIBILITY_EPS)
        continue;
      value = rest;
      taken = true;
      feasible = !check || (value >= lower - FEASIBILITY_EPS &&
                            value <= upper + FEASIBILITY_EPS);
      break;
    }
  }
  if (break_taken != nullptr)
    *break_taken = taken;
  // The bounded arrival value, before the departure term and before any reset:
  // this is the quantity the bound was tested against and the one a tightness
  // report should read.
  if (bounded_value != nullptr)
    *bounded_value = value;
  float margin = std::numeric_limits<float>::infinity();
  if (check) {
    if (std::isfinite(lower))
      margin = std::min(margin, value - lower);
    if (std::isfinite(upper))
      margin = std::min(margin, upper - value);
  }
  // Arm optional checkpoints before post-bound additions so service/departure
  // terms are charged after the restored base, exactly like the frozen path.
  for (const ResourceTerm &term : spec.terms) {
    if (term.operation == TermOperation::CHECKPOINT &&
        term_event_matches(term, from, next, depot, false) &&
        gate_matches(term))
      rest = term_value(term, from, next);
  }
  apply_add_terms(spec, TermPhase::AFTER_BOUND, from, next, value);
  apply_add_terms(spec, TermPhase::AFTER_BOUND, from, next, rest);
  // `horizon: return` projects the bound one depot leg further. It is a sound
  // necessary condition for a row that cannot be replenished mid-route, and it
  // is what the compiled route- and tour-limit kernels have always enforced.
  if (feasible && spec.horizon == BoundHorizon::RETURN && !depot &&
      !problem_.open_route && route_depot >= 0 &&
      spec.bound_check != BoundCheck::SOLUTION_END) {
    const float return_margin = return_horizon_slack(
        spec, next, route_depot, value, remaining, resource_index);
    margin = std::min(margin, return_margin);
    feasible = return_margin >= -FEASIBILITY_EPS;
  }
  if (admissibility_margin != nullptr)
    *admissibility_margin = std::isfinite(margin) ? margin : spec.scale;
  for (const ResourceTerm &term : spec.terms) {
    if (term.operation != TermOperation::ASSIGN ||
        !term_event_matches(term, from, next, depot, false) ||
        !gate_matches(term))
      continue;
    value = term_value(term, from, next);
    rest = value;
  }
  return feasible;
}

bool RoutingDecoder::construction_return_reachable(const State &state,
                                                   int32_t next) const {
  // Construction-time stranding guard for rows declaring
  // `horizon: return_construction`. The per-transition bound only certifies
  // that `next` is reachable on arrival, not that any onward move remains -- so
  // greedy construction can drive to a far customer and then reach neither the
  // depot nor another customer. Because every customer is feasible as a depot
  // singleton, requiring the depot to stay reachable after the move keeps a
  // feasible completion available (return, reset, serve the rest) and never
  // blocks legitimate progress. It is deliberately *not* a feasibility
  // condition: a resource that resets at interior nodes may reach the depot
  // through a charger or rest stop, so a closed route that fails this
  // projection can still be valid. Rows needing the sound, always-enforced
  // version declare `horizon: return` instead, which is checked in
  // resource_transition_feasible.
  if (next < problem_.depot_count || problem_.depot_count == 0 ||
      problem_.open_route)
    return true;
  for (int32_t index : active_resource_indices_) {
    const ResourceSpec &spec = resource(index);
    if (spec.horizon != BoundHorizon::RETURN_CONSTRUCTION)
      continue;
    // Post-arrival (post-reset) resource value at `next`.
    float value = state.resource_state[index];
    if (!resource_transition_feasible(state, next, index, &value))
      return false;
    // Cheapest depot return leg keeps the guard least restrictive under
    // multiple depots.
    float best_slack = -std::numeric_limits<float>::infinity();
    for (int32_t depot = 0; depot < problem_.depot_count; ++depot)
      best_slack = std::max(
          best_slack,
          return_horizon_slack(spec, next, depot, value,
                               state.reset_guard_remaining, index));
    if (best_slack < -FEASIBILITY_EPS)
      return false;
  }
  return true;
}

bool RoutingDecoder::legal_node(const State &state, int32_t node) const {
  const int32_t depots = problem_.depot_count;
  if (node < 0 || node >= problem_.node_count)
    return false;
  if (node >= depots) {
    if (state.visited[node])
      return false;
    for (int32_t index : active_resource_indices_) {
      if (!resource_transition_feasible(state, node, index))
        return false;
    }
    return construction_return_reachable(state, node);
  }

  if (depots == 0 || state.at_depot)
    return false;

  bool depot_allowed = problem_.multi_route || !problem_.has(VISIT_ALL);
  for (int32_t index : active_resource_indices_) {
    if (!depot_allowed)
      break;
    if (!resource_transition_feasible(state, node, index))
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
    for (int32_t index : active_resource_indices_) {
      if (!resource_transition_feasible(state, next, index)) {
        error = "resource transition failed: " + resource(index).name;
        return false;
      }
    }
    error = "route contains an infeasible transition to node " +
            std::to_string(next);
    return false;
  }
  if (find_edge(state.current, next) < 0) {
    ++state.off_graph_edges;
  }
  const float break_duration = transition_break_duration(state, next);
  for (int32_t index : scalar_resource_indices_) {
    float value = state.resource_state[index];
    float rest = state.resource_since_rest[index];
    (void)resource_transition_feasible(state, next, index, &value, false,
                                       nullptr, &rest);
    state.resource_state[index] = value;
    state.resource_since_rest[index] = rest;
  }
  for (int32_t index : precedence_resource_indices_) {
    state.resource_state[index] = precedence_next_state(
        resource(index), next, next < problem_.depot_count,
        state.resource_state[index]);
  }
  if (next < problem_.depot_count) {
    if (!problem_.open_route) {
      state.distance += problem_.dist(state.current, state.route_depot);
    }
    state.route.push_back(next);
    state.current = next;
    state.route_depot = next;
    state.at_depot = true;
    slot(state, FieldChannel::BACKHAUL_ORDER) = false;
    slot(state, FieldChannel::ROUTE_LIMIT) = 0.0f;
    slot(state, FieldChannel::TOUR_LIMIT) = 0.0f;
    slot(state, FieldChannel::TIME_WINDOW) = 0.0f;
    slot(state, FieldChannel::CAPACITY) = depot_reload(state);
    return true;
  }

  const float edge = problem_.dist(state.current, next);
  state.distance += edge;
  slot(state, FieldChannel::ROUTE_LIMIT) += edge;
  slot(state, FieldChannel::TOUR_LIMIT) += edge;
  slot(state, FieldChannel::TIME_WINDOW) =
      std::max(slot(state, FieldChannel::TIME_WINDOW) + break_duration + edge,
               problem_.tw_start[next]) +
      problem_.service_time[next];
  slot(state, FieldChannel::CAPACITY) = opening_load(state, next) - problem_.demand[next];
  if (problem_.demand[next] < -FEASIBILITY_EPS) {
    slot(state, FieldChannel::BACKHAUL_ORDER) = true;
  }
  if (problem_.delivery_of_pickup[next] >= 0) {
    slot(state, FieldChannel::PICKUP_DELIVERY) += 1.0f;
  }
  if (problem_.pickup_of_delivery[next] >= 0) {
    slot(state, FieldChannel::PICKUP_DELIVERY) -= 1.0f;
  }
  state.current = next;
  state.at_depot = false;
  state.visited[next] = 1;
  state.unvisited_linehauls -=
      problem_.demand[next] > FEASIBILITY_EPS ? 1 : 0;
  state.unvisited_backhauls -=
      problem_.demand[next] < -FEASIBILITY_EPS ? 1 : 0;
  consume_reset_guards(state.reset_guard_remaining, next);
  ++state.visited_customers;
  slot(state, FieldChannel::PRIZE_QUOTA) += problem_.prize[next];
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
  const int32_t unvisited_linehauls = state.unvisited_linehauls;
  const int32_t unvisited_backhauls = state.unvisited_backhauls;
  const bool at_depot = state.at_depot;
  const float distance = state.distance;
  const int32_t off_graph_edges = state.off_graph_edges;
  // Every compiled quantity lives in resource_state, so restoring the vector
  // restores all of them. Seven individual save/restore pairs used to sit
  // alongside this copy and were already dead -- the vector assignment below
  // ran last and overwrote each one.
  const std::vector<float> resource_state = state.resource_state;
  const std::vector<float> resource_since_rest = state.resource_since_rest;
  const std::vector<int32_t> reset_guard_remaining =
      state.reset_guard_remaining;

  std::string error;
  const bool transitioned = transition(state, next, error);
  const bool feasible = transitioned && has_feasible_lookahead(state, depth);

  state.route.resize(route_size);
  state.visited[next] = was_visited;
  state.current = current;
  state.route_depot = route_depot;
  state.visited_customers = visited_customers;
  state.unvisited_linehauls = unvisited_linehauls;
  state.unvisited_backhauls = unvisited_backhauls;
  state.at_depot = at_depot;
  state.distance = distance;
  state.off_graph_edges = off_graph_edges;
  state.resource_state = resource_state;
  state.resource_since_rest = resource_since_rest;
  state.reset_guard_remaining = reset_guard_remaining;
  return feasible;
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
  if (!problem_.open_route && !state.at_depot) {
    const int32_t end =
        problem_.depot_count == 0 ? state.start_node : state.route_depot;
    for (int32_t index : active_resource_indices_) {
      float value = state.resource_state[index];
      float rest = state.resource_since_rest[index];
      if (!resource_transition_feasible(state, end, index, &value, true,
                                        nullptr, &rest)) {
        solution.error = "closing resource bound failed: " +
                         resource(index).name;
        return solution;
      }
      state.resource_state[index] = value;
      state.resource_since_rest[index] = rest;
    }
  }
  for (int32_t index : active_resource_indices_) {
    if (!resource_terminal_feasible(state, index)) {
      solution.error = "terminal resource bound failed: " +
                       resource(index).name;
      return solution;
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
  solution.collected_prize = slot(state, FieldChannel::PRIZE_QUOTA);
  solution.off_graph_edges = state.off_graph_edges;
  solution.objective = problem_.objective.report(
      solution.distance, solution.collected_prize, solution.missed_penalty);
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
                                    const float *edge_state_field,
                                    const float *multipliers,
                                    const float *coupler_weights,
                                    const float *coupler_bias,
                                    const float *objective_residual,
                                    RolloutTrace *trace, bool greedy) const {
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
  // ranking energy no longer contains a distortion the learned field must
  // fight.
  if (problem_.open_route) {
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
                    static_cast<float>(log_probability), live_state);
    return pool[index].node;
  };
  if (pool.size() == 1) {
    return selected(0, false, 0.0);
  }

  std::vector<double> log_weights(pool.size());
  double maximum = -std::numeric_limits<double>::infinity();
  for (size_t index = 0; index < pool.size(); ++index) {
    const int32_t edge = pool[index].edge;
    const double energy = edge_energy(
        state.current, pool[index].node, edge, edge_field, edge_additive, edge_state_field,
        multipliers, coupler_weights, coupler_bias, live_state.data(),
        objective_residual);
    const double value = -beta_ * energy;
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
                                   const float *edge_state_field,
                                   const float *multipliers,
                                   const float *coupler_weights,
                                   const float *coupler_bias,
                                   const float *objective_residual,
                                   RolloutTrace *trace, bool greedy) const {
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
        state, rng, edge_field, edge_additive, edge_state_field, multipliers,
        coupler_weights, coupler_bias, objective_residual,
        trace, greedy);
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
                                   const float *edge_state_field,
                                   const float *multipliers,
                                   const float *coupler_weights,
                                   const float *coupler_bias,
                                   const float *objective_residual,
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
  const std::vector<float> live_state =
      incumbent_state_features(current);
  for (int32_t edge = edge_offsets_[current]; edge < edge_offsets_[current + 1];
       ++edge) {
    const int32_t node = edge_to_[edge];
    if (node == current || (node >= problem_.depot_count && used[node]) ||
        (node < problem_.depot_count && !problem_.multi_route)) {
      continue;
    }
    const double log_weight =
        -beta_ * edge_energy(current, node, edge, edge_field, edge_additive, edge_state_field,
                             multipliers, coupler_weights, coupler_bias,
                             live_state.data(), objective_residual);
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

bool RoutingDecoder::metric_symmetric() const { return metric_symmetric_; }

float RoutingDecoder::metric_skew() const { return metric_skew_; }

bool RoutingDecoder::relational() const {
  return (active_kernel_capabilities_ & KERNEL_RELATIONAL) != 0;
}

Solution RoutingDecoder::scope_restricted_refine(
    Solution solution, const std::vector<int32_t> &initial_scope,
    const float *edge_field, const float *edge_additive,
    const float *edge_state_field,
    const float *multipliers,
    const float *coupler_weights, const float *coupler_bias,
    const float *objective_residual,
    RolloutTrace *trace, std::mt19937_64 &rng) const {
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
    bool is_predecessor = false;
    const int32_t partner = relation_partner(pair_node, &is_predecessor);
    if (partner < 0)
      return false;
    const int32_t pickup = is_predecessor ? pair_node : partner;
    const int32_t delivery = is_predecessor ? partner : pair_node;
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
    double objective_residual = 0.0;
    std::vector<double> resource;
    // Per (resource, live-state feature) sums of the state-conditioned field.
    // The anchor's live state multiplies these at QUERY time, which is what
    // keeps the guided energy anchor-varying while the sums themselves stay
    // edge-additive -- the property the prefix sums below depend on.
    std::vector<double> resource_state;
    explicit GuidanceValue(int32_t resource_count = 0, int32_t state_count = 0)
        : resource(resource_count, 0.0),
          resource_state(
              static_cast<size_t>(resource_count) * state_count, 0.0) {}
  };
  struct GuidedSequence : GuidanceValue {
    bool empty = true;
    int32_t first = -1;
    int32_t last = -1;
  };
  const auto add_guidance = [&](GuidanceValue lhs, const GuidanceValue &rhs) {
    lhs.objective += rhs.objective;
    lhs.objective_residual += rhs.objective_residual;
    lhs.resource.resize(resource_count(), 0.0);
    for (int32_t channel = 0; channel < resource_count(); ++channel)
      lhs.resource[channel] += channel < static_cast<int32_t>(rhs.resource.size())
                                   ? rhs.resource[channel]
                                   : 0.0;
    const size_t state_slots =
        static_cast<size_t>(resource_count()) * live_state_feature_count();
    lhs.resource_state.resize(state_slots, 0.0);
    for (size_t slot = 0; slot < state_slots; ++slot)
      lhs.resource_state[slot] +=
          slot < rhs.resource_state.size() ? rhs.resource_state[slot] : 0.0;
    return lhs;
  };
  const auto subtract_guidance = [&](GuidanceValue lhs,
                                    const GuidanceValue &rhs) {
    lhs.objective -= rhs.objective;
    lhs.objective_residual -= rhs.objective_residual;
    lhs.resource.resize(resource_count(), 0.0);
    for (int32_t channel = 0; channel < resource_count(); ++channel)
      lhs.resource[channel] -= channel < static_cast<int32_t>(rhs.resource.size())
                                   ? rhs.resource[channel]
                                   : 0.0;
    const size_t state_slots =
        static_cast<size_t>(resource_count()) * live_state_feature_count();
    lhs.resource_state.resize(state_slots, 0.0);
    for (size_t slot = 0; slot < state_slots; ++slot)
      lhs.resource_state[slot] -=
          slot < rhs.resource_state.size() ? rhs.resource_state[slot] : 0.0;
    return lhs;
  };
  // Both halves of the learned resource energy are accumulated here WITHOUT a
  // live state, so these values stay pure functions of the edge and the prefix
  // sums built from them remain exactly additive. The graph-level coupler is
  // applied to `resource` at query time, and the anchor's live state contracts
  // `resource_state` at query time, so the guided energy is anchor-varying
  // without the aggregate ever having to know which anchor it will be read at.
  const auto edge_guidance = [&](int32_t from, int32_t to) {
    GuidanceValue value(resource_count(), live_state_feature_count());
    value.objective = objective_edge_cost(from, to);
    const int32_t edge = find_edge(from, to);
    value.objective_residual =
        edge >= 0 && objective_residual != nullptr ? objective_residual[edge] : 0.0;
    const int32_t states = live_state_feature_count();
    for (int32_t channel = 0; channel < resource_count(); ++channel) {
      if (!resource(channel).active)
        continue;
      value.resource[channel] = resource_field_value(
          from, to, edge, channel, edge_field, edge_additive);
      if (edge < 0 || edge_state_field == nullptr)
        continue;
      const float *row =
          edge_state_field +
          (static_cast<size_t>(edge) * resource_count() + channel) * states;
      for (int32_t feature = 0; feature < states; ++feature) {
        value.resource_state[static_cast<size_t>(channel) * states + feature] =
            row[feature];
      }
    }
    return value;
  };
  const auto guidance_energy = [&](const GuidanceValue &value,
                                   const float *live_state) {
    double result = coupled_multiplier(
                        objective_multiplier(), multipliers, coupler_weights,
                        coupler_bias, live_state) *
                    (value.objective / objective_energy_scale_ +
                     value.objective_residual);
    const int32_t states = live_state_feature_count();
    for (int32_t channel = 0; channel < resource_count(); ++channel) {
      if (!resource(channel).active)
        continue;
      result += coupled_multiplier(channel, multipliers, coupler_weights,
                                   coupler_bias, live_state) *
                value.resource[channel];
      if (live_state == nullptr)
        continue;
      const double base = multipliers == nullptr ? 1.0 : multipliers[channel];
      double interaction = 0.0;
      for (int32_t feature = 0; feature < states; ++feature) {
        interaction +=
            value.resource_state[static_cast<size_t>(channel) * states +
                                 feature] *
            live_state[feature];
      }
      result += base * interaction;
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
    double pickup_binding = 0.0;
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
  const double pickup_scale = std::max(
      static_cast<double>(std::count_if(
          problem_.delivery_of_pickup.begin() + problem_.depot_count,
          problem_.delivery_of_pickup.end(),
          [](int32_t delivery) { return delivery >= 0; })),
      1.0);
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
    result.capacity_binding =
        std::max(sequence.positive_load, sequence.negative_load) /
        capacity_scale;

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
    result.exact =
        !problem_.has(BACKHAUL_ORDER) || !sequence.backhaul_violation;
    return result;
  };

  enum RouteRank : int32_t {
    CAPACITY_EXCESS_RANK = 0,
    CAPACITY_BINDING_RANK,
    TIME_BINDING_RANK,
    ROUTE_RATIO_RANK,
    TOUR_RATIO_RANK,
    PICKUP_BINDING_RANK,
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
    route.forward_guidance.assign(nodes.size(), GuidanceValue(resource_count(), live_state_feature_count()));
    route.reverse_guidance.assign(nodes.size(), GuidanceValue(resource_count(), live_state_feature_count()));
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
    total_guidance = GuidanceValue(resource_count(), live_state_feature_count());
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
        if (local > 0)
          open += open_relation_delta(node);
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
          open += open_relation_delta(node);
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
      if (problem_.has(PICKUP_DELIVERY)) {
        const int32_t max_open = *std::max_element(
            route.open_pickups.begin(), route.open_pickups.end());
        route.resources.pickup_binding = max_open / pickup_scale;
        ranked_routes[PICKUP_BINDING_RANK].push(
            {route.resources.pickup_binding, route_id, version});
      }
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
  std::vector<int32_t> pickup_seen(problem_.node_count, 0);
  int32_t pickup_epoch = 0;
  struct PickupMetrics {
    int32_t violations = 0;
    int32_t max_open = 0;
  };
  const auto planned_pickup_metrics = [&](const PlannedRoute &plan) {
    PickupMetrics result;
    if (!relational())
      return result;
    if (pickup_epoch == std::numeric_limits<int32_t>::max()) {
      std::fill(pickup_seen.begin(), pickup_seen.end(), 0);
      pickup_epoch = 1;
    } else {
      ++pickup_epoch;
    }
    int32_t open = 0;
    bool first = true;
    const auto visit = [&](int32_t node) {
      bool is_predecessor = false;
      const int32_t partner = relation_partner(node, &is_predecessor);
      const int32_t delivery = is_predecessor ? partner : -1;
      const int32_t pickup = is_predecessor ? -1 : partner;
      // Depot-free evaluation treats the first route token as the already
      // visited start node: its pickup identity is visible to later deliveries,
      // but it does not itself change the open-pair counters.
      if (first && plan.depot < 0) {
        if (delivery >= 0)
          pickup_seen[node] = pickup_epoch;
        first = false;
        return;
      }
      first = false;
      if (delivery >= 0) {
        pickup_seen[node] = pickup_epoch;
        ++open;
      }
      if (pickup >= 0) {
        if (pickup_seen[pickup] != pickup_epoch)
          ++result.violations;
        if (open > 0)
          --open;
      }
      result.max_open = std::max(result.max_open, open);
    };
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
    result.violations += open;
    return result;
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
    if (!relational() || piece.singleton >= 0)
      return true;
    const CachedRoute &route = cached_routes[piece.route];
    return route.open_pickups[piece.begin] == 0 &&
           route.open_pickups[piece.end] == 0;
  };
  rebuild_cache();
  rebuild_resource_cache();
  static constexpr size_t MAX_SCREENING_LABELS = 512;
  // Planned summaries exactly certify capacity, time-window, route/tour-limit,
  // prize, and depot assignment for customer-only route edits. Custom algebra
  // rows are replayed below without rebuilding the compiled evaluator state.
  const bool has_negative_demand = std::any_of(
      problem_.demand.begin() + problem_.depot_count,
      problem_.demand.end(),
      [](float demand) { return demand < -FEASIBILITY_EPS; });
  const bool needs_load_partition_certificate =
      problem_.has(CAPACITY) && has_negative_demand;
  const bool planned_commit_certificate = true;
  std::vector<int32_t> structure_seen(problem_.node_count, 0);
  int32_t structure_epoch = 0;
  // Trial-local record of what a candidate route has already served, so a
  // declared precedence row can be certified without allocating per trial.
  std::vector<int32_t> precedence_seen(
      precedence_resource_indices_.empty() ? 0 : problem_.node_count, 0);
  int32_t precedence_epoch = 0;
  const auto route_structure_certificate =
      [&](const std::vector<int32_t> &trial) {
        if (trial.empty())
          return false;
        if (structure_epoch == std::numeric_limits<int32_t>::max()) {
          std::fill(structure_seen.begin(), structure_seen.end(), 0);
          structure_epoch = 1;
        } else {
          ++structure_epoch;
        }
        int32_t seen_customers = 0;
        if (problem_.depot_count > 0) {
          if (trial.front() < 0 || trial.front() >= problem_.depot_count)
            return false;
          bool at_depot = true;
          for (size_t index = 1; index < trial.size(); ++index) {
            const int32_t node = trial[index];
            if (node < 0 || node >= problem_.node_count)
              return false;
            if (node < problem_.depot_count) {
              if (at_depot)
                return false;
              at_depot = true;
              continue;
            }
            if (structure_seen[node] == structure_epoch)
              return false;
            structure_seen[node] = structure_epoch;
            ++seen_customers;
            at_depot = false;
          }
          if (problem_.multi_route && !at_depot)
            return false;
        } else {
          for (int32_t node : trial) {
            if (node < 0 || node >= problem_.node_count ||
                structure_seen[node] == structure_epoch)
              return false;
            structure_seen[node] = structure_epoch;
            ++seen_customers;
          }
        }
        if (problem_.has(VISIT_ALL) &&
            seen_customers != problem_.customer_count())
          return false;
        return true;
      };
  const auto runtime_resource_certificate =
      [&](const std::vector<int32_t> &trial) {
        if (declared_resource_indices_.empty())
          return true;
        if (!precedence_resource_indices_.empty()) {
          if (precedence_epoch == std::numeric_limits<int32_t>::max()) {
            std::fill(precedence_seen.begin(), precedence_seen.end(), 0);
            precedence_epoch = 1;
          } else {
            ++precedence_epoch;
          }
          precedence_seen[trial.front()] = precedence_epoch;
        }
        std::vector<float> algebra(resource_count(), 0.0f);
        std::vector<float> algebra_rest(resource_count(), 0.0f);
        // A guarded reset reads what the route still has left to serve, so
        // this replay has to track it too; the construction path keeps the
        // same count in State.
        std::vector<int32_t> algebra_guards = initial_reset_guards();
        for (int32_t index : declared_resource_indices_) {
          algebra[index] = resource(index).initial;
          algebra_rest[index] = resource(index).initial;
        }
        int32_t current = trial.front();
        int32_t route_depot = problem_.depot_count > 0 ? current : -1;
        bool at_depot = problem_.depot_count > 0;
        const auto extend = [&](int32_t next, bool force_route_end) {
          const bool depot = next < problem_.depot_count;
          for (int32_t index : declared_resource_indices_) {
            const ResourceSpec &spec = resource(index);
            float value = algebra[index];
            float rest = algebra_rest[index];
            if (spec.op == ResourceOperator::PRECEDENCE) {
              const bool served = precedence_prerequisites_met(
                  spec, next, [&](int32_t node) {
                    return precedence_seen[static_cast<size_t>(node)] ==
                           precedence_epoch;
                  });
              if (!precedence_admits(spec, next, depot, value, served))
                return false;
              algebra[index] = precedence_next_state(spec, next, depot, value);
              continue;
            }
            if (!extend_declared(spec, current, next, route_depot,
                                 force_route_end,
                                 algebra_guards, index,
                                 value, rest))
              return false;
            algebra[index] = value;
            algebra_rest[index] = rest;
          }
          consume_reset_guards(algebra_guards, next);
          if (!precedence_resource_indices_.empty())
            precedence_seen[static_cast<size_t>(next)] = precedence_epoch;
          current = next;
          if (depot) {
            route_depot = next;
            at_depot = true;
          } else {
            at_depot = false;
          }
          return true;
        };
        for (size_t index = 1; index < trial.size(); ++index) {
          if (!extend(trial[index], false))
            return false;
        }
        if (problem_.has(VISIT_ALL) && !problem_.multi_route && !at_depot &&
            !problem_.open_route) {
          const int32_t end =
              problem_.depot_count == 0 ? trial.front() : route_depot;
          if (!extend(end, true))
            return false;
        }
        for (int32_t index : precedence_resource_indices_) {
          if (resource(index).relation == PrecedenceRelation::PAIRWISE &&
              algebra[index] > FEASIBILITY_EPS)
            return false;
        }
        for (int32_t index : declared_resource_indices_) {
          const ResourceSpec &spec = resource(index);
          // Resolve the bound at the node the state ends on: a per-node bound
          // must agree with resource_terminal_feasible, which construction uses.
          if (spec.bound_check == BoundCheck::SOLUTION_END &&
              (algebra[index] < spec_lower(spec, current) - FEASIBILITY_EPS ||
               algebra[index] > spec_upper(spec, current) + FEASIBILITY_EPS))
            return false;
        }
        return true;
      };
  const auto load_partition_certificate = [&]
      (const std::vector<PlannedRoute> &plans,
       const std::vector<SequenceSummary> &sequences) {
        if (!needs_load_partition_certificate)
          return true;
        if (plans.size() != sequences.size())
          return false;
        bool backhaul_only_suffix = false;
        int32_t visited_routes = 0;
        for (int32_t route = route_head; route >= 0;
             route = cached_routes[route].next) {
          if (route >= static_cast<int32_t>(cached_routes.size()) ||
              !cached_routes[route].active)
            return false;
          const SequenceSummary *sequence = &cached_routes[route].summary;
          for (size_t index = 0; index < plans.size(); ++index) {
            if (plans[index].slot == route) {
              sequence = &sequences[index];
              break;
            }
          }
          if (sequence->empty)
            return false;
          // The evaluator reloads capacity while any positive-demand customer
          // remains globally. Therefore negative-only routes form one suffix;
          // inside that suffix they start empty, and before it routes start at
          // capacity exactly as route_resource_metrics() assumes.
          if (backhaul_only_suffix && sequence->has_linehaul)
            return false;
          if (sequence->has_backhaul && !sequence->has_linehaul)
            backhaul_only_suffix = true;
          ++visited_routes;
          if (visited_routes > static_cast<int32_t>(cached_routes.size()))
            return false;
        }
        return visited_routes > 0;
      };
  ResourceEvaluation current_resource;
  if (trace != nullptr)
    current_resource = evaluate_resources(solution.route);
  // One label per registry row. This was a FIELD_CHANNEL_COUNT array while
  // ResourceEvaluation was already registry-wide, so the label silently
  // truncated to the seven compiled channels: an appended row got no resource
  // supervision at all, and the width disagreed with the field head's, which is
  // what made the auxiliary resource loss unable to consume a declared row.
  const auto screening_delta = [&](const ResourceEvaluation &candidate) {
    std::vector<float> delta(static_cast<size_t>(resource_count()), 0.0f);
    for (int32_t index = 0; index < resource_count(); ++index) {
      delta[index] = std::clamp(
          candidate.binding[index] - current_resource.binding[index] +
              candidate.violation[index],
          0.0f, 1.0f);
    }
    return delta;
  };
  const auto append_screening_edge = [&]
      (int32_t from, int32_t to, const std::vector<float> &delta) {
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
    if (!load_partition_certificate(plans, sequences))
      return std::nullopt;
    // Time-window and pickup-delivery labels scan only the affected planned
    // route pieces, preserving summed lateness and pair-identity semantics.
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
    double max_pickup_binding =
        problem_.has(PICKUP_DELIVERY)
            ? unaffected_rank(PICKUP_BINDING_RANK, affected, affected_count,
                              0.0)
            : 0.0;
    int32_t pickup_violations = 0;

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
      if (problem_.has(PICKUP_DELIVERY)) {
        const PickupMetrics pickup = planned_pickup_metrics(plan);
        pickup_violations += pickup.violations;
        max_pickup_binding = std::max(
            max_pickup_binding,
            static_cast<double>(pickup.max_open) / pickup_scale);
      }
      route_excess += resource.route_excess;
      tour_excess += resource.tour_excess;
      time_warp += resource.time_warp;
      prize += resource.prize;
      backhauls += resource.backhaul_count;
    }

    ResourceEvaluation result;
    // Registry-indexed, matching evaluate_resources and the field head. A row a
    // compiled kernel does not back has no entry to write here, which is why
    // this fast path abstains for such a row rather than reporting a zero it
    // never checked (see planned_screening_covers_registry_ above).
    result.violation.assign(resource_count(), 0.0f);
    result.binding.assign(resource_count(), 0.0f);
    const auto write = [&](FieldChannel channel, double violation,
                           double binding) {
      const int32_t index = field_resource_index(channel);
      if (index < 0)
        return;
      result.violation[index] = static_cast<float>(violation);
      result.binding[index] = static_cast<float>(std::clamp(binding, 0.0, 1.0));
    };
    write(FieldChannel::CAPACITY, capacity_excess / capacity_scale,
          capacity_binding);
    write(FieldChannel::TIME_WINDOW, time_warp / time_scale, time_binding);
    write(FieldChannel::ROUTE_LIMIT, route_excess / route_scale,
          max_route_ratio);
    write(FieldChannel::TOUR_LIMIT, tour_excess / tour_scale, max_tour_ratio);
    write(FieldChannel::BACKHAUL_ORDER, 0.0, backhauls > 0 ? 1.0 : 0.0);
    write(FieldChannel::PICKUP_DELIVERY, pickup_violations / pickup_scale,
          max_pickup_binding);
    write(FieldChannel::PRIZE_QUOTA,
          std::max(problem_.prize_quota - prize, 0.0) / quota_scale,
          problem_.has(PRIZE_QUOTA) ? prize / quota_scale : 0.0);
    for (int32_t index = 0; index < resource_count(); ++index) {
      if (!resource(index).active) {
        result.violation[index] = 0.0f;
        result.binding[index] = 0.0f;
      } else if (result.violation[index] > FEASIBILITY_EPS) {
        result.binding[index] = 1.0f;
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
    route.guidance = GuidanceValue(resource_count(), live_state_feature_count());
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
    for (int32_t local = 0;
         local < static_cast<int32_t>(route.sequence.nodes.size()); ++local) {
      const int32_t node = route.sequence.nodes[local];
      if (route.depot >= 0 || local > 0)
        open += open_relation_delta(node);
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
    if (problem_.has(PICKUP_DELIVERY)) {
      const int32_t max_open =
          *std::max_element(route.open_pickups.begin(),
                            route.open_pickups.end());
      route.resources.pickup_binding = max_open / pickup_scale;
    }
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
    case PICKUP_BINDING_RANK:
      return resource.pickup_binding;
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
        if (rank == PICKUP_BINDING_RANK && !relational())
          continue;
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
        if (rank == PICKUP_BINDING_RANK && !relational())
          continue;
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
    // Sized from the rank array itself, not from ROUTE_RANK_COUNT. The
    // verification loop at the end of this block iterates ranked_routes, so the
    // two must agree by construction; while they were written independently a
    // single added rank wrote past the end of this array and smashed the stack.
    std::vector<double> incremental_rank_max(ranked_routes.size(), 0.0);
    const std::array<int32_t, 2> no_affected{-1, -1};
    for (size_t rank = 0; rank < ranked_routes.size(); ++rank) {
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
          !close(lhs.objective_residual, rhs.objective_residual)) {
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
  std::vector<double> ranking_energy(edge_count());
  for (int32_t node = 0; node < problem_.node_count; ++node) {
    const auto state = incumbent_state_features(node);
    for (int32_t edge = edge_offsets_[node];
         edge < edge_offsets_[node + 1]; ++edge) {
      ranked_local_edges[edge] = edge;
      ranking_energy[edge] = edge_energy(
          node, edge_to_[edge], edge, edge_field, edge_additive, edge_state_field,
          multipliers, coupler_weights, coupler_bias, state.data(),
          objective_residual);
    }
    std::stable_sort(
        ranked_local_edges.begin() + edge_offsets_[node],
        ranked_local_edges.begin() + edge_offsets_[node + 1],
        [&](int32_t lhs, int32_t rhs) {
          if (ranking_energy[lhs] != ranking_energy[rhs])
            return ranking_energy[lhs] < ranking_energy[rhs];
          return lhs < rhs;
        });
  }

  int32_t moves = 0;
  int32_t full_evaluations = 0;
  int32_t certified_evaluations = 0;
  int32_t incremental_rebuilds = 0;
  int32_t full_rebuilds = 0;
  int64_t rebuilt_nodes = 0;
  // Bounded objective-worsening exploration and the champion (best objective
  // ever seen) that is returned so an uphill excursion never degrades the
  // result. Learned mode requires guided-energy descent; the explicit baseline
  // control selects a feasible non-improving move by seeded random priority.
  int32_t exploration_remaining = search_config_.srr_exploration_budget;
  std::uniform_real_distribution<double> random_escape_priority(0.0, 1.0);
  Solution champion = solution;
  while (!checklist.empty()) {
    const int32_t anchor = checklist.front();
    checklist.pop_front();
    in_queue[anchor] = 0;
    ++visits[anchor];
    if (anchor < problem_.depot_count || node_route[anchor] < 0) {
      continue;
    }
    Solution best_move;
    AcceptedPlan best_plan;
    StructuralMove best_structural;
    const GuidanceValue current_guidance = total_guidance;
    const std::vector<float> anchor_state =
        incumbent_state_features(anchor);
    double best_guided_energy = std::numeric_limits<double>::infinity();
    // Exploration slot for this anchor: either the lowest guided-energy move
    // that strictly lowers anchor energy or the random control's lowest-priority
    // draw. It is committed only when no improving move exists and budget remains.
    const double current_anchor_energy =
        guidance_energy(current_guidance, anchor_state.data());
    Solution best_explore_move;
    AcceptedPlan best_explore_plan;
    double best_explore_energy = std::numeric_limits<double>::infinity();
    double best_explore_priority = std::numeric_limits<double>::infinity();
    const auto consider = [&](const std::vector<int32_t> &trial,
                              StructuralMove structural) {
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
          if (plans.empty())
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
          // Screening labels need resource values immediately, but the commit
          // certificate does not: defer its exact stateful work until the
          // cheap objective and guidance gates identify a promising move.
          const bool needs_screening_resource =
              trace != nullptr &&
              (trace->screened_edges.size() < MAX_SCREENING_LABELS ||
               search_config_.verify_screening_resources);
          std::optional<ResourceEvaluation> planned_resource =
              needs_screening_resource
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
                for (int32_t channel = 0; channel < resource_count();
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
              if (planned_resource.has_value() &&
                  planned_screening_covers_registry_) {
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
          scored.objective = problem_.objective.report(
              scored.distance, scored.collected_prize, scored.missed_penalty);
          const double planned_energy =
              guidance_energy(guided, anchor_state.data());
          const bool random_escape_enabled =
              exploration_remaining > 0 && search_config_.random_escape;
          const double planned_escape_priority = random_escape_enabled
                                                     ? random_escape_priority(rng)
                                                     : 0.0;
          // The guided energy is anchor-specific rather than a global
          // potential: both the graph-level coupler and the per-edge
          // state-conditioned field are evaluated at THIS anchor's live state.
          // Keep the global objective as the monotone acceptance gate and use
          // energy to select among improving moves. A bounded exploration
          // budget may admit energy-descending uphill escapes.
          const bool improving_gate =
              better(scored, solution) &&
              planned_energy < best_guided_energy - 1.0e-12;
          const bool explore_gate =
              exploration_remaining > 0 &&
              (random_escape_enabled
                   ? planned_escape_priority < best_explore_priority
                   : planned_energy < current_anchor_energy -
                                          search_config_.srr_exploration_margin &&
                         planned_energy < best_explore_energy);
          if (!improving_gate && !explore_gate)
            return;
          std::vector<int32_t> trial;
          if (!materialize(trial))
            return;
          if (planned_commit_certificate && !planned_resource.has_value()) {
            planned_resource = evaluate_planned_resources(
                plans, planned_sequences, affected_routes);
          }
          bool planned_resources_feasible = planned_resource.has_value();
          if (planned_resources_feasible) {
            // violation is registry-indexed; a compiled channel reaches its own
            // row through field_resource_index rather than by sharing its
            // ordinal with it.
            for (int32_t channel = 0; channel < FIELD_CHANNEL_COUNT;
                 ++channel) {
              const int32_t index = field_resource_index(
                  static_cast<FieldChannel>(channel));
              if (index < 0)
                continue;
              if (planned_resource->violation[index] >
                  FEASIBILITY_EPS /
                      std::max(runtime_resource_scale(index), EPS)) {
                planned_resources_feasible = false;
                break;
              }
            }
          }
          const bool certificate_feasible =
              planned_commit_certificate && planned_resource.has_value() &&
              planned_resource->structurally_valid &&
              route_structure_certificate(trial) &&
              runtime_resource_certificate(trial) &&
              load_partition_certificate(plans, planned_sequences) &&
              planned_resources_feasible;
          Solution candidate;
          if (certificate_feasible) {
            ++certified_evaluations;
            // `scored` was produced from the same exact sequence summaries;
            // attach the materialized route for cache replacement without
            // replaying every transition a second time.
            candidate = std::move(scored);
            candidate.route = std::move(trial);
            candidate.raw_objective = candidate.objective;
          } else {
            ++full_evaluations;
            candidate = evaluate(trial);
          }
          if (planned_resource.has_value() &&
              planned_screening_covers_registry_) {
            record_planned_screening(plans, *planned_resource);
          } else {
            // A row the planned summaries do not describe would be labelled
            // "unstressed" here purely because nothing looked at it. Replay the
            // route instead: as a certificate the planned evaluation is still
            // sound (runtime_resource_certificate checks the rows it omits),
            // but as supervision a silent zero is a wrong label, not a missing
            // one.
            record_screening(candidate);
          }
          if (!candidate.feasible)
            return;
          if (better(candidate, solution) &&
              planned_energy < best_guided_energy - 1.0e-12) {
            best_move = std::move(candidate);
            best_guided_energy = planned_energy;
            best_plan.valid = true;
            best_plan.plans = plans;
            best_plan.resources = planned_resource;
            best_structural = {};
          } else if (
              exploration_remaining > 0 && !better(candidate, solution) &&
              (search_config_.random_escape
                   ? planned_escape_priority < best_explore_priority
                   : planned_energy < current_anchor_energy -
                                          search_config_.srr_exploration_margin &&
                         planned_energy < best_explore_energy)) {
            // No strictly-improving move at this anchor: record either the
            // field's lowest-energy move or the random control's selected
            // feasible move as a bounded uphill escape.
            best_explore_move = std::move(candidate);
            best_explore_energy = planned_energy;
            best_explore_priority = planned_escape_priority;
            best_explore_plan.valid = true;
            best_explore_plan.plans = plans;
            best_explore_plan.resources = planned_resource;
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
      if (relational() &&
          (relation_partner(lhs) >= 0 || relation_partner(rhs) >= 0)) {
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
      if (relational() &&
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
      if (relational() && relation_partner(node) >= 0) {
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
      bool is_predecessor = false;
      const int32_t partner = relation_partner(pair_node, &is_predecessor);
      if (partner < 0)
        return;
      const int32_t pickup = is_predecessor ? pair_node : partner;
      const int32_t delivery = is_predecessor ? partner : pair_node;
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
      if (problem_.multi_route) {
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
        if (route_local + 1 == route_size &&
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
      for (int32_t rank = edge_offsets_[anchor];
           rank < edge_offsets_[anchor + 1]; ++rank) {
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
            consider_insert_plan(candidate_node, anchor, true);
            consider_replace_plan(anchor, candidate_node);
          }
          continue;
        }
        ++served_candidate_rank;
        consider_two_opt_plan(anchor, candidate_node);
        consider_two_opt_star_plan(anchor, candidate_node);
        consider_swap_plan(anchor, candidate_node);
        for (int32_t length = 1;
             length <= search_config_.or_opt_max_segment; ++length) {
          consider_relocate_plan(candidate_node, anchor, length, false);
          if (candidate_rank <= SRR_DIRECTED_CANDIDATES)
            consider_relocate_plan(candidate_node, anchor, length, true);
        }
        consider_relocate_plan(anchor, candidate_node, 1, false);
        if (candidate_rank <= SRR_DIRECTED_CANDIDATES)
          consider_relocate_plan(anchor, candidate_node, 1, true);
        if (problem_.multi_route &&
            node_route[anchor] != node_route[candidate_node] &&
            served_candidate_rank <= SRR_STRING_CANDIDATES) {
          for (int32_t length = 2;
               length <= search_config_.or_opt_max_segment; ++length) {
            consider_exchange_plan(anchor, 1, candidate_node, length);
            consider_exchange_plan(anchor, length, candidate_node, 1);
            consider_exchange_plan(anchor, length, candidate_node, length);
          }
        }
        if (relational()) {
          consider_relocate_pair_plan(anchor, candidate_node);
          consider_relocate_pair_plan(candidate_node, anchor);
        }
      }
    // Prefer a strictly-improving move; otherwise, if the field found a bounded
    // guided-energy-descending escape, commit that and spend one unit of budget.
    const bool commit_improving = best_move.feasible;
    const bool commit_explore = !commit_improving && exploration_remaining > 0 &&
                                best_explore_plan.valid;
    if (commit_improving || commit_explore) {
      AcceptedPlan &commit_plan = commit_improving ? best_plan : best_explore_plan;
      std::vector<int32_t> old_route;
      if (commit_explore) {
        solution = std::move(best_explore_move);
        --exploration_remaining;
      } else {
        if (!best_plan.valid && !best_structural.valid())
          old_route = solution.route;
        solution = std::move(best_move);
      }
      ++moves;
      std::vector<int32_t> touched;
      // An exploration move is always a planned move (only consider_plans records
      // the escape slot), so it takes the incremental planned-route path.
      if (commit_explore || best_plan.valid) {
        if (!replace_planned_routes(commit_plan.plans, touched, rebuilt_nodes)) {
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
        current_resource = commit_plan.resources.has_value()
                               ? *commit_plan.resources
                               : evaluate_resources(solution.route);
      }
      for (int32_t node : touched) {
        enqueue(node);
        const int32_t route_slot = node_route[node];
        const int32_t local = node_local[node];
        if (route_slot < 0 || local < 0)
          continue;
        const std::vector<int32_t> &nodes =
            cached_routes[route_slot].sequence.nodes;
        if (local > 0)
          enqueue(nodes[local - 1]);
        if (local + 1 < static_cast<int32_t>(nodes.size()))
          enqueue(nodes[local + 1]);
        if (problem_.depot_count == 0 && !problem_.open_route &&
            !nodes.empty()) {
          const int32_t previous =
              nodes[(local + nodes.size() - 1) % nodes.size()];
          const int32_t next = nodes[(local + 1) % nodes.size()];
          enqueue(previous);
          enqueue(next);
        }
      }
      enqueue(anchor);
      // Track the best objective ever seen so an uphill exploration excursion
      // can never degrade the returned solution.
      if (better(solution, champion))
        champion = solution;
    }
  }
  // Return the champion (best objective seen); with exploration disabled it is
  // always the final monotone-descent solution, so behaviour is unchanged.
  Solution result =
      better(champion, solution) ? std::move(champion) : std::move(solution);
  result.srr_moves = moves;
  result.srr_evaluations = full_evaluations;
  result.srr_certified_evaluations = certified_evaluations;
  result.srr_incremental_rebuilds = incremental_rebuilds;
  result.srr_full_rebuilds = full_rebuilds;
  result.srr_rebuilt_nodes = rebuilt_nodes;
  for (int32_t count : visits) {
    if (count > 0)
      ++result.srr_scope_nodes;
    if (count > 1)
      result.srr_revisits += count - 1;
  }
  return result;
}

Solution RoutingDecoder::perturb(uint64_t rollout_seed, const float *edge_field,
                                 const float *edge_additive,
                                 const float *edge_state_field,
                                 const float *multipliers,
                                 const float *coupler_weights,
                                 const float *coupler_bias,
                                 const float *objective_residual,
                                 RolloutTrace *trace, bool greedy) const {
  const Solution source = evaluate(incumbent_route_);
  if (!source.feasible) {
    return construct(rollout_seed, edge_field, edge_additive, edge_state_field, multipliers,
                     coupler_weights, coupler_bias, objective_residual, trace,
                     greedy);
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
        current, used, rng, edge_field, edge_additive, edge_state_field, multipliers,
        coupler_weights, coupler_bias, objective_residual,
        greedy);
    std::vector<int32_t> valid_indices;
    valid_indices.reserve(order.size());
    State prefix_state;
    const bool has_prefix =
        trace != nullptr && incumbent_prefix_state(current, prefix_state);
    if (has_prefix)
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
                      static_cast<float>(log_probability), live_state);
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
      scope_restricted_refine(raw, initial_scope, edge_field, edge_additive, edge_state_field,
                              multipliers, coupler_weights, coupler_bias,
                              objective_residual, trace,
                              rng);
  refined.raw_objective = raw.raw_objective;
  refined.changed_edges = raw.changed_edges;
  return refined;
}

std::vector<Solution> RoutingDecoder::sample(const float *edge_field,
                                             const float *edge_additive,
                                             const float *edge_state_field,
                                             const float *multipliers,
                                             const float *coupler_weights,
                                             const float *coupler_bias,
                                             const float *objective_residual,
                                             DecisionTrace *trace) {
  validate_guidance(edge_field, edge_additive, edge_state_field, multipliers, coupler_weights,
                    coupler_bias, objective_residual);
  std::vector<Solution> solutions(n_rollouts_);
  std::vector<RolloutTrace> rollout_traces(trace == nullptr ? 0 : n_rollouts_);
  // The per-row verification counter is sized by the registry, so it has to be
  // allocated where the registry is known rather than by the struct's default
  // member initializer.
  for (RolloutTrace &rollout : rollout_traces) {
    rollout.screening_verification_failures_by_channel.assign(
        static_cast<size_t>(resource_count()), 0);
  }
  const uint64_t generation_seed = splitmix64(seed_ ^ generation_++);
  const int32_t thread_count = std::min(n_rollouts_, omp_get_max_threads());
#pragma omp parallel for schedule(static) num_threads(thread_count)
  for (int32_t rollout = 0; rollout < n_rollouts_; ++rollout) {
    const uint64_t rollout_seed = splitmix64(generation_seed + rollout);
    RolloutTrace *rollout_trace = trace == nullptr ? nullptr : &rollout_traces[rollout];
    solutions[rollout] =
        incumbent_route_.empty()
            ? construct(rollout_seed, edge_field, edge_additive, edge_state_field, multipliers,
                        coupler_weights, coupler_bias, objective_residual,
                        rollout_trace, rollout < std::max(1, n_rollouts_ / 2))
            : perturb(rollout_seed, edge_field, edge_additive, edge_state_field, multipliers,
                      coupler_weights, coupler_bias, objective_residual,
                      rollout_trace);
  }
  if (trace != nullptr) {
    *trace = DecisionTrace{};
    trace->starts.reserve(n_rollouts_ + 1);
    trace->starts.push_back(0);
    trace->screening_verification_failures_by_channel.assign(
        static_cast<size_t>(resource_count()), 0);
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
      if (trace->screening_verification_failures_by_channel.size() <
          rollout.screening_verification_failures_by_channel.size()) {
        trace->screening_verification_failures_by_channel.resize(
            rollout.screening_verification_failures_by_channel.size(), 0);
      }
      for (size_t channel = 0;
           channel < rollout.screening_verification_failures_by_channel.size();
           ++channel) {
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
    const float *edge_state_field,
    const float *multipliers, const float *coupler_weights,
    const float *coupler_bias, const float *objective_residual) const {
  validate_guidance(edge_field, edge_additive, edge_state_field, multipliers, coupler_weights,
                    coupler_bias, objective_residual);
  const uint64_t deterministic_seed = splitmix64(seed_);
  return incumbent_route_.empty()
             ? construct(deterministic_seed, edge_field, edge_additive, edge_state_field,
                         multipliers, coupler_weights, coupler_bias,
                         objective_residual, nullptr, true)
             : perturb(deterministic_seed, edge_field, edge_additive, edge_state_field,
                       multipliers, coupler_weights, coupler_bias,
                       objective_residual, nullptr, true);
}

bool RoutingDecoder::better(const Solution &lhs, const Solution &rhs) const {
  if (!lhs.feasible) {
    return false;
  }
  if (!rhs.feasible) {
    return true;
  }
  // Search always minimizes `sense * objective`; ties break toward shorter
  // travel. For a pure-distance objective the tie-break is a no-op (objective is
  // the distance) so this reduces to the former strict distance comparison.
  const float sense = problem_.objective.sense;
  if (std::abs(lhs.objective - rhs.objective) > FEASIBILITY_EPS) {
    return sense * lhs.objective < sense * rhs.objective;
  }
  return lhs.distance < rhs.distance;
}

Solution RoutingDecoder::solve(int32_t iterations, const float *edge_field,
                               const float *edge_additive,
                               const float *edge_state_field,
                               const float *multipliers,
                               const float *coupler_weights,
                               const float *coupler_bias,
                               const float *objective_residual) {
  if (iterations <= 0) {
    throw std::invalid_argument("iterations must be positive");
  }
  validate_guidance(edge_field, edge_additive, edge_state_field, multipliers, coupler_weights,
                    coupler_bias, objective_residual);
  std::vector<float> working_field;
  std::vector<float> working_additive;
  std::vector<float> working_objective_residual;
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
  if (objective_residual != nullptr)
    working_objective_residual.assign(
        objective_residual, objective_residual + edge_to_.size());
  // The state-conditioned field is per edge, so it is aligned to the candidate
  // graph and has to survive a rebuild the same way the field does.
  std::vector<float> working_state_field;
  if (edge_state_field != nullptr) {
    working_state_field.assign(
        edge_state_field,
        edge_state_field + static_cast<size_t>(edge_to_.size()) *
                               resource_count() * live_state_feature_count());
  }
  for (int32_t iteration = 0; iteration < iterations; ++iteration) {
    std::vector<Solution> solutions = sample(
        working_field.empty() ? nullptr : working_field.data(),
        working_additive.empty() ? nullptr : working_additive.data(),
        working_state_field.empty() ? nullptr : working_state_field.data(),
        multipliers, coupler_weights, coupler_bias,
        working_objective_residual.empty()
            ? nullptr
            : working_objective_residual.data());
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
                            working_objective_residual.empty()
                                ? nullptr
                                : &working_objective_residual,
                            working_state_field.empty()
                                ? nullptr
                                : &working_state_field);
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
  // Declared precedence rows carry one counter each, in the same slot layout the
  // construction state uses.
  std::vector<float> precedence_state(
      precedence_resource_indices_.empty() ? 0 : resource_count(), 0.0f);
  const auto precedence_ok = [&](int32_t node, bool depot) {
    for (int32_t index : precedence_resource_indices_) {
      if (!precedence_admits(
              resource(index), node, depot, precedence_state[index],
              precedence_prerequisites_met(
                  resource(index), node, [&](int32_t required) {
                    return visited[static_cast<size_t>(required)] != 0;
                  })))
        return false;
    }
    return true;
  };
  const auto precedence_advance = [&](int32_t node, bool depot) {
    for (int32_t index : precedence_resource_indices_)
      precedence_state[index] = precedence_next_state(
          resource(index), node, depot, precedence_state[index]);
  };
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

  // Always sized: these are copied into a State whose slot accessors index the
  // full registry, so an empty vector here is an out-of-bounds read there.
  std::vector<float> scalar_state(state_slot_count(), 0.0f);
  // Guarded resets depend on the unserved remainder; this replay tracks it the
  // same way construction does.
  std::vector<int32_t> scalar_guards = initial_reset_guards();
  std::vector<float> scalar_since_rest(state_slot_count(), 0.0f);
  for (int32_t resource_index : scalar_resource_indices_) {
    scalar_state[resource_index] = resource(resource_index).initial;
    scalar_since_rest[resource_index] = resource(resource_index).initial;
  }
  int32_t resource_current = current;
  float scalar_break_duration = 0.0f;
  const auto extend_resource_kernels = [&](int32_t next,
                                           bool force_route_end) {
    scalar_break_duration = 0.0f;
    for (int32_t resource_index : scalar_resource_indices_) {
      const ResourceSpec &spec = resource(resource_index);
      float value = scalar_state[resource_index];
      float rest = scalar_since_rest[resource_index];
      bool break_taken = false;
      const bool feasible =
          extend_declared(spec, resource_current, next, route_depot,
                          force_route_end,
                          scalar_guards, resource_index,
                          value, rest, &break_taken);
      if (break_taken)
        scalar_break_duration += spec.optional_reset_duration;
      if (!feasible) {
        failed.error = "resource bound failed: " + spec.name;
        return false;
      }
      scalar_state[resource_index] = value;
      scalar_since_rest[resource_index] = rest;
    }
    consume_reset_guards(scalar_guards, next);
    resource_current = next;
    return true;
  };

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
    if (!extend_resource_kernels(next, false))
      return failed;

    if (next < problem_.depot_count) {
      if (problem_.has(TIME_WINDOWS) && !problem_.open_route &&
          current_time + scalar_break_duration +
                  problem_.dist(current, route_depot) >
              problem_.tw_end[route_depot] + FEASIBILITY_EPS) {
        failed.error = "route contains an infeasible transition to node " +
                       std::to_string(next);
        return failed;
      }
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
      if (!depot_allowed || !precedence_ok(next, true)) {
        failed.error = "route contains an infeasible transition to node " +
                       std::to_string(next);
        return failed;
      }
      precedence_advance(next, true);
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

    if (visited[next] || !precedence_ok(next, false)) {
      failed.error = "route contains an infeasible transition to node " +
                     std::to_string(next);
      return failed;
    }
    precedence_advance(next, false);
    const int32_t pickup = problem_.pickup_of_delivery[next];
    if (problem_.has(PICKUP_DELIVERY) && pickup >= 0 && !visited[pickup]) {
      failed.error = "route contains an infeasible transition to node " +
                     std::to_string(next);
      return failed;
    }
    // Benchmark rule: a route that opens on a backhaul starts empty, whether or
    // not linehauls remain elsewhere (URS UniVRPEnv.py:517). `reload` only
    // covers the case where none do.
    if (at_depot && class_ordered() &&
        problem_.demand[next] < -FEASIBILITY_EPS) {
      load = 0.0f;
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
    const float arrival = std::max(current_time + scalar_break_duration + edge,
                                   problem_.tw_start[next]);
    float return_break_duration = 0.0f;
    if (!problem_.open_route) {
      State projected;
      // resource_state has to be in place before any slot accessor is used.
      projected.resource_state = scalar_state;
      projected.resource_since_rest = scalar_since_rest;
      projected.current = next;
      projected.route_depot = route_depot;
      slot(projected, FieldChannel::TIME_WINDOW) = arrival + problem_.service_time[next];
      return_break_duration = transition_break_duration(projected, route_depot);
    }
    if (problem_.has(TIME_WINDOWS) &&
        (arrival > problem_.tw_end[next] + FEASIBILITY_EPS ||
         (!problem_.open_route &&
          arrival + problem_.service_time[next] + return_break_duration +
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
    if (!extend_resource_kernels(end, true))
      return failed;
    distance += problem_.dist(current, end);
    if (find_edge(current, end) < 0)
      ++off_graph_edges;
  }

  for (int32_t resource_index : precedence_resource_indices_) {
    const ResourceSpec &spec = resource(resource_index);
    if (spec.relation == PrecedenceRelation::PAIRWISE &&
        precedence_state[resource_index] > FEASIBILITY_EPS) {
      failed.error = "unresolved precedence relation: " + spec.name;
      return failed;
    }
  }
  for (int32_t resource_index : scalar_resource_indices_) {
    const ResourceSpec &spec = resource(resource_index);
    if (spec.bound_check == BoundCheck::SOLUTION_END &&
        (scalar_state[resource_index] <
             spec_lower(spec, resource_current) - FEASIBILITY_EPS ||
         scalar_state[resource_index] >
             spec_upper(spec, resource_current) + FEASIBILITY_EPS)) {
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
  solution.objective = problem_.objective.report(
      solution.distance, solution.collected_prize, solution.missed_penalty);
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
  float capacity_binding = 0.0f;
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
    // Accumulate into a local, not into result.binding: the report vector is
    // registry-sized, so a problem that declares no capacity row -- or no rows
    // at all -- has no slot here to accumulate into. `report` below places the
    // total once, if there is a row to place it in.
    capacity_binding = std::max(capacity_binding,
                                std::max(route_positive, route_negative) /
                                    capacity_scale);
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

  // A compiled kernel reaches its report through its registry row. These were
  // written at the channel's own ordinal, which is the same number only while
  // the registry keeps one row per channel in channel order.
  const auto report = [&](FieldChannel channel, float violation,
                          float binding) {
    const int32_t index = field_resource_index(channel);
    if (index < 0)
      return;
    result.violation[index] = violation;
    result.binding[index] = std::clamp(binding, 0.0f, 1.0f);
  };
  report(FieldChannel::CAPACITY, capacity_excess / capacity_scale,
         capacity_binding);
  report(FieldChannel::TIME_WINDOW, time_warp / time_scale,
         problem_.has(TIME_WINDOWS) ? 1.0f - min_time_slack / time_scale
                                    : 0.0f);
  report(FieldChannel::ROUTE_LIMIT, route_excess / route_scale,
         max_route_ratio);
  report(FieldChannel::TOUR_LIMIT, tour_excess / tour_scale, max_tour_ratio);
  report(FieldChannel::BACKHAUL_ORDER,
         static_cast<float>(backhaul_violations) /
             std::max(problem_.customer_count(), 1),
         any_backhaul ? 1.0f : 0.0f);
  report(FieldChannel::PICKUP_DELIVERY,
         static_cast<float>(precedence_violations) / std::max(pair_count, 1),
         static_cast<float>(max_open_pickups) / std::max(pair_count, 1));
  report(FieldChannel::PRIZE_QUOTA,
         std::max(problem_.prize_quota - collected_prize, 0.0f) / quota_scale,
         problem_.has(PRIZE_QUOTA) ? collected_prize / quota_scale : 0.0f);
  for (int32_t index = 0; index < resource_count(); ++index) {
    if (!resource(index).active) {
      result.violation[index] = 0.0f;
      result.binding[index] = 0.0f;
    } else if (result.violation[index] > FEASIBILITY_EPS) {
      result.binding[index] = 1.0f;
    }
  }
  // Always sized: these are copied into a State whose slot accessors index the
  // full registry, so an empty vector here is an out-of-bounds read there.
  std::vector<float> scalar_state(state_slot_count(), 0.0f);
  // Guarded resets depend on the unserved remainder; this replay tracks it the
  // same way construction does.
  std::vector<int32_t> scalar_guards = initial_reset_guards();
  std::vector<float> scalar_since_rest(state_slot_count(), 0.0f);
  for (int32_t resource_index : scalar_resource_indices_) {
    scalar_state[resource_index] = resource(resource_index).initial;
    scalar_since_rest[resource_index] = resource(resource_index).initial;
  }
  int32_t resource_current = route.front();
  // `node` resolves per-node bounds; reporting must use the same bound the
  // feasibility path used, or a rejected route reports zero violation.
  const auto record_scalar = [&](int32_t resource_index, float value,
                                 bool check_bound, int32_t node) {
    const ResourceSpec &spec = resource(resource_index);
    const float scale = runtime_resource_scale(resource_index);
    const float lower = spec_lower(spec, node);
    const float upper = spec_upper(spec, node);
    float violation = 0.0f;
    if (check_bound) {
      if (std::isfinite(lower))
        violation = std::max(violation, lower - value);
      if (std::isfinite(upper))
        violation = std::max(violation, value - upper);
    }
    result.violation[resource_index] = std::max(
        result.violation[resource_index], violation / std::max(scale, EPS));
    float binding = 0.0f;
    if (std::isfinite(lower) && std::isfinite(upper)) {
      const float slack = std::min(value - lower, upper - value);
      binding = 1.0f - slack / std::max(0.5f * (upper - lower), EPS);
    } else if (std::isfinite(lower)) {
      binding = 1.0f - (value - lower) / std::max(scale, EPS);
    } else if (std::isfinite(upper)) {
      binding = 1.0f - (upper - value) / std::max(scale, EPS);
    }
    result.binding[resource_index] =
        std::max(result.binding[resource_index],
                 std::clamp(binding, 0.0f, 1.0f));
  };
  // Declared precedence rows report the share of their relations a route puts in
  // the wrong order, and the depth of unresolved obligations as tightness. They
  // are counted here rather than left at zero: a route evaluate() rejects must
  // not be reported as violation-free.
  std::vector<float> precedence_open(
      precedence_resource_indices_.empty() ? 0 : resource_count(), 0.0f);
  std::vector<uint8_t> precedence_visited(
      precedence_resource_indices_.empty() ? 0 : problem_.node_count, 0);
  if (!precedence_resource_indices_.empty())
    precedence_visited[route.front()] = 1;
  const auto extend_precedence_kernels = [&](int32_t next) {
    if (precedence_resource_indices_.empty())
      return;
    const bool depot = next < problem_.depot_count;
    for (int32_t index : precedence_resource_indices_) {
      const ResourceSpec &spec = resource(index);
      const bool served = precedence_prerequisites_met(
          spec, next, [&](int32_t node) {
            return precedence_visited[static_cast<size_t>(node)] != 0;
          });
      if (!precedence_admits(spec, next, depot, precedence_open[index], served))
        result.violation[index] += 1.0f / std::max(spec.relation_count, 1);
      precedence_open[index] =
          precedence_next_state(spec, next, depot, precedence_open[index]);
      result.binding[index] = std::max(
          result.binding[index],
          std::clamp(precedence_open[index] /
                         std::max(static_cast<float>(spec.relation_count), 1.0f),
                     0.0f, 1.0f));
    }
    precedence_visited[static_cast<size_t>(next)] = 1;
  };
  const auto extend_scalar_kernels = [&](int32_t next, bool force_route_end) {
    const bool depot = next < problem_.depot_count;
    for (int32_t resource_index : scalar_resource_indices_) {
      const ResourceSpec &spec = resource(resource_index);
      const bool check = spec.bound_check == BoundCheck::TRANSITION ||
                         ((depot || force_route_end) &&
                          spec.bound_check == BoundCheck::ROUTE_END);
      float value = scalar_state[resource_index];
      float rest = scalar_since_rest[resource_index];
      float bounded = value;
      (void)extend_declared(spec, resource_current, next, route_depot,
                            force_route_end,
                            scalar_guards, resource_index,
                            value, rest, nullptr, &bounded);
      record_scalar(resource_index, bounded, check, next);
      scalar_state[resource_index] = value;
      scalar_since_rest[resource_index] = rest;
    }
    consume_reset_guards(scalar_guards, next);
    resource_current = next;
  };
  for (size_t route_index = 1; route_index < route.size(); ++route_index) {
    extend_scalar_kernels(route[route_index], false);
    extend_precedence_kernels(route[route_index]);
  }
  if (problem_.has(VISIT_ALL) && !problem_.multi_route && !at_depot &&
      !problem_.open_route) {
    const int32_t end = problem_.depot_count == 0 ? route.front() : route_depot;
    extend_scalar_kernels(end, true);
  }
  for (int32_t resource_index : scalar_resource_indices_) {
    if (resource(resource_index).bound_check == BoundCheck::SOLUTION_END)
      record_scalar(resource_index, scalar_state[resource_index], true,
                    resource_current);
    if (result.violation[resource_index] > FEASIBILITY_EPS)
      result.binding[resource_index] = 1.0f;
  }
  for (int32_t index : precedence_resource_indices_) {
    const ResourceSpec &spec = resource(index);
    // An obligation still open at the end of the solution is never resolved.
    if (spec.relation == PrecedenceRelation::PAIRWISE &&
        precedence_open[index] > FEASIBILITY_EPS)
      result.violation[index] +=
          precedence_open[index] / std::max(spec.relation_count, 1);
    if (result.violation[index] > FEASIBILITY_EPS)
      result.binding[index] = 1.0f;
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
