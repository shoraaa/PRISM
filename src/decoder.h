#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace prism {

static constexpr int32_t FIELD_CHANNEL_COUNT = 7;
static constexpr int32_t CONSTRAINT_KERNEL_COUNT = 8;
static constexpr int32_t RESOURCE_KERNEL_COUNT = 10;
// Constraint-named slots (forward/backward load, time, slack and open pickups)
// used to live here, and so did opens_relation / requires_relation: a pairwise
// row's node semantics now travel as its published node term (the relation
// delta, +1 at an opener and -1 at a requirer) through node_resource_features,
// the same shared weights demand uses, so no slot here names a constraint. They only ever existed for three of the seven compiled
// kernels, so a declared row had no route-state column at all. They are now
// read per resource from incumbent_live_state / incumbent_suffix_state and
// pooled through the same shared weights, leaving this vector free of any slot
// named after a constraint.
static constexpr int32_t NODE_FEATURE_COUNT = 9;
// Node-side terms of the declared objective: the quantity charged on VISITED
// nodes and the one charged on UNVISITED nodes. Unlike the resource registry
// this list is closed -- ObjectiveSpec declares exactly three quantities and
// the third (travel) is an edge term -- so these are slots of the objective
// language rather than a registry that grows. Which slot means what, and
// whether it is live at all, is carried by the declared coefficient vector, so
// no column is named after prize or penalty.
static constexpr int32_t OBJECTIVE_NODE_TERM_COUNT = 2;
// Per-node, per-resource attributes, read off each row's published algebra:
// the node increment split by sign, the tropical clamp, the finite bound, and
// the departure term. This is the node-side counterpart of the per-edge
// resource pressure: variable in the number of resources, with no slot named
// after any constraint, so a declared row's node attributes reach the model
// through the same shared weights that demand and time windows do. The last
// slot is the node's rank within whatever order the row declares -- its class
// for a class order, which side of the relation it sits on for a pairwise one --
// because membership in an order is not an increment and no other slot could
// carry it. A class-ordered row previously had no node-level representation at
// all.
static constexpr int32_t NODE_RESOURCE_FEATURE_COUNT = 6;
// Reverse-route sufficient statistics published for every resource row.  The
// legacy scalar suffix below keeps |signed total| for compatibility, but that
// scalar destroys information whenever positive and negative increments cancel
// (capacity with deliveries/backhauls is the common case).  These four columns
// retain the positive/negative parts of the complete remaining increment and of
// its departure component.  They are algebraic quantities rather than
// constraint-named features, so the same contract applies to appended rows.
enum class ResourceSuffixFeature : int32_t {
  POSITIVE_TOTAL = 0,
  NEGATIVE_TOTAL = 1,
  POSITIVE_DEPARTURE = 2,
  NEGATIVE_DEPARTURE = 3,
};
static constexpr int32_t RESOURCE_SUFFIX_FEATURE_COUNT = 4;
// Minimal behavioral view of a resource transition at an incumbent-prefix
// state. The native interpreter supplies the normalized post-transition state
// and a signed admissibility margin directly to the shared neural row encoder;
// no constraint name or learned reconstruction head is involved.
enum class ResourceTransitionFeature : int32_t {
  NEXT_STATE = 0,
  SIGNED_MARGIN = 1,
};
static constexpr int32_t RESOURCE_TRANSITION_FEATURE_COUNT = 2;
// Slot 3 carried a waived open-route return leg, which was a hard-coded special
// case for one route structure. It now carries the DECLARED objective's own edge
// term (squashed, scale-normalized), the edge-side counterpart of
// node_objective_features: an open route's depot leg reads as genuinely free
// because the objective says so, not because a flag says so.
// One slot per compiled channel used to sit between `distance` and the
// incumbent flags. They duplicated the first seven columns of the per-resource
// resource_features tensor exactly, and a declared row had no slot, so they are
// gone: every row now reaches the model through resource_features alone.
static constexpr int32_t EDGE_FEATURE_COUNT = 5;
// The neural contract is factored exactly like the executable resource row:
// fixed row properties plus a variable-length set of term properties.  Terms
// are pooled by shared learned weights in net.py, so adding a term never adds a
// cold input column.
static constexpr int32_t RESOURCE_ROW_PROPERTY_DIM = 14;
static constexpr int32_t RESOURCE_TERM_PROPERTY_DIM = 20;
// One multiplier slot beyond the resource rows carries the objective weight
// applied to the objective edge cost. ConstraintFieldNet fixes this slot to one
// (and its coupler to zero). The optional objective_residual argument is a
// low-level external-control hook; ConstraintFieldNet does not produce one.
// Both the slot's position
// and the total count are properties of the registry, so they are
// RoutingDecoder::objective_multiplier()/multiplier_count() rather than
// constants; the constants were only ever right while every problem carried
// exactly FIELD_CHANNEL_COUNT rows.

enum class FieldChannel : uint8_t {
  CAPACITY = 0,
  TIME_WINDOW = 1,
  ROUTE_LIMIT = 2,
  TOUR_LIMIT = 3,
  BACKHAUL_ORDER = 4,
  PICKUP_DELIVERY = 5,
  PRIZE_QUOTA = 6,
};

enum class ResourceOperator : uint8_t {
  CAPACITY,
  TIME_WINDOW,
  ROUTE_LIMIT,
  TOUR_LIMIT,
  BACKHAUL_ORDER,
  PICKUP_DELIVERY,
  PRIZE_QUOTA,
  // One affine family, parameterized by the semiring its update runs in. The
  // tropical case used to be a separate AFFINE_MAX operator, but execution
  // always shared this path: the only difference was a hardcoded max against a
  // per-node operand. Declaring the semiring makes that a parameter, so
  // min_plus comes for free and the descriptor stops carrying the same fact in
  // two places (an operator one-hot and a join-operand-present bit).
  AFFINE_ACCUMULATOR,
  // Precedence relation over nodes. Not a resource extension function at all:
  // there is no accumulated quantity, only an admissibility predicate over what
  // the route has already served. Pickup-delivery and backhaul ordering are its
  // two declared relation forms, and an arbitrary precedence DAG is a third.
  PRECEDENCE,
};

// PAIRWISE:    a node may not be served before its declared predecessor, and a
//              route may not close while a predecessor it served is unresolved.
//              The relation is a matching: each node takes part in at most one,
//              which is what pickup-delivery pairing means.
// CLASS_ORDER: within a route, node classes must be served in non-decreasing
//              order. Backhaul is the two-class case (linehaul then backhaul).
// DAG:         a node may not be served until *every* declared predecessor has
//              been. This is the general precedence relation -- the sequential
//              ordering problem's constraint -- and it is not expressible as a
//              set of PAIRWISE rows, because those are matchings and a node may
//              take part in only one of them across the whole registry.
enum class PrecedenceRelation : uint8_t { PAIRWISE, CLASS_ORDER, DAG };

// Which compiled kernel, if any, executes a row. This is a dispatch tag, not an
// identity: `ResourceOperator` says what a row IS (its algebra), and this says
// who runs it. They used to be the same field, so a row's operator doubled as
// the name of a constraint and no declared row could ever reach a fast path.
// NONE means the declarative interpreter runs the row.
enum class FastPath : uint8_t {
  NONE,
  CAPACITY,
  TIME_WINDOW,
  ROUTE_LIMIT,
  TOUR_LIMIT,
  BACKHAUL_ORDER,
  PICKUP_DELIVERY,
  PRIZE_QUOTA,
};

// The join a row applies against its per-node operand. This is the (+) of the
// semiring the row's update runs in; the affine accumulation below is its (*).
// Declaring it is what lets one affine family cover both a plain accumulator
// and a tropical one, rather than branching on an operator name: max_plus is a
// time window (arrival waits for the window to open), min_plus its dual (a
// value capped at a per-node ceiling), and arithmetic has no join at all.
//
// The identity differs per semiring (-inf for max, +inf for min), so a row with
// no declared operand is inert under any of them -- see join_identity.
enum class ResourceSemiring : uint8_t { ARITHMETIC, MAX_PLUS, MIN_PLUS };

// One contribution to a row's transition, described by where it reads and the
// independent operation, phase, trigger, and gate coordinates that consume it.
//
// These used to be six named struct fields -- node_values, edge_values,
// departure_values, reset_value and so on -- each of which hardcoded one
// (source, point, behavior) combination in its name. That is why every new routing
// rule needed a new field: `departure` exists only because it is `node_values`
// at a later stage. Naming the coordinates instead makes a new rule a new
// *point* in this space rather than a new axis, which is the difference between
// an algebra and a list of special cases.
enum class TermSource : uint8_t {
  CONSTANT,
  NODE_ATTRIBUTE,
  EDGE_ATTRIBUTE,
  DISTANCE,
};
// Which endpoint a node attribute is read at. Edge sources read the pair.
enum class TermPoint : uint8_t { TO, FROM };

// Independent term coordinates.  The former stage enum bundled all four and
// made RESET impossible to state as a term.  CHECKPOINT/RESTORE represent the
// latest-feasible optional-reset schedule: CHECKPOINT records the accumulator
// at an eligible node and RESTORE installs that saved state on bound failure.
enum class TermOperation : uint8_t {
  ADD,
  JOIN,
  ASSIGN,
  CHECKPOINT,
  RESTORE,
};
enum class TermPhase : uint8_t { BEFORE_BOUND, AFTER_BOUND };
enum class TermTrigger : uint8_t {
  ALWAYS,
  RESET_DEPARTURE,
  RESET_ARRIVAL,
  BOUND_FAILURE,
  CHECKPOINT_ARRIVAL,
};
enum class TermGate : uint8_t {
  ALWAYS,
  REMAINDER_DEFAULT,
  REMAINDER_ALTERNATIVE,
};

struct ResourceTerm {
  TermSource source = TermSource::CONSTANT;
  TermPoint point = TermPoint::TO;
  TermOperation operation = TermOperation::ADD;
  TermPhase phase = TermPhase::BEFORE_BOUND;
  TermTrigger trigger = TermTrigger::ALWAYS;
  TermGate gate = TermGate::ALWAYS;
  float coefficient = 1.0f;
  float constant = 0.0f;
  // [node_count] for NODE_ATTRIBUTE, [node_count^2] for EDGE_ATTRIBUTE.
  std::vector<float> values;
  // Event selector owned by the term.  This keeps reset firing conditions out
  // of ResourceSpec and distinguishes depot events from declared node events.
  bool trigger_at_depot = false;
  std::vector<uint8_t> trigger_nodes;
  // Remainder predicate owned by guarded ASSIGN terms.  The alternative gate
  // is true iff matching==0 and opposing>0; DEFAULT is its exact complement,
  // including the empty remainder.
  std::vector<float> gate_values;
  float gate_sign = 0.0f;
};

enum class ResourceDirection : uint8_t { FORWARD, BACKWARD, BIDIRECTIONAL };
enum class ResourceScope : uint8_t { ROUTE, TOUR, SOLUTION };
enum class BoundCheck : uint8_t { TRANSITION, ROUTE_END, SOLUTION_END };

// How far ahead a bound is projected before it is tested.
//   TRANSITION           -- the bound holds at the arrival state, nothing more.
//   RETURN               -- the bound must also hold after the cheapest depot
//                           return leg. Sound (implied by any closed feasible
//                           route under the triangle inequality) whenever the
//                           resource cannot be replenished at an interior node,
//                           so it is enforced as a feasibility condition.
//   RETURN_CONSTRUCTION  -- the same projection applied only while constructing
//                           a route. Resources that reset at interior nodes
//                           (chargers, rest stops) may legitimately reach the
//                           depot via a reset, so the projection is a stranding
//                           guard rather than a necessary condition and must not
//                           reject an already-closed route.
enum class BoundHorizon : uint8_t { TRANSITION, RETURN, RETURN_CONSTRUCTION };

// Runtime resource row. Named input references from the Python algebra schema
// are resolved into dense arrays once in the binding, so the hot decoder path
// never calls Python and never performs string lookup.
struct ResourceSpec {
  std::string name;
  // Compiled constraint rows and declarative rows share this representation.
  // Inactive constraint rows remain present so tensor positions are stable
  // within the current model contract.
  bool active = true;
  ResourceOperator op = ResourceOperator::AFFINE_ACCUMULATOR;
  // AFFINE_ACCUMULATOR only. ARITHMETIC leaves join_values unused.
  ResourceSemiring semiring = ResourceSemiring::ARITHMETIC;
  // Set by build_resource_registry for the rows the constraint frontend
  // creates. A declared row is interpreted: see the note at that call site for
  // why a kernel cannot yet be selected by matching a row's algebra.
  FastPath fast_path = FastPath::NONE;
  ResourceDirection direction = ResourceDirection::FORWARD;
  ResourceScope scope = ResourceScope::ROUTE;
  BoundCheck bound_check = BoundCheck::TRANSITION;
  BoundHorizon horizon = BoundHorizon::TRANSITION;
  int32_t state_dim = 1;
  float initial = 0.0f;
  float scale = 1.0f;
  float lower = -std::numeric_limits<float>::infinity();
  float upper = std::numeric_limits<float>::infinity();
  float edge_coefficient = 0.0f;
  float node_coefficient = 0.0f;
  // Node term charged *after* the bound is tested (a service time: it delays
  // departure but is not itself subject to the arrival window).
  float departure_coefficient = 0.0f;
  bool edge_uses_distance = false;
  bool reset_at_depot = false;
  float reset_value = 0.0f;
  // A reset whose value depends on what the route still has left to do.
  //
  // A constant reset cannot express a reload that is conditional on the
  // remaining work: a vehicle serving mixed pickups and deliveries reloads to
  // full while deliveries remain and to empty once only pickups do. The guard
  // is a predicate over the *unvisited* set -- "does some unvisited node still
  // carry a positive `demand`" -- and the reset takes `reset_value` while it
  // holds and `reset_otherwise` once it does not.
  //
  // Empty `reset_guard_values` means the reset is the plain constant. The sign
  // is +1 to count strictly positive attributes and -1 for strictly negative;
  // zero-valued attributes satisfy neither, matching the kernel convention that
  // a zero-demand customer is neither a linehaul nor a backhaul.
  std::vector<float> reset_guard_values;
  float reset_guard_sign = 0.0f;
  float reset_otherwise = 0.0f;
  // Optional resets are taken at the current node before traversing an edge,
  // but only when the unreset transition would violate this resource's bound.
  // This gives route-only representations a canonical latest-feasible schedule
  // for renewable resources such as continuous driver time.
  float optional_reset_duration = 0.0f;
  std::vector<float> edge_values;
  std::vector<float> node_values;
  std::vector<float> departure_values;
  // PRECEDENCE only. `predecessor[j]` is the node that must precede j (-1 for
  // none) and `successor[j]` its inverse, used to count unresolved obligations.
  // `node_class[j]` is the ordering class for CLASS_ORDER.
  PrecedenceRelation relation = PrecedenceRelation::PAIRWISE;
  std::vector<int32_t> predecessor;
  std::vector<int32_t> successor;
  std::vector<int32_t> node_class;
  // DAG only. CSR over the predecessors of each node: node j must wait for
  // every entry in [predecessor_offsets[j], predecessor_offsets[j + 1]).
  // `predecessor`/`successor` stay empty, so nothing that reads the pairwise
  // matching picks a DAG row up by accident.
  std::vector<int32_t> predecessor_offsets;
  std::vector<int32_t> predecessor_list;
  std::vector<int32_t> successor_count;
  // Number of declared relations in this row (pairs, or class-ordered nodes),
  // resolved once so pressure does not rescan the arrays.
  int32_t relation_count = 0;
  // Per-node operand of the row's join, applied to the accumulated value before
  // the bound is tested. Under MAX_PLUS this is a floor (a time window's ready
  // time, so arrival waits); under MIN_PLUS a ceiling. Empty means the row has
  // no join operand and is inert under any semiring.
  std::vector<float> join_values;
  // Per-node bounds. Empty means the scalar `lower`/`upper` applies everywhere;
  // a time window needs a bound that varies node by node.
  std::vector<float> lower_values;
  std::vector<float> upper_values;
  std::vector<uint8_t> reset_nodes;
  std::vector<uint8_t> optional_reset_nodes;
  // Opening correction: added to the value a route carries out of a reset node,
  // read at the node it opens toward. A route whose initial state depends on
  // its first customer -- a backhaul-ordered vehicle leaves empty when it opens
  // on a pickup -- has nowhere else to be stated, because a reset fires on
  // *arriving* at the depot and cannot see where the route goes next.
  std::vector<float> opening_values;
  float opening_coefficient = 1.0f;
  // Execution form of the extension, derived from optional input shorthand by
  // build_terms(). Published declarations expose only terms. They are what the
  // transition actually runs, so a phase can no longer be
  // implied by a field name. Anything reading increments -- resource pressure
  // above all -- filters on ADD, which is why a reset correction can no
  // longer leak into a quantity that means "what this edge costs".
  std::vector<ResourceTerm> terms;

};

// The objective is declared as pure data: a signed linear combination over the
// three quantities every route accumulates (total travel, prize collected on
// visited nodes, penalty incurred on unvisited nodes). `sense` is +1 to
// minimize the reported value and -1 to maximize it, so search always minimizes
// `sense * report(...)`. `distance_regularizer` is a tiny travel tie-break that
// only matters when `distance_coeff == 0` (e.g. pure prize). A new objective is
// a new coefficient vector -- no enum, no switch, no new code path.
struct ObjectiveSpec {
  float distance_coeff = 1.0f;
  float visit_coeff = 0.0f;
  float miss_coeff = 0.0f;
  float distance_regularizer = 0.0f;
  float sense = 1.0f;
  std::string name = "distance";

  float report(float distance, float collected_prize,
               float missed_penalty) const {
    return distance_coeff * distance + visit_coeff * collected_prize +
           miss_coeff * missed_penalty;
  }
  bool has_node_term() const {
    return visit_coeff != 0.0f || miss_coeff != 0.0f;
  }
  const char *direction() const { return sense < 0.0f ? "maximize" : "minimize"; }
};

enum Constraint : uint32_t {
  VISIT_ALL = 1u << 0,
  CAPACITY = 1u << 1,
  BACKHAUL_ORDER = 1u << 2,
  PICKUP_DELIVERY = 1u << 3,
  ROUTE_LIMIT = 1u << 4,
  TIME_WINDOWS = 1u << 5,
  TOUR_LIMIT = 1u << 6,
  PRIZE_QUOTA = 1u << 7,
};

// Capabilities describe semantic properties needed by generic search
// operators. They are registered once with each compiled constraint kernel,
// instead of being reconstructed from variant names or ad-hoc flag lists.
enum ConstraintCapability : uint32_t {
  KERNEL_ROUTE_STATE = 1u << 0,
  KERNEL_SOLUTION_STATE = 1u << 1,
  KERNEL_ORDER_SENSITIVE = 1u << 2,
  KERNEL_REVERSAL_SENSITIVE = 1u << 3,
  KERNEL_RELATIONAL = 1u << 4,
};

struct ConstraintKernelSpec {
  Constraint constraint;
  const char *schema_name;
  // VISIT_ALL has no learned resource row and therefore uses -1.
  int32_t field_channel;
  ResourceOperator resource_operator;
  uint32_t capabilities;
};

struct ResourceKernelSpec {
  ResourceOperator op;
  const char *name;
  int32_t field_channel;
};

const std::array<ConstraintKernelSpec, CONSTRAINT_KERNEL_COUNT> &
constraint_kernel_registry();
const std::array<ResourceKernelSpec, RESOURCE_KERNEL_COUNT> &
resource_kernel_registry();
const ConstraintKernelSpec *constraint_kernel(Constraint constraint);
const ConstraintKernelSpec *field_channel_kernel(int32_t channel);
const ConstraintKernelSpec *constraint_kernel(const std::string &schema_name);
const ResourceKernelSpec &resource_kernel(ResourceOperator op);

struct Problem {
  std::string name;
  int32_t node_count = 0;
  int32_t depot_count = 0;
  uint32_t constraints = 0;
  ObjectiveSpec objective;
  bool multi_route = false;
  bool open_route = false;

  float capacity = 1.0f;
  float route_limit = std::numeric_limits<float>::infinity();
  float tour_limit = std::numeric_limits<float>::infinity();
  float prize_quota = 1.0f;

  // All arrays use the same node indexing. Distances are row-major n x n when
  // supplied; Euclidean instances may compute them from coordinates instead.
  std::vector<float> distance;
  // Optional Euclidean coordinates used by the KD-tree geometric channel.
  std::vector<float> coordinates;
  std::vector<float> demand;
  std::vector<float> prize;
  std::vector<float> penalty;
  std::vector<float> tw_start;
  std::vector<float> tw_end;
  std::vector<float> service_time;

  // -1 means no relation. Both arrays have node_count entries.
  std::vector<int32_t> delivery_of_pickup;
  std::vector<int32_t> pickup_of_delivery;

  // RoutingDecoder materializes declared constraint kernels and then appends
  // explicit resource rows using the same ResourceSpec/kernel contract.
  std::vector<ResourceSpec> resources;

  bool has(Constraint constraint) const;
  bool has_capability(ConstraintCapability capability) const;
  float dist(int32_t from, int32_t to) const;
  int32_t customer_count() const;
  void validate() const;
};

// The neighborhood is the distance neighborhood: the K nearest nodes plus the
// required depot overlay. A schema-derived variant that allocated part of K by
// per-resource relevance (optionally reweighted by a learned quota policy) was
// tried and measured consistently worse than the plain distance neighborhood,
// so it is gone.
struct CandidateConfig {
  int32_t max_candidates = 64;

  void validate() const;
};

struct SearchConfig {
  int32_t min_changed_edges = 8;
  int32_t max_perturb_attempts = 64;
  int32_t or_opt_max_segment = 3;
  int32_t feasibility_lookahead_depth = 2;
  bool use_srr = true;
  bool verify_screening_resources = false;
  bool verify_incremental_srr = false;
  // Field-gated exploration: a bounded number of guided-energy-descending but
  // objective-worsening moves the SRR descent may accept per invocation before
  // reverting to strict monotone descent. It lets the learned field steer the
  // search uphill (in objective) to escape local optima, while the champion
  // (best objective seen) and solve()'s best_solution_ guarantee we never
  // return worse. Constant energy has no gradient, so no guided exploration
  // move qualifies unless the explicit random_escape control below is enabled.
  // 0 disables the phase (default), preserving the pure monotone hill-climb.
  int32_t srr_exploration_budget = 0;
  // Select objective-nonimproving exploration moves by seeded random priority
  // instead of requiring a decrease in guided energy. This is an explicit
  // baseline control: it lets flat constant guidance use the same bounded
  // budget without changing construction or improving-move ranking.
  bool random_escape = false;
  // A candidate qualifies for exploration only if it lowers the anchor guided
  // energy by more than this margin, so numerical ties never trigger a kick.
  float srr_exploration_margin = 1.0e-6f;

  void validate() const;
};

struct Solution {
  std::vector<int32_t> route;
  bool feasible = false;
  float objective = std::numeric_limits<float>::infinity();
  float distance = 0.0f;
  float collected_prize = 0.0f;
  float missed_penalty = 0.0f;
  float raw_objective = std::numeric_limits<float>::infinity();
  int32_t changed_edges = 0;
  int32_t srr_moves = 0;
  int32_t srr_scope_nodes = 0;
  int32_t srr_revisits = 0;
  int32_t srr_evaluations = 0;
  int32_t srr_certified_evaluations = 0;
  int32_t srr_incremental_rebuilds = 0;
  int32_t srr_full_rebuilds = 0;
  int64_t srr_rebuilt_nodes = 0;
  int32_t off_graph_edges = 0;
  std::string error;
};

struct ResourceEvaluation {
  std::vector<float> violation;
  std::vector<float> binding;
  bool structurally_valid = false;
  std::string error;
};

struct DecisionTrace {
  // starts has n_rollouts + 1 entries and partitions every other decision array.
  std::vector<int32_t> starts;
  std::vector<int32_t> current_nodes;
  std::vector<int32_t> valid_offsets;
  std::vector<int32_t> valid_indices;
  std::vector<int32_t> chosen_indices;
  std::vector<uint8_t> stochastic;
  std::vector<float> log_probabilities;
  std::vector<float> live_state;
  std::vector<int32_t> screened_edges;
  std::vector<float> screened_resource_delta;
  int64_t screening_fast_evaluations = 0;
  int64_t screening_fallback_evaluations = 0;
  int64_t screening_verification_failures = 0;
  // One counter per registry row, sized by the decoder that fills it. This was
  // a fixed FIELD_CHANNEL_COUNT array, so a declared row's screening error was
  // not merely uncounted but unrepresentable.
  std::vector<int64_t> screening_verification_failures_by_channel;
};

class RoutingDecoder {
public:
  RoutingDecoder(Problem problem, CandidateConfig candidate_config = {},
             SearchConfig search_config = {}, int32_t n_rollouts = 20,
             float beta = 2.0f);

  void seed(uint64_t value);
  std::vector<Solution> sample(const float *edge_field = nullptr,
                               const float *edge_additive = nullptr,
                               const float *edge_state_field = nullptr,
                               const float *multipliers = nullptr,
                               const float *coupler_weights = nullptr,
                               const float *coupler_bias = nullptr,
                               const float *objective_residual = nullptr,
                               DecisionTrace *trace = nullptr);
  Solution sample_greedy(const float *edge_field = nullptr,
                         const float *edge_additive = nullptr,
                         const float *edge_state_field = nullptr,
                         const float *multipliers = nullptr,
                         const float *coupler_weights = nullptr,
                         const float *coupler_bias = nullptr,
                         const float *objective_residual = nullptr) const;
  Solution solve(int32_t iterations, const float *edge_field = nullptr,
                 const float *edge_additive = nullptr,
                 const float *edge_state_field = nullptr,
                 const float *multipliers = nullptr,
                 const float *coupler_weights = nullptr,
                 const float *coupler_bias = nullptr,
                 const float *objective_residual = nullptr);
  Solution evaluate(const std::vector<int32_t> &route) const;
  ResourceEvaluation
  evaluate_resources(const std::vector<int32_t> &route) const;
  void set_incumbent(const std::vector<int32_t> &route);
  std::vector<uint8_t> mask(const std::vector<int32_t> &prefix) const;

  const Problem &problem() const { return problem_; }
  const CandidateConfig &candidate_config() const { return candidate_config_; }
  const SearchConfig &search_config() const { return search_config_; }
  int32_t n_rollouts() const { return n_rollouts_; }
  int32_t edge_count() const { return static_cast<int32_t>(edge_to_.size()); }
  uint64_t graph_version() const { return graph_version_; }
  float beta() const { return beta_; }
  const std::vector<int32_t> &edge_offsets() const { return edge_offsets_; }
  const std::vector<int32_t> &edge_to() const { return edge_to_; }
  const std::vector<float> &edge_features() const { return edge_features_; }
  const std::vector<float> &node_features() const { return node_features_; }
  const std::vector<float> &incumbent_live_state() const {
    return incumbent_live_state_;
  }
  const std::vector<float> &incumbent_suffix_state() const {
    return incumbent_suffix_state_;
  }
  const std::vector<float> &incumbent_suffix_features() const {
    return incumbent_suffix_features_;
  }
  const std::vector<float> &incumbent_transition_features() const {
    return incumbent_transition_features_;
  }
  const std::vector<uint8_t> &incumbent_transition_feature_mask() const {
    return incumbent_transition_feature_mask_;
  }
  const std::vector<float> &node_objective_features() const {
    return node_objective_features_;
  }
  const std::vector<float> &resource_features() const {
    return resource_features_;
  }
  const std::vector<float> &resource_pressure() const {
    return resource_pressure_;
  }
  // [edge_count, resource_count] priced from each registry row directly rather
  // than from its published declaration. Diagnostic only: it is the reference a
  // published row must reproduce, and nothing in the search or the model reads
  // it.
  std::vector<float> compiled_resource_pressures() const;
  const std::vector<float> &resource_events() const {
    return resource_events_;
  }
  // [node_count, resource_count, NODE_RESOURCE_FEATURE_COUNT], row-major.
  const std::vector<float> &node_resource_features() const {
    return node_resource_features_;
  }
  int32_t resource_count() const {
    return static_cast<int32_t>(resources_.size());
  }
  // Declarative resource row a compiled kernel implements. `declared` reports
  // whether the kernel's semantics fit the current language at all. Compiled
  // kernels are fast paths for these rows, not a separate semantics.
  ResourceSpec declared_algebra(int32_t resource_index,
                                bool *declared = nullptr) const;
  // The row a kernel implements, always populated. `declared_algebra` publishes
  // it only when `algebra_is_exact`; node attributes read it unconditionally,
  // because what escapes the language is the extension rule, not the per-node
  // quantities.
  ResourceSpec published_algebra(int32_t resource_index) const;
  bool algebra_is_exact(int32_t resource_index) const;
  // Bounds and clamp resolved at a node; the scalar fields are the fallback.
  static float spec_lower(const ResourceSpec &spec, int32_t node);
  static float spec_upper(const ResourceSpec &spec, int32_t node);
  // The row's join operand at a node, or the semiring identity where the row
  // declares none -- so an operand-free row is inert under every semiring and
  // callers need no emptiness test of their own.
  // Derive the execution terms from a published row's named fields.
  std::vector<ResourceTerm> build_terms(const ResourceSpec &spec) const;
  // Value a term contributes on this transition, coefficient applied.
  float term_value(const ResourceTerm &term, int32_t from, int32_t to) const;
  // Node-sourced accumulation charged at one node.
  double node_term_total(const ResourceSpec &spec, int32_t node) const;
  // Summed contribution of additive terms at one phase.
  double phase_total(const ResourceSpec &spec, TermPhase phase, int32_t from,
                     int32_t to) const;
  void apply_add_terms(const ResourceSpec &spec, TermPhase phase, int32_t from,
                       int32_t to, float &value) const;
  static bool has_operation(const ResourceSpec &spec,
                            TermOperation operation);
  bool term_event_matches(const ResourceTerm &term, int32_t from, int32_t to,
                          bool depot, bool bound_failed) const;
  // Whether a node counts toward `spec`'s reset guard, and whether it counts
  // toward the opposite sign.
  static bool guarded_node(const ResourceSpec &spec, int32_t node);
  static bool opposing_node(const ResourceSpec &spec, int32_t node);
  // Per-row count of guarded nodes not yet served, as of a route's start.
  std::vector<int32_t> initial_reset_guards() const;
  // Decrement every row's guard count for a node just served. The raw-vector
  // form is what the three stateless route replays use; the State form
  // delegates to it, so construction and replay cannot drift.
  void consume_reset_guards(std::vector<int32_t> &remaining,
                            int32_t node) const;
  static float spec_join_operand(const ResourceSpec &spec, int32_t node);
  static float join_identity(const ResourceSpec &spec);
  // (+) of the row's semiring. ARITHMETIC has none and returns the accumulated
  // value unchanged.
  static float join(const ResourceSpec &spec, float accumulated, float operand);
  // Whether the row's join is a real operation. Tropical rows lose order
  // invariance, which the search capabilities and the pressure model both key
  // off; this replaces the old `op == AFFINE_MAX` test.
  static bool tropical(const ResourceSpec &spec);
  // Search capabilities implied by a row's algebra, replacing the hand-written
  // per-kernel capability table for every row the language can express.
  static uint32_t derived_capabilities(const ResourceSpec &spec);
  // Whether a declared precedence row admits `next` from the current state, and
  // whether the route may close. Precedence carries no accumulated quantity, so
  // it does not travel through extend_declared; its per-row state is the count
  // of unresolved obligations, held in the same resource_state slot.
  // The node that must precede `next` under this row, or -1 for none. Callers
  // resolve "has it been served" from whatever record they keep -- a visited
  // array during construction, a trial-local stamp inside SRR.
  static int32_t precedence_required(const ResourceSpec &spec, int32_t next);
  static bool precedence_admits(const ResourceSpec &spec, int32_t next,
                                bool depot, float state_value,
                                bool predecessor_served);
  static float precedence_next_state(const ResourceSpec &spec, int32_t next,
                                     bool depot, float state_value);
  // Net change in unresolved pairwise obligations at `node`, over the compiled
  // pickup-delivery relation and every declared pairwise precedence row.
  int32_t open_relation_delta(int32_t node) const;
  // The node paired with `node` under whichever active row declares a pairwise
  // relation, or -1. `node_is_predecessor` reports which end `node` is. Generic
  // operators move a pair together through this rather than naming pickups and
  // deliveries.
  int32_t relation_partner(int32_t node,
                           bool *node_is_predecessor = nullptr) const;
  int32_t multiplier_count() const { return resource_count() + 1; }
  int32_t objective_multiplier() const { return resource_count(); }
  int32_t live_state_feature_count() const { return resource_count(); }
  const std::vector<ResourceSpec> &resources() const { return resources_; }
  const std::vector<const ConstraintKernelSpec *> &active_constraint_kernels()
      const {
    return active_constraint_kernels_;
  }
  const std::vector<float> &resource_row_properties() const {
    return resource_row_properties_;
  }
  const std::vector<float> &resource_term_properties() const {
    return resource_term_properties_;
  }
  const std::vector<int32_t> &resource_term_counts() const {
    return resource_term_counts_;
  }
  const std::vector<float> &objective_edge_costs() const {
    return objective_edge_costs_;
  }
  std::vector<float> resource_scales() const;
  // Bounded per-graph magnitude of the objective edge cost relative to the
  // distance scale, in [0, 1). Gives the field the raw objective scale that
  // per-edge normalization hides, so the multiplier can calibrate its units.
  float objective_scale() const;
  // Row-centered graph scale used to make objective energy dimensionless.
  // Unlike objective_scale(), this is an energy-unit conversion, not a model
  // feature.
  float objective_energy_scale() const { return objective_energy_scale_; }
  const Solution &best_solution() const { return best_solution_; }
  // Whether the cost metric is symmetric, and by how much it is not. Model
  // conditioning, so unlike reversal_safe() these are public.
  bool metric_symmetric() const;
  float metric_skew() const;

private:
  struct RolloutTrace {
    std::vector<int32_t> current_nodes;
    std::vector<int32_t> valid_offsets = {0};
    std::vector<int32_t> valid_indices;
    std::vector<int32_t> chosen_indices;
    std::vector<uint8_t> stochastic;
    std::vector<float> log_probabilities;
    std::vector<float> live_state;
    std::vector<int32_t> screened_edges;
    std::vector<float> screened_resource_delta;
    int64_t screening_fast_evaluations = 0;
    int64_t screening_fallback_evaluations = 0;
    int64_t screening_verification_failures = 0;
    std::vector<int64_t> screening_verification_failures_by_channel;
  };

  struct OrderedChoice {
    int32_t node = -1;
    int32_t local_index = -1;
    double log_weight = 0.0;
  };

  struct State {
    std::vector<int32_t> route;
    std::vector<uint8_t> visited;
    int32_t current = -1;
    int32_t route_depot = -1;
    int32_t start_node = -1;
    int32_t visited_customers = 0;
    int32_t unvisited_linehauls = 0;
    int32_t unvisited_backhauls = 0;
    bool at_depot = false;
    float distance = 0.0f;
    // One live value per registry row, plus a scratch tail. A compiled kernel
    // used to reach its state through a named accessor keyed on a constant
    // FieldChannel ordinal -- load() was resource_state[CAPACITY] -- which is
    // what forced the registry to keep every field channel at a fixed position
    // whether the problem declared it or not. Slots are now addressed by
    // registry index, resolved through RoutingDecoder::channel_slot, so a row's
    // position is a property of the registry.
    //
    // The construction path advances every compiled quantity unconditionally
    // (a running clock, a running load) without first asking whether the
    // problem constrains it. A channel with no row is therefore given a private
    // scratch slot past the end of the registry: the write lands somewhere
    // harmless, nothing reads it, resource_count() does not count it, and it is
    // never exported. One scratch slot per channel rather than one shared, so
    // two absent kernels never alias.
    std::vector<float> resource_state =
        std::vector<float>(FIELD_CHANNEL_COUNT, 0.0f);
    // Two counts per row, at 2*index and 2*index+1: unvisited nodes matching
    // that row's guard sign, and unvisited nodes matching the opposite sign.
    // Both are needed because the alternative reset applies only while the
    // remaining work is all of the opposite kind -- a route with nothing left
    // to serve takes the default, since its reload is immaterial. Maintained
    // incrementally on visit rather than rescanned, because a reset is
    // evaluated on the transition path.
    std::vector<int32_t> reset_guard_remaining;
    // Parallel to resource_state, only meaningful for accumulators that carry
    // interior optional resets (a driver break): the drive accumulated since the
    // last rest-eligible node was passed. It lets a break be placed retroactively
    // at the latest rest node (minimal-break scheduling) instead of eagerly at
    // every one. For every other resource it simply tracks resource_state.
    std::vector<float> resource_since_rest =
        std::vector<float>(FIELD_CHANNEL_COUNT, 0.0f);
    int32_t off_graph_edges = 0;
  };

  Problem problem_;
  std::vector<const ConstraintKernelSpec *> active_constraint_kernels_;
  uint32_t active_kernel_capabilities_ = 0;
  std::array<uint8_t, FIELD_CHANNEL_COUNT> active_field_channels_{};
  std::vector<ResourceSpec> resources_;
  std::vector<int32_t> active_resource_indices_;
  // Rows the INTERPRETER advances, split by algebra. A row a compiled kernel
  // executes is excluded: its running scalar is maintained by that kernel, and
  // advancing it twice would double-count. Both lists therefore hold every
  // active row when interpret_all_resources is set.
  std::vector<int32_t> scalar_resource_indices_;
  std::vector<int32_t> precedence_resource_indices_;
  // Whether each row's compiled kernel is exactly what the row declares. The
  // published copy that used to sit beside this is gone: the row itself is the
  // declaration, so there was nothing left to cache.
  std::vector<uint8_t> declared_specs_valid_;

  std::array<int32_t, FIELD_CHANNEL_COUNT> field_resource_index_{};
  // True when every active registry row is backed by a compiled field-channel
  // kernel, so the cached-route-summary screening path describes the whole
  // registry. One declared row clears it and screening falls back to exact
  // replay, which is the only way the fast path can stay a fast path rather
  // than become a different, narrower semantics.
  bool planned_screening_covers_registry_ = true;
  // Active rows no compiled kernel backs. The certificate and screening paths
  // used to find these by index -- everything at or past FIELD_CHANNEL_COUNT --
  // which only worked while the registry opened with one row per field channel.
  std::vector<int32_t> declared_resource_indices_;
  CandidateConfig candidate_config_;
  SearchConfig search_config_;
  int32_t n_rollouts_;
  float beta_;
  uint64_t seed_ = 1;
  uint64_t generation_ = 0;
  uint64_t graph_version_ = 0;
  bool reversal_safe_ = false;
  // Whether the cost metric itself is symmetric, kept separate from
  // reversal_safe_: that predicate also folds in KERNEL_REVERSAL_SENSITIVE, so
  // a symmetric time-window instance is reversal-unsafe while its metric is
  // still symmetric. Only the metric belongs in the model's conditioning.
  bool metric_symmetric_ = true;
  // Mean |d(i,j) - d(j,i)| over ordered pairs, normalized by distance_scale_.
  // Zero for every coordinate-derived instance.
  float metric_skew_ = 0.0f;
  float distance_scale_ = 1.0f;
  float time_scale_ = 1.0f;
  float prize_scale_ = 1.0f;
  float penalty_scale_ = 1.0f;
  float objective_energy_scale_ = 1.0f;
  int32_t pair_count_ = 0;

  std::vector<float> edge_features_;
  std::vector<float> node_features_;
  std::vector<float> resource_features_;
  std::vector<float> resource_pressure_;
  std::vector<float> resource_events_;
  std::vector<float> node_resource_features_;
  std::vector<float> resource_row_properties_;
  std::vector<float> resource_term_properties_;
  std::vector<int32_t> resource_term_counts_;
  std::vector<float> objective_edge_costs_;
  std::vector<float> incumbent_live_state_;
  // Per-resource accumulation over the REMAINING route, the reverse-pass
  // counterpart of incumbent_live_state_. Route-scoped rows reset at depots, so
  // each node reports what its own route still has to spend.
  std::vector<float> incumbent_suffix_state_;
  // [node_count, resource_count, RESOURCE_SUFFIX_FEATURE_COUNT].
  std::vector<float> incumbent_suffix_features_;
  // [edge_count, resource_count, RESOURCE_TRANSITION_FEATURE_COUNT] plus a
  // matching [edge_count, resource_count] validity mask. Sources absent from
  // the incumbent have no authoritative prefix state and remain masked.
  std::vector<float> incumbent_transition_features_;
  std::vector<uint8_t> incumbent_transition_feature_mask_;
  std::vector<float> node_objective_features_;
  // Unified pairwise-relation index, built once from every active row's
  // PUBLISHED declaration -- the compiled pickup-delivery kernel included, via
  // the PRECEDENCE spec it publishes. One source of truth, so no consumer needs
  // a branch on which path executes the relation.
  std::vector<int32_t> relation_predecessor_;
  std::vector<int32_t> relation_successor_;
  std::vector<int32_t> edge_offsets_;
  std::vector<int32_t> edge_to_;
  std::vector<int32_t> incumbent_route_;
  Solution best_solution_;

  void build_candidate_graph(const std::vector<int32_t> &incumbent,
                             std::vector<float> *edge_field = nullptr,
                             std::vector<float> *edge_additive = nullptr,
                             std::vector<float> *edge_state_field = nullptr,
                             std::vector<float> *objective_residual = nullptr,
                             std::vector<float> *coupler_weights = nullptr,
                             std::vector<float> *coupler_bias = nullptr);
  std::vector<int32_t> rank_by_distance(int32_t from, int32_t limit) const;
  float objective_edge_cost(int32_t from, int32_t to) const;
  float resource_scale(int32_t channel) const;
  float runtime_resource_scale(int32_t resource) const;
  void refresh_objective_energy_scale();
  float analytic_resource_pressure(int32_t from, int32_t to,
                                   int32_t channel) const;
  float runtime_resource_pressure(int32_t from, int32_t to,
                                  int32_t resource) const;
  float compiled_resource_pressure(int32_t from, int32_t to,
                                   int32_t resource) const;
  float resource_pressure_of(const ResourceSpec &spec, int32_t from,
                             int32_t to) const;
  float compiled_pressure_of(const ResourceSpec &spec, int32_t from,
                             int32_t to) const;
  double resource_field_value(int32_t from, int32_t to, int32_t edge,
                              int32_t channel, const float *edge_field,
                              const float *edge_additive) const;
  void build_model_features();
  void build_incumbent_suffix_state();
  void build_relation_index();
  bool field_channel_active(int32_t channel) const;
  int32_t field_resource_index(FieldChannel channel) const;
  // Slot in State::resource_state backing a compiled kernel's live state: its
  // registry row when the problem declares the kernel, otherwise that channel's
  // private scratch slot past the end of the registry.
  int32_t channel_slot(FieldChannel channel) const;
  // Which kernel executes a row; FastPath::NONE means the interpreter does.
  // The single point where that policy is read, so a future switch that
  // interprets a row a kernel could have run has one place to change.
  FastPath fast_path(const ResourceSpec &spec) const;
  FastPath fast_path(int32_t resource_index) const;
  // Rewrite a compiled kernel row into the declarative language it implements.
  ResourceSpec publish(const ResourceSpec &row) const;
  float &slot(State &state, FieldChannel channel) const;
  float slot(const State &state, FieldChannel channel) const;
  // Registry rows plus the per-channel scratch tail.
  int32_t state_slot_count() const {
    return resource_count() + FIELD_CHANNEL_COUNT;
  }
  const ResourceSpec &resource(int32_t index) const;
  void build_resource_registry();
  void build_constraint_kernel_set();
  void build_resource_properties();
  float return_horizon_slack(const ResourceSpec &spec, int32_t next,
                             int32_t depot, float arrival_value,
                             const std::vector<int32_t> &remaining,
                             int32_t resource_index) const;
  // The single definition of a declared row's extension. Construction masks,
  // the SRR trial validator, incumbent evaluation, and the resource report all
  // call it, so a new operator primitive cannot be wired into one replay path
  // and silently missed by the others. `route_depot < 0` disables the return
  // horizon for callers that do not track which depot the route started from.
  bool extend_declared(const ResourceSpec &spec, int32_t from, int32_t next,
                       int32_t route_depot, bool force_route_end,
                       const std::vector<int32_t> &remaining,
                       int32_t resource_index, float &value, float &rest,
                       bool *break_taken = nullptr,
                       float *bounded_value = nullptr,
                       float *admissibility_margin = nullptr) const;
  float resource_state_feature(const State &state, int32_t resource) const;
  bool resource_transition_feasible(const State &state, int32_t next,
                                    int32_t resource,
                                    float *next_value = nullptr,
                                    bool force_route_end = false,
                                    bool *optional_reset_taken = nullptr,
                                    float *next_since_rest = nullptr) const;
  float transition_break_duration(const State &state, int32_t next) const;
  bool resource_terminal_feasible(const State &state, int32_t resource) const;
  bool construction_return_reachable(const State &state, int32_t next) const;
  void validate_guidance(const float *edge_field,
                         const float *edge_additive,
                         const float *edge_state_field,
                         const float *multipliers,
                         const float *coupler_weights,
                         const float *coupler_bias,
                         const float *objective_residual) const;
  std::vector<float> live_state_features(const State &state) const;
  std::vector<float> incumbent_state_features(int32_t current) const;
  bool incumbent_prefix_state(int32_t current, State &state) const;
  // The graph-level half of the learned live-state response: one bounded gain
  // per channel, evaluated at whatever live state the caller supplies.
  double coupled_multiplier(int32_t channel, const float *multipliers,
                            const float *coupler_weights,
                            const float *coupler_bias,
                            const float *live_state) const;
  // The per-edge half, linear in the live state so it survives the SRR
  // aggregate. [edge_count, resource_count, live_state_feature_count].
  double resource_state_field_value(int32_t edge, int32_t channel,
                                    const float *edge_state_field,
                                    const float *live_state) const;
  // live_state is passed by reference, not as a pointer: an empty registry
  // makes vector::data() null, and the null check this used to perform then
  // silently dropped every decision on a problem that declares no constraint.
  void record_decision(RolloutTrace *trace, int32_t current,
                       const std::vector<int32_t> &valid_indices,
                       int32_t chosen_index, bool stochastic,
                       float log_probability,
                       const std::vector<float> &live_state) const;
  double field_score(int32_t from, int32_t to, int32_t edge,
                     const float *edge_field,
                     const float *edge_additive,
                     const float *edge_state_field,
                     const float *multipliers,
                     const float *coupler_weights = nullptr,
                     const float *coupler_bias = nullptr,
                     const float *live_state = nullptr) const;
  double edge_energy(int32_t from, int32_t to, int32_t edge,
                     const float *edge_field,
                     const float *edge_additive,
                     const float *edge_state_field,
                     const float *multipliers,
                     const float *coupler_weights = nullptr,
                     const float *coupler_bias = nullptr,
                     const float *live_state = nullptr,
                     const float *objective_residual = nullptr) const;
  int32_t find_edge(int32_t from, int32_t to) const;
  State initial_state(int32_t start_node) const;
  float depot_reload(const State &state) const;
  // Load a route starts with, given the customer it opens with. The benchmark's
  // backhaul-and-priority rule (URS UniVRPEnv.py:517) lets a route that opens on
  // a backhaul start empty, independently of whether linehauls remain elsewhere;
  // depot_reload alone only covers the case where none do.
  float opening_load(const State &state, int32_t next) const;
  // Whether some active row orders node classes within a route -- the compiled
  // backhaul kernel or a declared class_order precedence row. The
  // opens-on-a-backhaul reload is a `bp` rule, so it must key on the ordering
  // being enforced, not on which path enforces it.
  bool class_ordered() const;
  bool legal_node(const State &state, int32_t node) const;
  std::vector<uint8_t> legal_mask(const State &state) const;
  bool transition(State &state, int32_t next, std::string &error) const;
  bool has_feasible_lookahead(State &state, int32_t depth) const;
  bool feasible_after_lookahead_transition(State &state, int32_t next,
                                           int32_t depth) const;
  bool complete(const State &state) const;
  Solution finish(State state) const;
  Solution construct(uint64_t rollout_seed, const float *edge_field,
                     const float *edge_additive,
                     const float *edge_state_field,
                     const float *multipliers,
                     const float *coupler_weights,
                     const float *coupler_bias, const float *objective_residual, RolloutTrace *trace,
                     bool greedy = false) const;
  Solution perturb(uint64_t rollout_seed, const float *edge_field,
                   const float *edge_additive,
                   const float *edge_state_field,
                   const float *multipliers,
                   const float *coupler_weights, const float *coupler_bias,
                   const float *objective_residual, RolloutTrace *trace,
                   bool greedy = false) const;
  Solution
  scope_restricted_refine(Solution solution,
                          const std::vector<int32_t> &initial_scope,
                          const float *edge_field,
                          const float *edge_additive,
                          const float *edge_state_field,
                          const float *multipliers,
                          const float *coupler_weights,
                          const float *coupler_bias,
                          const float *objective_residual,
                          RolloutTrace *trace, std::mt19937_64 &rng) const;
  int32_t select_next(State &state, std::mt19937_64 &rng,
                      const float *edge_field,
                      const float *edge_additive,
                      const float *edge_state_field,
                      const float *multipliers,
                      const float *coupler_weights,
                      const float *coupler_bias, const float *objective_residual, RolloutTrace *trace,
                      bool greedy = false) const;
  std::vector<OrderedChoice>
  perturbation_order(int32_t current, const std::vector<uint8_t> &used,
                     std::mt19937_64 &rng, const float *edge_field,
                     const float *edge_additive,
                     const float *edge_state_field,
                     const float *multipliers,
                     const float *coupler_weights, const float *coupler_bias,
                     const float *objective_residual,
                     bool greedy = false) const;
  std::vector<int32_t> changed_scope(const std::vector<int32_t> &source,
                                     const std::vector<int32_t> &candidate,
                                     int32_t *new_edge_count = nullptr) const;
  bool reversal_safe() const;
  // Some active row ties two nodes together, so generic operators must not cut a
  // route between them. This is the consumer of KERNEL_RELATIONAL: it holds for
  // the compiled pickup-delivery kernel and for any declared pairwise
  // precedence row, without either being named.
  bool relational() const;
  bool better(const Solution &lhs, const Solution &rhs) const;
};

std::vector<std::string> constraint_names(uint32_t constraints);
std::vector<std::string> candidate_feature_names();
std::vector<std::string> node_feature_names();
std::vector<std::string> field_channel_names();
// Name of the compiled kernel a FastPath selects; "" for FastPath::NONE.
const char *fast_path_name(FastPath fast_path);

} // namespace prism
