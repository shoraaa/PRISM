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
static constexpr int32_t RESOURCE_KERNEL_COUNT = 8;
static constexpr int32_t LIVE_STATE_FEATURE_COUNT = FIELD_CHANNEL_COUNT;
static constexpr int32_t NODE_FEATURE_COUNT = 24;
static constexpr int32_t EDGE_FEATURE_COUNT = 11;
static constexpr int32_t RESOURCE_DESCRIPTOR_DIM = 32;
// One multiplier slot beyond the resource channels carries the objective
// weight applied to the objective edge cost. ConstraintFieldNet fixes this slot
// to one (and its coupler to zero). A signed learned objective residual is
// added to the canonical edge cost inside the same energy formula.
static constexpr int32_t OBJECTIVE_MULTIPLIER = FIELD_CHANNEL_COUNT;
static constexpr int32_t MULTIPLIER_COUNT = FIELD_CHANNEL_COUNT + 1;

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
  AFFINE_ACCUMULATOR,
};

enum class ResourceDirection : uint8_t { FORWARD, BACKWARD, BIDIRECTIONAL };
enum class ResourceScope : uint8_t { ROUTE, TOUR, SOLUTION };
enum class BoundCheck : uint8_t { TRANSITION, ROUTE_END, SOLUTION_END };
enum class VisitPolicy : uint8_t {
  DEPOT,
  REQUIRED_ONCE,
  OPTIONAL_ONCE,
  OPTIONAL_REPEATABLE,
};
enum class EventEffectOperator : uint8_t { ADD, ASSIGN };

// Declarative node-arrival event program. Expressions are affine in live
// scalar resources; names are resolved to dense resource indices once the
// decoder registry has been built. The reserved target `route_time` lets an
// event contribute state-dependent duration to the compiled time-window
// kernel without making charging (or breaks/reloads) a named special case.
struct EventExpressionTerm {
  std::string resource;
  float coefficient = 0.0f;
};

struct EventExpression {
  float constant = 0.0f;
  std::vector<EventExpressionTerm> terms;
};

struct NodeEventGuardSpec {
  std::string resource;
  float lower = -std::numeric_limits<float>::infinity();
  float upper = std::numeric_limits<float>::infinity();
};

struct NodeEventEffectSpec {
  EventEffectOperator op = EventEffectOperator::ADD;
  std::string target;
  EventExpression expression;
};

struct NodeEventSpec {
  std::string name;
  std::vector<uint8_t> trigger_nodes;
  std::vector<NodeEventGuardSpec> guards;
  std::vector<NodeEventEffectSpec> effects;
};

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
  ResourceDirection direction = ResourceDirection::FORWARD;
  ResourceScope scope = ResourceScope::ROUTE;
  BoundCheck bound_check = BoundCheck::TRANSITION;
  int32_t state_dim = 1;
  float initial = 0.0f;
  float scale = 1.0f;
  float lower = -std::numeric_limits<float>::infinity();
  float upper = std::numeric_limits<float>::infinity();
  float edge_coefficient = 0.0f;
  float node_coefficient = 0.0f;
  bool edge_uses_distance = false;
  bool reset_at_depot = false;
  float reset_value = 0.0f;
  std::vector<float> edge_values;
  std::vector<float> node_values;
  std::vector<uint8_t> reset_nodes;

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
  // Defaults to DEPOT for depot indices and REQUIRED_ONCE elsewhere. Support
  // facilities can instead be OPTIONAL_ONCE or OPTIONAL_REPEATABLE.
  std::vector<VisitPolicy> visit_policy;
  std::vector<NodeEventSpec> node_events;

  bool has(Constraint constraint) const;
  bool has_capability(ConstraintCapability capability) const;
  float dist(int32_t from, int32_t to) const;
  int32_t customer_count() const;
  int32_t required_count() const;
  bool is_service_node(int32_t node) const;
  bool is_required_node(int32_t node) const;
  bool is_repeatable_node(int32_t node) const;
  bool has_repeatable_nodes() const;
  void validate() const;
};

// SCHEMA (default) builds the neighborhood from the schema-derived
// candidate_resource_relevance, so any declared resource gets coverage without
// per-variant tuning; when no learned quota is installed it uses a uniform
// equal-share prior over active resources. GEOMETRIC is an explicit ablation
// that drops every resource channel and keeps only the distance neighborhood.
enum class CandidateMode : uint8_t { SCHEMA, GEOMETRIC };

struct CandidateConfig {
  int32_t max_candidates = 64;
  CandidateMode candidate_mode = CandidateMode::SCHEMA;

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
  // return worse. Constant energy has no gradient, so
  // no exploration move ever qualifies -- the baseline is strictly unaffected.
  // 0 disables the phase (default), preserving the pure monotone hill-climb.
  int32_t srr_exploration_budget = 0;
  // A candidate qualifies for exploration only if it lowers the anchor guided
  // energy by more than this margin, so numerical ties never trigger a kick.
  float srr_exploration_margin = 1.0e-6f;
  // Unified feasible-move-plus-STOP SRR policy. Positive beta samples from
  // softmax(-beta * guided_energy) and emits replayable action traces; beta
  // zero selects greedily with STOP winning ties for deterministic inference.
  bool srr_policy_enabled = false;
  // Maximum accepted moves made by the unified SRR policy in one refinement
  // invocation.  This is a generic finite horizon: improving and worsening
  // moves consume the same budget, and the returned champion remains safe.
  int32_t srr_policy_horizon = 64;
  float srr_policy_beta = 0.0f;

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
  std::vector<int32_t> feasibility_edges;
  // Aligned with feasibility_edges.
  std::vector<float> feasibility_risk_labels;
  std::vector<int32_t> screened_edges;
  std::vector<float> screened_resource_delta;
  // Nested unified SRR policy trace: rollout -> decisions -> candidates ->
  // signed changed-edge terms.  Every decision includes STOP as a zero-delta
  // candidate.
  std::vector<int32_t> srr_starts;
  std::vector<int32_t> srr_candidate_offsets;
  std::vector<int32_t> srr_edge_offsets;
  std::vector<int32_t> srr_edges;
  std::vector<int8_t> srr_edge_signs;
  std::vector<float> srr_candidate_objective_delta;
  std::vector<float> srr_candidate_energy_delta;
  std::vector<float> srr_candidate_constant_delta;
  std::vector<int32_t> srr_chosen_indices;
  std::vector<float> srr_log_probabilities;
  std::vector<float> srr_live_state;
  std::vector<uint8_t> srr_stop;
  int64_t screening_fast_evaluations = 0;
  int64_t screening_fallback_evaluations = 0;
  int64_t screening_verification_failures = 0;
  std::array<int64_t, FIELD_CHANNEL_COUNT>
      screening_verification_failures_by_channel{};
};

class RoutingDecoder {
public:
  RoutingDecoder(Problem problem, CandidateConfig candidate_config = {},
             SearchConfig search_config = {}, int32_t n_rollouts = 20,
             float beta = 2.0f);

  void seed(uint64_t value);
  std::vector<Solution> sample(const float *edge_field = nullptr,
                               const float *edge_additive = nullptr,
                               const float *multipliers = nullptr,
                               const float *coupler_weights = nullptr,
                               const float *coupler_bias = nullptr,
                               const float *objective_residual = nullptr,
                               const float *edge_risk = nullptr,
                               float risk_penalty = 0.0f,
                               DecisionTrace *trace = nullptr);
  Solution sample_greedy(const float *edge_field = nullptr,
                         const float *edge_additive = nullptr,
                         const float *multipliers = nullptr,
                         const float *coupler_weights = nullptr,
                         const float *coupler_bias = nullptr,
                         const float *objective_residual = nullptr,
                         const float *edge_risk = nullptr,
                         float risk_penalty = 0.0f) const;
  Solution solve(int32_t iterations, const float *edge_field = nullptr,
                 const float *edge_additive = nullptr,
                 const float *multipliers = nullptr,
                 const float *coupler_weights = nullptr,
                 const float *coupler_bias = nullptr,
                 const float *objective_residual = nullptr,
                 const float *edge_risk = nullptr,
                 float risk_penalty = 0.0f);
  Solution evaluate(const std::vector<int32_t> &route) const;
  ResourceEvaluation
  evaluate_resources(const std::vector<int32_t> &route) const;
  void set_incumbent(const std::vector<int32_t> &route);
  void set_candidate_resource_quotas(const std::vector<float> &quotas);
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
  const std::vector<float> &resource_features() const {
    return resource_features_;
  }
  const std::vector<float> &resource_pressure() const {
    return resource_pressure_;
  }
  const std::vector<float> &resource_events() const {
    return resource_events_;
  }
  int32_t resource_count() const {
    return static_cast<int32_t>(resources_.size());
  }
  int32_t multiplier_count() const { return resource_count() + 1; }
  int32_t objective_multiplier() const { return resource_count(); }
  int32_t live_state_feature_count() const { return resource_count(); }
  const std::vector<ResourceSpec> &resources() const { return resources_; }
  const std::vector<const ConstraintKernelSpec *> &active_constraint_kernels()
      const {
    return active_constraint_kernels_;
  }
  const std::vector<float> &resource_descriptors() const {
    return resource_descriptors_;
  }
  const std::vector<float> &candidate_resource_quotas() const {
    return candidate_resource_quotas_;
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

private:
  struct RolloutTrace {
    std::vector<int32_t> current_nodes;
    std::vector<int32_t> valid_offsets = {0};
    std::vector<int32_t> valid_indices;
    std::vector<int32_t> chosen_indices;
    std::vector<uint8_t> stochastic;
    std::vector<float> log_probabilities;
    std::vector<float> live_state;
    std::vector<int32_t> feasibility_edges;
    std::vector<float> feasibility_risk_labels;
    std::vector<int32_t> screened_edges;
    std::vector<float> screened_resource_delta;
    std::vector<int32_t> srr_candidate_offsets = {0};
    std::vector<int32_t> srr_edge_offsets = {0};
    std::vector<int32_t> srr_edges;
    std::vector<int8_t> srr_edge_signs;
    std::vector<float> srr_candidate_objective_delta;
    std::vector<float> srr_candidate_energy_delta;
    std::vector<float> srr_candidate_constant_delta;
    std::vector<int32_t> srr_chosen_indices;
    std::vector<float> srr_log_probabilities;
    std::vector<float> srr_live_state;
    std::vector<uint8_t> srr_stop;
    int64_t screening_fast_evaluations = 0;
    int64_t screening_fallback_evaluations = 0;
    int64_t screening_verification_failures = 0;
    std::array<int64_t, FIELD_CHANNEL_COUNT>
        screening_verification_failures_by_channel{};
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
    int32_t visited_required = 0;
    int32_t route_service_visits = 0;
    int32_t open_pickups = 0;
    int32_t unvisited_linehauls = 0;
    int32_t unvisited_backhauls = 0;
    bool at_depot = false;
    bool route_has_backhaul = false;
    float load = 0.0f;
    float route_distance = 0.0f;
    float current_time = 0.0f;
    float distance = 0.0f;
    float collected_prize = 0.0f;
    std::vector<float> resource_state;
    int32_t off_graph_edges = 0;
  };

  Problem problem_;
  std::vector<const ConstraintKernelSpec *> active_constraint_kernels_;
  uint32_t active_kernel_capabilities_ = 0;
  std::array<uint8_t, FIELD_CHANNEL_COUNT> active_field_channels_{};
  std::vector<ResourceSpec> resources_;
  std::vector<int32_t> active_resource_indices_;
  std::vector<int32_t> scalar_resource_indices_;
  struct CompiledEventExpressionTerm {
    int32_t resource = -1;
    float coefficient = 0.0f;
  };
  struct CompiledEventExpression {
    float constant = 0.0f;
    std::vector<CompiledEventExpressionTerm> terms;
  };
  struct CompiledNodeEventGuard {
    int32_t resource = -1;
    float lower = -std::numeric_limits<float>::infinity();
    float upper = std::numeric_limits<float>::infinity();
  };
  struct CompiledNodeEventEffect {
    EventEffectOperator op = EventEffectOperator::ADD;
    // -1 is the reserved built-in route_time target.
    int32_t target = -1;
    CompiledEventExpression expression;
  };
  struct CompiledNodeEvent {
    std::string name;
    std::vector<uint8_t> trigger_nodes;
    std::vector<CompiledNodeEventGuard> guards;
    std::vector<CompiledNodeEventEffect> effects;
  };
  std::vector<CompiledNodeEvent> node_events_;
  std::array<int32_t, FIELD_CHANNEL_COUNT> field_resource_index_{};
  CandidateConfig candidate_config_;
  SearchConfig search_config_;
  int32_t n_rollouts_;
  float beta_;
  uint64_t seed_ = 1;
  uint64_t generation_ = 0;
  uint64_t graph_version_ = 0;
  bool reversal_safe_ = false;
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
  std::vector<float> resource_descriptors_;
  std::vector<float> candidate_resource_quotas_;
  std::vector<float> objective_edge_costs_;
  std::vector<float> incumbent_live_state_;
  std::vector<int32_t> edge_offsets_;
  std::vector<int32_t> edge_to_;
  std::vector<int32_t> incumbent_route_;
  Solution best_solution_;

  void build_candidate_graph(const std::vector<int32_t> &incumbent,
                             std::vector<float> *edge_field = nullptr,
                             std::vector<float> *edge_additive = nullptr,
                             std::vector<float> *objective_residual = nullptr,
                             std::vector<float> *edge_risk = nullptr);
  std::vector<int32_t> rank_by_distance(int32_t from, int32_t limit) const;
  float objective_edge_cost(int32_t from, int32_t to) const;
  float resource_scale(int32_t channel) const;
  float runtime_resource_scale(int32_t resource) const;
  void refresh_objective_energy_scale();
  float analytic_resource_pressure(int32_t from, int32_t to,
                                   int32_t channel) const;
  float runtime_resource_pressure(int32_t from, int32_t to,
                                  int32_t resource) const;
  float candidate_resource_relevance(int32_t from, int32_t to,
                                     int32_t resource) const;
  double resource_field_value(int32_t from, int32_t to, int32_t edge,
                              int32_t channel, const float *edge_field,
                              const float *edge_additive) const;
  void build_model_features();
  bool field_channel_active(int32_t channel) const;
  int32_t field_resource_index(FieldChannel channel) const;
  const ResourceSpec &resource(int32_t index) const;
  void build_resource_registry();
  void build_node_events();
  bool node_event_uses_resource(int32_t node, int32_t resource) const;
  void build_constraint_kernel_set();
  void build_resource_descriptors();
  float resource_state_feature(const State &state, int32_t resource) const;
  bool resource_transition_feasible(const State &state, int32_t next,
                                    int32_t resource,
                                    float *next_value = nullptr,
                                    bool force_route_end = false,
                                    float event_time = 0.0f) const;
  bool scalar_transition(int32_t current, int32_t next,
                         const std::vector<float> &before,
                         std::vector<float> &after, bool force_route_end,
                         float *event_time, std::string *error,
                         std::vector<float> *max_violation = nullptr) const;
  bool resource_terminal_feasible(const State &state, int32_t resource) const;
  void validate_guidance(const float *edge_field,
                         const float *edge_additive,
                         const float *multipliers,
                         const float *coupler_weights,
                         const float *coupler_bias,
                         const float *objective_residual,
                         const float *edge_risk,
                         float risk_penalty) const;
  std::vector<float> live_state_features(const State &state) const;
  std::vector<float> incumbent_state_features(int32_t current) const;
  bool incumbent_prefix_state(int32_t current, State &state) const;
  double coupled_multiplier(int32_t channel, const float *multipliers,
                            const float *coupler_weights,
                            const float *coupler_bias,
                            const float *live_state) const;
  void record_decision(RolloutTrace *trace, int32_t current,
                       const std::vector<int32_t> &valid_indices,
                       int32_t chosen_index, bool stochastic,
                       float log_probability,
                       const float *live_state) const;
  void record_feasibility_labels(RolloutTrace *trace, State &state) const;
  double field_score(int32_t from, int32_t to, int32_t edge,
                     const float *edge_field,
                     const float *edge_additive,
                     const float *multipliers,
                     const float *coupler_weights = nullptr,
                     const float *coupler_bias = nullptr,
                     const float *live_state = nullptr) const;
  double edge_energy(int32_t from, int32_t to, int32_t edge,
                     const float *edge_field,
                     const float *edge_additive,
                     const float *multipliers,
                     const float *coupler_weights = nullptr,
                     const float *coupler_bias = nullptr,
                     const float *live_state = nullptr,
                     const float *objective_residual = nullptr,
                     const float *edge_risk = nullptr,
                     float risk_penalty = 0.0f) const;
  int32_t find_edge(int32_t from, int32_t to) const;
  State initial_state(int32_t start_node) const;
  float depot_reload(const State &state) const;
  bool legal_node(const State &state, int32_t node) const;
  std::vector<uint8_t> legal_mask(const State &state) const;
  bool transition(State &state, int32_t next, std::string &error) const;
  bool has_feasible_lookahead(State &state, int32_t depth) const;
  bool feasible_after_lookahead_transition(State &state, int32_t next,
                                           int32_t depth) const;
  float feasibility_risk_label(State &state, int32_t next) const;
  bool complete(const State &state) const;
  Solution finish(State state) const;
  Solution construct(uint64_t rollout_seed, const float *edge_field,
                     const float *edge_additive,
                     const float *multipliers,
                     const float *coupler_weights,
                     const float *coupler_bias, const float *objective_residual,
                     const float *edge_risk,
                     float risk_penalty, RolloutTrace *trace,
                     bool greedy = false) const;
  Solution perturb(uint64_t rollout_seed, const float *edge_field,
                   const float *edge_additive, const float *multipliers,
                   const float *coupler_weights, const float *coupler_bias,
                   const float *objective_residual, const float *edge_risk,
                   float risk_penalty, RolloutTrace *trace,
                   bool greedy = false) const;
  Solution
  scope_restricted_refine(Solution solution,
                          const std::vector<int32_t> &initial_scope,
                          const float *edge_field,
                          const float *edge_additive,
                          const float *multipliers,
                          const float *coupler_weights,
                          const float *coupler_bias,
                          const float *objective_residual,
                          const float *edge_risk, float risk_penalty,
                          RolloutTrace *trace, std::mt19937_64 &rng) const;
  int32_t select_next(State &state, std::mt19937_64 &rng,
                      const float *edge_field,
                      const float *edge_additive,
                      const float *multipliers,
                      const float *coupler_weights,
                      const float *coupler_bias, const float *objective_residual,
                      const float *edge_risk,
                      float risk_penalty, RolloutTrace *trace,
                      bool greedy = false) const;
  std::vector<OrderedChoice>
  perturbation_order(int32_t current, const std::vector<uint8_t> &used,
                     std::mt19937_64 &rng, const float *edge_field,
                     const float *edge_additive, const float *multipliers,
                     const float *coupler_weights, const float *coupler_bias,
                     const float *objective_residual, const float *edge_risk,
                     float risk_penalty,
                     bool greedy = false) const;
  std::vector<int32_t> changed_scope(const std::vector<int32_t> &source,
                                     const std::vector<int32_t> &candidate,
                                     int32_t *new_edge_count = nullptr) const;
  bool reversal_safe() const;
  bool better(const Solution &lhs, const Solution &rhs) const;
};

std::vector<std::string> constraint_names(uint32_t constraints);
std::vector<std::string> candidate_feature_names();
std::vector<std::string> node_feature_names();
std::vector<std::string> field_channel_names();

} // namespace prism
