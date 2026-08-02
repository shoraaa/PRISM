#include "legacy_schema.h"

#include <algorithm>
#include <cctype>

namespace prism {
namespace {

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

} // namespace

LegacySchemaDefaults legacy_schema_defaults(std::string name) {
  std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });

  LegacySchemaDefaults result;
  result.name = std::move(name);
  if (!is_orienteering(result.name) && !is_pctsp(result.name))
    result.constraints |= VISIT_ALL;
  if (is_vrp(result.name))
    result.constraints |= CAPACITY;
  if (contains(result.name, "bp"))
    result.constraints |= BACKHAUL_ORDER;
  if (contains(result.name, "pd"))
    result.constraints |= PICKUP_DELIVERY;
  if (contains(result.name, "tw"))
    result.constraints |= TIME_WINDOWS;
  if (contains(result.name, "l"))
    result.constraints |= ROUTE_LIMIT;
  if (is_orienteering(result.name))
    result.constraints |= TOUR_LIMIT;
  if (is_pctsp(result.name))
    result.constraints |= PRIZE_QUOTA;

  result.objective = is_orienteering(result.name)
                         ? Objective::MAX_PRIZE
                         : is_pctsp(result.name)
                               ? Objective::MIN_DISTANCE_PLUS_PENALTY
                               : Objective::MIN_DISTANCE;
  const bool no_depot = result.name == "tsp" || result.name == "atsp";
  result.depot_count =
      no_depot ? 0 : (contains(result.name, "md") ? 3 : 1);
  result.multi_route = is_vrp(result.name);
  result.open_route = is_open_vrp(result.name);
  result.tour_limit = result.name == "aop" ? 1.0f : 4.0f;
  return result;
}

} // namespace prism
