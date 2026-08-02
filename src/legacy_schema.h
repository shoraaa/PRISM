#pragma once

#include "decoder.h"

#include <string>

namespace prism {

// Frozen compatibility defaults for callers of the historical name-only
// pybind contract. Native execution must consume the normalized fields derived
// from this value and must never inspect the legacy name itself.
struct LegacySchemaDefaults {
  std::string name;
  uint32_t constraints = 0;
  Objective objective = Objective::MIN_DISTANCE;
  int32_t depot_count = 1;
  bool multi_route = false;
  bool open_route = false;
  float tour_limit = 4.0f;
};

LegacySchemaDefaults legacy_schema_defaults(std::string name);

} // namespace prism
