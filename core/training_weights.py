"""Shared constants for persisted training-sample weights."""

# Greedy fallback is a completed route, but is a weaker teacher than a route
# found directly by Search3. Historical replay items stored 0.2; the model
# also caps their effective weight so the change applies without rewriting
# saved replay data.
SEARCH3_GREEDY_FALLBACK_WEIGHT = 0.05
