#
# A set of enums in gkeyll. They have to match those in the Gkeyll source code.
#

# Identifiers for specific geometry types
gkyl_geometry_id = [
  "GKYL_GEOMETRY_NONE", # No geometry, use Cartesian.
  "GKYL_GEOMETRY_TOKAMAK", # Tokamak Geometry from Efit.
  "GKYL_GEOMETRY_MIRROR", # Mirror Geometry from Efit.
  "GKYL_GEOMETRY_MAPC2P", # General geometry from user provided mapc2p.
  "GKYL_GEOMETRY_FROMFILE", # Geometry from file.
]

def enum_idx_to_key(enum, idx):
  # Given an enum list, return the string corresponding to the index idx
  # provided.
  return enum[idx];


def enum_key_to_idx(enum, key):
  # Given an enum list, return the index of the string key provided.
  return enum.index(key);
