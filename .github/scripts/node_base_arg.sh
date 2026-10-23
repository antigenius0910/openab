#!/usr/bin/env bash
# Print the Node.js version ARG (e.g. NODE22_VERSION) that a Dockerfile stage
# is ultimately built FROM, so CI can verify every Node-based image against its
# pin without a hand-maintained allowlist of agent names.
#
# Usage: node_base_arg.sh <dockerfile> [target]
#        node_base_arg.sh --all <dockerfile>
#   target  stage name to resolve; defaults to the last stage in the file.
#   --all   check every FROM line instead of one stage: print each
#           NODE<n>_VERSION used (one per line), fail on any unpinned node: base.
#
# Output / exit status:
#   prints NODE<n>_VERSION, exit 0  -> stage resolves to node:${NODE<n>_VERSION}-...
#   prints nothing,         exit 0  -> stage is not Node-based
#   error message,          exit 1  -> unpinned node: base, or unknown target
set -euo pipefail

all=0
if [ "${1:-}" = "--all" ]; then all=1; shift; fi
dockerfile=${1:?usage: node_base_arg.sh [--all] <dockerfile> [target]}
target=${2:-}

awk -v target="$target" -v all="$all" '
  toupper($1) == "FROM" {
    image = ""; name = ""
    for (i = 2; i <= NF; i++) {
      if ($i ~ /^--/) continue                      # skip --platform=...
      if (image == "") { image = $i; continue }
      if (toupper($i) == "AS" && i < NF) { name = $(i + 1); break }
    }
    if (name != "") by_name[name] = image
    last_image = image
    if (all) {
      if (match(image, /^node:[$][{]NODE[0-9]+_VERSION[}]-/)) used[substr(image, 8, RLENGTH - 9)] = 1
      else if (image ~ /^node:/) {
        print "::error::" FILENAME ":" FNR " uses unpinned base image " image "; use node:${NODE<major>_VERSION}-..." > "/dev/stderr"
        bad = 1
      }
    }
  }
  END {
    if (all) {
      if (bad) exit 1
      for (arg in used) print arg
      exit 0
    }
    if (target == "") {
      image = last_image
    } else if (target in by_name) {
      image = by_name[target]
    } else {
      print "::error::Stage \"" target "\" not found in " FILENAME > "/dev/stderr"
      exit 1
    }
    # Follow stage-to-stage inheritance (e.g. claude -> base-node -> node:...).
    for (hops = 0; (image in by_name) && hops < 64; hops++) image = by_name[image]
    if (match(image, /^node:[$][{]NODE[0-9]+_VERSION[}]-/)) {
      arg = substr(image, 8, RLENGTH - 9)            # strip "node:${" and "}-"
      print arg
    } else if (image ~ /^node:/) {
      print "::error::" FILENAME " uses unpinned base image " image "; use node:${NODE<major>_VERSION}-..." > "/dev/stderr"
      exit 1
    }
  }
' "$dockerfile"
