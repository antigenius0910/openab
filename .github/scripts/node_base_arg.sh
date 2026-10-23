#!/usr/bin/env bash
# Single parser for Node.js base images in Dockerfiles. CI uses it both to
# decide which images need a runtime `node --version` check and to enforce that
# every Node.js base is pinned as node:${NODE<major>_VERSION}-<suffix>.
#
# Usage:
#   node_base_arg.sh <dockerfile> [target]
#       Print the NODE<n>_VERSION ARG that <target> (default: the last stage)
#       is ultimately built FROM, or nothing if it is not Node-based.
#   node_base_arg.sh --check <dockerfile>...
#       Validate every stage of every file and print "<file> NODE<n>_VERSION"
#       once per pin each file uses.
#
# Both modes scan every FROM in the file and exit 1 if any stage:
#   - references a Node.js image (node:, node@, library/node, docker.io/library/node)
#     not written as node:${NODE<n>_VERSION}-<suffix>; or
#   - takes its base from a build arg (FROM ${VAR}) other than an allowlisted one,
#     since that can hide an unpinned Node.js base.
# The first mode also fails if <target> is not a stage in the file.
# Stage names are matched case-insensitively, as Docker does.
set -euo pipefail

# Build args that may be used directly as a base image. BUILDER_IMAGE selects
# the openab builder in Dockerfile.unified and is never a Node.js runtime base.
ALLOWED_VAR_BASES="BUILDER_IMAGE"

parse() { # parse <mode> <dockerfile> [target]
  awk -v mode="$1" -v target="${3:-}" -v allowed=" $ALLOWED_VAR_BASES " '
    # Classify one FROM image reference. Returns "node <ARG>", "none", or
    # "error <message>". Stage references inherit the referenced stage.
    function classify(img,   key, ref, var) {
      key = tolower(img)
      if (key in stage) return stage[key]
      if (img ~ /^[$][{]?[A-Za-z_][A-Za-z0-9_]*[}]?$/) {
        var = img; gsub(/[${}]/, "", var)
        if (index(allowed, " " var " ")) return "none"
        return "error FROM " img " takes its base image from a build arg, which can hide an unpinned Node.js base; use node:${NODE<major>_VERSION}-<suffix> or a named stage (allowed build args:" allowed ")"
      }
      ref = img
      if (tolower(substr(ref, 1, 10)) == "docker.io/") ref = substr(ref, 11)
      if (tolower(substr(ref, 1, 8)) == "library/") ref = substr(ref, 9)
      if (match(ref, /^node:[$][{]NODE[0-9]+_VERSION[}]-./))
        return "node " substr(ref, 8, index(ref, "}") - 8)   # text between "node:${" and "}"
      if (ref ~ /^node:[$][{]NODE[0-9]+_VERSION[}]$/)
        return "error " img " has no tag suffix; use node:${NODE<major>_VERSION}-<suffix> (e.g. -trixie-slim)"
      if (tolower(ref) ~ /^node([:@]|$)/)
        return "error " img " is not pinned; use node:${NODE<major>_VERSION}-<suffix>"
      return "none"
    }
    toupper($1) == "FROM" {
      img = ""; name = ""
      for (i = 2; i <= NF; i++) {
        if ($i ~ /^--/) continue                       # skip --platform=... and other flags
        if (img == "") { img = $i; continue }
        if (toupper($i) == "AS" && i < NF) { name = tolower($(i + 1)); break }
      }
      class = classify(img)
      if (class ~ /^error / && !(tolower(img) in stage)) {   # report each bad base once, not per child stage
        print "::error file=" FILENAME ",line=" FNR "::" substr(class, 7) > "/dev/stderr"
        failed = 1
      }
      if (class ~ /^node /) used[substr(class, 6)] = 1
      if (name != "") stage[name] = class
      last = class
    }
    END {
      if (failed) exit 1
      if (mode == "check") {
        for (arg in used) print FILENAME, arg
        exit 0
      }
      if (target == "") {
        class = last
      } else if (tolower(target) in stage) {
        class = stage[tolower(target)]
      } else {
        print "::error::Stage \"" target "\" not found in " FILENAME > "/dev/stderr"
        exit 1
      }
      if (class ~ /^node /) print substr(class, 6)
    }
  ' "$2"
}

if [ "${1:-}" = "--check" ]; then
  shift
  [ "$#" -gt 0 ] || { echo "usage: node_base_arg.sh --check <dockerfile>..." >&2; exit 2; }
  rc=0
  for f in "$@"; do parse check "$f" || rc=1; done   # report every bad file, then fail
  exit "$rc"
fi

[ "$#" -ge 1 ] && [ "$#" -le 2 ] || { echo "usage: node_base_arg.sh <dockerfile> [target]" >&2; exit 2; }
parse resolve "$1" "${2:-}"
