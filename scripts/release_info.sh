#!/usr/bin/env bash
# Print release facts for CI as key=value lines (append them to $GITHUB_OUTPUT).
#
#   scripts/release_info.sh            # version taken from __version__
#   scripts/release_info.sh v2.3.0     # version given explicitly
#
#   version=2.3.0      the version, without the leading "v"
#   tag=v2.3.0
#   tagged=false       whether refs/tags/<tag> already exists on origin
#   highest=true       whether the version is >= every vX.Y.Z tag on origin,
#                      i.e. whether it may become :latest
set -euo pipefail

INIT_FILE="src/image_deidentification/__init__.py"

version="${1:-}"
if [[ -z "$version" ]]; then
    version="$(sed -n 's/^__version__ = "\(.*\)"$/\1/p' "$INIT_FILE")"
fi
version="${version#v}"

if [[ ! "$version" =~ ^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]]; then
    echo "::error::'$version' is not a MAJOR.MINOR.PATCH version" >&2
    exit 1
fi

tags="$(git ls-remote --tags --refs origin 'v*' | sed -n 's#.*refs/tags/v##p' \
    | grep -E '^[0-9]+\.[0-9]+\.[0-9]+$' || true)"

tagged=false
if grep -qxF "$version" <<< "$tags"; then
    tagged=true
fi

newest="$(printf '%s\n%s\n' "$tags" "$version" | sed '/^$/d' | sort -V | tail -n 1)"
highest=false
if [[ "$newest" == "$version" ]]; then
    highest=true
fi

echo "version=$version"
echo "tag=v$version"
echo "tagged=$tagged"
echo "highest=$highest"
