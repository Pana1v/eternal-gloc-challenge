#!/usr/bin/env bash
# Downloads the prior map into map/prior_map.pcd.
#
# The map is 202 MB, past GitHub's 100 MiB limit on a single tracked file, so it
# ships as a release asset instead of in the tree. Everything else the dev split
# needs (the 40 Track A scenarios and their ground truth) is already in
# scenarios/dev/.
set -euo pipefail

TAG=dev-map-v1
URL="https://github.com/Pana1v/eternal-gloc-challenge/releases/download/$TAG/prior_map.pcd"
SHA256=7d9e86e7f5c38dff2d07ac5847a9d1ce17d04632df70ff09c5139b217f110e58
DEST="$(dirname "$0")/../map/prior_map.pcd"

mkdir -p "$(dirname "$DEST")"

if [ -f "$DEST" ] && echo "$SHA256  $DEST" | sha256sum -c --status; then
    echo "map already present and matches the published checksum"
    exit 0
fi

echo "downloading 202 MB from $TAG ..."
curl -fL --progress-bar -o "$DEST" "$URL"

# A truncated or proxy-mangled download reads as a valid but wrong map, and the
# baselines would score badly for a reason no one would think to look for.
echo "$SHA256  $DEST" | sha256sum -c -
echo "map ready at $DEST"
