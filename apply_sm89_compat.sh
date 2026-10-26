#!/usr/bin/env bash
# Target-specific compatibility patch for the pinned RTX 4090 / SM89 build.
set -Eeuo pipefail
source "$HOME/apxinf-runtime/env.sh"
engine="$APX_PROJECT/apxinf"
patch="$APX_RUNTIME/bootstrap/sm89-gdn-compat.patch"
test "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)" = 8.9
test "$(git -C "$engine" rev-parse HEAD)" = 7f37642dd6c5896bca66d7640d9944c2a3fe04b4
if git -C "$engine" apply --reverse --check "$patch" 2>/dev/null; then
  echo SM89_PATCH_ALREADY_APPLIED
else
  git -C "$engine" apply --check "$patch"
  git -C "$engine" apply "$patch"
fi
if ! grep -q '^export APXINF_SM89_GDN_COMPAT=1$' "$APX_RUNTIME/env.sh"; then
  printf '\nexport APXINF_SM89_GDN_COMPAT=1\n' >> "$APX_RUNTIME/env.sh"
fi
echo SM89_COMPAT_READY

