#!/usr/bin/env python3
"""Supply the approved GitHub account credential only to Git's GitHub prompt."""

import os
import sys

prompt = sys.argv[1] if len(sys.argv) == 2 else ""
if "github.com" not in prompt:
    raise SystemExit("Only GitHub credential prompts are supported")
if "Username" in prompt:
    print("x-access-token")
elif "Password" in prompt:
    print(os.environ["GITHUB_TOKEN"])
else:
    raise SystemExit("Unrecognized Git credential prompt")
