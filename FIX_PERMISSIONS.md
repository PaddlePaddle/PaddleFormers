# Fix Claude Path Permissions

This PR fixes the permission issue by using /usr/local/bin/claude instead of /root path.

Changes:
- Removed unsupported `anthropic_base_url` parameter
- Updated Claude path to /usr/local/bin/claude
- Added debug step to verify user and permissions

Date: 2026-04-22

