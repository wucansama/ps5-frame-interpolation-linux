# MoonGlide MCP server

`moonglide mcp` is a [Model Context Protocol](https://modelcontextprotocol.io) server (JSON-RPC 2.0 over stdio) that
lets Claude Code, Claude Desktop or any MCP client read and control a **running** MoonGlide. It talks to the instance
through the control fifo and the status file (see the README), starts no GPU or window code, needs no network, and
exits when stdin closes.

```sh
moonglide mcp [--control PATH] [--status PATH] [--config PATH]
```

Defaults match the app: control `$XDG_RUNTIME_DIR/moonglide.ctl` (`MOONGLIDE_CTL`), status `/dev/shm/moonglide.status`
(`MOONGLIDE_STATUS`), config `~/.config/moonglide/config.toml` (`MOONGLIDE_CONFIG`). If a second MoonGlide started
with the fifo busy, it uses `<path>.<pid>`; point `--control` / `--status` at those to drive it.
When MoonGlide is not running, tools answer "MoonGlide is not running" instead of hanging.

## Tools

| Tool | What it does |
| --- | --- |
| `get_status` | Live status: fps shown / real, latency, source and Remote Play state, FG mode, drops, errors (summary + full JSON). |
| `get_settings` | Live values from the status plus the saved config file; lists restart-only and live keys. |
| `set_setting {key, value}` | Set one known setting; says `needs_restart` for restart-only keys (pending until `restart`). |
| `command {command}` | One allowlisted control command (`nr`, `intensity`, `overlay ...`, `fg ...`, `hud ...`, `ui`, `fullscreen`, `restart`, `quit`, ...). Anything else is rejected. `quit` stops MoonGlide and `restart` re-executes it. |
| `presets {action, name}` | `list`, `load`, `save` named presets (no delete). |
| `set_frame_generation` | `mode` off/own/rife/lsfg, `multiplier`, `flags`, `hold`, `real_delay`. |
| `set_denoise` | `enabled`, `intensity`, `local_structure`, `local_tone`, `skin_structure`, `clean`, `clean_strength`, `deband`, `deblock`. |
| `set_upscale` | `mode` fsr/bicubic, `sharpen`. |
| `static_zones` | `list`, `add` (rect or ellipse as 0..1 fractions, optional feather), `pop`, `clear`. |
| `hud` | The performance overlay: `on`, `position` (tl/tr/bl/br), `mode` (full/compact), `size` 0..6. |
| `wait_for_stream {timeout_s}` | Wait until frames flow (useful after `restart`). |

Notes: changes apply asynchronously (the status refreshes once a second). The status only reports the number of static
zones; `static_zones list` also shows the zones added through this server. Credential-like fields (Remote Play
registration, PSN account, keys, tokens) are never returned, and no tool can set them.

## Claude Code

```sh
claude mcp add moonglide -- /path/to/moonglide mcp
```

or a project `.mcp.json`:

```json
{
  "mcpServers": {
    "moonglide": { "command": "/path/to/moonglide", "args": ["mcp"] }
  }
}
```

## Claude Desktop

Add to `claude_desktop_config.json` (Linux: `~/.config/Claude/claude_desktop_config.json`), then restart Claude Desktop:

```json
{
  "mcpServers": {
    "moonglide": { "command": "/path/to/moonglide", "args": ["mcp"] }
  }
}
```

Add `"--control", "/path", "--status", "/path"` to `args` to target a non-default instance.

## AppImage

The AppImage passes its arguments through to the binary, so use it as the command:

```sh
claude mcp add moonglide -- /path/to/MoonGlide-x86_64.AppImage mcp
```

(`"command": "/path/to/MoonGlide-x86_64.AppImage", "args": ["mcp"]` in the JSON forms.)
