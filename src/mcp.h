// `moonglide mcp [--control PATH] [--status PATH]`: an MCP server (JSON-RPC 2.0 over stdio) that reads
// and controls a *running* MoonGlide through its control fifo and status file. No GPU, no window.
#pragma once

int runMcp(int argc, char** argv);
