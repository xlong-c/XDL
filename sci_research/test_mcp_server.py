#!/usr/bin/env python3
"""
MCP Server Test Script
测试 MCP 服务器功能
"""

import sys
import json
from pathlib import Path

# Add sci_research to path
sys.path.insert(0, str(Path(__file__).parent))

from skills.mcp_server import MCPServer
import asyncio


async def test_mcp_server():
    """测试 MCP 服务器"""
    server = MCPServer(verbose=False)
    
    # Test 1: Initialize
    print("📌 Test 1: Initialize")
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"id": 1}
    }
    response = await server.handle_request(request)
    print(f"  Response: {json.dumps(response, indent=2)[:200]}...")
    
    # Test 2: List tools
    print("\n📌 Test 2: List tools")
    request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    }
    response = await server.handle_request(request)
    tools = response.get("result", {}).get("tools", [])
    print(f"  Available tools: {len(tools)}")
    for tool in tools:
        print(f"    - {tool['name']}: {tool['description'][:50]}...")
    
    # Test 3: Call dblp_analyze (mock)
    print("\n📌 Test 3: Call dblp_analyze (mock)")
    request = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "dblp_analyze",
            "arguments": {
                "url": "https://dblp.org/db/conf/icml/icml2024.html"
            }
        }
    }
    # Note: This will fail without network, but tests the interface
    try:
        response = await server.handle_request(request)
        print(f"  Response keys: {list(response.keys())}")
    except Exception as e:
        print(f"  Expected error (no network): {str(e)[:100]}")
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(test_mcp_server())
