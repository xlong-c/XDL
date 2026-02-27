#!/usr/bin/env python3
"""
MCP Server Template for OpenCode Skill
MCP 服务器模板 - 可根据实际需求修改
"""

import sys
import json
import asyncio
from typing import Any, Dict, List, Optional
from pathlib import Path


class MCPServer:
    """MCP 服务器基类"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.tools: Dict[str, callable] = {}
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理 MCP 请求"""
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")
        
        try:
            if method == "initialize":
                return await self.initialize(params)
            elif method == "tools/list":
                return await self.list_tools()
            elif method == "tools/call":
                tool_name = params.get("name")
                tool_args = params.get("arguments", {})
                return await self.call_tool(tool_name, tool_args)
            elif method == "notifications/initialized":
                return {"jsonrpc": "2.0", "result": {}}
            else:
                return self._error_response(request_id, f"Unknown method: {method}")
        except Exception as e:
            if self.verbose:
                print(f"Error handling request: {e}", file=sys.stderr)
            return self._error_response(request_id, str(e))
    
    async def initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """初始化 MCP 连接"""
        return {
            "jsonrpc": "2.0",
            "id": params.get("id"),
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "skill-mcp",
                    "version": "1.0.0"
                }
            }
        }
    
    async def list_tools(self) -> Dict[str, Any]:
        """列出所有可用工具"""
        # 子类需要实现此方法以返回工具列表
        tools = []
        return {
            "jsonrpc": "2.0",
            "id": "list",
            "result": {"tools": tools}
        }
    
    async def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """调用工具"""
        if tool_name not in self.tools:
            return {
                "jsonrpc": "2.0",
                "id": "call",
                "error": {"message": f"Tool not found: {tool_name}"}
            }
        
        try:
            tool_func = self.tools[tool_name]
            result = await tool_func(tool_args)
            return {
                "jsonrpc": "2.0",
                "id": "call",
                "result": result
            }
        except Exception as e:
            if self.verbose:
                print(f"Error calling tool {tool_name}: {e}", file=sys.stderr)
            return {
                "jsonrpc": "2.0",
                "id": "call",
                "error": {"message": str(e)}
            }
    
    def _error_response(self, request_id: Any, message: str) -> Dict[str, Any]:
        """生成错误响应"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"message": message}
        }


async def run_server(server: MCPServer):
    """运行 MCP 服务器"""
    # 从 stdin 读取请求
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)
    
    # 创建写入管道
    write_transport, write_protocol = await asyncio.get_event_loop().create_subprocess_exec(
        "cat",
        stdout=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.PIPE
    )
    
    writer = asyncio.StreamWriter(write_transport, write_protocol, reader, None)
    
    if server.verbose:
        print("MCP Server started", file=sys.stderr)
    
    while True:
        try:
            line = await reader.readline()
            if not line:
                break
            
            try:
                request = json.loads(line.decode())
                if server.verbose:
                    print(f"Received request: {request.get('method')}", file=sys.stderr)
                
                response = await server.handle_request(request)
                
                response_json = json.dumps(response, ensure_ascii=False)
                writer.write((response_json + "\n").encode())
                await writer.drain()
                
                if server.verbose:
                    print(f"Sent response", file=sys.stderr)
            except json.JSONDecodeError as e:
                error_response = server._error_response(None, f"Invalid JSON: {e}")
                writer.write((json.dumps(error_response) + "\n").encode())
                await writer.drain()
                
        except Exception as e:
            if server.verbose:
                print(f"Server error: {e}", file=sys.stderr)
            error_response = server._error_response(None, str(e))
            writer.write((json.dumps(error_response) + "\n").encode())
            await writer.drain()
    
    writer.close()
    await writer.wait_closed()


# ============================================================================
# 示例实现 - 请根据实际需求修改
# ============================================================================

class ExampleSkillServer(MCPServer):
    """示例 Skill 服务器实现"""
    
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)
        # 注册工具
        self.tools = {
            "example_tool": self.example_tool,
        }
    
    async def example_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """示例工具实现"""
        # 获取参数
        param1 = args.get("param1", "default")
        param2 = args.get("param2", 0)
        
        # 实现逻辑
        result = f"Processed: {param1} with value {param2}"
        
        # 返回结果
        return {
            "success": True,
            "message": result,
            "data": {
                "processed_param1": param1,
                "processed_param2": param2
            }
        }
    
    async def list_tools(self) -> Dict[str, Any]:
        """列出工具"""
        tools = [
            {
                "name": "example_tool",
                "description": "示例工具 - 处理输入参数",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "param1": {
                            "type": "string",
                            "description": "第一个参数"
                        },
                        "param2": {
                            "type": "integer",
                            "description": "第二个参数",
                            "default": 0
                        }
                    },
                    "required": []
                }
            }
        ]
        return {
            "jsonrpc": "2.0",
            "id": "list",
            "result": {"tools": tools}
        }


# ============================================================================
# 入口点
# ============================================================================

async def main():
    """主函数"""
    # 创建服务器实例（使用示例实现）
    # 实际使用时请替换为您自己的服务器类
    server = ExampleSkillServer(verbose=True)
    
    # 运行服务器
    await run_server(server)


if __name__ == "__main__":
    asyncio.run(main())
