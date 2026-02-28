#!/usr/bin/env python3
"""
MCP Server for gitU - 自动 Git 提交推送工具
"""

import sys
import json
import asyncio
from typing import Any, Dict
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from tools.auto_commit import auto_commit_tool


class GitUSkillServer:
    """gitU Skill MCP 服务器"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.tools = {
            "auto_commit": self.auto_commit,
        }
    
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
                print(f"Error: {e}", file=sys.stderr)
            return self._error_response(request_id, str(e))
    
    async def initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """初始化 MCP 连接"""
        return {
            "jsonrpc": "2.0",
            "id": params.get("id"),
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "gitu-mcp",
                    "version": "1.0.0"
                }
            }
        }
    
    async def list_tools(self) -> Dict[str, Any]:
        """列出所有可用工具"""
        tools = [
            {
                "name": "auto_commit",
                "description": "自动分析代码变更，生成智能提交描述，执行 git add、commit 和 push 一键提交推送所有变更",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "cwd": {
                            "type": "string",
                            "description": "工作目录路径（默认当前目录）",
                            "default": "."
                        },
                        "push": {
                            "type": "boolean",
                            "description": "是否自动推送到远程仓库",
                            "default": True
                        },
                        "remote": {
                            "type": "string",
                            "description": "远程仓库名称",
                            "default": "origin"
                        },
                        "branch": {
                            "type": "string",
                            "description": "分支名称（默认当前分支）"
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
    
    async def auto_commit(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """自动提交工具入口"""
        try:
            result = await auto_commit_tool(args)
            return {
                "jsonrpc": "2.0",
                "id": "call",
                "result": result
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": "call",
                "error": {"message": str(e)}
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


async def main():
    """主函数"""
    server = GitUSkillServer(verbose=True)
    
    if server.verbose:
        print("gitU MCP Server started", file=sys.stderr)
    
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
    
    while True:
        try:
            line = await reader.readline()
            if not line:
                break
            
            try:
                request = json.loads(line.decode())
                if server.verbose:
                    print(f"Request: {request.get('method')}", file=sys.stderr)
                
                response = await server.handle_request(request)
                
                response_json = json.dumps(response, ensure_ascii=False)
                writer.write((response_json + "\n").encode())
                await writer.drain()
                
                if server.verbose:
                    print(f"Response sent", file=sys.stderr)
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


if __name__ == "__main__":
    asyncio.run(main())
