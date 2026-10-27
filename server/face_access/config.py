from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ServerConfig:
    runtime_dir: Path
    qt_socket: Path
    local_secret: str = field(repr=False)
    host: str = "0.0.0.0"
    port: int = 8080

    def __post_init__(self):
        if not isinstance(self.local_secret, str) or not 16 <= len(self.local_secret) <= 128:
            raise ValueError("本机启动密钥无效")
        if not 0 <= self.port <= 65535:
            raise ValueError("端口无效")
