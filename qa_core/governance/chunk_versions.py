"""MySQL 控制面的文档 chunk 有效期索引。(★★ 理解)

为引用式增量版本维护 chunk 的有效期窗口。每个 chunk 有 valid_from_seq 和 valid_to_seq，
检索时根据当前活跃版本的 version_seq 过滤可见 chunk：
- valid_from_seq <= active_seq：chunk 在 active_seq 版本时已存在。
- valid_to_seq == 0 OR valid_to_seq > active_seq：chunk 尚未被过期或在此版本后仍有效。

设计决策：
- 引用式增量不复制数据，只记录有效期窗口，切换版本是 O(1) 操作。
- 入库脚本只写入 valid_from_seq，valid_to_seq 初始为 0（表示未过期）。
- 新版本发布时调用 expire_chunks() 将旧版 chunk 的 valid_to_seq 设为新版本号。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

from qa_core.memory.base import _MySqlStore, safe_sql_identifier


KB_CHUNK_VERSIONS_TABLE = "kb_chunk_versions"


@dataclass(frozen=True)
class ChunkVersionRecord:
    """文档 chunk 有效期记录，从 Milvus metadata 镜像到 MySQL 控制面。

    一条记录表示一个 chunk 在某个版本范围内的可见性：
    - valid_from_seq：该 chunk 从哪个版本序列号开始可见（包含）。
    - valid_to_seq：该 chunk 到哪个版本序列号结束可见（独占），0 表示未过期。

    调用顺序：治理或版本管理入口 -> ChunkVersionRecord。
    """

    scenario_id: str
    chunk_id: str
    source: str
    kb_version: str
    file_path: str
    valid_from_seq: int
    valid_to_seq: int = 0

    @classmethod
    def from_row(cls, row: Any) -> "ChunkVersionRecord":
        """从 SQLAlchemy 行映射构建记录。

        参数：
            row: SQLAlchemy RowMapping 对象。

        返回：
            ChunkVersionRecord 实例，所有字段已转换为正确类型。

        调用顺序：list_visible() -> ChunkVersionRecord.from_row()。
        """
        payload = dict(row)
        payload["valid_from_seq"] = int(payload.get("valid_from_seq") or 0)
        payload["valid_to_seq"] = int(payload.get("valid_to_seq") or 0)
        payload["file_path"] = str(payload.get("file_path") or "")
        return cls(**payload)


class ChunkVersionIndex(_MySqlStore):
    """chunk 有效期索引，支持引用式增量版本的可见性查询。（★★★ 核心）

    提供以下操作：
    - upsert_chunks: 创建或刷新 chunk 有效期记录。
    - expire_chunks: 将 chunk 标记为在指定版本后不可见。
    - list_visible: 查询在特定版本序列号下可见的 chunk 列表。

    调用顺序：治理或版本管理入口 -> ChunkVersionIndex。
    """

    def __init__(self) -> None:
        """初始化 chunk 有效期索引。MySQL 表结构由启动期 bootstrap 统一建表。

        调用顺序：治理或版本管理入口 -> ChunkVersionIndex.__init__()。
        """
        super().__init__()
        self.table_name = safe_sql_identifier(KB_CHUNK_VERSIONS_TABLE, label="KB chunk versions table")

    def upsert_chunks(
        self,
        chunk_ids: list[str],
        *,
        scenario_id: str,
        source: str,
        kb_version: str,
        valid_from_seq: int,
        valid_to_seq: int = 0,
        file_path: str = "",
    ) -> None:
        """创建或刷新 chunk 有效期记录。（★★★ 核心）

        使用 INSERT ON DUPLICATE KEY UPDATE 实现幂等写入。
        如果 chunk_id 已存在，更新其有效期范围和相关元数据。

        参数：
            chunk_ids: 要写入的 chunk ID 列表。
            scenario_id: 业务场景标识。
            source: 数据来源（如 faq / document）。
            kb_version: 知识库版本号。
            valid_from_seq: 有效起始版本序列号（必须 > 0）。
            valid_to_seq: 有效结束版本序列号，0 表示未过期（默认值）。
            file_path: 文档源文件路径（可选）。

        异常：
            ValueError: valid_from_seq <= 0。

        调用顺序：入库或版本治理流程 -> upsert_chunks()。
        """
        if not chunk_ids:
            # 无 chunk id 时直接跳过，避免空列表执行无效的 SQL INSERT
            return
        if valid_from_seq <= 0:
            raise ValueError("valid_from_seq must be a positive version sequence")
        sql = f"""
        INSERT INTO {self.table_name}
            (
                scenario_id, chunk_id, source, kb_version, file_path,
                valid_from_seq, valid_to_seq
            )
        VALUES
            (
                :scenario_id, :chunk_id, :source, :kb_version, :file_path,
                :valid_from_seq, :valid_to_seq
            )
        ON DUPLICATE KEY UPDATE
            source=VALUES(source),
            kb_version=VALUES(kb_version),
            file_path=VALUES(file_path),
            valid_from_seq=VALUES(valid_from_seq),
            valid_to_seq=VALUES(valid_to_seq),
            updated_at=CURRENT_TIMESTAMP
        """
        rows = [
            {
                "scenario_id": scenario_id,
                "chunk_id": chunk_id,
                "source": source,
                "kb_version": kb_version,
                "file_path": file_path,
                "valid_from_seq": int(valid_from_seq),
                "valid_to_seq": int(valid_to_seq or 0),
            }
            for chunk_id in chunk_ids
        ]
        with self.engine.begin() as conn:
            conn.execute(text(sql), rows)

    def expire_chunks(
        self,
        chunk_ids: list[str],
        *,
        scenario_id: str,
        source: str,
        kb_version: str,
        valid_from_seq: int,
        valid_to_seq: int,
        file_path: str = "",
    ) -> None:
        """将 chunk 标记为在 ``valid_to_seq`` 版本后不可见。（★★★ 核心）

        在新版本发布时调用，将旧版本的 chunk 过期（设置 valid_to_seq > 0）。
        检索时 list_visible 会自动排除 valid_to_seq > active_seq 的 chunk。

        参数：
            chunk_ids: 要过期的 chunk ID 列表。
            scenario_id: 业务场景标识。
            source: 数据来源。
            kb_version: 知识库版本号。
            valid_from_seq: 有效起始版本序列号。
            valid_to_seq: 有效结束版本序列号（必须 > 0，表示该 chunk 在此版本后不可见）。
            file_path: 文档源文件路径（可选）。

        异常：
            ValueError: valid_to_seq <= 0。

        调用顺序：版本发布/回滚流程 -> ChunkVersionIndex.expire_chunks()。
        """
        if not chunk_ids:
            return
        if valid_to_seq <= 0:
            raise ValueError("valid_to_seq must be a positive version sequence")
        # 先确保有效期记录存在，再更新关闭序列号。
        # 在此调用 upsert_chunks 时传入 valid_to_seq > 0，
        # 检索时 list_visible 会排除 valid_to_seq > active_seq 的 chunk
        self.upsert_chunks(
            chunk_ids,
            scenario_id=scenario_id,
            source=source,
            kb_version=kb_version,
            valid_from_seq=valid_from_seq,
            valid_to_seq=valid_to_seq,
            file_path=file_path,
        )

    def list_visible(
        self,
        *,
        scenario_id: str,
        active_seq: int,
        source: str | None = None,
    ) -> list[ChunkVersionRecord]:
        """查询在指定活跃版本序列号下可见的 chunk 列表。（★★★ 核心）

        可见性判断逻辑：
        - valid_from_seq <= active_seq：chunk 在 active_seq 版本时已存在。
        - valid_to_seq == 0 OR valid_to_seq > active_seq：chunk 尚未被过期。

        参数：
            scenario_id: 业务场景标识。
            active_seq: 当前活跃版本序列号。
            source: 可选的数据来源过滤。

        返回：
            可见的 ChunkVersionRecord 列表，按 source 和 chunk_id 排序。

        调用顺序：检索链路 -> ChunkVersionIndex.list_visible()。
        """
        filters = [
            "scenario_id = :scenario_id",
            "valid_from_seq <= :active_seq",
            # valid_to_seq == 0 表示该 chunk 尚未被过期（仍在有效窗口内）；
            # valid_to_seq > active_seq 表示该 chunk 在此版本后仍有效
            "(valid_to_seq = 0 OR valid_to_seq > :active_seq)",
        ]
        params: dict[str, Any] = {"scenario_id": scenario_id, "active_seq": int(active_seq)}
        if source:
            filters.append("source = :source")
            params["source"] = source
        sql = f"""
        SELECT
            scenario_id, chunk_id, source, kb_version, file_path,
            valid_from_seq, valid_to_seq
        FROM {self.table_name}
        WHERE {' AND '.join(filters)}
        ORDER BY source ASC, chunk_id ASC
        """
        with self.engine.begin() as conn:
            rows = conn.execute(text(sql), params).mappings().all()
        return [ChunkVersionRecord.from_row(row) for row in rows]
