/* 管理控制台入口：加载版本、质量、评测和反馈数据，并统一携带管理令牌调用治理 API。 */
(function () {
  const tokenInput = document.getElementById('adminTokenInput');
  const scenarioSelect = document.getElementById('scenarioSelect');
  const refreshBtn = document.getElementById('refreshBtn');

  tokenInput.value = localStorage.getItem('qa_admin_token') || '';

  function adminHeaders() {
    const token = tokenInput.value.trim();
    return token ? { 'X-Admin-Token': token } : {};
  }

  async function fetchJson(url, withToken = true, options = {}) {
    const headers = {
      ...(withToken ? adminHeaders() : {}),
      ...(options.headers || {}),
    };
    const response = await fetch(url, { ...options, headers });
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`);
    }
    return response.json();
  }

  function escapeHtml(value) {
    return String(value ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function text(value) {
    return value === undefined || value === null || value === '' ? '-' : String(value);
  }

  function dateText(value) {
    if (!value) return '-';
    if (typeof value === 'number') return new Date(value * 1000).toLocaleString();
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
  }

  function shortText(value, max = 48) {
    const raw = text(value);
    return raw.length > max ? `${raw.slice(0, max - 1)}...` : raw;
  }

  function showBanner(message, type = 'warn') {
    const banner = document.getElementById('adminBanner');
    banner.textContent = message;
    banner.className = `admin-banner is-${type}`;
    banner.style.display = '';
  }

  function hideBanner() {
    const banner = document.getElementById('adminBanner');
    banner.style.display = 'none';
  }

  function item(label, value, icon = 'fa-circle-info') {
    return `
      <div class="quality-item">
        <div class="quality-icon"><i class="fas ${icon}"></i></div>
        <div>
          <div class="quality-title">${escapeHtml(label)}</div>
          <div class="quality-desc">${escapeHtml(text(value))}</div>
        </div>
      </div>
    `;
  }

  function cacheStat(label, value) {
    return `
      <div class="cache-stat">
        <span>${escapeHtml(label)}</span>
        <strong>${escapeHtml(text(value))}</strong>
      </div>
    `;
  }

  function actionButton(label, action, payload = {}) {
    return `<button class="btn btn-secondary btn-sm admin-action" data-action="${escapeHtml(action)}" data-payload='${escapeHtml(JSON.stringify(payload))}'>${escapeHtml(label)}</button>`;
  }

  function prettyJson(payload) {
    return JSON.stringify(payload ?? {}, null, 2);
  }

  function setDetail(title, payload) {
    document.getElementById('detailSubtitle').textContent = title;
    document.getElementById('detailView').textContent = prettyJson(payload);
  }

  function fileSummary(label, summary) {
    const available = summary && summary.available;
    const value = available
      ? `${summary.file || '-'} | ${dateText(summary.updated_at)}`
      : '暂无报告';
    return item(label, value, available ? 'fa-file-circle-check' : 'fa-circle-exclamation');
  }

  function renderLangSmith(status) {
    document.getElementById('langsmithEnabledValue').textContent = status.enabled ? '已开启' : '未开启';
    document.getElementById('langsmithProjectValue').textContent = status.project || '-';
    document.getElementById('langsmithStatus').innerHTML = [
      item('Tracing', status.enabled ? 'LANGSMITH_TRACING=true' : 'LANGSMITH_TRACING=false', 'fa-wave-square'),
      item('Project', status.project || '-', 'fa-folder-tree'),
      item('Endpoint', status.endpoint || '-', 'fa-link'),
      item('API Key', status.has_api_key ? '已配置' : '未配置', status.has_api_key ? 'fa-key' : 'fa-circle-exclamation'),
    ].join('');
  }

  function renderLlmStatus(status) {
    const ok = status.ok === true;
    const known = status.ok !== null && status.ok !== undefined;
    document.getElementById('llmStatusValue').textContent = ok ? '可用' : known ? '不可用' : '未探测';
    document.getElementById('llmModelValue').textContent = status.model || '-';
    document.getElementById('llmStatusList').innerHTML = [
      item('连接状态', ok ? 'LLM 服务可用' : status.error || '尚未完成连通性探测', ok ? 'fa-circle-check' : 'fa-circle-exclamation'),
      item('模型', status.model || '-', 'fa-brain'),
      item('Base URL', status.base_url || '-', 'fa-link'),
      item('最近探测', status.checked_at ? dateText(status.checked_at) : '-', 'fa-clock'),
    ].join('');
  }

  function renderCacheStatus(status) {
    const redis = status.redis || {};
    const config = status.config || {};
    const stats = status.stats || {};
    const namespaces = status.namespaces || [];
    const events = status.recent_events || [];
    const namespaceText = namespaces.length
      ? namespaces.map(row => `${row.scenario_id || '-'} / ${row.tenant_id || 'default'} / ${row.dataset_id || 'default'}: epoch ${row.cache_epoch ?? '-'}`).join('\n')
      : '暂无 namespace 记录';

    document.getElementById('cacheEnabledValue').textContent = status.enabled ? '已开启' : '未开启';
    document.getElementById('cacheFootValue').textContent = status.enabled
      ? `${redis.host || '-'}:${redis.port || '-'} / db ${redis.db ?? '-'}`
      : 'CACHE_ENABLED=false';

    document.getElementById('cacheStats').innerHTML = [
      cacheStat('Redis', status.enabled ? (redis.ok ? '正常' : '异常') : '未启用'),
      cacheStat('检索命中', stats.retrieval_hits ?? 0),
      cacheStat('检索未命中', stats.retrieval_misses ?? 0),
      cacheStat('Embedding 命中', stats.embedding_hits ?? 0),
    ].join('');

    document.getElementById('cacheStatusList').innerHTML = [
      item('Key Prefix', config.key_prefix || '-', 'fa-key'),
      item('TTL', `FAQ ${config.faq_ttl_seconds ?? '-'}s | Doc ${config.doc_ttl_seconds ?? '-'}s | Embedding ${config.embedding_ttl_seconds ?? '-'}s`, 'fa-clock'),
      item('版本边界', `embedding ${config.embedding_model_version || '-'} | reranker ${config.reranker_model_version || '-'} | chunk ${config.chunk_schema_version || '-'}`, 'fa-code-branch'),
      item('Namespace Epoch', namespaceText, 'fa-layer-group'),
      item(
        '最近缓存事件',
        events.length
          ? events.slice(0, 5).map(event => `${dateText(event.created_at)} | ${event.kind || '-'} | ${event.source_type || '-'} | ${event.hit === true ? 'hit' : event.hit === false ? 'miss' : 'write'}`).join('\n')
          : '暂无事件',
        'fa-wave-square',
      ),
    ].join('');
  }

  function renderIntentModelStatus(status) {
    const payload = status.payload || {};
    const model = payload.model || {};
    const evaluation = payload.evaluation || {};
    const policy = payload.decision_policy || {};
    const closure = payload.closure || {};
    const accuracy = evaluation.accuracy === undefined ? '-' : `${Math.round(Number(evaluation.accuracy) * 1000) / 10}%`;

    document.getElementById('intentModelStats').innerHTML = [
      cacheStat('模型状态', payload.ok ? '正常' : '异常'),
      cacheStat('版本', model.model_version || '-'),
      cacheStat('准确率', accuracy),
      cacheStat('策略', policy.policy_version || '-'),
    ].join('');

    document.getElementById('intentModelList').innerHTML = [
      item('模型路径', model.model_path || '-', 'fa-folder-tree'),
      item('标签', (model.labels || []).join(' / ') || '-', 'fa-tags'),
      item('评测集', `train ${model.training_examples ?? '-'} | eval ${model.eval_examples ?? '-'}`, 'fa-clipboard-check'),
      item('闭环脚本', [
        closure.training_script,
        closure.model_eval_script,
        closure.policy_eval_script,
      ].filter(Boolean).join('\n') || '-', 'fa-terminal'),
      item('最近报告', status.available ? `${status.file || '-'} | ${dateText(status.updated_at)}` : '暂无报告，运行 demo_intent_model.py --eval-only --output latest 生成', 'fa-file-circle-check'),
    ].join('');
  }

  function renderVersions(payload) {
    const versions = payload.versions || [];
    document.getElementById('kbValue').textContent = shortText(payload.effective_active_version || '-', 24);
    document.getElementById('versionSource').textContent = `来源：${text(payload.active_version_source)}`;
    if (!versions.length) {
      document.getElementById('versionRows').innerHTML = '<tr><td colspan="6">暂无版本清单</td></tr>';
      return;
    }
    document.getElementById('versionRows').innerHTML = versions.map(version => `
      <tr>
        <td class="mono">${escapeHtml(version.kb_version || '-')}</td>
        <td>${escapeHtml(version.status || '-')}</td>
        <td>${escapeHtml(dateText(version.created_at))}</td>
        <td>${escapeHtml(shortText(version.description || version.notes || '-', 70))}</td>
        <td>${escapeHtml([
          `doc ${version.stats?.last_doc_count ?? version.stats?.total_doc_written ?? 0}`,
          `faq ${version.stats?.last_faq_count ?? version.stats?.total_faq_written ?? 0}`,
          version.version_seq ? `seq ${version.version_seq}` : null,
        ].filter(Boolean).join(' | '))}</td>
        <td>
          <div class="admin-actions">
            ${actionButton('详情', 'version_detail', { version })}
            ${version.status !== 'ACTIVE' && version.status !== 'ARCHIVED' && version.activated_at ? actionButton('回滚', 'activate_version', { kb_version: version.kb_version }) : ''}
            ${version.status !== 'ACTIVE' && version.status !== 'ARCHIVED' ? actionButton('归档', 'archive_version', { kb_version: version.kb_version }) : ''}
          </div>
        </td>
      </tr>
    `).join('');
  }

  function renderActivationHistory(payload) {
    const rows = payload.activation_history || [];
    const target = document.getElementById('activationHistoryList');
    if (!rows.length) {
      target.innerHTML = item('版本切换历史', '暂无激活或回滚记录', 'fa-circle-info');
      return;
    }
    target.innerHTML = rows.slice(0, 8).map(row => {
      const actionLabel = row.action === 'rollback'
        ? '回滚'
        : '激活';
      const value = [
        `${shortText(row.from_kb_version || '-', 20)} -> ${shortText(row.to_kb_version || '-', 20)}`,
        `seq ${row.from_version_seq ?? 0} -> ${row.to_version_seq ?? 0}`,
        row.activated_by ? `by ${row.activated_by}` : null,
        row.reason ? `reason ${row.reason}` : null,
      ].filter(Boolean).join(' | ');
      return `
        <div class="quality-item quality-item-action">
          <div class="quality-icon"><i class="fas ${row.action === 'rollback' ? 'fa-clock-rotate-left' : 'fa-code-branch'}"></i></div>
          <div class="quality-content">
            <div class="quality-title">${escapeHtml(actionLabel)} · ${escapeHtml(dateText(row.created_at))}</div>
            <div class="quality-desc mono">${escapeHtml(value)}</div>
          </div>
          <div class="quality-tail">
            ${actionButton('详情', 'activation_detail', { row })}
          </div>
        </div>
      `;
    }).join('');
  }

  function renderIngestion(rows) {
    if (!rows.length) {
      document.getElementById('ingestionList').innerHTML = item('入库质量报告', '暂无报告', 'fa-circle-exclamation');
      return;
    }
    document.getElementById('ingestionList').innerHTML = rows.map(row => {
      const summary = row.summary || {};
      const value = [
        `文件 ${summary.files_loaded_count ?? 0}/${summary.files_scanned ?? 0}`,
        `FAQ 冲突 ${summary.faq_document_conflicts?.count ?? 0}`,
        `表格 ${summary.table_files_count ?? 0}`,
        `OCR 风险 ${summary.ocr_risk_files_count ?? 0}`,
      ].join(' | ');
      return `
        <div class="quality-item quality-item-action">
          <div class="quality-icon"><i class="fas ${summary.ok === false ? 'fa-circle-exclamation' : 'fa-file-circle-check'}"></i></div>
          <div class="quality-content">
            <div class="quality-title">${escapeHtml(row.file_name || row.path || '入库报告')}</div>
            <div class="quality-desc">${escapeHtml(value)}</div>
          </div>
          <div class="quality-tail">
            ${actionButton('查看详情', 'ingestion_detail', { path: row.path, file_name: row.file_name })}
          </div>
        </div>
      `;
    }).join('');
  }

  function renderGates(gates, performance) {
    const rows = [
      ...((gates.reports || []).map(report => ({ label: '质量回归', report }))),
      ...((performance.reports || []).map(report => ({ label: '性能回归', report }))),
    ];
    document.getElementById('gateList').innerHTML = rows.length ? rows.map(({ label, report }) => {
      const available = report && report.available;
      return `
        <div class="quality-item quality-item-action">
          <div class="quality-icon"><i class="fas ${available ? 'fa-file-circle-check' : 'fa-circle-exclamation'}"></i></div>
          <div class="quality-content">
            <div class="quality-title">${escapeHtml(label)}</div>
            <div class="quality-desc">${escapeHtml(available ? `${report.file || '-'} | ${dateText(report.updated_at)}` : '暂无报告')}</div>
          </div>
          <div class="quality-tail">
            ${available ? actionButton('查看详情', 'report_detail', { path: report.file, label }) : ''}
          </div>
        </div>
      `;
    }).join('') : item('回归报告', '暂无报告', 'fa-circle-exclamation');
  }

  function renderGovernance(payload) {
    const rows = [
      { label: '资料真实度', report: payload.data_realism },
      { label: '增强包预检', report: payload.overlay_readiness },
    ];
    document.getElementById('enterpriseGovernanceList').innerHTML = rows.map(({ label, report }) => {
      const available = report && report.available;
      return `
        <div class="quality-item quality-item-action">
          <div class="quality-icon"><i class="fas ${available ? 'fa-file-circle-check' : 'fa-circle-exclamation'}"></i></div>
          <div class="quality-content">
            <div class="quality-title">${escapeHtml(label)}</div>
            <div class="quality-desc">${escapeHtml(available ? `${report.file || '-'} | ${dateText(report.updated_at)}` : '暂无报告')}</div>
          </div>
          <div class="quality-tail">
            ${available ? actionButton('查看详情', 'report_detail', { path: report.file, label }) : ''}
          </div>
        </div>
      `;
    }).join('');
  }

  function renderBadFeedback(payload) {
    const items = payload.items || [];
    if (!items.length) {
      document.getElementById('badFeedbackList').innerHTML = item('低质量反馈', '暂无需要复盘的反馈', 'fa-circle-check');
      return;
    }
    document.getElementById('badFeedbackList').innerHTML = items.map(row => `
      <div class="quality-item quality-item-action">
        <div class="quality-icon"><i class="fas fa-circle-exclamation"></i></div>
        <div class="quality-content">
          <div class="quality-title">${escapeHtml(shortText(row.question || '-', 42))}</div>
          <div class="quality-desc">${escapeHtml(`${row.scenario_id || '-'} | ${dateText(row.created_at)} | ${row.rating || '-'}`)}</div>
        </div>
        <div class="quality-tail">
          ${actionButton('查看详情', 'bad_feedback_detail', { row })}
        </div>
      </div>
    `).join('');
  }

  async function handleAction(action, payload) {
    const scenarioId = scenarioSelect.value;
    if (action === 'version_detail') {
      setDetail(`版本详情 · ${payload.version?.kb_version || '-'}`, payload.version || {});
      return;
    }
    if (action === 'bad_feedback_detail') {
      setDetail(`低质量反馈 · ${payload.row?.id || '-'}`, payload.row || {});
      return;
    }
    if (action === 'activation_detail') {
      setDetail(`版本切换历史 · ${payload.row?.action || '-'}`, payload.row || {});
      return;
    }
    if (action === 'ingestion_detail') {
      const detail = await fetchJson(`/api/admin/ingestion_report_detail?path=${encodeURIComponent(payload.path)}`);
      setDetail(`入库质量报告 · ${payload.file_name || detail.file_name}`, detail.payload || {});
      return;
    }
    if (action === 'report_detail') {
      const detail = await fetchJson(`/api/admin/report_detail?path=${encodeURIComponent(payload.path)}`);
      setDetail(`${payload.label || '报告'} · ${detail.file_name}`, detail.payload || {});
      return;
    }
    if (action === 'activate_version') {
      await fetchJson(
        `/api/kb_versions/${encodeURIComponent(payload.kb_version)}/activate?scenario_id=${encodeURIComponent(scenarioId)}`,
        true,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ reason: 'admin_activate', activated_by: 'admin_page' }),
        },
      );
      await loadDashboard();
      showBanner(`已切换版本：${payload.kb_version}`, 'ok');
      return;
    }
    if (action === 'archive_version') {
      await fetchJson(
        `/api/kb_versions/${encodeURIComponent(payload.kb_version)}/archive?scenario_id=${encodeURIComponent(scenarioId)}`,
        true,
        { method: 'POST' },
      );
      await loadDashboard();
      showBanner(`已归档版本：${payload.kb_version}`, 'ok');
      return;
    }
    if (action === 'invalidate_cache') {
      await fetchJson('/api/admin/cache/invalidate', true, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scenario_id: scenarioId }),
      });
      await loadDashboard();
      showBanner(`已刷新场景缓存：${scenarioId}`, 'ok');
    }
  }

  document.addEventListener('click', async event => {
    const button = event.target.closest('.admin-action');
    if (!button) return;
    try {
      const action = button.dataset.action || '';
      const payload = JSON.parse(button.dataset.payload || '{}');
      await handleAction(action, payload);
    } catch (error) {
      showBanner(`操作失败：${error.message}`, 'warn');
    }
  });

  async function loadScenarios() {
    const scenarioId = document.body.dataset.runtimeScenarioId || 'tender_contract_risk';
    const displayName = document.body.dataset.contractProductDescriptor || '';
    scenarioSelect.innerHTML = `<option value="${escapeHtml(scenarioId)}">${escapeHtml(displayName)}</option>`;
    scenarioSelect.value = scenarioId;
    document.getElementById('scenarioCountValue').textContent = '单一合同场景';
  }

  async function loadDashboard() {
    hideBanner();
    localStorage.setItem('qa_admin_token', tokenInput.value.trim());
    const scenarioId = scenarioSelect.value;
    try {
      const [
        adminStatus,
        langsmith,
        versions,
        ingestion,
        gates,
        performance,
        governance,
        badFeedback,
        cacheStatus,
        intentModel,
      ] = await Promise.all([
        fetchJson('/api/admin/status'),
        fetchJson('/api/admin/langsmith'),
        fetchJson(`/api/kb_versions?scenario_id=${encodeURIComponent(scenarioId)}`),
        fetchJson(`/api/admin/ingestion_reports?scenario_id=${encodeURIComponent(scenarioId)}&limit=10`),
        fetchJson('/api/admin/gate_reports'),
        fetchJson('/api/admin/performance_reports'),
        fetchJson('/api/admin/enterprise_governance'),
        fetchJson(`/api/admin/bad_feedback?scenario_id=${encodeURIComponent(scenarioId)}&limit=10`),
        fetchJson(`/api/admin/cache/status?scenario_id=${encodeURIComponent(scenarioId)}`),
        fetchJson('/api/admin/intent_model'),
      ]);

      document.getElementById('updatedAt').textContent = `更新于 ${new Date().toLocaleString()}`;
      const activeScenario = (adminStatus.scenarios || []).includes(scenarioId) ? scenarioSelect.selectedOptions[0]?.textContent : scenarioId;
      document.getElementById('scenarioValue').textContent = activeScenario || '-';
      renderLangSmith(langsmith);
      renderLlmStatus(adminStatus.llm || {});
      renderVersions(versions);
      renderActivationHistory(versions);
      renderIngestion(ingestion.reports || []);
      renderGates(gates || {}, performance || {});
      renderGovernance(governance || {});
      renderBadFeedback(badFeedback || {});
      renderCacheStatus(cacheStatus || {});
      renderIntentModelStatus(intentModel || {});
    } catch (error) {
      showBanner(`加载失败：${error.message}。请确认 ADMIN_API_TOKEN 是否正确。`, 'warn');
    }
  }

  refreshBtn.addEventListener('click', loadDashboard);
  tokenInput.addEventListener('change', loadDashboard);
  scenarioSelect.addEventListener('change', loadDashboard);

  loadScenarios()
    .then(loadDashboard)
    .catch(error => showBanner(`场景加载失败：${error.message}`, 'warn'));
})();
