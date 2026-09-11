/* 合同工作台：只消费合同 API，不在浏览器复制抽取、风险或日期计算规则。 */
const contractState = { items: [], selectedId: null, detail: null, socket: null, sessionId: null };
const byId = id => document.getElementById(id);

function escapeHtml(value) {
  const node = document.createElement('div');
  node.textContent = String(value ?? '');
  return node.innerHTML;
}

function label(value) {
  const labels = {
    active: '履行中', draft: '草稿', expired: '已到期', terminated: '已终止', archived: '已归档',
    ready: '分析完成', uploaded: '已上传', parsing: '解析中', parsed: '已解析', analyzing: '分析中', failed: '处理失败',
    pending: '待处理', waiting_trigger: '待触发', in_progress: '进行中', completed: '已完成', overdue: '已逾期', waived: '已豁免', cancelled: '已取消',
    critical: '重大', high: '高', medium: '中', low: '低', rule: '规则', llm: '模型', hybrid: '混合'
  };
  return labels[value] || value || '-';
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, { cache: 'no-store', ...options, headers: { ...(options.headers || {}) } });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || payload.error || `${response.status} ${response.statusText}`);
  return payload;
}

function toast(message) {
  const node = byId('toast');
  node.textContent = message;
  node.classList.remove('hidden');
  window.clearTimeout(toast.timer);
  toast.timer = window.setTimeout(() => node.classList.add('hidden'), 2800);
}

async function loadContracts() {
  const params = new URLSearchParams();
  if (byId('statusFilter').value) params.set('status', byId('statusFilter').value);
  if (byId('riskFilter').value) params.set('risk_level', byId('riskFilter').value);
  try {
    const payload = await requestJson(`/api/contracts?${params}`);
    contractState.items = payload.contracts || [];
    renderContractList();
    if (!contractState.selectedId && contractState.items.length) selectContract(contractState.items[0].id);
  } catch (error) {
    byId('contractList').innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
  }
}

function renderContractList() {
  byId('contractCount').textContent = `${contractState.items.length} 份合同`;
  const root = byId('contractList');
  root.replaceChildren();
  if (!contractState.items.length) {
    root.innerHTML = '<div class="empty-state">暂无合同，点击右上角上传。</div>';
    return;
  }
  contractState.items.forEach(item => {
    const button = document.createElement('button');
    button.className = `contract-item${item.id === contractState.selectedId ? ' active' : ''}`;
    button.innerHTML = `<strong>${escapeHtml(item.contract_name)}</strong><div class="contract-item-meta"><span>${escapeHtml(item.contract_type || label(item.status))}</span><span class="tag ${escapeHtml(item.overall_risk_level)}">${label(item.overall_risk_level)}</span></div>`;
    button.addEventListener('click', () => selectContract(item.id));
    root.appendChild(button);
  });
}

async function selectContract(contractId) {
  contractState.selectedId = contractId;
  renderContractList();
  byId('emptyDetail').classList.add('hidden');
  byId('contractDetail').classList.remove('hidden');
  byId('detailName').textContent = '加载中…';
  try {
    contractState.detail = await requestJson(`/api/contracts/${contractId}`);
    contractState.sessionId = `contract:${contractId}:${crypto.randomUUID()}`;
    renderDetail(contractState.detail);
  } catch (error) {
    toast(error.message);
    byId('detailName').textContent = '合同加载失败';
  }
}

function contractFieldValue(field) {
  if (!field) return '-';
  const value = field.extracted_value;
  return Array.isArray(value) ? value.join('、') : (value ?? '-');
}

function evidenceButton(evidence, text = '查看原文') {
  if (!evidence || !evidence.quote) return '<span class="helper-text">无引用</span>';
  return `<button class="evidence-link" data-evidence="${encodeURIComponent(JSON.stringify(evidence))}"><i class="fas fa-quote-left"></i> ${text}</button>`;
}

function bindEvidenceButtons(root = document) {
  root.querySelectorAll('[data-evidence]').forEach(button => button.addEventListener('click', () => {
    const evidence = JSON.parse(decodeURIComponent(button.dataset.evidence));
    const body = byId('sourceBody');
    body.innerHTML = `<div class="helper-text">文档 ${escapeHtml(evidence.document_id || '-')} · 第 ${escapeHtml(evidence.page_number || '-')} 页 · ${escapeHtml(evidence.section || '')} ${escapeHtml(evidence.clause_no || '')}</div><div class="source-quote">${escapeHtml(evidence.quote)}</div><div class="helper-text">chunk_id: ${escapeHtml(evidence.chunk_id || '-')}</div>`;
    byId('sourceDialog').showModal();
  }));
}

function renderDetail(detail) {
  byId('detailName').textContent = detail.contract_name;
  byId('detailMeta').innerHTML = `<span>编号：${escapeHtml(detail.contract_number || '待抽取')}</span><span>类型：${escapeHtml(detail.contract_type || '待抽取')}</span><span>状态：${label(detail.status)}</span><span>更新：${escapeHtml(detail.updated_at || '-')}</span>`;
  byId('processBadge').className = `status-badge ${escapeHtml(detail.processing_status)}`;
  byId('processBadge').textContent = label(detail.processing_status);
  renderDashboard(detail.risk_dashboard || {});
  renderBasicInfo(detail.analysis?.extraction?.basic_info || {});
  renderSummary(detail.analysis?.summary || [], detail.analysis || null);
  renderTimeline(detail.timeline || []);
  renderObligations(detail.obligations || []);
  renderRisks(detail.risks || []);
  renderRecords(detail.documents || [], detail.analysis_runs || []);
  byId('workdayNote').textContent = detail.workday_calculation_note || '';
  bindEvidenceButtons(byId('contractDetail'));
}

function renderDashboard(data) {
  const metrics = [
    ['重大风险', data.critical_count || 0], ['高风险', data.high_count || 0], ['中风险', data.medium_count || 0],
    ['低风险', data.low_count || 0], ['逾期事项', data.overdue_obligation_count || 0], ['近期到期', data.upcoming_obligation_count || 0]
  ];
  byId('riskOverview').innerHTML = metrics.map(([name, value]) => `<div class="metric"><span>${name}</span><strong>${value}</strong></div>`).join('');
}

function renderBasicInfo(info) {
  const fields = [
    ['合同名称', 'contract_name'], ['合同编号', 'contract_number'], ['合同类型', 'contract_type'], ['甲方', 'party_a'], ['乙方', 'party_b'],
    ['签署日期', 'signing_date'], ['生效日期', 'effective_date'], ['终止日期', 'termination_date'], ['合同期限', 'contract_term'], ['合同金额', 'contract_amount'], ['币种', 'currency']
  ];
  byId('basicFields').innerHTML = fields.map(([name, key]) => {
    const field = info[key];
    return `<div class="key-value"><span>${name}${field?.needs_review ? ' · 待复核' : ''}</span><strong>${escapeHtml(contractFieldValue(field))}</strong>${evidenceButton(field?.source_evidence)}</div>`;
  }).join('');
}

function renderSummary(items, analysis) {
  const summaryFailed = analysis?.stage_status?.summary === 'failed';
  const emptyMessage = !analysis
    ? '尚无成功分析结果。'
    : summaryFailed
      ? '摘要生成失败，其他已成功分析结果仍然可用；请查看分析记录后重试。'
      : '当前分析未生成摘要内容。';
  byId('summarySections').innerHTML = items.length ? items.map(item => `<article class="summary-item"><h4>${escapeHtml(item.title)}</h4><p>${escapeHtml(item.content)}</p>${evidenceButton((item.source_evidence || [])[0])}</article>`).join('') : `<div class="empty-state">${emptyMessage}</div>`;
}

function renderTimeline(items) {
  byId('timeline').innerHTML = items.length ? items.map(item => `<article class="timeline-item"><time>${escapeHtml(item.display_date || '待触发')}</time><h4>${escapeHtml(item.name)}</h4><p>${escapeHtml(item.responsible_party || item.description || '')} <span class="tag ${escapeHtml(item.status)}">${label(item.status)}</span></p>${evidenceButton(item.source_evidence)}</article>`).join('') : '<div class="empty-state">未识别到时间节点。</div>';
}

function renderObligations(items) {
  const statuses = ['pending', 'waiting_trigger', 'in_progress', 'completed', 'overdue', 'waived', 'cancelled'];
  byId('obligationRows').innerHTML = items.length ? items.map(item => `<tr><td><strong>${escapeHtml(item.title)}</strong><div class="helper-text">${escapeHtml(item.description || '')}</div></td><td>${escapeHtml(item.responsible_party || '-')}</td><td>${escapeHtml(item.obligation_type)}</td><td>${escapeHtml(item.planned_date || item.time_rule?.original_text || '待触发')}</td><td><span class="tag ${escapeHtml(item.status)}">${label(item.status)}</span></td><td><span class="tag ${escapeHtml(item.risk_level)}">${label(item.risk_level)}</span></td><td>${evidenceButton(item.evidence || item.source_evidence)}</td><td><select class="inline-select obligation-status" data-id="${escapeHtml(item.id)}">${statuses.map(status => `<option value="${status}"${status === item.status ? ' selected' : ''}>${label(status)}</option>`).join('')}</select></td></tr>`).join('') : '<tr><td colspan="8" class="empty-state">未识别到履约事项。</td></tr>';
  document.querySelectorAll('.obligation-status').forEach(select => select.addEventListener('change', async () => {
    select.disabled = true;
    try {
      const detail = await requestJson(`/api/contracts/${contractState.selectedId}/obligations/${select.dataset.id}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ status: select.value }) });
      contractState.detail = detail;
      renderDetail(detail);
      toast('履约状态已更新并重新计算风险');
    } catch (error) { toast(error.message); select.disabled = false; }
  }));
}

function renderRisks(items) {
  const statuses = ['open', 'acknowledged', 'resolved', 'ignored'];
  byId('riskCards').innerHTML = items.length ? items.map(item => `<article class="risk-card ${escapeHtml(item.risk_level)}"><div class="risk-card-head"><div><strong>${escapeHtml(item.risk_name)}</strong> <span class="tag ${escapeHtml(item.risk_level)}">${label(item.risk_level)}</span></div><div><span class="helper-text">${label(item.risk_source)} · ${item.needs_review ? '待复核' : '已校验证据'}</span> <select class="inline-select risk-status" data-id="${escapeHtml(item.id)}">${statuses.map(status => `<option value="${status}"${status === item.status ? ' selected' : ''}>${{open:'待处理',acknowledged:'已确认',resolved:'已解决',ignored:'已忽略'}[status]}</option>`).join('')}</select></div></div><p>${escapeHtml(item.description)}</p><p><strong>原因：</strong>${escapeHtml(item.reason)}</p><p class="suggestion"><strong>建议：</strong>${escapeHtml(item.suggestion)}</p>${evidenceButton(item.evidence || item.source_evidence)}</article>`).join('') : '<div class="empty-state">当前成功分析中未识别到风险。</div>';
  document.querySelectorAll('.risk-status').forEach(select => select.addEventListener('change', async () => {
    select.disabled = true;
    try {
      const detail = await requestJson(`/api/contracts/${contractState.selectedId}/risks/${select.dataset.id}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ status: select.value }) });
      contractState.detail = detail;
      renderDetail(detail);
      toast('风险处理状态已更新');
    } catch (error) { toast(error.message); select.disabled = false; }
  }));
}

async function submitEvent(event) {
  event.preventDefault();
  if (!contractState.selectedId) return;
  const form = event.currentTarget;
  const payload = Object.fromEntries(new FormData(form));
  byId('eventStatus').textContent = '正在保存事件并重新计算关联义务…';
  try {
    const detail = await requestJson(`/api/contracts/${contractState.selectedId}/events`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    contractState.detail = detail;
    renderDetail(detail);
    byId('eventDialog').close();
    form.reset();
    toast('事件已保存，履约日期与规则风险已重算');
  } catch (error) { byId('eventStatus').textContent = error.message; }
}

function renderRecords(documents, analyses) {
  byId('documentList').innerHTML = documents.length ? documents.map(item => `<div class="record-item"><strong>${escapeHtml(item.file_name)}</strong><span>${escapeHtml(item.document_type)} · 版本 ${escapeHtml(item.document_version)} · ${item.page_count || 0} 页 / ${item.chunk_count || 0} chunks · ${label(item.parse_status)}</span></div>`).join('') : '<div class="empty-state">暂无文件。</div>';
  byId('analysisList').innerHTML = analyses.length ? analyses.map(item => `<div class="record-item"><strong>${escapeHtml(item.analysis_version)} · ${label(item.status)}</strong><span>${escapeHtml(item.started_at || '-')} → ${escapeHtml(item.finished_at || '进行中')}${item.error_summary ? ` · ${escapeHtml(item.error_summary)}` : ''}</span></div>`).join('') : '<div class="empty-state">暂无分析记录。</div>';
}

async function submitUpload(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const data = new FormData(form);
  data.set('auto_analyze', form.elements.auto_analyze.checked ? 'true' : 'false');
  byId('uploadStatus').textContent = '正在解析、向量化并分析，请稍候…';
  const submit = form.querySelector('[type="submit"]');
  submit.disabled = true;
  try {
    const detail = await requestJson('/api/contracts', { method: 'POST', body: data });
    byId('uploadDialog').close();
    form.reset();
    await loadContracts();
    await selectContract(detail.id);
    toast('合同上传处理完成');
  } catch (error) { byId('uploadStatus').textContent = error.message; }
  finally { submit.disabled = false; }
}

async function reanalyze() {
  if (!contractState.selectedId) return;
  byId('reanalyzeBtn').disabled = true;
  byId('processBadge').textContent = '分析中';
  try {
    contractState.detail = await requestJson(`/api/contracts/${contractState.selectedId}/analyze`, { method: 'POST' });
    renderDetail(contractState.detail);
    await loadContracts();
    toast('合同重新分析完成');
  } catch (error) { toast(error.message); }
  finally { byId('reanalyzeBtn').disabled = false; }
}

function appendQaMessage(role, content) {
  const node = document.createElement('div');
  node.className = `qa-message ${role}`;
  if (role === 'assistant' && window.marked && window.DOMPurify) node.innerHTML = DOMPurify.sanitize(marked.parse(content || ''));
  else node.textContent = content;
  byId('qaMessages').appendChild(node);
  byId('qaMessages').scrollTop = byId('qaMessages').scrollHeight;
  return node;
}

function sendContractQuestion() {
  const query = byId('qaInput').value.trim();
  if (!query || !contractState.selectedId || byId('qaSend').disabled) return;
  byId('qaMessages').querySelector('.qa-welcome')?.remove();
  appendQaMessage('user', query);
  const answerNode = appendQaMessage('assistant', '正在检索当前合同条款…');
  byId('qaInput').value = '';
  byId('qaSend').disabled = true;
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const socket = new WebSocket(`${protocol}//${location.host}/api/contracts/${contractState.selectedId}/stream`);
  contractState.socket = socket;
  let answer = '';
  socket.onopen = () => socket.send(JSON.stringify({ query, session_id: contractState.sessionId }));
  socket.onmessage = event => {
    const data = JSON.parse(event.data);
    if (data.session_id) contractState.sessionId = data.session_id;
    if (data.type === 'status' && !answer) answerNode.textContent = data.message || '正在处理…';
    if (data.type === 'token') { answer += data.token || ''; answerNode.innerHTML = DOMPurify.sanitize(marked.parse(answer)); }
    if (data.type === 'end') {
      if (!answer) answerNode.textContent = '根据当前合同内容，未发现明确约定。';
      const sources = data.sources || [];
      if (sources.length) {
        const list = document.createElement('div');
        list.className = 'qa-source-list';
        list.innerHTML = '<strong>来源依据</strong>' + sources.map(source => `<div>${evidenceButton({ document_id: source.document_id, chunk_id: source.chunk_id, page_number: source.page, section: source.metadata?.section, clause_no: source.metadata?.clause_no || source.metadata?.clause_number, quote: source.source_text }, escapeHtml(source.citation || '查看原文'))}</div>`).join('');
        answerNode.appendChild(list);
        bindEvidenceButtons(list);
      }
      socket.close();
    }
    if (data.type === 'error') { answerNode.textContent = data.error || '问答失败'; socket.close(); }
  };
  socket.onerror = () => { answerNode.textContent = '问答连接失败，请稍后重试。'; };
  socket.onclose = () => { byId('qaSend').disabled = false; };
}

document.addEventListener('DOMContentLoaded', () => {
  byId('openUpload').addEventListener('click', () => byId('uploadDialog').showModal());
  document.querySelectorAll('[data-close]').forEach(button => button.addEventListener('click', () => button.closest('dialog').close()));
  byId('uploadForm').addEventListener('submit', submitUpload);
  byId('openEvent').addEventListener('click', () => byId('eventDialog').showModal());
  byId('eventForm').addEventListener('submit', submitEvent);
  byId('refreshList').addEventListener('click', loadContracts);
  byId('statusFilter').addEventListener('change', loadContracts);
  byId('riskFilter').addEventListener('change', loadContracts);
  byId('reanalyzeBtn').addEventListener('click', reanalyze);
  byId('qaSend').addEventListener('click', sendContractQuestion);
  byId('qaInput').addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); sendContractQuestion(); } });
  byId('detailTabs').addEventListener('click', event => {
    const button = event.target.closest('[data-tab]');
    if (!button) return;
    document.querySelectorAll('#detailTabs button').forEach(item => item.classList.toggle('active', item === button));
    document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.toggle('active', panel.id === `tab-${button.dataset.tab}`));
  });
  loadContracts();
});
