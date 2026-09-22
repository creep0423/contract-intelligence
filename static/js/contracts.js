/* 合同工作台：只消费合同 API，不在浏览器复制抽取、风险或日期计算规则。
   本文件只负责展示层：状态文案、业务概览卡片、列表与详情渲染、原文依据弹窗与问答会话。 */
const contractState = { items: [], selectedId: null, detail: null, socket: null, sessionId: null };
const byId = id => document.getElementById(id);

const EMPTY_TEXT = '未明确';
const PENDING_STATUSES = ['pending', 'waiting_trigger', 'in_progress', 'overdue'];

/* 展示层标签：仅做值到文案的映射，不改变任何业务判定。 */
const LABELS = {
  active: '履行中', draft: '草稿', expired: '已到期', terminated: '已终止', archived: '已归档',
  ready: '分析完成', uploaded: '待解析', parsing: '解析中', parsed: '已解析', analyzing: '分析中', failed: '处理失败',
  pending: '待处理', waiting_trigger: '待触发', in_progress: '进行中', completed: '已完成', overdue: '已逾期', waived: '已豁免', cancelled: '已取消',
  critical: '重大', high: '高', medium: '中', low: '低',
  rule: '规则识别', llm: '模型识别', hybrid: '规则与模型',
  open: '待处理', acknowledged: '已确认', resolved: '已解决', ignored: '已忽略',
  running: '分析中', succeeded: '分析成功',
  payment: '付款', receivable: '收款', invoice: '开票', delivery: '交付', implementation: '实施',
  milestone: '里程碑', acceptance: '验收', document_submission: '资料提交', service: '服务',
  deposit_payment: '保证金支付', deposit_refund: '保证金退还', notice: '通知', renewal: '续约',
  termination: '终止', warranty: '质保', sla: '服务级别', confidentiality: '保密', insurance: '保险',
  audit: '审计', other: '其他',
  contract_signed: '合同签署', contract_effective: '合同生效', delivery_completed: '完成交付',
  acceptance_started: '开始验收', acceptance_passed: '验收通过', acceptance_failed: '验收未通过',
  invoice_received: '收到发票', payment_completed: '完成付款', milestone_completed: '完成里程碑',
  notice_sent: '发出通知', contract_terminated: '合同终止', warranty_ended: '质保期结束', custom: '其他事件',
  master_contract: '主合同', supplementary_agreement: '补充协议', amendment: '修订协议'
};

const ANALYSIS_STAGE_LABELS = {
  extraction: '要素抽取', obligations: '义务识别', deterministic_risk: '规则风险',
  semantic_risk: '模型风险', summary: '摘要生成', persistence: '结果保存'
};

function label(value) {
  if (value === null || value === undefined || value === '') return '—';
  return LABELS[value] || String(value);
}

function escapeHtml(value) {
  const node = document.createElement('div');
  node.textContent = String(value ?? '');
  return node.innerHTML;
}

function fieldText(field) {
  if (!field) return '';
  const value = field.extracted_value;
  const text = Array.isArray(value) ? value.join('、') : value;
  return text === null || text === undefined ? '' : String(text).trim();
}

function formatDate(value, withTime = false) {
  const raw = String(value ?? '').trim();
  if (!raw) return '';
  const match = raw.match(/^(\d{4}-\d{2}-\d{2})(?:[T ](\d{2}:\d{2}))?/);
  if (!match) return raw;
  return withTime && match[2] ? `${match[1]} ${match[2]}` : match[1];
}

function friendlyError(error) {
  const raw = String(error?.message || '').trim();
  if (!raw || /failed to fetch|networkerror|load failed|fetch failed/i.test(raw)) {
    return '无法连接服务，请确认服务可用后重试。';
  }
  if (/^\d{3}\s/.test(raw)) return '服务暂时不可用，请稍后重试。';
  return raw;
}

/* ---- 通用片段 ---- */
function panelEmpty(title, message, icon = 'fa-circle-info', compact = false) {
  return `<div class="panel-empty${compact ? ' is-compact' : ''}"><span class="panel-empty-icon"><i class="fas ${icon}"></i></span><strong>${escapeHtml(title)}</strong><p>${escapeHtml(message)}</p></div>`;
}

function panelError(title, message, actionLabel = '', action = '') {
  const button = actionLabel ? `<button type="button" class="btn btn-secondary btn-sm" data-action="${escapeHtml(action)}"><i class="fas fa-sync-alt"></i> ${escapeHtml(actionLabel)}</button>` : '';
  return `<div class="panel-empty is-error"><span class="panel-empty-icon"><i class="fas fa-circle-exclamation"></i></span><strong>${escapeHtml(title)}</strong><p>${escapeHtml(message)}</p>${button}</div>`;
}

function evidenceButton(evidence, text = '查看原文', context = '') {
  if (!evidence || !evidence.quote) return '<span class="evidence-missing">未标注原文出处</span>';
  const payload = encodeURIComponent(JSON.stringify({ ...evidence, label: context }));
  return `<button type="button" class="evidence-link" data-evidence="${payload}"><i class="fas fa-quote-left"></i> ${escapeHtml(text)}</button>`;
}

function setStatusNode(id, text, tone = 'info', hidden = false) {
  const node = byId(id);
  if (!node) return;
  if (hidden || !text) {
    node.classList.add('hidden');
    node.textContent = '';
    return;
  }
  node.classList.remove('hidden');
  node.className = `dialog-status${tone === 'error' ? ' is-error' : tone === 'success' ? ' is-success' : ''}`;
  node.innerHTML = `<i class="fas ${tone === 'error' ? 'fa-circle-exclamation' : tone === 'success' ? 'fa-circle-check' : 'fa-circle-notch fa-spin'}"></i><span>${escapeHtml(text)}</span>`;
}

function toast(message, tone = 'info') {
  const node = byId('toast');
  const icon = tone === 'success' ? 'fa-circle-check' : tone === 'error' ? 'fa-circle-exclamation' : 'fa-circle-info';
  node.className = `toast${tone === 'success' ? ' is-success' : tone === 'error' ? ' is-error' : ''}`;
  node.innerHTML = `<i class="fas ${icon}"></i><span>${escapeHtml(message)}</span>`;
  window.clearTimeout(toast.timer);
  toast.timer = window.setTimeout(() => node.classList.add('hidden'), 3200);
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, { cache: 'no-store', ...options, headers: { ...(options.headers || {}) } });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || payload.error || `${response.status} ${response.statusText}`);
  return payload;
}

/* ---- 合同台账 ---- */
function renderListLoading() {
  byId('contractCount').textContent = '加载中…';
  byId('contractList').innerHTML = Array.from({ length: 3 }).map(() =>
    '<div class="skeleton-item"><div class="skeleton-line"></div><div class="skeleton-line w-70"></div><div class="skeleton-line w-40"></div></div>'
  ).join('');
}

async function loadContracts() {
  const params = new URLSearchParams();
  if (byId('statusFilter').value) params.set('status', byId('statusFilter').value);
  if (byId('riskFilter').value) params.set('risk_level', byId('riskFilter').value);
  if (!contractState.items.length) renderListLoading();
  try {
    const payload = await requestJson(`/api/contracts?${params}`);
    contractState.items = payload.contracts || [];
    renderContractList();
    if (!contractState.selectedId && contractState.items.length) selectContract(contractState.items[0].id);
  } catch (error) {
    contractState.items = [];
    byId('contractCount').textContent = '加载失败';
    byId('contractList').innerHTML = panelError('合同台账加载失败', friendlyError(error), '重新加载', 'reload-list');
  }
}

function contractStatusClass(status) {
  return ['active', 'draft', 'expired', 'terminated', 'archived'].includes(status) ? status : '';
}

function renderContractList() {
  const root = byId('contractList');
  byId('contractCount').textContent = contractState.items.length ? `${contractState.items.length} 份合同` : '暂无合同';
  root.replaceChildren();
  if (!contractState.items.length) {
    const hasFilter = byId('statusFilter').value || byId('riskFilter').value;
    root.innerHTML = hasFilter
      ? panelEmpty('没有符合条件的合同', '可调整状态或风险筛选条件，或清空筛选后查看全部合同。', 'fa-filter', true)
      : panelEmpty('暂无合同', '上传第一份合同后即可查看条款、履约与风险分析结果。', 'fa-file-contract', true);
    return;
  }
  contractState.items.forEach(item => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `contract-item${item.id === contractState.selectedId ? ' active' : ''}`;
    const riskBadge = item.overall_risk_level
      ? `<span class="tag ${escapeHtml(item.overall_risk_level)}">${label(item.overall_risk_level)}风险</span>`
      : '<span class="tag">风险待评估</span>';
    const statusBadge = `<span class="tag ${contractStatusClass(item.status)}">${label(item.status)}</span>`;
    const analysisBadge = item.processing_status && item.processing_status !== 'ready'
      ? `<span class="status-badge ${escapeHtml(item.processing_status)}">${label(item.processing_status)}</span>`
      : '';
    const updated = formatDate(item.updated_at, true);
    button.innerHTML = `<strong class="contract-item-name">${escapeHtml(item.contract_name)}</strong>`
      + `<div class="contract-item-badges">${statusBadge}${riskBadge}${analysisBadge}</div>`
      + (updated ? `<div class="contract-item-meta"><i class="fas fa-clock"></i><span>更新于 ${escapeHtml(updated)}</span></div>` : '');
    button.addEventListener('click', () => selectContract(item.id));
    root.appendChild(button);
  });
}

function setBanner(tone, title, message, action = '') {
  const node = byId('detailBanner');
  if (!tone) {
    node.className = 'detail-banner hidden';
    node.innerHTML = '';
    return;
  }
  const defaultAction = { running: 'reload-detail', failed: 'reanalyze', degraded: 'open-documents' }[tone] || '';
  const actions = {
    'reload-detail': '<button type="button" class="btn btn-secondary btn-sm" data-action="reload-detail"><i class="fas fa-sync-alt"></i> 刷新状态</button>',
    reanalyze: '<button type="button" class="btn btn-secondary btn-sm" data-action="reanalyze"><i class="fas fa-search"></i> 重新分析</button>',
    'open-documents': '<button type="button" class="btn btn-secondary btn-sm" data-action="open-documents"><i class="fas fa-folder-tree"></i> 查看分析记录</button>'
  };
  const icon = tone === 'running' ? 'fa-circle-notch fa-spin' : tone === 'failed' ? 'fa-circle-exclamation' : 'fa-triangle-exclamation';
  node.className = `detail-banner is-${tone}`;
  node.innerHTML = `<i class="fas ${icon}"></i><div><strong>${escapeHtml(title)}</strong><p>${escapeHtml(message)}</p></div>${actions[action || defaultAction] || ''}`;
}

function renderDetailUnavailable(message) {
  byId('riskOverview').innerHTML = '';
  ['basicFields', 'summarySections'].forEach(id => {
    byId(id).innerHTML = panelEmpty('暂无合同数据', message, 'fa-circle-exclamation');
  });
  byId('timelineSummary').textContent = '';
  byId('timeline').innerHTML = panelEmpty('暂无合同数据', message, 'fa-clock');
  byId('obligationSummary').textContent = '';
  byId('obligationRows').innerHTML = `<tr><td colspan="8">${panelEmpty('暂无合同数据', message, 'fa-clipboard-check')}</td></tr>`;
  byId('riskSummary').textContent = '';
  byId('riskCards').innerHTML = panelEmpty('暂无合同数据', message, 'fa-triangle-exclamation');
  byId('documentList').innerHTML = panelEmpty('暂无合同数据', message, 'fa-folder-tree', true);
  byId('analysisList').innerHTML = panelEmpty('暂无合同数据', message, 'fa-history', true);
}

async function selectContract(contractId) {
  contractState.selectedId = contractId;
  renderContractList();
  byId('emptyDetail').classList.add('hidden');
  byId('contractDetail').classList.remove('hidden');
  const item = contractState.items.find(entry => entry.id === contractId);
  byId('detailName').textContent = item?.contract_name || '合同详情';
  byId('statusBadge').className = 'status-badge hidden';
  byId('processBadge').className = 'status-badge hidden';
  byId('detailMeta').innerHTML = '<span class="detail-meta-item"><i class="fas fa-circle-notch fa-spin"></i> 正在加载合同数据…</span>';
  setBanner(null);
  try {
    contractState.detail = await requestJson(`/api/contracts/${contractId}`);
    contractState.sessionId = `contract:${contractId}:${crypto.randomUUID()}`;
    renderDetail(contractState.detail);
  } catch (error) {
    contractState.detail = null;
    byId('detailMeta').innerHTML = '';
    setBanner('failed', '合同数据加载失败', friendlyError(error), 'reload-detail');
    renderDetailUnavailable(friendlyError(error));
    toast(friendlyError(error), 'error');
  }
}

function renderDetail(detail) {
  const info = detail.analysis?.extraction?.basic_info || {};
  byId('detailName').textContent = detail.contract_name;
  const meta = [
    ['fa-tags', '类型', detail.contract_type || fieldText(info.contract_type) || '未抽取'],
    ['fa-circle-info', '编号', detail.contract_number || fieldText(info.contract_number) || '未抽取'],
    ['fa-clock', '更新', formatDate(detail.updated_at, true) || '—']
  ];
  byId('detailMeta').innerHTML = meta.map(([icon, name, value]) =>
    `<span class="detail-meta-item${value === '未抽取' ? ' is-empty' : ''}"><i class="fas ${icon}"></i>${escapeHtml(name)}：${escapeHtml(value)}</span>`
  ).join('');
  renderStatusBadges(detail);
  renderBanner(detail);
  renderOverview(detail);
  renderBasicInfo(info);
  renderSummary(detail.analysis?.summary || [], detail.analysis || null);
  renderTimeline(detail.timeline || []);
  renderObligations(detail.obligations || []);
  renderRisks(detail.risks || []);
  renderRecords(detail.documents || [], detail.analysis_runs || []);
  byId('workdayNote').textContent = detail.workday_calculation_note || '';
}

function renderStatusBadges(detail) {
  const statusTone = { active: 'is-success', expired: 'is-warning', terminated: '', draft: '', archived: '' }[detail.status] ?? '';
  const statusNode = byId('statusBadge');
  statusNode.className = `status-badge ${statusTone}`.trim();
  statusNode.innerHTML = `<span class="badge-dot"></span>${escapeHtml(label(detail.status))}`;

  const live = ['parsing', 'analyzing'].includes(detail.processing_status);
  const processTone = { ready: 'is-success', failed: 'is-danger', parsing: 'is-info is-live', analyzing: 'is-info is-live', parsed: '', uploaded: '' }[detail.processing_status] ?? '';
  const processNode = byId('processBadge');
  processNode.className = `status-badge ${processTone}`.trim();
  processNode.innerHTML = `${live ? '<span class="badge-dot"></span>' : ''}${escapeHtml(label(detail.processing_status))}`;
}

function renderBanner(detail) {
  const analysis = detail.analysis || null;
  const failedStages = Object.entries(analysis?.stage_status || {})
    .filter(([, value]) => value === 'failed')
    .map(([stage]) => ANALYSIS_STAGE_LABELS[stage] || stage);
  if (['parsing', 'analyzing'].includes(detail.processing_status)) {
    setBanner('running', '正在处理本合同', '系统正在解析文档、抽取要素并分析风险，完成后可刷新查看最新结果。');
    return;
  }
  if (detail.processing_status === 'failed') {
    setBanner('failed', '合同处理未完成', '最近一次解析或分析未成功，其他已有数据仍然可用；可重新分析后再次查看。');
    return;
  }
  if (analysis?.is_degraded) {
    const detailText = failedStages.length ? `未成功的阶段：${failedStages.join('、')}。` : '';
    setBanner('degraded', '部分分析阶段未成功', `${detailText}其余分析结果仍然可用，可在「文件与记录」中查看分析记录。`);
    return;
  }
  setBanner(null);
}

/* ---- 业务概览 ---- */
function metricCard({ name, value, foot, icon, tone = '', empty = false }) {
  return `<div class="metric${tone ? ` tone-${tone}` : ''}">`
    + `<div class="metric-head"><span class="metric-label">${escapeHtml(name)}</span><span class="metric-icon"><i class="fas ${icon}"></i></span></div>`
    + `<div class="metric-value${empty ? ' is-empty' : ''}">${escapeHtml(value)}</div>`
    + `<div class="metric-foot">${escapeHtml(foot)}</div>`
    + '</div>';
}

function riskTone(level) {
  return { critical: 'critical', high: 'danger', medium: 'warning', low: 'primary' }[level] || 'neutral';
}

function renderOverview(detail) {
  const info = detail.analysis?.extraction?.basic_info || {};
  const dashboard = detail.risk_dashboard || {};
  const obligations = detail.obligations || [];
  const risks = detail.risks || [];

  const amount = fieldText(info.contract_amount);
  const currency = fieldText(info.currency);
  const amountText = amount ? `${amount}${currency ? ` ${currency}` : ''}` : EMPTY_TEXT;

  const keyDate = [['termination_date', '合同终止日期'], ['effective_date', '合同生效日期'], ['signing_date', '合同签署日期']]
    .map(([key, name]) => [fieldText(info[key]), name])
    .find(([value]) => value);

  const riskCounts = ['critical', 'high', 'medium', 'low'];
  const riskTotal = riskCounts.reduce((sum, level) => sum + (dashboard[`${level}_count`] || 0), 0);
  const riskOpen = risks.filter(item => item.status === 'open').length;

  const openObligations = obligations.filter(item => PENDING_STATUSES.includes(item.status)).length;
  const overdue = dashboard.overdue_obligation_count || 0;
  const upcoming = dashboard.upcoming_obligation_count || 0;

  const analysis = detail.analysis || null;
  const degradedStages = Object.entries(analysis?.stage_status || {})
    .filter(([, value]) => value === 'failed')
    .map(([stage]) => ANALYSIS_STAGE_LABELS[stage] || stage);
  const analysisTone = { ready: 'success', failed: 'danger', parsing: 'primary', analyzing: 'primary' }[detail.processing_status] || '';

  byId('riskOverview').innerHTML = [
    metricCard({
      name: '合同金额', value: amountText, icon: 'fa-coins', tone: amount ? 'primary' : '', empty: !amount,
      foot: amount ? '来自合同要素抽取结果' : '合同中未明确金额'
    }),
    metricCard({
      name: '当前状态', value: label(detail.status), icon: 'fa-file-circle-check',
      tone: detail.status === 'active' ? 'success' : '', foot: `更新于 ${formatDate(detail.updated_at, true) || '—'}`
    }),
    metricCard({
      name: '风险情况', value: riskTotal ? `${riskTotal} 项` : '暂无风险', icon: 'fa-triangle-exclamation',
      tone: riskTotal ? riskTone(detail.overall_risk_level) : '', empty: !riskTotal,
      foot: riskTotal ? `最高等级 ${label(detail.overall_risk_level)} · 待处理 ${riskOpen} 项` : '当前分析未识别到风险'
    }),
    metricCard({
      name: '待履约事项', value: upcoming ? `近期 ${upcoming} 项` : '暂无', icon: 'fa-clipboard-check',
      tone: overdue ? 'danger' : upcoming ? 'warning' : '', empty: !upcoming,
      foot: obligations.length ? `未完成 ${openObligations} 项 · 已逾期 ${overdue} 项` : '尚未识别到履约事项'
    }),
    metricCard({
      name: '关键日期', value: keyDate ? keyDate[0] : EMPTY_TEXT, icon: 'fa-calendar-check',
      tone: keyDate ? 'primary' : '', empty: !keyDate,
      foot: keyDate ? keyDate[1] : '合同中未明确关键日期'
    }),
    metricCard({
      name: '分析状态', value: label(detail.processing_status), icon: 'fa-history', tone: analysisTone,
      foot: analysis?.is_degraded && degradedStages.length
        ? `部分阶段未成功：${degradedStages.join('、')}`
        : `更新于 ${formatDate(detail.updated_at, true) || '—'}`
    })
  ].join('');
}

/* ---- 关键字段与摘要 ---- */
function renderBasicInfo(info) {
  const fields = [
    ['合同名称', 'contract_name'], ['合同编号', 'contract_number'], ['合同类型', 'contract_type'],
    ['甲方', 'party_a'], ['乙方', 'party_b'], ['合同金额', 'contract_amount'], ['币种', 'currency'],
    ['签署日期', 'signing_date'], ['生效日期', 'effective_date'], ['终止日期', 'termination_date'], ['合同期限', 'contract_term']
  ];
  byId('basicFields').innerHTML = fields.map(([name, key]) => {
    const field = info[key];
    const text = fieldText(field);
    const review = field?.needs_review ? '<span class="chip-review"><i class="fas fa-circle-exclamation"></i> 待复核</span>' : '';
    return `<div class="key-value">`
      + `<div class="key-value-head"><span class="key-value-label">${escapeHtml(name)}</span>${review}</div>`
      + `<div class="key-value-value${text ? '' : ' is-empty'}">${escapeHtml(text || EMPTY_TEXT)}</div>`
      + `<div class="key-value-foot">${evidenceButton(field?.source_evidence, '查看原文', `关键字段 · ${name}`)}</div>`
      + '</div>';
  }).join('');
}

function renderSummary(items, analysis) {
  const summaryFailed = analysis?.stage_status?.summary === 'failed';
  const title = !analysis
    ? '尚无分析结果'
    : summaryFailed
      ? '摘要生成未成功'
      : '本次分析未生成摘要';
  const message = !analysis
    ? '上传合同并完成分析后，这里会展示按主题归纳的合同要点。'
    : summaryFailed
      ? '摘要阶段未成功，其他已完成的抽取与风险结果仍然可用；可在「文件与记录」查看分析记录后重试。'
      : '本次分析结果中没有摘要内容，可查看关键字段与履约事项了解合同要点。';
  byId('summarySections').innerHTML = items.length
    ? items.map(item => `<article class="summary-item"><h4>${escapeHtml(item.title)}</h4><p>${escapeHtml(item.content)}</p><div class="summary-item-foot">${evidenceButton((item.source_evidence || [])[0], '查看原文', `合同摘要 · ${item.title}`)}</div></article>`).join('')
    : panelEmpty(title, message, 'fa-file-lines');
}

/* ---- 履约时间轴 ---- */
function renderTimeline(items) {
  const counts = {
    completed: items.filter(item => item.status === 'completed').length,
    overdue: items.filter(item => item.status === 'overdue').length,
    waiting: items.filter(item => item.status === 'waiting_trigger').length
  };
  byId('timelineSummary').textContent = items.length
    ? `共 ${items.length} 个节点 · 已完成 ${counts.completed} · 已逾期 ${counts.overdue} · 待触发 ${counts.waiting}`
    : '';
  byId('timeline').innerHTML = items.length ? items.map(item => {
    const milestone = item.type === 'contract';
    const name = item.type === 'event' ? label(item.name) : item.name;
    const hasDate = Boolean(item.date);
    const meta = [];
    if (item.responsible_party) meta.push(`<span><i class="fas fa-user"></i>${escapeHtml(item.responsible_party)}</span>`);
    if (item.obligation_type) meta.push(`<span><i class="fas fa-tags"></i>${escapeHtml(label(item.obligation_type))}</span>`);
    if (milestone) meta.push('<span><i class="fas fa-file-contract"></i>合同节点</span>');
    return `<article class="timeline-item status-${escapeHtml(item.status || 'pending')}${milestone ? ' is-milestone' : ''}">`
      + `<div class="timeline-date${hasDate ? '' : ' is-empty'}"><time>${escapeHtml(item.display_date || '待触发')}</time>${hasDate ? '' : '<span>按事件触发</span>'}</div>`
      + `<div class="timeline-rail"><span class="timeline-dot"></span></div>`
      + '<div class="timeline-body">'
      + `<div class="timeline-head"><h4>${escapeHtml(name)}</h4><span class="tag ${escapeHtml(item.status || '')}">${escapeHtml(label(item.status))}</span></div>`
      + (meta.length ? `<div class="timeline-meta">${meta.join('')}</div>` : '')
      + (item.description ? `<p>${escapeHtml(item.description)}</p>` : '')
      + `<div class="timeline-foot">${evidenceButton(item.source_evidence, '查看原文', `时间轴 · ${name}`)}</div>`
      + '</div></article>';
  }).join('') : panelEmpty('未识别到时间节点', '当前分析结果中没有可展示的合同节点或履约义务，可先完成合同分析或记录履约事件。', 'fa-clock');
}

/* ---- 履约事项 ---- */
function renderObligations(items) {
  const statuses = ['pending', 'waiting_trigger', 'in_progress', 'completed', 'overdue', 'waived', 'cancelled'];
  const overdue = items.filter(item => item.status === 'overdue').length;
  const completed = items.filter(item => item.status === 'completed').length;
  byId('obligationSummary').textContent = items.length
    ? `共 ${items.length} 项履约事项 · 已完成 ${completed} · 已逾期 ${overdue}`
    : '';
  byId('obligationRows').innerHTML = items.length ? items.map(item => {
    const planned = item.planned_date || '';
    const ruleText = item.time_rule?.original_text || '';
    return '<tr>'
      + `<td class="cell-primary" data-label="事项"><div class="cell-title">${escapeHtml(item.title)}</div>${item.description ? `<div class="cell-desc">${escapeHtml(item.description)}</div>` : ''}</td>`
      + `<td data-label="状态"><span class="tag ${escapeHtml(item.status)}">${escapeHtml(label(item.status))}</span></td>`
      + `<td data-label="风险"><span class="tag ${escapeHtml(item.risk_level)}">${escapeHtml(label(item.risk_level))}</span></td>`
      + `<td data-label="责任方" class="cell-value${item.responsible_party ? '' : ' is-empty'}">${escapeHtml(item.responsible_party || EMPTY_TEXT)}</td>`
      + `<td data-label="计划时间"><div class="cell-date${planned ? '' : ' is-empty'}">${escapeHtml(planned || '待触发')}</div>${ruleText && ruleText !== planned ? `<div class="cell-note">${escapeHtml(ruleText)}</div>` : ''}</td>`
      + `<td data-label="类型" class="cell-muted">${escapeHtml(label(item.obligation_type))}</td>`
      + `<td data-label="依据">${evidenceButton(item.evidence || item.source_evidence, '查看原文', `履约事项 · ${item.title}`)}</td>`
      + `<td data-label="更新状态"><select class="inline-select obligation-status" data-id="${escapeHtml(item.id)}" aria-label="更新履约状态">${statuses.map(status => `<option value="${status}"${status === item.status ? ' selected' : ''}>${escapeHtml(label(status))}</option>`).join('')}</select></td>`
      + '</tr>';
  }).join('') : `<tr><td colspan="8">${panelEmpty('未识别到履约事项', '当前分析结果中没有可展示的履约义务，可重新分析或先在时间轴记录履约事件。', 'fa-clipboard-check')}</td></tr>`;
  document.querySelectorAll('.obligation-status').forEach(select => select.addEventListener('change', async () => {
    select.disabled = true;
    try {
      const detail = await requestJson(`/api/contracts/${contractState.selectedId}/obligations/${select.dataset.id}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ status: select.value }) });
      contractState.detail = detail;
      renderDetail(detail);
      toast('履约状态已更新，相关风险已重新计算', 'success');
    } catch (error) {
      toast(friendlyError(error), 'error');
      select.disabled = false;
      select.value = contractState.detail?.obligations?.find(entry => entry.id === select.dataset.id)?.status || select.value;
    }
  }));
}

/* ---- 风险清单 ---- */
function renderRisks(items) {
  const statuses = [['open', '待处理'], ['acknowledged', '已确认'], ['resolved', '已解决'], ['ignored', '已忽略']];
  const open = items.filter(item => item.status === 'open').length;
  const resolved = items.filter(item => item.status === 'resolved').length;
  const breakdown = ['critical', 'high', 'medium', 'low']
    .map(level => [level, items.filter(item => item.risk_level === level).length])
    .filter(([, count]) => count > 0)
    .map(([level, count]) => `${label(level)} ${count}`)
    .join(' · ');
  byId('riskSummary').textContent = items.length
    ? `共 ${items.length} 项风险 · ${breakdown} · 待处理 ${open} · 已解决 ${resolved}`
    : '';
  byId('riskCards').innerHTML = items.length ? items.map(item => {
    const supporting = (item.supporting_evidence || []).length;
    return `<article class="risk-card risk-${escapeHtml(item.risk_level)}">`
      + '<div class="risk-card-head">'
      + `<div class="risk-title"><h4>${escapeHtml(item.risk_name)}</h4><span class="tag ${escapeHtml(item.risk_level)}">${escapeHtml(label(item.risk_level))}</span></div>`
      + `<div class="risk-card-actions"><select class="inline-select risk-status" data-id="${escapeHtml(item.id)}" aria-label="更新风险处理状态">${statuses.map(([status, text]) => `<option value="${status}"${status === item.status ? ' selected' : ''}>${text}</option>`).join('')}</select></div>`
      + '</div>'
      + `<div class="risk-block"><span class="risk-block-label">风险描述</span><p>${escapeHtml(item.description)}</p></div>`
      + (item.reason ? `<div class="risk-block"><span class="risk-block-label">风险原因</span><p>${escapeHtml(item.reason)}</p></div>` : '')
      + (item.suggestion ? `<div class="risk-suggestion"><span class="risk-block-label">处置建议</span><p>${escapeHtml(item.suggestion)}</p></div>` : '')
      + '<div class="risk-card-foot">'
      + `<div class="risk-meta"><span>${escapeHtml(label(item.risk_source))}</span><span class="dot-sep">·</span><span>${item.needs_review ? '待复核' : '已校验原文证据'}</span>${supporting ? `<span class="dot-sep">·</span><span>另有 ${supporting} 处原文支持</span>` : ''}</div>`
      + `<div>${evidenceButton(item.evidence || item.source_evidence, '查看原文依据', `风险 · ${item.risk_name}`)}</div>`
      + '</div></article>';
  }).join('') : panelEmpty('当前分析未识别到风险', '规则与模型均未在本合同中识别到风险条目；重新分析后可再次查看结果。', 'fa-shield-alt');
  document.querySelectorAll('.risk-status').forEach(select => select.addEventListener('change', async () => {
    select.disabled = true;
    try {
      const detail = await requestJson(`/api/contracts/${contractState.selectedId}/risks/${select.dataset.id}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ status: select.value }) });
      contractState.detail = detail;
      renderDetail(detail);
      toast('风险处理状态已更新', 'success');
    } catch (error) {
      toast(friendlyError(error), 'error');
      select.disabled = false;
    }
  }));
}

/* ---- 文件与分析记录 ---- */
function renderRecords(documents, analyses) {
  byId('documentList').innerHTML = documents.length ? documents.map(item => {
    const meta = [
      label(item.document_type),
      `版本 ${item.document_version || '1'}`,
      `${item.page_count || 0} 页 · ${item.chunk_count || 0} 片段`,
      label(item.parse_status)
    ].filter(text => text && text !== '—');
    return `<div class="record-item"><strong>${escapeHtml(item.file_name)}</strong><div class="record-item-meta">${meta.map(text => `<span>${escapeHtml(text)}</span>`).join('')}</div></div>`;
  }).join('') : panelEmpty('暂无合同文件', '上传合同后，这里会显示文件解析状态与切片结果。', 'fa-folder-tree', true);

  byId('analysisList').innerHTML = analyses.length ? analyses.map(item => {
    const failedStages = (item.stage_errors || []).map(entry => ANALYSIS_STAGE_LABELS[entry.stage] || entry.stage);
    const started = formatDate(item.started_at, true);
    const finished = item.finished_at ? formatDate(item.finished_at, true) : '';
    const time = value => (value ? value.slice(-5) : '');
    const meta = [label(item.status)];
    if (started && finished) meta.push(`${time(started)} → ${time(finished)}`);
    else if (started) meta.push(`${time(started)} 开始`);
    if (item.analysis_version) meta.push(`版本 ${item.analysis_version}`);
    if (item.is_degraded) meta.push('部分阶段未成功');
    return `<div class="record-item${item.status === 'failed' ? ' is-error' : ''}"><strong>分析记录${started ? ` · ${escapeHtml(started)}` : ''}</strong>`
      + `<div class="record-item-meta">${meta.map(text => `<span>${escapeHtml(text)}</span>`).join('')}</div>`
      + (failedStages.length ? `<div class="record-item-meta"><span>未成功阶段：${escapeHtml(failedStages.join('、'))}</span></div>` : '')
      + '</div>';
  }).join('') : panelEmpty('暂无分析记录', '完成一次合同分析后，这里会显示各阶段执行结果。', 'fa-history', true);
}

/* ---- 履约事件 ---- */
async function submitEvent(event) {
  event.preventDefault();
  if (!contractState.selectedId) return;
  const form = event.currentTarget;
  const payload = Object.fromEntries(new FormData(form));
  setStatusNode('eventStatus', '正在保存事件并重新计算关联义务…', 'info');
  try {
    const detail = await requestJson(`/api/contracts/${contractState.selectedId}/events`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    contractState.detail = detail;
    renderDetail(detail);
    byId('eventDialog').close();
    form.reset();
    setStatusNode('eventStatus', '', 'info', true);
    toast('事件已保存，履约日期与规则风险已重新计算', 'success');
  } catch (error) {
    setStatusNode('eventStatus', friendlyError(error), 'error');
  }
}

/* ---- 上传与重新分析 ---- */
function openUploadDialog() {
  setStatusNode('uploadStatus', '', 'info', true);
  byId('uploadDialog').showModal();
}

async function submitUpload(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const data = new FormData(form);
  data.set('auto_analyze', form.elements.auto_analyze.checked ? 'true' : 'false');
  setStatusNode('uploadStatus', '正在解析、向量化并分析合同，请稍候…', 'info');
  const submit = form.querySelector('[type="submit"]');
  submit.disabled = true;
  try {
    const detail = await requestJson('/api/contracts', { method: 'POST', body: data });
    byId('uploadDialog').close();
    form.reset();
    setStatusNode('uploadStatus', '', 'info', true);
    await loadContracts();
    await selectContract(detail.id);
    toast('合同上传处理完成', 'success');
  } catch (error) {
    setStatusNode('uploadStatus', friendlyError(error), 'error');
  } finally {
    submit.disabled = false;
  }
}

async function reanalyze() {
  if (!contractState.selectedId) return;
  const button = byId('reanalyzeBtn');
  button.disabled = true;
  button.innerHTML = '<i class="fas fa-circle-notch fa-spin"></i> 分析中…';
  setBanner('running', '正在重新分析合同', '系统正在重新执行要素抽取、义务识别与风险分析。');
  try {
    contractState.detail = await requestJson(`/api/contracts/${contractState.selectedId}/analyze`, { method: 'POST' });
    renderDetail(contractState.detail);
    await loadContracts();
    toast('合同重新分析完成', 'success');
  } catch (error) {
    setBanner('failed', '重新分析未完成', friendlyError(error));
    toast(friendlyError(error), 'error');
  } finally {
    button.disabled = false;
    button.innerHTML = '<i class="fas fa-search"></i> 重新分析';
  }
}

/* ---- 原文依据 ---- */
function evidenceDocumentName(evidence) {
  if (evidence.document_name) return evidence.document_name;
  const documents = contractState.detail?.documents || [];
  const document = documents.find(item => item.id === evidence.document_id);
  return document?.file_name || '';
}

function openEvidence(evidence) {
  const body = byId('sourceBody');
  const documentName = evidenceDocumentName(evidence);
  const clause = [evidence.clause_no, evidence.clause_title].filter(Boolean).join(' ');
  const meta = [
    ['来源文件', documentName || '合同文件'],
    ['页码', evidence.page_number ? `第 ${evidence.page_number} 页` : '未标注页码']
  ];
  if (clause) meta.push(['条款', clause]);
  if (evidence.section) meta.push(['章节', evidence.section]);
  body.innerHTML = (evidence.label ? `<div class="source-context"><i class="fas fa-link"></i> ${escapeHtml(evidence.label)}</div>` : '')
    + `<dl class="source-meta">${meta.map(([name, value]) => `<div><dt>${escapeHtml(name)}</dt><dd>${escapeHtml(value)}</dd></div>`).join('')}</dl>`
    + `<figure class="source-quote"><blockquote>${escapeHtml(evidence.quote)}</blockquote></figure>`
    + '<div class="source-actions">'
    + '<span class="source-note"><i class="fas fa-circle-check"></i> 引用内容摘自合同原文，可用于核对结论</span>'
    + '<button type="button" class="btn btn-secondary btn-sm" data-copy-evidence><i class="fas fa-copy"></i> 复制引用</button>'
    + '</div>';
  body.dataset.copyText = `${evidence.quote}${documentName ? `\n——${documentName}` : ''}${evidence.page_number ? ` 第 ${evidence.page_number} 页` : ''}`;
  byId('sourceDialog').showModal();
}

async function copyEvidence() {
  const text = byId('sourceBody').dataset.copyText || '';
  try {
    await navigator.clipboard.writeText(text);
    toast('引用已复制', 'success');
  } catch (error) {
    toast('当前浏览器不支持自动复制，请手动选择原文内容。', 'error');
  }
}

/* ---- 合同问答 ---- */
function scrollQa() {
  const node = byId('qaMessages');
  node.scrollTop = node.scrollHeight;
}

function qaRow(role) {
  const row = document.createElement('div');
  row.className = `qa-row ${role}`;
  const avatar = document.createElement('span');
  avatar.className = `qa-avatar ${role}`;
  avatar.innerHTML = role === 'assistant' ? '<i class="fas fa-sparkles"></i>' : '<i class="fas fa-user"></i>';
  const wrap = document.createElement('div');
  wrap.className = 'qa-bubble-wrap';
  const roleLabel = document.createElement('span');
  roleLabel.className = 'qa-role';
  roleLabel.textContent = role === 'assistant' ? '合同助手' : '我';
  const bubble = document.createElement('div');
  bubble.className = `qa-message ${role}`;
  wrap.append(roleLabel, bubble);
  row.append(avatar, wrap);
  byId('qaMessages').appendChild(row);
  scrollQa();
  return bubble;
}

function appendQaMessage(role, content) {
  const bubble = qaRow(role);
  if (role === 'assistant' && window.marked && window.DOMPurify) bubble.innerHTML = DOMPurify.sanitize(marked.parse(content || ''));
  else bubble.textContent = content;
  scrollQa();
  return bubble;
}

function setQaPending(bubble, text) {
  bubble.classList.add('is-pending');
  bubble.innerHTML = '<span class="qa-typing"><b></b><b></b><b></b></span><span class="qa-pending-text"></span>';
  bubble.querySelector('.qa-pending-text').textContent = text;
  scrollQa();
}

function qaSourceList(sources) {
  const list = document.createElement('div');
  list.className = 'qa-source-list';
  const title = document.createElement('div');
  title.className = 'qa-source-title';
  title.innerHTML = '<i class="fas fa-quote-left"></i> 来源依据';
  list.appendChild(title);
  sources.forEach((source, index) => {
    const row = document.createElement('div');
    row.className = 'qa-source';
    const badge = document.createElement('span');
    badge.className = 'qa-source-index';
    badge.textContent = String(index + 1);
    const holder = document.createElement('div');
    holder.innerHTML = evidenceButton({
      document_id: source.document_id,
      document_name: source.document_name,
      chunk_id: source.chunk_id,
      page_number: source.page,
      section: source.metadata?.section,
      clause_no: source.metadata?.clause_no || source.metadata?.clause_number,
      quote: source.source_text
    }, source.citation || (source.page ? `第 ${source.page} 页` : '查看原文'), '合同问答引用');
    row.append(badge, holder.firstElementChild || holder);
    list.appendChild(row);
  });
  return list;
}

function setQaStatus(text) {
  const node = byId('qaStatus');
  if (!text) {
    node.classList.add('hidden');
    node.innerHTML = '';
    return;
  }
  node.classList.remove('hidden');
  node.innerHTML = `<i class="fas fa-circle-notch fa-spin"></i>${escapeHtml(text)}`;
}

function sendContractQuestion() {
  const query = byId('qaInput').value.trim();
  if (!query || !contractState.selectedId || byId('qaSend').disabled) return;
  byId('qaMessages').querySelector('.qa-welcome')?.remove();
  appendQaMessage('user', query);
  const bubble = qaRow('assistant');
  setQaPending(bubble, '正在检索当前合同条款…');
  byId('qaInput').value = '';
  autoGrowQaInput();
  byId('qaSend').disabled = true;
  setQaStatus('正在生成回答…');
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const socket = new WebSocket(`${protocol}//${location.host}/api/contracts/${contractState.selectedId}/stream`);
  contractState.socket = socket;
  let answer = '';
  let finished = false;
  socket.onopen = () => socket.send(JSON.stringify({ query, session_id: contractState.sessionId }));
  socket.onmessage = event => {
    const data = JSON.parse(event.data);
    if (data.session_id) contractState.sessionId = data.session_id;
    if (data.type === 'status' && !answer) setQaPending(bubble, data.message || '正在处理…');
    if (data.type === 'token') {
      answer += data.token || '';
      bubble.classList.remove('is-pending');
      bubble.innerHTML = DOMPurify.sanitize(marked.parse(answer));
      scrollQa();
    }
    if (data.type === 'end') {
      finished = true;
      bubble.classList.remove('is-pending');
      if (!answer) bubble.textContent = '根据当前合同内容，未发现明确约定。';
      const sources = data.sources || [];
      if (sources.length) {
        bubble.appendChild(qaSourceList(sources));
        scrollQa();
      }
      socket.close();
    }
    if (data.type === 'error') {
      finished = true;
      bubble.classList.remove('is-pending');
      bubble.classList.add('is-error');
      bubble.textContent = data.error || '问答未能完成，请重新提问。';
      byId('qaSend').disabled = false;
      setQaStatus('');
      socket.close();
    }
  };
  socket.onerror = () => {
    finished = true;
    bubble.classList.remove('is-pending');
    bubble.classList.add('is-error');
    bubble.textContent = '问答连接失败，请确认服务可用后重试。';
    byId('qaSend').disabled = false;
    setQaStatus('');
  };
  socket.onclose = () => {
    byId('qaSend').disabled = false;
    setQaStatus('');
    if (finished) return;
    finished = true;
    bubble.classList.remove('is-pending');
    bubble.classList.add('is-error');
    bubble.textContent = '问答连接已中断，请重新发送问题。';
  };
}

function autoGrowQaInput() {
  const input = byId('qaInput');
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, 140)}px`;
}

function useQaSuggestion(button) {
  const input = byId('qaInput');
  input.value = button.textContent.trim();
  input.focus();
  autoGrowQaInput();
}

/* ---- 事件绑定 ---- */
document.addEventListener('DOMContentLoaded', () => {
  byId('openUpload').addEventListener('click', openUploadDialog);
  byId('openUploadEmpty').addEventListener('click', openUploadDialog);
  document.querySelectorAll('[data-close]').forEach(button => button.addEventListener('click', () => button.closest('dialog').close()));
  byId('uploadForm').addEventListener('submit', submitUpload);
  byId('openEvent').addEventListener('click', () => {
    setStatusNode('eventStatus', '', 'info', true);
    byId('eventDialog').showModal();
  });
  byId('eventForm').addEventListener('submit', submitEvent);
  byId('refreshList').addEventListener('click', loadContracts);
  byId('statusFilter').addEventListener('change', loadContracts);
  byId('riskFilter').addEventListener('change', loadContracts);
  byId('reanalyzeBtn').addEventListener('click', reanalyze);
  byId('qaSend').addEventListener('click', sendContractQuestion);
  byId('qaInput').addEventListener('input', autoGrowQaInput);
  byId('qaInput').addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendContractQuestion();
    }
  });
  byId('qaMessages').addEventListener('click', event => {
    const suggestion = event.target.closest('.qa-suggestion');
    if (suggestion) useQaSuggestion(suggestion);
  });
  byId('detailTabs').addEventListener('click', event => {
    const button = event.target.closest('[data-tab]');
    if (!button) return;
    document.querySelectorAll('#detailTabs button').forEach(item => item.classList.toggle('active', item === button));
    document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.toggle('active', panel.id === `tab-${button.dataset.tab}`));
  });
  byId('detailBanner').addEventListener('click', event => {
    const action = event.target.closest('[data-action]')?.dataset.action;
    if (action === 'reanalyze') reanalyze();
    if (action === 'reload-detail' && contractState.selectedId) selectContract(contractState.selectedId);
    if (action === 'open-documents') {
      const tab = document.querySelector('#detailTabs [data-tab="documents"]');
      if (tab) tab.click();
    }
  });
  byId('contractList').addEventListener('click', event => {
    if (event.target.closest('[data-action="reload-list"]')) loadContracts();
  });
  document.addEventListener('click', event => {
    const button = event.target.closest('[data-evidence]');
    if (!button) return;
    try {
      openEvidence(JSON.parse(decodeURIComponent(button.dataset.evidence)));
    } catch (error) {
      toast('原文依据读取失败，请重新打开。', 'error');
    }
  });
  byId('sourceBody').addEventListener('click', event => {
    if (event.target.closest('[data-copy-evidence]')) copyEvidence();
  });
  loadContracts();
});
