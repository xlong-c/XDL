'use strict';

/* XDL graph workflow editor (design preview).
 *
 * Nodes are created from the palette, ordered left to right, and wired with an
 * explicit multi-select of upstream nodes. Nothing here executes Python; the
 * editor only assembles a reviewable workflow document and hands it to the
 * shared design preview through xdl:workflow-queue.
 */
(() => {
  const root = document.querySelector('#page-workflow');
  if (!root) return;

  const $ = (selector, scope = root) => scope.querySelector(selector);
  const $$ = (selector, scope = root) => [...scope.querySelectorAll(selector)];
  const ESCAPES = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
  const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (char) => ESCAPES[char]);
  const icon = (name) => `<svg class="icon" aria-hidden="true"><use href="#i-${name}"/></svg>`;
  const uid = (type) => `${type}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 5)}`;
  const pad2 = (value) => String(value).padStart(2, '0');

  const STORAGE_KEY = 'xdl.graph-workflow.v3';
  const LEGACY_KEY = 'xdl.graph-workflow.v2';

  /* ---------- node type catalog ---------- */

  const TYPES = {
    dataset: {
      title: '数据集', short: 'DATASET', group: 'data', icon: 'data', color: '#127a6d',
      role: '输出图像 + 标签', hint: '定义训练样本来源. 真实训练时 target 决定读取哪个数据集, params 控制样本量与形状.',
      defaults: { name: 'train_dataset', target: 'torchvision.datasets:FakeData', params: { size: 512, image_size: '[3, 32, 32]', num_classes: 10 } },
    },
    transform: {
      title: '预处理', short: 'TRANSFORM', group: 'data', icon: 'settings', color: '#0f7f95',
      role: '张量归一化', hint: '把原始图像转成模型需要的张量. 多个变换可用 Compose 串联.',
      defaults: { name: 'train_transforms', target: 'torchvision.transforms:Compose', params: { transforms: 'ToTensor + Normalize' } },
    },
    dataloader: {
      title: '批次加载', short: 'DATALOADER', group: 'data', icon: 'layers', color: '#4a6fb5',
      role: '按批次产出', hint: '决定 batch size, 打乱与并行加载. 训练循环按这个粒度推进 step.',
      defaults: { name: 'train_loader', target: 'torch.utils.data:DataLoader', params: { batch_size: 64, num_workers: 2, shuffle: true } },
    },
    model: {
      title: '模型', short: 'MODEL', group: 'train', icon: 'chip', color: '#4a5fd0',
      role: '前向 logits', hint: 'registry 里的模型名会被解析成 XDL 注册的模型类, 输出类别数由 num_classes 决定.',
      defaults: { name: 'classifier', target: 'registry:vgg11_bn', params: { num_classes: 10, dropout: 0.2 } },
    },
    loss: {
      title: '损失函数', short: 'LOSS', group: 'train', icon: 'function', color: '#a8701f',
      role: '计算标量损失', hint: '接收模型输出与标签, 产出可反向传播的标量. 同类损失可以并行挂多个.',
      defaults: { name: 'classification_loss', target: 'torch.nn:CrossEntropyLoss', params: { label_smoothing: 0 } },
    },
    optimizer: {
      title: '优化器', short: 'OPTIMIZER', group: 'train', icon: 'bolt', color: '#7d57ab',
      role: '更新参数', hint: '在 CoreModel.training_step 里被显式调用, 学习率等超参写在这里.',
      defaults: { name: 'optimizer', target: 'torch.optim:AdamW', params: { lr: 0.001, weight_decay: 0.0005 } },
    },
    trainer: {
      title: '训练循环', short: 'TRAINER', group: 'run', icon: 'play', color: '#4d6a92',
      role: '驱动 epoch / step', hint: 'Trainer 负责生命周期, 回调与日志. device 与 precision 决定运行方式.',
      defaults: { name: 'trainer', target: 'xdl.trainer:Trainer', params: { max_epochs: 20, precision: '32', device: 'cuda' } },
    },
    checkpoint: {
      title: '检查点', short: 'CHECKPOINT', group: 'run', icon: 'save', color: '#5c6f85',
      role: '保存模型状态', hint: '通过回调在每个 epoch 边界保存 model + optimizer 状态, 用于断点续训.',
      defaults: { name: 'checkpoint', target: 'xdl.callbacks:ModelCheckpoint', params: { every_n_epochs: 1, save_last: true } },
    },
  };

  const GROUPS = [
    { id: 'data', label: 'DATA', title: '数据' },
    { id: 'train', label: 'TRAIN', title: '训练' },
    { id: 'run', label: 'RUN', title: '运行' },
  ];

  const TARGETS = {
    dataset: ['torchvision.datasets:FakeData', 'torchvision.datasets:CIFAR10', 'torchvision.datasets:CIFAR100', 'torchvision.datasets:ImageFolder'],
    transform: ['torchvision.transforms:Compose', 'torchvision.transforms:Resize', 'torchvision.transforms:Normalize'],
    dataloader: ['torch.utils.data:DataLoader'],
    model: ['registry:vgg11_bn', 'registry:vgg16_bn', 'registry:resnet18', 'registry:resnet34'],
    loss: ['torch.nn:CrossEntropyLoss', 'registry:FocalLoss', 'registry:HuberLoss'],
    optimizer: ['torch.optim:AdamW', 'torch.optim:Adam', 'torch.optim:SGD'],
    trainer: ['xdl.trainer:Trainer'],
    checkpoint: ['xdl.callbacks:ModelCheckpoint'],
  };

  const PRESETS = [
    { id: 'vgg', title: 'VGG-11 BN 分类', note: '合成图像 → 模型 → 损失 → 优化器 → 训练循环', types: ['dataset', 'dataloader', 'model', 'loss', 'optimizer', 'trainer'] },
    { id: 'resnet', title: 'ResNet-18 分类', note: '同一任务换模型, 用于横向对比实验', types: ['dataset', 'transform', 'dataloader', 'model', 'loss', 'optimizer', 'trainer'] },
    { id: 'smoke', title: '最小冒烟流', note: '快速验证训练入口与回调生命周期', types: ['dataset', 'model', 'loss', 'trainer'] },
  ];

  /* ---------- state ---------- */

  const state = { name: 'new-training-flow', nodes: [], selected: null, tab: 'config', menu: null, review: null };
  let saveTimer;

  function makeNode(type) {
    const base = TYPES[type].defaults;
    return { id: uid(type), type, name: base.name, target: base.target, params: { ...base.params }, inputs: [], code: '' };
  }

  function restore() {
    const read = (key, version) => {
      try {
        const saved = JSON.parse(localStorage.getItem(key) || 'null');
        return saved && saved.version === version && Array.isArray(saved.nodes) ? saved : null;
      } catch {
        return null;
      }
    };
    const saved = read(STORAGE_KEY, 3) || read(LEGACY_KEY, 2);
    if (!saved) return;
    state.name = typeof saved.name === 'string' && saved.name ? saved.name : state.name;
    state.nodes = saved.nodes
      .filter((entry) => entry && TYPES[entry.type])
      .map((entry) => ({
        id: String(entry.id || uid(entry.type)),
        type: entry.type,
        name: String(entry.name || TYPES[entry.type].defaults.name),
        target: String(entry.target || TYPES[entry.type].defaults.target),
        params: entry.params && typeof entry.params === 'object' ? { ...entry.params } : { ...TYPES[entry.type].defaults.params },
        inputs: Array.isArray(entry.inputs) ? entry.inputs.map(String) : [],
        code: typeof entry.code === 'string' ? entry.code : '',
      }));
    state.nodes.forEach((entry) => {
      entry.inputs = entry.inputs.filter((id) => state.nodes.some((other) => other.id === id && other !== entry));
    });
    state.selected = saved.selected && state.nodes.some((n) => n.id === saved.selected) ? saved.selected : state.nodes.at(-1)?.id ?? null;
    state.tab = saved.tab === 'code' ? 'code' : 'config';
  }

  const nodeOf = (id) => state.nodes.find((entry) => entry.id === id);
  const indexOf = (id) => state.nodes.findIndex((entry) => entry.id === id);
  const tell = (text) => document.dispatchEvent(new CustomEvent('xdl:workflow-message', { detail: text }));

  /* ---------- persistence ---------- */

  function setSaveState(text, tone) {
    const el = $('#wf-save-state');
    if (!el) return;
    el.className = `wf-save-state ${tone || ''}`.trim();
    el.innerHTML = `${icon(tone === 'is-error' ? 'info' : 'check')}${esc(text)}`;
  }

  function persist(immediate) {
    clearTimeout(saveTimer);
    if (!immediate) {
      setSaveState('未保存的改动', 'is-dirty');
      saveTimer = setTimeout(() => persist(true), 400);
      return;
    }
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ version: 3, name: state.name, nodes: state.nodes, selected: state.selected, tab: state.tab }));
      setSaveState('草稿已保存', 'is-saved');
    } catch {
      setSaveState('无法写入本地草稿', 'is-error');
    }
  }

  function download(name, content, type = 'application/json') {
    const url = URL.createObjectURL(new Blob([content], { type: `${type};charset=utf-8` }));
    const link = document.createElement('a');
    link.href = url;
    link.download = name;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  /* ---------- derived data ---------- */

  const primaryInput = (entry) => nodeOf(entry.inputs[0]);
  const upstreamLabel = (entry) => entry.inputs.map((id) => nodeOf(id)?.name || 'missing').join(' + ');

  function paramSummary(entry) {
    const pairs = Object.entries(entry.params).slice(0, 2).map(([key, value]) => `${key}=${value}`);
    return pairs.join(' · ') || '无参数';
  }

  function shapeOf(entry) {
    if (entry.type === 'dataset') return entry.params.image_size || '[3, 32, 32]';
    if (entry.type === 'dataloader') return `[${entry.params.batch_size ?? '?'}, 3, 32, 32]`;
    if (entry.type === 'model') return `[${TYPES.dataloader.defaults.params.batch_size}, ${entry.params.num_classes ?? 10}] logits`;
    if (entry.type === 'loss') return 'scalar';
    return '';
  }

  function diagnostic() {
    if (!state.nodes.length) return { tone: 'idle', text: '空画布' };
    const orphan = state.nodes.filter((entry) => indexOf(entry.id) > 0 && !entry.inputs.length);
    if (orphan.length) return { tone: 'warn', text: `${orphan.length} 个节点未连接` };
    return { tone: 'ok', text: '连接完整' };
  }

  function checks() {
    const rows = [];
    const types = new Set(state.nodes.map((entry) => entry.type));
    rows.push({ ok: state.nodes.length > 0, title: '训练图非空', detail: state.nodes.length ? `${state.nodes.length} 个节点按从左到右的顺序执行.` : '至少需要一个节点才能创建实验.' });
    const orphans = state.nodes.filter((entry, index) => index > 0 && !entry.inputs.length);
    rows.push({
      ok: orphans.length === 0,
      tone: orphans.length ? 'warn' : '',
      title: orphans.length ? `${orphans.length} 个节点未连接输入` : '节点连接完整',
      detail: orphans.length ? `${orphans.map((entry) => entry.name).join(', ')} 会以默认输入启动, 预览中按孤立节点处理.` : '每个节点都至少声明了一个上游输入.',
    });
    const missing = ['dataset', 'model', 'loss', 'trainer'].filter((type) => !types.has(type));
    rows.push({
      ok: missing.length === 0,
      tone: missing.length ? 'warn' : '',
      title: missing.length ? `缺少 ${missing.map((type) => TYPES[type].title).join(' / ')}` : '训练闭环组件齐全',
      detail: missing.length ? '预览允许继续, 但真实训练入口需要这些组件才能运行.' : '数据, 模型, 损失与训练循环都已就位.',
    });
    rows.push({ ok: false, tone: 'warn', title: '检查点未配置', detail: '当前工作流没有检查点节点, 无法从历史 step 恢复. 预览可以继续.', hidden: types.has('checkpoint') });
    const codeCount = state.nodes.filter((entry) => entry.code.trim()).length;
    rows.push({ ok: true, title: '局部代码不执行', detail: codeCount ? `${codeCount} 个节点带有 Python 草稿, 只会写入 JSON.` : '没有自定义 Python 草稿.', tone: codeCount ? 'warn' : '' });
    return rows.filter((row) => !row.hidden);
  }

  /* ---------- markup ---------- */

  function addbarMarkup() {
    const groups = GROUPS.map((group) => {
      const items = Object.entries(TYPES).filter(([, type]) => type.group === group.id);
      return items.map(([type, meta]) => `<button class="wf-add" data-add="${type}" style="--node-color:${meta.color}" title="添加${esc(meta.title)}节点">
        <span class="wf-add-icon">${icon(meta.icon)}</span>${esc(meta.title)}
      </button>`).join('');
    });
    return `<span class="wf-addbar-label">ADD</span>${groups[0]}<span class="wf-add-group"></span>${groups[1]}<span class="wf-add-group"></span>${groups[2]}
      <span class="wf-addbar-tail">
        <button class="btn small" data-action="save" title="保存草稿 (Ctrl + S)">${icon('save')}保存</button>
      </span>`;
  }

  function nodeMarkup(entry, index) {
    const type = TYPES[entry.type];
    const linked = entry.inputs.length > 0;
    const previous = index > 0 ? nodeOf(state.nodes[index - 1].id) : null;
    const edgeLabel = linked ? entry.inputs.map((id) => nodeOf(id) ? 'x' : '?').length > 1 ? 'merge' : 'input' : 'unlinked';
    const menuOpen = state.menu === entry.id;
    const facts = [];
    if (!linked) facts.push(`<div class="wf-node-fact is-unlinked"><span>IN</span><b>未连接上游</b></div>`);
    else facts.push(`<div class="wf-node-fact"><span>IN</span><b>${esc(upstreamLabel(entry))}</b></div>`);
    facts.push(`<div class="wf-node-fact"><span>OUT</span><b>${esc(shapeOf(entry) || type.role)}</b></div>`);
    const paramLine = Object.entries(entry.params).map(([key, value]) => `${key}=${value}`).join(' · ');
    if (paramLine) facts.push(`<div class="wf-node-fact"><span>P</span><b>${esc(paramLine)}</b></div>`);

    return `${index > 0 ? `<div class="wf-edge ${linked ? 'is-linked' : ''}"><span>${esc(edgeLabel)}</span></div>` : ''}
      <article class="wf-node ${state.selected === entry.id ? 'is-selected' : ''}" data-node="${esc(entry.id)}" style="--node-color:${type.color}">
        <span class="wf-port is-in" aria-hidden="true"></span>
        <span class="wf-port is-out" aria-hidden="true"></span>
        <div class="wf-node-top">
          <span class="wf-node-seq">${pad2(index + 1)}</span>
          <span class="wf-node-icon">${icon(type.icon)}</span>
          <span class="wf-node-kind">${esc(type.short)}</span>
          <button class="wf-node-menu-btn" data-menu="${esc(entry.id)}" aria-label="节点操作" aria-expanded="${menuOpen}">${icon('settings')}</button>
        </div>
        <button class="wf-node-main" data-select="${esc(entry.id)}" aria-pressed="${state.selected === entry.id}">
          <span class="wf-node-name">${esc(entry.name)}</span>
          <span class="wf-node-target">${esc(entry.target)}</span>
        </button>
        <div class="wf-node-facts">${facts.join('')}</div>
        <div class="wf-node-foot">
          <span class="wf-tag">${esc(primaryInput(entry) ? 'WIRED' : 'OPEN')}</span>
          <span class="wf-tag is-accent">${esc(type.title)}</span>
        </div>
        ${menuOpen ? `<div class="wf-node-menu" role="menu">
          <button data-move="left" data-id="${esc(entry.id)}" ${index === 0 ? 'disabled' : ''} role="menuitem">${icon('back')}向左移动</button>
          <button data-move="right" data-id="${esc(entry.id)}" ${index === state.nodes.length - 1 ? 'disabled' : ''} role="menuitem">${icon('arrow')}向右移动</button>
          <button data-duplicate="${esc(entry.id)}" role="menuitem">${icon('copy')}复制节点</button>
          <hr>
          <button data-delete="${esc(entry.id)}" class="is-danger" role="menuitem">${icon('close')}删除节点</button>
        </div>` : ''}
      </article>`;
  }

  function canvasMarkup() {
    if (!state.nodes.length) {
      return `<div class="wf-empty">
        <div class="wf-empty-art">${icon('flow')}</div>
        <h2>从空白训练图开始</h2>
        <p>点击上方任意节点加入画布, 选中后再声明上游输入. 顺序即执行顺序, 不需要拖拽连线.</p>
        <div class="wf-presets">${PRESETS.map((preset) => `<button class="wf-preset" data-preset="${preset.id}">
          <span class="wf-preset-head">${icon('bolt')}<strong>${esc(preset.title)}</strong></span>
          <small>${esc(preset.note)}</small>
          <span class="wf-preset-flow">${preset.types.map((type, index) => `${index ? '<u></u>' : ''}<i style="--c:${TYPES[type].color}"></i>`).join('')}</span>
        </button>`).join('')}</div>
      </div>`;
    }
    return `<div class="wf-graph">${state.nodes.map((entry, index) => nodeMarkup(entry, index)).join('')}</div>`;
  }

  function inspectorMarkup() {
    const entry = state.selected ? nodeOf(state.selected) : null;
    if (!entry) {
      return `<div class="wf-inspector-empty">
        <div class="wf-empty-art">${icon('mouse')}</div>
        <h2>${state.nodes.length ? '选择一个节点' : '还没有可编辑的节点'}</h2>
        <p>${state.nodes.length ? '点击画布中的卡片, 在这里编辑名称, target, 参数与局部 Python.' : '添加节点后, 这里会显示它的完整配置.'}</p>
        <div class="wf-kbd-list">
          <div class="wf-kbd-row">添加节点<b>点击左侧</b></div>
          <div class="wf-kbd-row">选择节点<b>Enter</b></div>
          <div class="wf-kbd-row">保存草稿<b>Ctrl S</b></div>
          <div class="wf-kbd-row">取消选中<b>Esc</b></div>
        </div>
      </div>`;
    }

    const type = TYPES[entry.type];
    const tab = state.tab;
    const available = state.nodes.filter((other) => other.id !== entry.id && indexOf(other.id) < indexOf(entry.id));
    const downstream = state.nodes.filter((other) => other.inputs.includes(entry.id));
    const params = Object.entries(entry.params)
      .map(([key, value]) => `<label class="wf-param"><span title="${esc(key)}">${esc(key)}</span><input data-param="${esc(key)}" value="${esc(value)}" aria-label="${esc(key)}"></label>`)
      .join('');
    const inputs = available.length
      ? available.map((other) => `<label class="wf-input-row"><input type="checkbox" data-input="${esc(other.id)}" ${entry.inputs.includes(other.id) ? 'checked' : ''}><span class="wf-input-name">${esc(other.name)}</span><span class="wf-input-type">${esc(TYPES[other.type].short)}</span></label>`).join('')
      : `<div class="wf-empty-note">没有可用的上游节点. 先添加其他节点, 并让它排在当前节点左侧.</div>`;

    const body = tab === 'code'
      ? `<div class="wf-field">
          <div class="wf-label">节点 Python<small>${entry.code.trim() ? '已自定义' : '模板'}</small></div>
          <textarea data-node-code spellcheck="false" aria-label="节点 Python 代码">${esc(codeOf(entry))}</textarea>
          <small>代码随节点写入 JSON, 设计稿不会执行 Python.</small>
        </div>`
      : `<div class="wf-field">
          <label for="wf-node-name">节点名称</label>
          <input id="wf-node-name" type="text" data-node-name value="${esc(entry.name)}">
        </div>
        <div class="wf-field">
          <label for="wf-node-target">组件 target</label>
          <select id="wf-node-target" data-node-target>${(TARGETS[entry.type] || [entry.target]).map((option) => `<option ${option === entry.target ? 'selected' : ''}>${esc(option)}</option>`).join('')}</select>
        </div>
        <div class="wf-field">
          <div class="wf-label">输入连接<small>${entry.inputs.length} 个上游</small></div>
          <div class="wf-input-list">${inputs}</div>
        </div>
        <div class="wf-field">
          <div class="wf-label">组件参数<small>target + params</small></div>
          <div class="wf-params">${params || `<div class="wf-empty-note">该节点暂无参数.</div>`}</div>
        </div>
        <div class="wf-field">
          <div class="wf-label">下游节点<small>${downstream.length} 个</small></div>
          <div class="wf-input-list">${downstream.length
            ? downstream.map((other) => `<div class="wf-input-row is-down"><span class="wf-input-name">${esc(other.name)}</span><span class="wf-input-type">${esc(TYPES[other.type].short)}</span></div>`).join('')
            : `<div class="wf-empty-note">没有节点使用它的输出.</div>`}</div>
        </div>
`;

    return `<div class="wf-insp-head" style="--node-color:${type.color}">
        <div class="wf-insp-title"><span class="wf-node-kind">${esc(type.short)}</span><span class="wf-insp-id">${esc(entry.id)}</span></div>
        <div class="wf-insp-name">${esc(entry.name)}</div>
        <div class="wf-insp-meta"><span>${esc(type.role)}</span><i></i><span>${esc(shapeOf(entry) || '—')}</span><i></i><span>${entry.inputs.length} 上游</span></div>
      </div>
      <div class="wf-tabs" role="tablist" aria-label="节点配置分区">
        <button class="wf-tab" role="tab" data-wf-tab="config" aria-selected="${tab === 'config'}">配置</button>
        <button class="wf-tab" role="tab" data-wf-tab="code" aria-selected="${tab === 'code'}">局部 Python<span class="wf-tab-count">${entry.code.trim() ? '1' : '0'}</span></button>
      </div>
      <div class="wf-insp-body">${body}</div>
      <div class="wf-insp-foot">
        <button class="btn btn-danger" data-delete="${esc(entry.id)}">${icon('close')}删除节点</button>
        <button class="btn primary" data-action="apply" title="应用修改 (Ctrl + Enter)">应用修改</button>
      </div>`;
  }

  function codeOf(entry) {
    if (entry.code.trim()) return entry.code;
    return `# ${entry.name}\n\nSPEC = {\n    "target": ${JSON.stringify(entry.target)},\n    "params": ${JSON.stringify(entry.params, null, 4)}\n}\n\n# 通过 XDL config builder 接到训练入口.\n# 设计稿只保存代码, 不执行 Python.\n`;
  }

  function render() {
    if (state.review) state.review.close();
    const diag = diagnostic();
    const linked = state.nodes.filter((entry) => entry.inputs.length).length;
    root.innerHTML = `
      <header class="wf-head">
        <h1>训练流</h1>
        <span class="wf-head-sep"></span>
        <input class="wf-name" id="wf-name" value="${esc(state.name)}" aria-label="工作流名称" spellcheck="false" autocomplete="off">
        <span class="wf-stats"><b>${state.nodes.length}</b> 节点 <i></i> <b>${linked}</b> 连接</span>
        <span class="wf-flag ${diag.tone === 'ok' ? 'is-ok' : diag.tone === 'warn' ? 'is-warn' : 'is-idle'}"><i class="dot"></i>${esc(diag.text)}</span>
        <span id="wf-save-state" class="wf-save-state">${icon('check')}本机草稿</span>
        <div class="wf-head-actions">
          <button class="btn wf-quiet-btn" data-action="clear" ${state.nodes.length ? '' : 'disabled'}>清空</button>
          <button class="btn" data-action="export" ${state.nodes.length ? '' : 'disabled'}>${icon('download')}导出</button>
          <button class="btn primary" data-action="run" ${state.nodes.length ? '' : 'disabled'}>${icon('check')}检查并创建</button>
        </div>
      </header>

      <nav class="wf-addbar" aria-label="添加节点">${addbarMarkup()}</nav>

      <div class="wf-body">
        <main class="wf-canvas" aria-label="训练图画布">
          <div class="wf-scroll">${canvasMarkup()}</div>
        </main>
        <aside class="wf-inspector" aria-label="节点详情">${inspectorMarkup()}</aside>
      </div>`;

    if (state.selected) {
      const card = $(`[data-node="${state.selected}"]`);
      if (card && card.parentElement) card.parentElement.scrollLeft = Math.max(0, card.offsetLeft - card.parentElement.clientWidth / 2 + card.clientWidth / 2);
    }
    syncScrollHint();
  }

  /* fade on the canvas right edge only while more nodes are off-screen */
  function syncScrollHint() {
    const canvas = $('.wf-canvas');
    const scroller = $('.wf-scroll');
    if (!canvas || !scroller) return;
    const more = scroller.scrollWidth - scroller.clientWidth - scroller.scrollLeft > 8;
    canvas.classList.toggle('is-scrollable', more);
  }

  function closeMenu() {
    if (!state.menu) return;
    state.menu = null;
    render();
  }

  /* ---------- actions ---------- */

  function addNode(type, { silent } = {}) {
    if (!TYPES[type]) return;
    const entry = makeNode(type);
    const previous = state.nodes.at(-1);
    if (previous) entry.inputs = [previous.id];
    state.nodes.push(entry);
    state.selected = entry.id;
    state.tab = 'config';
    persist(true);
    render();
    if (!silent) tell(`已添加${TYPES[type].title}节点`);
    $('#wf-save-state')?.scrollIntoView({ block: 'nearest' });
  }

  function applyPreset(id) {
    const preset = PRESETS.find((item) => item.id === id);
    if (!preset) return;
    const built = [];
    preset.types.forEach((type) => {
      const entry = makeNode(type);
      const previous = built.at(-1);
      entry.inputs = previous ? [previous.id] : [];
      built.push(entry);
    });
    state.nodes = built;
    state.selected = state.nodes.at(-1)?.id ?? null;
    state.tab = 'config';
    persist(true);
    render();
    tell(`已载入「${preset.title}」预设, 参数仍可逐个调整`);
  }

  function removeNode(id) {
    state.nodes = state.nodes.filter((entry) => entry.id !== id);
    state.nodes.forEach((entry) => {
      entry.inputs = entry.inputs.filter((input) => input !== id);
    });
    if (state.selected === id) state.selected = state.nodes.at(-1)?.id ?? null;
    state.menu = null;
    persist(true);
    render();
    tell('节点已删除');
  }

  function duplicateNode(id) {
    const original = nodeOf(id);
    if (!original) return;
    const copy = { ...original, id: uid(original.type), name: `${original.name}_copy`, params: { ...original.params }, inputs: [...original.inputs] };
    state.nodes.splice(indexOf(id) + 1, 0, copy);
    state.selected = copy.id;
    state.menu = null;
    persist(true);
    render();
    tell(`已复制为 ${copy.name}`);
  }

  function moveNode(id, direction) {
    const from = indexOf(id);
    const to = direction === 'left' ? from - 1 : from + 1;
    if (from < 0 || to < 0 || to >= state.nodes.length) return;
    const [entry] = state.nodes.splice(from, 1);
    state.nodes.splice(to, 0, entry);
    state.menu = null;
    persist(true);
    render();
  }

  function applyInspector() {
    const entry = state.selected ? nodeOf(state.selected) : null;
    if (!entry) return;
    const name = $('[data-node-name]');
    const target = $('[data-node-target]');
    if (name) entry.name = name.value.trim() || TYPES[entry.type].title;
    if (target) entry.target = target.value;
    entry.inputs = $$('[data-input]:checked').map((input) => input.dataset.input);
    const next = {};
    $$('[data-param]').forEach((input) => {
      const raw = input.value.trim();
      next[input.dataset.param] = raw === 'true' ? true : raw === 'false' ? false : raw !== '' && !Number.isNaN(Number(raw)) ? Number(raw) : raw;
    });
    entry.params = next;
    const code = $('[data-node-code]');
    if (code) entry.code = code.value;
    persist(true);
    render();
    tell(`${entry.name} 已应用修改`);
  }

  function exportBundle(kind) {
    if (!state.nodes.length) {
      tell('请先添加至少一个节点');
      return;
    }
    persist(true);
    const yaml = toYaml();
    if (kind === 'yaml') {
      download(`${slug(state.name)}.yaml`, yaml, 'text/yaml');
      tell('已导出工作流 YAML');
      return;
    }
    download(`${slug(state.name)}-workflow.json`, JSON.stringify({ version: 3, preview_only: true, name: state.name, nodes: state.nodes, yaml }, null, 2));
    tell('已导出节点, 连接和局部代码');
  }

  function slug(value) {
    return String(value || 'workflow').trim().replace(/[^\w.-]+/g, '-').replace(/^-+|-+$/g, '') || 'workflow';
  }

  function toYaml() {
    const lines = ['# XDL graph workflow v3', '# 用户创建的节点与显式多输入连接.', 'runtime:', `  experiment_name: ${JSON.stringify(state.name)}`, 'nodes:'];
    state.nodes.forEach((entry) => {
      lines.push(`  - id: ${entry.id}`, `    type: ${entry.type}`, `    target: ${JSON.stringify(entry.target)}`, `    name: ${JSON.stringify(entry.name)}`, `    inputs: [${entry.inputs.map((input) => JSON.stringify(input)).join(', ')}]`, '    params:');
      Object.entries(entry.params).forEach(([key, value]) => {
        lines.push(`      ${key}: ${typeof value === 'string' ? JSON.stringify(value) : value}`);
      });
    });
    return lines.join('\n');
  }

  function workflowValues() {
    const pick = (type, key) => state.nodes.find((entry) => entry.type === type)?.params[key];
    const datasetTarget = state.nodes.find((entry) => entry.type === 'dataset')?.target || '';
    const modelTarget = state.nodes.find((entry) => entry.type === 'model')?.target || '';
    return {
      name: state.name,
      dataset: datasetTarget.split(':').pop(),
      samples: Number(pick('dataset', 'size')) || 512,
      model: modelTarget.replace('registry:', ''),
      batch: Number(pick('dataloader', 'batch_size')) || 64,
      lr: Number(pick('optimizer', 'lr')) || 0.001,
      epochs: Number(pick('trainer', 'max_epochs')) || 20,
      precision: String(pick('trainer', 'precision') ?? '32'),
      device: String(pick('trainer', 'device') ?? 'cuda'),
      optimizer: (state.nodes.find((entry) => entry.type === 'optimizer')?.target || 'torch.optim:AdamW').replace('torch.optim:', ''),
      loss: (state.nodes.find((entry) => entry.type === 'loss')?.target || 'torch.nn:CrossEntropyLoss').split(':').pop(),
      saveEvery: pick('checkpoint', 'every_n_epochs') ? `每 ${pick('checkpoint', 'every_n_epochs')} 个 epoch` : '每个 epoch',
      classes: Number(pick('dataset', 'num_classes')) || 10,
    };
  }

  function openReview() {
    if (!state.nodes.length) {
      tell('请先添加至少一个节点');
      return;
    }
    if (state.review) return;
    const rows = checks();
    const values = workflowValues();
    const blockers = rows.filter((row) => row.tone === 'fail');
    const mask = document.createElement('div');
    mask.className = 'wf-modal-mask';
    mask.innerHTML = `<div class="wf-modal" role="dialog" aria-modal="true" aria-labelledby="wf-review-title">
      <div class="wf-modal-head">
        <small>PRE-FLIGHT CHECK</small>
        <h2 id="wf-review-title">创建前检查</h2>
        <p>确认节点与参数后再加入队列. 设计预览只生成模拟实验记录, 不会启动训练进程.</p>
      </div>
      <div class="wf-modal-body">
        <div class="wf-summary">
          <div><small>工作流</small><strong>${esc(state.name)}</strong></div>
          <div><small>节点 / 连接</small><strong>${state.nodes.length} / ${state.nodes.filter((entry) => entry.inputs.length).length}</strong></div>
          <div><small>模型</small><strong>${esc(values.model)}</strong></div>
          <div><small>批次 / Epochs</small><strong>${esc(values.batch)} / ${esc(values.epochs)}</strong></div>
        </div>
        <div class="wf-check">${rows.map((row) => `<div class="wf-check-row ${row.tone ? `is-${row.tone}` : ''}">
          <span class="wf-check-mark">${icon(row.tone === 'fail' ? 'close' : row.tone === 'warn' ? 'info' : 'check')}</span>
          <span><b>${esc(row.title)}</b><br>${esc(row.detail)}</span>
        </div>`).join('')}</div>
      </div>
      <div class="wf-modal-foot">
        <button class="btn" data-action="close-review">继续编辑</button>
        <button class="btn primary" data-action="queue" ${blockers.length ? 'disabled' : ''}>${icon('play')}加入模拟队列</button>
      </div>
    </div>`;

    const close = () => {
      mask.remove();
      state.review = null;
      document.removeEventListener('keydown', onKey);
    };
    const onKey = (event) => {
      if (event.key === 'Escape') {
        event.stopPropagation();
        close();
      }
    };
    state.review = { close };
    mask.addEventListener('click', (event) => {
      if (event.target === mask) close();
    });
    mask.querySelector('[data-action="close-review"]').addEventListener('click', close);
    mask.querySelector('[data-action="queue"]').addEventListener('click', () => {
      close();
      queueExperiment(values);
    });
    root.append(mask);
    mask.querySelector('[data-action="close-review"]').focus();
  }

  function queueExperiment(values) {
    if (!values.dataset || !values.model) {
      tell('数据集与模型节点是创建实验的必要条件');
      return;
    }
    document.dispatchEvent(new CustomEvent('xdl:workflow-queue', { detail: { version: 3, values, nodes: state.nodes.map((entry) => ({ ...entry, code_applied: Boolean(entry.code.trim()), code_dirty: false })), yaml: toYaml(), code_execution: 'disabled' } }));
  }

  /* ---------- events ---------- */

  root.addEventListener('scroll', syncScrollHint, true);
  window.addEventListener('resize', syncScrollHint);

  root.addEventListener('input', (event) => {
    const target = event.target;
    if (target.id === 'wf-name') {
      state.name = target.value;
      persist();
    }
  });

  root.addEventListener('change', (event) => {
    if (event.target.matches('[data-input]')) applyInspector();
  });

  root.addEventListener('click', (event) => {
    const button = event.target.closest('button');
    if (!button) {
      if (state.menu) closeMenu();
      return;
    }
    const dataset = button.dataset;

    if (dataset.add) return addNode(dataset.add);
    if (dataset.preset) return applyPreset(dataset.preset);
    if (dataset.select) {
      if (state.menu) {
        state.menu = null;
      }
      state.selected = dataset.select;
      state.tab = 'config';
      persist(true);
      return render();
    }
    if (dataset.menu) {
      state.menu = state.menu === dataset.menu ? null : dataset.menu;
      state.selected = dataset.menu;
      return render();
    }
    if (dataset.delete) return removeNode(dataset.delete);
    if (dataset.duplicate) return duplicateNode(dataset.duplicate);
    if (dataset.move) return moveNode(dataset.id, dataset.move);
    if (dataset.wfTab) {
      state.tab = dataset.wfTab;
      return render();
    }

    switch (dataset.action) {
      case 'apply':
        return applyInspector();
      case 'save':
        persist(true);
        return tell('工作流草稿已保存');
      case 'clear':
        if (!state.nodes.length) return;
        state.nodes = [];
        state.selected = null;
        state.menu = null;
        persist(true);
        render();
        return tell('画布已清空');
      case 'export':
        return exportBundle('json');
      case 'run':
        return openReview();
      case 'help':
        return tell('点击左侧节点添加, 点击卡片选中, 在右侧勾选上游输入即可连线');
      default:
        return undefined;
    }
  });

  document.addEventListener('keydown', (event) => {
    if (!document.body.classList.contains('workflow-mode')) return;
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(event.target.tagName || '');
    if (event.key === 'Escape') {
      if (state.review) return state.review.close();
      if (state.menu) return closeMenu();
      if (typing) return event.target.blur();
      if (state.selected) {
        state.selected = null;
        render();
      }
      return;
    }
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 's') {
      event.preventDefault();
      persist(true);
      tell('工作流草稿已保存');
      return;
    }
    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
      event.preventDefault();
      applyInspector();
    }
  });

  window.addEventListener('beforeunload', () => persist(true));

  /* ---------- boot ---------- */

  restore();
  render();
})();
