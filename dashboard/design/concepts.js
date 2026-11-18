'use strict';

(() => {
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const esc = (value) => String(value).replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character]);
  const app = $('#app');
  const toast = $('#toast');
  let toastTimer;

  const STATUS = { running: '训练中', paused: '已暂停', queued: '排队中', completed: '已完成', failed: '失败', stopped: '已停止' };
  const TONE = { running: 'running', paused: 'paused', queued: 'queued', completed: 'done', failed: 'failed', stopped: 'stopped' };

  const runs = [
    { id: 'r1', name: 'vgg11-cifar10-baseline', template: 'VGG-11 BN', status: 'running', step: 98, total: 156, epoch: 12, epochs: 20, loss: 2.307, val: 2.321, acc: 10.2, gpu: 'GPU 0', lr: '1e-3', batch: 64, launcher: 'single', started: '10:42:07', elapsed: '18m 24s', eta: '约 11 分钟', checkpoint: 'epoch_12.ckpt' },
    { id: 'r2', name: 'resnet18-lr-sweep', template: 'ResNet-18', status: 'queued', step: 0, total: 156, epoch: 0, epochs: 20, loss: null, val: null, acc: null, gpu: 'GPU 1', lr: '3e-4', batch: 64, launcher: 'torchrun', started: null, elapsed: '-', eta: '等待 GPU 1', checkpoint: null },
    { id: 'r3', name: 'mlp-smoke-test', template: 'MLP', status: 'completed', step: 64, total: 64, epoch: 4, epochs: 4, loss: 2.299, val: 2.314, acc: 10.4, gpu: 'CPU', lr: '1e-2', batch: 32, launcher: 'single', started: '09:58:11', elapsed: '2m 18s', eta: '已完成', checkpoint: 'final.ckpt' },
    { id: 'r4', name: 'vgg11-lr-3e-3', template: 'VGG-11 BN', status: 'paused', step: 48, total: 156, epoch: 6, epochs: 20, loss: 2.329, val: 2.338, acc: 9.8, gpu: 'GPU 0', lr: '3e-3', batch: 64, launcher: 'accelerate', started: '10:21:40', elapsed: '9m 02s', eta: '已暂停', checkpoint: 'epoch_6.ckpt' },
    { id: 'r5', name: 'cifar10-download-check', template: 'VGG-11 BN', status: 'failed', step: 0, total: 156, epoch: 0, epochs: 20, loss: null, val: null, acc: null, gpu: 'CPU', lr: '1e-3', batch: 64, launcher: 'single', started: '09:40:02', elapsed: '12s', eta: '-', checkpoint: null, error: '数据归档校验失败, 训练尚未开始. 可改用 FakeData 模板验证流程.' },
  ];

  const state = {
    concept: 'a',
    selected: 'r1',
    metric: 'loss',
    logs: [],
    bStatus: 'all',
    bQuery: '',
    bSort: { key: 'step', dir: 'desc' },
    bPicked: new Set(['r1', 'r4', 'r3']),
    cNode: 1,
    cPattern: 'full',
    cConfig: { epochs: 20, batch: 64, lr: 0.001, gpus: 1, launcher: 'single', val: true, amp: true, checkpoint: true },
    cTab: 'params',
    cRunState: 'draft',
    cShowPreflight: false,
    cActivePanel: 'library',
    cNodes: [],
    cSelected: null,
    cNodeId: 0,
    cNodeDrafts: {
      dataset: { source: 'FakeData', samples: 512 },
      preprocess: { normalize: true, augment: false },
      model: { name: 'VGG-11 BN' },
      loss: { name: 'CrossEntropy' },
      optimizer: { name: 'SGD', lr: '0.001' },
      validation: { enabled: true, samples: 128 },
      checkpoint: { policy: '每个 epoch' },
    },
    logFilter: 'all',
    bDevice: 'all',
  };

  const CONCEPTS = {
    a: { eyebrow: 'DIRECTION 01 / FOCUS', title: '让正在进行的训练, 成为主角.', description: '单实验优先. 曲线占据视觉中心, 控制就近放置, 配置与日志按需展开. 适合日常训练与快速排错.', note: '建议优先落地 · 复用现有 API 的成本较低' },
    b: { eyebrow: 'DIRECTION 02 / COMPARE', title: '把多次实验, 放在同一张桌面上.', description: '多实验优先. 左侧按状态筛选, 中部叠加对比曲线, 底部是可排序的结果表. 适合调参与结果复盘.', note: '适合实验密集阶段 · 需要补一个批量指标接口' },
    c: { eyebrow: 'DIRECTION 03 / PIPELINE', title: '从空白画布, 搭建一条训练流水线.', description: '节点优先. 从节点库添加组件, 自由拖动排布, 在检查器里声明输入关系, 让 preflight 随着画布逐步变绿.', note: '已选方向 · 适合作为工作流编辑器主界面' },
  };

  /* ---------- shared helpers ---------- */

  function seedOf(text) {
    let hash = 2166136261;
    for (let index = 0; index < text.length; index += 1) {
      hash ^= text.charCodeAt(index);
      hash = Math.imul(hash, 16777619);
    }
    return hash >>> 0;
  }

  function rng(seed) {
    let value = seed;
    return () => {
      value = (value + 0x6d2b79f5) >>> 0;
      let t = value;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function curve(run, key = 'loss', points = 48) {
    const random = rng(seedOf(run.id + key));
    const isLoss = key !== 'accuracy';
    const start = isLoss ? 2.34 : 9.4;
    const end = isLoss ? (run.status === 'failed' ? 2.34 : 1.12) : (run.status === 'completed' ? 38 : 26);
    const data = [];
    let wobble = 0;
    for (let index = 0; index < points; index += 1) {
      const ratio = index / (points - 1);
      const drift = start + (end - start) * Math.pow(ratio, isLoss ? 0.72 : 0.6);
      wobble = wobble * 0.62 + (random() - 0.5) * (isLoss ? 0.05 : 0.9);
      data.push({ x: Math.round(ratio * run.total), y: Number((drift + wobble).toFixed(isLoss ? 3 : 2)) });
    }
    return data;
  }

  // One shared domain for every series in a chart: independent per-line scaling made
  // falling curves look like rising ones, so min/max are computed across all series.
  function domainOf(seriesList) {
    const xs = seriesList.flatMap((item) => item.points.map((point) => point.x));
    const ys = seriesList.flatMap((item) => item.points.map((point) => point.y));
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const padY = (maxY - minY) * 0.12 || (maxY === 0 ? 1 : Math.abs(maxY) * 0.1);
    return { minX: Math.min(...xs), maxX: Math.max(...xs), lo: minY - padY, hi: maxY + padY };
  }

  function linePath(points, box, domain) {
    const scaleX = (value) => box.x + ((value - domain.minX) / (domain.maxX - domain.minX || 1)) * box.w;
    const scaleY = (value) => box.y + box.h - ((value - domain.lo) / (domain.hi - domain.lo || 1)) * box.h;
    const d = points.map((point, index) => `${index ? 'L' : 'M'}${scaleX(point.x).toFixed(1)},${scaleY(point.y).toFixed(1)}`).join(' ');
    return { d, scaleX, scaleY };
  }

  function gridMarks(box, domain, rows = 4, digits = 1) {
    return Array.from({ length: rows + 1 }, (_, index) => {
      const y = box.y + (box.h / rows) * index;
      const value = domain.hi - ((domain.hi - domain.lo) / rows) * index;
      return `<line class="grid" x1="${box.x}" y1="${y}" x2="${box.x + box.w}" y2="${y}"/><text class="axis" x="${box.x - 8}" y="${y + 3.5}" text-anchor="end">${value.toFixed(digits)}</text>`;
    }).join('');
  }

  function showToast(message) {
    window.clearTimeout(toastTimer);
    toast.textContent = message;
    toast.hidden = false;
    toastTimer = window.setTimeout(() => { toast.hidden = true; }, 3200);
  }

  function openDialog(title, body) {
    $('#dialogTitle').textContent = title;
    $('#dialogBody').innerHTML = body;
    $('#detailDialog').showModal();
  }

  function addLog(text, level = 'INFO') {
    const now = new Date().toLocaleTimeString('zh-CN', { hour12: false });
    state.logs.unshift({ at: now, level, text });
  }

  function tagOf(run) {
    return `<span class="tag ${TONE[run.status]}"><i class="dot"></i>${STATUS[run.status]}</span>`;
  }

  function progressOf(run) {
    if (!run.total) return 0;
    return Math.min(100, Math.round((run.step / run.total) * 100));
  }

  /* ---------- A / 专注训练 ---------- */

  function renderA() {
    const run = runs.find((item) => item.id === state.selected) || runs[0];
    const rail = runs.map((item) => `<button class="a-run" data-pick="${item.id}" aria-current="${item.id === run.id}">
      <div class="a-run-top"><strong>${esc(item.name)}</strong>${tagOf(item)}</div>
      <small>${esc(item.template)} · ${item.step}/${item.total} steps</small>
      <div class="progress" style="color:${item.status === 'failed' ? '#a4363f' : item.status === 'paused' ? '#c08b37' : '#3a5a8c'}"><span style="width:${progressOf(item)}%"></span></div>
    </button>`).join('');

    const metricLabel = state.metric === 'loss' ? '验证损失' : '验证准确率';
    const value = state.metric === 'loss' ? (run.val ?? 0).toFixed(3) : `${(run.acc ?? 0).toFixed(1)}%`;
    const allLogs = state.logs.length ? state.logs : baseLogs(run);
    const logs = state.logFilter === 'warn' ? allLogs.filter((line) => line.level === 'ERROR' || line.level === 'WARN') : allLogs;

    app.innerHTML = `<div class="app-a shell">
      <aside class="rail">
        <div class="brand"><i>X</i><div><strong>XDL</strong><small>training observatory</small></div></div>
        <div class="a-section-label"><span>运行队列</span><span>${String(runs.length).padStart(2, '0')}</span></div>
        <div class="a-run-list">${rail}</div>
        <button class="button primary" data-new-run>+ 新建实验</button>
        <div class="a-rail-foot"><b><i class="dot"></i>本地 / loopback</b>设计预览 · 数据为模拟值</div>
      </aside>
      <main>
        <div class="a-top">
          <div>
            <div class="eyebrow">${esc(run.template)} · ${esc(run.id)}</div>
            <h2>${esc(run.name)}</h2>
            <div class="a-meta">${tagOf(run)}<i></i><span>${esc(run.gpu)}</span><i></i><span>epoch ${run.epoch} / ${run.epochs}</span><i></i><span class="mono">step ${run.step} / ${run.total}</span></div>
          </div>
          <div class="a-controls">${controlsFor(run)}</div>
        </div>

        <figure class="a-figure">
          <div class="a-figure-head">
            <div><h3>训练曲线</h3><p>纵轴 ${state.metric === 'loss' ? 'loss' : 'accuracy (%)'} · 横轴 global step</p></div>
            <div class="a-seg" role="group" aria-label="切换指标"><button data-metric="loss" aria-pressed="${state.metric === 'loss'}">Loss</button><button data-metric="accuracy" aria-pressed="${state.metric === 'accuracy'}">Accuracy</button></div>
          </div>
          <div class="a-chart" id="aChart">
            ${chartSvg(run)}
            <div class="a-tooltip" id="aTip" hidden></div>
          </div>
          <div class="a-foot"><span><i class="dot"></i>曲线为模拟采样, 每 2 秒聚合一次</span><span>${metricLabel} <b class="mono">${value}</b></span></div>
        </figure>

        <div class="a-stats">
          ${statCell('训练损失', run.loss === null ? '-' : run.loss.toFixed(3), 'train/loss')}
          ${statCell('验证损失', run.val === null ? '-' : run.val.toFixed(3), 'val/loss')}
          ${statCell('验证准确率', run.acc === null ? '-' : `${run.acc.toFixed(1)}%`, 'val/accuracy')}
          ${statCell('已运行', run.elapsed, run.eta)}
        </div>

        <div class="a-lower">
          <section class="card a-log">
            <div class="a-log-head"><h3>运行日志</h3><div style="display:flex;gap:6px"><button class="button ghost" data-log-filter aria-pressed="${state.logFilter === 'warn'}">${state.logFilter === 'warn' ? '显示全部' : '只看异常'}</button><button class="button ghost" data-log-clear>清屏</button></div></div>
            <div class="a-log-body">${logs.length ? logs.map((line) => `<div class="a-log-line ${line.level === 'ERROR' || line.level === 'WARN' ? 'bad' : ''}"><time>${line.at}</time><b>${line.level}</b><span>${esc(line.text)}</span></div>`).join('') : '<div class="a-log-line"><span style="grid-column:1/-1;color:#a39b8d">没有匹配的日志</span></div>'}</div>
          </section>
          <section class="card a-aside">
            <div class="a-aside-head"><h3>实验记录</h3><button class="button ghost" data-open-config>查看配置</button></div>
            <div class="a-facts">
              <div class="a-fact"><span>启动方式</span><b>${esc(run.launcher)}</b></div>
              <div class="a-fact"><span>Batch size</span><b>${run.batch}</b></div>
              <div class="a-fact"><span>学习率</span><b>${esc(run.lr)}</b></div>
              <div class="a-fact"><span>最近检查点</span><b>${esc(run.checkpoint ?? '尚未保存')}</b></div>
            </div>
            <div class="a-next"><span>→</span><span>${nextHint(run)}</span></div>
          </section>
        </div>
      </main>
    </div>`;

    bindA(run);
  }

  function statCell(label, value, note) {
    return `<div class="a-stat"><span>${label}</span><strong>${value}</strong><small>${esc(note)}</small></div>`;
  }

  function controlsFor(run) {
    if (run.status === 'running') {
      return `<button class="button" data-action="pause">暂停</button><button class="button" data-action="checkpoint">保存检查点</button><button class="button danger" data-action="stop">停止</button>`;
    }
    if (run.status === 'paused') {
      return `<button class="button primary" data-action="resume">继续训练</button><button class="button danger" data-action="stop">停止</button>`;
    }
    if (run.status === 'queued') {
      return `<button class="button" data-action="cancel">取消排队</button>`;
    }
    return `<button class="button primary" data-action="restart">从检查点新建</button>`;
  }

  function nextHint(run) {
    if (run.status === 'running') return '指标持续写入 SQLite. 暂停会在 batch 边界生效, 不会中断当前 step.';
    if (run.status === 'paused') return '训练已停在安全边界. 继续后从最近检查点恢复优化器状态.';
    if (run.status === 'queued') return `等待 ${esc(run.gpu)} 释放. 同一张卡不会被两个活动任务同时占用.`;
    if (run.status === 'failed') return '先检查数据路径与归档校验, 再重新排队.';
    return '该实验已结束. 可以从最终检查点派生一次新的对比实验.';
  }

  function baseLogs(run) {
    if (run.status === 'failed') {
      return [
        { at: '09:40:02', level: 'INFO', text: `design_preview=true / run=${run.id}` },
        { at: '09:40:14', level: 'ERROR', text: run.error },
      ];
    }
    if (run.status === 'queued') return [{ at: '10:46:00', level: 'INFO', text: `${run.name} 已加入队列, 等待 ${run.gpu}.` }];
    return [
      { at: '10:42:07', level: 'INFO', text: `CoreModel.setup("fit") → configure_optimizers() → Trainer.fit()` },
      { at: '10:42:09', level: 'INFO', text: `dataset=FakeData / batch=${run.batch} / device=${run.gpu.toLowerCase()}` },
      { at: '10:44:31', level: 'WARN', text: 'FakeData 为随机图像与标签, 指标仅用于验证链路.' },
      { at: '10:52:18', level: 'INFO', text: `checkpoint_saved / ${run.checkpoint ?? 'epoch_0.ckpt'} / model + optimizer` },
      { at: '10:58:40', level: 'INFO', text: `step=${run.step} / train/loss=${run.loss ?? '-'} / val/loss=${run.val ?? '-'}` },
    ];
  }

  function chartSvg(run) {
    const box = { x: 44, y: 12, w: 648, h: 208 };
    const digits = state.metric === 'loss' ? 1 : 0;
    const series = state.metric === 'loss'
      ? [{ key: 'loss', className: 'train', points: curve(run, 'loss', 60) }, { key: 'val', className: 'val', points: curve(run, 'val', 60) }]
      : [{ key: 'acc', className: 'train', points: curve(run, 'accuracy', 60) }];
    const domain = domainOf(series);
    const cursorX = box.x + box.w * 0.63;
    return `<svg viewBox="0 0 720 240" role="img" aria-label="训练曲线">
      ${gridMarks(box, domain, 4, digits)}
      <line class="grid" x1="${box.x}" y1="${box.y + box.h}" x2="${box.x + box.w}" y2="${box.y + box.h}"/>
      ${series.map((item) => `<path class="${item.className}" d="${linePath(item.points, box, domain).d}"/>`).join('')}
      <line class="cursor-line" x1="${cursorX}" y1="${box.y}" x2="${cursorX}" y2="${box.y + box.h}"/>
      <text class="axis" x="${box.x}" y="236">0</text><text class="axis" x="${box.x + box.w}" y="236" text-anchor="end">${run.total} steps</text>
    </svg>`;
  }

  function bindA(run) {
    $$('[data-pick]').forEach((button) => button.addEventListener('click', () => {
      state.selected = button.dataset.pick;
      state.logs = [];
      renderA();
    }));
    $$('[data-metric]').forEach((button) => button.addEventListener('click', () => {
      state.metric = button.dataset.metric;
      renderA();
    }));
    $$('[data-action]').forEach((button) => button.addEventListener('click', () => {
      const action = button.dataset.action;
      if (action === 'stop') {
        openDialog('停止这个实验?', `<p style="font-size:13px;color:#4a505c">已保存的检查点会保留, 可以从检查点派生新的实验. 此操作只影响本页面的模拟状态.</p><div style="display:flex;justify-content:flex-end;gap:8px;margin-top:18px"><button class="button" data-close>继续训练</button><button class="button danger" id="confirmStop">停止实验</button></div>`);
        $('#confirmStop').addEventListener('click', () => {
          run.status = 'stopped';
          addLog(`stop_requested / run=${run.id} / 等待安全边界`, 'WARN');
          $('#detailDialog').close();
          renderA();
          showToast('实验已停止, 检查点保留');
        });
        return;
      }
      if (action === 'pause') {
        run.status = 'paused';
        addLog('pause_requested / 将在 batch 边界生效', 'WARN');
        showToast('已请求暂停, 将在 batch 边界生效');
      }
      if (action === 'resume') {
        run.status = 'running';
        addLog('resume / 从最近检查点恢复', 'INFO');
        showToast('训练已继续');
      }
      if (action === 'checkpoint') {
        addLog(`checkpoint_saved / epoch_${run.epoch}.ckpt / model + optimizer`, 'INFO');
        showToast('检查点已保存');
      }
      if (action === 'cancel') {
        run.status = 'stopped';
        addLog('queue_cancelled', 'WARN');
        showToast('已取消排队');
      }
      if (action === 'restart') {
        showToast('已创建派生实验, 正在排队');
      }
      renderA();
    }));
    $('[data-new-run]')?.addEventListener('click', () => openDialog('新建实验', `<dl class="kv"><dt>模板</dt><dd>VGG-11 BN</dd><dt>数据</dt><dd>FakeData · 512 样本</dd><dt>设备</dt><dd>GPU 0</dd><dt>默认参数</dt><dd>20 epochs / batch 64 / lr 1e-3</dd></dl><p style="margin-top:14px;font-size:12px;color:#7d766a">正式界面会在这里插入模板参数表单, 只允许 catalog 内的字段.</p><div style="display:flex;justify-content:flex-end;gap:8px;margin-top:18px"><button class="button" data-close>取消</button><button class="button primary" data-close>加入队列</button></div>`));
    $('[data-open-config]')?.addEventListener('click', () => openDialog('实验配置', `<dl class="kv"><dt>模板</dt><dd>${esc(run.template)}</dd><dt>启动方式</dt><dd>${esc(run.launcher)}</dd><dt>设备</dt><dd>${esc(run.gpu)}</dd><dt>batch size</dt><dd>${run.batch}</dd><dt>学习率</dt><dd>${esc(run.lr)}</dd><dt>epochs</dt><dd>${run.epochs}</dd><dt>输出目录</dt><dd>outputs/${esc(run.id)}</dd></dl>`));
    $('[data-log-filter]')?.addEventListener('click', () => { state.logFilter = state.logFilter === 'warn' ? 'all' : 'warn'; renderA(); });
    $('[data-log-clear]')?.addEventListener('click', () => { state.logs = []; renderA(); });
    bindChartHover(run);
  }

  function bindChartHover(run) {
    const chart = $('#aChart');
    const tip = $('#aTip');
    if (!chart || !tip) return;
    chart.addEventListener('mousemove', (event) => {
      const rect = chart.getBoundingClientRect();
      const ratio = Math.min(1, Math.max(0, (event.clientX - rect.left - 44) / 648));
      const step = Math.round(ratio * run.total);
      const train = curve(run, state.metric === 'loss' ? 'loss' : 'accuracy', 60);
      const index = Math.min(train.length - 1, Math.round(ratio * (train.length - 1)));
      const sample = train[index];
      const second = state.metric === 'loss' ? curve(run, 'val', 60)[index] : null;
      tip.hidden = false;
      tip.innerHTML = `step ${step}<br>train ${sample.y}${second ? `<br>val ${second.y}` : ''}`;
      tip.style.left = `${Math.min(rect.width - 110, Math.max(8, event.clientX - rect.left + 12))}px`;
      tip.style.top = `${Math.max(6, event.clientY - rect.top - 34)}px`;
    });
    chart.addEventListener('mouseleave', () => { tip.hidden = true; });
  }

  /* ---------- B / 实验研究台 ---------- */

  const PALETTE = ['#2a58b8', '#c2772f', '#2f8a6d', '#8b5aa8', '#9aa4b2'];

  function bFiltered() {
    const query = state.bQuery.trim().toLowerCase();
    return runs.filter((run) => {
      const statusOk = state.bStatus === 'all' || run.status === state.bStatus;
      const queryOk = !query || run.name.toLowerCase().includes(query) || run.template.toLowerCase().includes(query);
      const isGpu = run.gpu.toLowerCase().startsWith('gpu');
      const deviceOk = state.bDevice === 'all' || (state.bDevice === 'gpu' ? isGpu : !isGpu);
      return statusOk && queryOk && deviceOk;
    });
  }

  function bSorted(list) {
    const { key, dir } = state.bSort;
    const factor = dir === 'asc' ? 1 : -1;
    return [...list].sort((left, right) => {
      const a = key === 'name' ? left.name : key === 'status' ? left.status : key === 'step' ? left.step : key === 'val' ? (left.val ?? 9) : (left.acc ?? 0);
      const b = key === 'name' ? right.name : key === 'status' ? right.status : key === 'step' ? right.step : key === 'val' ? (right.val ?? 9) : (right.acc ?? 0);
      return a > b ? factor : a < b ? -factor : 0;
    });
  }

  function renderB() {
    const filtered = bSorted(bFiltered());
    const picked = runs.filter((run) => state.bPicked.has(run.id));
    const counts = runs.reduce((acc, run) => ({ ...acc, [run.status]: (acc[run.status] ?? 0) + 1 }), {});
    const chips = [['all', '全部实验', runs.length], ['running', '训练中', counts.running ?? 0], ['queued', '排队中', counts.queued ?? 0], ['paused', '已暂停', counts.paused ?? 0], ['completed', '已完成', counts.completed ?? 0], ['failed', '失败', counts.failed ?? 0]];

    const box = { x: 46, y: 14, w: 632, h: 178 };
    const series = picked.slice(0, 4).map((run, index) => ({ run, color: PALETTE[index], points: curve(run, 'val', 44) }));
    const domain = domainOf(series);
    const legend = series.map((item) => `<span><i style="background:${item.color}"></i>${esc(item.run.name)}<button data-toggle-series="${item.run.id}" aria-label="移除 ${esc(item.run.name)}">×</button></span>`).join('');

    const rows = filtered.map((run) => {
      const bit = Array.from({ length: 14 }, (_, index) => {
        const on = run.total ? index / 14 < run.step / run.total : false;
        return `<i class="${on ? (run.status === 'failed' ? 'warn' : 'on') : ''}"></i>`;
      }).join('');
      return `<tr aria-selected="${state.bPicked.has(run.id)}">
        <td><input class="b-row-pick" type="checkbox" data-pick-row="${run.id}" ${state.bPicked.has(run.id) ? 'checked' : ''} aria-label="选择 ${esc(run.name)}"></td>
        <td><span class="b-name">${esc(run.name)}<small>${esc(run.template)} · ${esc(run.launcher)}</small></span></td>
        <td>${tagOf(run)}</td>
        <td class="mono">${run.step}/${run.total}</td>
        <td><div class="b-bitbar" title="进度">${bit}</div></td>
        <td class="mono">${run.val === null ? '-' : run.val.toFixed(3)}</td>
        <td class="mono">${run.acc === null ? '-' : `${run.acc.toFixed(1)}%`}</td>
        <td class="mono">${esc(run.gpu)}</td>
        <td><div class="b-actions"><button class="b-icon" data-open-run="${run.id}" aria-label="查看 ${esc(run.name)} 详情">···</button></div></td>
      </tr>`;
    }).join('');

    app.innerHTML = `<div class="app-b shell">
      <header class="b-top">
        <h2>实验研究台</h2>
        <nav class="b-nav" aria-label="视图"><button aria-pressed="true">对比</button><button>时间线</button><button>模板</button></nav>
        <div class="b-top-right"><span class="b-live"><i class="dot"></i>SSE 已连接</span><button class="button">导出 CSV</button><button class="button primary">新建实验</button></div>
      </header>
      <div class="b-body">
        <aside class="b-filters" aria-label="筛选">
          <div class="search" style="margin-bottom:14px"><input id="bSearch" placeholder="搜索实验名称或模板..." value="${esc(state.bQuery)}" aria-label="搜索实验"></div>
          <div class="b-filter-group"><span>状态</span>${chips.map(([key, label, count]) => `<button class="b-chip" data-status="${key}" aria-pressed="${state.bStatus === key}">${label}<em>${count}</em></button>`).join('')}</div>
          <div class="b-filter-group"><span>设备</span>${[['all', '全部'], ['gpu', 'GPU'], ['cpu', 'CPU']].map(([key, label]) => `<button class="b-chip" data-device="${key}" aria-pressed="${state.bDevice === key}">${label}</button>`).join('')}</div>
          <div class="b-filter-group"><span>已选对比</span><div style="padding:2px 9px;color:#4e5a6c;font-size:12px">${picked.length} 个实验</div><div style="padding:4px 9px;color:#8792a3;font-size:11px">勾选表格左侧复选框即可加入对比, 最多 4 个.</div></div>
        </aside>
        <div class="b-main">
          <div class="b-strip">
            <div class="b-kpi"><span>进行中</span><strong>${counts.running ?? 0}</strong><small>1 个 GPU 被占用</small></div>
            <div class="b-kpi"><span>排队</span><strong>${counts.queued ?? 0}</strong><small>等待 GPU 释放</small></div>
            <div class="b-kpi"><span>对比中</span><strong>${picked.length}</strong><small>上限 4 个</small></div>
            <div class="b-kpi"><span>最佳验证损失</span><strong>${bestVal()}</strong><small>来自已完成实验</small></div>
          </div>
          <section class="b-compare">
            <div class="b-compare-head"><h3>验证损失对比</h3><div class="b-legend">${legend || '<span style="color:#8792a3">从表格勾选实验加入对比</span>'}</div></div>
            <div class="b-chart">${picked.length ? `<svg viewBox="0 0 720 210" role="img" aria-label="多实验验证损失对比">${gridMarks(box, domain, 4, 1)}<line class="grid" x1="${box.x}" y1="${box.y + box.h}" x2="${box.x + box.w}" y2="${box.y + box.h}"/>${series.map((item) => `<path class="line" stroke="${item.color}" d="${linePath(item.points, box, domain).d}"/>`).join('')}<text class="b-axis-label" x="${box.x}" y="206">0</text><text class="b-axis-label" x="${box.x + box.w}" y="206" text-anchor="end">step</text></svg>` : `<div style="display:grid;place-items:center;height:100%;color:#8792a3;font-size:12px">还没有选中的实验</div>`}</div>
          </section>
          <div class="b-table-wrap">
            <div class="b-table-head"><h3>全部实验</h3><span class="muted" style="font-size:11.5px">${filtered.length} 条记录</span><div style="margin-left:auto;display:flex;gap:6px"><button class="button" data-sort="step">按 step 排序</button><button class="button" data-sort="val">按验证损失排序</button></div></div>
            <table class="b-table"><thead><tr><th style="width:30px"></th><th>实验</th><th>状态</th><th>step</th><th>进度</th><th>验证损失</th><th>准确率</th><th>设备</th><th></th></tr></thead><tbody>${rows || '<tr><td colspan="9" style="padding:28px;text-align:center;color:#8792a3">没有匹配的实验</td></tr>'}</tbody></table>
          </div>
        </div>
      </div>
    </div>`;

    bindB();
  }

  function bestVal() {
    const done = runs.filter((run) => run.val !== null);
    if (!done.length) return '-';
    return Math.min(...done.map((run) => run.val)).toFixed(3);
  }

  function bindB() {
    $$('[data-status]').forEach((button) => button.addEventListener('click', () => { state.bStatus = button.dataset.status; renderB(); }));
    $$('[data-device]').forEach((button) => button.addEventListener('click', () => { state.bDevice = button.dataset.device; renderB(); }));
    $('#bSearch')?.addEventListener('input', (event) => { state.bQuery = event.target.value; renderB(); const input = $('#bSearch'); if (input) { input.focus(); input.setSelectionRange(input.value.length, input.value.length); } });
    $$('[data-pick-row]').forEach((input) => input.addEventListener('change', () => {
      const id = input.dataset.pickRow;
      if (input.checked && state.bPicked.size >= 4) { input.checked = false; showToast('对比最多 4 个实验'); return; }
      if (input.checked) state.bPicked.add(id); else state.bPicked.delete(id);
      renderB();
    }));
    $$('[data-toggle-series]').forEach((button) => button.addEventListener('click', () => { state.bPicked.delete(button.dataset.toggleSeries); renderB(); }));
    $$('[data-sort]').forEach((button) => button.addEventListener('click', () => {
      const key = button.dataset.sort;
      state.bSort = { key, dir: state.bSort.key === key && state.bSort.dir === 'desc' ? 'asc' : 'desc' };
      renderB();
    }));
    $$('[data-open-run]').forEach((button) => button.addEventListener('click', () => {
      const run = runs.find((item) => item.id === button.dataset.openRun);
      if (!run) return;
      openDialog(run.name, `<dl class="kv"><dt>状态</dt><dd>${STATUS[run.status]}</dd><dt>模板</dt><dd>${esc(run.template)}</dd><dt>进度</dt><dd>${run.step} / ${run.total} steps</dd><dt>验证损失</dt><dd>${run.val === null ? '-' : run.val.toFixed(3)}</dd><dt>准确率</dt><dd>${run.acc === null ? '-' : `${run.acc.toFixed(1)}%`}</dd><dt>设备</dt><dd>${esc(run.gpu)}</dd><dt>启动方式</dt><dd>${esc(run.launcher)}</dd></dl><p style="margin-top:14px;font-size:12px;color:#7d8798">从详情可以直接进入单实验视图, 或把该实验加入对比.</p>`);
    }));
  }

  /* ---------- C / 流程工作室 ---------- */

  const NODES = [
    { name: '数据集', detail: 'FakeData · 512 样本', metric: 'shape 3×32×32' },
    { name: '预处理', detail: 'ToTensor + Normalize', metric: 'pipeline 3 步' },
    { name: '模型', detail: 'VGG-11 BN', metric: '133M 参数' },
    { name: '损失函数', detail: 'CrossEntropy', metric: 'reduction=mean' },
    { name: '优化器', detail: 'SGD + momentum', metric: 'lr 动态读取' },
    { name: '验证', detail: '每 epoch 一次', metric: '128 样本' },
    { name: '检查点', detail: 'ModelCheckpoint', metric: '保存到 outputs' },
  ];

  function preflight() {
    const issues = [];
    const config = state.cConfig;
    if (config.lr > 0.1) issues.push({ level: 'warn', text: `学习率 ${config.lr} 偏高, 分类任务通常从 1e-3 起步.` });
    if (config.batch > 128) issues.push({ level: 'warn', text: `batch size ${config.batch} 可能超出单卡显存, 建议配合梯度累积.` });
    if (config.epochs > 60) issues.push({ level: 'warn', text: `${config.epochs} 个 epoch 预计耗时较长, 建议先跑 2 个 epoch 冒烟.` });
    if (!config.val) issues.push({ level: 'warn', text: '已关闭验证集, 将没有 val/loss 与 accuracy 曲线.' });
    if (config.gpus > 1 && state.cConfig.launcher === 'single') issues.push({ level: 'warn', text: '多卡必须选择 torchrun 或 accelerate 启动方式.' });
    return issues;
  }

  const C_LIBRARY = [
    { type: 'dataset', title: '数据集', short: 'DATA', group: '数据', color: '#3f6ad1', detail: '输出样本与标签', target: 'FakeData', icon: 'D' },
    { type: 'transform', title: '预处理', short: 'TRANSFORM', group: '数据', color: '#2f8a9b', detail: '图像到张量', target: 'Compose', icon: 'T' },
    { type: 'dataloader', title: '批次加载', short: 'LOADER', group: '数据', color: '#607db8', detail: '组成 batch', target: 'DataLoader', icon: 'B' },
    { type: 'model', title: '模型', short: 'MODEL', group: '训练', color: '#6072d0', detail: '输出 logits', target: 'VGG-11 BN', icon: 'M' },
    { type: 'loss', title: '损失函数', short: 'LOSS', group: '训练', color: '#c2772f', detail: '产生 scalar loss', target: 'CrossEntropy', icon: 'L' },
    { type: 'optimizer', title: '优化器', short: 'OPTIMIZER', group: '训练', color: '#a06bc0', detail: '更新模型参数', target: 'AdamW', icon: 'O' },
    { type: 'trainer', title: '训练循环', short: 'TRAINER', group: '运行', color: '#5c83ad', detail: '驱动 epoch / step', target: 'XDL Trainer', icon: 'R' },
    { type: 'checkpoint', title: '检查点', short: 'CHECKPOINT', group: '运行', color: '#718194', detail: '保存模型状态', target: 'ModelCheckpoint', icon: 'C' },
  ];
  const C_REQUIRED = ['dataset', 'model', 'loss', 'trainer'];

  function cLibrary(type) {
    return C_LIBRARY.find((item) => item.type === type) || C_LIBRARY[0];
  }

  function cAddNode(type) {
    const meta = cLibrary(type);
    const index = state.cNodes.length;
    const node = { id: `node-${++state.cNodeId}`, type, name: meta.title, target: meta.target, x: 56 + (index % 3) * 208, y: 64 + Math.floor(index / 3) * 138, inputs: [] };
    state.cNodes.push(node);
    state.cSelected = node.id;
    state.cActivePanel = 'library';
    showToast(`已添加 ${meta.title} 节点, 可拖动到合适位置`);
    renderC();
  }

  function cNodePosition(node) {
    return { x: Number(node.x) || 0, y: Number(node.y) || 0 };
  }

  function cLinksMarkup() {
    return state.cNodes.flatMap((target) => target.inputs.map((sourceId) => {
      const source = state.cNodes.find((node) => node.id === sourceId);
      if (!source) return '';
      const from = cNodePosition(source);
      const to = cNodePosition(target);
      const x1 = from.x + 172;
      const y1 = from.y + 48;
      const x2 = to.x;
      const y2 = to.y + 48;
      const curve = Math.max(34, Math.abs(x2 - x1) * 0.45);
      return `<path class="c-graph-link" d="M${x1} ${y1} C${x1 + curve} ${y1}, ${x2 - curve} ${y2}, ${x2} ${y2}"/>`;
    })).join('');
  }

  function cGraphMarkup() {
    if (!state.cNodes.length) {
      return `<div class="c-graph-empty"><div class="c-empty-mark">+</div><strong>从空白画布开始</strong><p>从左侧节点库添加第一个节点, 再拖动节点并在检查器中选择上游输入.</p><span>建议顺序: 数据集 → 模型 → 损失 → 训练循环</span></div>`;
    }
    return `<svg class="c-graph-links" viewBox="0 0 1160 520" preserveAspectRatio="none" aria-hidden="true"><defs><marker id="cArrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7z" fill="#607080"/></marker></defs>${cLinksMarkup()}</svg>${state.cNodes.map((node) => {
      const meta = cLibrary(node.type);
      const isSelected = state.cSelected === node.id;
      const connected = node.inputs.length > 0;
      return `<article class="c-graph-node ${isSelected ? 'is-selected' : ''}" data-c-node="${node.id}" tabindex="0" aria-label="${esc(meta.title)}节点: ${esc(node.name)}" style="left:${node.x}px;top:${node.y}px;--node-color:${meta.color}">
        <div class="c-graph-node-head"><span class="c-node-icon">${meta.icon}</span><span>${meta.short}</span><button data-delete-node="${node.id}" aria-label="删除${esc(node.name)}">×</button></div>
        <strong>${esc(node.name)}</strong><small>${esc(node.target)}</small>
        <div class="c-graph-node-foot"><span class="c-port-label">${connected ? `${node.inputs.length} IN` : 'OPEN IN'}</span><span>${meta.detail}</span></div>
        <i class="c-port c-port-in" aria-hidden="true"></i><i class="c-port c-port-out" aria-hidden="true"></i>
      </article>`;
    }).join('')}`;
  }

  function cSettingsMarkup(config) {
    return `<div class="c-field"><label>Epochs <em>int</em></label><input data-c-setting="epochs" type="number" min="1" max="500" value="${config.epochs}"></div>
      <div class="c-field"><label>Batch size <em>int</em></label><input data-c-setting="batch" type="number" min="1" max="1024" value="${config.batch}"></div>
      <div class="c-field"><label>学习率 <em>float</em></label><input data-c-setting="lr" value="${esc(config.lr)}"></div>
      <div class="c-field"><label>启动方式 <em>enum</em></label><select data-c-setting="launcher"><option value="single" ${config.launcher === 'single' ? 'selected' : ''}>single process</option><option value="torchrun" ${config.launcher === 'torchrun' ? 'selected' : ''}>torchrun</option><option value="accelerate" ${config.launcher === 'accelerate' ? 'selected' : ''}>accelerate</option></select></div>
      <div class="c-field"><label>GPU 数量 <em>int</em></label><input data-c-setting="gpus" type="number" min="1" max="8" value="${config.gpus}"></div>
      <label class="c-check"><input data-c-setting="val" type="checkbox" ${config.val ? 'checked' : ''}> 启用验证集</label>
      <label class="c-check"><input data-c-setting="amp" type="checkbox" ${config.amp ? 'checked' : ''}> 混合精度 AMP</label>
      <label class="c-check"><input data-c-setting="checkpoint" type="checkbox" ${config.checkpoint ? 'checked' : ''}> 每个 epoch 保存检查点</label>
      <div class="c-budget"><strong><span>显存预算</span><span>${(config.batch * 0.012 + 2.1).toFixed(1)} GB</span></strong><div class="bar"><i style="width:${Math.min(100, (config.batch * 0.012 + 2.1) / 16 * 100)}%;background:#f0b35a"></i></div><small>RTX 4080 · 16 GB · 估算值</small></div>`;
  }

  function cInspectorMarkup() {
    const node = state.cNodes.find((item) => item.id === state.cSelected);
    if (!node) return `<div class="c-inspector-empty"><div class="c-empty-mark">⌖</div><h3>选择一个节点</h3><p>点击画布中的节点, 在这里编辑名称, 组件和输入连接.</p><div class="c-shortcuts"><span><b>拖动</b>重新排布</span><span><b>Delete</b>删除节点</span></div></div>`;
    const meta = cLibrary(node.type);
    const options = C_LIBRARY.filter((item) => item.type !== node.type && item.type !== 'checkpoint').map((item) => `<label class="c-input-row"><input type="checkbox" data-c-input="${item.type}" ${node.inputs.includes(state.cNodes.find((candidate) => candidate.type === item.type)?.id) ? 'checked' : ''}> <span>${item.title}</span></label>`).join('');
    return `<div class="c-inspector-head" style="--node-color:${meta.color}"><span class="c-node-icon">${meta.icon}</span><div><small>${meta.short}</small><strong>${esc(node.name)}</strong></div><button data-delete-node="${node.id}" class="c-delete">删除</button></div>
      <div class="c-inspector-body"><div class="c-field"><label>节点名称</label><input data-c-node-name value="${esc(node.name)}"></div><div class="c-field"><label>组件 target</label><select data-c-node-target><option>${esc(node.target)}</option><option>${esc(meta.target)}</option>${node.type === 'model' ? '<option>ResNet-18</option>' : ''}</select></div>
      <div class="c-inspector-section"><span>上游输入</span><small>勾选节点后建立连线</small>${options || '<em>暂无可用上游节点</em>'}</div><div class="c-shape"><span>输出形状</span><b>${meta.type === 'dataset' ? `[${state.cConfig.batch}, 3, 32, 32]` : meta.type === 'model' ? `[${state.cConfig.batch}, 10] logits` : meta.type === 'loss' ? 'scalar loss' : meta.detail}</b></div></div>`;
  }

  function cDiagnostics() {
    if (!state.cNodes.length) return [{ level: 'error', text: '画布为空, 至少添加数据集, 模型, 损失函数和训练循环节点.' }];
    const types = new Set(state.cNodes.map((node) => node.type));
    const issues = [];
    const missing = C_REQUIRED.filter((type) => !types.has(type));
    if (missing.length) issues.push({ level: 'error', text: `缺少必需节点: ${missing.map((type) => cLibrary(type).title).join(' / ')}` });
    const open = state.cNodes.filter((node) => node.type !== 'dataset' && !node.inputs.length);
    if (open.length) issues.push({ level: 'warn', text: `${open.length} 个节点没有上游输入: ${open.map((node) => node.name).join(', ')}` });
    if (!types.has('checkpoint')) issues.push({ level: 'warn', text: '没有检查点节点, 训练中断后无法从历史 step 恢复.' });
    if (state.cConfig.gpus > 1 && state.cConfig.launcher === 'single') issues.push({ level: 'error', text: '多卡配置需要 torchrun 或 accelerate 启动方式.' });
    if (state.cConfig.batch > 128) issues.push({ level: 'warn', text: `batch size ${state.cConfig.batch} 可能超出单卡显存, 建议配合梯度累积.` });
    return issues;
  }

  function renderC() {
    const issues = cDiagnostics();
    const stateLabels = { draft: '草稿', queued: '已排队', running: '运行中', failed: '上次失败' };
    const selectedCount = state.cNodes.length;
    const library = ['数据', '训练', '运行'].map((group) => `<div class="c-library-group"><span>${group}</span>${C_LIBRARY.filter((item) => item.group === group).map((item) => `<button class="c-library-item" data-add-node="${item.type}" style="--node-color:${item.color}"><i>${item.icon}</i><span><b>${item.title}</b><small>${item.detail}</small></span><em>+</em></button>`).join('')}</div>`).join('');
    const isSettings = state.cActivePanel === 'settings';
    const preflightOpen = state.cShowPreflight || issues.length > 0;
    const diagnosticLabel = issues.some((item) => item.level === 'error') ? `${issues.filter((item) => item.level === 'error').length} 个错误` : issues.length ? `${issues.length} 个提醒` : '全部通过';
    const nodeSummary = selectedCount ? `${selectedCount} nodes · ${state.cNodes.filter((node) => node.inputs.length).length} links` : 'empty canvas';

    app.innerHTML = `<div class="app-c shell">
      <div class="railbar" aria-label="工作流导航"><span class="logo">X</span><button aria-pressed="true" title="节点画布">流</button><button title="实验">实</button><button title="产物">产</button><hr><button title="设置">设</button></div>
      <aside class="c-config" aria-label="节点库和流程设置"><div class="c-config-title"><div><h3>工作流构建器</h3><p class="c-config-sub">${nodeSummary}</p></div><button class="c-icon-btn" data-clear-canvas aria-label="清空画布">清空</button></div>
        <div class="c-tabs"><button data-c-panel="library" aria-pressed="${!isSettings}">节点库</button><button data-c-panel="settings" aria-pressed="${isSettings}">流程设置</button></div>
        ${isSettings ? cSettingsMarkup(state.cConfig) : `<div class="c-library-search"><input placeholder="搜索节点类型" aria-label="搜索节点类型"><span>⌕</span></div>${library}<div class="c-library-note">点击添加节点, 再拖动节点排布. 连线在右侧检查器中建立.</div>`}
      </aside>
      <div class="c-stage"><div class="c-toolbar"><div class="c-flow-title"><input id="cFlowName" value="新建训练流水线" aria-label="流水线名称"><small>${stateLabels[state.cRunState]} · ${nodeSummary}</small></div><div class="c-state-pills" aria-label="预览状态"><span>状态预览</span>${Object.entries(stateLabels).map(([key, label]) => `<button data-c-state="${key}" aria-pressed="${state.cRunState === key}">${label}</button>`).join('')}</div><button class="button" data-auto-layout>自动排布</button><button class="button" data-show-preflight>预检 ${issues.length || '✓'}</button><button class="button primary" data-launch>校验并排队</button></div>
        <div class="c-preflight ${issues.some((item) => item.level === 'error') ? 'has-error' : issues.length ? 'has-warn' : ''}"><span>PREFLIGHT</span><b>${diagnosticLabel}</b><span>·</span><span>${selectedCount ? `${selectedCount} 个节点 / ${state.cNodes.reduce((sum, node) => sum + node.inputs.length, 0)} 条连线` : '尚未开始搭建'}</span><button data-show-preflight>${preflightOpen ? '收起详情' : '查看详情'}</button></div>
        ${preflightOpen ? `<div class="c-preflight-panel">${issues.length ? issues.map((item) => `<div class="c-issue ${item.level}"><i>${item.level === 'error' ? '!' : 'i'}</i><span>${esc(item.text)}</span></div>`).join('') : '<div class="c-issue ok"><i>✓</i><span>节点齐全, 输入连接完整, 可以加入训练队列.</span></div>'}</div>` : ''}
        <div class="c-main"><div class="c-canvas-panel"><div class="c-canvas-head"><div><h3>画布</h3><span>拖动节点自由排布 · 点击节点编辑输入</span></div><span class="c-zoom">100%　−　+</span></div><div class="c-graph-canvas" data-canvas>${cGraphMarkup()}</div></div><aside class="c-inspector" aria-label="节点检查器">${cInspectorMarkup()}</aside></div>
      </div>
    </div>`;
    bindCGraph();
  }

  function bindCGraph() {
    $$('[data-add-node]').forEach((button) => button.addEventListener('click', () => cAddNode(button.dataset.addNode)));
    $$('[data-c-panel]').forEach((button) => button.addEventListener('click', () => { state.cActivePanel = button.dataset.cPanel; renderC(); }));
    $$('[data-c-state]').forEach((button) => button.addEventListener('click', () => { state.cRunState = button.dataset.cState; renderC(); }));
    $$('[data-show-preflight]').forEach((button) => button.addEventListener('click', () => { state.cShowPreflight = !state.cShowPreflight; renderC(); }));
    $('[data-clear-canvas]')?.addEventListener('click', () => { state.cNodes = []; state.cSelected = null; state.cShowPreflight = true; renderC(); showToast('画布已清空'); });
    $('[data-auto-layout]')?.addEventListener('click', () => { state.cNodes.forEach((node, index) => { node.x = 56 + (index % 4) * 214; node.y = 66 + Math.floor(index / 4) * 140; }); renderC(); showToast('已按节点顺序自动排布'); });
    $('[data-canvas]')?.addEventListener('click', (event) => { if (event.target === event.currentTarget) { state.cSelected = null; renderC(); } });
    $$('[data-c-node]').forEach((element) => {
      element.addEventListener('click', () => { state.cSelected = element.dataset.cNode; renderC(); });
      element.addEventListener('pointerdown', (event) => {
        if (event.target.closest('button, input, select')) return;
        const node = state.cNodes.find((item) => item.id === element.dataset.cNode);
        if (!node) return;
        const start = { x: event.clientX, y: event.clientY, left: node.x, top: node.y };
        const move = (moveEvent) => { node.x = Math.max(10, start.left + moveEvent.clientX - start.x); node.y = Math.max(10, start.top + moveEvent.clientY - start.y); element.style.left = `${node.x}px`; element.style.top = `${node.y}px`; const links = $('.c-graph-links'); if (links) links.innerHTML = `<defs><marker id="cArrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7z" fill="#607080"/></marker></defs>${cLinksMarkup()}`; };
        const up = () => { element.classList.remove('is-dragging'); window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); };
        element.classList.add('is-dragging'); window.addEventListener('pointermove', move); window.addEventListener('pointerup', up);
      });
    });
    $$('[data-delete-node]').forEach((button) => button.addEventListener('click', (event) => { event.stopPropagation(); const id = button.dataset.deleteNode; state.cNodes = state.cNodes.filter((node) => node.id !== id).map((node) => ({ ...node, inputs: node.inputs.filter((input) => input !== id) })); state.cSelected = null; renderC(); showToast('节点已移除'); }));
    $('[data-c-node-name]')?.addEventListener('change', (event) => { const node = state.cNodes.find((item) => item.id === state.cSelected); if (node) { node.name = event.target.value || cLibrary(node.type).title; renderC(); } });
    $('[data-c-node-target]')?.addEventListener('change', (event) => { const node = state.cNodes.find((item) => item.id === state.cSelected); if (node) { node.target = event.target.value; renderC(); } });
    $$('[data-c-input]').forEach((input) => input.addEventListener('change', () => { const node = state.cNodes.find((item) => item.id === state.cSelected); const source = state.cNodes.find((item) => item.type === input.dataset.cInput); if (!node || !source) return; node.inputs = input.checked ? [...new Set([...node.inputs, source.id])] : node.inputs.filter((id) => id !== source.id); renderC(); }));
    $$('[data-c-setting]').forEach((input) => input.addEventListener('change', () => { const key = input.dataset.cSetting; state.cConfig[key] = input.type === 'checkbox' ? input.checked : input.type === 'number' ? Number(input.value) || 1 : input.value; renderC(); }));
    $('#cFlowName')?.addEventListener('change', (event) => { event.target.value = event.target.value.trim() || '新建训练流水线'; });
    $('[data-launch]')?.addEventListener('click', () => { const issues = cDiagnostics(); state.cShowPreflight = true; if (issues.some((item) => item.level === 'error')) { renderC(); showToast('请先处理阻断错误'); return; } state.cRunState = 'queued'; renderC(); showToast('流水线已校验并加入队列'); });
  }

  function renderCLegacy() {
    const config = state.cConfig;
    const issues = preflight();
    const blocked = issues.length > 0;
    const node = NODES[state.cNode];

    const chain = NODES.map((item, index) => {
      const isBlocked = item.name === '验证' && !config.val;
      const status = isBlocked ? '已跳过 · 验证集关闭' : index < 4 ? '就绪' : '待启动';
      return `<button class="c-node ${isBlocked ? 'blocked' : ''}" data-node="${index}" aria-current="${index === state.cNode}">
        <span class="num">${String(index + 1).padStart(2, '0')}</span>
        <strong>${item.name}</strong>
        <small>${esc(item.detail)}</small>
        <span class="status-note">${status}</span>
      </button>${index < NODES.length - 1 ? '<span class="c-link"></span>' : ''}`;
    }).join('');

    app.innerHTML = `<div class="app-c shell">
      <div class="railbar" aria-label="主导航">
        <span class="logo">X</span>
        <button aria-pressed="true" title="流程">流</button>
        <button title="实验">实</button>
        <button title="产物">产</button>
        <hr>
        <button title="设置">设</button>
      </div>
      <aside class="c-config" aria-label="实验配置">
        <h3>实验配置</h3>
        <p class="c-config-sub">workflow / vgg-cifar10</p>
        <div class="c-tabs"><button aria-pressed="true">参数</button><button>设备</button><button>输出</button></div>
        <div class="c-field"><label>Epochs <em>int</em></label><input id="cEpochs" type="number" min="1" max="500" value="${config.epochs}"></div>
        <div class="c-field"><label>Batch size <em>int</em></label><input id="cBatch" type="number" min="1" max="1024" value="${config.batch}"></div>
        <div class="c-field"><label>学习率 <em>float</em></label><input id="cLr" value="${esc(config.lr)}"></div>
        <div class="c-field"><label>启动方式 <em>enum</em></label><select id="cLauncher"><option value="single" ${config.launcher === 'single' ? 'selected' : ''}>single process</option><option value="torchrun" ${config.launcher === 'torchrun' ? 'selected' : ''}>torchrun</option><option value="accelerate" ${config.launcher === 'accelerate' ? 'selected' : ''}>accelerate</option></select></div>
        <div class="c-field"><label>GPU 数量 <em>int</em></label><input id="cGpus" type="number" min="1" max="8" value="${config.gpus}"></div>
        <label class="c-check"><input type="checkbox" id="cVal" ${config.val ? 'checked' : ''}> 启用验证集</label>
        <label class="c-check"><input type="checkbox" id="cAmp" ${config.amp ? 'checked' : ''}> 混合精度 AMP</label>
        <label class="c-check"><input type="checkbox" id="cCkpt" ${config.checkpoint ? 'checked' : ''}> 每个 epoch 保存检查点</label>
        <div class="c-budget"><strong><span>显存预算</span><span>${(config.batch * 0.012 + 2.1).toFixed(1)} GB</span></strong><div class="bar"><span style="width:${Math.min(100, (config.batch * 0.012 + 2.1) / 16 * 100)}%;background:#f0b35a"></span><span style="width:${Math.max(0, 100 - (config.batch * 0.012 + 2.1) / 16 * 100)}%;background:#232a33"></span></div><small>RTX 4080 · 16 GB · 估算值</small></div>
      </aside>
      <div class="c-stage">
        <div class="c-toolbar"><h2>vgg-cifar10 <small>单卡 · ${config.epochs} epochs</small></h2><button class="button">另存为</button><button class="button primary" data-launch>校验并排队</button></div>
        <div class="c-preflight"><span>PREFLIGHT</span><b>${blocked ? `${issues.length} 项提醒` : '全部通过'}</b><span>·</span><span>${config.epochs} epochs × ${Math.ceil(512 / config.batch)} steps</span><span>·</span><span>约 ${Math.max(1, Math.round(config.epochs * 0.9))} 分钟</span></div>
        <div class="c-chain" aria-label="训练流程">${chain}</div>
        <div class="c-main">
          <div class="c-preview">
            <div class="c-preview-head"><h3>数据形状预览 · 单 batch</h3><div class="c-pattern"><button data-pattern="full" aria-pressed="${state.cPattern === 'full'}">完整链路</button><button data-pattern="check" aria-pressed="${state.cPattern === 'check'}">仅冒烟</button></div></div>
            <div class="c-viz">${previewSvg()}</div>
            <div class="c-timeline">${Array.from({ length: 20 }, (_, index) => `<i class="${index < Math.round(20 * 0.6) ? 'done' : index === Math.round(20 * 0.6) ? 'now' : ''}"></i>`).join('')}</div>
            <div class="c-timeline-caption"><span>epoch 12 / ${config.epochs}</span><span>预计剩余 ${Math.max(1, Math.round((config.epochs - 12) * 0.9))} 分钟</span></div>
          </div>
          <aside class="c-inspector" aria-label="节点检查">
            <h3>节点检查</h3>
            <dl class="c-kv"><dt>节点</dt><dd>${node.name}</dd><dt>实现</dt><dd>${esc(node.detail)}</dd><dt>指标</dt><dd>${esc(node.metric)}</dd><dt>输入</dt><dd>[${config.batch}, 3, 32, 32]</dd><dt>设备</dt><dd>cuda:${Math.max(0, config.gpus - 1)}</dd><dt>状态</dt><dd style="color:${blocked ? '#f0b35a' : '#5fbc8f'}">${blocked ? '需确认' : '可运行'}</dd></dl>
            <div class="c-rule"><b>为什么这样连</b>数据先经过预处理再进入模型, 损失函数读取 logits 与标签, 优化器只在 backward 之后更新参数. 检查点挂在 epoch 边界, 不参与前向计算.</div>
            <div class="c-launch-flag">${blocked ? `启动前有 ${issues.length} 项需要确认, 点击右上角查看明细.` : '配置校验通过, 可以直接加入队列.'}</div>
          </aside>
        </div>
      </div>
    </div>`;

    bindCLegacy();
  }

  function previewSvg() {
    const batch = state.cConfig.batch;
    const blocks = [['数据', `${batch}×3×32×32`, '#3f6ad1'], ['模型', `${batch}×10`, '#2f8a6d'], ['损失', 'scalar', '#c2772f']];
    return `<svg viewBox="0 0 460 170" role="img" aria-label="单 batch 数据形状">
      ${blocks.map((block, index) => {
        const x = 24 + index * 148;
        return `<g><rect x="${x}" y="42" width="116" height="62" rx="9" fill="#1a1f26" stroke="#333a45"/><text x="${x + 58}" y="68" text-anchor="middle" fill="#eef1f6" font-size="12" font-family="sans-serif">${block[0]}</text><text x="${x + 58}" y="88" text-anchor="middle" fill="#8b95a3" font-size="10" font-family="monospace">${block[1]}</text></g>${index < 2 ? `<path d="M${x + 120} 73 L${x + 142} 73" stroke="#3c4552" stroke-width="1.5" marker-end="url(#arrow)"/>` : ''}`;
      }).join('')}
      <defs><marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6z" fill="#3c4552"/></marker></defs>
      <text x="24" y="132" fill="#6d7784" font-size="10" font-family="monospace">forward → loss → backward → step</text>
      <text x="24" y="150" fill="#6d7784" font-size="10" font-family="monospace">checkpoint 挂在 epoch 边界, 不参与前向</text>
    </svg>`;
  }

  function bindCLegacy() {
    $$('[data-node]').forEach((button) => button.addEventListener('click', () => { state.cNode = Number(button.dataset.node); renderC(); }));
    $$('[data-pattern]').forEach((button) => button.addEventListener('click', () => { state.cPattern = button.dataset.pattern; renderC(); }));
    $('#cEpochs')?.addEventListener('change', (event) => { state.cConfig.epochs = Number(event.target.value) || 1; renderC(); });
    $('#cBatch')?.addEventListener('change', (event) => { state.cConfig.batch = Number(event.target.value) || 1; renderC(); });
    $('#cLr')?.addEventListener('change', (event) => { state.cConfig.lr = Number(event.target.value) || 0.001; renderC(); });
    $('#cGpus')?.addEventListener('change', (event) => { state.cConfig.gpus = Number(event.target.value) || 1; renderC(); });
    $('#cLauncher')?.addEventListener('change', (event) => { state.cConfig.launcher = event.target.value; renderC(); });
    $('#cVal')?.addEventListener('change', (event) => { state.cConfig.val = event.target.checked; renderC(); });
    $('#cAmp')?.addEventListener('change', (event) => { state.cConfig.amp = event.target.checked; renderC(); });
    $('#cCkpt')?.addEventListener('change', (event) => { state.cConfig.checkpoint = event.target.checked; renderC(); });
    $('[data-launch]')?.addEventListener('click', () => {
      const issues = preflight();
      if (!issues.length) { showToast('配置校验通过, 实验已加入队列'); return; }
      openDialog('启动前需要确认', `<ul style="margin:0;padding-left:18px;font-size:12.5px;line-height:2;color:#4a505c">${issues.map((issue) => `<li>${esc(issue.text)}</li>`).join('')}</ul><div style="display:flex;justify-content:flex-end;gap:8px;margin-top:18px"><button class="button" data-close>返回修改</button><button class="button primary" id="forceLaunch">仍然排队</button></div>`);
      $('#forceLaunch')?.addEventListener('click', () => { $('#detailDialog').close(); showToast('已按当前配置加入队列'); });
    });
  }

  /* ---------- shell ---------- */

  function setConcept(concept) {
    state.concept = concept;
    $$('.review-bar nav button').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.concept === concept)));
    $('#conceptEyebrow').textContent = CONCEPTS[concept].eyebrow;
    $('#conceptTitle').textContent = CONCEPTS[concept].title;
    $('#conceptDescription').textContent = CONCEPTS[concept].description;
    $('#directionNote').textContent = CONCEPTS[concept].note;
    if (concept === 'a') renderA();
    if (concept === 'b') renderB();
    if (concept === 'c') renderC();
  }

  $$('.review-bar nav button').forEach((button) => button.addEventListener('click', () => setConcept(button.dataset.concept)));
  $('#closeDialog').addEventListener('click', () => $('#detailDialog').close());
  document.addEventListener('click', (event) => { if (event.target.closest('[data-close]')) $('#detailDialog').close(); });
  $('#detailDialog').addEventListener('click', (event) => { if (event.target === $('#detailDialog')) $('#detailDialog').close(); });
  document.addEventListener('keydown', (event) => { if (event.key === 'Escape' && $('#detailDialog').open) $('#detailDialog').close(); });

  const initial = ['a', 'b', 'c'].includes(window.location.hash.slice(1)) ? window.location.hash.slice(1) : 'a';
  setConcept(initial);
  window.addEventListener('hashchange', () => {
    const target = window.location.hash.slice(1);
    if (['a', 'b', 'c'].includes(target)) setConcept(target);
  });
})();
