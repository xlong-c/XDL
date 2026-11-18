'use strict';

(() => {
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const icon = (name) => `<svg class="icon" aria-hidden="true"><use href="#i-${name}"/></svg>`;
  const escape = (value) => String(value).replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character]);
  const labels = { running: '训练中', paused: '已暂停', queued: '排队中', completed: '已完成', failed: '失败', stopped: '已停止' };
  const badge = (status) => `<span class="status ${status}"><i class="dot"></i>${labels[status]}</span>`;
  const templates = {
    vgg: { model: 'VGG-11 BN', title: '图像分类 / VGG', icon: 'layers', description: '用 CIFAR-10 形状的合成图像走通训练,评估和检查点流程.', meta: 'FakeData · 3 × 32 × 32', epochs: 20, batch: 64, lr: 0.001 },
    resnet: { model: 'ResNet-18', title: '图像分类 / ResNet', icon: 'box', description: '同一任务, 不同模型. 用于对比实验配置和训练操作流程.', meta: 'FakeData · 3 × 32 × 32', epochs: 20, batch: 64, lr: 0.001 },
    mlp: { model: 'MLP', title: '快速验证 / MLP', icon: 'bolt', description: '更轻量的合成数据任务, 快速检查训练入口与回调生命周期.', meta: 'FakeData · 3 × 32 × 32', epochs: 4, batch: 32, lr: 0.01 },
  };
  const runs = [
    { id: 'demo-a31f82', name: 'vgg11-local-demo', template: 'vgg', status: 'running', epochs: 20, step: 98, batch: 64, lr: 0.001, launcher: 'single', loss: 2.307, valLoss: 2.321, accuracy: 10.2, elapsed: '08:42', gpu: 74, saved: 96 },
    { id: 'demo-b72c16', name: 'resnet18-baseline', template: 'resnet', status: 'queued', epochs: 20, step: 0, batch: 64, lr: 0.001, launcher: 'torchrun', loss: null, valLoss: null, accuracy: null, elapsed: '00:00', gpu: 0, saved: 0 },
    { id: 'demo-c93d40', name: 'mlp-smoke-test', template: 'mlp', status: 'completed', epochs: 4, step: 64, batch: 32, lr: 0.01, launcher: 'single', loss: 2.299, valLoss: 2.314, accuracy: 10.4, elapsed: '02:18', gpu: 0, saved: 64 },
    { id: 'demo-d18e59', name: 'vgg11-lr-check', template: 'vgg', status: 'paused', epochs: 20, step: 48, batch: 64, lr: 0.003, launcher: 'accelerate', loss: 2.329, valLoss: 2.338, accuracy: 9.8, elapsed: '04:07', gpu: 0, saved: 48 },
    { id: 'demo-e56f20', name: 'cifar10-download-check', template: 'vgg', status: 'failed', epochs: 20, step: 0, batch: 64, lr: 0.001, launcher: 'single', loss: null, valLoss: null, accuracy: null, elapsed: '00:12', gpu: 0, saved: 0, error: '示例错误: 数据归档校验失败, 训练尚未开始. 可使用 FakeData 模板验证训练流程.' },
  ];
  let selectedId = runs[0].id;
  let currentPage = 'overview';
  let currentTab = 'overview';
  let currentMetric = 'loss';
  let toastTimer;
  let chartData = [];
  let resumeParent = null;
  const activeRun = () => runs.find((run) => run.id === selectedId);
  const runModel = (run) => run.workflowSnapshot ? ({ vgg11_bn: 'VGG-11 BN', vgg16_bn: 'VGG-16 BN', resnet18: 'ResNet-18', resnet34: 'ResNet-34' }[run.workflowSnapshot.values.model] || run.workflowSnapshot.values.model) : templates[run.template].model;
  const runSamples = (run) => run.workflowSnapshot ? (run.workflowSnapshot.values.dataset === 'FakeData' ? run.workflowSnapshot.values.samples : null) : 512;
  const stepsPerEpoch = (run) => Math.ceil((runSamples(run) || run.batch) / run.batch);
  const totalSteps = (run) => run.epochs * stepsPerEpoch(run);
  const progress = (run) => Math.min(100, Math.round(run.step / totalSteps(run) * 100));
  const epochLabel = (run) => run.step ? `${Math.min(run.epochs, Math.floor((run.step - 1) / stepsPerEpoch(run)) + 1)} / ${run.epochs}` : `0 / ${run.epochs}`;
  const checkpointName = (run) => `step_${String(run.saved).padStart(4, '0')}.ckpt`;
  const canControl = (run) => ['running', 'paused'].includes(run.status);
  const time = () => new Intl.DateTimeFormat('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }).format(new Date());

  function showToast(message) {
    clearTimeout(toastTimer);
    $('#toastText').textContent = message;
    $('#toast').hidden = false;
    toastTimer = setTimeout(() => { $('#toast').hidden = true; }, 4300);
  }

  function download(name, content, type = 'application/json') {
    const blob = new Blob([content], { type: `${type};charset=utf-8` });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = name;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function initialLogs(run) {
    if (run.status === 'failed') return [{ at: '15:28:00', level: 'INFO', message: 'design_preview=true / checking CIFAR-10 archive' }, { at: '15:28:12', level: 'ERROR', message: run.error }];
    if (!run.step) return [{ at: '15:30:00', level: 'INFO', message: `模拟实验 ${run.name} 已加入队列, 等待 GPU 0.` }];
    return [
      { at: '15:20:00', level: 'INFO', message: `design_preview=true / run=${run.id} / model=${runModel(run)}` },
      { at: '15:20:01', level: 'WARN', message: 'FakeData contains random images and labels. Metrics are illustrative; this is not official CIFAR-10.' },
      { at: '15:20:02', level: 'INFO', message: `dataset=FakeData / train_samples=512 / batch_size=${run.batch} / device=cuda:0` },
      { at: '15:20:03', level: 'INFO', message: 'CoreModel.setup("fit") → configure_optimizers() → Trainer.fit()' },
      { at: '15:28:12', level: 'INFO', message: `checkpoint_saved / step=${run.saved} / model + optimizer state` },
      { at: '15:28:40', level: 'INFO', message: `step=${run.step} / train/loss=${run.loss} / val/loss=${run.valLoss} / val/accuracy=${run.accuracy}%` },
    ];
  }
  runs.forEach((run) => { run.logs = initialLogs(run); });

  function addLog(run, message, level = 'INFO') {
    run.logs.push({ at: time(), level, message });
  }

  function renderRecent() {
    const recent = runs.slice(0, 3);
    $('#pinnedRuns').innerHTML = recent.map((run) => `<button class="pinned ${run.id === selectedId ? 'selected' : ''}" data-select-run="${run.id}" aria-pressed="${run.id === selectedId}"><i class="dot ${run.status}"></i><span><strong>${escape(run.name)}</strong><small>${runModel(run)} · ${labels[run.status]}</small></span></button>`).join('');
    $('#navRunCount').textContent = runs.length;
  }

  function selectRun(id) {
    if (!runs.some((run) => run.id === id)) return;
    selectedId = id;
    currentTab = 'overview';
    setPage('overview');
  }

  function setPage(page) {
    const titles = { overview: ['训练工作台', '从一次实验开始, 看清训练的每一步.', '训练概览'], experiments: ['实验管理', '每一次尝试, 都有迹可循.', '全部实验'], templates: ['训练模板', '把常用配置变成下一次实验的起点.', '训练模板'], artifacts: ['模型产物', '集中管理训练留下的检查点与配置.', '模型产物'], resources: ['计算资源', '看清设备占用, 再安排下一次实验.', '计算资源'], workflow: ['工作流编辑器', '固定节点训练流程.', '工作流编辑器'] };
    if (!titles[page]) page = 'overview';
    currentPage = page;
    document.body.classList.toggle('overview-mode', page === 'overview');
    document.body.classList.toggle('workflow-mode', page === 'workflow');
    $('#pageTitle').textContent = titles[page][0];
    $('#pageSubtitle').textContent = titles[page][1];
    $('#breadcrumbCurrent').textContent = titles[page][2];
    $('#runWorkspace').hidden = page !== 'overview';
    $$('[id^="page-"]').forEach((section) => { section.hidden = section.id !== `page-${page}`; });
    $$('[data-page]').forEach((button) => {
      button.classList.toggle('active', button.dataset.page === page);
      if (button.dataset.page === page) button.setAttribute('aria-current', 'page');
      else button.removeAttribute('aria-current');
    });
    $('#sidebar').classList.remove('open');
    $('#menuToggle').setAttribute('aria-expanded', 'false');
    if (page === 'experiments') renderExperiments();
    if (page === 'artifacts') renderArtifacts();
    if (page === 'resources') renderResourceQueue();
    if (page === 'overview') renderRun();
    history.replaceState(null, '', `#${page}`);
  }

  function setTab(tab, moveFocus = false) {
    currentTab = tab;
    $$('[data-tab]').forEach((button) => {
      const isSelected = button.dataset.tab === tab;
      button.setAttribute('aria-selected', String(isSelected));
      button.tabIndex = isSelected ? 0 : -1;
      if (moveFocus && isSelected) button.focus();
    });
    $$('[role="tabpanel"]').forEach((panel) => { panel.hidden = panel.id !== `panel-${tab}`; });
    if (tab === 'overview') renderChart();
    if (tab === 'logs') renderLogs();
  }

  function renderRun() {
    const run = activeRun();
    const model = runModel(run);
    renderRecent();
    $('#runName').textContent = run.name;
    const status = $('#runStatus');
    status.className = `status ${run.status}`;
    status.innerHTML = `<i class="dot"></i>${labels[run.status]}`;
    $('#runMeta').innerHTML = `<span class="mono">${escape(run.id)}</span><i class="divider"></i><span>${model}</span><i class="divider"></i><span>FakeData · 512 张</span><i class="divider"></i><span>GPU 0</span>`;
    $('#pauseButton').disabled = !canControl(run);
    $('#pauseButton').innerHTML = `${icon(run.status === 'paused' ? 'play' : 'pause')}<span>${run.status === 'paused' ? '继续训练' : '暂停训练'}</span>`;
    $('#checkpointButton').disabled = !canControl(run);
    $('#stopButton').disabled = !['running', 'paused', 'queued'].includes(run.status);
    $('#stopButton').title = run.status === 'queued' ? '取消排队' : '停止实验';
    $('#runNotice').hidden = true;
    if (run.status !== 'running') {
      const descriptions = { paused: '实验已暂停. 模型状态保留在内存中, 点击继续训练可恢复模拟运行状态.', queued: '实验正在等待 GPU 0. 训练开始后, 这里会显示曲线,指标和检查点.', completed: '实验已完成. 可查看最终指标, 或从已保存的检查点创建新实验.', failed: run.error, stopped: '实验已停止. 已保存的检查点仍可用于创建恢复实验.' };
      $('#runNotice').hidden = false;
      $('#runNotice').innerHTML = `<div class="notice ${run.status === 'failed' ? 'error' : 'warning'}">${icon('info')}${escape(descriptions[run.status])}</div>`;
    }
    renderMetrics(run);
    $('#gpuUsage').innerHTML = `${run.gpu}<small>%</small>`;
    $('#gpuBar').style.width = `${run.gpu}%`;
    $('#chartStep').textContent = `${run.step} steps`;
    $('#chartState').textContent = run.status === 'running' ? '模拟指标已就绪' : `模拟快照 · ${labels[run.status]}`;
    $('#overviewPipeline').innerHTML = pipelineMarkup(run, false);
    $('#expandedPipeline').innerHTML = pipelineMarkup(run, true);
    document.body.classList.toggle('is-paused', run.status === 'paused');
    document.body.classList.toggle('is-inactive', !['running', 'paused'].includes(run.status));
    renderActivity(run);
    renderCheckpoint(run);
    renderConfig(run);
    setTab(currentTab);
  }

  function renderMetrics(run) {
    const values = [
      { title: '训练进度', icon: 'layers', value: `${epochLabel(run).split(' / ')[0]} <small>/ ${run.epochs} epochs</small>`, foot: `Step ${run.step} / ${totalSteps(run)}`, bar: true },
      { title: '训练损失', icon: 'chart', value: run.loss === null ? '-' : run.loss.toFixed(3), foot: run.loss === null ? '训练尚未开始' : '<span class="positive">稳定</span> · CrossEntropy' },
      { title: '验证准确率', icon: 'check', value: run.accuracy === null ? '-' : `${run.accuracy.toFixed(1)}<small>%</small>`, foot: '随机标签 · 不评估收敛效果' },
      { title: '已用时间', icon: 'clock', value: `${run.elapsed} <small>mm:ss</small>`, foot: run.status === 'running' ? '186 ms / step · 示例耗时' : `${labels[run.status]} · 保存当前快照` },
    ];
    $('#metrics').innerHTML = values.map((item) => `<article class="metric"><div class="metric-label">${item.title}${icon(item.icon)}</div><div class="metric-value">${item.value}</div>${item.bar ? `<div class="progress-line" role="progressbar" aria-label="训练进度" aria-valuenow="${progress(run)}" aria-valuemin="0" aria-valuemax="100"><span style="width:${progress(run)}%"></span></div>` : ''}<div class="metric-foot">${item.foot}</div></article>`).join('');
  }

  function getNodes(run) {
    const started = run.step > 0;
    const batch = run.batch;
    return [
      { id: 'dataset', label: '数据集', desc: 'FakeData · 合成图像', metric: '512 samples', icon: 'data', intro: '从本地 FakeData 生成 RGB 图像和 10 类随机标签. 用它验证训练流程, 无需等待外部数据下载. 这些样本不是官方 CIFAR-10.', facts: [['样本数量', '512'], ['图像形状', '[3, 32, 32]'], ['标签范围', '0 ... 9'], ['数据位置', 'data/ (local)']], output: 'image: Tensor[3, 32, 32]\nlabel: int64', samples: true },
      { id: 'loader', label: '加载与预处理', desc: `batch ${batch} · 2 workers`, metric: started ? '8.2 ms / batch' : '等待加载', icon: 'layers', intro: '在 CPU 侧读取样本, 执行 ToTensor 与 Normalize 后, 由 DataLoader 整理为 batch. 预处理在样本进入 GPU 之前完成.', facts: [['batch size', batch], ['worker 数量', '2'], ['预处理', 'ToTensor / Normalize'], ['pinned memory', 'true']], output: `images: float32[${batch}, 3, 32, 32]\nlabels: int64[${batch}]` },
      { id: 'transfer', label: '设备迁移', desc: 'CPU → CUDA:0', metric: started ? '1.6 ms / batch' : '等待迁移', icon: 'swap', intro: 'Trainer 将 batch 中的图像和标签一并迁移到计算设备. 数据形状不变, 存储位置从 CPU 内存进入 GPU 显存.', facts: [['输入设备', 'CPU / pinned'], ['输出设备', 'CUDA:0'], ['图像类型', 'float32'], ['迁移路径', 'Trainer batch transfer']], output: `images.device: cuda:0\nlabels.device: cuda:0\nshape: [${batch}, 3, 32, 32]` },
      { id: 'forward', label: '模型前向', desc: runModel(run), metric: started ? '62.4 ms' : '等待前向', icon: 'chip', intro: 'CoreModel.training_step 调用模型进行前向传播. 分类头将每张图像映射到 10 个类别的未归一化 logits.', facts: [['模型', runModel(run)], ['输入张量', `[${batch}, 3, 32, 32]`], ['输出张量', `[${batch}, 10]`], ['计算设备', 'CUDA:0']], output: `logits = model(images)\nlogits: float32[${batch}, 10]` },
      { id: 'loss', label: '损失计算', desc: 'CrossEntropy', metric: run.loss === null ? '等待计算' : `loss ${run.loss.toFixed(3)}`, icon: 'function', intro: 'CrossEntropy 对比模型 logits 与真实标签, 得到用于反向传播的标量损失. 此设计稿展示的是示例值, 不可用于判断模型质量.', facts: [['输入 logits', `[${batch}, 10]`], ['输入 labels', `[${batch}]`], ['归约方式', 'mean'], ['输出', 'scalar loss']], output: 'loss = criterion(logits, labels)\nself.log("loss", loss, prefix="train")' },
      { id: 'backward', label: '反向传播', desc: 'manual_backward', metric: started ? '108.8 ms' : '等待反向', icon: 'back', intro: 'XDL 使用手动优化. 在 training_step 中显式清零梯度,调用 manual_backward, 再按需执行梯度裁剪.', facts: [['执行入口', 'CoreModel.training_step'], ['反向方法', 'manual_backward(loss)'], ['优化模式', 'manual'], ['梯度目标', 'trainable parameters']], output: 'optimizer.zero_grad()\nself.manual_backward(loss)\nself.clip_gradients(optimizer)' },
      { id: 'optimizer', label: '参数更新', desc: `Adam · lr ${run.lr}`, metric: started ? '5.0 ms / step' : '等待更新', icon: 'settings', intro: 'optimizer.step 应用梯度并更新模型参数. 下一批数据使用新的参数继续前向传播. Callback 在训练边界保存检查点.', facts: [['优化器', 'Adam'], ['学习率', run.lr], ['当前 step', run.step], ['下一步', 'next batch → forward']], output: 'optimizer.step()\n# Callback saves model + optimizer\n# at the configured training boundary.' },
    ];
  }

  function pipelineMarkup(run, expanded) {
    const nodes = getNodes(run);
    return `<section class="card pipeline-card"><div class="card-head"><div class="card-heading">${icon('flow')}<h2>训练数据流</h2></div><div class="pipeline-head-right"><span class="flow-live"><i class="dot"></i>${run.step ? `Batch ${run.step} · 单步示例` : '等待第一个 batch'}</span>${expanded ? '<span class="card-subtitle">点击节点检查细节</span>' : `<button class="text-btn" data-open-pipeline>展开视图${icon('arrow-up')}</button>`}</div></div><div class="pipeline-canvas"><div class="lane-labels"><span>${icon('monitor')} CPU / DATA PIPELINE</span><span>${icon('chip')} GPU 0 / TRAINING STEP</span></div><div class="flow-track">${nodes.map((node, index) => `${index ? '<span class="flow-arrow" aria-hidden="true"><i></i></span>' : ''}<button class="flow-node" data-node="${node.id}" aria-label="查看${node.label}详情" aria-pressed="false"><span class="flow-node-head"><span class="node-icon">${icon(node.icon)}</span><span class="node-number">0${index + 1}</span></span><strong>${node.label}</strong><span class="node-desc">${node.desc}</span><span class="node-metric">${node.metric}</span></button>`).join('')}</div><div class="feedback-row"><span class="flow-caption">${icon('arrow')} images + labels</span><div class="feedback-loop"><span>更新模型权重 · 进入下一个 batch</span></div></div></div><div class="pipeline-footer"><span class="hint">${icon('info')}<span>张量从左向右流动, 点击节点查看输入与输出.</span></span><span>${icon('save')}<span> 检查点: <strong>${run.saved ? `step ${run.saved} 已保存` : '每个 epoch 保存'}</strong></span></span></div></section>`;
  }

  function renderChart() {
    const run = activeRun();
    const svg = $('#trainingChart');
    const isAccuracy = currentMetric === 'accuracy';
    const chartWidth = Math.max(280, svg.getBoundingClientRect().width);
    svg.setAttribute('viewBox', `0 0 ${chartWidth} 218`);
    svg.setAttribute('preserveAspectRatio', 'none');
    $('#chartUnit').textContent = isAccuracy ? '/ 纵轴: accuracy (%)' : '/ 纵轴: loss';
    svg.setAttribute('aria-label', `${isAccuracy ? '准确率' : '损失'}随 step 变化的模拟曲线`);
    $$('[data-metric]').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.metric === currentMetric)));
    $('#chartTooltip').hidden = true;
    if (!run.step) {
      chartData = [];
      svg.innerHTML = `<text x="${chartWidth / 2}" y="93" text-anchor="middle" style="font-size:13px">${run.status === 'failed' ? '训练未开始, 暂无指标' : '等待训练开始'}</text><text x="${chartWidth / 2}" y="118" text-anchor="middle">${run.status === 'failed' ? '请在运行日志中查看错误原因' : '第一个 batch 完成后将在此显示曲线'}</text>`;
      return;
    }
    const left = 42; const right = chartWidth - 13; const top = 18; const bottom = 180;
    const min = isAccuracy ? 0 : 2.25;
    const max = isAccuracy ? 20 : 2.65;
    const y = (value) => bottom - (value - min) / (max - min) * (bottom - top);
    const x = (step) => left + (step / run.step) * (right - left);
    const count = Math.min(55, run.step);
    chartData = Array.from({ length: count + 1 }, (_, index) => {
      const fraction = index / count;
      const step = Math.round(run.step * fraction);
      let train = isAccuracy ? 9.8 + Math.sin(index * 1.2) * .7 + Math.sin(index * .39) * .4 : 2.307 + .273 * Math.exp(-fraction * 7) + Math.sin(index * 1.9) * .009 + Math.cos(index * .6) * .006;
      let val = isAccuracy ? 10.1 + Math.cos(index * .75) * .65 : 2.321 + .255 * Math.exp(-fraction * 6) + Math.sin(index * .67) * .006;
      if (index === count) { train = isAccuracy ? run.accuracy : run.loss; val = isAccuracy ? 10.5 : run.valLoss; }
      return { step, train, val, x: x(step), y: y(train) };
    });
    const path = (key) => chartData.map((point, index) => `${index ? 'L' : 'M'}${point.x.toFixed(2)},${y(point[key]).toFixed(2)}`).join(' ');
    const ticks = isAccuracy ? [0, 5, 10, 15, 20] : [2.25, 2.35, 2.45, 2.55, 2.65];
    const grid = ticks.map((value) => `<line class="gridline" x1="${left}" x2="${right}" y1="${y(value)}" y2="${y(value)}"/><text x="${left - 9}" y="${y(value) + 3}" text-anchor="end">${isAccuracy ? `${value}%` : value.toFixed(2)}</text>`).join('');
    const axis = Array.from({ length: 6 }, (_, index) => { const value = Math.round(run.step * index / 5); return `<text x="${x(value)}" y="202" text-anchor="middle">${value}</text>`; }).join('');
    const line = path('train');
    const end = chartData[chartData.length - 1];
    svg.innerHTML = `<defs><linearGradient id="chartFill" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#3c5df5" stop-opacity=".09"/><stop offset="1" stop-color="#3c5df5" stop-opacity="0"/></linearGradient></defs>${grid}${axis}<path d="${line} L${right},${bottom} L${left},${bottom}Z" fill="url(#chartFill)"/><path class="chart-line val" d="${path('val')}"/><path class="chart-line" d="${line}"/><circle cx="${end.x}" cy="${end.y}" r="3.5" fill="#3c5df5" stroke="white" stroke-width="2"/><line id="chartCrosshair" class="chart-crosshair" x1="0" x2="0" y1="${top}" y2="${bottom}" visibility="hidden"/><circle id="chartHoverDot" r="4" fill="#3c5df5" stroke="white" stroke-width="2" visibility="hidden"/>`;
  }

  function chartHover(event) {
    if (!chartData.length) return;
    const svg = $('#trainingChart');
    const point = svg.createSVGPoint();
    point.x = event.clientX; point.y = event.clientY;
    const transformed = point.matrixTransform(svg.getScreenCTM().inverse());
    const nearest = chartData.reduce((best, item) => Math.abs(item.x - transformed.x) < Math.abs(best.x - transformed.x) ? item : best);
    const line = $('#chartCrosshair'); const dot = $('#chartHoverDot');
    line.setAttribute('x1', nearest.x); line.setAttribute('x2', nearest.x); line.setAttribute('visibility', 'visible');
    dot.setAttribute('cx', nearest.x); dot.setAttribute('cy', nearest.y); dot.setAttribute('visibility', 'visible');
    const tooltip = $('#chartTooltip');
    const format = (value) => currentMetric === 'accuracy' ? `${value.toFixed(1)}%` : value.toFixed(3);
    tooltip.innerHTML = `Step ${nearest.step}<br>train ${format(nearest.train)}<br>val&nbsp;&nbsp; ${format(nearest.val)}`;
    tooltip.hidden = false;
    const surface = $('#chartSurface').getBoundingClientRect();
    tooltip.style.left = `${Math.max(5, Math.min(event.clientX - surface.left + 12, surface.width - 135))}px`;
  }

  function renderActivity(run) {
    const activities = run.logs.slice(-3).reverse();
    $('#activityList').innerHTML = activities.map((entry) => `<div class="activity-row"><span class="activity-icon">${icon(entry.message.includes('checkpoint') ? 'save' : entry.level === 'WARN' ? 'info' : 'chart')}</span><div><strong>${entry.message.includes('checkpoint') ? '检查点已保存' : entry.message.includes('step=') ? '训练指标已记录' : entry.level === 'WARN' ? '使用本地合成数据' : '实验状态更新'}</strong><small>${escape(entry.message.length > 72 ? `${entry.message.slice(0, 72)}...` : entry.message)}</small></div><time>${entry.at}</time></div>`).join('');
  }

  function renderCheckpoint(run) {
    $('#checkpointPreview').innerHTML = run.saved ? `<div class="file-preview">${icon('file')}<div><strong>${checkpointName(run)}</strong><small>model + optimizer · step ${run.saved}</small></div><span class="status">已保存</span></div><div class="checkpoint-note"><span>每个 epoch 保存 · 模拟产物</span><button class="text-btn" data-resume-run="${run.id}">从此处恢复${icon('arrow')}</button></div>` : `<div class="file-preview">${icon('save')}<div><strong>尚无检查点</strong><small>训练开始后可手动保存或按 epoch 保存.</small></div></div><div class="checkpoint-note"><span>保存模型和优化器状态</span><span>等待训练</span></div>`;
  }

  function configText(run) {
    return `# XDL design preview. Simulated configuration.\n# This is not an executable training manifest.\nexperiment: ${JSON.stringify(run.name)}\nmodel: ${runModel(run)}\ndataset: FakeData\nsamples: 512\nimage_shape: [3, 32, 32]\nnum_classes: 10\nbatch_size: ${run.batch}\nepochs: ${run.epochs}\noptimizer: Adam\nlearning_rate: ${run.lr}\nlauncher: ${run.launcher}\ndevice: cuda:0\n${run.parentId ? `parent_run_id: ${run.parentId}\n` : ''}design_preview: true\n`;
  }

  function renderConfig(run) {
    const section = (heading, rows) => `<section class="card"><div class="card-head"><h2>${heading}</h2></div><dl class="config-body">${rows.map(([label, value]) => `<div class="config-row"><dt>${label}</dt><dd>${escape(value)}</dd></div>`).join('')}</dl></section>`;
    const snapshot = run.workflowSnapshot;
    const workflowSection = snapshot ? `<section class="card" style="grid-column:1/-1"><div class="card-head"><div><h2>固定节点工作流快照</h2><p class="card-subtitle">此模拟实验保留了排队时的参数与已应用代码.</p></div><button class="btn" id="downloadConfig">${icon('download')}导出 YAML + 代码 JSON</button></div><div class="config-grid">${section('数据与模型', [['数据集', snapshot.values.dataset], ['类别数', snapshot.values.classes], ['模型', snapshot.values.model], ['输出类别', snapshot.values.modelClasses], ['损失函数', snapshot.values.loss]])}${section('优化与训练', [['优化器', snapshot.values.optimizer], ['学习率', snapshot.values.lr], ['Epochs', snapshot.values.epochs], ['Batch size', snapshot.values.batch], ['精度 / 设备', `${snapshot.values.precision} / ${snapshot.values.device}`]])}</div><pre class="config-code">${escape(snapshot.yaml)}\n# 局部代码: ${snapshot.nodes.filter((node) => node.code_applied).length} applied / ${snapshot.nodes.filter((node) => node.code_dirty).length} draft\n# ${snapshot.code_execution}</pre></section>` : '';
    $('#panel-config').innerHTML = `<div class="view-heading"><div><h2>实验配置</h2><p>当前实验的模拟参数快照.</p></div>${snapshot ? '' : `<button class="btn" id="downloadConfig">${icon('download')}导出配置</button>`}</div>${workflowSection}<div class="config-grid">${section('模型与数据', [['模型', runModel(run)], ['数据集', 'FakeData (非官方 CIFAR-10)'], ['训练样本', 512], ['输入形状', '[3, 32, 32]'], ['类别数', 10]])}${section('训练与优化', [['Epochs', run.epochs], ['Batch size', run.batch], ['优化器', snapshot?.values.optimizer || 'Adam'], ['学习率', run.lr], ['启动方式', run.launcher]])}${section('运行信息', [['实验 ID', run.id], ['状态', labels[run.status]], ['父实验', run.parentId || '无'], ['计算设备', snapshot?.values.device || 'GPU 0 / CUDA:0'], ['数据来源', 'design preview']])}${section('保存策略', [['检查点周期', snapshot?.values.saveEvery || '每个 epoch'], ['最近保存 step', run.saved || '尚未保存'], ['保存内容', 'model + optimizer'], ['恢复方式', '创建新实验'], ['产物类型', '模拟清单']])}</div>`;
  }

  function renderLogs() {
    const query = $('#logSearch').value.trim().toLowerCase();
    const level = $('#logLevel').value;
    const logs = activeRun().logs.filter((entry) => (level === 'all' || entry.level === level) && `${entry.message} ${entry.at} ${entry.level}`.toLowerCase().includes(query));
    $('#logsContainer').innerHTML = logs.length ? logs.map((entry) => `<div class="log-line ${entry.level === 'WARN' ? 'warning' : ''}"><span class="log-time">${entry.at}</span><span class="log-tag">${entry.level}</span><span class="log-copy">${escape(entry.message)}</span></div>`).join('') : `<div class="empty">${icon('search')}<h3>没有匹配的日志</h3><p>尝试其他关键词或选择所有级别.</p></div>`;
  }

  function renderExperiments() {
    const query = $('#runSearch').value.trim().toLowerCase();
    const status = $('#statusFilter').value;
    const filtered = runs.filter((run) => `${run.name} ${run.id} ${runModel(run)}`.toLowerCase().includes(query) && (status === 'all' || run.status === status));
    if (!filtered.length) {
      $('#experimentTable').innerHTML = `<div class="empty">${icon('search')}<h3>没有找到实验</h3><p>换一个关键词, 或清除当前筛选条件.</p><button class="btn small" id="clearRunSearch" style="margin-top:15px">清除筛选</button></div>`;
      return;
    }
    $('#experimentTable').innerHTML = `<table><thead><tr><th>实验名称</th><th>状态</th><th>模型</th><th>训练进度</th><th>训练损失</th><th>设备</th><th></th></tr></thead><tbody>${filtered.map((run) => `<tr><td><button class="table-name" data-select-run="${run.id}">${escape(run.name)}</button><div class="table-sub mono">${run.id}</div></td><td>${badge(run.status)}</td><td>${runModel(run)}</td><td>Epoch ${epochLabel(run)}</td><td class="mono">${run.loss === null ? '-' : run.loss.toFixed(3)}</td><td>GPU 0</td><td><button class="icon-button" data-select-run="${run.id}" aria-label="查看 ${escape(run.name)}">${icon('arrow')}</button></td></tr>`).join('')}</tbody></table>`;
  }

  function renderTemplates() {
    $('#templateGrid').innerHTML = Object.entries(templates).map(([id, template]) => `<article class="card template-card"><div class="template-symbol">${icon(template.icon)}</div><h3>${template.title}</h3><p>${template.description}</p><div class="template-meta"><span>${template.meta}</span><button class="text-btn" data-use-template="${id}">使用模板${icon('arrow')}</button></div></article>`).join('');
  }

  function renderArtifacts() {
    const select = $('#artifactRun');
    const previous = select.value || selectedId;
    select.innerHTML = runs.map((run) => `<option value="${run.id}" ${run.id === previous ? 'selected' : ''}>${escape(run.name)}</option>`).join('');
    const run = runs.find((item) => item.id === select.value) || activeRun();
    const artifacts = [{ title: 'resolved-config.yaml', kind: '配置快照', at: '创建时', type: 'config' }];
    if (run.step) artifacts.push({ title: 'metrics-preview.json', kind: '模拟指标', at: `step ${run.step}`, type: 'metrics' });
    if (run.saved) artifacts.unshift({ title: `${checkpointName(run)}.manifest.json`, kind: '检查点示例清单', at: `step ${run.saved}`, type: 'manifest' });
    $('#artifactTable').innerHTML = `<table><thead><tr><th>产物名称</th><th>类型</th><th>保存位置</th><th>操作</th></tr></thead><tbody>${artifacts.map((artifact) => `<tr><td><div class="table-name">${artifact.title}</div><div class="table-sub">${escape(run.name)}</div></td><td>${artifact.kind}</td><td class="mono">${artifact.at}</td><td><button class="text-btn" data-download-artifact="${artifact.type}" data-artifact-run="${run.id}">${icon('download')}下载${artifact.type === 'manifest' ? '清单' : '预览'}</button>${artifact.type === 'manifest' ? ` <button class="text-btn" data-resume-run="${run.id}" style="margin-left:12px">从此处恢复${icon('arrow')}</button>` : ''}</td></tr>`).join('')}</tbody></table>`;
  }

  function renderResourceQueue() {
    const allocated = runs.filter((run) => ['running', 'queued', 'paused'].includes(run.status));
    $('#resourceQueue').innerHTML = allocated.length ? allocated.map((run) => `<div class="activity-row"><span class="activity-icon">${icon(run.status === 'queued' ? 'clock' : 'chip')}</span><div><strong>${escape(run.name)}</strong><small>${runModel(run)} · GPU 0</small></div><span style="margin-left:auto">${badge(run.status)}</span><button class="text-btn" data-select-run="${run.id}">查看${icon('arrow')}</button></div>`).join('') : `<div class="empty"><h3>队列为空</h3><p>可以新建实验体验 GPU 分配流程.</p></div>`;
  }

  function openNewRun(templateId = 'vgg', parentId = null) {
    const template = templates[templateId] || templates.vgg;
    resumeParent = parentId;
    $('#newTemplate').value = templateId;
    $('#newName').value = parentId ? `${runs.find((run) => run.id === parentId).name}-resume` : `${templateId === 'vgg' ? 'vgg11' : templateId === 'resnet' ? 'resnet18' : 'mlp'}-experiment-${String(runs.length + 1).padStart(2, '0')}`;
    $('#newEpochs').value = template.epochs;
    $('#newBatch').value = template.batch;
    $('#newLr').value = template.lr;
    $('#newRunTitle').textContent = parentId ? '从检查点恢复实验' : '新建训练实验';
    if (parentId) {
      const parent = runs.find((run) => run.id === parentId);
      $('#newEpochs').value = parent.epochs;
      $('#newBatch').value = parent.batch;
      $('#newLr').value = parent.lr;
    }
    $('#newRunDialog').showModal();
    $('#newName').focus();
    $('#newName').select();
  }

  function showNode(id) {
    const node = getNodes(activeRun()).find((item) => item.id === id);
    if (!node) return;
    $$('[data-node]').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.node === id)));
    $('#nodeTitle').textContent = node.label;
    $('#nodeSubtitle').textContent = `${node.desc} · 模拟 batch ${activeRun().step || '-'}`;
    $('#nodeDetail').innerHTML = `<p class="detail-intro">${node.intro}</p><div class="detail-grid">${node.facts.map(([label, value]) => `<div class="detail-tile"><small>${label}</small><strong>${escape(value)}</strong></div>`).join('')}</div>${node.samples ? '<div class="detail-label">合成样本预览 / 程序生成的随机像素</div><div class="sample-grid" id="sampleGrid"></div>' : ''}<div class="detail-label">${node.id === 'dataset' || node.id === 'loader' || node.id === 'transfer' ? '输出张量' : '执行摘要'}</div><pre class="config-code">${escape(node.output)}</pre><p class="card-subtitle" style="margin-top:12px">耗时与数值均为设计示例, 未连接实际训练.</p>`;
    if (node.samples) renderSamples();
    $('#nodeDialog').showModal();
  }

  function renderSamples() {
    const grid = $('#sampleGrid');
    for (let sample = 0; sample < 8; sample++) {
      const canvas = document.createElement('canvas');
      canvas.width = 32; canvas.height = 32;
      canvas.setAttribute('aria-label', `合成随机样本 ${sample + 1}`);
      canvas.setAttribute('role', 'img');
      const context = canvas.getContext('2d');
      const image = context.createImageData(32, 32);
      let seed = 719 + sample * 127;
      for (let pixel = 0; pixel < image.data.length; pixel += 4) {
        for (let channel = 0; channel < 3; channel++) {
          seed = (seed * 16807) % 2147483647;
          image.data[pixel + channel] = seed % 256;
        }
        image.data[pixel + 3] = 255;
      }
      context.putImageData(image, 0, 0);
      grid.append(canvas);
    }
  }

  document.addEventListener('click', (event) => {
    const target = event.target.closest('button, a.brand');
    if (!target) return;
    if (target.matches('[data-page]')) setPage(target.dataset.page);
    if (target.matches('a.brand')) { event.preventDefault(); setPage('overview'); }
    if (target.matches('[data-select-run]')) selectRun(target.dataset.selectRun);
    if (target.matches('[data-tab]')) setTab(target.dataset.tab);
    if (target.matches('[data-metric]')) { currentMetric = target.dataset.metric; renderChart(); }
    if (target.matches('[data-node]')) showNode(target.dataset.node);
    if (target.matches('[data-new-run]')) openNewRun();
    if (target.matches('[data-use-template]')) openNewRun(target.dataset.useTemplate);
    if (target.matches('[data-resume-run]')) {
      const parent = runs.find((run) => run.id === target.dataset.resumeRun);
      openNewRun(parent.template, parent.id);
    }
    if (target.matches('[data-close-dialog]')) target.closest('dialog').close();
    if (target.matches('[data-open-pipeline]')) setTab('pipeline');
    if (target.matches('[data-go-logs]')) setTab('logs');
    if (target.matches('[data-go-artifacts]')) { $('#artifactRun').value = selectedId; setPage('artifacts'); }
    if (target.matches('[data-go-resources]')) setPage('resources');
    if (target.matches('[data-go-experiments]')) setPage('experiments');
    if (target.id === 'clearRunSearch') { $('#runSearch').value = ''; $('#statusFilter').value = 'all'; renderExperiments(); }
    if (target.id === 'downloadConfig') {
      const run = activeRun();
      if (run.workflowSnapshot) download(`${run.id}-workflow-preview.json`, JSON.stringify(run.workflowSnapshot, null, 2));
      else download(`${run.id}-preview.yaml`, configText(run), 'text/yaml');
    }
    if (target.matches('[data-download-artifact]')) {
      const run = runs.find((item) => item.id === target.dataset.artifactRun);
      const type = target.dataset.downloadArtifact;
      if (type === 'config') download(`${run.id}-preview.yaml`, configText(run), 'text/yaml');
      else download(`${run.id}-${type}-preview.json`, JSON.stringify({ design_preview: true, run_id: run.id, type, model: runModel(run), step: type === 'manifest' ? run.saved : run.step, note: type === 'manifest' ? 'This is a simulated manifest, not a model checkpoint.' : 'Illustrative metrics, not a real training result.', ...(type === 'metrics' ? { loss: run.loss, accuracy: run.accuracy } : {}) }, null, 2));
    }
  });

  $('#pauseButton').addEventListener('click', () => {
    const run = activeRun();
    if (!canControl(run)) return;
    if (run.status === 'running') { run.status = 'paused'; run.gpu = 0; }
    else {
      if (runs.some((other) => other.id !== run.id && other.status === 'running')) { showToast('GPU 0 已有实验运行, 请先暂停或停止当前运行的实验.'); return; }
      run.status = 'running'; run.gpu = 74;
    }
    addLog(run, `模拟控制: ${run.status === 'paused' ? 'pause acknowledged / 已暂停' : 'resume acknowledged / 已继续'}`);
    renderRun();
    showToast(run.status === 'paused' ? '模拟训练已暂停, 可以继续或保存检查点.' : '模拟训练已继续.');
  });
  $('#checkpointButton').addEventListener('click', () => {
    const run = activeRun();
    if (!canControl(run)) return;
    run.saved = run.step;
    addLog(run, `checkpoint_saved / step=${run.step} / simulated model + optimizer snapshot`);
    renderRun();
    showToast(`检查点已保存到 step ${run.step}. 可在模型产物中查看示例清单.`);
  });
  $('#stopButton').addEventListener('click', () => {
    if (!['running', 'paused', 'queued'].includes(activeRun().status)) return;
    const queued = activeRun().status === 'queued';
    $('#stopTitle').textContent = queued ? '取消这个排队实验?' : '停止当前实验?';
    $('#confirmStop').textContent = queued ? '取消排队' : '停止实验';
    $('#stopDialog .confirm-copy').textContent = queued ? '取消后, 实验将从等待队列中移出. 这只会更新设计预览中的模拟记录.' : '实验将停止推进, 已保存的检查点仍会保留. 可以从检查点创建一个新实验继续训练.';
    $('#stopDialog').showModal();
  });
  $('#confirmStop').addEventListener('click', () => {
    const run = activeRun();
    run.status = 'stopped'; run.gpu = 0;
    addLog(run, '模拟控制: stop acknowledged / 实验已停止');
    $('#stopDialog').close();
    renderRun();
    showToast('模拟实验已停止. 已保存的检查点仍然保留.');
  });
  $('#allExperiments').addEventListener('click', () => setPage('experiments'));
  $('#helpButton').addEventListener('click', () => $('#helpDialog').showModal());
  $('#menuToggle').addEventListener('click', () => { const open = $('#sidebar').classList.toggle('open'); $('#menuToggle').setAttribute('aria-expanded', String(open)); });
  $('#newTemplate').addEventListener('change', () => {
    const template = templates[$('#newTemplate').value];
    $('#newEpochs').value = template.epochs; $('#newBatch').value = template.batch; $('#newLr').value = template.lr;
    resumeParent = null;
    $('#newRunTitle').textContent = '新建训练实验';
  });
  $('#newRunForm').addEventListener('submit', (event) => {
    event.preventDefault();
    if (!event.currentTarget.reportValidity()) return;
    const name = $('#newName').value.trim();
    if (!name) return;
    const run = { id: `demo-${Date.now().toString(36)}`, name, template: $('#newTemplate').value, status: 'queued', epochs: Number($('#newEpochs').value), step: 0, batch: Number($('#newBatch').value), lr: Number($('#newLr').value), launcher: $('#newLauncher').value, loss: null, valLoss: null, accuracy: null, elapsed: '00:00', gpu: 0, saved: 0, parentId: resumeParent };
    run.logs = initialLogs(run);
    if (resumeParent) addLog(run, `resume_parent=${resumeParent} / a new simulated run was created`);
    runs.unshift(run);
    $('#newRunDialog').close();
    selectedId = run.id;
    currentTab = 'overview';
    setPage('overview');
    renderRecent();
    showToast(`${run.name} 已加入模拟队列, 等待 GPU 0.`);
  });
  $('#runSearch').addEventListener('input', renderExperiments);
  $('#statusFilter').addEventListener('change', renderExperiments);
  $('#logSearch').addEventListener('input', renderLogs);
  $('#logLevel').addEventListener('change', renderLogs);
  $('#artifactRun').addEventListener('change', renderArtifacts);
  $('#exportRuns').addEventListener('click', () => download('xdl-experiments-preview.json', JSON.stringify({ design_preview: true, runs: runs.map(({ logs, ...run }) => run) }, null, 2)));
  $('#exportLogs').addEventListener('click', () => download(`${activeRun().id}-preview.log`, activeRun().logs.map((entry) => `${entry.at} ${entry.level} ${entry.message}`).join('\n'), 'text/plain'));
  $('#chartSurface').addEventListener('pointermove', chartHover);
  $('#chartSurface').addEventListener('pointerleave', () => { $('#chartTooltip').hidden = true; $('#chartCrosshair')?.setAttribute('visibility', 'hidden'); $('#chartHoverDot')?.setAttribute('visibility', 'hidden'); });
  $('.tabbar').addEventListener('keydown', (event) => {
    const tabs = $$('[data-tab]');
    const current = tabs.indexOf(document.activeElement);
    if (current < 0) return;
    let next;
    if (event.key === 'ArrowRight') next = (current + 1) % tabs.length;
    if (event.key === 'ArrowLeft') next = (current - 1 + tabs.length) % tabs.length;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = tabs.length - 1;
    if (next !== undefined) { event.preventDefault(); setTab(tabs[next].dataset.tab, true); }
  });
  $$('dialog').forEach((dialog) => dialog.addEventListener('click', (event) => {
    if (event.target !== dialog) return;
    const bounds = dialog.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
  }));
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') { $('#sidebar').classList.remove('open'); $('#menuToggle').setAttribute('aria-expanded', 'false'); }
  });
  $$('[data-page]').forEach((button) => button.setAttribute('aria-label', button.querySelector('span').textContent));
  let resizeFrame;
  new ResizeObserver(() => {
    cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(() => { if (currentPage === 'overview' && currentTab === 'overview') renderChart(); });
  }).observe($('#chartSurface'));
  window.addEventListener('hashchange', () => setPage(location.hash.slice(1)));
  document.addEventListener('xdl:workflow-message', (event) => showToast(String(event.detail)));
  document.addEventListener('xdl:workflow-queue', (event) => {
    const workflow = event.detail;
    const values = workflow.values;
    const source = { FakeData: 'Synthetic image data', CIFAR10: 'CIFAR-10', CIFAR100: 'CIFAR-100', ImageFolder: 'ImageFolder' }[values.dataset];
    const model = { vgg11_bn: 'VGG-11 BN', vgg16_bn: 'VGG-16 BN', resnet18: 'ResNet-18', resnet34: 'ResNet-34' }[values.model] || values.model;
    const now = new Date();
    const run = {
      id: `workflow-${now.getTime().toString(36)}`, name: values.name, template: 'vgg', status: 'queued',
      epochs: values.epochs, step: 0, batch: values.batch, lr: values.lr, launcher: 'single',
      loss: null, valLoss: null, accuracy: null, elapsed: '00:00', gpu: 0, saved: 0,
      logs: [{ at: time(), level: 'INFO', message: `workflow_preview=${workflow.version} / model=${model} / dataset=${source} / code_edits=${workflow.nodes.filter((node) => node.code_applied).length} / code_execution=disabled` }],
      workflowSnapshot: workflow,
    };
    runs.unshift(run);
    selectedId = run.id;
    currentTab = 'config';
    setPage('overview');
    showToast(`${run.name} 已加入模拟队列. 工作流配置与局部代码已保存在实验记录中; Python 未执行.`);
  });
  renderRecent();
  renderTemplates();
  renderArtifacts();
  const initialPage = location.hash.replace('#', '');
  setPage(['overview', 'experiments', 'templates', 'artifacts', 'resources', 'workflow'].includes(initialPage) ? initialPage : 'overview');
})();
