// app.js — TorusFold SPA controller
//
// Wires the sequence input, Predict button, job polling, result fetch,
// Mol* mounting (via CircRNAViewer), fingerprint panel, and download buttons
// to the /api/* endpoints.

(function () {
  'use strict';

  const seqArea = document.getElementById('sequence');
  const seqCounter = document.getElementById('seq-counter');
  const seqError = document.getElementById('seq-error');
  const predictBtn = document.getElementById('predict-btn');
  const progressCard = document.getElementById('progress-card');
  const progressEl = document.getElementById('progress');
  const methodTag = document.getElementById('method-tag');
  const resultCard = document.getElementById('result-card');
  const dlPdb = document.getElementById('dl-pdb');
  const dlJson = document.getElementById('dl-json');
  const metaSummary = document.getElementById('meta-summary');
  const seqCard = document.getElementById('sequence-card');
  const seqBox = document.getElementById('seq-box');
  const serverStatus = document.getElementById('server-status');

  const methodFoot = document.getElementById('method-foot');
  const closureFoot = document.getElementById('closure-foot');
  const elapsedFoot = document.getElementById('elapsed-foot');

  let viewer = null;
  let currentJobId = null;
  let lastResult = null;
  let pollTimer = null;

  // ----- sequence input -----
  seqArea.addEventListener('input', () => {
    const cleaned = (seqArea.value || '').replace(/\s+/g, '').toUpperCase();
    const len = cleaned.length;
    seqCounter.textContent = len + ' nt';
    const bad = cleaned.replace(/[ACGUTN]/g, '');
    if (bad) {
      seqError.textContent = '非法字符: ' + bad.split('').slice(0, 10).join(',');
      predictBtn.disabled = true;
    } else if (len === 0) {
      seqError.textContent = '';
      predictBtn.disabled = true;
    } else if (len < 10) {
      seqError.textContent = `过短 (${len} < 10)`;
      predictBtn.disabled = true;
    } else if (len > 500) {
      seqError.textContent = `过长 (${len} > 500)`;
      predictBtn.disabled = true;
    } else {
      seqError.textContent = '';
      predictBtn.disabled = false;
    }
  });

  // ----- health probe on load -----
  async function probeHealth() {
    try {
      const r = await fetch('/api/health');
      if (!r.ok) throw new Error('health ' + r.status);
      const h = await r.json();
      serverStatus.textContent =
        `backend: ${h.backend}  |  device: ${h.device}` +
        (h.weights_loaded ? `  |  weights: ✓` : '');
    } catch (e) {
      serverStatus.textContent = 'server unreachable';
      serverStatus.style.color = '#ff6b6b';
    }
  }
  probeHealth();

  // ----- representation / opacity controls -----
  const reprSelect = document.getElementById('repr-select');
  const opacitySlider = document.getElementById('opacity-slider');
  const opacityVal = document.getElementById('opacity-val');
  if (reprSelect) {
    reprSelect.addEventListener('change', () => {
      if (viewer && viewer.setRepresentation)
        viewer.setRepresentation(reprSelect.value);
    });
  }
  if (opacitySlider) {
    opacitySlider.addEventListener('input', () => {
      const v = parseFloat(opacitySlider.value);
      if (opacityVal) opacityVal.textContent = v.toFixed(2);
      if (viewer && viewer.setSurfaceOpacity) viewer.setSurfaceOpacity(v);
    });
  }

  // ----- predict -----
  predictBtn.addEventListener('click', async () => {
    predictBtn.disabled = true;
    const seq = (seqArea.value || '').replace(/\s+/g, '').toUpperCase().replace(/T/g, 'U');

    resultCard.style.display = 'none';
    progressCard.style.display = 'block';
    progressEl.textContent = '提交中...';
    progressEl.classList.add('loading');
    methodTag.textContent = '';
    if (viewer && viewer.setStatus) viewer.setStatus('');

    try {
      const r = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sequence: seq }),
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({ detail: r.statusText }));
        throw new Error(err.detail || 'predict failed');
      }
      const { job_id } = await r.json();
      currentJobId = job_id;
      startPolling(job_id);
    } catch (e) {
      progressEl.textContent = '提交失败: ' + e.message;
      progressEl.classList.remove('loading');
      predictBtn.disabled = false;
    }
  });

  function startPolling(jobId) {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(() => pollJob(jobId), 500);
    pollJob(jobId);
  }

  async function pollJob(jobId) {
    try {
      const r = await fetch('/api/jobs/' + jobId);
      if (!r.ok) throw new Error('job fetch ' + r.status);
      const s = await r.json();
      progressEl.textContent =
        s.status === 'pending' ? '排队中...'
        : s.status === 'running' ? '预测中（可能需联网 AF3）...'
        : s.status;
      if (s.method) methodTag.textContent = 'method: ' + s.method;
      if (s.status === 'done') {
        clearInterval(pollTimer);
        pollTimer = null;
        progressEl.classList.remove('loading');
        await fetchResult(jobId);
      } else if (s.status === 'error') {
        clearInterval(pollTimer);
        pollTimer = null;
        progressEl.classList.remove('loading');
        progressEl.textContent = '错误: ' + (s.error || 'unknown');
        predictBtn.disabled = false;
      }
    } catch (e) {
      progressEl.textContent = '轮询失败: ' + e.message;
    }
  }

  async function fetchResult(jobId) {
    try {
      const r = await fetch('/api/result/' + jobId);
      if (!r.ok) throw new Error('result fetch ' + r.status);
      const res = await r.json();
      lastResult = res;

      // Render fingerprint + activate download buttons.
      renderResult(res);
      resultCard.style.display = 'block';
      seqCard.style.display = 'block';
      seqBox.textContent = res.metadata && res.metadata.sequence
        ? res.metadata.sequence
        : (lastResult.fingerprint ? JSON.parse(lastResult.fingerprint).sequence : '');

      // Mount Mol*. The fingerprint JSON is embedded in lastResult.fingerprint.
      const fp = JSON.parse(res.fingerprint);
      progressEl.textContent = '结构已就绪';
      if (methodFoot) methodFoot.textContent = res.method;
      if (closureFoot)
        closureFoot.textContent = res.metadata && res.metadata.closure_error != null
          ? res.metadata.closure_error.toFixed(3) + ' Å'
          : '—';
      if (elapsedFoot)
        elapsedFoot.textContent = res.metadata && res.metadata.elapsed_s != null
          ? res.metadata.elapsed_s + ' s'
          : '—';

      try {
        if (!viewer) viewer = new CircRNAViewer('viewer');
        await viewer.mount(res.pdb, fp);
      } catch (e) {
        console.error('Mol* mount failed:', e);
        progressEl.textContent = 'Mol* 初始化失败: ' + e.message;
      }

      predictBtn.disabled = false;
    } catch (e) {
      progressEl.textContent = '取结果失败: ' + e.message;
      predictBtn.disabled = false;
    }
  }

  function renderResult(res) {
    // Activate download buttons with current job id.
    dlPdb.disabled = false;
    dlPdb.onclick = () => (window.location.href = '/api/result/' + currentJobId + '/download?format=pdb');
    dlJson.disabled = false;
    dlJson.onclick = () => (window.location.href = '/api/result/' + currentJobId + '/download?format=json');

    // Summary metadata.
    const lines = [];
    if (res.metadata) {
      if (res.metadata.backend) lines.push('backend: ' + res.metadata.backend);
      if (res.metadata.fallback_reason)
        lines.push('fallback: ' + res.metadata.fallback_reason);
    }
    metaSummary.innerHTML = lines.map((l) => `<div>${l}</div>`).join('');
  }
})();
