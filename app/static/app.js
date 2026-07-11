/* ============================================================
   BIFROST — Frontend Application
   Pure vanilla JS, zero dependencies
   ============================================================ */

(() => {
  'use strict';

  // ── State ─────────────────────────────────────────────────
  // Change this to your backend URL (e.g. 'http://203.0.113.5:8000') if hosted separately
  const API_BASE = ''; // Use relative paths since frontend and backend share the same server
  const state = {
    messages: [],
    stats: null,
    models: [],
    sending: false,
    compareMode: false,
    messageCount: 0,
    totalTokensUsed: 0,
    naiveTokensEstimate: 0,
  };

  // Naive multiplier — cloud model uses ~2.5x tokens vs local routing
  const NAIVE_MULTIPLIER = 2.6;

  // ── DOM refs ──────────────────────────────────────────────
  const $ = (sel) => document.querySelector(sel);
  const chatMessages   = $('#chatMessages');
  const chatInput      = $('#chatInput');
  const sendBtn        = $('#sendBtn');
  const compareToggle  = $('#compareToggle');
  const compareOverlay = $('#compareOverlay');
  const compareGrid    = $('#compareGrid');
  const compareQuery   = $('#compareQuery');

  // ── Init ──────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', () => {
    setupInput();
    fetchModels();
    fetchStats();
  });

  // ── Input handling ────────────────────────────────────────
  function setupInput() {
    chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    });

    chatInput.addEventListener('input', () => {
      chatInput.style.height = 'auto';
      chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
    });

    compareToggle.addEventListener('change', () => {
      state.compareMode = compareToggle.checked;
    });
  }

  // ── Send handler ──────────────────────────────────────────
  window.handleSend = async function handleSend() {
    const text = chatInput.value.trim();
    if (!text || state.sending) return;

    if (state.compareMode) {
      sendComparison(text);
    } else {
      sendMessage(text);
    }
  };

  // ── Hint chips ────────────────────────────────────────────
  window.useHint = function useHint(el) {
    const raw = el.textContent.replace(/^"|"$/g, '');
    chatInput.value = raw;
    chatInput.dispatchEvent(new Event('input'));
    chatInput.focus();
  };

  // ── Chat: send message ───────────────────────────────────
  async function sendMessage(text) {
    state.sending = true;
    sendBtn.disabled = true;
    chatInput.value = '';
    chatInput.style.height = 'auto';

    clearWelcome();
    appendUserMsg(text);
    const typingEl = appendTyping();

    try {
      const res = await fetch(API_BASE + '/v1/chat', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Bypass-Tunnel-Reminder': 'true'
        },
        body: JSON.stringify({ message: text }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      removeTyping(typingEl);
      appendBotMsg(data);

      // Update savings — only LOCAL routing saves tokens vs naive (always-remote) approach
      const tokens = data.total_tokens || 0;
      state.totalTokensUsed += tokens;
      if ((data.tier || data.routed_to) === 'LOCAL') {
        // LOCAL call: naive approach would have used remote model (more expensive/verbose)
        state.naiveTokensEstimate += Math.round(tokens * NAIVE_MULTIPLIER);
      } else {
        // REMOTE call: no savings, naive estimate equals actual
        state.naiveTokensEstimate += tokens;
      }
      updateSavings();

      state.messageCount++;
      updateChatCount();
      fetchStats();
    } catch (err) {
      removeTyping(typingEl);
      appendError(err.message);
    } finally {
      state.sending = false;
      sendBtn.disabled = false;
      chatInput.focus();
    }
  }

  // ── Compare mode ──────────────────────────────────────────
  async function sendComparison(text) {
    state.sending = true;
    sendBtn.disabled = true;
    chatInput.value = '';
    chatInput.style.height = 'auto';

    compareQuery.textContent = `"${text}"`;
    compareGrid.innerHTML = `
      <div style="grid-column:1/-1;display:flex;justify-content:center;padding:60px 0;">
        <div class="typing-indicator"><span></span><span></span><span></span></div>
      </div>`;
    compareOverlay.classList.add('active');

    try {
      const res = await fetch(API_BASE + '/v1/compare', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Bypass-Tunnel-Reminder': 'true'
        },
        body: JSON.stringify({ message: text }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      renderCompare(data);
      fetchStats();
    } catch (err) {
      compareGrid.innerHTML = `<div class="msg-error" style="grid-column:1/-1;">
        <span class="msg-error-icon">${iconError()}</span>
        <span class="msg-error-text">${escHtml(err.message)}</span>
      </div>`;
    } finally {
      state.sending = false;
      sendBtn.disabled = false;
    }
  }

  window.closeCompare = () => compareOverlay.classList.remove('active');

  function renderCompare(data) {
    const { category, confidence, recommended_tier, results } = data;
    compareGrid.innerHTML = '';

    (results || []).forEach((r, i) => {
      const card = document.createElement('div');
      const tierLabel = r.tier || r.routed_to || 'UNKNOWN';
      const tierClass = tierLabel === 'LOCAL' ? 'LOCAL' : 'REMOTE';
      const isRecommended = tierLabel === recommended_tier;
      card.className = `compare-card tier-${tierClass} ${isRecommended ? 'recommended' : ''}`;
      card.style.animationDelay = `${i * 0.1}s`;

      if (r.error) {
        card.innerHTML = `
          <div class="compare-card-tier">${escHtml(tierLabel)}</div>
          <div class="compare-card-model">${escHtml(r.model_used || '—')}</div>
          <div class="compare-card-error">${escHtml(r.error)}</div>`;
      } else {
        card.innerHTML = `
          <div class="compare-card-tier">${escHtml(tierLabel)}</div>
          <div class="compare-card-model">${escHtml(r.model_used || '—')}</div>
          <div class="compare-card-response">${escHtml(r.response || '')}</div>
          <div class="compare-card-stats">
            <div class="compare-stat"><span class="compare-stat-val">${fmtNum(r.total_tokens || 0)}</span><span class="compare-stat-label">tokens</span></div>
            <div class="compare-stat"><span class="compare-stat-val">${fmtNum(Math.round(r.latency_ms || 0))}ms</span><span class="compare-stat-label">latency</span></div>
            <div class="compare-stat"><span class="compare-stat-val">${fmtNum(r.prompt_tokens || 0)}</span><span class="compare-stat-label">prompt tok</span></div>
            <div class="compare-stat"><span class="compare-stat-val">${fmtNum(r.completion_tokens || 0)}</span><span class="compare-stat-label">compl tok</span></div>
          </div>`;
      }
      compareGrid.appendChild(card);
    });
  }

  // ── Fetch stats ───────────────────────────────────────────
  async function fetchStats() {
    try {
      const res = await fetch(API_BASE + '/v1/stats', {
        headers: { 'Bypass-Tunnel-Reminder': 'true' }
      });
      if (!res.ok) return;
      const data = await res.json();
      state.stats = data;
      renderStats(data);
    } catch (_) { /* silent */ }
  }
  window.fetchStats = fetchStats;

  function renderStats(s) {
    animateCounter('kpiCalls',   s.total_calls || 0);
    animateCounter('kpiTokens',  s.total_tokens || 0);
    animateCounter('kpiErrors',  s.total_errors || 0);
    animateCounter('kpiVerifications', s.total_verification_retries || 0);
    animateCounterWithSuffix('kpiLatency', Math.round(s.avg_latency_ms || 0), 'ms');

    renderDonut(s.tier_usage || {});
    renderCategoryBars(s.category_usage || {});
  }

  // ── Fetch models ──────────────────────────────────────────
  async function fetchModels() {
    try {
      const res = await fetch(API_BASE + '/v1/models', {
        headers: { 'Bypass-Tunnel-Reminder': 'true' }
      });
      if (!res.ok) return;
      const data = await res.json();
      state.models = data.models || [];

      // provider badge
      const provEl = $('#providerName');
      if (data.provider && provEl) {
        provEl.textContent = data.provider;
      }

      renderModelTiers(data.models || [], data.tier_map || {});
    } catch (_) {
      const mt = $('#modelTiers');
      if (mt) mt.innerHTML = `<p class="empty-state">Could not load models</p>`;
    }
  }

  function renderModelTiers(models, tierMap) {
    const container = $('#modelTiers');
    if (!container) return;
    if (!models.length) {
      container.innerHTML = `<p class="empty-state">No models configured</p>`;
      return;
    }
    container.innerHTML = models.map((m, idx) => `
      <div class="model-tier-row" style="animation: slideIn 0.3s var(--ease-out) both; animation-delay: ${idx * 50}ms">
        <span class="tier-dot t-${m.tier}"></span>
        <span class="model-tier-name" title="${escHtml(m.model_id)}">${escHtml(m.model_id)}</span>
        <span class="model-tier-label">${m.tier}</span>
      </div>
    `).join('');
  }

  // ── Donut Chart (SVG) ────────────────────────────────────
  function renderDonut(tierUsage) {
    const svg = $('#donutChart');
    if (!svg) return;
    const local  = tierUsage['LOCAL']  || 0;
    const remote = tierUsage['REMOTE'] || 0;
    const total  = local + remote;

    // Update legend
    const ls = $('#legendSmall');
    const lm = $('#legendMedium');
    const ll = $('#legendLarge');
    const dt = $('#donutTotal');
    if (ls) ls.textContent = local;
    if (lm) lm.textContent = remote;
    if (ll) ll.textContent = 0;
    if (dt) dt.textContent = total;

    // Remove old segments
    svg.querySelectorAll('.donut-seg').forEach(s => s.remove());

    if (total === 0) return;

    const R = 60;
    const C = 2 * Math.PI * R;
    const segments = [
      { val: local,  color: 'var(--tier-small)' },
      { val: remote, color: 'var(--tier-medium)' },
    ];

    let offset = 0;
    segments.forEach(seg => {
      if (seg.val === 0) return;
      const pct = seg.val / total;
      const len = pct * C;
      const gap = total > 1 ? 3 : 0;
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('class', 'donut-seg');
      circle.setAttribute('cx', '80');
      circle.setAttribute('cy', '80');
      circle.setAttribute('r', String(R));
      circle.setAttribute('fill', 'none');
      circle.setAttribute('stroke', seg.color);
      circle.setAttribute('stroke-width', '22');
      circle.setAttribute('stroke-dasharray', `${Math.max(0, len - gap)} ${C - Math.max(0, len - gap)}`);
      circle.setAttribute('stroke-dashoffset', String(-offset));
      circle.setAttribute('stroke-linecap', 'round');
      svg.appendChild(circle);
      offset += len;
    });
  }

  // ── Bar Chart (CSS) ──────────────────────────────────────
  function renderCategoryBars(catUsage) {
    const container = $('#categoryBars');
    if (!container) return;
    const entries = Object.entries(catUsage).filter(([, v]) => v > 0);
    if (!entries.length) {
      container.innerHTML = '<p class="empty-state">No data yet</p>';
      return;
    }

    const max = Math.max(...entries.map(([, v]) => v));
    const catColors = {
      factual: 'var(--cat-general, #6366f1)',
      mathematical: 'var(--cat-math, #f59e0b)',
      sentiment: 'var(--cat-creative, #ec4899)',
      summarization: 'var(--cat-analysis, #06b6d4)',
      ner: 'var(--cat-translation, #8b5cf6)',
      code_debugging: 'var(--cat-code, #ef4444)',
      code_generation: 'var(--cat-code, #ef4444)',
      logical_reasoning: 'var(--cat-conversation, #22c55e)',
      unknown: 'var(--accent)',
    };

    container.innerHTML = entries.map(([cat, val]) => {
      const pct = (val / max) * 100;
      const color = catColors[cat.toLowerCase()] || 'var(--accent)';
      const displayName = cat.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
      return `<div class="bar-row">
        <div class="bar-label-row">
          <span class="bar-label">${escHtml(displayName)}</span>
          <span class="bar-value">${fmtNum(val)}</span>
        </div>
        <div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${color}"></div></div>
      </div>`;
    }).join('');
  }

  // ── Message rendering ────────────────────────────────────
  function clearWelcome() {
    const w = chatMessages.querySelector('.chat-welcome');
    if (w) w.remove();
  }

  function appendUserMsg(text) {
    const div = document.createElement('div');
    div.className = 'msg msg--user';
    div.innerHTML = `<div class="msg-bubble">${escHtml(text)}</div>`;
    chatMessages.appendChild(div);
    scrollChat();
  }

  function appendTyping() {
    const div = document.createElement('div');
    div.className = 'msg msg--bot';
    div.innerHTML = `<div class="typing-indicator"><span></span><span></span><span></span></div>`;
    chatMessages.appendChild(div);
    scrollChat();
    return div;
  }

  function removeTyping(el) {
    if (el && el.parentNode) el.remove();
  }

  function appendBotMsg(data) {
    const div = document.createElement('div');
    div.className = 'msg msg--bot';

    const catClass = `cat-${(data.category || '').toLowerCase().replace(/[^a-z]/g, '')}`;
    const tierLabel = data.tier || data.routed_to || 'LOCAL';
    const tierClass = `tier-${tierLabel}`;
    const confPct = Math.round((data.confidence || data.complexity_score || 0) * 100);

    const escalatedBadge = data.escalated
      ? `<span class="meta-badge meta-badge--escalated">↑ Escalated</span>`
      : '';

    const routingVisualizer = `
      <div class="routing-trace" style="display:flex; align-items:center; gap:8px; font-size:0.65rem; color:var(--text-tertiary); margin-bottom:8px; font-weight:600; text-transform:uppercase;">
         <span style="color:var(--text-secondary)">USER</span>
         <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
         <span class="meta-badge meta-badge--category ${catClass}" style="font-size:0.55rem; padding:1px 6px;">${escHtml(data.category || '—')}</span>
         <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
         <span class="meta-badge meta-badge--tier ${tierClass}" style="font-size:0.55rem; padding:1px 6px;">${escHtml(tierLabel)}</span>
      </div>
    `;

    div.innerHTML = `
      ${routingVisualizer}
      <div class="msg-bubble bot-bubble-content" style="opacity:0"></div>
      <div class="msg-meta" style="margin-top:6px;">
        <span class="meta-badge meta-badge--category ${catClass}">${escHtml(data.category || '—')}</span>
        <span class="meta-badge meta-badge--tier ${tierClass}">${escHtml(tierLabel)}</span>
        ${escalatedBadge}
        <span class="meta-stat">${iconModel()} ${escHtml(data.model_used || '—')}</span>
        <span class="meta-stat">${iconToken()} <span class="counter-animate" data-target="${data.total_tokens || 0}">0</span> tok</span>
        <span class="meta-stat">${iconClock()} ${fmtNum(Math.round(data.latency_ms || 0))}ms</span>
        <span class="confidence-bar meta-stat">
          ${iconConfidence()} ${confPct}%
          <span class="conf-track"><span class="conf-fill" style="width:0%"></span></span>
        </span>
      </div>`;

    chatMessages.appendChild(div);

    // Markdown render instead of type out
    const contentEl = div.querySelector('.bot-bubble-content');
    renderMarkdown(contentEl, data.response || '(no response)');
    contentEl.animate([{opacity: 0}, {opacity: 1}], {duration: 400, fill: 'forwards'});

    // Animate confidence bar
    requestAnimationFrame(() => {
      const fill = div.querySelector('.conf-fill');
      if (fill) fill.style.width = confPct + '%';
    });

    // Counter animation
    div.querySelectorAll('.counter-animate').forEach(el => {
      animateEl(el, parseInt(el.dataset.target, 10));
    });

    scrollChat();
  }

  function appendError(msg) {
    const div = document.createElement('div');
    div.className = 'msg-error';
    div.innerHTML = `
      <span class="msg-error-icon">${iconError()}</span>
      <span class="msg-error-text">Request failed: ${escHtml(msg)}</span>`;
    chatMessages.appendChild(div);
    scrollChat();
  }

  // ── Markdown Render ────────────────────────────────────
  function renderMarkdown(el, text) {
    if (window.marked) {
      marked.setOptions({
        highlight: function(code, lang) {
          if (window.hljs) {
            const language = hljs.getLanguage(lang) ? lang : 'plaintext';
            return hljs.highlight(code, { language }).value;
          }
          return code;
        }
      });
      el.innerHTML = marked.parse(text);
      el.classList.add('markdown-body');
      
      el.querySelectorAll('pre').forEach(pre => {
        const code = pre.querySelector('code');
        if (!code) return;
        const lang = code.className.replace('hljs language-', '').replace('language-', '') || 'text';
        
        const header = document.createElement('div');
        header.className = 'code-header';
        header.innerHTML = `<span>${lang}</span><button class="copy-btn">Copy</button>`;
        pre.insertBefore(header, code);
        
        const btn = header.querySelector('.copy-btn');
        btn.addEventListener('click', () => {
          navigator.clipboard.writeText(code.textContent);
          btn.textContent = 'Copied!';
          btn.classList.add('copied');
          setTimeout(() => {
            btn.textContent = 'Copy';
            btn.classList.remove('copied');
          }, 2000);
        });
      });
    } else {
      el.textContent = text;
    }
    scrollChat();
  }

  // ── Counter animations ────────────────────────────────────
  function animateCounter(elId, target) {
    const el = document.getElementById(elId);
    if (!el) return;
    const current = parseInt(el.textContent.replace(/[^0-9]/g, ''), 10) || 0;
    if (current === target) return;
    animateEl(el, target, current);
  }

  function animateCounterWithSuffix(elId, target, suffix) {
    const el = document.getElementById(elId);
    if (!el) return;
    const current = parseInt(el.textContent.replace(/[^0-9]/g, ''), 10) || 0;
    if (current === target) return;
    const dur = 600;
    const start = performance.now();
    function tick(now) {
      const p = Math.min(1, (now - start) / dur);
      const eased = 1 - Math.pow(1 - p, 3);
      const val = Math.round(current + (target - current) * eased);
      el.innerHTML = `${fmtNum(val)}<small>${suffix}</small>`;
      if (p < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  function animateEl(el, target, from) {
    from = from !== undefined ? from : 0;
    const dur = 600;
    const start = performance.now();
    function tick(now) {
      const p = Math.min(1, (now - start) / dur);
      const eased = 1 - Math.pow(1 - p, 3);
      el.textContent = fmtNum(Math.round(from + (target - from) * eased));
      if (p < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  // ── Token Savings ─────────────────────────────────────────
  function updateSavings() {
    const actual = state.totalTokensUsed;
    const naive  = state.naiveTokensEstimate;
    const saved  = naive > 0 ? Math.round(((naive - actual) / naive) * 100) : 0;

    animateCounter('savActual', actual);
    animateCounter('savNaive', naive);

    const pctEl = document.getElementById('savPct');
    if (pctEl) pctEl.textContent = saved + '%';
    
    // Dollar savings
    const tokensSaved = Math.max(0, naive - actual);
    const dollarsSaved = (tokensSaved / 1_000_000) * 0.90;
    
    let dollarEl = document.getElementById('savDollars');
    if (!dollarEl) {
       const labelGroup = document.querySelector('.savings-label-group > div');
       if (labelGroup) {
         const div = document.createElement('div');
         div.className = 'savings-dollars';
         div.style = 'margin-top:6px; font-weight:700; color:#10B981; font-size:0.75rem;';
         div.innerHTML = `Est. Saved: $<span id="savDollars">0.00</span>`;
         labelGroup.appendChild(div);
         dollarEl = document.getElementById('savDollars');
       }
    }
    if (dollarEl) {
      dollarEl.textContent = dollarsSaved.toFixed(4);
    }
  }

  // ── Helpers ───────────────────────────────────────────────
  function scrollChat() {
    requestAnimationFrame(() => {
      chatMessages.scrollTop = chatMessages.scrollHeight;
    });
  }

  function updateChatCount() {
    const badge = document.querySelector('.chrome-badge');
    if (badge) badge.textContent = `${state.messageCount} message${state.messageCount !== 1 ? 's' : ''}`;
  }

  function fmtNum(n) {
    if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
    if (n >= 1e4) return (n / 1e3).toFixed(1) + 'k';
    return n.toLocaleString();
  }

  function escHtml(str) {
    const d = document.createElement('div');
    d.textContent = str;
    return d.innerHTML;
  }

  // ── Inline SVG icons (small, crisp) ───────────────────────
  function iconModel() {
    return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="4" y="4" width="16" height="16" rx="2"/><path d="M9 9h6M9 13h4"/></svg>`;
  }
  function iconToken() {
    return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><path d="M12 6v12M8 10h8M8 14h8"/></svg>`;
  }
  function iconClock() {
    return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>`;
  }
  function iconConfidence() {
    return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`;
  }
  function iconError() {
    return `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>`;
  }

})();
