/* ═══════════════════════════════════════════════════════════
   MOOCs Generator Pro — pro.js
   Particle canvas · Hero animations · Scroll reveals · UX polish
   ═══════════════════════════════════════════════════════════ */
'use strict';

// ── Particle Network Canvas ───────────────────────────────────
(function initParticles() {
  const canvas = document.getElementById('bg'); if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let W, H, pts = [];

  function resize() { W = canvas.width = window.innerWidth; H = canvas.height = window.innerHeight; }

  class Pt {
    constructor() { this.reset(); }
    reset() {
      this.x  = Math.random() * W;   this.y  = Math.random() * H;
      this.vx = (Math.random()-.5) * .45; this.vy = (Math.random()-.5) * .45;
      this.r  = Math.random() * 1.5 + .5;
      this.a  = Math.random() * .38 + .07;
      this.c  = Math.random() > .5 ? '124,58,237' : '6,182,212';
    }
    tick() { this.x += this.vx; this.y += this.vy; if (this.x<0||this.x>W||this.y<0||this.y>H) this.reset(); }
    draw() { ctx.beginPath(); ctx.arc(this.x,this.y,this.r,0,Math.PI*2); ctx.fillStyle=`rgba(${this.c},${this.a})`; ctx.fill(); }
  }

  function connections() {
    const D = 135;
    for (let i=0;i<pts.length;i++) for (let j=i+1;j<pts.length;j++) {
      const dx=pts[i].x-pts[j].x, dy=pts[i].y-pts[j].y, d=Math.sqrt(dx*dx+dy*dy);
      if (d<D) { ctx.beginPath(); ctx.moveTo(pts[i].x,pts[i].y); ctx.lineTo(pts[j].x,pts[j].y); ctx.strokeStyle=`rgba(124,58,237,${.07*(1-d/D)})`; ctx.lineWidth=.6; ctx.stroke(); }
    }
  }

  function frame() { ctx.clearRect(0,0,W,H); connections(); pts.forEach(p=>{p.tick();p.draw();}); requestAnimationFrame(frame); }

  window.addEventListener('resize', resize, {passive:true});
  resize();
  pts = Array.from({length:80}, ()=>new Pt());
  frame();
})();


// ── Hero title entrance (stagger) ─────────────────────────────
(function heroTitleAnim() {
  ['hl1','hl2','hl3'].forEach((cls,i) => {
    const el = document.querySelector('.'+cls); if (!el) return;
    Object.assign(el.style, {opacity:'0', transform:'translateY(22px)',
      transition:`opacity .55s ease ${i*.13}s, transform .55s ease ${i*.13}s`});
    requestAnimationFrame(() => requestAnimationFrame(() => {
      el.style.opacity = '1'; el.style.transform = 'none';
    }));
  });
})();


// ── Hero subtitle typing effect ───────────────────────────────
(function heroTyping() {
  const el = document.getElementById('heroSub'); if (!el) return;
  const full = el.textContent; el.textContent = ''; let i = 0;
  const t = setInterval(() => { el.textContent += full[i++]; if (i >= full.length) clearInterval(t); }, 20);
  setTimeout(() => { clearInterval(t); el.textContent = full; }, 5500);
})();


// ── Flow node entrance animation ──────────────────────────────
(function flowNodeAnim() {
  document.querySelectorAll('.fn').forEach((n,i) => {
    Object.assign(n.style, {opacity:'0', transform:'translateY(14px)',
      transition:`opacity .5s ease ${.2+i*.08}s, transform .5s ease ${.2+i*.08}s`});
    requestAnimationFrame(() => requestAnimationFrame(() => {
      n.style.opacity = '1'; n.style.transform = 'none';
    }));
  });
  document.querySelectorAll('.fline').forEach((l,i) => {
    l.style.width = '0'; l.style.transition = `width .4s ease ${.38+i*.08}s`;
    requestAnimationFrame(() => requestAnimationFrame(() => { l.style.width = '28px'; }));
  });
})();


// ── Panel scroll reveal ───────────────────────────────────────
(function panelReveal() {
  if (!('IntersectionObserver' in window)) {
    document.querySelectorAll('.pnl').forEach(p => p.classList.add('vis')); return;
  }
  const obs = new IntersectionObserver(entries => {
    entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add('vis'); obs.unobserve(e.target); } });
  }, {threshold: .06});
  document.querySelectorAll('.pnl').forEach((p,i) => {
    p.style.transitionDelay = `${i*.04}s`;
    obs.observe(p);
  });
})();


// ── Active panel highlight on scroll ─────────────────────────
(function activePanel() {
  const panels = document.querySelectorAll('.pnl');
  if (!panels.length) return;
  const obs = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting && e.intersectionRatio > .22) {
        // Don't override done-p or act-p set by step logic
        // Just add a subtle viewport-visible indicator
        e.target.dataset.inView = '1';
      } else {
        delete e.target.dataset.inView;
      }
    });
  }, {threshold: .22});
  panels.forEach(p => obs.observe(p));
})();


// ── Orb scale on scroll ───────────────────────────────────────
window.addEventListener('scroll', () => {
  const orb = document.getElementById('orb'); if (!orb) return;
  orb.style.transform = window.scrollY > 80 ? 'scale(1.12)' : 'scale(1)';
}, {passive:true});


// ── Keyboard shortcut: / → course title ─────────────────────
document.addEventListener('keydown', e => {
  if (e.key === '/' && !['INPUT','TEXTAREA'].includes(document.activeElement?.tagName)) {
    e.preventDefault();
    const ct = document.getElementById('courseTitle');
    if (ct) { ct.focus(); ct.select(); }
  }
  // Escape closes overlay
  if (e.key === 'Escape') {
    const ov = document.getElementById('overlay');
    if (ov && ov.style.display !== 'none') {
      // Don't close during active fetch — just a UX hint
      ov.style.opacity = '.5';
      setTimeout(() => { ov.style.opacity = '1'; }, 300);
    }
  }
});


// ── Button hover sound (subtle visual feedback) ───────────────
document.querySelectorAll('.bp').forEach(btn => {
  btn.addEventListener('mouseenter', () => {
    if (!btn.disabled) btn.style.letterSpacing = '.01em';
  });
  btn.addEventListener('mouseleave', () => {
    btn.style.letterSpacing = '';
  });
});
