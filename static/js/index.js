document.addEventListener('DOMContentLoaded', () => {
  // ---- math
  if (window.renderMathInElement) {
    renderMathInElement(document.body, {
      delimiters: [
        { left: '\\[', right: '\\]', display: true },
        { left: '\\(', right: '\\)', display: false },
      ],
      throwOnError: false,
    });
  }

  // ---- single-choice button groups
  const activate = (group, btn) => {
    group.querySelectorAll('button').forEach((b) => {
      const on = b === btn;
      b.classList.toggle('is-active', on);
      b.setAttribute('aria-selected', on);
    });
  };

  // ---- ImageNet gallery: class x number of steps
  const img = document.getElementById('gallery-img');
  const caption = document.getElementById('gallery-caption');
  const classPills = document.getElementById('class-pills');
  const nfeToggle = document.getElementById('nfe-toggle');
  const state = { cls: '279', clsName: 'Arctic fox', nfe: '1nfe', nfeName: '1 step (1 NFE)' };

  const showGallery = () => {
    img.src = `./static/images/imagenet/c${state.cls}_${state.nfe}.jpg`;
    img.alt = `Uncurated CDMD samples, ImageNet class ${state.clsName}, ${state.nfeName}.`;
    caption.textContent = `${state.clsName} · ${state.nfeName} · uncurated`;
  };

  classPills.addEventListener('click', (e) => {
    const btn = e.target.closest('button');
    if (!btn) return;
    activate(classPills, btn);
    state.cls = btn.dataset.class;
    state.clsName = btn.textContent.trim();
    showGallery();
  });
  nfeToggle.addEventListener('click', (e) => {
    const btn = e.target.closest('button');
    if (!btn) return;
    activate(nfeToggle, btn);
    state.nfe = btn.dataset.nfe;
    state.nfeName = btn.dataset.caption || btn.textContent.trim();
    showGallery();
  });

  // ---- ImageNet table: teacher selector
  const teacherToggle = document.getElementById('teacher-toggle');
  teacherToggle.addEventListener('click', (e) => {
    const btn = e.target.closest('button');
    if (!btn) return;
    activate(teacherToggle, btn);
    document.querySelectorAll('tbody[data-teacher]').forEach((tb) => {
      tb.hidden = tb.dataset.teacher !== btn.dataset.teacher;
    });
  });

  // ---- BibTeX copy
  const copyBtn = document.getElementById('copy-bibtex');
  copyBtn.addEventListener('click', async () => {
    const text = document.getElementById('bibtex-code').textContent;
    const label = copyBtn.querySelector('span');
    try {
      await navigator.clipboard.writeText(text);
      label.textContent = 'Copied';
    } catch (err) {
      label.textContent = 'Select & copy';
    }
    setTimeout(() => { label.textContent = 'Copy'; }, 1600);
  });

  // ---- lightbox for large figures
  const box = document.getElementById('lightbox');
  const boxImg = box.querySelector('img');
  document.querySelectorAll('img.zoomable').forEach((el) => {
    el.addEventListener('click', () => {
      boxImg.src = el.src;
      boxImg.alt = el.alt;
      box.hidden = false;
    });
  });
  const close = () => { box.hidden = true; boxImg.src = ''; };
  box.addEventListener('click', close);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !box.hidden) close(); });
});

// ---- mobile navbar toggle
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.navbar-burger').forEach((burger) => {
    burger.addEventListener('click', () => {
      const menu = document.getElementById(burger.dataset.target);
      const open = !burger.classList.contains('is-active');
      burger.classList.toggle('is-active', open);
      menu.classList.toggle('is-active', open);
      burger.setAttribute('aria-expanded', open);
    });
  });
  document.querySelectorAll('#nav-menu a[href^="#"]').forEach((a) => {
    a.addEventListener('click', () => {
      document.querySelectorAll('.navbar-burger, #nav-menu').forEach((el) => el.classList.remove('is-active'));
    });
  });
});

