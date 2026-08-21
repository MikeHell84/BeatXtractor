import * as THREE from 'three';

const PARTICLE_COUNT = 2800;
const LERP_SPEED = 0.045;

const canvas = document.getElementById('bg-canvas');
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });

renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x05050f, 1);
camera.position.set(0, 1, 32);

// ── Partículas en forma de piezas de batería ───────────────────────

function writeParticle(out, idx, x, y, z) {
  out[idx * 3] = x;
  out[idx * 3 + 1] = y;
  out[idx * 3 + 2] = z;
}

function fillDrum(out, start, count, cx, cy, cz, radius, height, spread) {
  const shellN = Math.floor(count * 0.55);
  const headN = count - shellN;
  for (let j = 0; j < shellN; j++) {
    const t = (j / shellN) * Math.PI * 2 * 5 + j * 0.17;
    const y = ((j % 12) / 12 - 0.5) * height * spread;
    writeParticle(out, start + j,
      cx + radius * spread * Math.cos(t),
      cy + y,
      cz + radius * spread * Math.sin(t));
  }
  for (let j = 0; j < headN; j++) {
    const t = (j / headN) * Math.PI * 2;
    const r = Math.sqrt((j + 1) / headN) * radius * spread;
    const which = j % 2 === 0 ? 1 : -1;
    writeParticle(out, start + shellN + j,
      cx + r * Math.cos(t),
      cy + which * height * 0.52 * spread,
      cz + r * Math.sin(t));
  }
}

function fillCymbal(out, start, count, cx, cy, cz, radius, spread, tilt = 0.35) {
  for (let j = 0; j < count; j++) {
    const t = (j / count) * Math.PI * 2 * 3 + j * 0.11;
    const r = Math.sqrt((j + 1) / count) * radius * spread;
    const lx = r * Math.cos(t);
    const lz = r * Math.sin(t);
    writeParticle(out, start + j,
      cx + lx,
      cy + lz * Math.sin(tilt),
      cz + lz * Math.cos(tilt));
  }
}

function fillHiHat(out, start, count, cx, cy, cz, radius, spread) {
  const half = Math.floor(count / 2);
  fillCymbal(out, start, half, cx, cy + 0.35 * spread, cz, radius * 0.95, spread, 0.2);
  fillCymbal(out, start + half, count - half, cx, cy - 0.35 * spread, cz, radius, spread, -0.15);
}

function fillWaveformRing(out, start, count, cx, cy, cz, radius, spread, waves = 5) {
  for (let j = 0; j < count; j++) {
    const t = (j / count) * Math.PI * 2;
    const wave = Math.sin(t * waves + j * 0.04) * 0.55 + Math.sin(t * 11) * 0.15;
    const r = radius * spread * (1 + wave * 0.35);
    writeParticle(out, start + j,
      cx + r * Math.cos(t),
      cy + Math.sin(t * 3 + j * 0.02) * 1.2 * spread,
      cz + r * Math.sin(t));
  }
}

function layoutFullKit(spread) {
  const out = new Float32Array(PARTICLE_COUNT * 3);
  const parts = [
    { fn: fillDrum, args: [0, -2.5, 0, 3.2, 4.2, spread], n: 520 },
    { fn: fillDrum, args: [-3.5, 0.2, 2.2, 2.0, 2.0, spread], n: 280 },
    { fn: fillHiHat, args: [-5.0, 1.8, -1.0, 2.4, spread], n: 260 },
    { fn: fillCymbal, args: [4.5, 2.2, -2.5, 3.0, spread, 0.4], n: 240 },
    { fn: fillCymbal, args: [5.5, 1.5, 3.0, 2.6, spread, 0.25], n: 220 },
    { fn: fillDrum, args: [1.2, 1.0, 3.2, 1.4, 1.8, spread], n: 180 },
    { fn: fillDrum, args: [2.8, 0.4, 2.0, 1.6, 2.0, spread], n: 190 },
    { fn: fillDrum, args: [4.0, -0.4, 1.0, 2.0, 2.4, spread], n: 210 },
  ];
  let idx = 0;
  for (const p of parts) {
    const n = Math.min(p.n, PARTICLE_COUNT - idx);
    p.fn(out, idx, n, ...p.args);
    idx += n;
  }
  while (idx < PARTICLE_COUNT) {
    fillDrum(out, idx, 1, 0, 0, 0, 0.3, 0.3, spread * 0.2);
    idx++;
  }
  return out;
}

function layoutWaveformKick(spread) {
  const out = new Float32Array(PARTICLE_COUNT * 3);
  fillDrum(out, 0, 900, 0, -1.5, 0, 3.5, 4.5, spread);
  fillWaveformRing(out, 900, 1100, 0, 0.5, 0, 7.5, spread, 7);
  fillWaveformRing(out, 2000, 800, 0, 1.2, 0, 10, spread * 1.05, 4);
  return out;
}

function layoutFourStems(spread) {
  const out = new Float32Array(PARTICLE_COUNT * 3);
  const clusters = [
    { cx: -7 * spread, cy: 1, cz: 0, r: 2.8, kind: 'drum' },
    { cx: 7 * spread, cy: -1, cz: 0, r: 2.4, kind: 'bass' },
    { cx: 0, cy: 6 * spread, cz: -4, r: 2.2, kind: 'vox' },
    { cx: 0, cy: -6 * spread, cz: 4, r: 2.2, kind: 'other' },
  ];
  const per = Math.floor(PARTICLE_COUNT / 4);
  clusters.forEach((c, i) => {
    const start = i * per;
    const n = i === 3 ? PARTICLE_COUNT - start : per;
    if (c.kind === 'drum') {
      fillDrum(out, start, n, c.cx, c.cy, c.cz, c.r, c.r * 1.4, spread);
    } else if (c.kind === 'bass') {
      for (let j = 0; j < n; j++) {
        const t = (j / n) * Math.PI * 2 * 2;
        const y = ((j % 20) / 20 - 0.5) * 5 * spread;
        writeParticle(out, start + j, c.cx + Math.cos(t) * c.r * spread, c.cy + y, c.cz + Math.sin(t) * 0.6);
      }
    } else {
      fillCymbal(out, start, n, c.cx, c.cy, c.cz, c.r, spread, 0.5);
    }
  });
  return out;
}

function layoutSplitPieces(spread) {
  const out = new Float32Array(PARTICLE_COUNT * 3);
  const pieces = [
    { fn: fillDrum, args: [-6, -1, 0, 2.8, 3.5], n: 380 },
    { fn: fillDrum, args: [-2, 0.5, 2, 1.8, 1.6], n: 280 },
    { fn: fillHiHat, args: [-4, 2, -2, 2.0], n: 260 },
    { fn: fillCymbal, args: [2, 2.5, -3, 2.8], n: 240 },
    { fn: fillDrum, args: [4, 1.5, 1, 1.3, 1.5], n: 200 },
    { fn: fillDrum, args: [5.5, 0.8, 0, 1.5, 1.8], n: 220 },
    { fn: fillDrum, args: [6.5, 0, -1.5, 1.8, 2.2], n: 240 },
  ];
  let idx = 0;
  for (const p of pieces) {
    const [cx, cy, cz, r, h] = p.args;
    p.fn(out, idx, p.n, cx * spread, cy * spread, cz * spread, r, h, spread);
    idx += p.n;
  }
  while (idx < PARTICLE_COUNT) {
    fillCymbal(out, idx, 1, 0, 0, 0, 0.5, spread * 0.3);
    idx++;
  }
  return out;
}

function layoutKickBeats(spread) {
  const out = new Float32Array(PARTICLE_COUNT * 3);
  fillDrum(out, 0, 700, 0, -1, 0, 4, 5, spread);
  const hitAngles = [0, 0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 5.4];
  let idx = 700;
  for (let ring = 0; ring < 6 && idx < PARTICLE_COUNT; ring++) {
    const n = Math.min(220, PARTICLE_COUNT - idx);
    const r = (1.2 + ring * 0.55) * spread;
    for (let j = 0; j < n; j++) {
      const base = hitAngles[j % hitAngles.length];
      const t = base + (j / n) * 0.15;
      writeParticle(out, idx + j,
        Math.cos(t) * r * 4.2,
        2.8 + ring * 0.35 * spread,
        Math.sin(t) * r * 4.2);
    }
    idx += n;
  }
  while (idx < PARTICLE_COUNT) {
    writeParticle(out, idx, (Math.random() - 0.5) * 2, 4 + Math.random() * 3, (Math.random() - 0.5) * 2);
    idx++;
  }
  return out;
}

function layoutOneShotGrid(spread) {
  const out = new Float32Array(PARTICLE_COUNT * 3);
  const cols = 8;
  const rows = 5;
  const cell = 2.1 * spread;
  let idx = 0;
  for (let row = 0; row < rows && idx < PARTICLE_COUNT; row++) {
    for (let col = 0; col < cols && idx < PARTICLE_COUNT; col++) {
      const cx = (col - cols / 2 + 0.5) * cell;
      const cz = (row - rows / 2 + 0.5) * cell;
      const isCym = (row + col) % 3 === 0;
      const n = Math.min(55, PARTICLE_COUNT - idx);
      if (isCym) {
        fillCymbal(out, idx, n, cx, 0, cz, 0.75, spread * 0.85, 0.1);
      } else {
        fillDrum(out, idx, n, cx, 0, cz, 0.55, 0.7, spread * 0.85);
      }
      idx += n;
    }
  }
  while (idx < PARTICLE_COUNT) {
    fillDrum(out, idx, 1, 0, 0, 0, 0.4, 0.5, spread * 0.5);
    idx++;
  }
  return out;
}

function layoutSingleKick(spread) {
  const out = new Float32Array(PARTICLE_COUNT * 3);
  fillDrum(out, 0, 1800, 0, 0, 0, 5, 6.5, spread);
  fillWaveformRing(out, 1800, 1000, 0, 3.5, 0, 3.5, spread, 3);
  return out;
}

const layoutBuilders = {
  fullKit: layoutFullKit,
  waveformKick: layoutWaveformKick,
  fourStems: layoutFourStems,
  splitPieces: layoutSplitPieces,
  kickBeats: layoutKickBeats,
  oneShotGrid: layoutOneShotGrid,
  singleKick: layoutSingleKick,
};

const sectionStates = {
  hero: {
    cameraPos: { x: 0, y: 1, z: 32 },
    color: new THREE.Color(0x6366f1),
    color2: new THREE.Color(0xa855f7),
    bg: new THREE.Color(0x05050f),
    centralScale: 1,
    particleSpread: 1,
    drumLayout: 'fullKit',
    centralDrum: 'kick',
    ringMode: 'sticks',
  },
  features: {
    cameraPos: { x: 5, y: 3, z: 28 },
    color: new THREE.Color(0x10b981),
    color2: new THREE.Color(0x06b6d4),
    bg: new THREE.Color(0x050f0a),
    centralScale: 1.05,
    particleSpread: 1,
    drumLayout: 'waveformKick',
    centralDrum: 'kick',
    ringMode: 'waveform',
  },
  separacion: {
    cameraPos: { x: -6, y: 2, z: 30 },
    color: new THREE.Color(0x8b5cf6),
    color2: new THREE.Color(0x6366f1),
    bg: new THREE.Color(0x08050f),
    centralScale: 0.85,
    particleSpread: 1.15,
    drumLayout: 'fourStems',
    centralDrum: 'split',
    ringMode: 'orbit',
  },
  division: {
    cameraPos: { x: 4, y: -2, z: 30 },
    color: new THREE.Color(0xf59e0b),
    color2: new THREE.Color(0xef4444),
    bg: new THREE.Color(0x0f0805),
    centralScale: 0.75,
    particleSpread: 1.05,
    drumLayout: 'splitPieces',
    centralDrum: 'snare',
    ringMode: 'pieces',
  },
  beats: {
    cameraPos: { x: -4, y: -1, z: 28 },
    color: new THREE.Color(0xec4899),
    color2: new THREE.Color(0xf97316),
    bg: new THREE.Color(0x0f050a),
    centralScale: 1.2,
    particleSpread: 1.1,
    drumLayout: 'kickBeats',
    centralDrum: 'kick',
    ringMode: 'hits',
  },
  export: {
    cameraPos: { x: 0, y: 4, z: 27 },
    color: new THREE.Color(0x22d3ee),
    color2: new THREE.Color(0x6366f1),
    bg: new THREE.Color(0x050a0f),
    centralScale: 0.6,
    particleSpread: 1,
    drumLayout: 'oneShotGrid',
    centralDrum: 'pad',
    ringMode: 'grid',
  },
  install: {
    cameraPos: { x: 0, y: 0, z: 24 },
    color: new THREE.Color(0xfbbf24),
    color2: new THREE.Color(0xa855f7),
    bg: new THREE.Color(0x0a0805),
    centralScale: 1.35,
    particleSpread: 0.9,
    drumLayout: 'singleKick',
    centralDrum: 'kick',
    ringMode: 'download',
  },
};

function cloneState(s) {
  return {
    ...s,
    cameraPos: { ...s.cameraPos },
    color: s.color.clone(),
    color2: s.color2.clone(),
    bg: s.bg.clone(),
  };
}

const layoutCache = {};
for (const [key, state] of Object.entries(sectionStates)) {
  layoutCache[key] = layoutBuilders[state.drumLayout](state.particleSpread);
}

let currentState = cloneState(sectionStates.hero);
let targetState = cloneState(sectionStates.hero);

const particleGeo = new THREE.BufferGeometry();
const positions = new Float32Array(PARTICLE_COUNT * 3);
const targetPositions = new Float32Array(layoutCache.hero);
positions.set(layoutCache.hero);

const colors = new Float32Array(PARTICLE_COUNT * 3);
for (let i = 0; i < PARTICLE_COUNT; i++) {
  colors[i * 3] = 0.4;
  colors[i * 3 + 1] = 0.4;
  colors[i * 3 + 2] = 0.9;
}

particleGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
particleGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

const particles = new THREE.Points(particleGeo, new THREE.PointsMaterial({
  size: 0.11,
  vertexColors: true,
  transparent: true,
  opacity: 0.92,
  blending: THREE.AdditiveBlending,
  sizeAttenuation: true,
}));
scene.add(particles);

const drumGeometries = {
  kick: new THREE.CylinderGeometry(2.4, 2.8, 3.2, 32, 1, true),
  snare: new THREE.CylinderGeometry(1.9, 1.9, 1.4, 32, 1, true),
  cymbal: new THREE.CylinderGeometry(3.2, 3.2, 0.12, 48, 1, true),
  hihat: new THREE.CylinderGeometry(2.4, 2.4, 0.08, 48),
  pad: new THREE.BoxGeometry(1.4, 0.25, 1.4),
  split: new THREE.OctahedronGeometry(2.2, 1),
};

const centralMat = new THREE.MeshPhongMaterial({
  color: 0x6366f1,
  emissive: 0x6366f1,
  emissiveIntensity: 0.35,
  wireframe: true,
  transparent: true,
  opacity: 0.65,
  side: THREE.DoubleSide,
});

const centralMesh = new THREE.Mesh(drumGeometries.kick, centralMat);
scene.add(centralMesh);

const hiHatTop = new THREE.Mesh(
  new THREE.CylinderGeometry(2.6, 2.6, 0.06, 48),
  new THREE.MeshBasicMaterial({ color: 0xa855f7, wireframe: true, transparent: true, opacity: 0.45 })
);
hiHatTop.position.y = 0.5;
hiHatTop.visible = false;
scene.add(hiHatTop);

const stick1 = new THREE.Mesh(
  new THREE.CylinderGeometry(0.06, 0.08, 4.5, 8),
  new THREE.MeshBasicMaterial({ color: 0xfbbf24, transparent: true, opacity: 0.7 })
);
stick1.rotation.z = Math.PI / 3;
stick1.position.set(3, 2, 0);
scene.add(stick1);

const stick2 = stick1.clone();
stick2.rotation.z = -Math.PI / 4;
stick2.position.set(-3, 2.5, 1);
scene.add(stick2);

const ring = new THREE.Mesh(
  new THREE.TorusGeometry(5.5, 0.07, 12, 80),
  new THREE.MeshBasicMaterial({ color: 0xa855f7, transparent: true, opacity: 0.35 })
);
ring.rotation.x = Math.PI / 2;
scene.add(ring);

const ring2 = new THREE.Mesh(
  new THREE.TorusGeometry(7.5, 0.05, 12, 80),
  new THREE.MeshBasicMaterial({ color: 0xec4899, transparent: true, opacity: 0.25 })
);
ring2.rotation.x = Math.PI / 2.8;
scene.add(ring2);

scene.add(new THREE.AmbientLight(0x404040, 0.55));
const pointLight1 = new THREE.PointLight(0x6366f1, 2.2, 120);
pointLight1.position.set(10, 10, 10);
scene.add(pointLight1);
const pointLight2 = new THREE.PointLight(0xec4899, 2, 120);
pointLight2.position.set(-10, -8, 12);
scene.add(pointLight2);

let targetCentralDrum = 'kick';
let currentCentralDrum = 'kick';

function setCentralDrum(type) {
  if (type === currentCentralDrum) return;
  centralMesh.geometry = drumGeometries[type];
  currentCentralDrum = type;
  hiHatTop.visible = type === 'hihat';
  centralMesh.rotation.x = type === 'cymbal' ? 0.35 : 0;
}

const mouse = { x: 0, y: 0 };
const targetMouse = { x: 0, y: 0 };
document.addEventListener('mousemove', (e) => {
  targetMouse.x = (e.clientX / window.innerWidth) * 2 - 1;
  targetMouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
});

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function updateTargetPositions(sectionName) {
  targetPositions.set(layoutCache[sectionName]);
  targetState = cloneState(sectionStates[sectionName]);
  targetCentralDrum = sectionStates[sectionName].centralDrum;
}

const sections = document.querySelectorAll('section');
const navLinks = document.querySelectorAll('.nav-link');
const dots = document.querySelectorAll('.dot');
let currentSection = 'hero';

const sectionObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      const id = entry.target.id;
      if (id !== currentSection) {
        currentSection = id;
        updateTargetPositions(id);
        navLinks.forEach((link) => {
          link.classList.toggle('active', link.getAttribute('href') === `#${id}`);
        });
        dots.forEach((dot) => {
          dot.classList.toggle('active', dot.getAttribute('data-target') === id);
        });
      }
    }
  });
}, { threshold: 0.42 });

sections.forEach((s) => sectionObserver.observe(s));

dots.forEach((dot) => {
  dot.addEventListener('click', () => {
    document.getElementById(dot.getAttribute('data-target')).scrollIntoView({ behavior: 'smooth' });
  });
});

const clock = new THREE.Clock();
const mixedColor = new THREE.Color();

function animate() {
  requestAnimationFrame(animate);
  const time = clock.getElapsedTime();

  mouse.x += (targetMouse.x - mouse.x) * 0.05;
  mouse.y += (targetMouse.y - mouse.y) * 0.05;

  currentState.cameraPos.x += (targetState.cameraPos.x - currentState.cameraPos.x) * LERP_SPEED;
  currentState.cameraPos.y += (targetState.cameraPos.y - currentState.cameraPos.y) * LERP_SPEED;
  currentState.cameraPos.z += (targetState.cameraPos.z - currentState.cameraPos.z) * LERP_SPEED;
  currentState.color.lerp(targetState.color, LERP_SPEED);
  currentState.color2.lerp(targetState.color2, LERP_SPEED);
  currentState.bg.lerp(targetState.bg, LERP_SPEED);
  currentState.centralScale += (targetState.centralScale - currentState.centralScale) * LERP_SPEED;

  if (targetCentralDrum !== currentCentralDrum) setCentralDrum(targetCentralDrum);

  const posAttr = particleGeo.attributes.position;
  const colAttr = particleGeo.attributes.color;
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    posAttr.array[i * 3] += (targetPositions[i * 3] - posAttr.array[i * 3]) * LERP_SPEED;
    posAttr.array[i * 3 + 1] += (targetPositions[i * 3 + 1] - posAttr.array[i * 3 + 1]) * LERP_SPEED;
    posAttr.array[i * 3 + 2] += (targetPositions[i * 3 + 2] - posAttr.array[i * 3 + 2]) * LERP_SPEED;
    mixedColor.copy(currentState.color).lerp(currentState.color2, i / PARTICLE_COUNT);
    colAttr.array[i * 3] = mixedColor.r;
    colAttr.array[i * 3 + 1] = mixedColor.g;
    colAttr.array[i * 3 + 2] = mixedColor.b;
  }
  posAttr.needsUpdate = true;
  colAttr.needsUpdate = true;

  particles.rotation.y = time * 0.06;
  particles.rotation.x = mouse.y * 0.12;

  let beatPulse = 1;
  if (currentSection === 'beats') {
    beatPulse = 1 + Math.max(0, Math.sin(time * 8)) * 0.12;
  }

  centralMesh.rotation.y = time * 0.25;
  const s = currentState.centralScale * beatPulse;
  centralMesh.scale.set(s, s * (currentCentralDrum === 'cymbal' ? 0.15 : 1), s);
  centralMat.color.copy(currentState.color);
  centralMat.emissive.copy(currentState.color);

  hiHatTop.rotation.y = time * 0.4;
  hiHatTop.material.color.copy(currentState.color2);

  stick1.rotation.x = Math.sin(time * 3.5) * 0.6 - 0.3;
  stick2.rotation.x = Math.sin(time * 3.5 + Math.PI) * 0.5 + 0.2;
  const showSticks = currentSection === 'hero' || currentSection === 'beats' || currentSection === 'division';
  stick1.visible = showSticks;
  stick2.visible = showSticks;

  const mode = sectionStates[currentSection]?.ringMode || 'sticks';
  ring2.visible = true;
  if (mode === 'waveform') {
    ring.scale.setScalar(1 + Math.sin(time * 2) * 0.08);
    ring2.scale.setScalar(1 + Math.sin(time * 2 + 1) * 0.06);
  } else if (mode === 'hits') {
    ring.scale.setScalar(beatPulse);
    ring2.scale.setScalar(1 + Math.max(0, Math.sin(time * 8 - 0.5)) * 0.2);
  } else if (mode === 'orbit') {
    ring.rotation.z = time * 0.35;
    ring2.rotation.z = -time * 0.25;
    ring.scale.setScalar(1);
    ring2.scale.setScalar(1);
  } else if (mode === 'grid') {
    ring.scale.setScalar(1.3);
    ring2.visible = false;
  } else {
    ring.scale.setScalar(1);
    ring2.scale.setScalar(1);
  }

  ring.rotation.z = time * 0.18;
  ring.rotation.x = Math.PI / 2 + mouse.y * 0.25;
  ring.material.color.copy(currentState.color2);
  if (ring2.visible) {
    ring2.rotation.z = -time * 0.12;
    ring2.rotation.x = Math.PI / 2.8 + mouse.x * 0.2;
    ring2.material.color.copy(currentState.color);
  }

  pointLight1.color.copy(currentState.color);
  pointLight2.color.copy(currentState.color2);

  camera.position.x = currentState.cameraPos.x + mouse.x * 2;
  camera.position.y = currentState.cameraPos.y + mouse.y * 2;
  camera.position.z = currentState.cameraPos.z;
  camera.lookAt(0, 0, 0);

  renderer.setClearColor(currentState.bg);
  renderer.render(scene, camera);
}
animate();

setTimeout(() => document.getElementById('loader').classList.add('hidden'), 600);

const navbar = document.getElementById('navbar');
window.addEventListener('scroll', () => {
  navbar.classList.toggle('scrolled', window.scrollY > 50);
});
