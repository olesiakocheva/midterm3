// app.js — UI и пайплайн (+ ручной прогноз)
import { parseCSVFile, parseCSVUrl, inferSchema, prepareTensors, confusionMatrix } from './data_utils.js';
import { buildMLP, trainModel, evaluateModel } from './model.js';

const ui = {
  btnLoadEmbedded: document.getElementById('btnLoadEmbedded'),
  csv: document.getElementById('csv'),
  colTarget: document.getElementById('colTarget'),
  exclude: document.getElementById('exclude'),
  split: document.getElementById('split'),
  classWeight: document.getElementById('classWeight'),
  btnPrep: document.getElementById('btnPrep'),

  arch: document.getElementById('arch'),
  drop: document.getElementById('drop'),
  epochs: document.getElementById('epochs'),
  batch: document.getElementById('batch'),
  lr: document.getElementById('lr'),
  thr: document.getElementById('thr'),
  btnBuild: document.getElementById('btnBuild'),
  btnTrain: document.getElementById('btnTrain'),
  btnEval: document.getElementById('btnEval'),

  meta: document.getElementById('meta'),
  results: document.getElementById('results'),
  thead: document.getElementById('thead'),
  tbody: document.getElementById('tbody'),
  trainChart: document.getElementById('trainChart'),
  cmat: document.getElementById('cmat'),
  log: document.getElementById('log'),

  manualForm: document.getElementById('manualForm'),
  btnPredictManual: document.getElementById('btnPredictManual'),
  manualOut: document.getElementById('manualOut'),
};

let RAW = null;
let SCHEMA = null;
let DATA = null;
let MODEL = null;
let CHART = null; // ← Оставляем только одну декларацию здесь!

const logln = s => { ui.log.textContent += s + "\n"; ui.log.scrollTop = ui.log.scrollHeight; };
function disable(b){
  ui.btnPrep.disabled = b || !RAW;
  ui.btnBuild.disabled= b || !DATA;
  ui.btnTrain.disabled= b || !MODEL || !DATA;
  ui.btnEval.disabled = b || !MODEL || !DATA;
  ui.btnPredictManual.disabled = b || !MODEL || !DATA;
}

/* ---------- helpers ---------- */
async function initTF(){
  try {
    tf.env().set('WEBGL_VERSION', 1);
    tf.env().set('WEBGL_PACK', false);
    await tf.setBackend('webgl'); await tf.ready();
  } catch { await tf.setBackend('cpu'); await tf.ready(); }
  logln("TF backend: " + tf.getBackend());
}

let CHART;
function initChart(){
  if (CHART) CHART.destroy();
  CHART = new Chart(ui.trainChart.getContext('2d'), {
    type:'line',
    data:{labels:[], datasets:[
      {label:'loss', data:[], tension:.2},
      {label:'val_loss', data:[], tension:.2},
      {label:'acc', data:[], tension:.2},
    ]},
    options:{responsive:true, scales:{y:{beginAtZero:true}}, plugins:{legend:{position:'bottom'}}}
  });
}
function addPoint(epoch, loss, vloss, acc){
  CHART.data.labels.push(String(epoch));
  CHART.data.datasets[0].data.push(loss ?? null);
  CHART.data.datasets[1].data.push(vloss ?? null);
  CHART.data.datasets[2].data.push(acc ?? null);
  CHART.update();
}
function argmax(a){ let m=-1, mi=-1; for(let i=0;i<a.length;i++){ if(a[i]>m){m=a[i];mi=i;} } return mi; }
function invertMap(m){ return Object.entries(m).reduce((acc,[k,v])=> (acc[v]=k, acc), {}); }

function renderExamples(P, Y, labelMap, K=12){
  const inv = invertMap(labelMap);
  const rows=[];
  for (let i=0;i<Math.min(K,P.length);i++){
    const pi = P[i], yi = Y[i];
    const predIdx = argmax(pi);
    const trueIdx = argmax(yi);
    const prob = pi[predIdx];
    rows.push(`<tr><td>${i+1}</td><td>${inv[trueIdx]}</td><td>${inv[predIdx]}</td><td>${(prob*100).toFixed(1)}%</td></tr>`);
  }
  ui.thead.innerHTML = `<tr><th>#</th><th>True</th><th>Pred</th><th>Prob</th></tr>`;
  ui.tbody.innerHTML = rows.join("");
}

function renderConfusion(trueIdx, predIdx, nClasses, labelMap){
  const M = confusionMatrix(trueIdx, predIdx, nClasses);
  const inv = invertMap(labelMap);
  const labels = Array.from({length:nClasses}, (_,i)=> inv[i]);

  const ctx = ui.cmat.getContext('2d');
  const W = ui.cmat.width, H = ui.cmat.height;
  ctx.clearRect(0,0,W,H);
  const cellW = (W-40)/nClasses, cellH = (H-40)/nClasses;

  let maxv = 1;
  for (const r of M) for (const v of r) maxv = Math.max(maxv, v);

  ctx.font = '12px system-ui';
  ctx.fillStyle = '#0f172a';
  ctx.fillText('True →', 5, 14);
  ctx.save();
  ctx.translate(10, H-5);
  ctx.rotate(-Math.PI/2);
  ctx.fillText('Predicted →', 0, 0);
  ctx.restore();

  for (let i=0;i<nClasses;i++){
    for (let j=0;j<nClasses;j++){
      const x = 30 + j*cellW, y = 20 + i*cellH;
      const val = M[i][j];
      const t = val/maxv;
      const col = Math.floor(255*(1-t));
      ctx.fillStyle = `rgb(${col},${255-col},${180})`;
      ctx.fillRect(x,y,cellW-2,cellH-2);
      ctx.fillStyle = '#0f172a';
      ctx.fillText(String(val), x+cellW/2-8, y+cellH/2+4);
    }
  }
  ctx.fillStyle = '#0f172a';
  for (let i=0;i<nClasses;i++){
    ctx.fillText(labels[i], 0, 20 + i*cellH + 12);
    ctx.fillText(labels[i], 30 + i*cellW + 4, H-4);
  }
}
