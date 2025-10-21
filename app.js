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
let CHART = null; // единственное объявление

const logln = s => { ui.log.textContent += s + "\n"; ui.log.scrollTop = ui.log.scrollHeight; };
function disable(b){
  ui.btnPrep.disabled = b || !RAW;
  ui.btnBuild.disabled= b || !DATA;
  ui.btnTrain.disabled= b || !MODEL || !DATA;
  ui.btnEval.disabled = b || !MODEL || !DATA;
  ui.btnPredictManual.disabled = b || !MODEL || !DATA;
}

/* ==== загрузка CSV ==== */
ui.btnLoadEmbedded.onclick = async ()=>{
  try{
    ui.log.textContent="";
    // относительный путь работает на GitHub Pages под /midterm3/
    const url = './data/Sleep_health_and_lifestyle_dataset.csv';
    logln('Пробую загрузить: ' + url);
    RAW = await parseCSVUrl(url);
    if (!RAW?.length) throw new Error('CSV пуст или не найден');
    afterLoad();
  }catch(e){ logln("❌ Ошибка загрузки: " + (e.message||e)); }
};

ui.csv.onchange = async ()=>{
  const f = ui.csv.files?.[0]; if(!f) return;
  ui.log.textContent = "";
  RAW = await parseCSVFile(f);
  afterLoad();
};

function afterLoad(){
  logln(`Строк: ${RAW.length}`);
  SCHEMA = inferSchema(RAW);
  const cols = Object.keys(SCHEMA);

  ui.colTarget.innerHTML = cols.map(c=> `<option value="${c}">${c}</option>`).join('');
  ui.exclude.innerHTML   = cols.map(c=> `<option value="${c}">${c}</option>`).join('');

  const guess = cols.find(c=> /sleep\s*disorder/i.test(c)) || cols.find(c=> /disorder|insomnia|apnea/i.test(c)) || cols[0];
  ui.colTarget.value = guess;

  const numCnt = cols.filter(c=> SCHEMA[c].kind==='numeric').length;
  const catCnt = cols.length - numCnt;
  ui.meta.innerHTML = `<span class="pill">Колонки: ${cols.length}</span> <span class="pill">Num: ${numCnt}</span> <span class="pill">Cat: ${catCnt}</span>`;
  ui.btnPrep.disabled = false;
}

/* ==== подготовка ==== */
ui.btnPrep.onclick = async ()=>{
  try{
    disable(true);
    await initTF();

    const targetCol = ui.colTarget.value;
    const allCols = Object.keys(SCHEMA);
    const excluded = [...ui.exclude.selectedOptions].map(o=> o.value);
    const featureCols = allCols.filter(c=> c!==targetCol && !excluded.includes(c));
    if (featureCols.length === 0) throw new Error("Нет признаков — уберите что-нибудь из исключений.");

    const splitPct = Math.max(0.5, Math.min(0.95, (+ui.split.value||80)/100));
    const cWeight = ui.classWeight.value;

    DATA = prepareTensors(RAW, featureCols, targetCol, SCHEMA, splitPct, cWeight);

    ui.meta.innerHTML += ` <span class="pill">Train: ${DATA.Xtrain.shape[0]}</span> <span class="pill">Test: ${DATA.Xtest.shape[0]}</span> <span class="pill">Inputs: ${DATA.inputDim}</span> <span class="pill">Classes: ${DATA.nClasses}</span>`;
    logln(`Готово. Цель="${targetCol}". Классов: ${DATA.nClasses}.`);

    buildManualForm(featureCols, SCHEMA, DATA.pre);
    ui.btnBuild.disabled = false;
  } catch(e){
    logln("❌ Ошибка подготовки: " + (e.message||e));
  } finally { disable(false); }
};

/* ==== модель ==== */
ui.btnBuild.onclick = ()=>{
  try{
    disable(true);
    if (MODEL) MODEL.dispose();
    const cfg = {
      arch: ui.arch.value,
      drop: +ui.drop.value || 0.2,
      nClasses: DATA.nClasses,
      lr: +ui.lr.value || 1e-3
    };
    MODEL = buildMLP(DATA.inputDim, cfg);
    logln(`Модель: MLP ${cfg.arch}, drop=${cfg.drop}, classes=${cfg.nClasses}, lr=${cfg.lr}`);
    ui.btnTrain.disabled = false;
  } finally { disable(false); }
};

ui.btnTrain.onclick = async ()=>{
  try{
    disable(true);
    initChart();
    const ep = +ui.epochs.value||25, bs= +ui.batch.value||32;
    logln(`Обучение: epochs=${ep}, batch=${bs}`);
    await trainModel(MODEL, DATA.Xtrain, DATA.Ytrain, ep, bs,
      (e,logs)=> addPoint(e+1, logs.loss, logs.val_loss, logs.acc??logs.accuracy));
    logln("Обучение завершено.");
    ui.btnEval.disabled = false;
    ui.btnPredictManual.disabled = false;
  } catch(e){
    logln("❌ Ошибка обучения: " + (e.message||e));
  } finally { disable(false); }
};

ui.btnEval.onclick = async ()=>{
  try{
    disable(true);
    const res = await evaluateModel(MODEL, DATA.Xtest, DATA.Ytest);
    ui.results.textContent = `Test — loss=${res.loss.toFixed(4)}, accuracy=${(res.acc*100).toFixed(1)}%`;

    const P = await MODEL.predict(DATA.Xtest).array();
    const Y = await DATA.Ytest.array();
    const predsIdx = P.map(p=> argmax(p));
    const trueIdx  = Y.map(y=> argmax(y));
    renderExamples(P, Y, DATA.labelMap, 15);
    renderConfusion(trueIdx, predsIdx, DATA.nClasses, DATA.labelMap);
  } catch(e){
    logln("❌ Ошибка оценки: " + (e.message||e));
  } finally { disable(false); }
};

/* ==== ручной прогноз ==== */
function buildManualForm(featureCols, schema, pre){
  const form = ui.manualForm;
  form.innerHTML = '';
  const lists = {};

  featureCols.forEach(col=>{
    const kind = schema[col].kind;
    const wrap = document.createElement('div');

    const label = document.createElement('label');
    label.textContent = col;
    label.style.display = 'block';
    label.style.marginBottom = '4px';

    let input;
    if (kind === 'numeric'){
      input = document.createElement('input');
      input.type='number'; input.step='any';
      input.id = `mf_${col}`;
      const st = pre.numStats[col];
      if (st) input.title = `min=${st.min.toFixed(2)}, max=${st.max.toFixed(2)}`;
    } else {
      input = document.createElement('input');
      input.type='text';
      input.id = `mf_${col}`;
      const cats = pre.catMaps[col] || [];
      if (cats.length){
        const listId = `dl_${col}`;
        const dl = document.createElement('datalist');
        dl.id = listId;
        dl.innerHTML = cats.map(c=> `<option value="${c}"></option>`).join('');
        lists[listId]=dl;
        input.setAttribute('list', listId);
        input.title = `известные: ${cats.slice(0,10).join(', ')}${cats.length>10?'…':''}`;
      }
    }

    wrap.appendChild(label);
    wrap.appendChild(input);
    form.appendChild(wrap);
  });

  Object.values(lists).forEach(dl=> form.appendChild(dl));
}

ui.btnPredictManual.onclick = async ()=>{
  try{
    if (!MODEL || !DATA) return;
    const r = {};
    for (const c of DATA.featureCols){
      const el = document.getElementById(`mf_${c}`);
      if (!el) continue;
      if (el.type==='number'){
        r[c] = el.value==='' ? NaN : Number(el.value);
      } else {
        r[c] = el.value;
      }
    }
    const xi = DATA.pre.transformRow(r);
    const X = tf.tensor2d(xi, [1, DATA.inputDim]);
    const p = await MODEL.predict(X).array(); X.dispose();

    const inv = invertMap(DATA.labelMap);
    const preds = p[0].map((v,i)=> ({name: inv[i], p: v})).sort((a,b)=> b.p-a.p);
    const top = preds[0];
    ui.manualOut.innerHTML = `
      <div><b>Прогноз:</b> ${top.name} (${(top.p*100).toFixed(1)}%)</div>
      <div style="margin-top:6px">Вероятности:</div>
      <table style="margin-top:4px">
        <thead><tr><th>Класс</th><th>Prob</th></tr></thead>
        <tbody>${preds.map(r=> `<tr><td>${r.name}</td><td>${(r.p*100).toFixed(1)}%</td></tr>`).join('')}</tbody>
      </table>
      <div class="small muted" style="margin-top:6px">Демо, не мед. заключение.</div>
    `;
  } catch(e){
    logln('❌ Ошибка прогноза: ' + (e.message||e));
  }
};

/* ==== helpers ==== */
async function initTF(){
  try {
    tf.env().set('WEBGL_VERSION', 1);
    tf.env().set('WEBGL_PACK', false);
    await tf.setBackend('webgl'); await tf.ready();
  } catch { await tf.setBackend('cpu'); await tf.ready(); }
  logln("TF backend: " + tf.getBackend());
}
function argmax(a){ let m=-1, mi=-1; for(let i=0;i<a.length;i++){ if(a[i]>m){m=a[i];mi=i;} } return mi; }
function invertMap(m){ return Object.entries(m).reduce((acc,[k,v])=> (acc[v]=k, acc), {}); }

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
