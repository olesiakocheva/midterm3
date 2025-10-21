// data_utils.js — CSV → фичи: numeric (min-max), categorical (one-hot)

export async function parseCSVFile(file){
  return new Promise((resolve,reject)=>{
    Papa.parse(file, { header:true, dynamicTyping:true, skipEmptyLines:true,
      complete: res => resolve(cleanRows(res.data)), error: reject });
  });
}

export async function parseCSVUrl(url){
  return new Promise((resolve,reject)=>{
    Papa.parse(url, { download:true, header:true, dynamicTyping:true, skipEmptyLines:true,
      complete: res => resolve(cleanRows(res.data)), error: reject });
  });
}

function cleanRows(rows){
  return rows.filter(r => Object.values(r).some(v => v !== null && v !== undefined && v !== ''));
}

// определяем типы колонок
export function inferSchema(rows){
  const cols = Object.keys(rows[0] || {});
  const schema = {};
  for (const c of cols){
    const vals = rows.map(r=> r[c]).filter(v=> v!==null && v!==undefined);
    const nums = vals.filter(v=> typeof v === 'number' && Number.isFinite(v));
    const uniq = new Set(vals.map(v=> String(v)));
    schema[c] = { kind: (nums.length/vals.length > 0.7) ? 'numeric' : 'categorical', unique: uniq.size };
  }
  return schema;
}

// препроцессор табличных
export function buildPreprocessor(rows, featureCols, schema){
  const catMaps = {};
  const numStats = {};

  for (const c of featureCols){
    if (schema[c].kind === 'categorical'){
      const cats = [...new Set(rows.map(r=> r[c]).map(v=> String(v)))].filter(v=> v!=='undefined' && v!=='null' && v!=='');
      catMaps[c] = cats.slice(0, 200);
    } else {
      const vals = rows.map(r=> +r[c]).filter(Number.isFinite);
      const min = Math.min(...vals), max = Math.max(...vals);
      numStats[c] = {min, max};
    }
  }

  let dim = 0;
  for (const c of featureCols){
    dim += (schema[c].kind === 'categorical') ? catMaps[c].length : 1;
  }

  function transformRow(r){
    const out = new Float32Array(dim);
    let k = 0;
    for (const c of featureCols){
      if (schema[c].kind === 'categorical'){
        const cats = catMaps[c]; const v = String(r[c] ?? '');
        for (let i=0;i<cats.length;i++) out[k+i] = (cats[i]===v) ? 1 : 0;
        k += cats.length;
      } else {
        const {min,max} = numStats[c] || {min:0,max:1};
        const x = Number.isFinite(+r[c]) ? (+r[c] - min) / (max - min + 1e-9) : 0;
        out[k++] = x;
      }
    }
    return out;
  }

  return { dim, catMaps, numStats, transformRow };
}

// подготовка тензоров (теперь возвращает ещё и pre)
export function prepareTensors(rows, featureCols, targetCol, schema, splitPct=0.8, classWeight='auto'){
  const targetVals = [...new Set(rows.map(r=> String(r[targetCol])) )];
  const labelMap = Object.fromEntries(targetVals.map((v,i)=> [v,i]));
  const nClasses = Object.keys(labelMap).length;

  const pre = buildPreprocessor(rows, featureCols, schema);

  const shuffled = rows.slice();
  for (let i=shuffled.length-1;i>0;i--){ const j=Math.floor(Math.random()*(i+1)); [shuffled[i],shuffled[j]]=[shuffled[j],shuffled[i]]; }

  const X=[], y=[];
  for (const r of shuffled){
    const xi = pre.transformRow(r);
    X.push(xi);
    const yi = new Float32Array(nClasses);
    yi[labelMap[String(r[targetCol])]] = 1;
    y.push(yi);
  }

  const N = X.length, split = Math.floor(N*splitPct);
  const Xt = tf.tensor2d(X.slice(0,split), [split, pre.dim]);
  const Yt = tf.tensor2d(y.slice(0,split), [split, nClasses]);
  const Xv = tf.tensor2d(X.slice(split), [N-split, pre.dim]);
  const Yv = tf.tensor2d(y.slice(split), [N-split, nClasses]);

  let weights = null;
  if (classWeight==='auto'){
    const counts = new Array(nClasses).fill(0);
    for (let i=0;i<split;i++){
      const idx = y[i].findIndex(v=> v===1);
      counts[idx]++;
    }
    const maxc = Math.max(...counts);
    weights = counts.map(c=> c>0 ? maxc/c : 1);
  }

  return {
    Xtrain: Xt, Ytrain: Yt,
    Xtest: Xv,  Ytest: Yv,
    inputDim: pre.dim,
    nClasses, labelMap, featureCols, classWeights: weights,
    pre // <— добавлено: пригодится для ручного прогноза
  };
}

export function confusionMatrix(trueIdx, predIdx, nClasses){
  const M = Array.from({length:nClasses}, ()=> Array(nClasses).fill(0));
  for (let i=0;i<trueIdx.length;i++) M[trueIdx[i]][predIdx[i]]++;
  return M;
}
