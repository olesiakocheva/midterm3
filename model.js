// model.js — MLP классификатор для табличных данных

export function buildMLP(inputDim, {arch='128-64', drop=0.2, nClasses=2, lr=1e-3}){
  const m = tf.sequential();
  const sizes = arch.split('-').map(s=> +s);

  m.add(tf.layers.dense({inputShape:[inputDim], units:sizes[0], activation:'relu'}));
  if (drop>0) m.add(tf.layers.dropout({rate:drop}));
  for (let i=1;i<sizes.length;i++){
    m.add(tf.layers.dense({units:sizes[i], activation:'relu'}));
    if (drop>0) m.add(tf.layers.dropout({rate:drop}));
  }
  m.add(tf.layers.dense({units:nClasses, activation:'softmax'}));
  m.compile({optimizer: tf.train.adam(lr), loss: 'categoricalCrossentropy', metrics:['accuracy']});
  return m;
}

export async function trainModel(model, Xtrain, Ytrain, epochs=25, batch=32, onEpoch){
  return model.fit(Xtrain, Ytrain, {
    epochs, batchSize: Math.max(8, Math.min(64, batch)), validationSplit: 0.1, shuffle:true,
    callbacks: { onEpochEnd: async (ep, logs)=> { onEpoch?.(ep,logs); await tf.nextFrame(); } }
  });
}

export async function evaluateModel(model, Xtest, Ytest){
  const ev = await model.evaluate(Xtest, Ytest, {verbose:0});
  const [lossT, accT] = await Promise.all(ev.map(t=> t.data()));
  return { loss: lossT[0], acc: accT[0] };
}
