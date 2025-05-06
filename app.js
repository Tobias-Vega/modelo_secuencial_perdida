const fitButton = document.getElementById('fit-btn');
const inputNumber = document.getElementById('input-number');
const predictButton = document.getElementById('predict-btn');
const result = document.getElementById("result");
const predictionDiv = document.getElementById('prediction');
const graficoDiv = document.getElementById('grafico')

let model;

async function fitModel() {

  model = tf.sequential();

  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

  model.compile({
    loss: 'meanSquaredError',
    optimizer: 'sgd'
  });

  const xs = tf.tensor2d([-6, -5, -4, -3, -2, -1, 0, 1, 2], [9, 1])
  const ys = tf.tensor2d([-6, -4, -2, 0, 2, 4, 6, 8, 10], [9, 1])

  const surface = { name: 'Pérdida durante el entrenamiento', tab: 'Entrenamiento' }

  const history = await model.fit(xs, ys, {
    epochs: 350,
    callbacks: tfvis.show.fitCallbacks(surface, ['loss'], {
      callbacks: ['onEpochEnd']
    })
  })

  const losses = history.history.loss;
  const initialLoss = losses[0].toFixed(4);
  const finalLoss = losses[losses.length - 1].toFixed(4);
  const reduction = ((initialLoss - finalLoss) / initialLoss) * 100;

  graficoDiv.style.display = "block";

  document.getElementById("initial-loss").textContent = `Pérdida inicial: ${initialLoss}`;
  document.getElementById("final-loss").textContent = `Pérdida final: ${finalLoss}`;
  document.getElementById("reduction").textContent = `Reducción: ${reduction.toFixed(4)}%`;
};

predictButton.addEventListener("click", () => {

  const inputValue = inputNumber.value;

  if (inputValue.trim() === "") {
    alert("Ingrese uno o más números separados por coma");
    return;
  }

  const values = inputValue.split(',').map(Number);

  const tensor = tf.tensor2d(values, [values.length, 1])

  const prediction = model.predict(tensor)

  prediction.array().then(predict => {
    
    result.innerHTML = values.map((value, i) => {
      return `El resultaddo de predecir ${value} es: ${predict[i][0].toFixed(2)}`
    }).join("<br>")
  })
})

fitButton.addEventListener("click", () => {
  fitModel();

  alert("Entrenamiento del modelo finalizado")

  predictionDiv.style.display = 'block';
})